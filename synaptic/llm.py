"""LLM access: NVIDIA's OpenAI-compatible endpoint (Nemotron) with rate limiting, retries,
vision content, and reasoning capture. MockLLM (in mock.py) implements the same interface."""
from __future__ import annotations

import base64
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import MODELS


@dataclass
class Completion:
    text: str            # the visible answer (action JSON lives here)
    reasoning: str       # hidden chain of thought if the endpoint returns it
    tokens_in: int
    tokens_out: int
    model: str
    finish_reason: str = "stop"


class RateLimiter:
    """Sliding-window limiter shared by every agent thread, plus a shared cooldown: when any thread is told
    'Too Many Requests', every thread pauses, instead of each one hammering the API."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.calls: list[float] = []
        self.lock = threading.Lock()
        self.cooldown_until = 0.0

    def cool_down(self, seconds: float):
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.time() + seconds)

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                if now < self.cooldown_until:
                    sleep = self.cooldown_until - now
                    self.calls = [t for t in self.calls if now - t < 60]
                    pass
                else:
                    sleep = 0
                if not sleep:
                    self.calls = [t for t in self.calls if now - t < 60]
                    if len(self.calls) < self.rpm:
                        self.calls.append(now)
                        return
                    sleep = 60 - (now - self.calls[0]) + 0.05
            time.sleep(max(0.05, sleep))


def image_part(path: str | Path, max_px: int = 1600) -> dict:
    """Images go to the model as PNG or JPEG only, at most max_px on a side (other formats, e.g. an agent's GIF,
    TIFF or WEBP, or a huge PNG, were rejected live with HTTP 415)."""
    raw = Path(path).read_bytes()
    kind = "png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg" if raw[:3] == b"\xff\xd8\xff" else None
    try:
        from PIL import Image
        import io

        im = Image.open(io.BytesIO(raw))
        if kind is None or max(im.size) > max_px or len(raw) > 4_000_000:
            im.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG", optimize=True)
            raw, kind = buf.getvalue(), "png"
    except Exception:  # noqa: BLE001
        kind = kind or "png"
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{base64.b64encode(raw).decode()}"}}


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


class NvidiaLLM:
    name = "nvidia"

    def __init__(self, cfg=MODELS):
        from openai import OpenAI

        if not cfg.api_key:
            raise RuntimeError("NVIDIA_API_KEY is missing. Put it in .env (see .env.example) or use --mock.")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
        self.limiter = RateLimiter(cfg.rpm)
        self._thinking_supported: dict[str, bool] = {}
        self._structured_ok: dict[str, bool] = {}
        self._fails: dict[str, int] = {}

    def _extra(self, model: str, role: str = "worker") -> dict | None:
        mode = self.cfg.thinking_for(role)
        if mode == "off" or self._thinking_supported.get(model) is False:
            return None
        kw = {"enable_thinking": True}
        if mode == "low":
            kw["low_effort"] = True
        return {"chat_template_kwargs": kw}

    def complete(self, messages: list[dict], role: str, max_tokens: int | None = None,
                 temperature: float | None = None, schema: dict | None = None) -> Completion:
        model = self.cfg.model_for(role)
        if not self.cfg.vision or model in getattr(self.cfg, "text_only", ()):
            messages = _strip_images(messages)
        delay = 4.0
        last_err = None
        rate_limited = 0
        for attempt in range(14):
            self.limiter.wait()
            extra = self._extra(model, role)
            try:
                kwargs = dict(model=model, messages=messages,
                              temperature=self.cfg.temperature if temperature is None else temperature,
                              top_p=self.cfg.top_p, max_tokens=max_tokens or self.cfg.max_tokens)
                if extra:
                    kwargs["extra_body"] = extra
                if schema and self._structured_ok.get(model, True):
                    # constrained decoding: the reply IS valid JSON (v3 lost ~1 action in 34 to format slips)
                    kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "schema": schema}}
                t_call = time.time()
                r = self.client.chat.completions.create(**kwargs)
                self._fails[model] = 0
                msg = r.choices[0].message
                text = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or \
                    (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or ""
                m = _THINK_RE.search(text)
                if m:
                    reasoning = (reasoning + "\n" + m.group(1)).strip()
                    text = _THINK_RE.sub("", text).strip()
                if "</think>" in text:  # some templates emit only the closing tag
                    head, text = text.split("</think>", 1)
                    reasoning = (reasoning + "\n" + head).strip()
                u = r.usage
                return Completion(text.strip(), reasoning, getattr(u, "prompt_tokens", 0) or 0,
                                  getattr(u, "completion_tokens", 0) or 0, model,
                                  r.choices[0].finish_reason or "stop")
            except Exception as e:  # noqa: BLE001 - network errors come in many shapes
                last_err = e
                msg = str(e)
                status = getattr(e, "status_code", None)
                if status == 400 and schema and ("response_format" in msg or "json_schema" in msg):
                    self._structured_ok[model] = False  # the endpoint cannot constrain: fall back to the repair parser
                    continue
                if status is not None and status >= 500 or status is None and "timed out" in msg.lower():
                    # circuit breaker: repeated server errors on one model switch the call to the fallback model
                    self._fails[model] = self._fails.get(model, 0) + 1
                    fb = self.cfg.fallback_model
                    if fb and fb != model and self._fails[model] >= 2:
                        self.limiter.cool_down(2)
                        model = fb
                        if model in getattr(self.cfg, "text_only", ()):
                            messages = _strip_images(messages)
                        continue
                if status == 400 and extra and ("chat_template_kwargs" in msg or "enable_thinking" in msg):
                    self._thinking_supported[model] = False
                    continue
                if status == 400 and "image" in msg.lower() and self.cfg.vision and "model length" not in msg.lower():
                    raise RuntimeError(f"Model {model} rejected image input. Set SYNAPTIC_VISION=0 "
                                       f"or use a VL model. ({msg[:200]})") from e
                if (status in (415, 422) or "unsupported media" in msg.lower() or "could not be screened" in msg.lower()) \
                        and any(
                        isinstance(m.get("content"), list) for m in messages):
                    messages = _strip_images(messages)  # an image the endpoint cannot read: continue without it
                    continue
                if status in (401, 403, 404):
                    raise
                if status == 429:
                    rate_limited += 1
                    retry_after = 0.0
                    try:
                        retry_after = float(e.response.headers.get("retry-after", 0))  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                    self.limiter.cool_down(max(retry_after, min(15 * rate_limited, 120)))
                    continue  # the limiter's shared cooldown does the waiting
                if attempt >= 7:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 90)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


def _strip_images(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            txt = "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
            out.append({**m, "content": txt + "\n[image omitted: vision disabled]"})
        else:
            out.append(m)
    return out


def make_llm(mock: bool):
    if mock:
        from .mock import MockLLM

        return MockLLM()
    return NvidiaLLM()
