"""Model-agnostic action protocol.

Every agent turn must end with exactly one JSON object: {"action": "<tool>", "args": {...}}.
We do not depend on native function calling, so any chat model (including a reasoning model
that sometimes wraps or decorates its JSON) can drive the harness.
"""
from __future__ import annotations

import json
import re

FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?)\s*```", re.S)


def _balanced_objects(text: str):
    """Yield every top-level {...} substring, honouring strings and escapes."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]


_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
LAST_ERROR: dict = {}


def _close_brackets(s: str) -> str:
    """Append the closers a model forgot (Nemotron often drops the final '}'), ignoring brackets in strings."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return s + "".join(reversed(stack)) if stack and not in_str else s


def _loads(s: str):
    """json.loads with repairs for the slips models make when they embed code in JSON strings:
    raw newlines/tabs inside strings (strict=False), invalid escapes such as \\d or \\s from regexes or
    Windows paths, trailing commas, Python literals."""
    attempts = [s]
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    s2 = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", s2)))
    attempts += [s2, _BAD_ESCAPE.sub(r"\\\\", s), _BAD_ESCAPE.sub(r"\\\\", s2)]
    attempts += [_close_brackets(a) for a in list(attempts)]
    err = None
    for a in attempts:
        try:
            return json.loads(a, strict=False)
        except json.JSONDecodeError as e:
            err = e
    LAST_ERROR["msg"] = f"{err.msg} at line {err.lineno} column {err.colno}" if err else "unknown"
    fixed = _repair_delimiters(_close_brackets(_BAD_ESCAPE.sub(r"\\\\", s2)))
    if fixed is not None:
        return fixed
    raise err


def _repair_delimiters(s: str, max_fixes: int = 30):
    """Fix quote slips inside long code strings, one decode error at a time (seen live with Nemotron):
    a stray ' or ` right after a string's closing quote (`...\\n"', "timeout": 60`), and unescaped double
    quotes inside the code (`print("x")`) that close the JSON string early."""
    for _ in range(max_fixes):
        try:
            return json.loads(s, strict=False)
        except json.JSONDecodeError as e:
            p = e.pos
            if not e.msg.startswith("Expecting ',' delimiter") and not e.msg.startswith("Expecting ':' delimiter"):
                return None
            if p < len(s) and s[p] in "'`" and s[:p].rstrip().endswith('"'):
                s = s[:p] + s[p + 1:]
                continue
            q = len(s[:p].rstrip()) - 1  # the quote that closed the string too early
            if q > 0 and s[q] == '"' and s[q - 1] != "\\":
                s = s[:q] + '\\"' + s[q + 1:]
                continue
            return None
    return None


PY_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.S)


def parse_action(text: str) -> tuple[dict | None, str, str | None]:
    """Return (action, thought_text, error). Tries fenced JSON first, then any balanced JSON object.
    For the python action, code may also be given in a separate ```python block."""
    fenced = [m.group(1) for m in FENCE_RE.finditer(text)]
    balanced = [b for b in _balanced_objects(text) if b not in fenced]
    parsed, failed = [], False
    for raw in fenced[::-1] + balanced[::-1]:  # last fenced block first, then any other JSON object
        try:
            obj = _loads(raw)
        except Exception:
            failed = True
            continue
        if isinstance(obj, dict) and "action" in obj:
            obj.setdefault("args", {})
            if not isinstance(obj["args"], dict):
                return None, text, "`args` must be a JSON object."
            if obj["action"] == "python" and not obj["args"].get("code"):
                py = PY_FENCE_RE.findall(text)
                if py:
                    obj["args"]["code"] = py[-1]
            parsed.append((raw, obj))
    # Prefer an action that carries arguments: an empty one may be a quoted example from the instructions.
    NEEDS_ARGS = {"python", "plan", "look", "finish", "power_region", "power_point", "literature_search", "promote",
                  "set_verdict", "register_sweep", "family_info", "independent_series", "propose_followup"}
    for raw, obj in parsed:
        if obj["args"] or obj["action"] not in NEEDS_ARGS:
            thought = text.replace(raw, "").replace("```json", "").replace("```", "").strip()
            return obj, thought, None
    if parsed and not failed:
        raw, obj = parsed[0]
        thought = text.replace(raw, "").replace("```json", "").replace("```", "").strip()
        return obj, thought, None
    detail = ""
    if fenced or balanced:
        detail = (f" Your JSON could not be parsed ({LAST_ERROR.get('msg', 'invalid JSON')}). If you are sending code, "
                  "put the script in a ```python block and send {\"action\": \"python\", \"args\": {}} instead.")
    elif not text.strip():
        detail = " Your visible answer was empty (the action may have stayed in your reasoning)."
    return None, text, ("No valid action found." + detail + " End your reply with exactly one fenced JSON block like "
                        '```json\n{"action": "<tool name>", "args": {...}}\n```')


def parse_json(text: str) -> dict | None:
    """For roles that answer with a JSON verdict rather than an action."""
    candidates = [m.group(1) for m in FENCE_RE.finditer(text)] or list(_balanced_objects(text))
    for raw in reversed(candidates):
        try:
            obj = _loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None
