"""Central configuration. Everything secret comes from the environment (.env), never from code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.getenv("SYNAPTIC_RUNS_DIR", ROOT / "runs"))
SKILLS_DIR = ROOT / "skills"
SYNLIB_DIR = ROOT  # parent folder of the synlib package, put on PYTHONPATH for agent code


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class ModelConfig:
    """One model per role. The paper ran every role on the same model; we allow a stronger
    supervisor/judge (e.g. nemotron-3-ultra) while workers stay on the fast VL model."""

    base_url: str = _env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key: str = _env("NVIDIA_API_KEY")
    default: str = _env("SYNAPTIC_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    by_role: dict = field(default_factory=lambda: {
        "planner": _env("SYNAPTIC_MODEL_PLANNER"),
        "worker": _env("SYNAPTIC_MODEL_WORKER"),
        "supervisor": _env("SYNAPTIC_MODEL_SUPERVISOR"),
        "curator": _env("SYNAPTIC_MODEL_CURATOR"),
        "editor": _env("SYNAPTIC_MODEL_EDITOR"),
        # v4: the judgement roles use NVIDIA's largest model (Nemotron 3 Ultra 550B, text-only); workers keep the
        # fast vision model because they must look at figures
        "judge": _env("SYNAPTIC_MODEL_JUDGE", "nvidia/nemotron-3-ultra-550b-a55b"),
        "auditor": _env("SYNAPTIC_MODEL_AUDITOR", "nvidia/nemotron-3-ultra-550b-a55b"),
        "triage": _env("SYNAPTIC_MODEL_TRIAGE"),
        "pi": _env("SYNAPTIC_MODEL_PI", "nvidia/nemotron-3-ultra-550b-a55b"),
    })
    vision: bool = _env("SYNAPTIC_VISION", "1") not in ("0", "false", "no")
    temperature: float = float(_env("SYNAPTIC_TEMPERATURE", "0.6"))
    top_p: float = float(_env("SYNAPTIC_TOP_P", "0.95"))
    max_tokens: int = int(_env("SYNAPTIC_MAX_TOKENS", "16384"))
    # "on" | "off" | "low"  -> chat_template_kwargs.enable_thinking / low_effort. Measured on Nemotron 3.5 VL:
    # full reasoning on a worker step took ~28 s and often spent the whole output budget before writing the
    # action; low effort took ~8 s and acted reliably. Workers take many small steps -> low; judgement roles -> on.
    thinking: str = _env("SYNAPTIC_THINKING", "")
    thinking_by_role: dict = field(default_factory=lambda: {
        "worker": _env("SYNAPTIC_THINKING_WORKER", "low"), "planner": _env("SYNAPTIC_THINKING_PLANNER", "on"),
        "supervisor": _env("SYNAPTIC_THINKING_SUPERVISOR", "on"), "triage": _env("SYNAPTIC_THINKING_TRIAGE", "low"),
        "curator": _env("SYNAPTIC_THINKING_CURATOR", "low"), "editor": _env("SYNAPTIC_THINKING_EDITOR", "on"),
        "judge": _env("SYNAPTIC_THINKING_JUDGE", "on"),
        "auditor": _env("SYNAPTIC_THINKING_AUDITOR", "on"), "bench": _env("SYNAPTIC_THINKING_BENCH", "low"),
        "pi": _env("SYNAPTIC_THINKING_PI", "on")})

    def thinking_for(self, role: str) -> str:
        return (self.thinking or self.thinking_by_role.get(role) or "low").lower()
    rpm: int = int(_env("SYNAPTIC_RPM", "35"))  # NVIDIA trial endpoints are rate limited
    # v4: constrained JSON for worker actions (the endpoint supports response_format json_schema) and a fallback
    # model that takes over a call after repeated server errors (one judge call held a v3 run for 40 minutes)
    structured: bool = _env("SYNAPTIC_STRUCTURED", "1") not in ("0", "false", "no")
    fallback_model: str = _env("SYNAPTIC_MODEL_FALLBACK", "nvidia/nemotron-3-ultra-550b-a55b")
    text_only: tuple = ("nvidia/nemotron-3-ultra-550b-a55b",)  # images are removed before calling these
    timeout: int = int(_env("SYNAPTIC_TIMEOUT", "300"))

    def model_for(self, role: str) -> str:
        return self.by_role.get(role) or self.default


@dataclass
class HarnessConfig:
    concurrency: int = int(_env("SYNAPTIC_CONCURRENCY", "4"))
    max_tasks: int = int(_env("SYNAPTIC_MAX_TASKS", "60"))
    max_depth: int = int(_env("SYNAPTIC_MAX_DEPTH", "4"))
    max_followups_per_task: int = int(_env("SYNAPTIC_MAX_FOLLOWUPS", "3"))
    max_revisions: int = int(_env("SYNAPTIC_MAX_REVISIONS", "3"))
    max_steps: int = int(_env("SYNAPTIC_MAX_STEPS", "28"))  # tool calls per worker session
    max_promoted: int = int(_env("SYNAPTIC_MAX_PROMOTED", "8"))
    verification_share: float = float(_env("SYNAPTIC_VERIFICATION_SHARE", "0.15"))  # rest of the budget is science
    exec_timeout: int = int(_env("SYNAPTIC_EXEC_TIMEOUT", "600"))
    require_look: bool = _env("SYNAPTIC_REQUIRE_LOOK", "1") not in ("0", "false")
    # share of the task budget that survey-stage (sweep/classify/census) follow-ups may use; the rest is kept
    # for deep dives and reports
    survey_followup_share: float = float(_env("SYNAPTIC_SURVEY_FOLLOWUP_SHARE", "0.15"))
    # v3 research programme (PI + hypothesis board), run in parallel with the survey from the first hour
    program: bool = _env("SYNAPTIC_PROGRAM", "1") not in ("0", "false", "no")
    program_slots: float = float(_env("SYNAPTIC_PROGRAM_SLOTS", "0.5"))     # share of workers reserved for it
    program_budget: float = float(_env("SYNAPTIC_PROGRAM_BUDGET", "0.5"))   # share of max_tasks it may use
    meeting_every: int = int(_env("SYNAPTIC_MEETING_EVERY", "6"))           # finished tasks between lab meetings
    max_meetings: int = int(_env("SYNAPTIC_MAX_MEETINGS", "10"))
    program_wait_min: float = float(_env("SYNAPTIC_PROGRAM_WAIT_MIN", "15"))  # max wait for the GPU before the PI writes


@dataclass
class ServicesConfig:
    kaggle_username: str = _env("KAGGLE_USERNAME")
    kaggle_key: str = _env("KAGGLE_KEY")
    kaggle_accelerator: str = _env("KAGGLE_ACCELERATOR", "")  # optional machine_shape, e.g. NvidiaTeslaT4
    openalex_mailto: str = _env("OPENALEX_MAILTO")
    s2_api_key: str = _env("SEMANTIC_SCHOLAR_API_KEY")
    ads_token: str = _env("ADS_API_TOKEN")
    earthdata_token: str = _env("EARTHDATA_TOKEN")


MODELS = ModelConfig()
HARNESS = HarnessConfig()
SERVICES = ServicesConfig()
