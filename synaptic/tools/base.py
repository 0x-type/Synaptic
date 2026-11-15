from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..record import Record


@dataclass
class ToolResult:
    text: str
    images: list[str] = field(default_factory=list)  # PNG paths to show to the (vision) model
    data_chars: int = 0  # raw-data characters placed in context (the paper's "DNA read" metric)
    files: list[str] = field(default_factory=list)
    is_error: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class ToolContext:
    record: Record
    campaign_id: str
    task_id: str | None
    session_id: str | None
    workdir: Path        # private folder of this task
    campaign_dir: Path   # shared folder of the campaign (data cache, figures)
    mock: bool = False
    state: dict = field(default_factory=dict)  # per-session scratch (e.g. did the agent look?)

    @property
    def data_dir(self) -> Path:
        d = self.campaign_dir / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            for base in (self.workdir, self.campaign_dir, self.data_dir):
                if (base / p).exists():
                    return (base / p).resolve()
            p = self.workdir / p
        if not p.exists():  # agents often guess the folder wrong: fall back to the file name
            for base in (self.workdir, self.data_dir, self.campaign_dir):
                if (base / p.name).exists():
                    p = base / p.name
                    break
            else:
                hits = sorted((self.campaign_dir / "tasks").glob(f"*/{p.name}")) if (self.campaign_dir / "tasks").exists() else []
                if hits:
                    p = hits[-1]
        p = p.resolve()
        allowed = (self.campaign_dir.resolve(),)
        if not any(str(p).startswith(str(a)) for a in allowed):
            raise PermissionError(f"Path outside the campaign folder: {path}")
        return p


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, str]  # arg name -> short description (types in the text)
    fn: Callable[..., ToolResult]
    roles: tuple[str, ...] = ("worker",)
    kinds: tuple[str, ...] | None = None  # restrict to task kinds (None = all)

    def doc(self) -> str:
        a = "\n".join(f"      - {k}: {v}" for k, v in self.args.items()) or "      (no arguments)"
        return f"  * {self.name}: {self.description}\n{a}"

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        unknown = set(args) - set(self.args)
        if unknown:
            return ToolResult(f"Unknown argument(s) for {self.name}: {sorted(unknown)}. "
                              f"Allowed: {list(self.args)}", is_error=True)
        try:
            return self.fn(ctx, **args)
        except TypeError as e:
            return ToolResult(f"Bad arguments for {self.name}: {e}", is_error=True)
        except Exception as e:  # noqa: BLE001 - surface every failure to the agent
            return ToolResult(f"{self.name} failed: {type(e).__name__}: {e}", is_error=True)
