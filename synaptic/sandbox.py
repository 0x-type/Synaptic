"""Where agent-written Python runs (the expert review: agent code ran unsandboxed on the host and could read the
.env file). Modes, best first (SYNAPTIC_SANDBOX=auto picks the first available):

  docker      container with no network, no credentials and no host files except the task folder (read-write),
              the campaign data (read-only) and synlib (read-only). `python -m synaptic sandbox build` once.
  user        Linux, harness running as root: the script runs as the unprivileged user `nobody` in its own network
              namespace (no network), so root-only files such as .env (chmod 600) cannot be read.
  macos       macOS sandbox-exec profile: no network, reads of .env, ~/.kaggle and key files denied.
  plain       last resort: credentials removed from the environment only (a warning is logged).

In every mode every environment variable that looks like a credential is removed and memory is capped.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE = os.environ.get("SYNAPTIC_SANDBOX_IMAGE", "synaptic-sandbox:4")
SECRET_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I)
_MODE: str | None = None


def clean_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not SECRET_RE.search(k)}
    env.update(extra or {})
    return env


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def mode() -> str:
    global _MODE
    if _MODE:
        return _MODE
    want = os.environ.get("SYNAPTIC_SANDBOX", "auto").lower()
    if want != "auto":
        _MODE = want
    elif _docker_ok():
        _MODE = "docker"
    elif sys.platform.startswith("linux") and os.geteuid() == 0 and shutil.which("unshare"):
        _MODE = "user"
    elif sys.platform == "darwin" and shutil.which("sandbox-exec"):
        _MODE = "macos"
    else:
        _MODE = "plain"
    return _MODE


_NETNS: bool | None = None


def _netns_ok() -> bool:
    global _NETNS
    if _NETNS is None:
        try:
            _NETNS = subprocess.run(["unshare", "--net", "true"], capture_output=True, timeout=10).returncode == 0 \
                and bool(shutil.which("setpriv"))
        except Exception:  # noqa: BLE001
            _NETNS = False
    return _NETNS


def network_isolated() -> bool:
    m = mode()
    return m in ("docker", "macos") or (m == "user" and _netns_ok())


def _macos_profile(secret_paths: list[str]) -> str:
    deny = "".join(f'(deny file-read* (literal "{p}"))' for p in secret_paths)
    deny += f'(deny file-read* (subpath "{Path.home() / ".kaggle"}"))'
    return f"(version 1)(allow default)(deny network*){deny}"


def secret_files() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    cands = [root / ".env", Path.home() / ".kaggle" / "kaggle.json", Path.home() / ".netrc"]
    extra = os.environ.get("SYNAPTIC_SECRET_FILES", "")
    cands += [Path(p) for p in extra.split(os.pathsep) if p]
    return [str(p) for p in cands if p.exists()]


def run_python(script: Path, cwd: Path, data_dir: Path, synlib_dir: Path, timeout: int, mem_gb: float = 2.5,
               extra_env: dict | None = None) -> subprocess.CompletedProcess:
    m = mode()
    env = clean_env({"PYTHONPATH": str(synlib_dir), "MPLBACKEND": "Agg", **(extra_env or {})})
    cap = int(mem_gb * 1024 ** 3)
    if m == "docker":
        cmd = ["docker", "run", "--rm", "--network", "none", "--memory", f"{mem_gb}g", "--cpus", "2",
               "-v", f"{cwd}:/work", "-v", f"{data_dir}:/data:ro", "-v", f"{synlib_dir}:/synlib:ro",
               "-e", "PYTHONPATH=/synlib", "-e", "MPLBACKEND=Agg", "-e", "SYNAPTIC_DATA=/data", "-w", "/work",
               IMAGE, "python", f"/work/{script.relative_to(cwd)}"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def limits():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except Exception:  # noqa: BLE001
            pass

    if m == "user":
        _open_for_nobody(cwd, (data_dir, synlib_dir))
        env["HOME"] = str(cwd)
        if _netns_ok():  # own network namespace (no network), then drop to nobody
            cmd = ["unshare", "--net", "--", "setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
                   sys.executable, str(script)]
            return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                                  preexec_fn=limits)

        def drop():  # no namespace support (e.g. inside a container): still unprivileged, so root-only files stay closed
            limits()
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
        return subprocess.run([sys.executable, str(script)], cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=timeout, preexec_fn=drop)
    if m == "macos":
        return subprocess.run(["sandbox-exec", "-p", _macos_profile(secret_files()), sys.executable, str(script)],
                              cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, preexec_fn=limits)
    return subprocess.run([sys.executable, str(script)], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, preexec_fn=limits if os.name == "posix" else None)


def _open_for_nobody(cwd: Path, readable: tuple = ()):
    """The task folder must be writable by the unprivileged user, the data and synlib readable, and their parent
    folders traversable (execute bit only: their listings stay closed). Nothing else is opened up."""
    import stat

    for base in (cwd, *readable):
        for parent in Path(base).resolve().parents:
            try:
                mode_ = parent.stat().st_mode
                if not mode_ & stat.S_IXOTH:
                    parent.chmod(mode_ | stat.S_IXOTH)
            except OSError:
                pass
    for base in readable:
        for p in [Path(base), *Path(base).rglob("*")]:
            try:
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode | stat.S_IROTH | (stat.S_IXOTH if p.is_dir() else 0))
            except OSError:
                pass
    for p in [cwd, *cwd.rglob("*")]:
        try:
            if p.is_symlink():
                continue
            p.chmod(0o777 if p.is_dir() else 0o666)
        except OSError:
            pass


def build_image(root: Path) -> int:
    return subprocess.call(["docker", "build", "-t", IMAGE, str(root / "sandbox")])
