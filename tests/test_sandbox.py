"""Agent code cannot see credentials (environment or key files) and has no network in the isolated modes."""
import os
from pathlib import Path

import pytest

from synaptic import sandbox


def test_clean_env_drops_every_credential(monkeypatch):
    for k in ("NVIDIA_API_KEY", "KAGGLE_API_TOKEN", "EARTHDATA_TOKEN", "OPENALEX_API_KEY", "ADS_API_TOKEN", "MY_SECRET"):
        monkeypatch.setenv(k, "x")
    env = sandbox.clean_env()
    assert not [k for k in env if any(s in k for s in ("KEY", "TOKEN", "SECRET"))]


@pytest.mark.skipif(sandbox.mode() not in ("user", "docker"), reason="needs an isolated sandbox mode")
def test_isolated_script_cannot_read_secret_file_or_network(tmp_path):
    secret = tmp_path / "secret.env"
    secret.write_text("NVIDIA_API_KEY=abc")
    secret.chmod(0o600)
    work = tmp_path / "task"
    work.mkdir()
    script = work / "s.py"
    script.write_text(f"""
import os, socket
try:
    open({str(secret)!r}).read(); print('READ')
except Exception: print('NO_READ')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=3); print('NET')
except Exception: print('NO_NET')
print('KEY' if os.environ.get('NVIDIA_API_KEY') else 'NO_KEY')
""")
    os.environ["NVIDIA_API_KEY"] = "abc"
    data = tmp_path / "data"
    data.mkdir()
    r = sandbox.run_python(script, work, data, Path(__file__).resolve().parents[1] / "synlib", 60)
    assert "NO_READ" in r.stdout and "NO_KEY" in r.stdout, r.stdout + r.stderr
    if sandbox.network_isolated():
        assert "NO_NET" in r.stdout
