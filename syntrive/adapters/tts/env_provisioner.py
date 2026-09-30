from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENGINES_DIR = _REPO_ROOT / "engines"
_SETUP_SCRIPT = _REPO_ROOT / "tools" / "setup_tts_envs.py"

STATE_MISSING = "missing"
STATE_PROVISIONED = "provisioned"

VALID_DEVICES = ("cpu", "cuda", "mps")

_LogFn = Callable[[str], None]


def _default_log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class ProvisionResult:
    ok: bool
    engine: str
    device: str
    returncode: int
    error: str = ""


def engine_dir(engine: str) -> Path:
    return _ENGINES_DIR / engine


def venv_python(engine: str) -> Path:
    venv = engine_dir(engine) / ".venv"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def venv_state(engine: str) -> str:
    return STATE_PROVISIONED if venv_python(engine).is_file() else STATE_MISSING


def detect_compute_device() -> str:
    if shutil.which("nvidia-smi"):
        try:
            probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            if probe.returncode == 0 and "GPU" in probe.stdout:
                return "cuda"
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("nvidia-smi probe failed: %s", exc)
    if sys.platform == "darwin":
        return "mps"
    return "cpu"


def provision(engine: str, device: str, *, log: _LogFn = _default_log) -> ProvisionResult:
    if engine not in _known_engines():
        return ProvisionResult(False, engine, device, -1, f"unknown engine {engine!r}")
    if device not in VALID_DEVICES:
        return ProvisionResult(False, engine, device, -1,
                               f"device must be one of {VALID_DEVICES} (got {device!r})")
    if not _SETUP_SCRIPT.is_file():
        return ProvisionResult(False, engine, device, -1, f"{_SETUP_SCRIPT} not found")

    cmd = [sys.executable, str(_SETUP_SCRIPT), engine, "--torch-device", device]
    log(f"env_provision: engine={engine} device={device} cmd={' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except OSError as exc:
        log(f"env_provision_failed: engine={engine} error={exc}")
        return ProvisionResult(False, engine, device, -1, str(exc))

    assert proc.stdout is not None
    for line in proc.stdout:
        log(f"  [setup_tts_envs] {line.rstrip()}")
    rc = proc.wait()

    state = venv_state(engine)
    ok = rc == 0 and state == STATE_PROVISIONED
    log(f"env_provision_done: engine={engine} device={device} rc={rc} state={state} ok={ok}")
    return ProvisionResult(ok, engine, device, rc, "" if ok else f"exit {rc}, venv state {state}")


def _known_engines() -> tuple:
    from syntrive.adapters.tts import model_catalog

    return tuple(e for e in model_catalog.engines() if model_catalog.has_venv(e))
