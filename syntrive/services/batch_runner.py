from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from syntrive.io.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

RECORD = Path(".syntrive") / "runner.json"
CONSOLE_LOG = Path("logs") / "tts_batch.console.log"

_CHILDREN: dict[int, subprocess.Popen] = {}


def _alive(pid: int) -> bool:
    from syntrive.services.job_lease import pid_alive

    child = _CHILDREN.get(pid)
    return child.poll() is None if child is not None else pid_alive(pid)


@dataclass(frozen=True)
class RunnerProcess:
    pid: int
    started_at: str
    command: tuple[str, ...]
    alive: bool

    @property
    def offline(self) -> bool:
        return OFFLINE_FLAG in self.command


@dataclass(frozen=True)
class StartResult:
    started: bool
    pid: Optional[int] = None
    reason: Optional[str] = None


OFFLINE_FLAG = "--offline"


def default_command(repo_dir: Path, *, offline: bool = False) -> list[str]:
    argv = [sys.executable, str(PROJECT_ROOT / "tts_batch.py"), "-r", str(repo_dir)]
    return argv + [OFFLINE_FLAG] if offline else argv


def runner_process(repo_dir: Path) -> Optional[RunnerProcess]:
    path = repo_dir / RECORD
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data["pid"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("runner_record_unreadable: path=%s error=%s", path, exc)
        return None
    return RunnerProcess(pid, str(data.get("started_at", "")), tuple(data.get("command", ())), _alive(pid))


def _detach_kwargs() -> dict:
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return {"creationflags": flags}
    return {"start_new_session": True}


def start_runner(db_path: Path, command: Optional[Sequence[str]] = None, *, offline: bool = False) -> StartResult:
    from syntrive.services.queue_overview import live_runner

    repo_dir = db_path.parent
    lease = live_runner(db_path)
    if lease is not None:
        logger.info("runner_start_refused: repo=%s reason=lease_live job_id=%d op=%s", repo_dir, lease.job_id, lease.operation)
        return StartResult(False, lease.pid, "lease_live")
    recorded = runner_process(repo_dir)
    if recorded is not None and recorded.alive:
        logger.info("runner_start_refused: repo=%s reason=process_alive pid=%d", repo_dir, recorded.pid)
        return StartResult(False, recorded.pid, "process_alive")

    argv = list(command or default_command(repo_dir, offline=offline))
    log_path = repo_dir / CONSOLE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        child = subprocess.Popen(  # noqa: S603 - argv built here, no shell
            argv, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **_detach_kwargs(),
        )
    _CHILDREN[child.pid] = child
    record = {"pid": child.pid, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": argv}
    (repo_dir / RECORD).parent.mkdir(parents=True, exist_ok=True)
    (repo_dir / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("runner_started: repo=%s pid=%d offline=%s log=%s", repo_dir, child.pid, OFFLINE_FLAG in argv, log_path)
    return StartResult(True, child.pid)


OFFLINE_ENV = "SYNTRIVE_TTS_OFFLINE"


@dataclass(frozen=True)
class OfflineDefault:
    checked: bool
    locked: bool
    source: Optional[str]


def dotenv_value(text: str, key: str) -> Optional[str]:
    value = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, rest = line.removeprefix("export ").partition("=")
        if name.strip() == key:
            value = rest.split(" #", 1)[0].strip().strip("'\"")
    return value


def offline_default(environ: Optional[dict] = None, dotenv_path: Optional[Path] = None) -> OfflineDefault:
    env = os.environ if environ is None else environ
    if env.get(OFFLINE_ENV) == "1":
        return OfflineDefault(True, True, "environment")
    path = dotenv_path or (PROJECT_ROOT / ".env")
    try:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError as exc:
        logger.warning("dotenv_unreadable: path=%s error=%s", path, exc)
        text = ""
    if dotenv_value(text, OFFLINE_ENV) == "1":
        return OfflineDefault(True, False, "dotenv")
    return OfflineDefault(False, False, None)
