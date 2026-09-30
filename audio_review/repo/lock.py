from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from syntrive.io import server_registry

SERVER_NAME = "audio_review"


@dataclass(frozen=True)
class LockInfo:
    port: int
    pid: int
    started_at: str


def lock_path(repo_dir: Path) -> Path:
    return server_registry.record_path(repo_dir, SERVER_NAME)


def read_lock(repo_dir: Path) -> Optional[LockInfo]:
    record = server_registry.read_record(repo_dir, SERVER_NAME)
    return None if record is None else LockInfo(port=record.port, pid=record.pid, started_at=record.started_at)


def write_lock(repo_dir: Path, port: int) -> None:
    server_registry.write_record(repo_dir, SERVER_NAME, port, health_path="/api/health")


def remove_lock(repo_dir: Path) -> None:
    server_registry.remove_record(repo_dir, SERVER_NAME)


def check_existing_server(repo_dir: Path) -> Optional[str]:
    record = server_registry.find_live_server(repo_dir, SERVER_NAME)
    return None if record is None else record.local_url
