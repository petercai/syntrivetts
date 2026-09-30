from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SERVERS_DIRNAME = Path(".syntrive") / "servers"
_HEALTH_TIMEOUT_SECONDS = 0.5
_LOOPBACK = "127.0.0.1"


@dataclass(frozen=True)
class ServerRecord:
    name: str
    port: int
    pid: int
    started_at: str
    host: str = _LOOPBACK
    health_path: str = "/api/health"

    @property
    def local_url(self) -> str:
        return f"http://{_LOOPBACK}:{self.port}/"


def record_path(repo_dir: Path, name: str) -> Path:
    return repo_dir / SERVERS_DIRNAME / f"{name}.json"


def bind_port(host: str = _LOOPBACK, port: int = 0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    return sock


def bind_free_port(host: str = _LOOPBACK) -> socket.socket:
    return bind_port(host, 0)


def write_record(repo_dir: Path, name: str, port: int, *, host: str = _LOOPBACK, health_path: str = "/api/health") -> ServerRecord:
    record = ServerRecord(
        name=name, port=port, pid=os.getpid(), started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        host=host, health_path=health_path,
    )
    path = record_path(repo_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    logger.info("server_record_written: name=%s repo_dir=%s port=%s host=%s pid=%s", name, repo_dir, port, host, record.pid)
    return record


def read_record(repo_dir: Path, name: str) -> Optional[ServerRecord]:
    path = record_path(repo_dir, name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ServerRecord(
            name=str(data.get("name", name)), port=int(data["port"]), pid=int(data["pid"]),
            started_at=str(data["started_at"]), host=str(data.get("host", _LOOPBACK)),
            health_path=str(data.get("health_path", "/api/health")),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("server_record_read_failed: path=%s error=%s", path, exc)
        return None


def remove_record(repo_dir: Path, name: str) -> None:
    path = record_path(repo_dir, name)
    try:
        path.unlink()
        logger.info("server_record_removed: name=%s repo_dir=%s", name, repo_dir)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("server_record_remove_failed: path=%s error=%s", path, exc)


def _answers_for_repo(record: ServerRecord, repo_dir: Path) -> bool:
    url = f"http://{_LOOPBACK}:{record.port}{record.health_path}"
    try:
        with urllib.request.urlopen(url, timeout=_HEALTH_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.info("server_health_unreachable: name=%s port=%s error=%s", record.name, record.port, exc)
        return False
    return isinstance(payload, dict) and payload.get("repo_dir") == str(repo_dir.resolve())


def find_live_server(repo_dir: Path, name: str) -> Optional[ServerRecord]:
    record = read_record(repo_dir, name)
    if record is None:
        return None
    if _answers_for_repo(record, repo_dir):
        logger.info("server_already_running: name=%s repo_dir=%s url=%s", name, repo_dir, record.local_url)
        return record
    logger.warning("server_record_stale: name=%s repo_dir=%s recorded_port=%s -- treating as free", name, repo_dir, record.port)
    return None


def list_live_servers(repo_dir: Path, *, exclude: Optional[str] = None) -> list[ServerRecord]:
    folder = repo_dir / SERVERS_DIRNAME
    if not folder.is_dir():
        return []
    names = sorted(p.stem for p in folder.glob("*.json") if p.stem != exclude)
    return [record for record in (find_live_server(repo_dir, n) for n in names) if record is not None]
