from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_FILENAME = "syntrivetts.db"
SERVER_NAME = "webui"
HEALTH_PATH = "/api/v1/health"
HOLDER_KIND = "webui"


@dataclass(frozen=True)
class WebRepo:
    repo_dir: Path
    db_path: Path


@dataclass(frozen=True)
class ServerInfo:
    port: int
    host: str
    started_at: str
    version: str
    token: Optional[str] = None


class NotARepoError(ValueError):
    pass


def open_web_repo(repo_dir: Path) -> WebRepo:
    resolved = Path(repo_dir).expanduser().resolve()
    db_path = resolved / DB_FILENAME
    if not db_path.is_file():
        raise NotARepoError(f"Not a SyntriveTTS repo (missing {DB_FILENAME}): {resolved}")
    return WebRepo(repo_dir=resolved, db_path=db_path)
