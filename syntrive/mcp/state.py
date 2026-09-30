from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path

from syntrive.mcp.envelope import ConfirmTokens, fail
from syntrive.services.pipeline_service import StepRunRegistry

logger = logging.getLogger(__name__)

DB_FILENAME = "syntrivetts.db"


@dataclass(frozen=True)
class Repo:
    repo_dir: Path
    db_path: Path


def open_repo(repo_dir: Path) -> Repo:
    resolved = Path(repo_dir).expanduser().resolve()
    db_path = resolved / DB_FILENAME
    if not db_path.is_file():
        raise fail("not_found", f"Not a SyntriveTTS repo (missing {DB_FILENAME}): {resolved}")
    return Repo(resolved, db_path)


TRANSPORTS = ("stdio", "http")


@dataclass
class McpState:
    repo: Repo
    transport: str = "stdio"
    runs: StepRunRegistry = field(default_factory=StepRunRegistry)
    tokens: ConfirmTokens = field(default_factory=lambda: ConfirmTokens(secrets.token_bytes(32)))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def current(self) -> Repo:
        with self._lock:
            return self.repo

    def switch(self, repo_dir: Path) -> Repo:
        if self.transport == "http":
            raise fail("refused", f"This HTTP server is bound to {self.repo.repo_dir} and shared by its clients; "
                                  f"start another server for the other repo: python -m syntrive.mcp.server -r <repo> --transport http --port <port>")
        new = open_repo(repo_dir)
        with self._lock:
            if self.runs.any_running():
                raise fail("conflict", "A pipeline step is still running; switch the repo when it has finished.")
            old, self.repo = self.repo, new
        logger.info("mcp_repo_switched: from=%s to=%s", old.repo_dir, new.repo_dir)
        return new
