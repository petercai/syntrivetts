from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.engine import Engine

from syntrive.db.session import ensure_schema, make_engine

logger = logging.getLogger(__name__)

DB_FILENAME = "syntrivetts.db"


@dataclass(frozen=True)
class ActiveRepo:
    repo_dir: Path
    db_path: Path
    engine: Engine
    port: int


def open_repo(repo_dir: Path, port: int) -> ActiveRepo:
    db_path = repo_dir / DB_FILENAME
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Not a SyntriveTTS repo (missing {DB_FILENAME}): {repo_dir}"
        )

    engine = make_engine(db_path)
    ensure_schema(engine)
    logger.info("audio_review_repo_opened: repo_dir=%s db_path=%s", repo_dir, db_path)
    return ActiveRepo(repo_dir=repo_dir, db_path=db_path, engine=engine, port=port)
