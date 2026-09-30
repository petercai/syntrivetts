from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _canonicalize(path: Path) -> Path:
    return path.resolve()


def repo_dir_from_db(db_path: Path) -> Path:
    return _canonicalize(db_path).parent


def to_repo_relative(db_path: Path, abs_path: Path) -> str:
    repo = repo_dir_from_db(db_path)
    canonical = _canonicalize(abs_path)
    try:
        return canonical.relative_to(repo).as_posix()
    except ValueError:
        logger.error(
            "to_repo_relative: path %r is not under repo_dir %r",
            str(abs_path), str(repo),
        )
        raise


def resolve_abs(db_path: Path, rel_posix: str) -> Path:
    return repo_dir_from_db(db_path) / rel_posix
