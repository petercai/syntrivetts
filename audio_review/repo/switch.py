from __future__ import annotations

import logging
from pathlib import Path

from audio_review.repo.lock import check_existing_server, remove_lock, write_lock
from audio_review.repo.state import ActiveRepo, open_repo

logger = logging.getLogger(__name__)


class RepoAlreadyServedError(Exception):
    def __init__(self, url: str) -> None:
        super().__init__(f"Repo already served at {url}")
        self.url = url


def switch_active_repo(current: ActiveRepo, new_repo_dir: Path) -> ActiveRepo:
    resolved_new = new_repo_dir.resolve()

    existing_url = check_existing_server(resolved_new)
    if existing_url is not None:
        logger.info(
            "audio_review_repo_switch_refused: old_repo=%s new_repo=%s existing_url=%s",
            current.repo_dir, resolved_new, existing_url,
        )
        raise RepoAlreadyServedError(existing_url)

    new_state = open_repo(resolved_new, current.port)

    remove_lock(current.repo_dir)
    write_lock(resolved_new, current.port)
    logger.info(
        "audio_review_repo_switched: old_repo=%s new_repo=%s port=%s",
        current.repo_dir, resolved_new, current.port,
    )
    return new_state
