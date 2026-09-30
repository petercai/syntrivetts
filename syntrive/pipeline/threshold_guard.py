from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

THRESHOLD = 0.35


class ThresholdGuard:
    def __init__(
        self,
        job_id: int,
        chapter_id: str,
        db_path: Path,
    ) -> None:
        self._job_id = job_id
        self._chapter_id = chapter_id
        self._db_path = db_path

    def check(
        self,
        raw_chars: int,
        cleaned_chars: int,
        chapter_title: str,
        rule_distribution: Optional[dict] = None,
    ) -> bool:
        ratio = _deletion_ratio(raw_chars, cleaned_chars)

        if ratio <= THRESHOLD:
            return False

        logger.warning(
            "ThresholdGuard: exceeded %.1f%% for chapter=%s title=%r (%.1f%% deleted) "
            "— continuing without confirmation (non-blocking since 2026-08-25)",
            THRESHOLD * 100, self._chapter_id, chapter_title, ratio * 100,
        )
        self._record_event(ratio, rule_distribution)
        return True

    def _record_event(
        self,
        deletion_ratio: float,
        rule_distribution: Optional[dict],
    ) -> None:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import ThresholdBlockEvent

            with get_db_session(self._db_path) as db:
                event = ThresholdBlockEvent(
                    job_id=self._job_id,
                    chapter_id=self._chapter_id,
                    deletion_ratio=deletion_ratio,
                    rule_distribution=rule_distribution or {},
                )
                db.add(event)
        except Exception as exc:
            logger.warning("ThresholdGuard: could not record event: %s", exc)


def _deletion_ratio(raw_chars: int, cleaned_chars: int) -> float:
    if raw_chars == 0:
        return 0.0
    return max(0.0, (raw_chars - cleaned_chars) / raw_chars)

