from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from syntrive.db.models import SentenceReviewEvent

logger = logging.getLogger(__name__)


class SentenceReviewRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def record_delete(
        self,
        *,
        job_id: int,
        transcript_chapter_id: int,
        sentence_index: int,
        trash_path: str,
        description_tag_snapshot: Optional[str],
        issue_category: Optional[str] = None,
        note: Optional[str] = None,
    ) -> SentenceReviewEvent:
        event = SentenceReviewEvent(
            job_id=job_id,
            transcript_chapter_id=transcript_chapter_id,
            sentence_index=sentence_index,
            action="deleted",
            issue_category=issue_category,
            description_tag_snapshot=description_tag_snapshot,
            trash_path=trash_path,
            note=note,
            created_at=datetime.utcnow(),
        )
        self._db.add(event)
        self._db.flush()
        return event

    def get_open_delete(
        self, *, job_id: int, transcript_chapter_id: int, sentence_index: int
    ) -> Optional[SentenceReviewEvent]:
        return (
            self._db.query(SentenceReviewEvent)
            .filter(
                SentenceReviewEvent.job_id == job_id,
                SentenceReviewEvent.transcript_chapter_id == transcript_chapter_id,
                SentenceReviewEvent.sentence_index == sentence_index,
                SentenceReviewEvent.action == "deleted",
            )
            .order_by(SentenceReviewEvent.created_at.desc())
            .first()
        )

    def get(self, event_id: int) -> Optional[SentenceReviewEvent]:
        return self._db.query(SentenceReviewEvent).filter(SentenceReviewEvent.id == event_id).first()

    def record_resolved(self, event_id: int) -> None:
        event = self.get(event_id)
        if event is None:
            logger.warning("sentence_review_record_resolved_missing_event: event_id=%s", event_id)
            return
        event.action = "resolved"
        event.resolved_at = datetime.utcnow()
        event.trash_path = None
        self._db.flush()

    def delete_event(self, event_id: int) -> None:
        event = self.get(event_id)
        if event is None:
            logger.warning("sentence_review_delete_event_missing: event_id=%s", event_id)
            return
        self._db.delete(event)
        self._db.flush()

    def list_events_for_chapter(self, *, job_id: int, transcript_chapter_id: int) -> list[SentenceReviewEvent]:
        return (
            self._db.query(SentenceReviewEvent)
            .filter(
                SentenceReviewEvent.job_id == job_id,
                SentenceReviewEvent.transcript_chapter_id == transcript_chapter_id,
            )
            .order_by(SentenceReviewEvent.sentence_index, SentenceReviewEvent.created_at)
            .all()
        )

    def list_open_deletes_for_chapter(
        self, *, job_id: int, transcript_chapter_id: int
    ) -> list[SentenceReviewEvent]:
        return (
            self._db.query(SentenceReviewEvent)
            .filter(
                SentenceReviewEvent.job_id == job_id,
                SentenceReviewEvent.transcript_chapter_id == transcript_chapter_id,
                SentenceReviewEvent.action == "deleted",
            )
            .all()
        )
