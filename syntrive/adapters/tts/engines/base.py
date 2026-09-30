from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BaseTTSEngine:
    def __init__(self, options=None):
        ...

    def synthesize(self, text: str, voice=None, leading_silence_ms: int = 0) -> bool:
        ...

    def resolve_voice_for_tts_script(
        self, tts_config_id: int, voice_id: int, db, cache: dict,
    ) -> Optional[str]:
        from syntrive.db.models import ReferenceVoice, TtsVoice

        key = (tts_config_id, voice_id)
        if key in cache:
            return cache[key]

        def _bound_reference_voice_id(vid: int) -> Optional[int]:
            row = db.query(TtsVoice).filter_by(tts_config_id=tts_config_id, voice_id=vid).first()
            if row is None or row.excluded or row.reference_voice_id is None:
                return None
            return row.reference_voice_id

        reference_voice_id = _bound_reference_voice_id(voice_id)
        if reference_voice_id is None and voice_id != 0:
            reference_voice_id = _bound_reference_voice_id(0)

        resolved_path: Optional[str] = None
        if reference_voice_id is not None:
            ref = db.query(ReferenceVoice).filter_by(id=reference_voice_id).first()
            if ref is not None:
                from syntrive.io.paths import resolve_project_path

                resolved_path = str(resolve_project_path(ref.path))
            else:
                logger.warning(
                    "resolve_voice_for_tts_script: reference_voice_id=%d no longer exists "
                    "(tts_config_id=%d voice_id=%d)",
                    reference_voice_id, tts_config_id, voice_id,
                )

        cache[key] = resolved_path
        return resolved_path
