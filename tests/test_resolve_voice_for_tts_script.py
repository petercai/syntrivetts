from __future__ import annotations

from pathlib import Path

import pytest


def _seed(db_path: Path) -> int:
    from syntrive.db.session import get_db_session
    from syntrive.db.models import Book, Job, TtsConfig

    with get_db_session(db_path) as db:
        book = Book(title="Test Book")
        db.add(book)
        db.flush()
        job = Job(
            book_id=book.id,
            process_dir=str(db_path.parent),
            epub_path=str(db_path.parent / "book.epub"),
        )
        db.add(job)
        db.flush()
        cfg = TtsConfig(job_id=job.id)
        db.add(cfg)
        db.flush()
        return cfg.id


class TestResolveVoiceForTtsScript:
    def test_returns_none_when_narrator_unbound(self, tmp_path: Path) -> None:
        from syntrive.adapters.tts.engines.base import BaseTTSEngine
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)
        engine = BaseTTSEngine()

        with get_db_session(db_path) as db:
            result = engine.resolve_voice_for_tts_script(cfg_id, voice_id=1, db=db, cache={})
        assert result is None

    def test_returns_narrator_path_when_role_unbound(self, tmp_path: Path) -> None:
        from syntrive.adapters.tts.engines.base import BaseTTSEngine
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            db.add(ref)
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=ref.id, excluded=False,
            ))

        with get_db_session(db_path) as db:
            result = engine_resolve(db, cfg_id, 3)
        assert result is not None
        assert result.endswith("voices/en/adult/male/narrator.wav")

    def test_falls_back_to_narrator_when_role_excluded(self, tmp_path: Path) -> None:
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            narrator_ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            role_ref = ReferenceVoice(
                path="voices/en/adult/female/role.wav", name="role",
                gender="female", language="en",
            )
            db.add_all([narrator_ref, role_ref])
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=narrator_ref.id, excluded=False,
            ))
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="哲人", voice_id=1,
                reference_voice_id=role_ref.id, excluded=True,
            ))

        with get_db_session(db_path) as db:
            result = engine_resolve(db, cfg_id, 1)
        assert result is not None
        assert result.endswith("narrator.wav")

    def test_falls_back_to_narrator_when_role_has_no_reference_voice(self, tmp_path: Path) -> None:
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            narrator_ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            db.add(narrator_ref)
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=narrator_ref.id, excluded=False,
            ))
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="青年", voice_id=2,
                reference_voice_id=None, excluded=False,
            ))

        with get_db_session(db_path) as db:
            result = engine_resolve(db, cfg_id, 2)
        assert result is not None
        assert result.endswith("narrator.wav")

    def test_uses_role_own_reference_voice_when_bound(self, tmp_path: Path) -> None:
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            narrator_ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            role_ref = ReferenceVoice(
                path="voices/en/adult/female/role.wav", name="role",
                gender="female", language="en",
            )
            db.add_all([narrator_ref, role_ref])
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=narrator_ref.id, excluded=False,
            ))
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="青年", voice_id=2,
                reference_voice_id=role_ref.id, excluded=False,
            ))

        with get_db_session(db_path) as db:
            result = engine_resolve(db, cfg_id, 2)
        assert result is not None
        assert result.endswith("role.wav")

    def test_narrator_itself_resolves_via_voice_id_zero(self, tmp_path: Path) -> None:
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            db.add(ref)
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=ref.id, excluded=False,
            ))

        with get_db_session(db_path) as db:
            result = engine_resolve(db, cfg_id, 0)
        assert result is not None
        assert result.endswith("narrator.wav")

    def test_cache_memoizes_and_is_reused_across_calls(self, tmp_path: Path) -> None:
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            db.add(ref)
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=ref.id, excluded=False,
            ))

        cache: dict = {}
        with get_db_session(db_path) as db:
            first = engine_resolve(db, cfg_id, 0, cache=cache)
        assert first is not None

        with get_db_session(db_path) as db:
            db.query(TtsVoice).filter_by(tts_config_id=cfg_id, voice_id=0).delete()

        with get_db_session(db_path) as db:
            second = engine_resolve(db, cfg_id, 0, cache=cache)
        assert second == first

    def test_returns_absolute_path_string(self, tmp_path: Path) -> None:
        from syntrive.db.models import ReferenceVoice, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "db.sqlite"
        cfg_id = _seed(db_path)

        with get_db_session(db_path) as db:
            ref = ReferenceVoice(
                path="voices/en/adult/male/narrator.wav", name="narrator",
                gender="male", language="en",
            )
            db.add(ref)
            db.flush()
            db.add(TtsVoice(
                tts_config_id=cfg_id, name="narrator", voice_id=0,
                reference_voice_id=ref.id, excluded=False,
            ))

        with get_db_session(db_path) as db:
            result = engine_resolve(db, cfg_id, 0)
        assert Path(result).is_absolute()


def engine_resolve(db, cfg_id: int, voice_id: int, cache: dict | None = None) -> str | None:
    from syntrive.adapters.tts.engines.base import BaseTTSEngine

    return BaseTTSEngine().resolve_voice_for_tts_script(
        cfg_id, voice_id, db=db, cache=cache if cache is not None else {}
    )
