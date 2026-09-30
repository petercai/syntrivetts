import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from syntrive.db.models import Base

logger = logging.getLogger(__name__)

SQLITE_BUSY_TIMEOUT_MS = 30_000


def make_engine(db_path: Path):
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _set_wal_mode(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")

    logger.debug("SQLite engine created: %s", url)
    return engine


def seal_database_file(db_path: Path) -> str:
    import gc
    import sqlite3
    from contextlib import closing

    gc.collect()
    with closing(sqlite3.connect(str(db_path), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)) as conn:
        busy, frames, moved = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        try:
            mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        except sqlite3.OperationalError as exc:
            mode = "wal"
            logger.info("db_file_journal_kept: path=%s reason=%s", db_path, exc)
    logger.info("db_file_sealed: path=%s checkpoint_busy=%s frames=%s moved=%s journal_mode=%s",
                db_path, busy, frames, moved, mode)
    return mode


def create_tables(engine) -> None:
    Base.metadata.create_all(engine)
    logger.debug("Database tables ensured.")


def _apply_schema_migrations(engine) -> None:
    try:
        with engine.connect() as conn:
            books_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(books)")).fetchall()
            }
            if "cover" not in books_cols:
                conn.execute(text("ALTER TABLE books ADD COLUMN cover VARCHAR(1024)"))
                conn.commit()
                logger.info("Schema migration M001: added books.cover column")

            oc_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(output_configs)")).fetchall()
            }
            if "final_name" in oc_cols:
                conn.execute(text("ALTER TABLE output_configs DROP COLUMN final_name"))
                conn.commit()
                logger.info("Schema migration M002: dropped output_configs.final_name column")

            jobs_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "merge_mode_override" not in jobs_cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN merge_mode_override VARCHAR(32)"))
                conn.commit()
                logger.info("Schema migration M003: added jobs.merge_mode_override column")

            jobs_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "chapter_number_reset_per_volume" not in jobs_cols:
                conn.execute(
                    text(
                        "ALTER TABLE jobs ADD COLUMN "
                        "chapter_number_reset_per_volume BOOLEAN DEFAULT 0"
                    )
                )
                conn.commit()
                logger.info(
                    "Schema migration M005: added jobs.chapter_number_reset_per_volume column"
                )

            books_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(books)")).fetchall()
            }
            if "extraction_mode" not in books_cols:
                conn.execute(
                    text("ALTER TABLE books ADD COLUMN extraction_mode VARCHAR(32) DEFAULT 'auto'")
                )
                conn.commit()
                logger.info("Schema migration M006: added books.extraction_mode column")

            books_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(books)")).fetchall()
            }
            if "extraction_mode" in books_cols:
                needs_update = conn.execute(
                    text(
                        "SELECT 1 FROM books "
                        "WHERE extraction_mode != 'none' OR extraction_mode IS NULL LIMIT 1"
                    )
                ).first()
                if needs_update is not None:
                    result = conn.execute(
                        text(
                            "UPDATE books SET extraction_mode = 'none' "
                            "WHERE extraction_mode != 'none' OR extraction_mode IS NULL"
                        )
                    )
                    conn.commit()
                    logger.info(
                        "Schema migration M008: updated %d book record(s) "
                        "extraction_mode -> 'none'",
                        result.rowcount,
                    )

            tc_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(tts_configs)")).fetchall()
            }
            if "offline_mode" not in tc_cols:
                conn.execute(
                    text("ALTER TABLE tts_configs ADD COLUMN offline_mode BOOLEAN DEFAULT 0")
                )
                conn.commit()
                logger.info("Schema migration M009: added tts_configs.offline_mode column")

            jobs_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "offline_mode" in jobs_cols:
                conn.execute(text("ALTER TABLE jobs DROP COLUMN offline_mode"))
                conn.commit()
                logger.info("Schema migration M010: dropped jobs.offline_mode column")

            if "fine_tuned_model" not in tc_cols:
                conn.execute(
                    text(
                        "ALTER TABLE tts_configs ADD COLUMN fine_tuned_model "
                        "VARCHAR(256) DEFAULT 'internal'"
                    )
                )
                conn.commit()
                logger.info("Schema migration M011: added tts_configs.fine_tuned_model column")

            jobs_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "chapters_dir" in jobs_cols and "chapter_audio_dir" not in jobs_cols:
                conn.execute(
                    text("ALTER TABLE jobs RENAME COLUMN chapters_dir TO chapter_audio_dir")
                )
                conn.commit()
                logger.info(
                    "Schema migration M012: renamed jobs.chapters_dir -> chapter_audio_dir"
                )

            jobs_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "chapters_dir_sentences" in jobs_cols and "sentence_audio_dir" not in jobs_cols:
                conn.execute(
                    text(
                        "ALTER TABLE jobs RENAME COLUMN chapters_dir_sentences "
                        "TO sentence_audio_dir"
                    )
                )
                conn.commit()
                logger.info(
                    "Schema migration M013: renamed jobs.chapters_dir_sentences "
                    "-> sentence_audio_dir"
                )

            jobs_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "transcript_dir" not in jobs_cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN transcript_dir VARCHAR(1024)"))
                conn.commit()
                logger.info("Schema migration M014: added jobs.transcript_dir column")

            tc_cols = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(transcript_chapters)")
                ).fetchall()
            }
            if "text_path" in tc_cols and "transcript_path" not in tc_cols:
                conn.execute(
                    text(
                        "ALTER TABLE transcript_chapters RENAME COLUMN "
                        "text_path TO transcript_path"
                    )
                )
                conn.commit()
                logger.info(
                    "Schema migration M015: renamed transcript_chapters.text_path "
                    "-> transcript_path"
                )

            tc_cols = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(transcript_chapters)")
                ).fetchall()
            }
            if "tsv_path" in tc_cols:
                conn.execute(text("ALTER TABLE transcript_chapters DROP COLUMN tsv_path"))
                conn.commit()
                logger.info("Schema migration M016: dropped transcript_chapters.tsv_path column")

            needs_cover_backfill = conn.execute(
                text(
                    "SELECT 1 FROM jobs WHERE cover_path IS NOT ("
                    "  SELECT books.cover FROM books WHERE books.id = jobs.book_id"
                    ") LIMIT 1"
                )
            ).first()
            if needs_cover_backfill is not None:
                result = conn.execute(
                    text(
                        "UPDATE jobs SET cover_path = ("
                        "  SELECT books.cover FROM books WHERE books.id = jobs.book_id"
                        ") WHERE cover_path IS NOT ("
                        "  SELECT books.cover FROM books WHERE books.id = jobs.book_id"
                        ")"
                    )
                )
                conn.commit()
                logger.info(
                    "Schema migration M017: backfilled cover_path on %d job record(s) "
                    "from books.cover",
                    result.rowcount,
                )

            tc_cols = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(transcript_chapters)")
                ).fetchall()
            }
            if "transcript_lines" not in tc_cols:
                conn.execute(
                    text("ALTER TABLE transcript_chapters ADD COLUMN transcript_lines INTEGER")
                )
                conn.commit()
                logger.info(
                    "Schema migration M018: added transcript_chapters.transcript_lines column"
                )

            tables = {
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            }
            if "character_voice_mappings" in tables:
                conn.execute(text("DROP TABLE character_voice_mappings"))
                conn.commit()
                logger.info("Schema migration M019: dropped character_voice_mappings table")

            rv_cols = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(reference_voices)")
                ).fetchall()
            }
            if rv_cols and "duration_seconds" not in rv_cols:
                conn.execute(
                    text("ALTER TABLE reference_voices ADD COLUMN duration_seconds FLOAT")
                )
                conn.commit()
                logger.info("Schema migration M020: added reference_voices.duration_seconds column")

            tc_cols_m021 = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(tts_configs)")).fetchall()
            }
            if "voice_name" in tc_cols_m021:
                conn.execute(text("ALTER TABLE tts_configs DROP COLUMN voice_name"))
                conn.commit()
                logger.info("Schema migration M021: dropped tts_configs.voice_name column")
            if "voice_path" in tc_cols_m021:
                conn.execute(text("ALTER TABLE tts_configs DROP COLUMN voice_path"))
                conn.commit()
                logger.info("Schema migration M021: dropped tts_configs.voice_path column")

            jobs_cols_m022 = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if "repo_dir" in jobs_cols_m022:
                conn.execute(text("ALTER TABLE jobs DROP COLUMN repo_dir"))
                conn.commit()
                logger.info("Schema migration M022: dropped jobs.repo_dir column")

            tc_cols_m023 = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(transcript_chapters)")
                ).fetchall()
            }
            if "text_path" in tc_cols_m023:
                conn.execute(text("ALTER TABLE transcript_chapters DROP COLUMN text_path"))
                conn.commit()
                logger.info("Schema migration M023: dropped transcript_chapters.text_path column")

            tc_cols_m024 = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(transcript_chapters)")
                ).fetchall()
            }
            m024_columns = {
                "volume": "VARCHAR(512)",
                "volume_number": "VARCHAR(16)",
                "sequence_number": "VARCHAR(16)",
                "chapter_name": "VARCHAR(512)",
                "chapter_number": "VARCHAR(16)",
            }
            for col_name, col_type in m024_columns.items():
                if col_name not in tc_cols_m024:
                    conn.execute(
                        text(f"ALTER TABLE transcript_chapters ADD COLUMN {col_name} {col_type}")
                    )
                    conn.commit()
                    logger.info(
                        "Schema migration M024: added transcript_chapters.%s column", col_name
                    )

            tc_cols_m025 = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(transcript_chapters)")
                ).fetchall()
            }
            m025_columns = {
                "synthesis_batch_id": "INTEGER",
                "chapter_audio_path": "VARCHAR(1024)",
                "chapter_audio_seconds": "FLOAT",
            }
            for col_name, col_type in m025_columns.items():
                if col_name not in tc_cols_m025:
                    conn.execute(
                        text(f"ALTER TABLE transcript_chapters ADD COLUMN {col_name} {col_type}")
                    )
                    conn.commit()
                    logger.info(
                        "Schema migration M025: added transcript_chapters.%s column", col_name
                    )

            tts_cfg_cols = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(tts_configs)")).fetchall()
            }
            if "model" not in tts_cfg_cols:
                conn.execute(text("ALTER TABLE tts_configs ADD COLUMN model VARCHAR(128)"))
                conn.commit()
                logger.info("Schema migration M026: added tts_configs.model column")

            sb_cols_m027 = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(synthesis_batches)")
                ).fetchall()
            }
            if sb_cols_m027 and "paused_at" not in sb_cols_m027:
                conn.execute(text("ALTER TABLE synthesis_batches ADD COLUMN paused_at DATETIME"))
                conn.commit()
                logger.info("Schema migration M027: added synthesis_batches.paused_at column")

            tables_m028 = {
                row[0]
                for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            }
            if "speech_event_records" in tables_m028:
                dropped_rows = conn.execute(
                    text("SELECT COUNT(*) FROM speech_event_records")
                ).scalar()
                conn.execute(text("DROP TABLE speech_event_records"))
                conn.commit()
                logger.info(
                    "Schema migration M028: dropped speech_event_records table (%d row(s))",
                    dropped_rows,
                )

            needs_sentence_audio_repair = conn.execute(
                text(
                    "SELECT 1 FROM jobs "
                    "WHERE sentence_audio_dir = process_dir || '/chapters/sentences' LIMIT 1"
                )
            ).first()
            if needs_sentence_audio_repair is not None:
                repaired_sentence_audio_paths = conn.execute(
                    text(
                        "UPDATE jobs "
                        "SET sentence_audio_dir = process_dir || '/sentence_audio' "
                        "WHERE sentence_audio_dir = process_dir || '/chapters/sentences'"
                    )
                ).rowcount
                conn.commit()
                logger.info(
                    "Schema migration M029: repaired jobs.sentence_audio_dir (%d row(s))",
                    repaired_sentence_audio_paths,
                )

            jobs_cols_m030 = {
                row[1] for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            }
            if jobs_cols_m030 and "archived_at" not in jobs_cols_m030:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN archived_at DATETIME"))
                conn.commit()
                logger.info("Schema migration M030: added jobs.archived_at column")

    except Exception as exc:
        logger.warning("Schema migration failed (non-fatal): %s", exc)


def ensure_schema(engine, seed: bool = True) -> None:
    create_tables(engine)
    _apply_schema_migrations(engine)
    if seed:
        _seed_default_data(engine)


def _seed_default_data(engine) -> None:
    try:
        with Session(engine) as db:
            from syntrive.db.seed import seed_cleaning_rules
            added = seed_cleaning_rules(db)
            if added:
                logger.info("DB seed: inserted %d new cleaning rule(s)", added)
    except Exception as exc:
        logger.warning("DB seed failed (non-fatal): %s", exc)


def ensure_db_schema(db_path: Path) -> None:
    engine = make_engine(db_path)
    try:
        ensure_schema(engine)
    finally:
        engine.dispose()


@contextmanager
def get_db_session(db_path: Path):
    engine = make_engine(db_path)
    ensure_schema(engine)
    db = Session(engine)
    try:
        yield db
        db.commit()
        logger.debug("DB session committed.")
    except Exception:
        db.rollback()
        logger.warning("DB session rolled back due to exception.", exc_info=True)
        raise
    finally:
        db.close()
        engine.dispose()
