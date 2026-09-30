import logging
import shutil
from pathlib import Path
from typing import NamedTuple, Optional

from syntrive.io.paths import sanitize_filename

logger = logging.getLogger(__name__)


class ImageRecord(NamedTuple):
    name: str
    rel_path: str
    image_type: str

_PROCESS_DIR_PREFIX = "PROCESSING-"
_REQUIRED_SUBDIRS = [
    "chapter_audio",
    "sentence_audio",
    "audiobooks",
    "images",
    "transcript_html/raw",
    "transcript_html/cleaned",
    "transcript_html/index",
    "transcript_html/manifest",
    "transcript_text/raw",
]


class EpubMeta(NamedTuple):
    title: Optional[str]
    author: Optional[str]
    publisher: Optional[str]
    language: Optional[str]
    publish_date: Optional[str]


def bootstrap(repo_dir: Path, ebook_path: Path):
    from syntrive.db.session import get_db_session
    from syntrive.db.repository import JobRepository
    from syntrive.db.models import Job

    stem = _safe_stem(ebook_path.name)
    process_dir = repo_dir / f"{_PROCESS_DIR_PREFIX}{stem}"
    process_dir.mkdir(parents=True, exist_ok=True)
    _create_subdirs(process_dir)

    dest_epub = _copy_ebook(ebook_path, process_dir)
    meta = _extract_metadata(dest_epub)

    db_path = repo_dir / "syntrivetts.db"
    with get_db_session(db_path) as db:
        repo = JobRepository(db)

        book = repo.find_or_create_book(
            title=meta.title or stem,
            author=meta.author,
            publisher=meta.publisher,
            language=meta.language,
            publish_date=meta.publish_date,
            original_filename=ebook_path.stem,
        )

        from syntrive.db.path_utils import to_repo_relative
        rel_process_dir = to_repo_relative(db_path, process_dir)
        chapter_audio_dir = process_dir / "chapter_audio"
        sentence_audio_dir = process_dir / "sentence_audio"
        audiobooks_dir_path = process_dir / "audiobooks"

        existing_job = (
            db.query(Job)
            .filter_by(process_dir=rel_process_dir)
            .first()
        )
        if existing_job is not None:
            logger.info(
                "Bootstrap: reusing existing job_id=%d for process_dir=%s",
                existing_job.id, rel_process_dir,
            )
            db.expunge_all()
            return existing_job

        job = repo.create_job(
            book_id=book.id,
            process_dir=rel_process_dir,
            epub_path=to_repo_relative(db_path, dest_epub),
            chapter_audio_dir=to_repo_relative(db_path, chapter_audio_dir),
            sentence_audio_dir=to_repo_relative(db_path, sentence_audio_dir),
            audiobooks_dir=to_repo_relative(db_path, audiobooks_dir_path),
        )

        db.expunge_all()
        logger.info(
            "Bootstrap complete: job_id=%d book=%s process_dir=%s",
            job.id,
            book.title,
            process_dir,
        )
        return job


_SHARED_PYPROJECT_SRC = Path(__file__).resolve().parent / "shared" / "pyproject.toml"


def copy_shared_pyproject(process_dir: Path) -> Optional[Path]:
    if not _SHARED_PYPROJECT_SRC.is_file():
        logger.warning(
            "copy_shared_pyproject: shared template not found: %s", _SHARED_PYPROJECT_SRC
        )
        return None

    dest = process_dir / "pyproject.toml"
    if dest.exists():
        logger.debug("copy_shared_pyproject: already present, skipping: %s", dest)
        return dest

    shutil.copy2(_SHARED_PYPROJECT_SRC, dest)
    logger.info("copy_shared_pyproject: %s -> %s", _SHARED_PYPROJECT_SRC, dest)
    return dest


def extract_epub_images(
    epub_path: Path,
    images_dir: Path,
    process_dir: Path,
) -> list[ImageRecord]:
    try:
        from ebooklib import epub

        book = epub.read_epub(str(epub_path), {"ignore_ncx": True})

        cover_name = _detect_cover_name(book)

        records: list[ImageRecord] = []
        count = 0
        for item in book.get_items():
            media_type = getattr(item, "media_type", None) or ""
            if not media_type.startswith("image/"):
                continue
            raw_name = item.get_name() or f"image_{count}"
            image_name = Path(raw_name).name
            if not image_name:
                image_name = f"image_{count}.bin"

            dest = images_dir / image_name
            dest.write_bytes(item.get_content())

            try:
                rel_path = dest.relative_to(process_dir).as_posix()
            except ValueError:
                rel_path = f"images/{image_name}"

            if cover_name is None and "cover" in image_name.lower():
                cover_name = image_name
                logger.debug(
                    "extract_epub_images: cover detected by filename heuristic: %s",
                    image_name,
                )

            records.append(ImageRecord(
                name=image_name,
                rel_path=rel_path,
                image_type="placeholder",
            ))
            count += 1
            logger.debug("extract_epub_images: %s -> %s (rel=%s)", raw_name, dest, rel_path)

        if cover_name is None and records:
            cover_name = records[0].name
            logger.debug(
                "extract_epub_images: cover fallback to first image: %s", cover_name
            )

        finalised: list[ImageRecord] = []
        for rec in records:
            img_type = "cover" if rec.name == cover_name else "internal"
            finalised.append(ImageRecord(name=rec.name, rel_path=rec.rel_path, image_type=img_type))

        logger.info(
            "extract_epub_images: extracted %d image(s) from %s, cover=%s",
            count, epub_path.name, cover_name,
        )
        return finalised

    except Exception as exc:
        logger.warning("extract_epub_images: failed for %s: %s", epub_path, exc)
        return []


def _detect_cover_name(book) -> Optional[str]:
    try:
        opf_covers = book.metadata.get("OPF", {}).get("cover", [])
        if opf_covers:
            cover_id = opf_covers[0][0] if opf_covers[0] else None
            if cover_id:
                item = book.get_item_with_id(cover_id)
                if item and getattr(item, "media_type", "").startswith("image/"):
                    name = Path(item.get_name()).name
                    logger.debug(
                        "_detect_cover_name: strategy=opf id=%s name=%s", cover_id, name
                    )
                    return name

        for item in book.get_items():
            if not getattr(item, "media_type", "").startswith("image/"):
                continue
            item_id = getattr(item, "id", "") or ""
            if "cover" in item_id.lower():
                name = Path(item.get_name()).name
                logger.debug(
                    "_detect_cover_name: strategy=item_id id=%s name=%s", item_id, name
                )
                return name

    except Exception as exc:
        logger.debug("_detect_cover_name: error during detection: %s", exc)

    return None


def update_book_metadata(epub_path: Path, db_path: Path, book_id: int) -> None:
    from syntrive.db.session import get_db_session
    from syntrive.db.models import Book

    meta = _extract_metadata(epub_path)
    try:
        with get_db_session(db_path) as db:
            book = db.query(Book).filter_by(id=book_id).first()
            if book is None:
                logger.warning("update_book_metadata: book_id=%d not found", book_id)
                return

            if meta.title:
                book.title = meta.title
            if meta.author:
                book.author = meta.author
            if meta.publisher:
                book.publisher = meta.publisher
            if meta.language:
                book.language = meta.language
            if meta.publish_date:
                book.publish_date = meta.publish_date

            logger.info(
                "update_book_metadata: updated book_id=%d title=%r", book_id, book.title
            )
    except Exception as exc:
        logger.warning("update_book_metadata: failed for book_id=%d: %s", book_id, exc)


def _safe_stem(filename: str) -> str:
    return sanitize_filename(Path(filename).stem)


def _create_subdirs(base: Path) -> None:
    for sub in _REQUIRED_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)
    logger.debug("Subdirectories ensured under %s", base)


def _copy_ebook(ebook_path: Path, process_dir: Path) -> Path:
    dest = process_dir / ebook_path.name
    if not dest.exists():
        shutil.copy2(ebook_path, dest)
        logger.info("Copied ebook: %s -> %s", ebook_path, dest)
    else:
        logger.debug("Ebook already in process_dir: %s", dest)
    return dest


def _extract_metadata(epub_path: Path) -> EpubMeta:
    try:
        from ebooklib import epub

        book = epub.read_epub(str(epub_path), {"ignore_ncx": True})

        def _first(items):
            return items[0] if items else None

        title = _first(book.get_metadata("DC", "title"))
        author = _first(book.get_metadata("DC", "creator"))
        publisher = _first(book.get_metadata("DC", "publisher"))
        language = _first(book.get_metadata("DC", "language"))
        date = _first(book.get_metadata("DC", "date"))

        def _val(item):
            return item[0] if isinstance(item, (list, tuple)) else item

        from syntrive.adapters.epub.lang import normalize_language
        raw_lang = _val(language)
        lang_iso1 = normalize_language(raw_lang)
        if raw_lang and lang_iso1 is None:
            logger.warning(
                "_extract_metadata: unrecognised EPUB language=%r — stored as None;"
                " user must set language via TUI (press L)",
                raw_lang,
            )

        meta = EpubMeta(
            title=_val(title),
            author=_val(author),
            publisher=_val(publisher),
            language=lang_iso1,
            publish_date=_val(date),
        )
        logger.debug("Extracted metadata: %s", meta)
        return meta

    except Exception as exc:
        logger.warning("Failed to extract EPUB metadata: %s", exc)
        return EpubMeta(
            title=None,
            author=None,
            publisher=None,
            language=None,
            publish_date=None,
        )
