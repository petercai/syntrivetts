import json
import re
from pathlib import Path

import pytest

EBOOK_DIR = Path("ebooks")


def _ebook(filename: str) -> Path:
    path = (EBOOK_DIR / filename).resolve()
    if not path.is_file():
        pytest.skip(f"Test EPUB not found (place in ebooks/): {path}")
    return path


def _bootstrap_and_run(tmp_path: Path, epub_filename: str):
    from syntrive.bootstrap import bootstrap
    from syntrive.pipeline.transcript_html_stage import TranscriptHtmlStage

    ebook = _ebook(epub_filename)
    job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)
    db_path = tmp_path / "syntrivetts.db"

    stage = TranscriptHtmlStage(
        job=job,
        db_path=db_path,
    )
    result = stage.run()
    return job, db_path, result


def _assert_common(tmp_path: Path, job, result, min_chapters: int = 1):
    process_dir = Path(job.process_dir)
    transcript_dir = process_dir / "transcript_html"

    assert result.success, f"Stage must succeed; error={result.error}"
    assert result.artifacts.get("html", 0) >= min_chapters, (
        f"Expected >= {min_chapters} HTML chapters; got {result.artifacts}"
    )

    for sub in ("raw", "cleaned", "manifest"):
        assert (transcript_dir / sub).is_dir(), f"Missing subdir: transcript_html/{sub}"

    raw_files = list((transcript_dir / "raw").glob("*.html"))
    cleaned_files = list((transcript_dir / "cleaned").glob("ch_*.html"))
    manifest_files = list((transcript_dir / "manifest").glob("*.para_map.json"))

    assert len(raw_files) >= min_chapters, \
        f"Expected >= {min_chapters} raw HTML files; got {len(raw_files)}"
    assert len(cleaned_files) >= min_chapters, \
        f"Expected >= {min_chapters} cleaned HTML files; got {len(cleaned_files)}"
    assert len(manifest_files) >= min_chapters, \
        f"Expected >= {min_chapters} para_map.json files; got {len(manifest_files)}"


def _assert_para_ids_valid(tmp_path: Path, job):
    process_dir = Path(job.process_dir)
    manifest_dir = process_dir / "transcript_html" / "manifest"
    id_pattern = re.compile(r"^p[0-9a-f]{8}(_\d+)?$")

    total_paragraphs = 0
    for json_file in manifest_dir.glob("*.para_map.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data.get("version") == "1", f"para_map version must be '1' in {json_file}"
        assert data.get("hash_algo") == "sha256", f"hash_algo must be sha256 in {json_file}"
        for entry in data.get("paragraphs", []):
            pid = entry.get("id", "")
            assert id_pattern.match(pid), \
                f"Invalid para ID '{pid}' in {json_file} — must match p[0-9a-f]{{8}}"
            total_paragraphs += 1

    return total_paragraphs


def _assert_db_chapters(db_path: Path, job, min_count: int = 1):
    from syntrive.db.session import get_db_session
    from syntrive.db.models import TranscriptChapter

    with get_db_session(db_path) as db:
        count = (
            db.query(TranscriptChapter)
            .filter(TranscriptChapter.job_id == job.id)
            .count()
        )
    assert count >= min_count, \
        f"Expected >= {min_count} TranscriptChapter records; got {count}"
    return count


def _assert_group_ids_not_null(db_path: Path, job):
    from syntrive.db.session import get_db_session
    from syntrive.db.models import TranscriptChapter

    with get_db_session(db_path) as db:
        rows = (
            db.query(TranscriptChapter)
            .filter(TranscriptChapter.job_id == job.id)
            .all()
        )
    null_ids = [r.chapter_id for r in rows if not r.group_id]
    assert not null_ids, \
        f"TranscriptChapter rows with null/empty group_id: {null_ids[:5]}"
    return {r.group_id for r in rows}


class TestTranscriptHtmlStage1984:
    EPUB = "1984-Orwell_George.epub"
    MIN_CHAPTERS = 5

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_para_ids_are_hash_based(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        total = _assert_para_ids_valid(tmp_path, job)
        assert total > 0, "Must produce at least 1 tagged paragraph"

    def test_english_content_preserved(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        process_dir = Path(job.process_dir)
        cleaned_files = list((process_dir / "transcript_html" / "cleaned").glob("ch_*.html"))
        assert len(cleaned_files) > 0
        combined = " ".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in cleaned_files[:3]
        )
        english_words = ["the", "and", "was", "of", "in"]
        matches = sum(1 for w in english_words if w in combined.lower())
        assert matches >= 3, \
            "Expected common English words to survive; EnglishBlockRule may be mis-firing"

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)

    def test_deletion_ratio_not_excessive(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success

        with get_db_session(db_path) as db:
            chapters = (
                db.query(TranscriptChapter)
                .filter(TranscriptChapter.job_id == job.id)
                .all()
            )
        over_deleted = [
            c for c in chapters
            if c.deletion_ratio is not None and c.deletion_ratio > 0.40
        ]
        assert len(over_deleted) == 0, (
            f"{len(over_deleted)} chapters exceed 40% deletion ratio: "
            + str([(c.chapter_id, c.deletion_ratio) for c in over_deleted])
        )


class TestTranscriptHtmlStageLotR:
    EPUB = "The_Lord_of_the_Rings.epub"
    MIN_CHAPTERS = 5

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_para_ids_are_hash_based(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        total = _assert_para_ids_valid(tmp_path, job)
        assert total > 0

    def test_artifacts_written_to_disk(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)


class TestTranscriptHtmlStageDonQuixote:
    EPUB = "Don Quixote.epub"
    MIN_CHAPTERS = 5

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_para_ids_are_hash_based(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_para_ids_valid(tmp_path, job)

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)


class TestTranscriptHtmlStageMockingbirdEN:
    EPUB = "To_Kill_A_Mockingbird.epub"
    MIN_CHAPTERS = 3

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_english_content_not_over_deleted(self, tmp_path: Path):
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success

        with get_db_session(db_path) as db:
            chapters = (
                db.query(TranscriptChapter)
                .filter(TranscriptChapter.job_id == job.id)
                .all()
            )
        for c in chapters:
            if c.deletion_ratio is not None:
                assert c.deletion_ratio <= 0.40, (
                    f"Chapter {c.chapter_id} has unexpectedly high deletion "
                    f"ratio {c.deletion_ratio:.1%} — EnglishBlockRule may be mis-firing"
                )

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)


class TestTranscriptHtmlStageMockingbirdZH:
    EPUB = "杀死一只知更鸟_哈珀李.epub"
    MIN_CHAPTERS = 3

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_chinese_content_preserved(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        process_dir = Path(job.process_dir)
        cleaned_files = list(
            (process_dir / "transcript_html" / "cleaned").glob("ch_*.html")
        )
        assert len(cleaned_files) > 0
        combined = " ".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in cleaned_files[:5]
        )
        chinese_char_count = sum(1 for c in combined if "一" <= c <= "鿿")
        assert chinese_char_count > 100, (
            f"Expected > 100 Chinese characters in cleaned HTML; got {chinese_char_count}"
        )

    def test_para_ids_are_hash_based(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        total = _assert_para_ids_valid(tmp_path, job)
        assert total > 0

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)


class TestTranscriptHtmlStageDreamAnalysis:
    EPUB = "梦的解析.epub"
    MIN_CHAPTERS = 3

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_chinese_content_preserved(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        process_dir = Path(job.process_dir)
        cleaned_files = list(
            (process_dir / "transcript_html" / "cleaned").glob("ch_*.html")
        )
        combined = " ".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in cleaned_files[:5]
        )
        chinese_count = sum(1 for c in combined if "一" <= c <= "鿿")
        assert chinese_count > 50, "Chinese characters must survive cleaning"

    def test_para_ids_stable_across_runs(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        if not result.success:
            pytest.skip(f"Stage failed: {result.error}")

        process_dir = Path(job.process_dir)
        manifest_dir = process_dir / "transcript_html" / "manifest"
        ids_run1: set[str] = set()
        for f in manifest_dir.glob("*.para_map.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            for entry in data.get("paragraphs", []):
                ids_run1.add(entry["id"])

        from syntrive.pipeline.transcript_html_stage import TranscriptHtmlStage
        stage2 = TranscriptHtmlStage(job=job, db_path=db_path)
        stage2.run()

        ids_run2: set[str] = set()
        for f in manifest_dir.glob("*.para_map.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            for entry in data.get("paragraphs", []):
                ids_run2.add(entry["id"])

        assert ids_run1 == ids_run2, (
            "Para IDs must be stable across runs (content-hash guarantee). "
            f"Run 1: {len(ids_run1)} IDs, Run 2: {len(ids_run2)} IDs"
        )

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)


class TestTranscriptHtmlStageSapiens:
    EPUB = "人类简史.epub"
    MIN_CHAPTERS = 3

    def test_extracts_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        _assert_common(tmp_path, job, result, self.MIN_CHAPTERS)

    def test_chinese_content_preserved(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        process_dir = Path(job.process_dir)
        cleaned_files = list(
            (process_dir / "transcript_html" / "cleaned").glob("ch_*.html")
        )
        combined = " ".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in cleaned_files[:5]
        )
        chinese_count = sum(1 for c in combined if "一" <= c <= "鿿")
        assert chinese_count > 50, "Chinese characters must survive cleaning"

    def test_para_ids_are_hash_based(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        total = _assert_para_ids_valid(tmp_path, job)
        assert total > 0

    def test_db_chapter_records_created(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_db_chapters(db_path, job, self.MIN_CHAPTERS)

    def test_group_id_set_on_all_chapters(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        _assert_group_ids_not_null(db_path, job)

    def test_stage_result_artifacts_dict_structure(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run(tmp_path, self.EPUB)
        assert result.success
        assert "html" in result.artifacts, "artifacts must contain 'html' key"
        assert "json" in result.artifacts, "artifacts must contain 'json' key"
        assert result.artifacts["html"] >= self.MIN_CHAPTERS
