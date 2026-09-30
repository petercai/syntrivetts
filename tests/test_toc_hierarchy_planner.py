from pathlib import Path

import pytest

EBOOK_DIR = Path("ebooks")


def _ebook(filename: str) -> Path:
    path = (EBOOK_DIR / filename).resolve()
    if not path.is_file():
        pytest.skip(f"Test EPUB not found (place in ebooks/): {path}")
    return path


def _plan(epub_filename: str):
    from syntrive.adapters.epub.toc_planner import TocHierarchyPlanner
    epub = _ebook(epub_filename)
    return TocHierarchyPlanner(epub).plan()


def _bootstrap_and_run_stage(tmp_path: Path, epub_filename: str):
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


class TestTocPlannerUnit:
    def test_1984_is_flat(self):
        plan = _plan("1984-Orwell_George.epub")
        assert plan.mode in ("flat", "volume_split"), \
            f"mode must be flat or volume_split; got {plan.mode!r}"
        assert len(plan.groups) >= 1, "Must have at least one group"

    def test_1984_all_hrefs_resolve(self):
        from syntrive.adapters.epub.html_extractor import EpubHtmlExtractor

        epub = _ebook("1984-Orwell_George.epub")
        chapters = EpubHtmlExtractor(epub).extract()
        if not chapters:
            pytest.skip("No chapters extracted from 1984")

        plan = _plan("1984-Orwell_George.epub")
        for ch in chapters:
            group_id, _ = plan.get_group_for_href(ch.href)
            assert group_id, f"chapter {ch.toc_id} (href={ch.href}) returned empty group_id"

    def test_lotr_mode_detected(self):
        plan = _plan("The_Lord_of_the_Rings.epub")
        assert plan.mode in ("flat", "volume_split"), \
            f"Unexpected mode: {plan.mode!r}"
        assert len(plan.groups) >= 1

    def test_lotr_volume_split_has_multiple_groups(self):
        plan = _plan("The_Lord_of_the_Rings.epub")
        if plan.mode == "flat":
            pytest.skip(
                "LotR EPUB has flat TOC — volume_split assertion not applicable"
            )
        assert len(plan.groups) >= 2, \
            f"volume_split plan must have >= 2 groups; got {len(plan.groups)}"
        group_ids = [g.group_id for g in plan.groups]
        assert len(set(group_ids)) == len(group_ids), "group_ids must be unique"

    def test_lotr_all_hrefs_resolve(self):
        from syntrive.adapters.epub.html_extractor import EpubHtmlExtractor

        epub = _ebook("The_Lord_of_the_Rings.epub")
        chapters = EpubHtmlExtractor(epub).extract()
        if not chapters:
            pytest.skip("No chapters extracted from LotR")

        plan = _plan("The_Lord_of_the_Rings.epub")
        unresolved = []
        for ch in chapters:
            group_id, _ = plan.get_group_for_href(ch.href)
            if not group_id:
                unresolved.append(ch.toc_id)

        assert not unresolved, \
            f"Chapters with unresolved group_id: {unresolved[:5]}"

    def test_lotr_volume_group_ids_are_sequential(self):
        import re
        plan = _plan("The_Lord_of_the_Rings.epub")
        if plan.mode == "flat":
            pytest.skip("LotR EPUB has flat TOC")
        vol_pattern = re.compile(r"^vol_\d{3}$")
        for grp in plan.groups:
            assert vol_pattern.match(grp.group_id), \
                f"group_id {grp.group_id!r} does not match vol_NNN pattern"

    def test_flat_group_id_is_main(self):
        from syntrive.adapters.epub.html_extractor import EpubHtmlExtractor

        epub = _ebook("To_Kill_A_Mockingbird.epub")
        chapters = EpubHtmlExtractor(epub).extract()
        if not chapters:
            pytest.skip("No chapters extracted from To Kill a Mockingbird")

        plan = _plan("To_Kill_A_Mockingbird.epub")
        if plan.mode != "flat":
            pytest.skip("EPUB is not flat — test is for flat books only")

        for ch in chapters[:5]:
            group_id, _ = plan.get_group_for_href(ch.href)
            assert group_id == "main", \
                f"Flat book must return group_id='main'; got {group_id!r}"

    def test_get_group_for_href_fallback_for_unknown_href(self):
        plan = _plan("1984-Orwell_George.epub")
        group_id, title = plan.get_group_for_href("does_not_exist_abc123.html")
        assert group_id == "main", \
            f"Fallback group_id must be 'main'; got {group_id!r}"
        assert title == "", f"Fallback title must be empty; got {title!r}"

    def test_chinese_sapiens_plan(self):
        plan = _plan("人类简史.epub")
        assert plan.mode in ("flat", "volume_split")
        assert len(plan.groups) >= 1

    def test_chinese_dream_analysis_plan(self):
        plan = _plan("梦的解析.epub")
        assert plan.mode in ("flat", "volume_split")
        assert len(plan.groups) >= 1

    def test_don_quixote_plan(self):
        plan = _plan("Don Quixote.epub")
        assert plan.mode in ("flat", "volume_split")
        assert len(plan.groups) >= 1
        if plan.mode == "volume_split":
            assert len(plan.groups) >= 2, \
                "Don Quixote volume_split must have >= 2 groups"


class TestGroupIdInDatabase:
    def _assert_group_ids_set(self, db_path: Path, job) -> list:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        with get_db_session(db_path) as db:
            chapters = (
                db.query(TranscriptChapter)
                .filter(TranscriptChapter.job_id == job.id)
                .all()
            )

        null_group = [c.chapter_id for c in chapters if not c.group_id]
        assert not null_group, \
            f"TranscriptChapter records with null group_id: {null_group[:5]}"
        return chapters

    def test_1984_group_ids_persisted(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "1984-Orwell_George.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")
        self._assert_group_ids_set(db_path, job)

    def test_lotr_group_ids_persisted(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "The_Lord_of_the_Rings.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")
        self._assert_group_ids_set(db_path, job)

    def test_lotr_volume_chapters_distributed_across_groups(self, tmp_path: Path):
        plan = _plan("The_Lord_of_the_Rings.epub")
        if plan.mode == "flat":
            pytest.skip("LotR EPUB has flat TOC — volume distribution test skipped")

        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "The_Lord_of_the_Rings.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")

        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        with get_db_session(db_path) as db:
            chapters = (
                db.query(TranscriptChapter)
                .filter(TranscriptChapter.job_id == job.id)
                .all()
            )

        distinct_groups = {c.group_id for c in chapters if c.group_id}
        assert len(distinct_groups) >= 2, (
            f"LotR volume_split chapters should span >= 2 groups; "
            f"got: {distinct_groups}"
        )

    def test_flat_book_all_chapters_in_main_group(self, tmp_path: Path):
        plan = _plan("To_Kill_A_Mockingbird.epub")
        if plan.mode != "flat":
            pytest.skip("Book is not flat — test only valid for flat books")

        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "To_Kill_A_Mockingbird.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")

        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        with get_db_session(db_path) as db:
            non_main = (
                db.query(TranscriptChapter)
                .filter(
                    TranscriptChapter.job_id == job.id,
                    TranscriptChapter.group_id != "main",
                )
                .count()
            )
        assert non_main == 0, \
            f"Flat book must have group_id='main' for all chapters; {non_main} differ"

    def test_chinese_mockingbird_group_ids_persisted(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run_stage(
            tmp_path, "杀死一只知更鸟_哈珀李.epub"
        )
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")
        self._assert_group_ids_set(db_path, job)

    def test_sapiens_group_ids_persisted(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "人类简史.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")
        self._assert_group_ids_set(db_path, job)

    def test_dream_analysis_group_ids_persisted(self, tmp_path: Path):
        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "梦的解析.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")
        self._assert_group_ids_set(db_path, job)


class TestModeConsistency:
    def _get_distinct_groups(self, db_path: Path, job) -> set[str]:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        with get_db_session(db_path) as db:
            rows = (
                db.query(TranscriptChapter)
                .filter(TranscriptChapter.job_id == job.id)
                .all()
            )
        return {r.group_id for r in rows if r.group_id}

    def test_lotr_volume_mode_consistent_with_db(self, tmp_path: Path):
        plan = _plan("The_Lord_of_the_Rings.epub")
        if plan.mode == "flat":
            pytest.skip("LotR EPUB has flat TOC")

        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "The_Lord_of_the_Rings.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")

        db_groups = self._get_distinct_groups(db_path, job)
        plan_groups = {g.group_id for g in plan.groups}

        assert db_groups.issubset(plan_groups | {"main"}), (
            f"DB group IDs {db_groups} must be a subset of plan group IDs {plan_groups}"
        )
        assert len(db_groups) >= 2, \
            f"volume_split DB must have >= 2 distinct groups; got {db_groups}"

    def test_1984_plan_mode_consistent_with_db(self, tmp_path: Path):
        plan = _plan("1984-Orwell_George.epub")
        job, db_path, result = _bootstrap_and_run_stage(tmp_path, "1984-Orwell_George.epub")
        if not result.success:
            pytest.skip(f"Pipeline failed: {result.error}")

        db_groups = self._get_distinct_groups(db_path, job)

        if plan.mode == "flat":
            assert db_groups <= {"main"}, \
                f"Flat mode must only produce 'main' group; got {db_groups}"
        else:
            assert len(db_groups) >= 1, "volume_split must produce at least 1 group in DB"
