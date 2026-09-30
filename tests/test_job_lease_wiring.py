from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from syntrive.services.job_lease import (
    HolderIdentity,
    JobLeaseConflict,
    acquire_lease,
    get_lease,
    hold_job_lease,
    hold_job_leases,
    list_leases,
)

FOREIGN = HolderIdentity(holder_id="f" * 32, holder_kind="tts_batch", pid=4242, hostname="other-host")


def _plant_foreign_lease(db_path: Path, job_id: int) -> None:
    assert acquire_lease(db_path, job_id, holder=FOREIGN, operation="batch:99").ok


def _seed_book(db_path: Path, job_ids: tuple[int, ...] = (1,), language: str = "en") -> None:
    from syntrive.db.models import Book, Job
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        book = Book(title="Lease Book", language=language)
        db.add(book)
        db.flush()
        for job_id in job_ids:
            process_dir = f"job{job_id}"
            (db_path.parent / process_dir).mkdir(parents=True, exist_ok=True)
            db.add(Job(id=job_id, book_id=book.id, process_dir=process_dir, epub_path=f"{process_dir}/b.epub"))


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "syntrivetts.db"


@pytest.fixture()
def refreshes(monkeypatch) -> list[tuple[str, object]]:
    import syntrive.services.contract_export_service as ces

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        ces, "refresh_manifests_best_effort",
        lambda db_path, *, reason, job_id=None: calls.append((reason, job_id)),
    )
    return calls


class TestReentrancy:
    def test_nested_hold_in_same_thread_reuses_the_outer_lease(self, db_path: Path):
        _seed_book(db_path)
        with hold_job_lease(db_path, 1, holder_kind="tui", operation="outer") as outer:
            with hold_job_lease(db_path, 1, holder_kind="tui", operation="inner") as inner:
                assert inner is outer
            assert get_lease(db_path, 1).holder_id == outer.holder.holder_id
        assert get_lease(db_path, 1) is None

    def test_another_thread_of_the_same_process_still_conflicts(self, db_path: Path):
        _seed_book(db_path)
        errors: list[BaseException] = []

        def other_thread() -> None:
            try:
                with hold_job_lease(db_path, 1, holder_kind="tui"):
                    pass
            except BaseException as exc:  # noqa: BLE001 -- captured for the assertion
                errors.append(exc)

        with hold_job_lease(db_path, 1, holder_kind="tui", operation="step:x"):
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join()
        assert len(errors) == 1 and isinstance(errors[0], JobLeaseConflict)


class TestHoldJobLeases:
    def test_all_or_nothing_releases_what_it_took_on_conflict(self, db_path: Path):
        _seed_book(db_path, job_ids=(1, 2))
        _plant_foreign_lease(db_path, 2)

        with pytest.raises(JobLeaseConflict) as info:
            with hold_job_leases(db_path, [2, 1], holder_kind="tui"):
                pytest.fail("body must not run")

        assert info.value.job_id == 2
        assert get_lease(db_path, 1) is None
        assert get_lease(db_path, 2).holder_id == FOREIGN.holder_id

    def test_holds_every_job_and_releases_all(self, db_path: Path):
        _seed_book(db_path, job_ids=(1, 2))
        with hold_job_leases(db_path, [2, 1, 2], holder_kind="webui", operation="op") as handles:
            assert [h.job_id for h in handles] == [1, 2]
            assert {lease.holder_kind for lease in list_leases(db_path)} == {"webui"}
        assert list_leases(db_path) == []


class TestWorkflowEngineStep:
    def _engine(self, db_path: Path, monkeypatch, calls: list):
        from syntrive.workflow.engine import StepOutcome, WorkflowEngine

        engine = WorkflowEngine(SimpleNamespace(id=1), db_path, holder_kind="tui")

        def fake_dispatch(step):
            calls.append(get_lease(db_path, 1))
            return StepOutcome(success=True)

        monkeypatch.setattr(engine, "_dispatch_handler", fake_dispatch)
        return engine

    def test_step_runs_under_the_lease_and_releases_it(self, db_path: Path, monkeypatch):
        from syntrive.workflow.engine import WorkflowStep

        _seed_book(db_path)
        calls: list = []
        outcome = self._engine(db_path, monkeypatch, calls).execute_step(
            WorkflowStep.TRANSCRIPT_TEXT, force_overwrite=True,
        )

        assert outcome.success is True
        assert calls[0].holder_kind == "tui" and calls[0].operation == "step:transcript_text"
        assert get_lease(db_path, 1) is None

    def test_conflict_is_a_failed_outcome_and_the_handler_never_runs(self, db_path: Path, monkeypatch):
        from syntrive.workflow.engine import WorkflowStep

        _seed_book(db_path)
        _plant_foreign_lease(db_path, 1)
        calls: list = []
        outcome = self._engine(db_path, monkeypatch, calls).execute_step(
            WorkflowStep.TRANSCRIPT_TEXT, force_overwrite=True,
        )

        assert outcome.success is False
        assert "tts_batch" in outcome.error and "other-host" in outcome.error
        assert calls == []


def _seed_pending_chapter(db_path: Path) -> int:
    from syntrive.db.models import TranscriptChapter, TtsConfig
    from syntrive.db.session import get_db_session

    _seed_book(db_path)
    script = db_path.parent / "job1" / "transcript_text" / "tts_script" / "ch_0001.txt"
    script.parent.mkdir(parents=True)
    script.write_text("only line\n", encoding="utf-8")
    with get_db_session(db_path) as db:
        db.add(TtsConfig(job_id=1, engine="xtts"))
        chapter = TranscriptChapter(
            job_id=1, chapter_id="ch_0001", sequence_number="0001", chapter_number="0001",
            transcript_path="transcript_text/tts_script/ch_0001.txt", transcript_lines=1,
            synthesis_status="pending",
        )
        db.add(chapter)
        db.flush()
        return chapter.id


def _chapter_state(db_path: Path) -> tuple:
    from syntrive.db.models import SynthesisBatch, TranscriptChapter
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        chapter = db.query(TranscriptChapter).one()
        batch = db.query(SynthesisBatch).filter_by(synthesis_batch_id=chapter.synthesis_batch_id).first()
        return chapter.synthesis_status, batch.synth_error if batch else None


class TestEnqueueGuard:
    def test_refused_while_a_step_writer_holds_the_job(self, db_path: Path):
        from syntrive.services import synthesis_service

        chapter_db_id = _seed_pending_chapter(db_path)
        step_holder = HolderIdentity(holder_id="s" * 32, holder_kind="tui", pid=4242, hostname="other-host")
        assert acquire_lease(db_path, 1, holder=step_holder, operation="step:transcript_text").ok

        result = synthesis_service.enqueue(db_path, [chapter_db_id])

        assert result.ok is False and result.batch_ids == ()
        assert "tui (step:transcript_text)" in result.error
        assert _chapter_state(db_path) == ("pending", None)

    def test_allowed_while_tts_batch_synthesizes_the_same_book(self, db_path: Path):
        from syntrive.services import synthesis_service

        chapter_db_id = _seed_pending_chapter(db_path)
        _plant_foreign_lease(db_path, 1)

        result = synthesis_service.enqueue(db_path, [chapter_db_id])

        assert result.ok is True and len(result.batch_ids) == 1
        assert _chapter_state(db_path)[0] == "queued"

    def test_expired_holder_does_not_block(self, db_path: Path):
        import time

        from syntrive.services import synthesis_service

        chapter_db_id = _seed_pending_chapter(db_path)
        step_holder = HolderIdentity(holder_id="s" * 32, holder_kind="webui", pid=4242, hostname="other-host")
        assert acquire_lease(db_path, 1, holder=step_holder, ttl=90, clock=lambda: time.time() - 1000).ok

        assert synthesis_service.enqueue(db_path, [chapter_db_id]).ok is True


class TestRunActiveBatchesLease:
    def _enqueue_one(self, db_path: Path) -> int:
        from syntrive.services import synthesis_service

        return synthesis_service.enqueue(db_path, [_seed_pending_chapter(db_path)]).batch_ids[0]

    def test_stop_before_chapter_leaves_the_batch_resumable(self, db_path: Path, monkeypatch):
        import syntrive.orchestration.workflow as workflow_module
        from syntrive.services import synthesis_service

        batch_id = self._enqueue_one(db_path)
        monkeypatch.setattr(workflow_module.manager, "get_engine", lambda *a, **k: pytest.fail("engine loaded"))

        result = synthesis_service.run_one_batch(db_path, batch_id, should_stop=lambda: True)

        assert result.ok is False and result.stopped is True and result.chapters_completed == 0
        assert _chapter_state(db_path) == ("queued", None)

    def test_lease_lost_mid_batch_is_reported_as_a_skip_not_a_failure(self, db_path: Path, monkeypatch):
        from syntrive.orchestration.workflow import BatchRunResult
        from syntrive.services import synthesis_service

        batch_id = self._enqueue_one(db_path)
        probes: list[bool] = []

        def fake_run_one_batch(db_path_arg, bid, *, should_stop, **_kwargs):
            probes.append(should_stop())
            return BatchRunResult(ok=False, error="stopped before sentence 3", failed_chapter_id="ch_0001", stopped=True)

        monkeypatch.setenv("SYNTRIVE_MAX_CONSECUTIVE_BATCH_FAILURES", "1")
        monkeypatch.setattr(synthesis_service, "run_one_batch", fake_run_one_batch)
        result = synthesis_service.run_active_batches(db_path)

        assert probes == [False]
        assert result.ok is True and result.failed_results == []
        assert [s.batch_id for s in result.skipped] == [batch_id]
        assert result.skipped[0].reason.startswith("job lease lost mid-batch")

    def test_leased_job_batch_is_skipped_left_queued_and_not_a_failure(self, db_path: Path, monkeypatch):
        from syntrive.db.models import TranscriptChapter
        from syntrive.db.session import get_db_session
        from syntrive.services import synthesis_service

        batch_id = self._enqueue_one(db_path)
        _plant_foreign_lease(db_path, 1)
        monkeypatch.setenv("SYNTRIVE_MAX_CONSECUTIVE_BATCH_FAILURES", "1")
        ran: list[int] = []
        monkeypatch.setattr(synthesis_service, "run_one_batch", lambda *a, **k: ran.append(a[1]))

        result = synthesis_service.run_active_batches(db_path)

        assert result.ok is True and result.batches_run == 0 and result.failed_results == []
        assert [s.batch_id for s in result.skipped] == [batch_id]
        assert "tts_batch" in result.skipped[0].reason
        assert ran == []
        with get_db_session(db_path) as db:
            assert db.query(TranscriptChapter).one().synthesis_status == "queued"

    def test_batch_runs_under_a_tts_batch_lease_once_the_job_is_free(self, db_path: Path, monkeypatch):
        from syntrive.orchestration.workflow import BatchRunResult
        from syntrive.services import synthesis_service

        batch_id = self._enqueue_one(db_path)
        seen: list = []

        def fake_run_one_batch(db_path_arg, bid, **_kwargs):
            seen.append((bid, get_lease(db_path_arg, 1)))
            return BatchRunResult(ok=True, chapters_completed=1)

        monkeypatch.setattr(synthesis_service, "run_one_batch", fake_run_one_batch)
        result = synthesis_service.run_active_batches(db_path)

        assert result.ok is True and result.batches_run == 1 and result.skipped == []
        assert seen[0][0] == batch_id
        assert seen[0][1].holder_kind == "tts_batch" and seen[0][1].operation == f"batch:{batch_id}"
        assert get_lease(db_path, 1) is None


def _book_language(db_path: Path) -> str:
    from syntrive.db.models import Book
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        return db.query(Book).one().language


class TestJobServiceLease:
    def test_language_change_is_blocked_when_another_job_of_the_book_is_leased(self, db_path, refreshes):
        from syntrive.services.job_service import JobService

        _seed_book(db_path, job_ids=(1, 2), language="en")
        _plant_foreign_lease(db_path, 2)

        with pytest.raises(JobLeaseConflict):
            JobService(db_path, holder_kind="tui").update_book_language(1, "zh")

        assert _book_language(db_path) == "en"
        assert refreshes == []
        assert get_lease(db_path, 1) is None

    def test_unchanged_language_needs_no_lease(self, db_path, refreshes):
        from syntrive.services.job_service import JobService

        _seed_book(db_path, language="en")
        _plant_foreign_lease(db_path, 1)

        JobService(db_path).update_book_language(1, "en")
        assert refreshes == []

    def test_language_change_on_free_book_writes_and_refreshes(self, db_path, refreshes):
        from syntrive.services.job_service import JobService

        _seed_book(db_path, job_ids=(1, 2), language="en")
        JobService(db_path).update_book_language(1, "zh")

        assert _book_language(db_path) == "zh"
        assert refreshes == [("book:language", None)]
        assert list_leases(db_path) == []

    def test_title_and_cover_are_lease_free_and_refresh_the_contract(self, db_path, refreshes):
        from syntrive.db.models import Book
        from syntrive.db.session import get_db_session
        from syntrive.services.job_service import JobService

        _seed_book(db_path)
        _plant_foreign_lease(db_path, 1)
        svc = JobService(db_path)
        svc.update_book_title(1, "New Title")
        svc.update_book_cover(1, "images/cover.jpg")

        with get_db_session(db_path) as db:
            book = db.query(Book).one()
            assert (book.title, book.cover) == ("New Title", "images/cover.jpg")
        assert refreshes == [("book:title", None), ("book:cover", None)]

    def test_delete_is_blocked_and_deletes_nothing_while_leased(self, db_path, refreshes):
        from syntrive.db.models import Job
        from syntrive.db.session import get_db_session
        from syntrive.services.job_service import JobService

        _seed_book(db_path, job_ids=(1, 2))
        _plant_foreign_lease(db_path, 2)

        with pytest.raises(JobLeaseConflict):
            JobService(db_path).delete_book_job(1)

        with get_db_session(db_path) as db:
            assert db.query(Job).count() == 2
        assert (db_path.parent / "job1").is_dir() and (db_path.parent / "job2").is_dir()
        assert refreshes == []

    def test_delete_on_free_book_removes_and_refreshes_repo_json(self, db_path, refreshes):
        from syntrive.services.job_service import JobService

        _seed_book(db_path, job_ids=(1, 2))
        result = JobService(db_path).delete_book_job(1)

        assert result["jobs_deleted"] == 2
        assert not (db_path.parent / "job1").exists()
        assert refreshes == [("book:deleted", None)]
        assert list_leases(db_path) == []

    def test_reset_refreshes_that_job(self, db_path, refreshes):
        from syntrive.services.job_service import JobService
        from syntrive.workflow.engine import WorkflowStep

        _seed_book(db_path)
        JobService(db_path).reset_job_to_step(1, WorkflowStep.TRANSCRIPT_CLEAN)
        assert refreshes == [("job:reset:transcript_clean", 1)]

