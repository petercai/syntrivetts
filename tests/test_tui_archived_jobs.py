from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from syntrive.tui.job_list_view import (
    archived_notice,
    build_job_list_view,
    read_only_warning,
    restored_notice,
)


def _job(job_id: int, status: str = "running", archived: bool = False, title: str = "Book"):
    return SimpleNamespace(
        id=job_id, status=status, archived_at=datetime(2026, 9, 26) if archived else None,
        book=SimpleNamespace(title=title),
    )


class TestJobListView:
    def test_archived_hidden_by_default_with_a_hint_naming_the_key(self):
        jobs = [_job(3), _job(2, archived=True), _job(1, status="completed")]
        view = build_job_list_view(jobs, show_archived=False)

        assert [j.id for j in view.visible] == [3, 1]
        assert view.hint == "1 archived hidden (a to show)"
        assert view.counter == "1 incomplete / 2 active  ↺ live"

    def test_shown_keeps_order_and_the_hint_says_how_to_hide(self):
        jobs = [_job(3), _job(2, archived=True), _job(1)]
        view = build_job_list_view(jobs, show_archived=True)

        assert [j.id for j in view.visible] == [3, 2, 1]
        assert view.hint == "Showing 1 archived (a to hide)"
        assert view.counter == "2 incomplete / 2 active  ↺ live"

    def test_no_archived_jobs_means_no_hint(self):
        view = build_job_list_view([_job(1)], show_archived=False)
        assert view.hint == "" and view.archived_count == 0

    def test_toast_copy_names_the_book_and_the_next_key(self):
        job = _job(4, title="Elon Musk")
        assert read_only_warning(job) == "“Elon Musk” is archived and read-only. Press x to restore it, then continue."
        assert "press a to show archived jobs" in archived_notice(job, show_archived=False)
        assert restored_notice(job).startswith("Restored “Elon Musk”")


def _seed_repo(repo: Path) -> None:
    from syntrive.db.models import Book, Job
    from syntrive.db.session import get_db_session

    with get_db_session(repo / "syntrivetts.db") as db:
        for job_id, title, archived in ((1, "1984", False), (2, "Sapiens", True)):
            book = Book(title=title, language="en")
            db.add(book)
            db.flush()
            (repo / f"job{job_id}").mkdir()
            db.add(Job(
                id=job_id, book_id=book.id, process_dir=f"job{job_id}", epub_path=f"job{job_id}/b.epub",
                status="running", stage="transcript_review", current_step="transcript_review",
                archived_at=datetime(2026, 9, 26) if archived else None,
            ))


def _archived_at(repo: Path, job_id: int):
    from syntrive.db.models import Job
    from syntrive.db.session import get_db_session

    with get_db_session(repo / "syntrivetts.db") as db:
        return db.get(Job, job_id).archived_at


def _run(repo: Path, scenario) -> None:
    from syntrive.tui.app import SyntriveApp

    async def main() -> None:
        app = SyntriveApp(repo_dir=repo)
        async with app.run_test(size=(140, 40)) as pilot:
            await _settle(pilot)
            await scenario(app, pilot)

    asyncio.run(main())


async def _settle(pilot, rounds: int = 6) -> None:
    for _ in range(rounds):
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()


def _rows(app) -> list[str]:
    from textual.widgets import DataTable

    return [key.value for key in app.query_one("#job-table", DataTable).rows]


def _footer_label(app, key: str) -> str:
    binding = app.screen.active_bindings.get(key)
    return binding.binding.description if binding is not None else ""


class TestTuiArchivedJobs:
    def test_toggle_shows_hidden_archived_rows_and_the_footer_follows(self, tmp_path: Path):
        _seed_repo(tmp_path)

        async def scenario(app, pilot):
            panel = app.query_one("#job-panel")
            assert _rows(app) == ["1"]
            assert panel.border_subtitle == "1 archived hidden (a to show)"
            assert _footer_label(app, "a") == "Show archived"

            await pilot.press("a")
            await _settle(pilot)
            assert sorted(_rows(app)) == ["1", "2"]
            assert panel.border_subtitle == "Showing 1 archived (a to hide)"
            assert _footer_label(app, "a") == "Hide archived"

            await pilot.press("a")
            await _settle(pilot)
            assert _rows(app) == ["1"]

        _run(tmp_path, scenario)

    def test_x_archives_the_selected_job_and_restores_it(self, tmp_path: Path):
        _seed_repo(tmp_path)

        async def scenario(app, pilot):
            app._load_and_show_job(1)
            await _settle(pilot)
            assert _footer_label(app, "x") == "Archive"

            await pilot.press("x")
            await _settle(pilot)
            assert _archived_at(tmp_path, 1) is not None
            assert _rows(app) == []
            assert app._selected_job is None

            await pilot.press("a")
            await _settle(pilot)
            app._load_and_show_job(1)
            await _settle(pilot)
            assert _footer_label(app, "x") == "Restore"

            await pilot.press("x")
            await _settle(pilot)
            assert _archived_at(tmp_path, 1) is None
            assert _footer_label(app, "x") == "Archive"

        _run(tmp_path, scenario)

    def test_continue_on_an_archived_job_warns_and_opens_nothing(self, tmp_path: Path):
        _seed_repo(tmp_path)

        async def scenario(app, pilot):
            await pilot.press("a")
            await _settle(pilot)
            app._load_and_show_job(2)
            await _settle(pilot)

            screens_before = len(app.screen_stack)
            app.action_continue_job()
            await _settle(pilot)

            assert len(app.screen_stack) == screens_before
            messages = [n.message for n in app._notifications]
            assert any("“Sapiens” is archived and read-only" in m for m in messages)

        _run(tmp_path, scenario)
