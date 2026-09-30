from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar, Optional

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Footer, Header, Label, Rule, Static

from syntrive.tui.job_list_view import (
    JobListView,
    archived_notice,
    build_job_list_view,
    is_archived,
    read_only_warning,
    restored_notice,
)
from syntrive.tui.widgets import LogPanel, LogViewerModal

logger = logging.getLogger(__name__)


class _TuiLogHandler(logging.Handler):
    _LEVEL_COLOR: ClassVar[dict[int, str]] = {
        logging.DEBUG: "dim",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold red underline",
    }

    def __init__(self, app: "SyntriveApp") -> None:
        super().__init__(level=logging.INFO)
        self._app = app
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            color = self._LEVEL_COLOR.get(record.levelno, "")
            line = f"[{color}]{msg}[/{color}]" if color else msg
            self._app.call_from_thread(self._app._push_log_line, line)
        except Exception:
            pass


class JobSelected(Message):
    def __init__(self, job_id: int) -> None:
        super().__init__()
        self.job_id = job_id


class NewJobRequested(Message):
    def __init__(self, epub_path: Path) -> None:
        super().__init__()
        self.epub_path = epub_path


_ARCHIVE_ACTIONS = frozenset({"archive_job", "restore_job", "show_archived", "hide_archived"})


def _dimmed(cell) -> Text:
    text = cell.copy() if isinstance(cell, Text) else Text(str(cell))
    text.stylize("dim")
    return text



class JobListPanel(Vertical):
    BORDER_TITLE = "Jobs"

    DEFAULT_CSS = """
    JobListPanel {
        width: 40%;
        border: solid $primary;
        padding: 0 1;
    }
    JobListPanel #job-status {
        color: $text-muted;
        padding: 0 1;
    }
    JobListPanel {
        border-subtitle-color: $text-warning;  /* contrast-adjusted per theme */
    }
    """

    _jobs: reactive[list] = reactive([], always_update=True)
    show_archived: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        table = DataTable(id="job-table", cursor_type="row", show_cursor=True)
        table.add_columns("ID", "Book", "Lang", "Stage", "Status", "Cover")
        yield table
        yield Label("Loading…", id="job-status")

    @property
    def view(self) -> JobListView:
        return build_job_list_view(self._jobs, show_archived=self.show_archived)

    def watch__jobs(self, jobs: list) -> None:
        self._render_rows()

    def watch_show_archived(self, show_archived: bool) -> None:
        logger.info("job_list_archived_toggle: show_archived=%s", show_archived)
        self._render_rows()

    def _render_rows(self) -> None:
        view = self.view
        table = self.query_one("#job-table", DataTable)

        highlighted_key: str | None = None
        if table.row_count > 0:
            try:
                cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
                highlighted_key = cell_key.row_key.value
            except Exception:
                pass

        table.clear()

        _STATUS_COLOR = {
            "pending": "yellow",
            "running": "cyan",
            "completed": "green",
            "done": "green",
            "failed": "red",
            "blocked": "red",
            "paused": "yellow",
        }

        for job in view.visible:
            archived = is_archived(job)
            color = _STATUS_COLOR.get(job.status, "white")
            book = job.book if hasattr(job, "book") else None
            if book and book.title:
                title = Text(book.title)
                title.truncate(20, overflow="ellipsis")
            else:
                title = Text("—", style="dim")
            lang = book.language or "?" if book else "?"
            cover_rel = book.cover if book else None
            cover = Path(cover_rel).name if cover_rel else Text("—", style="dim")
            status = Text("archived", style="dim italic") if archived else Text(job.status, style=color)
            cells = [str(job.id), title, lang, job.stage or "—", status, cover]
            table.add_row(
                *(_dimmed(cell) if archived else cell for cell in cells),
                key=str(job.id),
            )

        if highlighted_key is not None:
            try:
                row_index = next(
                    (
                        i
                        for i, rk in enumerate(table.rows)
                        if rk.value == highlighted_key
                    ),
                    None,
                )
                if row_index is not None:
                    table.move_cursor(row=row_index)
            except Exception:
                pass

        self.query_one("#job-status", Label).update(view.counter)
        self.border_subtitle = view.hint

    @on(DataTable.RowSelected, "#job-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.post_message(JobSelected(int(event.row_key.value)))


class WorkflowPanel(Vertical):
    BORDER_TITLE = "Workflow"

    DEFAULT_CSS = """
    WorkflowPanel {
        width: 60%;
        border: solid $secondary;
        padding: 1 2;
    }
    WorkflowPanel #wf-title { color: $accent; text-style: bold; }
    WorkflowPanel #wf-meta  { color: $text-muted; }
    WorkflowPanel #wf-tasks { padding: 1 0; }
    """

    _job_id: reactive[Optional[int]] = reactive(None)
    _job_data: reactive[Optional[object]] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Static("Select a job from the list →", id="wf-title")
        yield Static("", id="wf-meta")
        yield Rule()
        yield Static("", id="wf-tasks")

    def update_job(self, job) -> None:
        self._job_id = job.id
        self._job_data = job

        title = self.query_one("#wf-title", Static)
        meta = self.query_one("#wf-meta", Static)
        tasks = self.query_one("#wf-tasks", Static)

        book_name = job.book.title if hasattr(job, "book") and job.book else "Unknown"
        title.update(f"Job #{job.id} — {book_name}")
        meta.update(
            f"Stage: {job.stage or 'init'}   Status: {'archived' if is_archived(job) else job.status}   "
            f"Step: {job.current_step or '—'}"
        )

        from syntrive.workflow.engine import _STEP_LABELS, WorkflowStep

        step_name = job.current_step or WorkflowStep.BOOTSTRAP.value
        try:
            step = WorkflowStep(step_name)
            label = _STEP_LABELS.get(step, step_name)
        except ValueError:
            label = step_name
        hint = (
            "Archived and read-only. Press x to restore it."
            if is_archived(job)
            else "Press Enter to continue, n for new book"
        )
        tasks.update(f"Current step: [bold]{label}[/bold]\n\n[dim]{hint}[/dim]")

    def clear(self) -> None:
        self._job_id = None
        self._job_data = None
        self.query_one("#wf-title", Static).update("Select a job from the list →")
        self.query_one("#wf-meta", Static).update("")
        self.query_one("#wf-tasks", Static).update("")


class SyntriveApp(App):
    TITLE = "SyntriveTTS"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("t", "toggle_dark", "Theme", show=True),
        Binding("enter", "continue_job", "Continue", show=True),
        Binding("s", "start_over", "Start Over", show=True),
        Binding("d", "delete_book_job", "Delete", show=True),
        Binding("x", "archive_job", "Archive", show=True),
        Binding("x", "restore_job", "Restore", show=True),
        Binding("a", "show_archived", "Show archived", show=True),
        Binding("a", "hide_archived", "Hide archived", show=True),
        Binding("n", "new_book", "New Book", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("v", "view_files", "View Files", show=True),
        Binding("m", "edit_metadata", "Metadata", show=True),
        Binding("l", "edit_language", "Set Lang", show=True),
        Binding("e", "edit_tts", "Edit TTS", show=True),
        Binding("c", "open_vscode", "VS Code", show=True),
        Binding("ctrl+l", "view_log", "View Log", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    DEFAULT_CSS = """
    Screen {
        layout: vertical;
    }
    #main-body {
        height: 1fr;
    }
    #theme-btn {
        dock: right;
        min-width: 12;
        margin: 0 1;
    }
    """

    _selected_job: reactive[Optional[object]] = reactive(None)

    def __init__(
        self,
        repo_dir: Path,
        initial_job_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._repo_dir = repo_dir
        self._initial_job_id = initial_job_id
        self._db_path = repo_dir / "syntrivetts.db"
        self._log_handler: Optional[_TuiLogHandler] = None
        self._log_buffer: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-body"):
            yield JobListPanel(id="job-panel")
            yield WorkflowPanel(id="workflow-panel")
        yield LogPanel(id="log-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self._repo_dir)
        header = self.query_one(Header)
        header.mount(Button("", id="theme-btn", variant="primary"))
        self.query_one("#theme-btn", Button).label = (
            "☀️ Light" if self.current_theme.dark else "🌙 Dark"
        )

        self.set_interval(3.0, self._refresh_jobs_background)
        self._refresh_jobs_background()

        if self._initial_job_id is not None:
            self.call_after_refresh(self._select_initial_job)

        self._log_handler = _TuiLogHandler(self)
        logging.getLogger().addHandler(self._log_handler)

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)

    @work(thread=True)
    def _refresh_jobs_background(self) -> None:
        try:
            from sqlalchemy.orm import joinedload
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job

            with get_db_session(self._db_path) as db:
                jobs = (
                    db.query(Job)
                    .options(joinedload(Job.book))
                    .order_by(Job.created_at.desc())
                    .all()
                )
                db.expunge_all()

            self.call_from_thread(self._apply_jobs_refresh, jobs)
        except Exception as exc:
            logger.warning("Job refresh failed: %s", exc)

    def _apply_jobs_refresh(self, jobs: list) -> None:
        panel = self.query_one("#job-panel", JobListPanel)
        panel._jobs = jobs
        selected = self._selected_job
        fresh = next((j for j in jobs if selected is not None and j.id == selected.id), None)
        if fresh is not None and is_archived(fresh) != is_archived(selected):
            logger.info("job_archive_state_synced: job_id=%d archived=%s", fresh.id, is_archived(fresh))
            if is_archived(fresh) and not panel.show_archived:
                self._clear_job_selection()
            else:
                self._apply_job_selection(fresh)
        self.refresh_bindings()

    def _select_initial_job(self) -> None:
        self._load_and_show_job(self._initial_job_id)

    @on(JobSelected)
    def _on_job_selected(self, msg: JobSelected) -> None:
        self._load_and_show_job(msg.job_id)

    @work(thread=True)
    def _load_and_show_job(self, job_id: int) -> None:
        try:
            from sqlalchemy.orm import joinedload
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job

            with get_db_session(self._db_path) as db:
                job = (
                    db.query(Job)
                    .options(joinedload(Job.book))
                    .filter(Job.id == job_id)
                    .first()
                )
                if job:
                    db.expunge_all()
                    self.call_from_thread(self._apply_job_selection, job)
        except Exception as exc:
            logger.error("Failed to load job %d: %s", job_id, exc)

    def _apply_job_selection(self, job) -> None:
        self._selected_job = job
        self.query_one("#workflow-panel", WorkflowPanel).update_job(job)
        self.refresh_bindings()

    def _selected_is_archived(self) -> bool:
        return self._selected_job is not None and is_archived(self._selected_job)

    def _guard_archived(self) -> bool:
        if not self._selected_is_archived():
            return False
        logger.info("job_action_blocked_archived: job_id=%d", self._selected_job.id)
        self.notify(read_only_warning(self._selected_job), title="Job is archived", severity="warning")
        return True

    def _set_show_archived(self, show: bool) -> None:
        panel = self.query_one("#job-panel", JobListPanel)
        panel.show_archived = show
        if not show and self._selected_is_archived():
            self._clear_job_selection()
        self.refresh_bindings()

    def action_show_archived(self) -> None:
        self._set_show_archived(True)

    def action_hide_archived(self) -> None:
        self._set_show_archived(False)

    def action_archive_job(self) -> None:
        if self._selected_job is not None:
            self._toggle_archive_background(self._selected_job.id, archive=True)

    def action_restore_job(self) -> None:
        if self._selected_job is not None:
            self._toggle_archive_background(self._selected_job.id, archive=False)

    @work(thread=True)
    def _toggle_archive_background(self, job_id: int, *, archive: bool) -> None:
        try:
            from syntrive.services.job_service import JobService

            svc = JobService(self._db_path, holder_kind="tui")
            result = svc.archive_job(job_id) if archive else svc.unarchive_job(job_id)
            if not result.ok:
                self.call_from_thread(self.notify, f"Archive failed: {result.error}", severity="error")
                return
            self.call_from_thread(self._apply_archive_result, job_id, archive)
        except Exception as exc:
            logger.error("Archive toggle failed for job %d: %s", job_id, exc, exc_info=True)
            self.call_from_thread(self.notify, f"Archive failed: {exc}", severity="error")

    def _apply_archive_result(self, job_id: int, archived: bool) -> None:
        job = self._selected_job if self._selected_job is not None and self._selected_job.id == job_id else None
        show = self.query_one("#job-panel", JobListPanel).show_archived
        if job is not None:
            message = archived_notice(job, show_archived=show) if archived else restored_notice(job)
            self.notify(message, title="Archived" if archived else "Restored")
        if archived and not show:
            self._clear_job_selection()
        else:
            self._load_and_show_job(job_id)
        self._refresh_jobs_background()

    def _push_log_line(self, line: str) -> None:
        try:
            plain = Text.from_markup(line).plain
        except Exception:
            plain = line
        self._log_buffer.append(plain)
        if len(self._log_buffer) > 200:
            self._log_buffer = self._log_buffer[-200:]

        try:
            for panel in self.screen.query(LogPanel):
                panel.write(line)
        except Exception:
            pass

    def action_toggle_dark(self) -> None:
        self.theme = "textual-light" if self.current_theme.dark else "textual-dark"
        btn = self.query_one("#theme-btn", Button)
        btn.label = "☀️ Light" if self.current_theme.dark else "🌙 Dark"

    @on(Button.Pressed, "#theme-btn")
    def _on_theme_btn(self) -> None:
        self.action_toggle_dark()

    async def action_new_book(self) -> None:
        from syntrive.tui.dialogs import FilePathInputScreen, is_gui_available

        if is_gui_available():
            self._open_file_picker()
        else:
            logger.debug("action_new_book: no GUI display — using text-input fallback")
            path_str = await self.push_screen_wait(
                FilePathInputScreen(
                    title="Enter EPUB File Path",
                    hint=(
                        "No GUI display detected (SSH/headless).\n"
                        "Paste the full server-side path to the EPUB file."
                    ),
                    expected_suffix=".epub",
                )
            )
            if path_str:
                logger.info("action_new_book: text-input path=%s", path_str)
                self._handle_file_selected(Path(path_str))
            else:
                logger.debug("action_new_book: text-input cancelled")

    @work(thread=True)
    def _open_file_picker(self) -> None:
        from syntrive.tui.dialogs import native_file_picker

        logger.debug("_open_file_picker: opening EPUB file dialog")
        path = native_file_picker(
            title="Select EPUB File",
            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")],
        )
        if path:
            logger.info("_open_file_picker: selected path=%s", path)
            self.call_from_thread(self._handle_file_selected, Path(path))
        else:
            logger.debug("_open_file_picker: no file selected (cancelled)")

    def _handle_file_selected(self, epub_path: Path) -> None:
        self.notify(f"Starting job for: {epub_path.name}")
        self._bootstrap_new_job(epub_path)

    @work(thread=True)
    def _bootstrap_new_job(self, epub_path: Path) -> None:
        try:
            from syntrive.bootstrap import bootstrap
            from syntrive.adapters.epub.lang import normalize_language
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Book

            job = bootstrap(repo_dir=self._repo_dir, ebook_path=epub_path)
            logger.info("New job created: id=%d", job.id)
            self.call_from_thread(
                self.notify, f"Job {job.id} created: {epub_path.name}"
            )
            self.call_from_thread(self._load_and_show_job, job.id)
            self._refresh_jobs_background()

            with get_db_session(self._db_path) as db:
                book = db.query(Book).filter_by(id=job.book_id).first()
                if book and normalize_language(book.language) is None:
                    self.call_from_thread(
                        self.notify,
                        f"Job #{job.id}: language unrecognised"
                        f" ('{book.language or 'None'}') — press L to set it",
                        severity="warning",
                    )
        except Exception as exc:
            logger.error("Bootstrap failed: %s", exc, exc_info=True)
            self.call_from_thread(self.notify, f"Failed: {exc}", severity="error")

    def action_continue_job(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return
        if self._guard_archived():
            return
        from syntrive.tui.dialogs import WorkflowRunScreen

        self.push_screen(WorkflowRunScreen(job=job, db_path=self._db_path))

    def action_start_over(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return
        if self._guard_archived():
            return

        job_id = job.id

        from syntrive.tui.dialogs import ChoiceScreen
        from syntrive.workflow.engine import _ORDERED_STEPS, _STEP_LABELS, WorkflowStep

        actionable = [s for s in _ORDERED_STEPS if s != WorkflowStep.DONE]
        choices = [f"{i + 1}. {_STEP_LABELS[s]}" for i, s in enumerate(actionable)]

        def _on_choice(choice: Optional[str]) -> None:
            if choice is None:
                return
            step = next(
                (
                    s
                    for i, s in enumerate(actionable)
                    if f"{i + 1}. {_STEP_LABELS[s]}" == choice
                ),
                None,
            )
            if step is None:
                logger.warning("Unknown step choice: %s", choice)
                return
            logger.info("User resetting job %d to step=%s", job_id, step.value)
            self._reset_job_to_step_background(job_id, step)

        self.push_screen(
            ChoiceScreen(
                title=f"Reset Job #{job_id} — choose start step",
                choices=choices,
            ),
            _on_choice,
        )

    @work(thread=True)
    def _reset_job_to_step_background(self, job_id: int, step) -> None:
        try:
            from syntrive.services.job_service import JobService

            JobService(self._db_path).reset_job_to_step(job_id, step)
            self.call_from_thread(
                self.notify,
                f"Job #{job_id} reset to: {step.value}. Press Enter to continue.",
            )
            self.call_from_thread(self._load_and_show_job, job_id)
            self._refresh_jobs_background()
        except Exception as exc:
            logger.error("Reset failed for job %d: %s", job_id, exc, exc_info=True)
            self.call_from_thread(self.notify, f"Reset failed: {exc}", severity="error")

    def action_view_files(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return
        from syntrive.tui.viewers import DirectoryViewScreen
        from syntrive.db.path_utils import resolve_abs

        self.push_screen(
            DirectoryViewScreen(process_dir=resolve_abs(self._db_path, job.process_dir))
        )

    def action_open_vscode(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return
        from syntrive.db.path_utils import resolve_abs

        abs_process_dir = resolve_abs(self._db_path, job.process_dir)
        _open_vscode(str(abs_process_dir))
        self.notify(f"Opening VS Code: {abs_process_dir}")

    def action_edit_metadata(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return
        from syntrive.tui.meta_editor import MetadataEditScreen
        from syntrive.db.path_utils import resolve_abs

        self.push_screen(
            MetadataEditScreen(
                job_id=job.id,
                process_dir=resolve_abs(self._db_path, job.process_dir),
                db_path=self._db_path,
            ),
            lambda _: self._load_and_show_job(job.id),
        )

    def action_edit_language(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return

        from syntrive.adapters.epub.lang import supported_languages
        from syntrive.tui.dialogs import ChoiceScreen

        job_id = job.id
        book = job.book if hasattr(job, "book") else None
        current = (book.language or "?") if book else "?"
        langs = supported_languages()

        def _on_choice(choice: Optional[str]) -> None:
            if choice is None:
                return
            self._update_language_background(job_id, choice)

        self.push_screen(
            ChoiceScreen(
                title=f"Set language for Job #{job_id}  (current: {current})",
                choices=langs,
            ),
            _on_choice,
        )

    @work(thread=True)
    def _update_language_background(self, job_id: int, language: str) -> None:
        try:
            from syntrive.services.job_service import JobService

            JobService(self._db_path, holder_kind="tui").update_book_language(job_id, language)
            self.call_from_thread(
                self.notify,
                f"Job #{job_id} language set to: {language}",
            )
            self.call_from_thread(self._load_and_show_job, job_id)
            self._refresh_jobs_background()
        except Exception as exc:
            logger.error(
                "Language update failed for job %d: %s", job_id, exc, exc_info=True
            )
            self.call_from_thread(
                self.notify, f"Language update failed: {exc}", severity="error"
            )

    def action_edit_tts(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return
        if self._guard_archived():
            return
        from syntrive.workflow.engine import WorkflowEngine

        WorkflowEngine(job, self._db_path).ensure_tts_config()
        WorkflowEngine(job, self._db_path).ensure_tts_voices()

        from syntrive.tui.tts_editor_gui import run_tts_editor_gui

        with self.suspend():
            saved = run_tts_editor_gui(job_id=job.id, db_path=self._db_path)
        if saved:
            self.notify("TTS configuration saved.")

    def action_refresh(self) -> None:
        self._refresh_jobs_background()
        self.notify("Refreshing jobs…")

    def action_delete_book_job(self) -> None:
        job = self._selected_job
        if job is None:
            self.notify("No job selected.", severity="warning")
            return

        from syntrive.tui.dialogs import ConfirmScreen

        book = getattr(job, "book", None)
        book_title = book.title if book and book.title else f"Job #{job.id}"
        job_id = job.id

        def _on_second_confirm(confirmed: bool) -> None:
            if confirmed:
                logger.info(
                    "User confirmed delete for job %d book=%r", job_id, book_title
                )
                self._delete_book_job_background(job_id)

        def _on_first_confirm(confirmed: bool) -> None:
            if not confirmed:
                return
            self.push_screen(
                ConfirmScreen(
                    title="Final Confirmation — Irreversible",
                    message=(
                        f"PERMANENTLY delete '{book_title}'?\n\n"
                        "This will remove ALL files and ALL database records.\n"
                        "There is NO undo.\n\n"
                        "Press Y to delete, N to cancel."
                    ),
                ),
                _on_second_confirm,
            )

        self.push_screen(
            ConfirmScreen(
                title=f"Delete: {book_title}",
                message=(
                    f"Delete book '{book_title}'?\n\n"
                    "This will permanently delete:\n"
                    "  • The PROCESSING folder and all files on disk\n"
                    "  • Book record and ALL database records\n"
                    "  • All jobs, chapters, TTS config, and audit logs\n\n"
                    "Press Y to proceed to final confirmation."
                ),
            ),
            _on_first_confirm,
        )

    @work(thread=True)
    def _delete_book_job_background(self, job_id: int) -> None:
        try:
            from syntrive.services.job_service import JobService

            result = JobService(self._db_path, holder_kind="tui").delete_book_job(job_id)
            book_title = result.get("book_title") or f"Job #{job_id}"
            dirs_deleted = result.get("dirs_deleted", 0)
            dirs_missing = result.get("dirs_missing", 0)
            jobs_deleted = result.get("jobs_deleted", 0)

            msg = (
                f"Deleted '{book_title}': "
                f"{jobs_deleted} job(s), {dirs_deleted} folder(s) removed."
            )
            if dirs_missing:
                msg += f" ({dirs_missing} folder(s) already missing on disk.)"

            logger.info("Delete complete: %s", msg)
            self.call_from_thread(self.notify, msg)
            self.call_from_thread(self._clear_job_selection)
            self._refresh_jobs_background()

        except Exception as exc:
            logger.error("Delete failed for job %d: %s", job_id, exc, exc_info=True)
            self.call_from_thread(
                self.notify, f"Delete failed: {exc}", severity="error"
            )

    def _clear_job_selection(self) -> None:
        self._selected_job = None
        self.query_one("#workflow-panel", WorkflowPanel).clear()
        self.refresh_bindings()

    def action_view_log(self) -> None:
        if not self._log_buffer:
            self.notify("Log is empty.", severity="warning")
            return
        self.push_screen(LogViewerModal(list(self._log_buffer)))

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        job_required = {
            "continue_job",
            "start_over",
            "delete_book_job",
            "view_files",
            "edit_tts",
            "open_vscode",
            "edit_language",
            "edit_metadata",
        }
        if action in job_required:
            return True if self._selected_job is not None else None
        if action in _ARCHIVE_ACTIONS:
            if self.screen is not self.screen_stack[0]:
                return False
            return self._archive_action_enabled(action)
        return True

    def _archive_action_enabled(self, action: str) -> bool:
        if action in ("archive_job", "restore_job"):
            return self._selected_job is not None and self._selected_is_archived() == (action == "restore_job")
        view = self.query_one("#job-panel", JobListPanel).view
        if action == "hide_archived":
            return view.show_archived
        return not view.show_archived and view.archived_count > 0


VSCODE_PREFERRED_PROFILE = "ai"


def _vscode_user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "Code"


def _vscode_profile_configured(profile_name: str) -> bool:
    storage_path = _vscode_user_data_dir() / "User" / "globalStorage" / "storage.json"
    try:
        data = json.loads(storage_path.read_text(encoding="utf-8"))
        profiles = data.get("userDataProfiles", [])
        return any(profile.get("name") == profile_name for profile in profiles)
    except (OSError, ValueError) as exc:
        logger.debug(
            "Could not read VS Code profile list from %s (%s); treating '%s' as not configured",
            storage_path,
            exc,
            profile_name,
        )
        return False


def _open_vscode(path: str, profile_name: str = VSCODE_PREFERRED_PROFILE) -> None:
    use_preferred_profile = _vscode_profile_configured(profile_name)
    cmd = ["code"]
    if use_preferred_profile:
        cmd += ["--profile", profile_name]
    cmd.append(path)
    logger.info(
        "Opening VS Code at %s (profile=%s)",
        path,
        profile_name if use_preferred_profile else "default",
    )
    try:
        subprocess.run(
            cmd,
            check=False,
            shell=(sys.platform == "win32"),
        )
    except FileNotFoundError:
        logger.warning("VS Code 'code' command not found on PATH")
