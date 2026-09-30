from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.style import Style
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Rule, Static

logger = logging.getLogger(__name__)


_PREVIEW_COLS = 44
_PREVIEW_ROWS = 28


def _render_image_preview(abs_path: Path, cols: int = _PREVIEW_COLS, rows: int = _PREVIEW_ROWS) -> Text:
    try:
        from PIL import Image, ImageEnhance  # type: ignore[import]
    except ImportError:
        return Text(
            f"(Pillow not installed — pip install Pillow for image preview)\n{abs_path.name}",
            style="dim",
        )

    try:
        img = Image.open(abs_path).convert("RGB")
        orig_w, orig_h = img.size

        target_w = cols
        target_h = rows * 2
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(2, int(orig_h * ratio))
        if new_h % 2 != 0:
            new_h += 1

        img = img.resize((new_w, new_h), Image.LANCZOS)

        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.3)

        pixels = list(img.getdata())

        result = Text(no_wrap=True)
        for row in range(0, new_h - 1, 2):
            if row > 0:
                result.append("\n")
            for col in range(new_w):
                tr, tg, tb = pixels[row * new_w + col]
                br, bg, bb = pixels[(row + 1) * new_w + col]
                result.append(
                    "▀",
                    style=Style(
                        color=f"rgb({tr},{tg},{tb})",
                        bgcolor=f"rgb({br},{bg},{bb})",
                    ),
                )

        size_kb = abs_path.stat().st_size / 1024 if abs_path.exists() else 0
        result.append(
            f"\n{abs_path.name}  {orig_w}×{orig_h}  {size_kb:.1f} KB",
            style="dim",
        )
        return result

    except Exception as exc:
        logger.debug("_render_image_preview: failed for %s: %s", abs_path, exc)
        return _metadata_fallback(abs_path, hint=str(exc))


def _metadata_fallback(abs_path: Path, hint: str = "") -> Text:
    if not abs_path.exists():
        return Text(f"{abs_path.name}\n(file not found)", style="dim")
    size_kb = abs_path.stat().st_size / 1024
    try:
        from PIL import Image  # type: ignore[import]
        img = Image.open(abs_path)
        w, h = img.size
        fmt = img.format or "?"
        suffix = f"  [{hint}]" if hint else ""
        return Text(f"{abs_path.name}\n{w}×{h}  {fmt}  {size_kb:.1f} KB{suffix}", style="dim")
    except Exception:
        return Text(f"{abs_path.name}\n{size_kb:.1f} KB", style="dim")


_LANGUAGE_CODES: tuple[str, ...] = (
    "zh", "en", "fr", "de", "ja", "ko", "es", "pt", "ru", "ar", "hi", "it",
)


class MetadataEditScreen(Screen):
    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("ctrl+o", "open_in_viewer", "Open Image", show=True),
        Binding("escape", "dismiss_screen", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    MetadataEditScreen {
        layout: vertical;
    }
    #meta-body {
        height: 1fr;
    }
    #meta-form {
        width: 60%;
        border: solid $primary;
        padding: 1 2;
    }
    #meta-preview {
        width: 40%;
        border: solid $secondary;
        padding: 1 1;
    }
    #form-header { text-style: bold; color: $accent; }
    .field-label { color: $accent; text-style: bold; margin-top: 1; }
    #lang-list  { height: 10; border: solid $border; }
    #cover-list { height: 12; border: solid $border; }
    #preview-title { color: $accent; text-style: bold; }
    #preview-content { height: 1fr; }
    """

    def __init__(self, job_id: int, process_dir: Path, db_path: Path) -> None:
        super().__init__()
        self._job_id = job_id
        self._process_dir = process_dir
        self._db_path = db_path

        self._images: list = []
        self._selected_language: Optional[str] = None
        self._selected_cover_rel: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="meta-body"):
            with ScrollableContainer(id="meta-form"):
                yield Label("Book Metadata Editor", id="form-header")
                yield Rule()
                yield Label("Title", classes="field-label")
                yield Input(placeholder="Book title…", id="title-input")
                yield Label("Language (ISO 639-1)", classes="field-label")
                yield ListView(
                    *[ListItem(Label(code), id=f"lang-{code}") for code in _LANGUAGE_CODES],
                    id="lang-list",
                )
                yield Label("Cover Image", classes="field-label")
                yield Label("[dim]Loading images…[/dim]", id="cover-loading-label")
                yield ListView(id="cover-list")
            with Vertical(id="meta-preview"):
                yield Label("Cover Preview", id="preview-title")
                yield Rule()
                yield Static("(no cover selected)", id="preview-content")
        yield Footer()

    def on_mount(self) -> None:
        self._load_data()

    @work(thread=True)
    def _load_data(self) -> None:
        try:
            from syntrive.services.job_service import JobService
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job, Book, BookImage

            svc = JobService(self._db_path)
            new_count = svc.ensure_book_images_synced(self._job_id)
            if new_count > 0:
                logger.info(
                    "MetadataEditScreen: synced %d new image(s) for job %d",
                    new_count, self._job_id,
                )

            with get_db_session(self._db_path) as db:
                job = db.query(Job).filter_by(id=self._job_id).first()
                if job is None:
                    self.app.call_from_thread(
                        self.notify, f"Job {self._job_id} not found.", severity="error"
                    )
                    return

                book = db.query(Book).filter_by(id=job.book_id).first()
                images = (
                    db.query(BookImage)
                    .filter_by(book_id=job.book_id)
                    .order_by(BookImage.name)
                    .all()
                )
                db.expunge_all()

            self.app.call_from_thread(self._apply_loaded_data, book, list(images))

        except Exception as exc:
            logger.error("MetadataEditScreen._load_data: %s", exc, exc_info=True)
            self.app.call_from_thread(self.notify, f"Load failed: {exc}", severity="error")

    def _apply_loaded_data(self, book, images: list) -> None:
        self._images = images

        self.query_one("#title-input", Input).value = book.title or ""

        self._selected_language = book.language
        if book.language in _LANGUAGE_CODES:
            idx = _LANGUAGE_CODES.index(book.language)
            lang_list = self.query_one("#lang-list", ListView)
            lang_list.index = idx

        cover_list = self.query_one("#cover-list", ListView)
        cover_list.clear()
        for img in images:
            cover_list.append(ListItem(Label(img.name), id=f"cover-img-{img.id}"))

        try:
            self.query_one("#cover-loading-label", Label).update("")
        except Exception:
            pass

        self._selected_cover_rel = book.cover
        if book.cover:
            current_name = Path(book.cover).name
            for i, img in enumerate(images):
                if img.name == current_name:
                    cover_list.index = i
                    self._update_preview_for_rel(img.path)
                    break

        self.notify("Metadata loaded.", timeout=2)

    @on(ListView.Highlighted, "#lang-list")
    def _on_lang_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        item_id = event.item.id or ""
        if item_id.startswith("lang-"):
            self._selected_language = item_id.removeprefix("lang-")

    @on(ListView.Highlighted, "#cover-list")
    def _on_cover_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        item_id = event.item.id or ""
        if not item_id.startswith("cover-img-"):
            return
        try:
            img_id = int(item_id.removeprefix("cover-img-"))
        except ValueError:
            return

        img = next((i for i in self._images if i.id == img_id), None)
        if img is None:
            return

        self._selected_cover_rel = img.path
        self._update_preview_for_rel(img.path)

    def _update_preview_for_rel(self, rel_path: str) -> None:
        abs_path = self._process_dir / rel_path
        preview = self.query_one("#preview-content", Static)
        w, h = preview.size
        cols = (w - 1) if w > 0 else _PREVIEW_COLS
        rows = (h - 1) if h > 0 else _PREVIEW_ROWS
        self._render_preview_worker(abs_path, max(10, cols), max(4, rows))

    @work(thread=True)
    def _render_preview_worker(self, abs_path: Path, cols: int, rows: int) -> None:
        rendered = _render_image_preview(abs_path, cols, rows)
        self.app.call_from_thread(self.query_one("#preview-content", Static).update, rendered)

    def action_save(self) -> None:
        title = self.query_one("#title-input", Input).value.strip()
        self._save_background(title, self._selected_language, self._selected_cover_rel)

    @work(thread=True)
    def _save_background(
        self,
        title: str,
        language: Optional[str],
        cover_rel: Optional[str],
    ) -> None:
        try:
            from syntrive.services.job_service import JobService

            svc = JobService(self._db_path, holder_kind="tui")
            if language:
                svc.update_book_language(self._job_id, language)
            if title:
                svc.update_book_title(self._job_id, title)
            svc.update_book_cover(self._job_id, cover_rel)

            logger.info(
                "MetadataEditScreen: saved job=%d title=%r lang=%r cover=%r",
                self._job_id, title, language, cover_rel,
            )
            self.app.call_from_thread(self.notify, "Metadata saved.")
            self.app.call_from_thread(self.dismiss)

        except Exception as exc:
            logger.error("MetadataEditScreen._save_background: %s", exc, exc_info=True)
            self.app.call_from_thread(self.notify, f"Save failed: {exc}", severity="error")

    def action_open_in_viewer(self) -> None:
        if self._selected_cover_rel is None:
            self.notify("No cover selected.", severity="warning")
            return
        abs_path = self._process_dir / self._selected_cover_rel
        if not abs_path.exists():
            self.notify(f"File not found: {abs_path.name}", severity="error")
            return
        self._launch_system_viewer(abs_path)

    @work(thread=True)
    def _launch_system_viewer(self, abs_path: Path) -> None:
        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                import os as _os
                _os.startfile(str(abs_path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(abs_path)])
            else:
                subprocess.Popen(["xdg-open", str(abs_path)])
            self.app.call_from_thread(
                self.notify, f"Opened: {abs_path.name}"
            )
        except Exception as exc:
            logger.error("_launch_system_viewer: %s", exc)
            self.app.call_from_thread(
                self.notify, f"Cannot open image: {exc}", severity="error"
            )

    def action_dismiss_screen(self) -> None:
        self.dismiss()
