from __future__ import annotations

import logging
from pathlib import Path

from rich.syntax import Syntax
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.geometry import Offset
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import DirectoryTree, Footer, Header, Rule, Static

from syntrive.tui.widgets import LogPanel

logger = logging.getLogger(__name__)

_SYNTAX_MAP = {
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".csv": "text",
}

_MAX_PREVIEW_BYTES = 256_000


class DirectoryViewScreen(Screen):
    BINDINGS = [
        Binding("q", "close", "Close", show=True),
        Binding("escape", "close", "Close", show=True),
    ]

    DEFAULT_CSS = """
    DirectoryViewScreen {
        layout: vertical;
        overflow: hidden hidden;
    }
    #dir-main {
        height: 1fr;
    }
    #dir-tree {
        width: 35%;
        border: solid $primary;
    }
    #file-preview {
        width: 65%;
        border: solid $secondary;
        padding: 0 1;
    }
    DirectoryViewScreen LogPanel { margin: 0; }
    """

    def __init__(self, process_dir: Path) -> None:
        super().__init__()
        self._process_dir = process_dir

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Horizontal(
            DirectoryTree(str(self._process_dir), id="dir-tree"),
            Static("Select a file to preview →", id="file-preview"),
            id="dir-main",
        )
        yield LogPanel(id="dir-log")
        yield Footer()

    def action_close(self) -> None:
        self.app.pop_screen()

    @on(DirectoryTree.FileSelected)
    def _on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = Path(event.path)
        preview = self.query_one("#file-preview", Static)
        preview.update(_render_file(path))


class FileViewScreen(Screen):
    BINDINGS = [
        Binding("q", "close", "Close", show=True),
        Binding("escape", "close", "Close", show=True),
    ]

    DEFAULT_CSS = """
    FileViewScreen {
        layout: vertical;
        padding: 1 2;
        overflow: hidden hidden;
    }
    #file-title { color: $accent; text-style: bold; margin-bottom: 1; }
    #file-content { height: 1fr; }
    FileViewScreen LogPanel { margin: 0; }
    """

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self._file_path = file_path

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(str(self._file_path), id="file-title")
        yield Rule()
        yield Static(_render_file(self._file_path), id="file-content")
        yield LogPanel(id="file-log")
        yield Footer()

    def action_close(self) -> None:
        self.app.pop_screen()


def _render_file(path: Path) -> str | Syntax:
    if not path.is_file():
        return f"[red]File not found:[/red] {path}"

    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
        size_kb = path.stat().st_size / 1024
        return (
            f"[bold]Image file[/bold]: {path.name}\n"
            f"Size: {size_kb:.1f} KB\n"
            f"Path: {path}\n\n"
            f"[dim]Terminal image rendering is not supported.\n"
            f"Use --code to open the directory in VS Code for image preview.[/dim]"
        )

    lexer = _SYNTAX_MAP.get(suffix, "text")
    try:
        data = path.read_bytes()
        if len(data) > _MAX_PREVIEW_BYTES:
            content = data[:_MAX_PREVIEW_BYTES].decode("utf-8", errors="replace")
            content += f"\n\n... [dim](truncated at {_MAX_PREVIEW_BYTES // 1024}KB)[/dim]"
        else:
            content = data.decode("utf-8", errors="replace")

        return Syntax(
            content,
            lexer,
            line_numbers=True,
            word_wrap=True,
            theme="monokai",
        )
    except Exception as exc:
        logger.warning("Cannot render file %s: %s", path, exc)
        return f"[red]Cannot read file:[/red] {exc}"
