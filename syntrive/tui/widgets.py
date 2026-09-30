from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.geometry import Offset
from textual.widgets import RichLog, Rule, Static, TextArea


class LogPanel(RichLog):
    BORDER_TITLE = "Log"
    DEFAULT_CSS = """
    LogPanel {
        height: 7;
        border: solid $surface-lighten-1;
        margin: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(max_lines=200, markup=True, auto_scroll=True, **kwargs)


class LogViewerModal(ModalScreen):
    BINDINGS = [Binding("escape,q", "close", "Close")]

    DEFAULT_CSS = """
    LogViewerModal {
        align: center middle;
    }
    #log-viewer-box {
        width: 90%;
        height: 80%;
        border: solid $primary;
        background: $surface;
    }
    #log-viewer-hint {
        padding: 0 1;
        color: $text-muted;
    }
    #log-viewer-area {
        height: 1fr;
    }
    """

    def __init__(self, log_lines: list[str]) -> None:
        super().__init__()
        self._log_lines = log_lines

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                "[bold]Log Viewer[/bold]  "
                "[dim]Select text then Ctrl+C to copy · Esc / q to close[/dim]",
                id="log-viewer-hint",
            ),
            Rule(),
            TextArea(
                "\n".join(self._log_lines),
                read_only=True,
                id="log-viewer-area",
            ),
            id="log-viewer-box",
        )

    def action_close(self) -> None:
        self.dismiss()
