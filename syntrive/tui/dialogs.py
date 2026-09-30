from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from rich.markup import escape as markup_escape
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.geometry import Offset
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Rule,
    Static,
)

from syntrive.tui import tk_widgets
from syntrive.tui.widgets import LogPanel

logger = logging.getLogger(__name__)

TTS_TEXT_CLEAN_NOTICE = (
    "[bold yellow]⚠ Tip:[/bold yellow] [yellow]run the [bold]tts-text-clean[/bold] "
    "skill for this book (title merge, dividers, …) after cleaning and before "
    "step 5/9. Each rule is checked against this book and needs your approval."
    "[/yellow]"
)


def clean_step_notice() -> str:
    return TTS_TEXT_CLEAN_NOTICE


_FILEPICKER_SCRIPT = """\
import json, sys
import tkinter as tk
from tkinter import filedialog

params = json.loads(sys.stdin.read())
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
filetypes = [tuple(ft) for ft in params.get('filetypes', [])]
path = filedialog.askopenfilename(
    title=params.get('title', 'Select file'),
    filetypes=filetypes,
    initialdir=params.get('initial_dir') or '',
)
root.destroy()
sys.stdout.write(json.dumps(path or ''))
"""


def native_file_picker(
    title: str,
    filetypes: list[tuple[str, str]],
    initial_dir: str = "",
    timeout: int = 300,
) -> str:
    params = json.dumps(
        {
            "title": title,
            "filetypes": [list(ft) for ft in filetypes],
            "initial_dir": initial_dir,
        }
    )
    logger.debug("native_file_picker: launching subprocess dialog -- title=%r", title)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _FILEPICKER_SCRIPT],
            input=params,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            detail = stderr[-300:]
            if "_tkinter" in stderr:
                detail = f"{detail} -- {tk_widgets.tkinter_unavailable_hint()}"
            logger.warning(
                "native_file_picker: subprocess exited with code %d -- stderr: %s",
                result.returncode,
                detail,
            )
            return ""
        if result.stderr.strip():
            logger.debug(
                "native_file_picker: subprocess stderr: %s",
                result.stderr.strip()[:200],
            )
        path = json.loads(result.stdout) if result.stdout else ""
        logger.debug(
            "native_file_picker: dialog returned path=%r", path or "(cancelled)"
        )
        return path
    except subprocess.TimeoutExpired:
        logger.warning("native_file_picker: dialog timed out after %ds", timeout)
        return ""
    except Exception as exc:
        logger.error("native_file_picker: subprocess failed -- %s", exc, exc_info=True)
        return ""


def is_gui_available() -> bool:
    system = platform.system()
    if system == "Windows":
        return True
    if system == "Darwin":
        has_display = bool(os.environ.get("DISPLAY"))
        via_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
        available = has_display or not via_ssh
        logger.debug(
            "is_gui_available: Darwin -- DISPLAY=%r SSH=%r result=%s",
            os.environ.get("DISPLAY"),
            via_ssh,
            available,
        )
        return available
    has_x11 = bool(os.environ.get("DISPLAY"))
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    available = has_x11 or has_wayland
    logger.debug(
        "is_gui_available: %s -- DISPLAY=%r WAYLAND_DISPLAY=%r result=%s",
        system,
        os.environ.get("DISPLAY"),
        os.environ.get("WAYLAND_DISPLAY"),
        available,
    )
    return available


class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-box {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #confirm-message { margin-bottom: 1; }
    #confirm-buttons { height: auto; }
    #btn-yes { margin-right: 2; }
    """

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self._title = title
        self._message = message

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"[bold]{self._title}[/bold]"),
            Rule(),
            Static(self._message, id="confirm-message"),
            Horizontal(
                Button("Yes [Y]", id="btn-yes", variant="default"),
                Button("No [N]", id="btn-no", variant="default"),
                id="confirm-buttons",
            ),
            id="confirm-box",
        )

    @on(Button.Pressed, "#btn-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#btn-no")
    def _no(self) -> None:
        self.dismiss(False)

    def on_key(self, event) -> None:
        if event.key == "y":
            self.dismiss(True)
        elif event.key in ("n", "escape"):
            self.dismiss(False)


class ChoiceScreen(ModalScreen[Optional[str]]):
    DEFAULT_CSS = """
    ChoiceScreen {
        align: center middle;
    }
    #choice-box {
        width: 60;
        height: auto;
        max-height: 30;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, title: str, choices: list[str]) -> None:
        super().__init__()
        self._title = title
        self._choices = choices

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"[bold]{self._title}[/bold]"),
            Rule(),
            ListView(
                *[ListItem(Label(c), name=c) for c in self._choices],
                id="choice-list",
            ),
            Rule(),
            Button("Cancel", id="btn-cancel", variant="default"),
            id="choice-box",
        )

    @on(ListView.Selected)
    def _selected(self, event: ListView.Selected) -> None:
        if event.item.name:
            self.dismiss(event.item.name)

    @on(Button.Pressed, "#btn-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class FilePathInputScreen(ModalScreen[Optional[str]]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    FilePathInputScreen {
        align: center middle;
    }
    #fp-box {
        width: 72;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #fp-title  { margin-bottom: 1; }
    #fp-hint   { margin-bottom: 1; color: $text-muted; }
    #fp-input  { margin-bottom: 0; }
    #fp-error  { height: 1; color: $error; margin-top: 0; }
    #fp-buttons { height: auto; margin-top: 1; }
    #fp-ok { margin-right: 2; }
    """

    def __init__(
        self,
        title: str = "Enter File Path",
        hint: str = "",
        expected_suffix: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._hint = hint
        self._expected_suffix = expected_suffix.lower() if expected_suffix else ""

    def compose(self) -> ComposeResult:
        hint_widgets: list = []
        if self._hint:
            hint_widgets.append(Label(self._hint, id="fp-hint"))
        yield Vertical(
            Label(f"[bold]{self._title}[/bold]", id="fp-title"),
            Rule(),
            *hint_widgets,
            Input(placeholder="/path/to/file", id="fp-input"),
            Label("", id="fp-error"),
            Horizontal(
                Button("OK [Enter]", id="fp-ok", variant="primary"),
                Button("Cancel [Esc]", id="fp-cancel", variant="default"),
                id="fp-buttons",
            ),
            id="fp-box",
        )

    def on_mount(self) -> None:
        self.query_one("#fp-input", Input).focus()

    def _validate_and_dismiss(self) -> None:
        input_widget = self.query_one("#fp-input", Input)
        error_label = self.query_one("#fp-error", Label)
        path_str = input_widget.value.strip()

        if not path_str:
            error_label.update("[red]Path cannot be empty.[/red]")
            return

        path = Path(path_str)
        if not path.exists():
            error_label.update(f"[red]File not found: {path_str}[/red]")
            logger.debug("FilePathInputScreen: path not found: %s", path_str)
            return

        if self._expected_suffix and not path_str.lower().endswith(
            self._expected_suffix
        ):
            error_label.update(f"[red]Expected a {self._expected_suffix} file.[/red]")
            return

        resolved = str(path.resolve())
        logger.debug("FilePathInputScreen: accepted path=%s", resolved)
        self.dismiss(resolved)

    @on(Button.Pressed, "#fp-ok")
    def _on_ok(self) -> None:
        self._validate_and_dismiss()

    @on(Button.Pressed, "#fp-cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#fp-input")
    def _on_submit(self) -> None:
        self._validate_and_dismiss()

    def action_cancel(self) -> None:
        self.dismiss(None)


class CleaningRulesScreen(ModalScreen[list[dict]]):
    DEFAULT_CSS = """
    CleaningRulesScreen {
        align: center middle;
    }
    #rules-box {
        width: 70;
        height: auto;
        max-height: 35;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #rules-hint  { color: $text-muted; margin-bottom: 1; }
    #rules-list  { height: auto; max-height: 20; }
    #rules-btn-row { height: auto; margin-top: 1; }
    #btn-save-rules { margin-right: 2; }
    """

    BINDINGS = [
        Binding("s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, rules: list[dict]) -> None:
        super().__init__()
        self._rules: list[dict] = [dict(r) for r in rules]

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        items = []
        for r in self._rules:
            mark = "[green]✓[/green]" if r["enabled"] else "[red]✗[/red]"
            label_text = f"{mark}  {r['display_name']}"
            if r.get("description"):
                label_text += f"  [dim]— {r['description']}[/dim]"
            items.append(ListItem(Label(label_text), name=r["name"]))

        yield Vertical(
            Static("[bold]Manage Cleaning Rules[/bold]"),
            Rule(),
            Static(
                "[dim]↑↓ navigate · Space/Enter to toggle · S to save · Esc to cancel[/dim]",
                id="rules-hint",
            ),
            ListView(*items, id="rules-list"),
            Rule(),
            Horizontal(
                Button("Save [S]", id="btn-save-rules", variant="default"),
                Button("Cancel [Esc]", id="btn-cancel-rules", variant="default"),
                id="rules-btn-row",
            ),
            id="rules-box",
        )

    @on(ListView.Selected)
    def _toggle_selected(self, event: ListView.Selected) -> None:
        rule_name = event.item.name
        if rule_name is None:
            return
        for r in self._rules:
            if r["name"] == rule_name:
                r["enabled"] = not r["enabled"]
                break
        self._refresh_list()

    def on_key(self, event) -> None:
        if event.key == "space":
            lv = self.query_one("#rules-list", ListView)
            if lv.highlighted_child is not None:
                lv.action_select_cursor()
        elif event.key in ("s",):
            self.dismiss(self._rules)
        elif event.key == "escape":
            self.dismiss(None)

    @on(Button.Pressed, "#btn-save-rules")
    def _save(self) -> None:
        self.dismiss(self._rules)

    @on(Button.Pressed, "#btn-cancel-rules")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self.dismiss(self._rules)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _refresh_list(self) -> None:
        lv = self.query_one("#rules-list", ListView)
        focused_idx = lv.index or 0
        lv.clear()
        for r in self._rules:
            mark = "[green]✓[/green]" if r["enabled"] else "[red]✗[/red]"
            label_text = f"{mark}  {r['display_name']}"
            if r.get("description"):
                label_text += f"  [dim]— {r['description']}[/dim]"
            lv.append(ListItem(Label(label_text), name=r["name"]))
        if self._rules:
            lv.index = min(focused_idx, len(self._rules) - 1)


class TextExtractionOptionsScreen(ModalScreen[Optional[dict]]):
    DEFAULT_CSS = """
    TextExtractionOptionsScreen {
        align: center middle;
    }
    #textfmt-box {
        width: 78;
        height: auto;
        max-height: 24;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #textfmt-hint { color: $text-muted; margin-bottom: 1; }
    #textfmt-list { height: auto; max-height: 10; }
    #textfmt-btn-row { height: auto; margin-top: 1; }
    #btn-save-textfmt { margin-right: 2; }
    """

    BINDINGS = [
        Binding("s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, formats: dict[str, bool], primary: str) -> None:
        super().__init__()
        self._formats: dict[str, bool] = dict(formats)
        self._primary: str = primary

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def _build_items(self) -> list[ListItem]:
        from syntrive.workflow.engine import TEXT_EXTRACTION_FORMAT_LABELS

        items = []
        for name, label in TEXT_EXTRACTION_FORMAT_LABELS.items():
            mark = "[green]✓[/green]" if self._formats.get(name) else "[red]✗[/red]"
            star = "  [yellow]★ primary[/yellow]" if name == self._primary else ""
            items.append(ListItem(Label(f"{mark}  {label}{star}"), name=name))
        return items

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold]Choose TRANSCRIPT_TEXT Output Format(s)[/bold]"),
            Rule(),
            Static(
                "[dim]↑↓ navigate · Space/Enter to toggle generate · "
                "D to set as primary (TTS input) · S to save · Esc to cancel[/dim]",
                id="textfmt-hint",
            ),
            ListView(*self._build_items(), id="textfmt-list"),
            Rule(),
            Horizontal(
                Button("Save [S]", id="btn-save-textfmt", variant="default"),
                Button("Cancel [Esc]", id="btn-cancel-textfmt", variant="default"),
                id="textfmt-btn-row",
            ),
            id="textfmt-box",
        )

    def _refresh_list(self) -> None:
        lv = self.query_one("#textfmt-list", ListView)
        focused_idx = lv.index or 0
        lv.clear()
        for item in self._build_items():
            lv.append(item)
        lv.index = min(focused_idx, len(self._formats) - 1)

    @on(ListView.Selected)
    def _toggle_selected(self, event: ListView.Selected) -> None:
        name = event.item.name
        if name is None:
            return
        self._formats[name] = not self._formats.get(name, False)
        if not self._formats[name] and self._primary == name:
            fallback = next((n for n, v in self._formats.items() if v), None)
            if fallback:
                self._primary = fallback
        self._refresh_list()

    def _set_primary_highlighted(self) -> None:
        lv = self.query_one("#textfmt-list", ListView)
        if lv.highlighted_child is None or lv.highlighted_child.name is None:
            return
        name = lv.highlighted_child.name
        self._formats[name] = True
        self._primary = name
        self._refresh_list()

    def on_key(self, event) -> None:
        if event.key == "space":
            lv = self.query_one("#textfmt-list", ListView)
            if lv.highlighted_child is not None:
                lv.action_select_cursor()
        elif event.key == "d":
            self._set_primary_highlighted()
        elif event.key == "s":
            self.dismiss({"formats": self._formats, "primary": self._primary})
        elif event.key == "escape":
            self.dismiss(None)

    @on(Button.Pressed, "#btn-save-textfmt")
    def _save(self) -> None:
        self.dismiss({"formats": self._formats, "primary": self._primary})

    @on(Button.Pressed, "#btn-cancel-textfmt")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self.dismiss({"formats": self._formats, "primary": self._primary})

    def action_cancel(self) -> None:
        self.dismiss(None)


class WorkflowRunScreen(Screen):
    BINDINGS = [
        Binding("y", "confirm", "Confirm"),
        Binding("n", "advance_step", "Next"),
        Binding("b", "go_back", "Back"),
        Binding("p", "pause", "Pause"),
        Binding("m", "set_merge_mode", "Merge Mode"),
        Binding("r", "manage_rules", "Manage Rules"),
        Binding("x", "set_text_formats", "Text Formats"),
        Binding("t", "set_transcript_path", "Transcript Path"),
        Binding("e", "edit_tts_params", "Edit Params"),
        Binding("q", "close_screen", "Close"),
    ]

    DEFAULT_CSS = """
    WorkflowRunScreen {
        layout: vertical;
        overflow: hidden hidden;
    }
    #wfrun-content { padding: 1 2; height: 1fr; }
    #wfrun-title  { color: $accent; text-style: bold; margin-bottom: 1; }
    #wfrun-body   { height: 1fr; }
    WorkflowRunScreen LogPanel { margin: 0 0; }
    """

    def __init__(self, job, db_path: Path) -> None:
        super().__init__()
        self._job = job
        self._db_path = db_path
        self._engine = None
        self._confirm_running: bool = False

    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        if widget is self:
            return None, None
        return widget, offset

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("", id="wfrun-title"),
            Rule(),
            Static("", id="wfrun-body"),
            id="wfrun-content",
        )
        yield Rule()
        yield LogPanel(id="wfrun-log")
        yield Footer()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        current = getattr(self, "_current_step", None)

        if action == "confirm":
            return not self._confirm_running

        if action == "advance_step":
            if current is None or self._engine.get_next_step(current) is None:
                return None
            return not self._confirm_running

        from syntrive.workflow.engine import WorkflowStep

        if action == "set_merge_mode":
            if current != WorkflowStep.TRANSCRIPT_EXTRACT:
                return None
            return not self._confirm_running


        if action == "manage_rules":
            if current != WorkflowStep.TRANSCRIPT_CLEAN:
                return None
            return not self._confirm_running

        if action == "set_text_formats":
            if current != WorkflowStep.TRANSCRIPT_TEXT:
                return None
            return not self._confirm_running

        if action == "set_transcript_path":
            if current != WorkflowStep.TRANSCRIPT_REVIEW:
                return None
            return not self._confirm_running

        if action == "edit_tts_params":
            if current != WorkflowStep.TTS_CONFIG:
                return None
            return not self._confirm_running

        return True

    def on_mount(self) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        self._event_loop = asyncio.get_running_loop()

        self._engine = WorkflowEngine(
            job=self._job,
            db_path=self._db_path,
            holder_kind="tui",
        )

        step_name = self._job.current_step or WorkflowStep.BOOTSTRAP.value
        if step_name == "transcript":
            step_name = WorkflowStep.TRANSCRIPT_EXTRACT.value
        try:
            step = WorkflowStep(step_name)
        except ValueError:
            step = WorkflowStep.BOOTSTRAP

        self._current_step = step
        if step == WorkflowStep.DONE:
            self._show_done_state()
        else:
            self._show_step_summary(step)

    def _show_step_summary(self, step) -> None:
        from syntrive.workflow.engine import _STEP_LABELS, _ORDERED_STEPS, WorkflowStep

        if step == WorkflowStep.DONE:
            self._show_done_state()
            return

        title = self.query_one("#wfrun-title", Static)
        body = self.query_one("#wfrun-body", Static)

        actionable = [s for s in _ORDERED_STEPS if s != WorkflowStep.DONE]
        idx = actionable.index(step) + 1
        total = len(actionable)
        label = _STEP_LABELS.get(step, step.value)

        title.update(f"STEP {idx}/{total}: {label}")

        tasks = self._engine.describe_tasks(step)
        task_lines = "\n".join(f"  • {t}" for t in tasks)

        artifact_warning = ""
        if self._engine.check_artifacts_exist(step):
            artifact_warning = (
                "\n\n[yellow]⚠ Existing artifacts detected.[/yellow] "
                "Confirming will prompt for Overwrite / Skip."
            )

        if step == WorkflowStep.TRANSCRIPT_CLEAN:
            rules = self._engine.get_all_cleaning_rules_with_status()
            enabled = [r for r in rules if r["enabled"]]
            if enabled:
                rules_lines = "\n".join(
                    f"  [green]✓[/green] {r['display_name']}" for r in enabled
                )
                rules_text = (
                    f"\n\n[bold]Active rules ({len(enabled)}):[/bold]\n{rules_lines}"
                )
            elif rules:
                rules_text = "\n\n[yellow]No cleaning rules enabled. Press R to enable rules.[/yellow]"
            else:
                rules_text = (
                    "\n\n[yellow]No cleaning rules found. Run Bootstrap first.[/yellow]"
                )
            logger.info(
                "Step 4/9 summary: tts-text-clean notice shown (enabled_rules=%d)",
                len(enabled),
            )
            body.update(
                f"{clean_step_notice()}\n\n"
                f"[bold]Tasks:[/bold]\n{task_lines}"
                f"{artifact_warning}"
                f"{rules_text}\n\n"
                f"[dim]Press Y to run cleaning · N to next · R to manage rules · B to go back[/dim]"
            )
        elif step == WorkflowStep.TRANSCRIPT_TEXT:
            body.update(
                f"[bold]Tasks:[/bold]\n{task_lines}"
                f"{artifact_warning}\n\n"
                f"[dim]Press X to change formats · Y to confirm · N to next · B to go back[/dim]"
            )
        elif step == WorkflowStep.TRANSCRIPT_REVIEW:
            body.update(
                f"[bold]Tasks:[/bold]\n{task_lines}"
                f"{artifact_warning}\n\n"
                f"[dim]Press T to change transcript_path · Y to confirm · "
                f"N to next · B to go back[/dim]"
            )
        elif step == WorkflowStep.TTS_CONFIG:
            summary = self._engine.get_tts_config_summary()
            if summary:
                cfg_lines = "\n".join(f"  {k}: {v}" for k, v in summary.items())
                cfg_text = f"[bold]Current TTS configuration:[/bold]\n{cfg_lines}"
            else:
                cfg_text = "[yellow]No TTS configuration saved yet.[/yellow]"
            body.update(
                f"{cfg_text}"
                f"{artifact_warning}\n\n"
                f"[dim]Press E to reconfigure — opens the same TTS editor as the "
                f"global E key · Y to confirm · N to next · B to go back[/dim]"
            )
        else:
            body.update(
                f"[bold]Tasks:[/bold]\n{task_lines}"
                f"{artifact_warning}\n\n"
                f"[dim]Press Y to confirm · N to next · B to go back · P to pause[/dim]"
            )

    def _show_done_state(self) -> None:
        self._confirm_running = True
        self.refresh_bindings()
        self.query_one("#wfrun-title", Static).update(
            "[bold green]Job Complete[/bold green]"
        )
        self.query_one("#wfrun-body", Static).update(
            "[green]All workflow steps completed successfully.[/green]\n\n"
            "[dim]Close this screen to return to the job list.[/dim]"
        )

    def _ask_overwrite_choice_sync(self, description: str) -> Optional[str]:
        result_holder: list[Optional[str]] = [None]
        done = threading.Event()

        async def _ask():
            try:
                with self.app._context():
                    choice = await self.app.push_screen_wait(
                        ChoiceScreen(
                            title="Artifacts already exist",
                            choices=["Overwrite", "Skip (keep existing)"],
                        )
                    )
                result_holder[0] = choice
            except Exception:
                result_holder[0] = None
            finally:
                done.set()

        asyncio.run_coroutine_threadsafe(_ask(), self._event_loop)
        done.wait(timeout=300)
        return result_holder[0]

    def action_confirm(self) -> None:
        if self._confirm_running:
            return
        self._confirm_running = True
        self.refresh_bindings()
        self._run_step_in_thread()

    def action_go_back(self) -> None:
        prev = self._engine.get_previous_step(self._current_step)
        if prev is not None:
            self._current_step = prev
            self._show_step_summary(prev)
            self.app.notify("Went back to previous step.")
        else:
            self.app.notify("Already at first step.")

    def action_pause(self) -> None:
        from syntrive.workflow.engine import StepAction

        self._engine.record_step(self._current_step, StepAction.PAUSED)
        self.app.notify("Progress saved. Resume anytime.")
        self.dismiss()

    def action_close_screen(self) -> None:
        self.dismiss()

    def action_advance_step(self) -> None:
        if self._confirm_running:
            return

        nxt = self._engine.get_next_step(self._current_step)
        if nxt is not None:
            self._current_step = nxt
            self._show_step_summary(nxt)
            self.app.notify("Previewed next step (no action taken).")
        else:
            self.app.notify("Already at last step.")

    async def action_set_merge_mode(self) -> None:
        from syntrive.workflow.engine import WorkflowStep, MERGE_MODE_LABELS

        if self._current_step != WorkflowStep.TRANSCRIPT_EXTRACT:
            return
        if self._confirm_running:
            return

        choices = list(MERGE_MODE_LABELS.values())
        choices.append("Auto-detect (reset override)")

        choice = await self.app.push_screen_wait(
            ChoiceScreen(title="Set Merge Mode Override", choices=choices)
        )
        if choice is None:
            return

        if choice == "Auto-detect (reset override)":
            self._engine.set_merge_mode_override(None)
            self.app.notify("Merge mode override cleared — auto-detect will be used.")
        else:
            selected_key = next(
                (k for k, v in MERGE_MODE_LABELS.items() if v == choice), None
            )
            if selected_key:
                self._engine.set_merge_mode_override(selected_key)
                self.app.notify(f"Merge mode override set to: {selected_key}")

        self._show_step_summary(self._current_step)

    @work
    async def action_manage_rules(self) -> None:
        from syntrive.workflow.engine import WorkflowStep

        if self._current_step != WorkflowStep.TRANSCRIPT_CLEAN:
            return
        if self._confirm_running:
            return

        rules = self._engine.get_all_cleaning_rules_with_status()
        if not rules:
            self.app.notify(
                "No cleaning rules found in the database. "
                "Run Bootstrap first to seed the rule definitions.",
                severity="warning",
            )
            return

        updated = await self.app.push_screen_wait(CleaningRulesScreen(rules))
        if updated is None:
            return

        changed = 0
        for new_r, old_r in zip(updated, rules):
            if new_r["enabled"] != old_r["enabled"]:
                self._engine.set_cleaning_rule_enabled(new_r["name"], new_r["enabled"])
                changed += 1

        if changed:
            self.app.notify(f"{changed} cleaning rule(s) updated.")
            self._show_step_summary(self._current_step)
        else:
            self.app.notify("No changes made.")

    @work
    async def action_set_text_formats(self) -> None:
        from syntrive.workflow.engine import WorkflowStep

        if self._current_step != WorkflowStep.TRANSCRIPT_TEXT:
            return
        if self._confirm_running:
            return

        opts = self._engine.get_text_extraction_options()
        result = await self.app.push_screen_wait(
            TextExtractionOptionsScreen(opts["formats"], opts["primary"])
        )
        if result is None:
            return

        for name, enabled in result["formats"].items():
            self._engine.set_text_extraction_format(name, enabled)
        self._engine.set_text_extraction_primary(result["primary"])

        self.app.notify(
            f"Formats updated -- primary: {result['primary']}"
        )
        self._show_step_summary(self._current_step)

    @work
    async def action_set_transcript_path(self) -> None:
        from syntrive.workflow.engine import WorkflowStep, TRANSCRIPT_PATH_LABELS

        if self._current_step != WorkflowStep.TRANSCRIPT_REVIEW:
            return
        if self._confirm_running:
            return

        current = self._engine.get_transcript_path_selection()
        value_by_label = {label: name for name, label in TRANSCRIPT_PATH_LABELS.items()}
        current_label = TRANSCRIPT_PATH_LABELS.get(current, current)

        result = await self.app.push_screen_wait(
            ChoiceScreen(
                title=f"Select transcript_path format (current: {current_label})",
                choices=list(TRANSCRIPT_PATH_LABELS.values()),
            )
        )
        if result is None:
            return

        selection = value_by_label.get(result)
        if selection is None:
            return

        self._engine.set_transcript_path_selection(selection)
        self.app.notify(f"transcript_path selection set to: {selection}")
        self._show_step_summary(self._current_step)

    def action_edit_tts_params(self) -> None:
        from syntrive.workflow.engine import WorkflowStep

        if self._current_step != WorkflowStep.TTS_CONFIG:
            return
        if self._confirm_running:
            return

        self._engine.ensure_tts_config()
        self._engine.ensure_tts_voices()


        from syntrive.tui.tts_editor_gui import run_tts_editor_gui

        with self.app.suspend():
            saved = run_tts_editor_gui(job_id=self._job.id, db_path=self._db_path)
        self._show_step_summary(self._current_step)
        if saved:
            self.app.notify("TTS configuration saved.")

    @work(thread=True)
    def _run_step_in_thread(self) -> None:
        from syntrive.workflow import engine as wf

        step_label = wf._STEP_LABELS.get(self._current_step, self._current_step.value)
        self.app.call_from_thread(
            self._set_body,
            f"[cyan]Running: {step_label}...[/cyan]\n\n"
            f"[dim]Watch the Log panel below for progress[/dim]",
        )

        outcome = self._engine.execute_step(self._current_step)

        if outcome.needs_overwrite_confirm:
            choice = self._ask_overwrite_choice_sync(
                outcome.existing_artifact_desc or ""
            )
            logger.info(
                "Overwrite confirm: step=%s choice=%s", self._current_step.value, choice
            )

            if choice == "Overwrite":
                outcome = self._engine.execute_step(
                    self._current_step, force_overwrite=True
                )
            elif choice == "Skip (keep existing)":
                self._engine.record_step(
                    self._current_step,
                    wf.StepAction.SKIPPED,
                    notes="User skipped — existing artifacts kept",
                )
                self._engine.advance_job_step(self._current_step)
                self._advance_to_next_step()
                return
            else:
                logger.info("Overwrite cancelled: step=%s", self._current_step.value)
                self.app.call_from_thread(self._show_step_summary, self._current_step)
                self.app.call_from_thread(self._set_confirm_enabled, True)
                return

        if not outcome.success:
            self.app.call_from_thread(
                self._set_body, f"[red]Failed:[/red] {outcome.error}"
            )
            self.app.call_from_thread(self._set_confirm_enabled, True)
            return

        self._engine.record_step(
            self._current_step,
            wf.StepAction.CONFIRMED,
            artifacts_summary=outcome.artifacts,
        )

        art_lines = "\n".join(
            f"  • {k}: {v}" for k, v in (outcome.artifacts or {}).items()
        )

        if self._current_step == wf.WorkflowStep.TRANSCRIPT_CLEAN:
            self.app.call_from_thread(
                self._set_body,
                f"[green]Cleaning complete[/green]\n\n"
                f"[bold]Outputs:[/bold]\n{art_lines}\n\n"
                f"{markup_escape(outcome.notes or '')}\n\n"
                f"[dim]Press Y to run again with updated rules · "
                f"N to preview the next step · R to manage rules[/dim]",
            )
            self.app.call_from_thread(self._set_confirm_enabled, True)
            return

        self._engine.advance_job_step(self._current_step)

        result_text = (
            f"[green]Complete[/green]\n\n"
            f"[bold]Outputs:[/bold]\n{art_lines}\n\n"
            f"{markup_escape(outcome.notes or '')}"
        )

        transitions = wf._TRANSITIONS.get(self._current_step, [])
        real_transitions = [s for s in transitions if s != wf.WorkflowStep.DONE]

        if real_transitions:
            options = [wf._STEP_LABELS[s] for s in real_transitions]
            labels_text = "\n".join(f"  [{i+1}] {o}" for i, o in enumerate(options))
            self.app.call_from_thread(
                self._set_body,
                result_text + f"\n\n[bold]Next steps:[/bold]\n{labels_text}",
            )
            next_step = real_transitions[0]
            self._current_step = next_step
            self.app.call_from_thread(self._show_step_summary, next_step)
            self.app.call_from_thread(self._set_confirm_enabled, True)
        else:
            self.app.call_from_thread(
                self._set_body,
                result_text + "\n\n[bold green]All steps complete![/bold green]",
            )
            self.app.call_from_thread(self._show_done_state)

    def _advance_to_next_step(self) -> None:
        from syntrive.workflow import engine as wf

        transitions = wf._TRANSITIONS.get(self._current_step, [])
        real_transitions = [s for s in transitions if s != wf.WorkflowStep.DONE]

        if real_transitions:
            next_step = real_transitions[0]
            self._current_step = next_step
            self.app.call_from_thread(self._show_step_summary, next_step)
            self.app.call_from_thread(self._set_confirm_enabled, True)
        else:
            self.app.call_from_thread(self._show_done_state)

    def _set_body(self, text: str) -> None:
        self.query_one("#wfrun-body", Static).update(text)

    def _set_confirm_enabled(self, enabled: bool) -> None:
        self._confirm_running = not enabled
        self.refresh_bindings()
