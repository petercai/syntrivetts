from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from syntrive.tui import tk_widgets  # noqa: E402

logger = logging.getLogger(__name__)


from syntrive.services import tts_settings  # noqa: E402
from syntrive.services.tts_settings import (  # noqa: E402
    TAB_BASIC,
    TAB_OUTPUT,
    TAB_TUNING,
    TAB_VOICES,
    InputType,
    TtsParam,
)

_TAB_TITLES = tts_settings.SECTION_TITLES
_SECTION_ORDER = tts_settings.SECTION_ORDER
_PARAMS = tts_settings.PARAMS
_OUTPUT_FIELDS = tts_settings.OUTPUT_FIELDS
model_choices_for = tts_settings.model_choices_for
_resolve_finetuned_base_dir = tts_settings.finetuned_base_dir
_scan_finetuned_model_dirs = tts_settings.scan_finetuned_model_dirs

_NO_REFERENCE_VOICE = "(none \u2014 unbound)"

_ROW_BREAK_AFTER: frozenset[str] = frozenset({"device"})

_FULL_WIDTH_FIELDS: frozenset[str] = frozenset()


def layout_tab_fields(tab_params: list[TtsParam]) -> list[list[TtsParam]]:
    rows: list[list[TtsParam]] = []
    current_row: list[TtsParam] = []
    for p in tab_params:
        if p.field in _FULL_WIDTH_FIELDS:
            if current_row:
                rows.append(current_row)
                current_row = []
            rows.append([p])
            continue
        current_row.append(p)
        if len(current_row) == 2 or p.field in _ROW_BREAK_AFTER:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)
    return rows


class TtsParamsGui:
    def __init__(self, job_id: int, db_path: Path):
        self._job_id = job_id
        self._db_path = db_path
        self._widgets: dict[str, object] = {}
        self._fine_tuned_options: tuple[str, ...] = ("internal",)
        self._model_id_by_display: dict[str, str] = {}
        self._model_display_by_id: dict[str, str] = {}
        self._download_btn = None
        self._current_config: dict = {}
        self._section_frames: dict = {}
        self._tts_voices: list[dict] = []
        self._reference_voices: list[dict] = []
        self._ref_display_map: dict[str, int] = {}
        self._ref_id_to_label: dict[int, str] = {}
        self._voice_row_widgets: dict[int, dict] = {}
        self._voice_player = None
        self._playing_voice_row_id: Optional[int] = None
        self.root = None
        self.saved = False
        self._test_after_build_hook = None

    def run(self) -> bool:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title(f"TTS Configuration \u2014 Job #{self._job_id}")
        self.root.geometry("900x820")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._current_config = self._load_config()
        self._load_tts_voices()
        self._build_ui()
        tk_widgets.bring_to_front(self.root)
        if self._test_after_build_hook is not None:
            self._test_after_build_hook()
        self.root.mainloop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        return self.saved

    def _load_config(self) -> dict:
        self._settings = self._read_settings()
        return dict(self._settings.values) if self._settings else {}

    def _read_settings(self):
        try:
            return tts_settings.read_tts_settings(self._db_path, self._job_id)
        except Exception as exc:
            logger.warning("Could not load TTS/Output config: %s", exc)
            return None

    def _build_ui(self) -> None:
        import tkinter as tk

        tk_widgets.apply_material_style(self.root)

        scroller = tk_widgets.ScrollFrame(self.root)
        scroller.pack(fill="both", expand=True)
        scroll_frame = scroller.body

        self._section_frames = {}
        for i, section_id in enumerate(_SECTION_ORDER):
            if i > 0:
                tk_widgets.section_divider(scroll_frame)
            tk_widgets.section_header(scroll_frame, _TAB_TITLES[section_id])
            body = tk.Frame(scroll_frame)
            body.pack(fill="x", padx=4, pady=(0, 4))
            self._section_frames[section_id] = body

        self._populate_field_sections()
        self._build_voices_section()
        self._build_footer()

        engine = str(self._current_config.get("engine") or "cosyvoice")
        output_split = str(self._current_config.get("output_split") or "by-chapter")
        self._apply_gating(engine, output_split)

    def _populate_field_sections(self) -> None:
        for section_id in (TAB_BASIC, TAB_OUTPUT, TAB_TUNING):
            frame = self._section_frames[section_id]
            for child in frame.winfo_children():
                child.destroy()
            section_params = [p for p in _PARAMS if p.tab == section_id]
            for row_i, row_params in enumerate(layout_tab_fields(section_params)):
                if len(row_params) == 1:
                    self._build_field_cell(frame, row_params[0], row_i, 0, columnspan=2)
                else:
                    for col_i, p in enumerate(row_params):
                        self._build_field_cell(frame, p, row_i, col_i)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=1)

    def _build_field_cell(self, parent, param: TtsParam, row: int, col: int, columnspan: int = 1) -> None:
        import tkinter as tk

        cell = tk.Frame(parent)
        cell.grid(row=row, column=col, columnspan=columnspan, sticky="w", padx=6, pady=2)
        tk.Label(cell, text=f"{param.label}:", width=20, anchor="e").pack(side="left", padx=(0, 6))
        current = self._current_config.get(param.field, param.default)
        widget = self._make_widget(cell, param, current)
        widget.pack(side="left")
        self._widgets[param.field] = widget
        if param.field == "model":
            self._build_download_button(cell)

    def _current_engine(self) -> str:
        w = self._widgets.get("engine")
        if w is not None:
            try:
                return str(w.get()) or "cosyvoice"
            except Exception:  # pragma: no cover - widget torn down
                pass
        return str(self._current_config.get("engine") or "cosyvoice")

    def _repopulate_model_combo(self, engine: str) -> None:
        combo = self._widgets.get("model")
        if combo is None:
            return
        pairs = model_choices_for(engine)
        self._model_id_by_display = {display: mid for mid, display in pairs}
        self._model_display_by_id = {mid: display for mid, display in pairs}
        displays = [display for _mid, display in pairs]
        combo["values"] = displays
        current_id = self._model_id_by_display.get(combo.get())
        if current_id is None:
            current_id = str(self._current_config.get("model") or "")
        selected = self._model_display_by_id.get(current_id) or (displays[0] if displays else "")
        combo.set(selected)
        logger.info(
            "model_combo_repopulated: engine=%s models=%s selected=%r",
            engine, [mid for mid, _ in pairs], selected,
        )

    def _build_download_button(self, cell) -> None:
        import tkinter as tk

        self._download_btn = tk.Button(cell, text="Download", width=10, command=self._on_download_model)
        self._download_btn.pack(side="left", padx=(6, 0))

    def _on_download_model(self) -> None:
        from tkinter import messagebox

        engine = self._current_engine()
        combo = self._widgets.get("model")
        model_id = self._model_id_by_display.get(combo.get() if combo else "", "")
        if not model_id or model_id not in {mid for mid, _ in model_choices_for(engine)}:
            messagebox.showinfo("Nothing to download", "Select a registry model first.")
            return
        from syntrive.adapters.tts import model_catalog
        spec = model_catalog.resolve(engine, model_id)
        if spec is None or spec.model_id != model_id:
            messagebox.showinfo("Nothing to download", f"{model_id!r} is not a downloadable registry model.")
            return

        self._download_btn.config(text="Downloading…", state="disabled")
        self.root.update_idletasks()
        rc = run_model_download(engine, model_id)
        self._download_btn.config(text="Download", state="normal")
        self._repopulate_model_combo(engine)
        if rc == 0:
            messagebox.showinfo("Download complete", f"{spec.label} is ready.\n({spec.local_path()})")
        else:
            messagebox.showerror("Download failed", f"model_dl.py exited {rc} — see the terminal for details.")

    def _make_widget(self, parent, param: TtsParam, current):
        import tkinter as tk
        from tkinter import ttk

        if param.field == "model":
            engine = str(self._current_config.get("engine") or "cosyvoice")
            pairs = model_choices_for(engine)
            self._model_id_by_display = {display: mid for mid, display in pairs}
            self._model_display_by_id = {mid: display for mid, display in pairs}
            displays = [display for _mid, display in pairs]
            current_id = str(current) if current else ""
            selected = self._model_display_by_id.get(current_id) or (displays[0] if displays else "")
            combo = ttk.Combobox(parent, values=displays, state="readonly", width=30)
            combo.set(selected)
            return combo

        if param.input_type == InputType.CHECKBOX:
            var = tk.BooleanVar(value=bool(current) if current is not None else bool(param.default))
            cb = ttk.Checkbutton(parent, variable=var)
            cb.var = var
            return cb

        if param.input_type == InputType.SELECT:
            choices = list(param.choices or [])
            selected = str(current) if current is not None else (choices[0] if choices else "")
            combo = ttk.Combobox(parent, values=choices, state="readonly", width=16)
            combo.set(selected)
            if param.field == "engine":
                combo.bind("<<ComboboxSelected>>", lambda e: self._on_engine_changed())
            elif param.field == "output_split":
                combo.bind("<<ComboboxSelected>>", lambda e: self._on_split_changed())
            return combo

        if param.input_type == InputType.NUMERIC:
            try:
                val = float(current) if current is not None else float(param.default or 0)
            except (TypeError, ValueError):
                val = float(param.default or 0)
            return tk_widgets.NumericSpinner(
                parent, value=val, min_val=param.min_val, max_val=param.max_val,
                step=param.step, default=float(param.default or 0),
            )

        entry = ttk.Entry(parent, width=32)
        entry.insert(0, str(current) if current is not None else "")
        return entry

    def _build_footer(self) -> None:
        import tkinter as tk

        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)
        tk.Button(bar, text="Reset Defaults", command=self._on_reset).pack(side="left")
        tk.Button(bar, text="Manage models…", command=self._on_manage_models).pack(side="left", padx=8)
        tk.Button(bar, text="Save", command=self._on_save).pack(side="right", padx=4)
        tk.Button(bar, text="Cancel", command=self._on_cancel).pack(side="right", padx=4)

    def _on_manage_models(self) -> None:
        from syntrive.tui.model_manager_gui import run_model_manager_gui

        run_model_manager_gui()
        self._repopulate_model_combo(self._current_engine())

    def _load_tts_voices(self) -> None:
        settings = getattr(self, "_settings", None) or self._read_settings()
        self._tts_voices = [
            {"id": v.id, "name": v.name, "voice_id": v.voice_id,
             "reference_voice_id": v.reference_voice_id, "excluded": v.excluded}
            for v in (settings.voices if settings else ())
        ]
        self._reference_voices = [
            {"id": r.id, "path": r.path, "name": r.name, "gender": r.gender, "language": r.language}
            for r in (settings.references if settings else ())
        ]

    def _build_reference_display_map(self) -> dict[str, int]:
        return {
            f'{r["name"]} [{r["gender"]}/{r["language"]}] (#{r["id"]})': r["id"]
            for r in self._reference_voices
        }

    def _build_voices_section(self) -> None:
        import tkinter as tk

        frame = self._section_frames[TAB_VOICES]
        for child in frame.winfo_children():
            child.destroy()

        self._ref_display_map = self._build_reference_display_map()
        self._ref_id_to_label = {v: k for k, v in self._ref_display_map.items()}
        ref_options = (_NO_REFERENCE_VOICE,) + tuple(sorted(self._ref_display_map))

        header = tk.Frame(frame)
        header.pack(fill="x", pady=(0, 4))
        for text, width in (
            ("Role", 18), ("Voice ID", 8), ("Reference Voice", 40), ("Preview", 8), ("Excluded", 8),
        ):
            tk.Label(header, text=text, width=width, anchor="w", font=("TkDefaultFont", 9, "bold")).pack(
                side="left", padx=2
            )

        self._voice_row_widgets = {}
        if not self._tts_voices:
            tk.Label(
                frame,
                text=(
                    "No voices yet for this job -- open the TTS_CONFIG step "
                    "(press Y or E from the workflow screen) to initialize the "
                    "narrator/role rows."
                ),
            ).pack(anchor="w", pady=8)
            return

        for row in self._tts_voices:
            self._build_voice_row(frame, row, ref_options)

    def _build_voice_row(self, parent, row: dict, ref_options: tuple[str, ...]) -> None:
        import tkinter as tk
        from tkinter import ttk

        row_frame = tk.Frame(parent)
        row_frame.pack(fill="x", pady=2)

        display_name = row["name"] + (" (narrator)" if row["voice_id"] == 0 else "")
        tk.Label(row_frame, text=display_name, width=18, anchor="w").pack(side="left", padx=2)
        tk.Label(row_frame, text=str(row["voice_id"]), width=8, anchor="w").pack(side="left", padx=2)

        current_label = self._ref_id_to_label.get(row["reference_voice_id"], _NO_REFERENCE_VOICE)
        combo = tk_widgets.FilterCombo(row_frame, ref_options, current_label, width=40)
        combo.pack(side="left", padx=2)
        combo.bind(
            "<<ComboboxSelected>>",
            lambda _e, rid=row["id"]: self._on_voice_row_selection_changed(rid),
        )

        play_btn = tk.Button(
            row_frame, text="Play", width=8,
            command=lambda rid=row["id"]: self._on_voice_row_play_click(rid),
        )
        play_btn.pack(side="left", padx=2)

        excluded_var = tk.BooleanVar(value=row["excluded"])
        ttk.Checkbutton(row_frame, variable=excluded_var).pack(side="left", padx=2)

        self._voice_row_widgets[row["id"]] = {"combo": combo, "play_btn": play_btn, "excluded_var": excluded_var}

    def _on_voice_row_selection_changed(self, _tts_voice_id: int) -> None:
        self._stop_voice_play()

    def _on_voice_row_play_click(self, tts_voice_id: int) -> None:
        import soundfile as sf

        widgets = self._voice_row_widgets.get(tts_voice_id)
        if widgets is None:
            return
        if self._playing_voice_row_id == tts_voice_id and self._voice_player is not None:
            self._stop_voice_play()
            return
        self._stop_voice_play()

        label = widgets["combo"].get().strip()
        reference_voice_id = self._ref_display_map.get(label)
        if reference_voice_id is None:
            from tkinter import messagebox
            messagebox.showinfo("No reference voice", "Select a reference voice first.")
            return
        ref = next((r for r in self._reference_voices if r["id"] == reference_voice_id), None)
        if ref is None:
            return

        tools_dir = _REPO_ROOT / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        import tts_import_voice as tiv
        from syntrive.io.paths import resolve_project_path

        abs_path = resolve_project_path(ref["path"])
        if not abs_path.is_file():
            from tkinter import messagebox
            messagebox.showwarning("File not found", f"{abs_path} does not exist on disk.")
            return

        data, samplerate = sf.read(str(abs_path), dtype="float32")
        self._voice_player = tiv._AudioPlayer(data, samplerate)
        self._voice_player.start()
        self._playing_voice_row_id = tts_voice_id
        widgets["play_btn"].config(text="Stop")
        self._poll_voice_playback()

    def _stop_voice_play(self) -> None:
        if self._voice_player is not None:
            self._voice_player.close()
            self._voice_player = None
        if self._playing_voice_row_id is not None:
            widgets = self._voice_row_widgets.get(self._playing_voice_row_id)
            if widgets is not None:
                widgets["play_btn"].config(text="Play")
        self._playing_voice_row_id = None

    def _poll_voice_playback(self) -> None:
        if self._voice_player is not None and self._voice_player.finished:
            self._stop_voice_play()
            return
        if self._voice_player is not None:
            self.root.after(150, self._poll_voice_playback)

    def _collect_tts_voices_values(self) -> list[dict]:
        values = []
        for row in self._tts_voices:
            widgets = self._voice_row_widgets.get(row["id"])
            if widgets is None:
                continue
            label = widgets["combo"].get().strip()
            values.append({
                "id": row["id"],
                "reference_voice_id": self._ref_display_map.get(label),
                "excluded": bool(widgets["excluded_var"].get()),
            })
        return values

    def _apply_gating(self, engine: str, output_split: str) -> None:
        split_widget = self._widgets.get("output_split_minutes")
        if split_widget is not None:
            split_widget.set_enabled(output_split == "by-duration")

    def _on_engine_changed(self) -> None:
        engine = self._widgets["engine"].get()
        output_split = self._widgets["output_split"].get()
        self._repopulate_model_combo(engine)
        self._apply_gating(engine, output_split)

    def _on_split_changed(self) -> None:
        engine = self._widgets["engine"].get()
        output_split = self._widgets["output_split"].get()
        self._apply_gating(engine, output_split)

    def _collect_values(self) -> dict:
        values: dict = {}
        for param in _PARAMS:
            widget = self._widgets.get(param.field)
            if widget is None:
                continue
            if param.input_type == InputType.CHECKBOX:
                values[param.field] = bool(widget.var.get())
            elif param.field == "model":
                values["model"] = self._model_id_by_display.get(widget.get(), "") or None
            elif param.input_type == InputType.FILTER_SELECT:
                raw = widget.get().strip()
                values[param.field] = raw if raw in self._fine_tuned_options else (
                    self._fine_tuned_options[0] if self._fine_tuned_options else ""
                )
            elif param.input_type == InputType.NUMERIC:
                values[param.field] = widget.value
            else:
                values[param.field] = widget.get()

        return values

    def _persist_under_lease(self, values: dict, voice_values) -> bool:
        from tkinter import messagebox

        from syntrive.services.job_lease import JobLeaseConflict

        bindings = [tts_settings.VoiceBinding(v["id"], v["reference_voice_id"], v["excluded"]) for v in voice_values]
        try:
            tts_settings.save_tts_settings(self._db_path, self._job_id, values, bindings, holder_kind="tui")
            return True
        except JobLeaseConflict as exc:
            logger.warning("TTS config save blocked by job lease: job=%d %s", self._job_id, exc)
            messagebox.showerror("Save blocked", str(exc))
        except tts_settings.SettingsError as exc:
            logger.warning("TTS config save refused: job=%d code=%s %s", self._job_id, exc.code, exc)
            messagebox.showerror("Save refused", str(exc))
        except Exception as exc:
            logger.error("Failed to save TTS configuration: %s", exc, exc_info=True)
            messagebox.showerror("Save failed", str(exc))
        return False

    def _on_save(self) -> None:
        self._stop_voice_play()
        values = self._collect_values()
        voice_values = self._collect_tts_voices_values()
        if self._persist_under_lease(values, voice_values):
            self.saved = True
            self.root.quit()

    def _on_cancel(self) -> None:
        self._stop_voice_play()
        self.saved = False
        self.root.quit()

    def _on_reset(self) -> None:
        self._current_config = {p.field: p.default for p in _PARAMS}
        self._widgets = {}
        self._populate_field_sections()
        engine = str(self._current_config.get("engine") or "cosyvoice")
        output_split = str(self._current_config.get("output_split") or "by-chapter")
        self._apply_gating(engine, output_split)


def run_model_download(engine: str, model_id: str) -> int:
    script = _REPO_ROOT / "tools" / "model_dl.py"
    logger.info("model_download_start: engine=%s model=%s script=%s", engine, model_id, script)
    try:
        result = subprocess.run([sys.executable, str(script), engine, model_id])
    except OSError as exc:
        logger.error("model_download_launch_failed: %s", exc, exc_info=True)
        return -1
    logger.info("model_download_finished: engine=%s model=%s rc=%d", engine, model_id, result.returncode)
    return result.returncode


def run_tts_editor_gui(job_id: int, db_path: Path) -> bool:
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(db_path), str(job_id)])
    except OSError as exc:
        logger.error("Failed to launch TTS parameter editor subprocess: %s", exc, exc_info=True)
        return False
    if result.returncode != 0:
        logger.warning("TTS parameter editor subprocess exited with code %d", result.returncode)
    return result.returncode == 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s"
    )
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <db_path> <job_id>", file=sys.stderr)
        sys.exit(2)
    _db_path_arg, _job_id_arg = Path(sys.argv[1]), int(sys.argv[2])
    _gui = TtsParamsGui(job_id=_job_id_arg, db_path=_db_path_arg)
    _autoclick = os.environ.get("SYNTRIVE_TTS_EDITOR_TEST_AUTOCLICK")
    if _autoclick in ("save", "cancel"):
        _handler = _gui._on_save if _autoclick == "save" else _gui._on_cancel
        _gui._test_after_build_hook = lambda: _gui.root.after(300, _handler)
    try:
        _saved = _gui.run()
    except ModuleNotFoundError as exc:
        if exc.name != "_tkinter":
            raise
        logger.error(tk_widgets.tkinter_unavailable_hint())
        sys.exit(1)
    sys.exit(0 if _saved else 1)
