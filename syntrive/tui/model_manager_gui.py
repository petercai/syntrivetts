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

from syntrive.adapters.tts import env_provisioner, model_catalog, model_downloader  # noqa: E402
from syntrive.tui import tk_widgets  # noqa: E402

from _shared import catalog as _catalog  # noqa: E402
from _shared import manifest as _manifest  # noqa: E402
from _shared.runtime_env import TTS_CACHE_DIR  # noqa: E402

logger = logging.getLogger(__name__)

from syntrive.services.model_library import (  # noqa: E402,F401
    LOW_SPACE_BYTES,
    ActionQueue as _ActionQueue,
    RowModel,
    Task as _Task,
    build_resource_row,
    build_row_model,
    disk_free_bytes,
    format_bytes,
    format_revision,
    load_compute_device,
    projected_free_after,
    save_compute_device,
    scan_finetuned_xtts,
    total_cache_bytes,
)

class ModelManagerGui:
    def __init__(self) -> None:
        self.root = None
        self._log_pane = None
        self._queue: Optional[_ActionQueue] = None
        self._groups: dict = {}
        self._device_var = None
        self._cache_label = None
        self._test_hook: Optional[str] = os.environ.get("SYNTRIVE_MODEL_MANAGER_TEST_AUTOCLICK") or None

    def run(self) -> int:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("TTS Model Manager")
        self.root.geometry("980x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        tk_widgets.apply_material_style(self.root)
        self._build_ui()
        tk_widgets.bring_to_front(self.root)
        self._queue = _ActionQueue(self._log, lambda cb: self.root.after(0, cb))
        self._refresh_all()

        detected = env_provisioner.detect_compute_device()
        self._device_var.set(load_compute_device(detected))
        self._log(f"model_manager_start: engines={list(model_catalog.engines())} "
                  f"detected_device={detected} cache={TTS_CACHE_DIR}")

        if self._test_hook:
            self.root.after(400, self._run_test_hook)
        self.root.mainloop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        return 0

    def _run_test_hook(self) -> None:
        hook = self._test_hook or ""
        self._log(f"test_hook: {hook}")
        if hook.startswith("files:"):
            _, engine, model_id = hook.split(":", 2)
            self._toggle_files(engine, model_id)
            self.root.after(300, self._on_close)
        else:
            self._on_close()

    def _on_close(self) -> None:
        if self.root is not None:
            self.root.quit()

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        scroller = tk_widgets.ScrollFrame(self.root)
        scroller.pack(fill="both", expand=True)
        body = scroller.body

        raw = _catalog.load_raw()
        for engine in model_catalog.engines():
            meta = _catalog.engine_meta(raw, engine)
            self._groups[engine] = _EngineGroup(self, body, engine, meta, raw)

        footer = tk.Frame(self.root)
        footer.pack(fill="x", padx=8, pady=(4, 4))
        self._cache_label = tk.Label(footer, text="Total cache: …", anchor="w")
        self._cache_label.pack(side="left")
        tk.Button(footer, text="Close", command=self._on_close).pack(side="right")
        tk.Label(footer, text="Compute device:").pack(side="right", padx=(0, 4))
        self._device_var = tk.StringVar(value="cpu")
        self._device_var.trace_add("write", lambda *_: save_compute_device(self._device_var.get()))
        ttk.OptionMenu(footer, self._device_var, "cpu", *env_provisioner.VALID_DEVICES).pack(side="right")

        self._log_pane = tk_widgets.LogPane(self.root, self.root, height=9)
        self._log_pane.pack(fill="both", expand=False, padx=8, pady=(0, 8))

    def _log(self, message: str) -> None:
        if self._log_pane is not None:
            self._log_pane.append(message)
        logger.info("%s", message)

    def device(self) -> str:
        return self._device_var.get() if self._device_var is not None else "cpu"

    def _submit(self, key: str, name: str, fn, on_done) -> None:
        if self._queue is None:
            return
        if self._queue.submit(key, _Task(name, fn, on_done)):
            self._refresh_all()

    def _refresh_all(self) -> None:
        for group in self._groups.values():
            group.refresh()
        if self._cache_label is not None:
            self._cache_label.config(text=f"Total cache: {format_bytes(total_cache_bytes())}")

    def download_model(self, engine: str, model_id: str, approx_gb: float) -> None:
        from tkinter import messagebox

        free = disk_free_bytes(TTS_CACHE_DIR)
        after = projected_free_after(free, approx_gb)
        proceed = messagebox.askyesno(
            "Download model",
            f"{engine}/{model_id}\n\nApprox size: ~{approx_gb} GB\n"
            f"Free space now: {format_bytes(free)}\n"
            f"Projected free after: {format_bytes(max(after, 0))}"
            + ("\n\n⚠ This will leave very little free space." if after < LOW_SPACE_BYTES else ""),
        )
        self._log(f"disk_check: need~{approx_gb}GB free={format_bytes(free)} "
                  f"proceed={proceed}")
        if not proceed:
            return
        self._submit(
            f"model:{engine}/{model_id}", f"download {engine}/{model_id}",
            lambda log: model_downloader.download(engine, model_id, log=log),
            lambda r: self._after_download(r),
        )

    def update_model(self, engine: str, model_id: str) -> None:
        self._submit(
            f"model:{engine}/{model_id}", f"update {engine}/{model_id}",
            lambda log: model_downloader.download(engine, model_id, log=log),
            lambda r: self._after_download(r),
        )

    def check_update(self, engine: str, model_id: str) -> None:
        self._submit(
            f"check:{engine}/{model_id}", f"check-update {engine}/{model_id}",
            lambda log: model_downloader.check_update(engine, model_id, log=log),
            lambda r: self._after_check_update(r),
        )

    def remove_model(self, engine: str, model_id: str, kind: str) -> None:
        from tkinter import messagebox

        if not messagebox.askyesno("Remove", f"Delete local files for {engine}/{model_id}?"):
            return
        self._submit(
            f"model:{engine}/{model_id}", f"remove {engine}/{model_id}",
            lambda log: model_downloader.remove(engine, model_id, is_resource=(kind == "resource"), log=log),
            lambda _r: self._refresh_all(),
        )

    def download_resource(self, engine: str, resource_id: str) -> None:
        self._submit(
            f"res:{engine}/{resource_id}", f"download {engine}/{resource_id}",
            lambda log: model_downloader.download_resource(engine, resource_id, log=log),
            lambda _r: self._refresh_all(),
        )

    def provision_env(self, engine: str) -> None:
        device = self.device()
        self._submit(
            f"env:{engine}", f"provision {engine} ({device})",
            lambda log: env_provisioner.provision(engine, device, log=log),
            lambda _r: self._refresh_all(),
        )

    def _after_download(self, result) -> None:
        from tkinter import messagebox

        self._refresh_all()
        if isinstance(result, Exception):
            messagebox.showerror("Download failed", str(result))
            return
        if getattr(result, "ok", False):
            free_after = disk_free_bytes(TTS_CACHE_DIR)
            if free_after < LOW_SPACE_BYTES:
                messagebox.showwarning("Low disk space",
                                       f"Only {format_bytes(free_after)} free after this download.")
        elif not getattr(result, "unknown", False):
            messagebox.showerror("Download failed", getattr(result, "error", "see the log pane"))

    def _after_check_update(self, result) -> None:
        from tkinter import messagebox

        if isinstance(result, model_downloader.UpdateStatus):
            group = self._groups.get(result.engine)
            if group is not None:
                group.mark_update_available(result.model_id, result.update_available)
            messagebox.showinfo(
                "Update check",
                f"{result.engine}/{result.model_id}\n\n{result.reason}\n\n"
                f"local: {format_revision(result.local_revision)}\n"
                f"upstream: {format_revision(result.upstream_revision)}",
            )

    def is_busy(self, key: str) -> bool:
        return self._queue.is_busy(key) if self._queue is not None else False

    def _toggle_files(self, engine: str, model_id: str) -> None:
        group = self._groups.get(engine)
        if group is not None:
            group.toggle_files(model_id)


class _EngineGroup:
    def __init__(self, app: ModelManagerGui, parent, engine: str, meta: dict, raw: dict):
        import tkinter as tk

        self._app = app
        self._engine = engine
        self._raw = raw
        self._expanded = False
        self._rows: dict = {}
        self._files_panels: dict = {}
        self._update_flags: dict = {}

        self._outer = tk.Frame(parent, bd=1, relief="solid")
        self._outer.pack(fill="x", padx=6, pady=3)

        self._header = tk.Frame(self._outer)
        self._header.pack(fill="x")
        self._toggle_btn = tk.Label(self._header, text="▸", width=2, cursor="hand2")
        self._toggle_btn.pack(side="left")
        self._toggle_btn.bind("<Button-1>", lambda _e: self.toggle())
        self._title = tk.Label(self._header, text=meta.get("label", engine),
                               font=("TkDefaultFont", 11, "bold"), cursor="hand2")
        self._title.pack(side="left")
        self._title.bind("<Button-1>", lambda _e: self.toggle())
        self._has_venv = model_catalog.has_venv(engine)
        self._env_label = tk.Label(self._header, text="", fg="#666")
        self._env_btn = tk.Button(self._header, text="Set up / update environment",
                                  command=lambda: self._app.provision_env(engine))
        if self._has_venv:
            self._env_label.pack(side="left", padx=8)
            self._env_btn.pack(side="right", padx=4, pady=2)

        self._body = tk.Frame(self._outer)

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._toggle_btn.config(text="▾" if self._expanded else "▸")
        if self._expanded:
            if not self._rows:
                self._build_body()
            self._body.pack(fill="x", padx=4, pady=(0, 4))
        else:
            self._body.pack_forget()

    def _build_body(self) -> None:
        import tkinter as tk

        for spec in model_catalog.models_for(self._engine):
            self._rows[spec.model_id] = _ModelRow(self._app, self._body, spec, kind="model")
        for res in _catalog.resources_for(self._raw, self._engine):
            self._rows[res["id"]] = _ModelRow(self._app, self._body, res, kind="resource")
        if self._engine == "xtts":
            discovered = scan_finetuned_xtts()
            if discovered:
                tk.Label(self._body, text="Discovered fine-tuned checkpoints (read-only):",
                         fg="#666", anchor="w").pack(fill="x", padx=6, pady=(6, 0))
                for name in discovered:
                    tk.Label(self._body, text=f"  ● {name}", anchor="w").pack(fill="x", padx=6)
        self.refresh()

    def refresh(self) -> None:
        if self._has_venv:
            state = env_provisioner.venv_state(self._engine)
            provisioned = state == env_provisioner.STATE_PROVISIONED
            self._env_label.config(
                text="env: ready" if provisioned else "env: not set up",
                fg="#2a7" if provisioned else "#a52",
            )
            self._env_btn.config(state="disabled" if self._app.is_busy(f"env:{self._engine}") else "normal")

        default_expand = False
        for spec in model_catalog.models_for(self._engine):
            entry = _manifest.get(self._engine, spec.model_id)
            downloaded = model_catalog.is_downloaded(spec)
            default_expand = default_expand or downloaded
            row = self._rows.get(spec.model_id)
            if row is not None:
                rm = build_row_model(spec, entry, downloaded)
                row.update(rm, self._update_flags.get(spec.model_id, False))
        for res in _catalog.resources_for(self._raw, self._engine):
            entry = _manifest.get(self._engine, res["id"])
            row = self._rows.get(res["id"])
            if row is not None:
                row.update(build_resource_row(res, self._engine, entry), False)

        if not self._expanded and default_expand and not self._rows:
            self.toggle()

    def mark_update_available(self, model_id: str, available: bool) -> None:
        self._update_flags[model_id] = available
        self.refresh()

    def toggle_files(self, model_id: str) -> None:
        row = self._rows.get(model_id)
        if row is not None:
            row.toggle_files()


class _ModelRow:
    def __init__(self, app: ModelManagerGui, parent, spec_or_res, kind: str):
        import tkinter as tk

        self._app = app
        self._kind = kind
        if kind == "model":
            self._engine, self._model_id = spec_or_res.engine, spec_or_res.model_id
            self._approx_gb = float(spec_or_res.approx_gb or 0)
        else:
            self._engine = None
            self._model_id = spec_or_res["id"]
            self._approx_gb = 0.0

        self._frame = tk.Frame(parent)
        self._frame.pack(fill="x", padx=6, pady=1)
        self._status = tk.Label(self._frame, text="○", width=2)
        self._status.pack(side="left")
        self._name = tk.Label(self._frame, text=self._model_id, width=30, anchor="w")
        self._name.pack(side="left")
        self._detail = tk.Label(self._frame, text="", fg="#666", anchor="w", width=42)
        self._detail.pack(side="left")

        self._btns = {}
        for label, cmd in self._button_specs():
            b = tk.Button(self._frame, text=label, command=cmd, width=8)
            b.pack(side="left", padx=1)
            self._btns[label] = b

        self._files_frame = None
        self._rm: Optional[RowModel] = None

    def _button_specs(self):
        if self._kind == "resource":
            return [
                ("Download", lambda: self._app.download_resource(self._engine_id(), self._model_id)),
                ("Update", lambda: self._app.download_resource(self._engine_id(), self._model_id)),
                ("Remove", lambda: self._app.remove_model(self._engine_id(), self._model_id, "resource")),
            ]
        return [
            ("Download", lambda: self._app.download_model(self._engine, self._model_id, self._approx_gb)),
            ("Check", lambda: self._app.check_update(self._engine, self._model_id)),
            ("Update", lambda: self._app.update_model(self._engine, self._model_id)),
            ("Remove", lambda: self._app.remove_model(self._engine, self._model_id, "model")),
            ("Files", self.toggle_files),
        ]

    def _engine_id(self) -> str:
        return self._engine or (self._rm.engine if self._rm else "")

    def update(self, rm: RowModel, update_available: bool) -> None:
        self._rm = rm
        self._engine = self._engine or rm.engine
        glyph = "↑" if update_available else rm.status_glyph
        self._status.config(text=glyph)
        label = rm.label + ("  (default)" if rm.is_default else "")
        self._name.config(text=label)
        self._detail.config(text=rm.detail_line)

        busy = self._app.is_busy(
            f"{'res' if self._kind == 'resource' else 'model'}:{self._engine_id()}/{self._model_id}"
        )
        dl = "disabled" if (rm.downloaded and rm.state == "complete") or busy else "normal"
        has_local = rm.downloaded or bool(rm.state)
        for name, btn in self._btns.items():
            if name == "Download":
                btn.config(state=dl)
            elif name in ("Update", "Remove", "Check", "Files"):
                btn.config(state="disabled" if (not has_local or busy) else "normal")

    def toggle_files(self) -> None:
        import tkinter as tk

        if self._files_frame is not None:
            self._files_frame.destroy()
            self._files_frame = None
            return
        entries = model_downloader.list_major_files(self._engine, self._model_id)
        self._files_frame = tk.Frame(self._frame.master, bd=1, relief="groove")
        self._files_frame.pack(fill="x", padx=24, pady=(0, 4))
        if not entries:
            tk.Label(self._files_frame, text="(no files on disk yet)", anchor="w").pack(fill="x")
            return
        for name, size, role in entries:
            line = tk.Frame(self._files_frame)
            line.pack(fill="x")
            tk.Label(line, text=name, width=34, anchor="w", font=("TkFixedFont", 9)).pack(side="left")
            tk.Label(line, text=format_bytes(size), width=10, anchor="e", fg="#666").pack(side="left")
            tk.Label(line, text=role or "(purpose not catalogued — likely an auxiliary asset)",
                     anchor="w", wraplength=560, justify="left").pack(side="left", padx=6)


def run_model_manager_gui() -> bool:
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())])
    except OSError as exc:
        logger.error("failed to launch model manager subprocess: %s", exc, exc_info=True)
        return False
    return result.returncode == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s")
    try:
        _exit_code = ModelManagerGui().run()
    except ModuleNotFoundError as exc:
        if exc.name != "_tkinter":
            raise
        logger.error(tk_widgets.tkinter_unavailable_hint())
        _exit_code = 1
    sys.exit(_exit_code)
