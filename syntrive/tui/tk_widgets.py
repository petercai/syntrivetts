from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def tkinter_unavailable_hint() -> str:
    hint = f"tkinter/_tkinter is not available for this Python interpreter ({sys.executable})."
    if sys.platform == "darwin":
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        hint += f" On macOS with Homebrew, install it via: brew install python-tk@{version}"
    elif sys.platform.startswith("linux"):
        hint += " On Debian/Ubuntu, install it via: sudo apt-get install python3-tk"
    return hint


def apply_material_style(root) -> None:
    import tkinter as tk
    from tkinter import ttk

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TButton", padding=(8, 4))
    style.configure("TCheckbutton", padding=(2, 2))
    style.configure("TCombobox", padding=(2, 2))
    style.configure("TEntry", padding=(2, 2))


def section_header(parent, title: str) -> None:
    import tkinter as tk

    tk.Label(parent, text=title, anchor="w", font=("TkDefaultFont", 11, "bold")).pack(
        fill="x", padx=4, pady=(4, 2)
    )


def section_divider(parent) -> None:
    import tkinter as tk

    tk.Frame(parent, height=1, bg="#d0d0d0").pack(fill="x", padx=4, pady=(4, 8))


def bring_to_front(root) -> None:
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    if sys.platform == "darwin":
        try:
            subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to set frontmost of '
                    f"first process whose unix id is {os.getpid()} to true",
                ],
                check=False, capture_output=True, timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("macOS foreground-activation osascript failed: %s", exc)
    root.after(250, lambda: root.attributes("-topmost", False))


class ScrollFrame:
    def __init__(self, parent):
        import tkinter as tk
        from tkinter import ttk

        self.outer = tk.Frame(parent)
        self._canvas = tk.Canvas(self.outer, highlightthickness=0)
        vsb = ttk.Scrollbar(self.outer, orient="vertical", command=self._canvas.yview)
        self.body = tk.Frame(self._canvas)

        self.body.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(window, width=e.width))
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True, padx=6, pady=(6, 0))
        vsb.pack(side="right", fill="y")

        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

    def pack(self, **kwargs) -> None:
        self.outer.pack(**kwargs)

    def _on_wheel(self, event) -> None:
        num = getattr(event, "num", None)
        if num == 5:
            delta = -1
        elif num == 4:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(delta, "units")

    def _bind_wheel(self, _event=None) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._canvas.bind_all("<Button-4>", self._on_wheel)
        self._canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event=None) -> None:
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")


class NumericSpinner:
    def __init__(
        self,
        parent,
        *,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        step: Optional[float] = None,
        default: float = 0.0,
    ):
        import tkinter as tk

        self._min = min_val
        self._max = max_val
        self._step = step
        self._default = default
        self.frame = tk.Frame(parent)
        self.var = tk.StringVar(value=self._fmt(value))
        self._entry = tk.Entry(self.frame, textvariable=self.var, width=10)
        self._entry.pack(side="left")

        self._stepper = tk.Frame(self.frame, width=18)
        self._stepper.pack(side="left", fill="y")
        self._stepper.grid_propagate(False)
        self._stepper.grid_columnconfigure(0, weight=1)
        self._stepper.grid_rowconfigure(0, weight=1)
        self._stepper.grid_rowconfigure(1, weight=0)
        self._stepper.grid_rowconfigure(2, weight=1)

        _btn_kwargs = dict(
            font=("TkDefaultFont", 7), padx=0, pady=0, bd=1, highlightthickness=0,
        )
        self._up = tk.Button(
            self._stepper, text="▲", command=lambda: self._bump(1), **_btn_kwargs
        )
        self._up.grid(row=0, column=0, sticky="nsew")
        tk.Frame(self._stepper, height=1, bg="#d0d0d0").grid(row=1, column=0, sticky="ew")
        self._down = tk.Button(
            self._stepper, text="▼", command=lambda: self._bump(-1), **_btn_kwargs
        )
        self._down.grid(row=2, column=0, sticky="nsew")

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def _bump(self, direction: int) -> None:
        try:
            v = float(self.var.get())
        except ValueError:
            v = float(self._default)
        v = round(v + (self._step or 1.0) * direction, 8)
        if self._min is not None:
            v = max(self._min, v)
        if self._max is not None:
            v = min(self._max, v)
        self.var.set(self._fmt(v))

    @property
    def value(self) -> float:
        try:
            return float(self.var.get())
        except ValueError:
            return float(self._default)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._entry.configure(state=state)
        self._up.configure(state=state)
        self._down.configure(state=state)

    @staticmethod
    def _fmt(v: float) -> str:
        if float(v) == int(v) and abs(v) < 1_000_000:
            return str(int(v))
        return f"{v:.4g}"


class FilterCombo:
    def __init__(self, parent, options: tuple, value: str, width: int = 28):
        from tkinter import ttk

        self._all_options = tuple(options)
        self.combo = ttk.Combobox(parent, values=list(options), width=width)
        self.combo.set(value)
        self.combo.bind("<KeyRelease>", self._on_key_release)

    def pack(self, **kwargs) -> None:
        self.combo.pack(**kwargs)

    def get(self) -> str:
        return self.combo.get()

    def set(self, value: str) -> None:
        self.combo.set(value)

    def configure(self, **kwargs) -> None:
        self.combo.configure(**kwargs)

    def bind(self, *args, **kwargs) -> None:
        self.combo.bind(*args, **kwargs)

    def set_options(self, options: tuple) -> None:
        self._all_options = tuple(options)
        self.combo["values"] = list(options)
        if self.combo.get() not in options:
            old_value = self.combo.get()
            self.combo.set(options[0] if options else "")
            logger.debug(
                "FilterCombo: value %r no longer in new option set, reset to %r",
                old_value, self.combo.get(),
            )

    def _on_key_release(self, event) -> None:
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab"):
            return
        needle = self.combo.get().strip().lower()
        matches = (
            [o for o in self._all_options if needle in o.lower()] if needle else list(self._all_options)
        )
        self.combo["values"] = matches
        if matches:
            try:
                self.combo.tk.call("ttk::combobox::Post", self.combo)
            except Exception as exc:  # pragma: no cover -- Tcl internals, platform-dependent
                logger.debug("Could not force-open combobox popdown: %s", exc)


class LogPane:
    def __init__(self, parent, root, *, height: int = 10):
        import tkinter as tk

        self._root = root
        self.frame = tk.Frame(parent)
        self._text = tk.Text(self.frame, height=height, wrap="none", state="disabled",
                             font=("TkFixedFont", 9), background="#1e1e1e", foreground="#d4d4d4")
        vsb = tk.Scrollbar(self.frame, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=vsb.set)
        self._text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def append(self, line: str) -> None:
        self._root.after(0, self._append_now, line)

    def clear(self) -> None:
        self._root.after(0, self._clear_now)

    def _append_now(self, line: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", line.rstrip("\n") + "\n")
        self._text.see("end")
        self._text.configure(state="disabled")

    def _clear_now(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


__all__ = [
    "apply_material_style",
    "section_header",
    "section_divider",
    "bring_to_front",
    "ScrollFrame",
    "NumericSpinner",
    "FilterCombo",
    "LogPane",
]
