from __future__ import annotations

from pathlib import Path
from typing import Optional

from syntrive.adapters.tts.ipc import SubprocessTTSEngineAdapter


class XTTSEngineAdapter(SubprocessTTSEngineAdapter):
    def __init__(self, options: Optional[dict] = None, engines_root: Optional[Path] = None):
        super().__init__(engine_name="xtts", options=options, engines_root=engines_root)
