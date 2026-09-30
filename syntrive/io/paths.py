import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def to_project_relative(abs_path: Path, root: Optional[Path] = None) -> str:
    resolved = Path(abs_path).resolve()
    try:
        return resolved.relative_to(root or PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_project_path(stored: str, root: Optional[Path] = None) -> Path:
    candidate = Path(stored)
    return candidate if candidate.is_absolute() else (root or PROJECT_ROOT) / candidate

def is_portable_stored_path(stored: str) -> bool:
    return not (PurePosixPath(stored).is_absolute() or PureWindowsPath(stored).is_absolute())

_FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F(), ]')


def sanitize_filename(name: str, replacement: str = "_") -> str:
    sanitized = name.replace("&", "And")
    sanitized = re.sub(r"\s+", replacement, sanitized)
    sanitized = _FORBIDDEN_CHARS.sub(replacement, sanitized)
    return sanitized.strip(replacement)


class PathResolver:
    def __init__(self, base_dir, temp_dir=None):
        ...

    def chapter_dir(self, chapter_index):
        ...

    def segment_audio_path(self, chapter_index, segment_index, audio_format):
        ...

    def chapter_audio_path(self, chapter_index, audio_format):
        ...

    def subtitle_path(self, chapter_index, subtitle_format):
        ...
