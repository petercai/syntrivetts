from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WriteStatus(str, Enum):
    WRITTEN = "written"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class WriteOutcome:
    path: Path
    status: WriteStatus
    content_sha256: str
    previous_sha256: Optional[str] = None


def read_existing_fingerprint(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("contract_manifest_unreadable: path=%s error=%s -- will overwrite", path, exc)
        return None
    if not isinstance(existing, dict):
        return None
    value = existing.get("content_sha256")
    return value if isinstance(value, str) else None


def write_manifest(path: Path, manifest: dict[str, Any]) -> WriteOutcome:
    new_sha = manifest["content_sha256"]
    previous = read_existing_fingerprint(path)
    if previous == new_sha:
        return WriteOutcome(path=path, status=WriteStatus.UNCHANGED, content_sha256=new_sha, previous_sha256=previous)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return WriteOutcome(path=path, status=WriteStatus.WRITTEN, content_sha256=new_sha, previous_sha256=previous)
