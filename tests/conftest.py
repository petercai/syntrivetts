from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

try:
    import torchaudio  # noqa: F401
except ImportError:
    import sys

    import _torchaudio_standin

    sys.modules["torchaudio"] = _torchaudio_standin
    logger.debug("conftest: torchaudio not installed -- registered soundfile-backed stand-in")

_REAL_ENV_OPTION = "--real-env"
_OFFLINE_OPTION = "--offline"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        _REAL_ENV_OPTION,
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.real_env "
            "(real models, subprocess engines, network downloads)."
        ),
    )
    parser.addoption(
        _OFFLINE_OPTION,
        action="store_true",
        default=False,
        help=(
            "Run the real-env TTS integration tests in offline mode: sets "
            "SYNTRIVE_TTS_OFFLINE=1 for every engine subprocess / in-process "
            "runner, so they must reuse the models/tts/ cache with no download."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_env: real-environment integration test — needs --real-env "
        "(real models, subprocess engines, network).",
    )
    if config.getoption(_OFFLINE_OPTION):
        os.environ["SYNTRIVE_TTS_OFFLINE"] = "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption(_REAL_ENV_OPTION):
        return
    skip_real_env = pytest.mark.skip(
        reason="needs --real-env (real models / subprocess engines / network)"
    )
    for item in items:
        if "real_env" in item.keywords:
            item.add_marker(skip_real_env)


_IT_RESULT_SUFFIXES = frozenset(
    {".flac", ".m4a", ".m4b", ".mp3", ".ogg", ".opus", ".aac", ".wav"}
)
_IT_RAW_WAV_RE = re.compile(r"_raw\.wav$", re.IGNORECASE)
_NODEID_UNSAFE_RUN_RE = re.compile(r"[^0-9A-Za-z_-]+")


def sanitize_nodeid(nodeid: str) -> str:
    file_part, separator, rest = nodeid.partition("::")
    stem = Path(file_part).name
    combined = f"{stem}::{rest}" if separator else stem
    combined = combined.replace("::", "__").replace("[", "__").replace("]", "")
    combined = combined.replace(".", "_")
    return _NODEID_UNSAFE_RUN_RE.sub("_", combined).strip("_")


def _is_integration_test(node: pytest.Item) -> bool:
    if node.get_closest_marker("real_env") is not None:
        return True
    node_path = getattr(node, "path", None) or getattr(node, "fspath", None)
    return node_path is not None and Path(str(node_path)).name.startswith("test_it_")


def _collect_result_files(search_root: Path) -> list[Path]:
    if not search_root.is_dir():
        return []
    return sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _IT_RESULT_SUFFIXES
        and not _IT_RAW_WAV_RE.search(path.name)
    )


def _artifact_name(stem: str, source: Path, index: int, total: int) -> str:
    suffix = source.suffix.lower()
    if total <= 1:
        return f"{stem}{suffix}"
    return f"{stem}__{index:02d}{suffix}"


def _publish_result_files(node: pytest.Item, search_root: Path, dest_dir: Path) -> None:
    results = _collect_result_files(search_root)
    if not results:
        return

    stem = sanitize_nodeid(node.nodeid)
    total = len(results)
    for index, source in enumerate(results, start=1):
        destination = dest_dir / _artifact_name(stem, source, index, total)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            logger.warning(
                "it_artifact_publish_failed test=%s src=%s dest=%s error=%s",
                node.nodeid, source, destination, exc,
            )
            continue
        logger.info(
            "it_artifact_published test=%s src=%s dest=%s bytes=%d",
            node.nodeid, source, destination, destination.stat().st_size,
        )


@pytest.fixture(autouse=True)
def _publish_it_result_artifacts(request: pytest.FixtureRequest):
    node = request.node
    if not _is_integration_test(node):
        yield
        return

    search_root = Path(request.getfixturevalue("tmp_path"))

    yield

    dest_dir = Path(request.config.invocation_params.dir)
    _publish_result_files(node, search_root, dest_dir)
