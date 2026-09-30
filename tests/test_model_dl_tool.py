from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "tools" / "model_dl.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_model_dl_tool", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_list_shows_every_engine_and_a_download_marker(capsys):
    tool = _load_tool()
    rc = tool.list_models()
    out = capsys.readouterr().out
    assert rc == 0
    for engine in ("xtts", "cosyvoice", "indextts", "voxcpm", "f5tts", "qwen3tts"):
        assert f"{engine}:" in out
    assert "[x]" in out or "[ ]" in out
    assert "cache dir:" in out


def test_unknown_model_id_exits_nonzero_without_touching_the_network(capsys):
    tool = _load_tool()
    rc = tool.download("cosyvoice", "no-such-model")
    out = capsys.readouterr().out
    assert rc == 2
    assert "model_dl_failed" in out


def test_unknown_engine_exits_nonzero(capsys):
    tool = _load_tool()
    assert tool.download("not-an-engine", "whatever") == 2


def test_cli_requires_both_positionals_or_list():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "--list" in (result.stderr + result.stdout)


def test_cli_list_flag_runs():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--list"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "qwen3tts:" in result.stdout
