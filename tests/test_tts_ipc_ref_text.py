from __future__ import annotations

from pathlib import Path

import pytest

from syntrive.adapters.tts.ipc import SubprocessTTSEngineAdapter


class TestReadRefText:
    def test_returns_none_when_sibling_txt_missing(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        assert SubprocessTTSEngineAdapter._read_ref_text(str(wav)) is None

    def test_reads_and_strips_sibling_txt(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        (tmp_path / "yunjian.txt").write_text("  你好，欢迎收听。  \n", encoding="utf-8")

        assert SubprocessTTSEngineAdapter._read_ref_text(str(wav)) == "你好，欢迎收听。"

    def test_blank_sibling_txt_is_treated_as_missing(self, tmp_path: Path):
        wav = tmp_path / "yunjian.wav"
        (tmp_path / "yunjian.txt").write_text("   \n", encoding="utf-8")

        assert SubprocessTTSEngineAdapter._read_ref_text(str(wav)) is None


@pytest.mark.parametrize("engine_name", ["f5tts", "cosyvoice"])
def test_required_reference_text_fails_before_starting_a_worker(tmp_path: Path, monkeypatch, engine_name: str):
    voice = tmp_path / "reference.wav"
    adapter = object.__new__(SubprocessTTSEngineAdapter)
    adapter.engine_name = engine_name
    monkeypatch.setattr(adapter, "ensure_worker", lambda: pytest.fail("worker must not start"))

    assert adapter.synthesize("A real sentence.", voice=str(voice)) is None
