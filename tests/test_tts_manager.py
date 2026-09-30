from __future__ import annotations

import pytest

from syntrive.adapters.tts.engines.cosyvoice import CosyVoiceEngineAdapter
from syntrive.adapters.tts.engines.f5tts import F5TTSEngineAdapter
from syntrive.adapters.tts.engines.indextts import IndexTTSEngineAdapter
from syntrive.adapters.tts.engines.qwen3tts import Qwen3TTSEngineAdapter
from syntrive.adapters.tts.engines.voxcpm import VoxCPMEngineAdapter
from syntrive.adapters.tts.engines.xtts import XTTSEngineAdapter
from syntrive.adapters.tts.manager import UnknownEngineError, get_engine


class TestGetEngine:
    @pytest.mark.parametrize(
        "engine_name,expected_class",
        [
            ("xtts", XTTSEngineAdapter),
            ("cosyvoice", CosyVoiceEngineAdapter),
            ("indextts", IndexTTSEngineAdapter),
            ("f5tts", F5TTSEngineAdapter),
            ("qwen3tts", Qwen3TTSEngineAdapter),
            ("voxcpm", VoxCPMEngineAdapter),
        ],
    )
    def test_returns_the_matching_adapter_class(self, engine_name, expected_class):
        adapter = get_engine(engine_name)

        assert isinstance(adapter, expected_class)
        assert adapter.engine_name == engine_name

    def test_unknown_engine_raises_with_the_available_list(self):
        with pytest.raises(UnknownEngineError, match="bogus"):
            get_engine("bogus")

    @pytest.mark.parametrize("not_ready_engine", ["bark"])
    def test_removed_engines_are_not_registered(self, not_ready_engine):
        with pytest.raises(UnknownEngineError):
            get_engine(not_ready_engine)

    def test_options_are_forwarded_to_the_adapter(self):
        adapter = get_engine("xtts", options={"device": "cpu"})

        assert adapter.options == {"device": "cpu"}
