from __future__ import annotations

import pytest

from syntrive.adapters.tts import engine_capabilities as ec


class TestFormatProvenanceTag:
    @pytest.mark.parametrize(
        "engine_name, short_code",
        [
            ("cosyvoice", "cosy"),
            ("indextts", "idx"),
            ("qwen3tts", "qwen"),
            ("voxcpm", "vox"),
        ],
    )
    def test_mapped_engines_use_their_short_code(self, engine_name, short_code):
        assert ec.format_provenance_tag(engine_name, 1.0) == f"syntrive ({short_code}-1.0)"

    def test_unmapped_engine_falls_back_to_its_full_name(self):
        assert ec.format_provenance_tag("f5tts", 1.0) == "syntrive (f5tts-1.0)"
        assert ec.format_provenance_tag("xtts", 1.0) == "syntrive (xtts-1.0)"

    def test_speed_none_defaults_to_1_0(self):
        assert ec.format_provenance_tag("cosyvoice", None) == "syntrive (cosy-1.0)"
        assert ec.format_provenance_tag("cosyvoice") == "syntrive (cosy-1.0)"

    def test_whole_tenth_speed_keeps_exactly_one_decimal(self):
        assert ec.format_provenance_tag("qwen3tts", 0.9) == "syntrive (qwen-0.9)"
        assert ec.format_provenance_tag("qwen3tts", 2.0) == "syntrive (qwen-2.0)"

    def test_finer_grained_speed_keeps_its_precision(self):
        assert ec.format_provenance_tag("indextts", 0.95) == "syntrive (index-0.95)"
        assert ec.format_provenance_tag("indextts", 1.15) == "syntrive (index-1.15)"
