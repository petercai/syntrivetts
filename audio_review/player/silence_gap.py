from __future__ import annotations

from typing import List, Tuple

from syntrive.adapters.muxing.chapter_assembler import plan_segments_and_silence


def plan_chapter_silence(
    transcript_content: str, expected_sentence_paths: List[str]
) -> Tuple[List[str], List[float]]:
    return plan_segments_and_silence(transcript_content, expected_sentence_paths)
