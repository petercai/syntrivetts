from __future__ import annotations


def applies_split_minutes(output_split: str) -> bool:
    return output_split == "by-duration"
