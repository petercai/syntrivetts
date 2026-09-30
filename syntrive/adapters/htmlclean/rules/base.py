from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from bs4 import BeautifulSoup


@dataclass
class CleaningStats:
    raw_chars: int = 0
    cleaned_chars: int = 0
    removals_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def deletion_ratio(self) -> float:
        if self.raw_chars == 0:
            return 0.0
        return max(0.0, (self.raw_chars - self.cleaned_chars) / self.raw_chars)

    def record(self, rule_name: str, chars_removed: int) -> None:
        self.removals_by_rule[rule_name] = (
            self.removals_by_rule.get(rule_name, 0) + chars_removed
        )


@runtime_checkable
class HtmlRule(Protocol):
    @property
    def name(self) -> str:
        ...

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        ...
