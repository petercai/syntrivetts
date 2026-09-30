from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from syntrive.adapters.htmlclean.rules.annotation_rule import AnnotationRemovalRule
from syntrive.adapters.htmlclean.rules.base import CleaningStats
from syntrive.adapters.htmlclean.rules.empty_rule import EmptyElementRule
from syntrive.adapters.htmlclean.rules.english_rule import EnglishBlockRule
from syntrive.adapters.htmlclean.rules.hard_newline_rule import HardNewlineToBrRule
from syntrive.adapters.htmlclean.rules.image_caption_rule import ImageCaptionRule
from syntrive.adapters.htmlclean.rules.inline_english_rule import InlineEnglishRule
from syntrive.adapters.htmlclean.rules.span_merge_rule import SpanMergeRule
from syntrive.adapters.htmlclean.rules.table_rule import TableSummaryRule
from syntrive.adapters.htmlclean.rules.whitespace_rule import WhitespaceRule

if TYPE_CHECKING:
    from syntrive.adapters.html.para_id_mapper import NumeralRule
    from syntrive.adapters.htmlclean.rules.base import HtmlRule

logger = logging.getLogger(__name__)


def _build_rule_chain(language: str, numeral_rules: "list[NumeralRule] | None" = None) -> tuple:
    chain: list = [
        AnnotationRemovalRule(),
        SpanMergeRule(),
        TableSummaryRule(),
        InlineEnglishRule(target_language=language),
        EnglishBlockRule(target_language=language),
        ImageCaptionRule(),
    ]
    if numeral_rules:
        from syntrive.adapters.htmlclean.rules.numeral_rule import NumeralConversionRule
        chain.append(NumeralConversionRule(numeral_rules))
    chain.extend([WhitespaceRule(), HardNewlineToBrRule(), EmptyElementRule()])
    return tuple(chain)


class HtmlCleaningEngine:
    def __init__(
        self,
        db_path: Path | None = None,
        job_id: int | None = None,
    ) -> None:
        self._db_path = db_path
        self._job_id = job_id

    def _resolve_book_language(self) -> str:
        if self._db_path is None or self._job_id is None:
            return "zh"
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job
            from syntrive.adapters.epub.lang import normalize_language

            with get_db_session(self._db_path) as db:
                job = db.query(Job).filter_by(id=self._job_id).first()
                if job and job.book and job.book.language:
                    normalized = normalize_language(job.book.language)
                    if normalized:
                        return normalized
        except Exception as exc:
            logger.debug("HtmlCleaningEngine: could not resolve book language: %s", exc)
        logger.warning(
            "HtmlCleaningEngine: book language not found for job=%s, defaulting to 'zh'",
            self._job_id,
        )
        return "zh"

    def run(
        self,
        raw_html: str,
        language: str = "en",
        numeral_rules: "list[NumeralRule] | None" = None,
    ) -> tuple[str, CleaningStats]:
        soup = BeautifulSoup(raw_html, "html.parser")
        stats = CleaningStats(raw_chars=len(soup.get_text()))

        rule_chain = self._load_rule_chain(language, numeral_rules)
        for rule in rule_chain:
            soup, chars_removed = rule.apply(soup)
            if chars_removed:
                stats.record(rule.name, chars_removed)
                logger.debug(
                    "HtmlCleaningEngine: rule=%s chars_removed=%d",
                    rule.name, chars_removed,
                )

        stats.cleaned_chars = len(soup.get_text())
        logger.info(
            "HtmlCleaningEngine: lang=%s raw=%d cleaned=%d ratio=%.1f%%",
            language, stats.raw_chars, stats.cleaned_chars,
            stats.deletion_ratio * 100,
        )
        return str(soup), stats

    def _load_rule_chain(
        self,
        language: str,
        numeral_rules: "list[NumeralRule] | None",
    ) -> tuple:
        if not self._db_path or not self._job_id:
            logger.debug(
                "HtmlCleaningEngine: using hardcoded rule chain (no db_path/job_id)"
            )
            return _build_rule_chain(language, numeral_rules)

        try:
            return self._load_from_db(language, numeral_rules)
        except Exception as exc:
            logger.warning(
                "HtmlCleaningEngine: DB rule load failed (%s) — fallback to hardcoded chain",
                exc,
            )
            return _build_rule_chain(language, numeral_rules)

    def _load_from_db(
        self,
        language: str,
        numeral_rules: "list[NumeralRule] | None",
    ) -> tuple:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import CleaningRule, JobCleaningRuleConfig

        with get_db_session(self._db_path) as db:
            rules = (
                db.query(CleaningRule)
                .order_by(CleaningRule.sort_order)
                .all()
            )
            skip_map: dict[str, bool] = {
                cfg.rule_name: cfg.enabled
                for cfg in db.query(JobCleaningRuleConfig)
                .filter_by(job_id=self._job_id)
                .all()
            }
            rule_defs = [
                (r.name, r.rule_class, r.sort_order, r.enabled_by_default)
                for r in rules
            ]

        effective_language = language or self._resolve_book_language()

        chain: list = []
        for name, rule_class, sort_order, enabled_default in rule_defs:
            is_enabled = skip_map.get(name, enabled_default)
            if not is_enabled:
                logger.info(
                    "HtmlCleaningEngine: rule=%s sort_order=%d SKIPPED (job=%d override)",
                    name, sort_order, self._job_id,
                )
                continue
            instance = _instantiate_rule(name, rule_class, effective_language, numeral_rules)
            if instance is not None:
                chain.append(instance)
                logger.debug(
                    "HtmlCleaningEngine: rule=%s sort_order=%d ACTIVE",
                    name, sort_order,
                )
            else:
                logger.debug(
                    "HtmlCleaningEngine: rule=%s skipped (instantiation returned None)",
                    name,
                )

        logger.info(
            "HtmlCleaningEngine: loaded %d active rules from DB for job=%d",
            len(chain), self._job_id,
        )
        return tuple(chain)


def _instantiate_rule(
    name: str,
    rule_class: str,
    language: str,
    numeral_rules: "list[NumeralRule] | None",
) -> "HtmlRule | None":
    try:
        module_path, class_name = rule_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as exc:
        logger.warning(
            "HtmlCleaningEngine: could not import rule class %r for rule=%s: %s — skipping",
            rule_class, name, exc,
        )
        return None

    if name in ("english_block", "inline_english"):
        return cls(target_language=language)

    if name == "numeral_conversion":
        if not numeral_rules:
            logger.warning(
                "HtmlCleaningEngine: numeral_conversion enabled but numeral_rules is "
                "empty — running as no-op. Add numeral_rules.yaml to activate conversions."
            )
            return cls([])
        return cls(numeral_rules)

    try:
        return cls()
    except Exception as exc:
        logger.warning(
            "HtmlCleaningEngine: could not instantiate rule=%s class=%r: %s — skipping",
            name, rule_class, exc,
        )
        return None
