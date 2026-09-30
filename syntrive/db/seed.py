from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_CLEANING_RULES: list[dict] = [
    {
        "name": "annotation_removal",
        "rule_class": "syntrive.adapters.htmlclean.rules.annotation_rule.AnnotationRemovalRule",
        "sort_order": 5,
        "display_name": "Annotation/Footnote Removal",
        "description": (
            "Remove inline footnote annotation markers (<sup> elements containing "
            "circled numbers or footnote links, e.g. ①) — must run before span_merge"
        ),
        "enabled_by_default": True,
    },
    {
        "name": "footnote_paragraph_removal",
        "rule_class": "syntrive.adapters.htmlclean.rules.footnote_paragraph_rule.FootnoteParagraphRemovalRule",
        "sort_order": 7,
        "display_name": "Footnote Paragraph Removal",
        "description": (
            "Remove free-standing Calibre footnote/endnote paragraphs (translator/"
            "editor notes etc.) identified by their filepos cross-reference "
            "structure — carries no spoken value and has no in-context meaning "
            "for a listener"
        ),
        "enabled_by_default": True,
    },
    {
        "name": "span_merge",
        "rule_class": "syntrive.adapters.htmlclean.rules.span_merge_rule.SpanMergeRule",
        "sort_order": 10,
        "display_name": "Span Merge",
        "description": "Merge adjacent span tags (eliminate typography-induced splits)",
        "enabled_by_default": True,
    },
    {
        "name": "table_summary",
        "rule_class": "syntrive.adapters.htmlclean.rules.table_rule.TableSummaryRule",
        "sort_order": 20,
        "display_name": "Table Summary",
        "description": "Convert table content to plain text summary",
        "enabled_by_default": True,
    },
    {
        "name": "inline_english",
        "rule_class": "syntrive.adapters.htmlclean.rules.inline_english_rule.InlineEnglishRule",
        "sort_order": 22,
        "display_name": "Inline English Removal",
        "description": (
            "Remove English-only parentheticals within Chinese text, e.g. "
            "'纳西姆·塔勒布（ Nassim Taleb）' → '纳西姆·塔勒布'"
        ),
        "enabled_by_default": True,
    },
    {
        "name": "english_block",
        "rule_class": "syntrive.adapters.htmlclean.rules.english_rule.EnglishBlockRule",
        "sort_order": 30,
        "display_name": "English Block Removal",
        "description": "Remove English-language blocks from non-English books (Chinese novels only)",
        "enabled_by_default": True,
    },
    {
        "name": "image_caption",
        "rule_class": "syntrive.adapters.htmlclean.rules.image_caption_rule.ImageCaptionRule",
        "sort_order": 35,
        "display_name": "Image Caption Removal",
        "description": (
            "Remove <p> caption paragraphs co-located with <img> inside a <div> — "
            "image captions carry no information for audio listeners"
        ),
        "enabled_by_default": True,
    },
    {
        "name": "numeral_conversion",
        "rule_class": "syntrive.adapters.htmlclean.rules.numeral_rule.NumeralConversionRule",
        "sort_order": 40,
        "display_name": "Numeral Conversion",
        "description": "Convert roman/arabic numerals to spoken words (requires numeral_rules.yaml)",
        "enabled_by_default": True,
    },
    {
        "name": "whitespace",
        "rule_class": "syntrive.adapters.htmlclean.rules.whitespace_rule.WhitespaceRule",
        "sort_order": 50,
        "display_name": "Whitespace Cleanup",
        "description": "Remove excess whitespace characters",
        "enabled_by_default": True,
    },
    {
        "name": "hard_newline_to_br",
        "rule_class": "syntrive.adapters.htmlclean.rules.hard_newline_rule.HardNewlineToBrRule",
        "sort_order": 55,
        "display_name": "Hard Newline to <br>",
        "description": (
            "Turn typesetting hard returns inside a leaf <p> into <br> "
            "(text extraction then joins the lines with sentence periods)"
        ),
        "enabled_by_default": True,
    },
    {
        "name": "empty_element",
        "rule_class": "syntrive.adapters.htmlclean.rules.empty_rule.EmptyElementRule",
        "sort_order": 60,
        "display_name": "Empty Element Removal",
        "description": "Remove empty HTML elements left by other rules",
        "enabled_by_default": True,
    },
]


def seed_cleaning_rules(db) -> int:
    from syntrive.db.models import CleaningRule

    existing_orders: set[int] = {
        row.sort_order
        for row in db.query(CleaningRule.sort_order).all()
    }

    inserted = 0
    for rule_def in DEFAULT_CLEANING_RULES:
        if db.query(CleaningRule).filter_by(name=rule_def["name"]).first():
            continue

        if rule_def["sort_order"] in existing_orders:
            logger.warning(
                "seed_cleaning_rules: sort_order=%d conflict for rule=%s — "
                "rule not inserted. Manually assign a free sort_order to add it.",
                rule_def["sort_order"], rule_def["name"],
            )
            continue

        db.add(CleaningRule(
            name=rule_def["name"],
            rule_class=rule_def["rule_class"],
            sort_order=rule_def["sort_order"],
            display_name=rule_def.get("display_name"),
            description=rule_def.get("description"),
            enabled_by_default=rule_def.get("enabled_by_default", True),
        ))
        existing_orders.add(rule_def["sort_order"])
        inserted += 1
        logger.debug(
            "seed_cleaning_rules: inserted rule name=%s sort_order=%d",
            rule_def["name"], rule_def["sort_order"],
        )

    if inserted:
        db.commit()
        logger.info("seed_cleaning_rules: inserted %d new cleaning rule(s)", inserted)
    return inserted
