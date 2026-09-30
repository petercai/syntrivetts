from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syntrive.adapters.epub.html_extractor import ChapterHtmlNode
    from syntrive.adapters.epub.toc_planner import HtmlGroupPlan
    from syntrive.adapters.epub.spine_scanner import SpineEntry

logger = logging.getLogger(__name__)

_MIN_CHAPTERS_PER_VOLUME = 3

_DOMINANT_GROUP_FRACTION = 0.70


@dataclass(frozen=True)
class MergedChapterFile:
    chapter_id: str
    title: str
    body_html: str
    is_discard: bool
    source_hrefs: list[str]
    merge_mode: str
    default_excluded: bool = False
    heading_level: int = 2


@dataclass(frozen=True)
class MergedHtmlDocument:
    merge_id: str
    title: str
    toc_nav_html: str = ""
    body_html: str = ""
    chapter_ids: list[str] = field(default_factory=list)
    merge_mode: str = "flat"
    non_toc_spine_count: int = 0


class HtmlMerger:
    def merge(
        self,
        chapters: list["ChapterHtmlNode"],
        spine_entries: list["SpineEntry"] | None,
        group_plan: "HtmlGroupPlan",
        book_title: str = "",
        override: str | None = None,
    ) -> list[MergedChapterFile]:
        if not chapters:
            logger.warning("HtmlMerger: empty chapter list, returning empty result")
            return []

        mode = _detect_merge_mode(group_plan, chapters, override)
        spine = sorted(spine_entries or [], key=lambda e: e.spine_pos)

        logger.info(
            "HtmlMerger: book=%r mode=%s chapters=%d spine=%d",
            book_title, mode, len(chapters), len(spine),
        )

        spine_pos_map: dict[str, int] = {}
        for e in spine:
            spine_pos_map.setdefault(_norm_href(e.href), e.spine_pos)

        toc_href_set: set[str] = {_norm_href(ch.href) for ch in chapters}
        first_toc_pos, last_toc_pos = _detect_spine_range(spine, toc_href_set, spine_pos_map)

        orphan_counter = itertools.count(9000)
        result: list[MergedChapterFile] = []

        pre_book = [
            e for e in spine
            if e.spine_pos < first_toc_pos and not e.has_toc_ref
        ]
        for entry in pre_book:
            discard_id = f"ch_{next(orphan_counter):04d}"
            result.append(_build_discard_file([entry], discard_id, mode))
            logger.info(
                "HtmlMerger: pre-book orphan href=%r → %s (Q7)", entry.href, discard_id
            )

        in_range_spine = [
            e for e in spine
            if first_toc_pos <= e.spine_pos <= last_toc_pos
        ]

        if mode in ("flat", "volume_continuous"):
            result.extend(
                _merge_flat_chapters(chapters, in_range_spine, spine_pos_map,
                                     toc_href_set, group_plan, mode)
            )
        else:
            result.extend(
                _merge_volume_chapters(chapters, in_range_spine, spine_pos_map,
                                       toc_href_set, group_plan, orphan_counter, mode)
            )

        post_book = [
            e for e in spine
            if e.spine_pos > last_toc_pos and not e.has_toc_ref
        ]
        for entry in post_book:
            discard_id = f"ch_{next(orphan_counter):04d}"
            result.append(_build_discard_file([entry], discard_id, mode))
            logger.info(
                "HtmlMerger: post-book orphan href=%r → %s (Q7)", entry.href, discard_id
            )

        emitted_ids = {f.chapter_id for f in result}
        for ch in sorted(chapters, key=lambda c: c.order):
            if ch.toc_id not in emitted_ids:
                is_container, heading_level = _node_info_for_chapter(ch, group_plan)
                result.append(_build_chapter_file(
                    [ch], [], [], mode,
                    default_excluded=is_container, heading_level=heading_level,
                ))
                logger.debug(
                    "HtmlMerger: chapter=%s not found in spine — added in TOC order",
                    ch.toc_id,
                )

        result = _sort_chapters_by_id(result)

        toc_count = sum(1 for f in result if not f.is_discard)
        discard_count = sum(1 for f in result if f.is_discard)
        logger.info(
            "HtmlMerger: complete — mode=%s toc_chapters=%d discard(ch_9NNN)=%d",
            mode, toc_count, discard_count,
        )
        return result


def _merge_flat_chapters(
    chapters: list["ChapterHtmlNode"],
    in_range_spine: list["SpineEntry"],
    spine_pos_map: dict[str, int],
    toc_href_set: set[str],
    group_plan: "HtmlGroupPlan",
    mode: str,
) -> list[MergedChapterFile]:
    href_to_chapters = _build_href_chapter_map(chapters)
    result: list[MergedChapterFile] = []

    groups = _partition_spine_into_chapter_groups(
        in_range_spine, toc_href_set, href_to_chapters
    )

    for toc_chs, trailing in groups:
        for i, ch in enumerate(toc_chs):
            is_last = i == len(toc_chs) - 1
            trailing_for_ch = trailing if is_last else []
            is_container, heading_level = _node_info_for_chapter(ch, group_plan)
            result.append(_build_chapter_file(
                [ch], [], trailing_for_ch, mode,
                default_excluded=is_container, heading_level=heading_level,
            ))
            logger.debug(
                "HtmlMerger: [flat] chapter=%s trailing=%d depth_level=H%d container=%s",
                ch.toc_id, len(trailing_for_ch), heading_level, is_container,
            )

    return result


def _merge_volume_chapters(
    chapters: list["ChapterHtmlNode"],
    in_range_spine: list["SpineEntry"],
    spine_pos_map: dict[str, int],
    toc_href_set: set[str],
    group_plan: "HtmlGroupPlan",
    orphan_counter,
    mode: str,
) -> list[MergedChapterFile]:
    href_to_chapters = _build_href_chapter_map(chapters)
    result: list[MergedChapterFile] = []

    named_groups = [g for g in group_plan.groups if g.group_id != "main"]

    def _vol_first_spine_pos(grp) -> int:
        positions = [
            spine_pos_map[_norm_href(h)]
            for h in grp.href_set
            if _norm_href(h) in spine_pos_map
        ]
        return min(positions) if positions else 0

    sorted_vols = sorted(named_groups, key=_vol_first_spine_pos)
    prev_vol_last_pos: int = -1

    for vol_idx, grp in enumerate(sorted_vols):
        norm_vol_hrefs: set[str] = {_norm_href(h) for h in grp.href_set}
        vol_chapters = [ch for ch in chapters if _norm_href(ch.href) in norm_vol_hrefs]
        if not vol_chapters:
            logger.debug("HtmlMerger: volume %s has no chapters — skipped", grp.group_id)
            continue

        vol_pos_list = [
            spine_pos_map[n] for n in norm_vol_hrefs if n in spine_pos_map
        ]
        if not vol_pos_list:
            for ch in sorted(vol_chapters, key=lambda c: c.order):
                is_container, heading_level = _node_info_for_chapter(ch, group_plan)
                result.append(_build_chapter_file(
                    [ch], [], [], mode,
                    default_excluded=is_container, heading_level=heading_level,
                ))
            continue

        vol_first = min(vol_pos_list)
        vol_last = max(vol_pos_list)
        vol_last = _extend_last_pos_with_trailing(vol_last, in_range_spine)

        if prev_vol_last_pos >= 0:
            gap = [
                e for e in in_range_spine
                if prev_vol_last_pos < e.spine_pos < vol_first
                and not e.has_toc_ref
            ]
            content_gap = [e for e in gap if _has_text_content(e.raw_html)]
            image_gap = [e for e in gap if not _has_text_content(e.raw_html)]

            for entry in image_gap:
                discard_id = f"ch_{next(orphan_counter):04d}"
                result.append(_build_discard_file([entry], discard_id, mode))
                logger.debug(
                    "HtmlMerger: inter-vol gap href=%r (images) → %s (Q8)",
                    entry.href, discard_id,
                )
            prepend_entries = content_gap
            if content_gap:
                logger.info(
                    "HtmlMerger: inter-vol gap %d content entries → prepend to vol %s first chapter (Q8)",
                    len(content_gap), grp.group_id,
                )
        else:
            prepend_entries = []

        vol_spine = [e for e in in_range_spine if vol_first <= e.spine_pos <= vol_last]

        groups = _partition_spine_into_chapter_groups(
            vol_spine, norm_vol_hrefs, href_to_chapters
        )

        first_group = True
        for toc_chs, trailing in groups:
            for i, ch in enumerate(toc_chs):
                is_last_in_group = i == len(toc_chs) - 1
                is_first_in_group = i == 0
                leading = prepend_entries if (first_group and is_first_in_group) else []
                trailing_for_ch = trailing if is_last_in_group else []
                is_container, heading_level = _node_info_for_chapter(ch, group_plan)
                result.append(_build_chapter_file(
                    [ch], leading, trailing_for_ch, mode,
                    default_excluded=is_container, heading_level=heading_level,
                ))
                logger.debug(
                    "HtmlMerger: [vol=%s] chapter=%s leading=%d trailing=%d depth_level=H%d container=%s",
                    grp.group_id, ch.toc_id, len(leading), len(trailing_for_ch),
                    heading_level, is_container,
                )
            first_group = False
            prepend_entries = []

        emitted = {f.chapter_id for f in result}
        for ch in sorted(vol_chapters, key=lambda c: c.order):
            if ch.toc_id not in emitted:
                is_container, heading_level = _node_info_for_chapter(ch, group_plan)
                result.append(_build_chapter_file(
                    [ch], [], [], mode,
                    default_excluded=is_container, heading_level=heading_level,
                ))

        prev_vol_last_pos = vol_last

    all_vol_hrefs = {_norm_href(h) for g in sorted_vols for h in g.href_set}
    ungrouped = [ch for ch in chapters if _norm_href(ch.href) not in all_vol_hrefs]
    if ungrouped:
        logger.debug(
            "HtmlMerger: %d ungrouped chapters added (default_excluded=True)", len(ungrouped)
        )
        for ch in sorted(ungrouped, key=lambda c: c.order):
            _, heading_level = _node_info_for_chapter(ch, group_plan)
            result.append(_build_chapter_file(
                [ch], [], [], mode, default_excluded=True, heading_level=heading_level,
            ))

    return result


def _partition_spine_into_chapter_groups(
    spine_entries: list["SpineEntry"],
    toc_href_set: set[str],
    href_to_chapters: dict[str, list["ChapterHtmlNode"]],
) -> list[tuple[list["ChapterHtmlNode"], list["SpineEntry"]]]:
    groups: list[tuple[list, list]] = []
    current_chs: list = []
    current_trailing: list = []
    emitted_hrefs: set[str] = set()

    for entry in spine_entries:
        norm = _norm_href(entry.href)
        if norm in toc_href_set and norm not in emitted_hrefs:
            if current_chs:
                groups.append((current_chs, current_trailing))
            current_chs = href_to_chapters.get(norm, [])
            current_trailing = []
            emitted_hrefs.add(norm)
        elif norm not in toc_href_set:
            current_trailing.append(entry)

    if current_chs:
        groups.append((current_chs, current_trailing))

    return groups


def _build_chapter_file(
    toc_chapters: list["ChapterHtmlNode"],
    leading_entries: list["SpineEntry"],
    trailing_entries: list["SpineEntry"],
    mode: str,
    default_excluded: bool = False,
    heading_level: int = 2,
) -> MergedChapterFile:
    if not toc_chapters:
        raise ValueError("_build_chapter_file: toc_chapters must not be empty")

    first_ch = toc_chapters[0]
    chapter_id = first_ch.toc_id
    title = first_ch.title
    source_hrefs: list[str] = []

    sections: list[str] = []

    for entry in leading_entries:
        sections.append(
            f'<section'
            f' data-spine-src="{_escape_attr(entry.href)}"'
            f' data-spine-pos="{entry.spine_pos}"'
            f' data-gap-prepend="true">\n'
            f'{_extract_body_html(entry.raw_html)}\n'
            f'</section>'
        )
        source_hrefs.append(entry.href)
        logger.debug(
            "HtmlMerger: chapter=%s prepend gap href=%r",
            chapter_id, entry.href,
        )

    for ch in toc_chapters:
        sections.append(
            f'<section id="{ch.toc_id}"'
            f' data-toc-title="{_escape_attr(ch.title)}"'
            f' data-spine-src="{_escape_attr(ch.href)}"'
            f' data-toc-index="{ch.order}">\n'
            f'{_extract_body_html(ch.raw_html)}\n'
            f'</section>'
        )
        if ch.href not in source_hrefs:
            source_hrefs.append(ch.href)

    for entry in trailing_entries:
        if entry.raw_html.strip():
            sections.append(
                f'<section'
                f' data-spine-src="{_escape_attr(entry.href)}"'
                f' data-spine-pos="{entry.spine_pos}">\n'
                f'{_extract_body_html(entry.raw_html)}\n'
                f'</section>'
            )
            if entry.href not in source_hrefs:
                source_hrefs.append(entry.href)

    body_content = "\n\n".join(sections)
    return MergedChapterFile(
        chapter_id=chapter_id,
        title=title,
        body_html=_wrap_html(title, body_content),
        is_discard=False,
        source_hrefs=source_hrefs,
        merge_mode=mode,
        default_excluded=default_excluded,
        heading_level=heading_level,
    )


def _build_discard_file(
    spine_entries: list["SpineEntry"],
    discard_id: str,
    mode: str,
) -> MergedChapterFile:
    sections: list[str] = []
    source_hrefs: list[str] = []

    for entry in spine_entries:
        sections.append(
            f'<section'
            f' data-spine-src="{_escape_attr(entry.href)}"'
            f' data-spine-pos="{entry.spine_pos}">\n'
            f'{_extract_body_html(entry.raw_html)}\n'
            f'</section>'
        )
        source_hrefs.append(entry.href)

    body_content = "\n\n".join(sections)
    label = f"(discard {discard_id})"
    return MergedChapterFile(
        chapter_id=discard_id,
        title="",
        body_html=_wrap_html(label, body_content),
        is_discard=True,
        source_hrefs=source_hrefs,
        merge_mode=mode,
    )


def _sort_chapters_by_id(result: list[MergedChapterFile]) -> list[MergedChapterFile]:
    regular = sorted(
        (cf for cf in result if not cf.is_discard),
        key=lambda cf: int(cf.chapter_id.split("_", 1)[1]),
    )
    discards = [cf for cf in result if cf.is_discard]
    return regular + discards


_MAX_HEADING_LEVEL = 6


def _node_info_for_chapter(
    ch: "ChapterHtmlNode", group_plan: "HtmlGroupPlan | None"
) -> tuple[bool, int]:
    if group_plan is None:
        return False, 2

    info = group_plan.get_node_info(ch.order)
    heading_level = info.depth + 2
    if heading_level > _MAX_HEADING_LEVEL:
        logger.warning(
            "HtmlMerger: chapter=%s TOC depth=%d exceeds heading cap — "
            "clamped to H%d",
            ch.toc_id, info.depth, _MAX_HEADING_LEVEL,
        )
        heading_level = _MAX_HEADING_LEVEL
    return info.is_container, heading_level


def _detect_merge_mode(
    group_plan: "HtmlGroupPlan",
    chapters: list["ChapterHtmlNode"],
    override: str | None,
) -> str:
    if override:
        logger.info("HtmlMerger: merge_mode OVERRIDDEN by user: %s", override)
        return override

    if group_plan.mode == "flat":
        return "flat"

    named_groups = [g for g in group_plan.groups if g.group_id != "main"]
    if not named_groups:
        return "flat"

    group_counts: dict[str, int] = {}
    for ch in chapters:
        norm = _norm_href(ch.href)
        for grp in named_groups:
            if any(_norm_href(h) == norm for h in grp.href_set):
                group_counts[grp.group_id] = group_counts.get(grp.group_id, 0) + 1
                break

    counts = [group_counts.get(g.group_id, 0) for g in named_groups]
    total = sum(counts) or 1
    max_count = max(counts) if counts else 0

    logger.debug(
        "HtmlMerger: volume distribution — groups=%d counts=%s total=%d max=%d",
        len(named_groups), counts, total, max_count,
    )

    if max_count / total > _DOMINANT_GROUP_FRACTION:
        logger.info(
            "HtmlMerger: largest group dominates (%.0f%%) → volume_continuous",
            max_count / total * 100,
        )
        return "volume_continuous"

    active_counts = [c for c in counts if c > 0]
    if active_counts and min(active_counts) < _MIN_CHAPTERS_PER_VOLUME:
        logger.info(
            "HtmlMerger: min group size=%d < threshold=%d → volume_continuous",
            min(active_counts), _MIN_CHAPTERS_PER_VOLUME,
        )
        return "volume_continuous"

    return "volume_split"


def _extend_last_pos_with_trailing(
    last_pos: int,
    in_range_spine: list["SpineEntry"],
) -> int:
    extended = last_pos
    trailing = sorted(
        (e for e in in_range_spine if e.spine_pos > last_pos),
        key=lambda e: e.spine_pos,
    )
    for entry in trailing:
        if entry.has_toc_ref:
            break
        extended = entry.spine_pos
    return extended


def _detect_spine_range(
    spine_entries: list["SpineEntry"],
    toc_href_set: set[str],
    spine_pos_map: dict[str, int],
) -> tuple[int, int]:
    positions = [
        spine_pos_map[norm]
        for norm in toc_href_set
        if norm in spine_pos_map
    ]
    if not positions:
        if spine_entries:
            all_pos = [e.spine_pos for e in spine_entries]
            return (min(all_pos), max(all_pos))
        return (0, 0)
    return (min(positions), max(positions))


def _has_text_content(html: str) -> bool:
    if not html or not html.strip():
        return False
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    return bool(
        soup.find(re.compile(r"^h[1-6]$"))
        or soup.find("p")
    )


def _extract_body_html(raw_html: str) -> str:
    if "<body" not in raw_html.lower():
        return raw_html
    from bs4 import BeautifulSoup
    body = BeautifulSoup(raw_html, "html.parser").find("body")
    if body is None:
        return raw_html
    return body.decode_contents()


def _wrap_html(title: str, body_content: str) -> str:
    safe_title = _escape_html(title)
    return (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '<head><meta charset="UTF-8">'
        f'<title>{safe_title}</title></head>\n'
        '<body>\n'
        f'{body_content}\n'
        '</body>\n'
        '</html>'
    )


def _build_href_chapter_map(
    chapters: list["ChapterHtmlNode"],
) -> dict[str, list["ChapterHtmlNode"]]:
    mapping: dict[str, list] = {}
    for ch in chapters:
        mapping.setdefault(_norm_href(ch.href), []).append(ch)
    for lst in mapping.values():
        lst.sort(key=lambda c: c.order)
    return mapping


def _norm_href(href: str) -> str:
    from pathlib import Path as _Path
    return _Path(href).name.lower()


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def toc_hrefs_from_chapters(chapters: list["ChapterHtmlNode"]) -> set[str]:
    return {ch.href for ch in chapters}
