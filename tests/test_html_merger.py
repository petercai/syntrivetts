from __future__ import annotations

from syntrive.adapters.epub.html_extractor import ChapterHtmlNode
from syntrive.adapters.epub.html_merger import HtmlMerger
from syntrive.adapters.epub.spine_scanner import SpineEntry
from syntrive.adapters.epub.toc_planner import GroupEntry, HtmlGroupPlan, TocNodeInfo


def _chapter(order: int, title: str, href: str) -> ChapterHtmlNode:
    return ChapterHtmlNode(
        toc_id=f"ch_{order + 1:04d}",
        title=title,
        href=href,
        anchor=None,
        raw_html=f"<html><body><p>{title}</p></body></html>",
        order=order,
    )


def _spine(pos: int, href: str) -> SpineEntry:
    return SpineEntry(
        href=href,
        spine_pos=pos,
        has_toc_ref=True,
        raw_html=f"<html><body><p>content {href}</p></body></html>",
    )


def _build_volume_split_book() -> tuple[list[ChapterHtmlNode], list[SpineEntry], HtmlGroupPlan]:
    chapters = [
        _chapter(0, "Preface", "preface.html"),
        _chapter(1, "Part One", "part1.html"),
        _chapter(2, "Chapter 1", "ch1.html"),
        _chapter(3, "Chapter 2", "ch2.html"),
        _chapter(4, "Part Two", "part2.html"),
        _chapter(5, "Chapter 3", "ch3.html"),
        _chapter(6, "Chapter 4", "ch4.html"),
        _chapter(7, "Appendix", "appendix.html"),
    ]
    spine = [
        _spine(0, "preface.html"),
        _spine(1, "part1.html"),
        _spine(2, "ch1.html"),
        _spine(3, "ch2.html"),
        _spine(4, "part2.html"),
        _spine(5, "ch3.html"),
        _spine(6, "ch4.html"),
        _spine(7, "appendix.html"),
    ]
    plan = HtmlGroupPlan(
        mode="volume_split",
        groups=[
            GroupEntry(group_id="vol_0001", title="Part One", href_set=frozenset({"ch1.html", "ch2.html"})),
            GroupEntry(group_id="vol_0002", title="Part Two", href_set=frozenset({"ch3.html", "ch4.html"})),
        ],
    )
    return chapters, spine, plan


class TestVolumeHeaderOrdering:
    def test_merged_order_matches_chapter_id_sequence(self):
        chapters, spine, plan = _build_volume_split_book()
        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )

        regular_ids = [f.chapter_id for f in files if not f.is_discard]
        assert regular_ids == sorted(regular_ids), (
            f"regular chapter_id order must be ascending; got {regular_ids}"
        )
        assert regular_ids == [
            "ch_0001", "ch_0002", "ch_0003", "ch_0004",
            "ch_0005", "ch_0006", "ch_0007", "ch_0008",
        ]

    def test_volume_header_and_frontmatter_default_excluded(self):
        chapters, spine, plan = _build_volume_split_book()
        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )
        by_id = {f.chapter_id: f for f in files if not f.is_discard}

        for cid in ("ch_0001", "ch_0002", "ch_0005", "ch_0008"):
            assert by_id[cid].default_excluded is True, f"{cid} should default_excluded=True"

        for cid in ("ch_0003", "ch_0004", "ch_0006", "ch_0007"):
            assert by_id[cid].default_excluded is False, f"{cid} should default_excluded=False"

    def test_discard_entries_unaffected_by_sort(self):
        chapters, spine, plan = _build_volume_split_book()
        spine = [SpineEntry(href="cover.html", spine_pos=-1, has_toc_ref=False, raw_html="<html><body><img/></body></html>")] + [
            SpineEntry(href=e.href, spine_pos=e.spine_pos + 1, has_toc_ref=e.has_toc_ref, raw_html=e.raw_html)
            for e in spine
        ]
        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )
        discards = [f for f in files if f.is_discard]
        assert len(discards) == 1
        assert discards[0].chapter_id == "ch_9000"


class TestQ8GapBoundaryTrailingContent:
    def test_trailing_content_after_last_chapter_stays_with_it_not_next_volume(self):
        chapters, spine, plan = _build_volume_split_book()

        gap_entries = [
            SpineEntry(href="gap1.html", spine_pos=4, has_toc_ref=False,
                       raw_html="<html><body><p>trailing content 1</p></body></html>"),
            SpineEntry(href="gap2.html", spine_pos=5, has_toc_ref=False,
                       raw_html="<html><body><p>trailing content 2</p></body></html>"),
        ]
        spine = [
            SpineEntry(href=e.href, spine_pos=e.spine_pos if e.spine_pos < 4 else e.spine_pos + 2,
                       has_toc_ref=e.has_toc_ref, raw_html=e.raw_html)
            for e in spine
        ]
        spine = sorted(spine + gap_entries, key=lambda e: e.spine_pos)

        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )
        by_id = {f.chapter_id: f for f in files if not f.is_discard}

        assert by_id["ch_0004"].source_hrefs == ["ch2.html", "gap1.html", "gap2.html"]
        assert by_id["ch_0006"].source_hrefs == ["ch3.html"]
        assert by_id["ch_0005"].source_hrefs == ["part2.html"]


def _build_three_level_book() -> tuple[list[ChapterHtmlNode], list[SpineEntry], HtmlGroupPlan]:
    chapters = [
        _chapter(0, "Book One", "book1.html"),
        _chapter(1, "Part One", "part1.html"),
        _chapter(2, "Chapter 1", "ch1.html"),
        _chapter(3, "Chapter 2", "ch2.html"),
        _chapter(4, "Part Two", "part2.html"),
        _chapter(5, "Chapter 3", "ch3.html"),
    ]
    spine = [_spine(i, ch.href) for i, ch in enumerate(chapters)]
    plan = HtmlGroupPlan(
        mode="volume_split",
        groups=[
            GroupEntry(
                group_id="vol_0001",
                title="Book One",
                href_set=frozenset({"part1.html", "ch1.html", "ch2.html", "part2.html", "ch3.html"}),
            ),
        ],
        node_info_by_order={
            0: TocNodeInfo(depth=0, is_container=True),
            1: TocNodeInfo(depth=1, is_container=True),
            2: TocNodeInfo(depth=2, is_container=False),
            3: TocNodeInfo(depth=2, is_container=False),
            4: TocNodeInfo(depth=1, is_container=True),
            5: TocNodeInfo(depth=2, is_container=False),
        },
    )
    return chapters, spine, plan


class TestArbitraryDepthHeadings:
    def test_heading_levels_mirror_toc_depth(self):
        chapters, spine, plan = _build_three_level_book()
        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )
        by_id = {f.chapter_id: f for f in files if not f.is_discard}

        assert by_id["ch_0001"].heading_level == 2
        assert by_id["ch_0002"].heading_level == 3
        assert by_id["ch_0003"].heading_level == 4
        assert by_id["ch_0004"].heading_level == 4
        assert by_id["ch_0005"].heading_level == 3
        assert by_id["ch_0006"].heading_level == 4

    def test_container_nodes_at_any_depth_default_excluded(self):
        chapters, spine, plan = _build_three_level_book()
        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )
        by_id = {f.chapter_id: f for f in files if not f.is_discard}

        for cid in ("ch_0001", "ch_0002", "ch_0005"):
            assert by_id[cid].default_excluded is True, f"{cid} should default_excluded=True"

        for cid in ("ch_0003", "ch_0004", "ch_0006"):
            assert by_id[cid].default_excluded is False, f"{cid} should default_excluded=False"

    def test_order_still_natural_at_three_levels(self):
        chapters, spine, plan = _build_three_level_book()
        files = HtmlMerger().merge(
            chapters, spine, plan, book_title="Test Book", override="volume_split"
        )
        regular_ids = [f.chapter_id for f in files if not f.is_discard]
        assert regular_ids == [f"ch_{i:04d}" for i in range(1, 7)]

