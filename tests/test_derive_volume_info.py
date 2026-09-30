from __future__ import annotations

from syntrive.adapters.epub.html_merger import MergedChapterFile
from syntrive.adapters.epub.toc_planner import GroupEntry, HtmlGroupPlan
from syntrive.pipeline.transcript_html_stage import _derive_volume_info


def _chapter_file(chapter_id: str, source_hrefs: list[str]) -> MergedChapterFile:
    return MergedChapterFile(
        chapter_id=chapter_id,
        title="Chapter",
        body_html="<html><body><p>x</p></body></html>",
        is_discard=False,
        source_hrefs=source_hrefs,
        merge_mode="volume_split",
    )


def _plan() -> HtmlGroupPlan:
    return HtmlGroupPlan(
        mode="volume_split",
        groups=[
            GroupEntry(group_id="vol_0001", title="Volume One", href_set=frozenset({"ch1.html", "ch2.html"})),
            GroupEntry(group_id="vol_0002", title="Volume Two", href_set=frozenset({"ch3.html", "ch4.html"})),
        ],
    )


class TestDeriveVolumeInfo:
    def test_normal_chapter_first_href_is_own_toc_href(self):
        cf = _chapter_file("ch_0002", ["ch2.html"])
        vol_num, vol_name = _derive_volume_info(cf, _plan())
        assert (vol_num, vol_name) == ("0001", "Volume One")

    def test_first_chapter_of_volume_with_prepended_gap_href(self):
        cf = _chapter_file("ch_0013", ["gap1.html", "gap2.html", "ch3.html"])
        vol_num, vol_name = _derive_volume_info(cf, _plan())
        assert (vol_num, vol_name) == ("0002", "Volume Two"), (
            "must skip leading non-TOC gap hrefs and find the real TOC href"
        )

    def test_no_href_resolves_to_a_volume(self):
        cf = _chapter_file("ch_0001", ["preface.html"])
        assert _derive_volume_info(cf, _plan()) == ("", "")

    def test_flat_book_returns_empty(self):
        flat_plan = HtmlGroupPlan(
            mode="flat",
            groups=[GroupEntry(group_id="main", title="", href_set=frozenset({"ch1.html"}))],
        )
        cf = _chapter_file("ch_0001", ["ch1.html"])
        assert _derive_volume_info(cf, flat_plan) == ("", "")

    def test_no_group_plan_returns_empty(self):
        cf = _chapter_file("ch_0001", ["ch1.html"])
        assert _derive_volume_info(cf, None) == ("", "")

    def test_no_source_hrefs_returns_empty(self):
        cf = _chapter_file("ch_0001", [])
        assert _derive_volume_info(cf, _plan()) == ("", "")
