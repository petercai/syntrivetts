from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest
from bs4 import BeautifulSoup


@dataclass
class RuleCase:
    id: str
    html_input: str
    expected_contains: list[str] = field(default_factory=list)
    expected_absent: list[str] = field(default_factory=list)
    chars_removed_min: int = 0
    chars_removed_eq: Optional[int] = None
    lang: str = "zh"


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _apply_case(soup, removed, case: RuleCase) -> None:
    text = soup.get_text()
    for expected in case.expected_contains:
        assert expected in text, f"Expected {expected!r} in output but not found.\nText: {text!r}"
    for absent in case.expected_absent:
        assert absent not in text, (
            f"Expected {absent!r} to be absent but found.\nText: {text!r}"
        )
    if case.chars_removed_eq is not None:
        assert removed == case.chars_removed_eq, (
            f"Expected chars_removed=={case.chars_removed_eq}, got {removed}"
        )
    elif case.chars_removed_min > 0:
        assert removed >= case.chars_removed_min, (
            f"Expected chars_removed>={case.chars_removed_min}, got {removed}"
        )


ENGLISH_BLOCK_CASES: list[RuleCase] = [
    RuleCase(
        id="removes_pure_ascii_block_for_zh",
        html_input="<p>This is a completely English paragraph with only ASCII characters.</p>",
        expected_absent=["This is"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="noop_for_english_target_language",
        html_input="<p>This entire paragraph is in English and must NOT be removed.</p>",
        expected_contains=["English"],
        chars_removed_eq=0,
        lang="en",
    ),
    RuleCase(
        id="en_gb_variant_also_skipped",
        html_input="<p>British English paragraph, must stay.</p>",
        expected_contains=["British English"],
        chars_removed_eq=0,
        lang="en-GB",
    ),
    RuleCase(
        id="keeps_chinese_dominant_paragraph",
        html_input="<p>这是一段中文文字，包含少量英文如ABC，但主体是中文，应保留。</p>",
        expected_contains=["中文"],
    ),
    RuleCase(
        id="mixed_below_70pct_ascii_kept",
        html_input="<p>这些是中文Hello World这些是中文</p>",
        expected_contains=["中文"],
    ),
    RuleCase(
        id="empty_block_not_counted",
        html_input="<p>   </p>",
        chars_removed_eq=0,
    ),
    RuleCase(
        id="multiple_leaf_paragraphs_selective_removal",
        html_input=(
            "<p>Fully ASCII English line that should be deleted.</p>"
            "<p>这行中文段落必须保留下来。</p>"
            "<p>Another English paragraph to be removed.</p>"
        ),
        expected_contains=["中文段落"],
        expected_absent=["English"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="above_70pct_ascii_removed",
        html_input="<p>AAAAAAAA中文</p>",
        expected_absent=["AAAAAAAA"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="at_exactly_70pct_ascii_kept",
        html_input="<p>AAAAAAA中文中</p>",
        expected_contains=["AAAAAAA"],
    ),
    RuleCase(
        id="just_below_70pct_ascii_kept",
        html_input="<p>AAAAAA中文中文</p>",
        expected_contains=["AAAAAA"],
    ),
    RuleCase(
        id="container_div_with_mixed_children_not_removed",
        html_input=(
            "<div>"
            "<p>Fully ASCII English line that should be deleted.</p>"
            "<p>这行中文段落必须保留下来。</p>"
            "<p>Another English paragraph to be removed.</p>"
            "</div>"
        ),
        expected_contains=["中文段落"],
        expected_absent=["ASCII English", "Another English"],
        chars_removed_min=1,
    ),
]


@pytest.mark.parametrize("case", ENGLISH_BLOCK_CASES, ids=[c.id for c in ENGLISH_BLOCK_CASES])
def test_english_block_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.english_rule import EnglishBlockRule

    rule = EnglishBlockRule(target_language=case.lang)
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


WHITESPACE_CASES: list[RuleCase] = [
    RuleCase(
        id="collapses_multiple_spaces",
        html_input="<p>This  has   many    spaces.</p>",
        expected_absent=["  "],
        chars_removed_min=1,
    ),
    RuleCase(
        id="collapses_tabs",
        html_input="<p>Word\t\t\tAnother</p>",
        chars_removed_min=1,
        expected_absent=["\t\t"],
    ),
    RuleCase(
        id="collapses_non_breaking_spaces",
        html_input="<p>Word\xa0\xa0Another</p>",
        chars_removed_min=1,
    ),
    RuleCase(
        id="single_spaces_untouched",
        html_input="<p>Normal single spaced text no changes needed.</p>",
        expected_contains=["Normal single spaced text"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="preserves_content_after_collapse",
        html_input="<p>Hello  World</p>",
        expected_contains=["Hello", "World"],
    ),
    RuleCase(
        id="collapses_mixed_whitespace_types",
        html_input="<p>A\t  B\xa0 C</p>",
        expected_contains=["A", "B", "C"],
        chars_removed_min=1,
    ),
]


@pytest.mark.parametrize("case", WHITESPACE_CASES, ids=[c.id for c in WHITESPACE_CASES])
def test_whitespace_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.whitespace_rule import WhitespaceRule

    rule = WhitespaceRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


EMPTY_ELEMENT_CASES: list[RuleCase] = [
    RuleCase(
        id="removes_empty_paragraph",
        html_input="<div><p></p><p>Real content.</p></div>",
        expected_contains=["Real content."],
    ),
    RuleCase(
        id="keeps_non_empty_paragraph",
        html_input="<p>Content</p>",
        expected_contains=["Content"],
    ),
    RuleCase(
        id="removes_whitespace_only_paragraph",
        html_input="<p>   </p>",
    ),
    RuleCase(
        id="cascade_removes_empty_parent_div",
        html_input="<div><p></p></div>",
    ),
    RuleCase(
        id="returns_zero_chars_removed",
        html_input="<p></p>",
        chars_removed_eq=0,
    ),
    RuleCase(
        id="removes_empty_headings",
        html_input="<h2></h2><p>Content here.</p>",
        expected_contains=["Content here."],
    ),
    RuleCase(
        id="leaves_nested_content_block_intact",
        html_input="<div><p>Outer</p><div><p>Inner</p></div></div>",
        expected_contains=["Outer", "Inner"],
    ),
]


@pytest.mark.parametrize("case", EMPTY_ELEMENT_CASES, ids=[c.id for c in EMPTY_ELEMENT_CASES])
def test_empty_element_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.empty_rule import EmptyElementRule

    rule = EmptyElementRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


TABLE_SUMMARY_CASES: list[RuleCase] = [
    RuleCase(
        id="replaces_table_with_summary_paragraph",
        html_input="<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>",
        expected_contains=["row"],
    ),
    RuleCase(
        id="summary_contains_row_count",
        html_input=(
            "<table>"
            "<tr><td>A</td></tr>"
            "<tr><td>B</td></tr>"
            "<tr><td>C</td></tr>"
            "</table>"
        ),
        expected_contains=["3"],
    ),
    RuleCase(
        id="summary_includes_headers_when_present",
        html_input=(
            "<table>"
            "<tr><th>Name</th><th>Age</th><th>City</th></tr>"
            "<tr><td>Alice</td><td>30</td><td>Paris</td></tr>"
            "</table>"
        ),
        expected_contains=["Name"],
    ),
    RuleCase(
        id="empty_table_produces_summary",
        html_input="<table></table>",
    ),
    RuleCase(
        id="net_removal_positive_for_large_table",
        html_input=(
            "<table>"
            + "".join(
                f"<tr><td>{'Long cell content here ' * 5}</td></tr>"
                for _ in range(10)
            )
            + "</table>"
        ),
        chars_removed_min=1,
    ),
    RuleCase(
        id="table_removed_not_present_after",
        html_input="<table><tr><th>Col</th></tr><tr><td>val</td></tr></table>",
        expected_absent=["<table>"],
    ),
]


@pytest.mark.parametrize("case", TABLE_SUMMARY_CASES, ids=[c.id for c in TABLE_SUMMARY_CASES])
def test_table_summary_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.table_rule import TableSummaryRule

    rule = TableSummaryRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)
    assert soup.find("table") is None, "Table element must not remain after TableSummaryRule"


ANNOTATION_REMOVAL_CASES: list[RuleCase] = [
    RuleCase(
        id="removes_calibre_sup_with_link_and_circled_number",
        html_input=(
            '<p>正文内容'
            '<sup class="calibre19">'
            '<small class="calibre20" id="filepos41108">'
            '<a href="part0007.html#filepos43871">①</a>'
            '</small></sup>'
            '继续。</p>'
        ),
        expected_contains=["正文内容", "继续"],
        expected_absent=["①"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_sup_with_plain_digit_and_link",
        html_input='<p>文字<sup><a href="#fn1">1</a></sup>更多</p>',
        expected_contains=["文字", "更多"],
        expected_absent=["<sup>"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_sup_wrapped_circled_number",
        html_input="<p>段落<sup>①</sup>内容</p>",
        expected_contains=["段落", "内容"],
        expected_absent=["①"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="keeps_bare_circled_number_in_text",
        html_input="<p>第①条规定</p>",
        expected_contains=["①"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="removes_sup_with_asterisk_footnote",
        html_input='<p>内容<sup><a href="#fn">*</a></sup>继续</p>',
        expected_contains=["内容", "继续"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="keeps_sup_with_long_content",
        html_input="<p>x<sup>This is not an annotation</sup></p>",
        expected_contains=["This is not an annotation"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="removes_multiple_sup_annotations",
        html_input=(
            '<p>第一段<sup><a href="#fn1">①</a></sup>'
            '中间<sup><a href="#fn2">②</a></sup>结尾</p>'
        ),
        expected_contains=["第一段", "中间", "结尾"],
        expected_absent=["①", "②"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="noop_when_no_sup_elements",
        html_input="<p>普通段落，没有上标。</p>",
        expected_contains=["普通段落"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="removes_calibre_span_cfs_bracket_number",
        html_input='<p>正文<span class="calibreclass10" id="cfs_1336">[1] </span>继续</p>',
        expected_contains=["正文", "继续"],
        expected_absent=["[1]"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_calibre_span_cfs_multidigit_number",
        html_input='<p>内容<span id="cfs_42">[12]</span>后续</p>',
        expected_contains=["内容", "后续"],
        expected_absent=["[12]"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_multiple_calibre_cfs_spans",
        html_input=(
            '<p>第一<span id="cfs_100">[1] </span>'
            '中间<span id="cfs_200">[2] </span>结尾</p>'
        ),
        expected_contains=["第一", "中间", "结尾"],
        expected_absent=["[1]", "[2]"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="keeps_span_without_cfs_id",
        html_input="<p>这是<span>[1]</span>正文</p>",
        expected_contains=["[1]"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="keeps_calibre_span_with_non_bracket_text",
        html_input='<p>内容<span id="cfs_100">重要说明</span>后续</p>',
        expected_contains=["重要说明"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="keeps_calibre_span_with_mixed_bracket_text",
        html_input='<p>内容<span id="cfs_100">[1a]</span>后续</p>',
        expected_contains=["[1a]"],
        chars_removed_eq=0,
    ),
]


@pytest.mark.parametrize(
    "case", ANNOTATION_REMOVAL_CASES, ids=[c.id for c in ANNOTATION_REMOVAL_CASES]
)
def test_annotation_removal_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.annotation_rule import AnnotationRemovalRule

    rule = AnnotationRemovalRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


FOOTNOTE_PARAGRAPH_CASES: list[RuleCase] = [
    RuleCase(
        id="removes_translator_note_paragraph",
        html_input=(
            '<p class="calibre29" id="filepos145255">'
            '<a href="part0024.html#filepos138501"><span class="calibre5">①</span></a>'
            '<span class="calibre5">语出乔治·奥威尔所著的《1984》。——译者注 </span>'
            "</p>"
        ),
        expected_absent=["译者注", "①"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_editor_note_paragraph",
        html_input=(
            '<p class="calibre29" id="filepos221055">'
            '<a href="part0034.html#filepos199865"><span class="calibre5">①</span></a>'
            '<span class="calibre5">《哈佛幸福课》中文版已由中信出版社出版。——编者注 </span>'
            "</p>"
        ),
        expected_absent=["编者注", "①"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_footnote_paragraph_without_attribution_suffix",
        html_input=(
            '<p class="calibre31" id="filepos175662">'
            '<a href="part0028.html#filepos161745"><span class="calibre5">①</span></a>'
            '<span class="calibre5">答案分别为5分钟和47天。 </span>'
            "</p>"
        ),
        expected_absent=["答案分别为5分钟和47天"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="keeps_inline_parenthetical_without_filepos_structure",
        html_input=(
            '<p class="calibre13">'
            "(禀赋效应是指当个人一旦拥有某个物品，那么他对该物品价值的评价要比未拥有"
            "之前大大提高。—译者注)"
            "</p>"
        ),
        expected_contains=["禀赋效应", "译者注"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="keeps_paragraph_with_filepos_id_but_no_leading_anchor",
        html_input='<p id="filepos99999">普通正文段落，恰好也有 filepos id。</p>',
        expected_contains=["普通正文段落"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="keeps_paragraph_with_leading_anchor_but_no_filepos_id",
        html_input='<p><a href="part0001.html#filepos1">①</a>正文引用，没有自身 filepos id。</p>',
        expected_contains=["正文引用"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="keeps_paragraph_when_anchor_href_is_not_filepos_pattern",
        html_input='<p id="filepos12345"><a href="#toc_1">目录</a>普通目录条目。</p>',
        expected_contains=["目录"],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="removes_multiple_footnote_paragraphs",
        html_input=(
            '<p id="filepos1"><a href="part1.html#filepos2">①</a><span>第一条脚注。——译者注</span></p>'
            '<p>正文段落保持不变。</p>'
            '<p id="filepos3"><a href="part2.html#filepos4">②</a><span>第二条脚注。——编者注</span></p>'
        ),
        expected_contains=["正文段落保持不变"],
        expected_absent=["第一条脚注", "第二条脚注"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="noop_when_no_footnote_paragraphs",
        html_input="<p>普通段落，没有任何脚注结构。</p>",
        expected_contains=["普通段落"],
        chars_removed_eq=0,
    ),
]


@pytest.mark.parametrize(
    "case", FOOTNOTE_PARAGRAPH_CASES, ids=[c.id for c in FOOTNOTE_PARAGRAPH_CASES]
)
def test_footnote_paragraph_removal_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.footnote_paragraph_rule import (
        FootnoteParagraphRemovalRule,
    )

    rule = FootnoteParagraphRemovalRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)




INLINE_ENGLISH_CASES: list[RuleCase] = [
    RuleCase(
        id="removes_english_name_in_fullwidth_parens",
        html_input="<p>作者纳西姆·塔勒布（ Nassim Taleb）的影响</p>",
        expected_contains=["作者纳西姆·塔勒布", "的影响"],
        expected_absent=["Nassim Taleb"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_english_term_in_ascii_parens",
        html_input="<p>自然语言处理(Natural Language Processing)技术</p>",
        expected_contains=["自然语言处理", "技术"],
        expected_absent=["Natural Language Processing"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_english_abbreviation",
        html_input="<p>人工智能（AI）研究</p>",
        expected_contains=["人工智能", "研究"],
        expected_absent=["AI"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="keeps_chinese_only_parens",
        html_input="<p>相关内容（第一章）请见</p>",
        expected_contains=["第一章"],
    ),
    RuleCase(
        id="keeps_mixed_cjk_english_parens",
        html_input="<p>说明（GDP增长）数据</p>",
        expected_contains=["GDP增长"],
    ),
    RuleCase(
        id="noop_for_english_target_language",
        html_input="<p>The author (Nassim Taleb) wrote this.</p>",
        expected_contains=["Nassim Taleb"],
        chars_removed_eq=0,
        lang="en",
    ),
    RuleCase(
        id="removes_multiple_english_parens_in_one_node",
        html_input="<p>塔勒布（Nassim Taleb）和马斯克（Elon Musk）的观点</p>",
        expected_contains=["塔勒布", "马斯克", "的观点"],
        expected_absent=["Nassim Taleb", "Elon Musk"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="keeps_year_in_parens",
        html_input="<p>出版于（2024年）的著作</p>",
        expected_contains=["2024年"],
    ),
    RuleCase(
        id="removes_company_name_in_parens",
        html_input="<p>谷歌（Google DeepMind）的研究</p>",
        expected_contains=["谷歌", "的研究"],
        expected_absent=["Google DeepMind"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="noop_when_no_parens",
        html_input="<p>纯中文段落，没有括号。</p>",
        expected_contains=["纯中文段落"],
        chars_removed_eq=0,
    ),
]


@pytest.mark.parametrize(
    "case", INLINE_ENGLISH_CASES, ids=[c.id for c in INLINE_ENGLISH_CASES]
)
def test_inline_english_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.inline_english_rule import InlineEnglishRule

    rule = InlineEnglishRule(target_language=case.lang)
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


SPAN_MERGE_CASES: list[RuleCase] = [
    RuleCase(
        id="pattern_a_merges_two_adjacent_spans",
        html_input=(
            '<p class="calibreclass6" id="cfs_177">'
            '<span class="calibreclass7" id="cfs_47">I</span>'
            '<span class="calibreclass8" id="cfs_1328">t was bright.</span>'
            '</p>'
        ),
        expected_contains=["It was bright."],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="pattern_a_merges_three_spans_in_heading",
        html_input=(
            '<h2><span>Chap</span><span>ter</span><span> One</span></h2>'
        ),
        expected_contains=["Chapter One"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="pattern_b_merges_text_node_and_single_span_in_h1",
        html_input='<h1 class="heading_s1u">P<span class="class_s7v">ROLOGUE</span> </h1>',
        expected_contains=["PROLOGUE"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="pattern_b_merges_text_node_and_span_in_paragraph",
        html_input='<p>EPI<span class="s">LOGUE</span></p>',
        expected_contains=["EPILOGUE"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="pattern_b_merges_text_then_multiple_spans",
        html_input='<p>A<span>B</span><span>C</span></p>',
        expected_contains=["ABC"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="no_merge_when_anchor_tag_present",
        html_input='<p>See <a href="#ref">link</a> for details.</p>',
        expected_contains=["See", "link", "for details."],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="no_merge_plain_text_block_no_spans",
        html_input="<p>Already plain text, no spans needed.</p>",
        expected_contains=["Already plain text"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="no_merge_single_span_no_text_node",
        html_input='<h1><span class="decorative">PROLOGUE</span></h1>',
        expected_contains=["PROLOGUE"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="no_merge_strong_tag_preserves_semantic",
        html_input="<p><strong>Important:</strong> this section matters.</p>",
        expected_contains=["Important:", "this section matters."],
        expected_absent=[],
        chars_removed_eq=0,
    ),
]


@pytest.mark.parametrize("case", SPAN_MERGE_CASES, ids=[c.id for c in SPAN_MERGE_CASES])
def test_span_merge_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.span_merge_rule import SpanMergeRule

    rule = SpanMergeRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


IMAGE_CAPTION_CASES: list[RuleCase] = [
    RuleCase(
        id="removes_caption_p_from_calibre_image_div",
        html_input=(
            '<div class="class_s3">'
            '<div class="calibre1" id="a7VW">'
            '<img alt="" class="class_s2" src="image_rsrc85Z.jpg"/>'
            "</div>"
            '<p class="caption_s3w">Cora and Walter Musk</p>'
            "</div>"
        ),
        expected_contains=[],
        expected_absent=["Cora and Walter Musk"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_caption_when_img_is_direct_child",
        html_input='<div><img src="photo.jpg"/><p>Some caption text</p></div>',
        expected_contains=[],
        expected_absent=["Some caption text"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="removes_multiple_captions_from_same_image_div",
        html_input=(
            "<div>"
            "<div><img src='a.jpg'/></div>"
            "<p>Caption line one</p>"
            "<p>Caption line two</p>"
            "</div>"
        ),
        expected_contains=[],
        expected_absent=["Caption line one", "Caption line two"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="preserves_p_when_div_has_non_p_text_content",
        html_input=(
            "<div>"
            "<div><img src='a.jpg'/><span>Figure 1</span></div>"
            "<p>This paragraph stays</p>"
            "</div>"
        ),
        expected_contains=["This paragraph stays"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="preserves_p_when_div_has_no_image",
        html_input="<div><p>Regular content paragraph</p></div>",
        expected_contains=["Regular content paragraph"],
        expected_absent=[],
        chars_removed_eq=0,
    ),
    RuleCase(
        id="inner_caption_removed_outer_paragraph_preserved",
        html_input=(
            "<div>"
            "<div><img src='b.jpg'/><p>Inner image caption</p></div>"
            "<p>Outer paragraph content</p>"
            "</div>"
        ),
        expected_contains=["Outer paragraph content"],
        expected_absent=["Inner image caption"],
        chars_removed_min=1,
    ),
]


@pytest.mark.parametrize(
    "case", IMAGE_CAPTION_CASES, ids=[c.id for c in IMAGE_CAPTION_CASES]
)
def test_image_caption_rule(case: RuleCase) -> None:
    from syntrive.adapters.htmlclean.rules.image_caption_rule import ImageCaptionRule

    rule = ImageCaptionRule()
    soup = _soup(case.html_input)
    soup, removed = rule.apply(soup)
    _apply_case(soup, removed, case)


ENGINE_CASES: list[RuleCase] = [
    RuleCase(
        id="returns_string_and_stats_for_simple_input",
        html_input="<p>Hello world.</p>",
        expected_contains=["Hello"],
        lang="en",
    ),
    RuleCase(
        id="near_zero_deletion_for_clean_english",
        html_input="<p>A clean English paragraph with no tables or excessive whitespace.</p>",
        lang="en",
    ),
    RuleCase(
        id="english_content_survives_for_en_book",
        html_input="<p>This is a completely English paragraph. All ASCII content. Must survive.</p>",
        expected_contains=["English paragraph"],
        lang="en",
    ),
    RuleCase(
        id="zh_book_removes_ascii_heavy_blocks",
        html_input=(
            "<div>"
            "<p>This is pure ASCII English text that should be removed.</p>"
            "<p>这里是中文内容，必须保留下来。</p>"
            "</div>"
        ),
        expected_contains=["中文内容"],
        expected_absent=["pure ASCII"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="annotation_removed_by_full_engine",
        html_input=(
            '<p>内容'
            '<sup><a href="#fn1">①</a></sup>'
            '继续</p>'
        ),
        expected_contains=["内容", "继续"],
        expected_absent=["①"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="inline_english_removed_by_full_engine",
        html_input="<p>作者（Nassim Taleb）的观点</p>",
        expected_contains=["作者", "的观点"],
        expected_absent=["Nassim Taleb"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="calibre_span_bracket_number_removed_by_full_engine",
        html_input='<p>内容<span class="calibreclass10" id="cfs_1336">[1] </span>继续</p>',
        expected_contains=["内容", "继续"],
        expected_absent=["[1]"],
        chars_removed_min=1,
    ),
    RuleCase(
        id="bracket_number_removed_when_surrounded_by_sibling_spans",
        html_input=(
            '<p class="calibreclass9" id="cfs_2">'
            '<span class="calibreclass8" id="cfs_1335">Newspeak</span>'
            '<span class="calibreclass10" id="cfs_1336">[1] </span>'
            '<span class="calibreclass8" id="cfs_1337">--was different</span>'
            '</p>'
        ),
        expected_contains=["Newspeak", "--was different"],
        expected_absent=["[1]"],
        chars_removed_min=1,
        lang="en",
    ),
    RuleCase(
        id="calibre_mixed_text_span_heading_merged_by_full_engine",
        html_input='<h1 class="heading_s1u">P<span class="class_s7v">ROLOGUE</span> </h1>',
        expected_contains=["PROLOGUE"],
        expected_absent=[],
        chars_removed_eq=0,
        lang="en",
    ),
    RuleCase(
        id="image_caption_and_empty_div_removed_by_full_engine",
        html_input=(
            '<div class="class_s3">'
            '<div class="calibre1" id="a7VW">'
            '<img alt="" class="class_s2" src="image_rsrc85Z.jpg"/>'
            "</div>"
            '<p class="caption_s3w">Cora and Walter Musk</p>'
            "</div>"
        ),
        expected_contains=[],
        expected_absent=["Cora and Walter Musk"],
        chars_removed_min=1,
        lang="en",
    ),
]


class TestHtmlCleaningEngine:
    def _engine(self):
        from syntrive.adapters.htmlclean.engine import HtmlCleaningEngine
        return HtmlCleaningEngine()

    @pytest.mark.parametrize("case", ENGINE_CASES, ids=[c.id for c in ENGINE_CASES])
    def test_engine_case(self, case: RuleCase) -> None:
        engine = self._engine()
        cleaned, stats = engine.run(case.html_input, language=case.lang)
        soup = BeautifulSoup(cleaned, "html.parser")
        _apply_case(soup, stats.raw_chars - stats.cleaned_chars, case)

    def test_run_returns_string_and_stats(self) -> None:
        engine = self._engine()
        cleaned, stats = engine.run("<p>Hello world.</p>")
        assert isinstance(cleaned, str) and len(cleaned) > 0
        assert stats.raw_chars > 0
        assert stats.cleaned_chars >= 0

    def test_deletion_ratio_near_zero_for_clean_english(self) -> None:
        engine = self._engine()
        _, stats = engine.run(
            "<p>A clean English paragraph with no tables or excessive whitespace.</p>",
            language="en",
        )
        assert stats.deletion_ratio < 0.05, (
            f"Expected near-zero deletion for clean English; got {stats.deletion_ratio:.2%}"
        )

    def test_stats_by_rule_populated(self) -> None:
        engine = self._engine()
        _, stats = engine.run("<p>  Multiple   spaces  here.  </p>", language="en")
        if stats.deletion_ratio > 0:
            assert len(stats.removals_by_rule) > 0

    def test_idempotent_on_clean_input(self) -> None:
        engine = self._engine()
        html = "<p>Clean single sentence here.</p>"
        cleaned1, stats1 = engine.run(html, language="en")
        _, stats2 = engine.run(cleaned1, language="en")
        assert abs(stats2.deletion_ratio - stats1.deletion_ratio) < 0.01
