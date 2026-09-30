from __future__ import annotations

from syntrive.cli.collapsible_tree import (
    TreeGroup,
    TreeLeaf,
    _CARET_COLLAPSED,
    _CARET_EXPANDED,
    _CollapsibleControl,
)


def _control(*, second_book_collapsed: bool = False) -> _CollapsibleControl:
    groups = (
        TreeGroup(
            "Book A",
            ("book", 1),
            tuple(TreeLeaf(f"ch {i}", ("chapter", i)) for i in (10, 11, 12)),
        ),
        TreeGroup(
            "Book B",
            ("book", 2),
            tuple(TreeLeaf(f"ch {i}", ("chapter", i)) for i in (20, 21)),
            collapsed=second_book_collapsed,
        ),
    )
    return _CollapsibleControl(groups, pointer="»", show_description=False)


def _visible(ic: _CollapsibleControl) -> list:
    return [c.value for c in ic.filtered_choices]


class TestVisibleRows:
    def test_all_rows_visible_when_nothing_is_collapsed(self):
        ic = _control()
        assert _visible(ic) == [
            ("book", 1), ("chapter", 10), ("chapter", 11), ("chapter", 12),
            ("book", 2), ("chapter", 20), ("chapter", 21),
        ]

    def test_a_group_flagged_collapsed_hides_its_children_from_the_start(self):
        ic = _control(second_book_collapsed=True)
        assert _visible(ic) == [
            ("book", 1), ("chapter", 10), ("chapter", 11), ("chapter", 12),
            ("book", 2),
        ]

    def test_group_caret_tracks_fold_state(self):
        ic = _control()
        titles = {c.value: c.title for c in ic.filtered_choices if c.value in ic._group_values}
        assert titles[("book", 1)].startswith(_CARET_EXPANDED)
        ic.collapsed.add(("book", 1))
        titles = {c.value: c.title for c in ic.filtered_choices if c.value in ic._group_values}
        assert titles[("book", 1)].startswith(_CARET_COLLAPSED)


class TestCollapseExpandHandlers:
    def test_left_on_a_group_row_collapses_it(self):
        ic = _control()
        ic._point_at_value(("book", 2))
        assert ic.collapse_at_cursor() is True
        assert ("book", 2) in ic.collapsed
        assert _visible(ic) == [
            ("book", 1), ("chapter", 10), ("chapter", 11), ("chapter", 12), ("book", 2),
        ]

    def test_left_on_a_child_row_collapses_the_parent_and_moves_cursor_to_it(self):
        ic = _control()
        ic._point_at_value(("chapter", 11))
        assert ic.collapse_at_cursor() is True
        assert ("book", 1) in ic.collapsed
        assert ic.get_pointed_at().value == ("book", 1)

    def test_left_is_a_noop_on_an_already_collapsed_group(self):
        ic = _control(second_book_collapsed=True)
        ic._point_at_value(("book", 2))
        assert ic.collapse_at_cursor() is False

    def test_right_on_a_collapsed_group_expands_it(self):
        ic = _control(second_book_collapsed=True)
        ic._point_at_value(("book", 2))
        assert ic.expand_at_cursor() is True
        assert ("book", 2) not in ic.collapsed
        assert ("chapter", 20) in _visible(ic)

    def test_right_is_a_noop_on_an_expanded_group_or_a_child_row(self):
        ic = _control()
        ic._point_at_value(("book", 1))
        assert ic.expand_at_cursor() is False
        ic._point_at_value(("chapter", 10))
        assert ic.expand_at_cursor() is False

    def test_cursor_never_points_past_the_visible_list_after_a_collapse(self):
        ic = _control()
        ic._point_at_value(("chapter", 21))
        last_index = ic.pointed_at
        ic._point_at_value(("chapter", 10))
        ic.collapse_at_cursor()
        assert ic.pointed_at < len(ic.filtered_choices)
        assert ic.get_pointed_at().value == ("book", 1)
        assert last_index >= 0
