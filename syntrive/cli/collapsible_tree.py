from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from questionary import utils
from questionary.constants import DEFAULT_QUESTION_PREFIX, DEFAULT_SELECTED_POINTER
from questionary.prompts import common
from questionary.prompts.common import Choice, InquirerControl, Separator
from questionary.question import Question
from questionary.styles import merge_styles_default

logger = logging.getLogger(__name__)

DEFAULT_AUTO_COLLAPSE_THRESHOLD = 15

_CARET_EXPANDED = "▾ "
_CARET_COLLAPSED = "▸ "


@dataclass(frozen=True)
class TreeLeaf:
    title: str
    value: Any


@dataclass(frozen=True)
class TreeGroup:
    title: str
    value: Any
    leaves: Tuple[TreeLeaf, ...] = field(default_factory=tuple)
    collapsed: bool = False


class _CollapsibleControl(InquirerControl):
    def __init__(self, groups: Sequence[TreeGroup], **kwargs: Any) -> None:
        self._group_values: set = set()
        self._leaf_parent: dict = {}
        self._group_label: dict = {}
        self.collapsed: set = set()

        flat: List[Choice] = []
        for group in groups:
            self._group_values.add(group.value)
            self._group_label[group.value] = group.title
            if group.collapsed:
                self.collapsed.add(group.value)
            flat.append(Choice(title=self._group_title(group.value), value=group.value))
            for leaf in group.leaves:
                self._leaf_parent[leaf.value] = group.value
                flat.append(Choice(title=leaf.title, value=leaf.value))

        super().__init__(flat, **kwargs)

    def _group_title(self, group_value: Any) -> str:
        caret = _CARET_COLLAPSED if group_value in self.collapsed else _CARET_EXPANDED
        return f"{caret}{self._group_label[group_value]}"

    @property
    def filtered_choices(self):  # noqa: D401 - overrides questionary property
        visible: List[Choice] = []
        for choice in self.choices:
            parent = self._leaf_parent.get(getattr(choice, "value", None))
            if parent is not None and parent in self.collapsed:
                continue
            if choice.value in self._group_values:
                choice.title = self._group_title(choice.value)
            visible.append(choice)
        return visible

    def get_pointed_at(self) -> Choice:
        fc = self.filtered_choices
        if self.pointed_at >= len(fc):
            self.pointed_at = max(0, len(fc) - 1)
        return fc[self.pointed_at]

    def is_selection_a_separator(self) -> bool:
        return isinstance(self.get_pointed_at(), Separator)

    def is_selection_disabled(self) -> Any:
        return self.get_pointed_at().disabled

    def _point_at_value(self, value: Any) -> None:
        fc = self.filtered_choices
        for i, choice in enumerate(fc):
            if choice.value == value:
                self.pointed_at = i
                return
        self.pointed_at = max(0, min(self.pointed_at, len(fc) - 1))

    def collapse_at_cursor(self) -> bool:
        pointed = self.get_pointed_at().value
        group = pointed if pointed in self._group_values else self._leaf_parent.get(pointed)
        if group is None or group in self.collapsed:
            return False
        self.collapsed.add(group)
        self._point_at_value(group)
        logger.debug("checkbox_tree: collapsed group=%r", group)
        return True

    def expand_at_cursor(self) -> bool:
        pointed = self.get_pointed_at().value
        if pointed not in self._group_values or pointed not in self.collapsed:
            return False
        self.collapsed.discard(pointed)
        logger.debug("checkbox_tree: expanded group=%r", pointed)
        return True


def checkbox_tree(
    message: str,
    groups: Sequence[TreeGroup],
    *,
    qmark: str = DEFAULT_QUESTION_PREFIX,
    pointer: str = DEFAULT_SELECTED_POINTER,
    instruction: str | None = None,
    style: Style | None = None,
    use_jk_keys: bool = True,
    use_emacs_keys: bool = True,
    **kwargs: Any,
) -> Question:
    merged_style = merge_styles_default(
        [Style([("bottom-toolbar", "noreverse")]), style]
    )

    ic = _CollapsibleControl(tuple(groups), pointer=pointer, show_description=False)

    auto_collapsed = sum(1 for g in groups if g.collapsed)
    logger.debug(
        "checkbox_tree: opened groups=%d leaves=%d auto_collapsed=%d",
        len(groups), sum(len(g.leaves) for g in groups), auto_collapsed,
    )

    def get_prompt_tokens() -> List[Tuple[str, str]]:
        tokens = [("class:qmark", qmark), ("class:question", f" {message} ")]
        if ic.is_answered:
            n = len(ic.selected_options)
            tokens.append(
                ("class:answer", "done" if n == 0 else f"done ({n} selection{'s' if n != 1 else ''})")
            )
        else:
            tokens.append((
                "class:instruction",
                instruction
                or "(↑↓ move · ←/→ fold/unfold book · <space> select · <a> all · <i> invert · <enter> confirm)",
            ))
        return tokens

    def selected_values() -> List[Any]:
        return [c.value for c in ic.get_selected_values()]

    layout = common.create_inquirer_layout(ic, get_prompt_tokens, **kwargs)

    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _abort(event):
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(" ", eager=True)
    def _toggle(event):
        value = ic.get_pointed_at().value
        if value in ic.selected_options:
            ic.selected_options.remove(value)
        else:
            ic.selected_options.append(value)

    @bindings.add("i", eager=True)
    def _invert(event):
        ic.selected_options = [
            c.value
            for c in ic.choices
            if not isinstance(c, Separator) and c.value not in ic.selected_options and not c.disabled
        ]

    @bindings.add("a", eager=True)
    def _all(event):
        missing = [
            c.value
            for c in ic.choices
            if not isinstance(c, Separator) and c.value not in ic.selected_options and not c.disabled
        ]
        ic.selected_options = [] if not missing else ic.selected_options + missing

    def _down(event):
        ic.select_next()
        while not ic.is_selection_valid():
            ic.select_next()

    def _up(event):
        ic.select_previous()
        while not ic.is_selection_valid():
            ic.select_previous()

    bindings.add(Keys.Down, eager=True)(_down)
    bindings.add(Keys.Up, eager=True)(_up)
    if use_jk_keys:
        bindings.add("j", eager=True)(_down)
        bindings.add("k", eager=True)(_up)
    if use_emacs_keys:
        bindings.add(Keys.ControlN, eager=True)(_down)
        bindings.add(Keys.ControlP, eager=True)(_up)

    @bindings.add(Keys.Left, eager=True)
    def _collapse(event):
        ic.collapse_at_cursor()

    @bindings.add(Keys.Right, eager=True)
    def _expand(event):
        ic.expand_at_cursor()

    @bindings.add(Keys.ControlM, eager=True)
    def _submit(event):
        ic.is_answered = True
        event.app.exit(result=selected_values())

    @bindings.add(Keys.Any)
    def _ignore(event):
        pass

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merged_style,
            **utils.used_kwargs(kwargs, Application.__init__),
        )
    )
