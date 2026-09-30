from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FLAT_GROUP_ID = "main"


@dataclass(frozen=True)
class GroupEntry:
    group_id: str
    title: str
    href_set: frozenset


@dataclass(frozen=True)
class TocNodeInfo:
    depth: int
    is_container: bool


_DEFAULT_NODE_INFO = TocNodeInfo(depth=0, is_container=False)


@dataclass
class HtmlGroupPlan:
    mode: str
    groups: list[GroupEntry] = field(default_factory=list)
    node_info_by_order: dict[int, TocNodeInfo] = field(default_factory=dict)

    def get_group_for_href(self, href: str) -> tuple[str, str]:
        for grp in self.groups:
            if href in grp.href_set:
                return grp.group_id, grp.title
        return _FLAT_GROUP_ID, ""

    def get_node_info(self, order: int) -> TocNodeInfo:
        return self.node_info_by_order.get(order, _DEFAULT_NODE_INFO)


class TocHierarchyPlanner:
    def __init__(self, epub_path: Path) -> None:
        self._epub_path = epub_path

    def plan(self) -> HtmlGroupPlan:
        try:
            from ebooklib import epub as eblib
        except ImportError:
            raise RuntimeError("ebooklib is required: pip install EbookLib") from None

        logger.info("TocHierarchyPlanner: reading TOC from %s", self._epub_path)
        book = eblib.read_epub(str(self._epub_path), {"ignore_ncx": True})
        plan = _build_plan(book.toc)
        logger.info(
            "TocHierarchyPlanner: mode=%s groups=%d max_depth=%d",
            plan.mode,
            len(plan.groups),
            max((info.depth for info in plan.node_info_by_order.values()), default=0),
        )
        return plan


def _build_plan(toc) -> HtmlGroupPlan:
    volume_info = _extract_volume_groups(toc)
    node_info_by_order = {
        idx: TocNodeInfo(depth=depth, is_container=is_container)
        for idx, (_, _, depth, is_container) in enumerate(_flatten_toc_with_depth(toc))
    }

    if not volume_info:
        all_hrefs = frozenset(
            _file_href(href)
            for _, href in _flatten_toc_hrefs(toc)
            if href
        )
        return HtmlGroupPlan(
            mode="flat",
            groups=[GroupEntry(group_id=_FLAT_GROUP_ID, title="", href_set=all_hrefs)],
            node_info_by_order=node_info_by_order,
        )

    groups = []
    for idx, (title, child_hrefs) in enumerate(volume_info):
        group_id = f"vol_{idx + 1:04d}"
        href_set = frozenset(_file_href(h) for h in child_hrefs if h)
        groups.append(GroupEntry(group_id=group_id, title=title, href_set=href_set))
        logger.debug("TocHierarchyPlanner: group %s = %r (%d hrefs)", group_id, title, len(href_set))

    return HtmlGroupPlan(mode="volume_split", groups=groups, node_info_by_order=node_info_by_order)


def _extract_volume_groups(toc) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []

    for item in toc:
        if isinstance(item, tuple) and len(item) >= 2:
            head, children = item[0], item[1]
            child_list = list(children) if isinstance(children, (list, tuple)) else []
            if child_list:
                child_hrefs = [href for _, href in _flatten_toc_hrefs(child_list) if href]
                groups.append((getattr(head, "title", None) or "Section", child_hrefs))

    return groups


def _flatten_toc_hrefs(toc) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in toc:
        if isinstance(item, tuple) and len(item) >= 2:
            head, children = item[0], item[1]
            if hasattr(head, "href") and head.href:
                result.append((getattr(head, "title", "") or "", head.href))
            if isinstance(children, (list, tuple)):
                result.extend(_flatten_toc_hrefs(children))
        elif hasattr(item, "href") and item.href:
            result.append((getattr(item, "title", "") or "", item.href))
    return result


def _flatten_toc_with_depth(
    toc, depth: int = 0
) -> list[tuple[str, str, int, bool]]:
    result: list[tuple[str, str, int, bool]] = []
    for item in toc:
        if isinstance(item, tuple) and len(item) >= 2:
            head, children = item[0], item[1]
            child_list = list(children) if isinstance(children, (list, tuple)) else []
            if hasattr(head, "href") and head.href:
                result.append((getattr(head, "title", "") or "", head.href, depth, bool(child_list)))
            if child_list:
                result.extend(_flatten_toc_with_depth(child_list, depth + 1))
        elif hasattr(item, "href") and item.href:
            result.append((getattr(item, "title", "") or "", item.href, depth, False))
    return result


def _file_href(href: str) -> str:
    return href.split("#", 1)[0] if href else ""
