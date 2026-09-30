from __future__ import annotations

import logging

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_MAX_HEADERS = 6
_MAX_SAMPLE_CELLS = 4


class TableSummaryRule:
    name = "table_summary"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        net_removed = 0
        for table in soup.find_all("table"):
            table_text_len = len(table.get_text())
            summary = _summarize_table(table)
            summary_tag = soup.new_tag("p")
            summary_tag["data-table-summary"] = "true"
            summary_tag.string = summary
            table.replace_with(summary_tag)
            net_removed += table_text_len - len(summary)
            logger.debug("TableSummaryRule: replaced table with: %s", summary[:80])

        return soup, max(0, net_removed)


def _summarize_table(table: Tag) -> str:
    headers = _extract_headers(table)
    data_rows = [r for r in table.find_all("tr") if r.find("td")]
    n_rows = len(data_rows)
    n_cols = _col_count(table, headers)

    if n_rows == 0 and not headers:
        return "An empty table."

    if headers:
        h_str = ", ".join(headers[:_MAX_HEADERS])
        if len(headers) > _MAX_HEADERS:
            h_str += f" and {len(headers) - _MAX_HEADERS} more"
        return f"A table with {n_rows} row{'s' if n_rows != 1 else ''}. Columns: {h_str}."

    if data_rows:
        cells = [td.get_text(strip=True) for td in data_rows[0].find_all("td")]
        sample = ", ".join(c for c in cells[:_MAX_SAMPLE_CELLS] if c)
        suffix = f" First row: {sample}." if sample else ""
        return f"A table with {n_rows} row{'s' if n_rows != 1 else ''} and {n_cols} column{'s' if n_cols != 1 else ''}.{suffix}"

    return f"A table with {n_cols} column{'s' if n_cols != 1 else ''}."


def _extract_headers(table: Tag) -> list[str]:
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    return [h for h in headers if h]


def _col_count(table: Tag, headers: list[str]) -> int:
    if headers:
        return len(headers)
    rows = table.find_all("tr")
    return max(
        (len(r.find_all(["td", "th"])) for r in rows),
        default=0,
    )
