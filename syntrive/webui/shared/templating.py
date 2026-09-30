from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional

from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, PrefixLoader, select_autoescape

from syntrive.webui.shared.i18n import LANG_COOKIE, pick_lang, translate, translate_or
from syntrive.webui.shared.state import WebRepo

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NavItem:
    key: str
    label_key: str
    href: str
    order: int = 100
    count: Optional[Callable[[WebRepo], object]] = None
    accent: bool = False
    section: str = "main"


def _format_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num} B"


def _format_time(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "—"


def build_environment() -> Environment:
    loaders = {"shared": FileSystemLoader(_WEBUI_DIR / "shared" / "templates")}
    for folder in sorted((_WEBUI_DIR / "features").glob("*/templates")):
        loaders[folder.parent.name] = FileSystemLoader(folder)
    env = Environment(loader=PrefixLoader(loaders), autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    env.filters["size"] = _format_size
    env.filters["when"] = _format_time
    return env


def request_lang(request: Request) -> str:
    return pick_lang(request.cookies.get(LANG_COOKIE), request.headers.get("accept-language"))


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _sidebar(request: Request) -> Callable[[], Mapping[str, object]]:
    def compute() -> Mapping[str, object]:
        from syntrive.io.server_registry import list_live_servers

        web: WebRepo = request.app.state.web
        items = []
        for item in request.app.state.nav:
            try:
                count = item.count(web) if item.count else None
            except Exception as exc:
                logger.warning("webui_nav_count_failed: key=%s error=%s", item.key, exc)
                count = None
            items.append({"item": item, "count": count, "active": request.url.path.startswith(item.href)})
        peers = {r.name: r.local_url for r in list_live_servers(web.repo_dir, exclude="webui")}
        return {"items": items, "peers": peers}

    return compute


def render(
    request: Request,
    name: str,
    context: Optional[Mapping[str, object]] = None,
    *,
    status_code: int = 200,
    headers: Optional[Mapping[str, str]] = None,
) -> HTMLResponse:
    lang = request_lang(request)
    env: Environment = request.app.state.jinja
    base = {
        "request": request,
        "lang": lang,
        "t": partial(translate, lang),
        "t_or": partial(translate_or, lang),
        "web": request.app.state.web,
        "server": request.app.state.server,
        "sidebar": _sidebar(request),
        "request_id": getattr(request.state, "request_id", ""),
        "htmx": is_htmx(request),
    }
    html = env.get_template(name).render({**base, **(context or {})})
    return HTMLResponse(html, status_code=status_code, headers=dict(headers or {}))
