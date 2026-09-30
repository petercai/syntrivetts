from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from syntrive.io import server_registry
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import LANG_COOKIE, translate
from syntrive.webui.shared.state import HEALTH_PATH, SERVER_NAME, NotARepoError, open_web_repo
from syntrive.webui.shared.templating import request_lang

logger = logging.getLogger(__name__)

router = APIRouter(tags=["repo"])

_ONE_YEAR = 365 * 24 * 3600


@router.post("/api/v1/repo/switch")
def switch_repo(request: Request, repo_dir: Annotated[str, Form()]) -> Response:
    lang = request_lang(request)
    current = request.app.state.web
    try:
        target = open_web_repo(Path(repo_dir.strip().strip('"')))
    except NotARepoError as exc:
        raise WebError(422, translate(lang, "repo.not_a_repo", path=repo_dir)) from exc
    if target.repo_dir == current.repo_dir:
        return Response(status_code=200, headers={"HX-Redirect": "/books"})
    if request.app.state.runs.any_running():
        raise WebError(409, translate(lang, "repo.busy"))
    live = server_registry.find_live_server(target.repo_dir, SERVER_NAME)
    if live is not None:
        logger.info("webui_repo_switch_refused: old=%s new=%s existing=%s", current.repo_dir, target.repo_dir, live.local_url)
        raise WebError(409, translate(lang, "repo.already_served", url=live.local_url))

    if request.app.state.register:
        server = request.app.state.server
        server_registry.remove_record(current.repo_dir, SERVER_NAME)
        server_registry.write_record(target.repo_dir, SERVER_NAME, server.port, host=server.host, health_path=HEALTH_PATH)
    request.app.state.web = target
    logger.info("webui_action: feature=repo action=switch old=%s new=%s result=ok", current.repo_dir, target.repo_dir)
    return Response(status_code=200, headers={"HX-Redirect": "/books"})


@router.post("/api/v1/repo/lang")
def set_lang(lang: Annotated[Literal["en", "zh"], Form()]) -> Response:
    response = Response(status_code=204, headers={"HX-Refresh": "true"})
    response.set_cookie(LANG_COOKIE, lang, max_age=_ONE_YEAR, samesite="lax")
    logger.info("webui_action: feature=repo action=lang value=%s result=ok", lang)
    return response
