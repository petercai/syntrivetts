from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal, Optional
from urllib.parse import parse_qsl, urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from syntrive.services import voice_library as vl
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import translate
from syntrive.webui.shared.state import WebRepo
from syntrive.webui.shared.templating import NavItem, render, request_lang

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voices"])

NAV = NavItem(key="voices", label_key="nav.voices", href="/voices", order=40,
              count=lambda web: vl.count_voices(web.db_path), section="library")

GENDERS = vl.VOICE_GENDERS
LANGUAGES = ("en", "zh")
LAN_FILTERS = ("en_zh", "all")
FILTER_KEYS = ("q", "lang", "gender", "flag")

Flag = Literal["", "in_use", "no_text", "fine_tuned"]


def _web(request: Request) -> WebRepo:
    return request.app.state.web


def _t(request: Request, key: str, **fields: object) -> str:
    return translate(request_lang(request), key, **fields)


def _back(notice: str, **query: object) -> Response:
    params = urlencode({k: v for k, v in {**query, "notice": notice}.items() if v not in (None, "")})
    return Response(status_code=200, headers={"HX-Redirect": f"/voices?{params}"})


def _filters(back: str) -> dict:
    return {k: v for k, v in parse_qsl(back) if k in FILTER_KEYS}


def _row(request: Request, voice_id: int) -> vl.VoiceRow:
    row = next((r for r in vl.list_voices(_web(request).db_path) if r.voice.db_id == voice_id), None)
    if row is None:
        raise WebError(404, _t(request, "voices.not_found", id=voice_id))
    return row


def _external(request: Request, external: str) -> Optional[Path]:
    if not external.strip():
        return None
    folder = Path(external.strip()).expanduser()
    if not folder.is_dir():
        raise WebError(422, _t(request, "voices.external_missing", path=external.strip()))
    return folder


def _href_builder(filters: dict):
    def href(**changes: object) -> str:
        params = {k: v for k, v in {**filters, **changes}.items() if v not in (None, "")}
        return "/voices" + ("?" + urlencode(params) if params else "")
    return href


@router.get("/voices", response_class=HTMLResponse)
def voices_page(
    request: Request,
    q: str = "", lang: str = "", gender: str = "", flag: Flag = "",
    voice: Optional[int] = None, mode: Literal["catalog", "import"] = "catalog",
    external: str = "", lan: Literal["en_zh", "all"] = "en_zh",
    notice: Optional[str] = None,
) -> HTMLResponse:
    web = _web(request)
    rows = vl.list_voices(web.db_path)
    selected = next((r for r in rows if r.voice.db_id == voice), None)
    context = {
        "rows": rows,
        "shown": vl.filter_voices(rows, q=q, lang=lang, gender=gender, flag=flag),
        "facets": vl.voice_facets(rows),
        "languages": sorted({r.voice.language for r in rows if r.voice.language}),
        "filters": {"q": q, "lang": lang, "gender": gender, "flag": flag},
        "href": _href_builder({"q": q, "lang": lang, "gender": gender, "flag": flag}),
        "selected": selected,
        "ref_text": vl.read_ref_text(selected.voice.abs_path) if selected and selected.file_present else "",
        "genders": GENDERS,
        "language_choices": sorted(set(LANGUAGES) | ({selected.voice.language} if selected and selected.voice.language else set())),
        "flags": vl.VOICE_FLAGS,
        "mode": mode,
        "notice": notice,
    }
    if mode == "import":
        folder = _external(request, external) if external else None
        context.update({
            "scan": vl.scan_new_voices(web.db_path, external=folder, lan_filter=lan),
            "external": external, "lan": lan, "lan_filters": LAN_FILTERS,
        })
    return render(request, "voices/page.html", context)


@router.get("/api/v1/voices/{voice_id:int}/audio")
def voice_audio(request: Request, voice_id: int) -> FileResponse:
    row = _row(request, voice_id)
    if not row.file_present:
        raise WebError(404, _t(request, "voices.file_missing", path=row.voice.stored_path))
    return FileResponse(row.voice.abs_path, media_type="audio/wav")


def _clean_fields(request: Request, gender: str, language: str) -> tuple[str, str]:
    try:
        return vl.clean_voice_fields(gender, language)
    except vl.VoiceFieldError as exc:
        raise WebError(422, _t(request, f"voices.bad_{exc.field}", value=exc.value)) from exc


@router.post("/api/v1/voices/{voice_id:int}")
def save_voice(
    request: Request, voice_id: int,
    gender: Annotated[str, Form()] = "", language: Annotated[str, Form()] = "",
    accent: Annotated[str, Form()] = "", tags: Annotated[str, Form()] = "",
    ref_text: Annotated[str, Form()] = "", back: Annotated[str, Form()] = "",
) -> Response:
    row = _row(request, voice_id)
    gender, language = _clean_fields(request, gender, language)
    changed = vl.update_reference_voice(_web(request).db_path, voice_id, gender=gender, language=language,
                                        accent=accent, tags=vl.parse_tags(tags))
    text_changed = vl.save_ref_text(row.voice.abs_path, ref_text) if row.file_present else False
    logger.info("webui_action: feature=voices action=save voice_id=%d fields_changed=%s ref_text_changed=%s",
                voice_id, changed, text_changed)
    return _back("saved" if changed or text_changed else "unchanged", **_filters(back), voice=voice_id)


@router.post("/api/v1/voices/{voice_id:int}/delete")
def delete_voice(request: Request, voice_id: int, back: Annotated[str, Form()] = "") -> Response:
    row = _row(request, voice_id)
    if row.in_use:
        raise WebError(409, _t(request, "voices.delete_in_use", books=" · ".join(row.used_by)))
    if not vl.delete_reference_voice(_web(request).db_path, voice_id):
        raise WebError(409, _t(request, "voices.delete_failed"))
    logger.info("webui_action: feature=voices action=delete voice_id=%d name=%s", voice_id, row.voice.name)
    return _back(f"deleted:{row.voice.name}", **_filters(back))


@router.post("/api/v1/voices/{voice_id:int}/transcribe", response_class=HTMLResponse)
def transcribe(request: Request, voice_id: int) -> HTMLResponse:
    row = _row(request, voice_id)
    if not row.file_present:
        raise WebError(404, _t(request, "voices.file_missing", path=row.voice.stored_path))
    try:
        text = vl.transcribe_reference_audio(row.voice.abs_path)
    except vl.AsrModelUnavailable as exc:
        logger.info("webui_action: feature=voices action=transcribe voice_id=%d result=unavailable", voice_id)
        raise WebError(409, str(exc)) from exc
    logger.info("webui_action: feature=voices action=transcribe voice_id=%d chars=%d", voice_id, len(text))
    return render(request, "voices/_ref_text.html", {"ref_text": text, "generated": True})


@router.post("/api/v1/voices/durations")
def update_durations(request: Request) -> Response:
    updated, missing = vl.update_reference_voice_durations(_web(request).db_path)
    logger.info("webui_action: feature=voices action=durations updated=%d missing=%d", updated, missing)
    return _back(f"durations:{updated}:{missing}")


@router.post("/api/v1/voices/import")
def import_voices(
    request: Request,
    path: Annotated[list[str], Form()] = [],  # noqa: B006
    external: Annotated[str, Form()] = "", lan: Annotated[Literal["en_zh", "all"], Form()] = "en_zh",
    gender: Annotated[str, Form()] = "", language: Annotated[str, Form()] = "",
    accent: Annotated[str, Form()] = "", tags: Annotated[str, Form()] = "",
) -> Response:
    wanted = set(path)
    if not wanted:
        raise WebError(422, _t(request, "voices.import_nothing"))
    gender, language = _clean_fields(request, gender, language)
    imported, matched = vl.import_scanned(_web(request).db_path, wanted, external=_external(request, external),
                                          lan_filter=lan, gender=gender, language=language, accent=accent, tags=tags)
    logger.info("webui_action: feature=voices action=import chosen=%d matched=%d imported=%d external=%s",
                len(wanted), matched, imported, external or None)
    return _back(f"imported:{imported}")


@router.post("/api/v1/voices/{voice_id:int}/relink")
def relink(
    request: Request, voice_id: int, external: Annotated[str, Form()] = "",
    lan: Annotated[Literal["en_zh", "all"], Form()] = "en_zh",
) -> Response:
    missing = vl.relink_scanned(_web(request).db_path, voice_id, external=_external(request, external), lan_filter=lan)
    if missing is None:
        raise WebError(409, _t(request, "voices.relink_none"))
    logger.info("webui_action: feature=voices action=relink voice_id=%d path=%s", voice_id, missing.replacement)
    return _back(f"relinked:{missing.name}", mode="import", external=external, lan=lan)
