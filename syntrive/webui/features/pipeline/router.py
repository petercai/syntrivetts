from __future__ import annotations

import logging
from typing import Annotated, Callable, Literal, Optional, TypeVar
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from syntrive.services import book_catalog, cover_images, tts_settings
from syntrive.services import step_decisions as decisions
from syntrive.services.job_lease import JobLeaseConflict, find_blocking_lease
from syntrive.services.job_service import JobService
from syntrive.services.pipeline_service import (
    DECISION_OPTIONS,
    PIPELINE_STEPS,
    OnExisting,
    PipelineStep,
    RunAlreadyActive,
    StepMode,
    StepOptions,
    StepRunError,
    StepRunRegistry,
    StepState,
    check_runnable,
    existing_output,
    load_job,
)
from syntrive.webui.shared.errors import WebError
from syntrive.webui.shared.i18n import translate
from syntrive.webui.shared.state import HOLDER_KIND, WebRepo
from syntrive.webui.shared.templating import render, request_lang
from syntrive.workflow.engine import WorkflowStep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pipeline"])

Tab = Literal["step", "chapters", "metadata", "activity"]
T = TypeVar("T")

SHORT_CHAPTER_CHARS = 200
DELETION_THRESHOLD = 0.35
TTS_HEADER_ORDER = ("engine", "model", "device", "language", "offline_mode")


def _web(request: Request) -> WebRepo:
    return request.app.state.web


def _runs(request: Request) -> StepRunRegistry:
    return request.app.state.runs


def _t(request: Request, key: str, **fields: object) -> str:
    return translate(request_lang(request), key, **fields)


def _parse_step(request: Request, value: Optional[str]) -> Optional[WorkflowStep]:
    if not value:
        return None
    try:
        step = WorkflowStep(value)
    except ValueError:
        step = None
    if step not in PIPELINE_STEPS:
        raise WebError(422, _t(request, "pipeline.unknown_step", step=value))
    return step


def _require_row(request: Request, job_id: int) -> book_catalog.BookRow:
    row = book_catalog.get_book_row(_web(request).db_path, job_id)
    if row is None:
        raise WebError(404, _t(request, "pipeline.refused_not_found"))
    return row


def _can_restart(row: book_catalog.BookRow, selected: PipelineStep) -> bool:
    return not row.archived and selected.state == StepState.DONE


def _can_run(row: book_catalog.BookRow, selected: PipelineStep) -> bool:
    decided_here = selected.step in DECISION_OPTIONS
    return not row.archived and selected.state == StepState.CURRENT and (selected.mode == StepMode.RUN or decided_here)


def _commands(web: WebRepo) -> dict[str, str]:
    repo = f'"{web.repo_dir}"'
    return {
        "tui": f"uv run python syntrive.py -r {repo}",
        "task": f"uv run python tts_task.py -r {repo}",
        "batch": f"uv run python tts_batch.py -r {repo}",
    }


def _raise_if_leased(web: WebRepo, job_id: int) -> None:
    conflict = find_blocking_lease(web.db_path, [job_id])
    if conflict is not None:
        raise conflict


def _decide(request: Request, call: Callable[[], T]) -> T:
    try:
        return call()
    except decisions.DecisionError as exc:
        status = 404 if exc.code == "not_found" else 422
        raise WebError(status, _t(request, f"pipeline.decision_{exc.code}")) from exc


def _page_url(job_id: int, step: WorkflowStep, **params: str) -> str:
    query = urlencode({"step": step.value, **{k: v for k, v in params.items() if v}})
    return f"/pipeline/{job_id}?{query}"


def _saved(job_id: int, step: WorkflowStep, what: str, result: decisions.DecisionResult) -> Response:
    notice = f"stale:{result.rerun_from.value}" if result.rerun_from else ("saved" if result.changed else "unchanged")
    logger.info("webui_action: feature=pipeline action=decide:%s job_id=%d result=%s", what, job_id, notice)
    return Response(status_code=200, headers={"HX-Redirect": _page_url(job_id, step, notice=notice)})


def _decision_context(request: Request, web: WebRepo, job_id: int, step: WorkflowStep, fmt: Optional[str]) -> dict:
    db = web.db_path
    if step == WorkflowStep.TRANSCRIPT_MERGE:
        return {"merge": _decide(request, lambda: decisions.read_merge_settings(db, job_id)),
                "selection": decisions.read_chapter_selection(db, job_id),
                "short_chars": SHORT_CHAPTER_CHARS}
    if step == WorkflowStep.TRANSCRIPT_CLEAN:
        results = sorted(decisions.read_clean_results(db, job_id), key=lambda r: -(r.deletion_ratio or 0))
        return {"rules": decisions.read_cleaning_rules(db, job_id), "results": results,
                "raw_total": sum(r.raw_chars or 0 for r in results), "clean_total": sum(r.cleaned_chars or 0 for r in results),
                "selection": decisions.read_chapter_selection(db, job_id), "threshold": DELETION_THRESHOLD}
    if step == WorkflowStep.TRANSCRIPT_TEXT:
        from syntrive.pipeline.text_extraction_stage import (
            DEFAULT_TEXT_EXTRACTION_FORMATS,
            DEFAULT_TEXT_EXTRACTION_PRIMARY,
            TEXT_EXTRACTION_FORMATS,
        )
        return {"text_formats": TEXT_EXTRACTION_FORMATS, "text_defaults": DEFAULT_TEXT_EXTRACTION_FORMATS,
                "text_primary": DEFAULT_TEXT_EXTRACTION_PRIMARY}
    if step == WorkflowStep.TRANSCRIPT_REVIEW:
        review = decisions.read_transcript_review(db, job_id)
        available = [f.name for f in review.formats if f.available]
        chosen = fmt if fmt in available else (review.selected if review.selected in available else
                                               (review.default if review.default in available else (available[0] if available else None)))
        current = next((f for f in review.formats if f.name == chosen), None)
        return {"review": review, "review_fmt": chosen, "review_files": current.files if current else ()}
    if step == WorkflowStep.TTS_CONFIG:
        return _tts_context(request, web, job_id)
    return {}


def _tts_context(request: Request, web: WebRepo, job_id: int) -> dict:
    row = _require_row(request, job_id)
    reached = row.pipeline.get(WorkflowStep.TTS_CONFIG).state != StepState.LATER
    settings = _settings(request, lambda: tts_settings.read_tts_settings(web.db_path, job_id))
    if reached and not row.archived and (not settings.exists or not settings.voices):
        try:
            tts_settings.prepare_tts_settings(web.db_path, job_id, holder_kind=HOLDER_KIND)
            settings = tts_settings.read_tts_settings(web.db_path, job_id)
        except JobLeaseConflict as exc:
            logger.info("webui_tts_prepare_skipped: job_id=%d reason=%s", job_id, exc)
    by_tab = {tab: [p for p in tts_settings.PARAMS if p.tab == tab] for tab in tts_settings.SECTION_ORDER}
    return {
        "tts": settings,
        "tts_header": sorted(by_tab[tts_settings.TAB_BASIC], key=lambda p: TTS_HEADER_ORDER.index(p.field)),
        "tts_output": by_tab[tts_settings.TAB_OUTPUT],
        "tts_tuning": by_tab[tts_settings.TAB_TUNING],
        "tts_ref_labels": {r.id: r.label for r in settings.references},
        "tts_genders": sorted({r.gender for r in settings.references if r.gender}),
    }


def _settings(request: Request, call: Callable[[], T]) -> T:
    try:
        return call()
    except tts_settings.SettingsError as exc:
        status = 404 if exc.code == "not_found" else 422
        label = tts_settings.PARAMS_BY_FIELD[exc.field].label if exc.field in tts_settings.PARAMS_BY_FIELD else ""
        raise WebError(status, _t(request, f"pipeline.tts_{exc.code}", field=label, detail=str(exc))) from exc


@router.get("/pipeline/{job_id}", response_class=HTMLResponse)
def book_page(
    request: Request, job_id: int, step: Optional[str] = None, tab: Tab = "step",
    notice: Optional[str] = None, fmt: Optional[str] = None,
) -> HTMLResponse:
    web = _web(request)
    row = _require_row(request, job_id)
    view = row.pipeline
    selected_step = _parse_step(request, step) or view.current or PIPELINE_STEPS[-1]
    selected = view.get(selected_step)
    context = {
        "row": row,
        "view": view,
        "selected": selected,
        "tab": tab,
        "run": _runs(request).get(web.db_path, job_id),
        "can_run": _can_run(row, selected),
        "can_restart": _can_restart(row, selected),
        "commands": _commands(web),
        "notice": notice,
        "chapters": book_catalog.list_chapters(web.db_path, job_id) if tab == "chapters" else (),
        "activity": book_catalog.list_activity(web.db_path, job_id) if tab == "activity" else (),
    }
    if tab == "step":
        context.update(_decision_context(request, web, job_id, selected_step, fmt))
    if tab == "metadata":
        context.update(_metadata_context(web, job_id))
    return render(request, "pipeline/page.html", context)


def _metadata_context(web: WebRepo, job_id: int, import_result=None) -> dict:
    return {
        "meta": book_catalog.read_book_metadata(web.db_path, job_id),
        "languages": tts_settings.PARAMS_BY_FIELD["language"].choices,
        "import_result": import_result,
    }


def _options(text_options: Optional[str], text_format: list[str],
             text_primary: Optional[str], transcript_path: Optional[str]) -> StepOptions:
    return StepOptions(
        text_formats=frozenset(text_format) if text_options else None,
        text_primary=text_primary or None,
        transcript_path=transcript_path or None,
    )


def _resubmit(options: StepOptions) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if options.text_formats is not None:
        fields.append(("text_options", "1"))
        fields.extend(("text_format", f) for f in sorted(options.text_formats))
    if options.text_primary:
        fields.append(("text_primary", options.text_primary))
    if options.transcript_path:
        fields.append(("transcript_path", options.transcript_path))
    return fields


@router.post("/api/v1/pipeline/{job_id}/run", response_class=HTMLResponse)
def start_run(
    request: Request, job_id: int,
    step: Annotated[str, Form()],
    on_existing: Annotated[OnExisting, Form()] = OnExisting.ASK,
    text_options: Annotated[Optional[str], Form()] = None,
    text_format: Annotated[list[str], Form()] = [],  # noqa: B006 - FastAPI form default
    text_primary: Annotated[Optional[str], Form()] = None,
    transcript_path: Annotated[Optional[str], Form()] = None,
) -> HTMLResponse:
    web = _web(request)
    wstep = _parse_step(request, step)
    options = _options(text_options, text_format, text_primary, transcript_path)
    try:
        check_runnable(load_job(web.db_path, job_id), wstep, options)
    except StepRunError as exc:
        raise WebError(404 if exc.code == "not_found" else 422, _t(request, f"pipeline.refused_{exc.code}")) from exc
    _raise_if_leased(web, job_id)

    if on_existing == OnExisting.ASK:
        desc = existing_output(web.db_path, job_id, wstep)
        if desc is not None:
            logger.info("webui_action: feature=pipeline action=run job_id=%d step=%s result=needs_confirm", job_id, wstep.value)
            return render(request, "pipeline/_run_panel.html",
                          {"job_id": job_id, "step": wstep, "confirm": desc, "resubmit": _resubmit(options)})

    try:
        run = _runs(request).start(web.db_path, job_id, wstep, on_existing=on_existing, holder_kind=HOLDER_KIND, options=options)
    except RunAlreadyActive as exc:
        raise WebError(409, _t(request, "pipeline.already_running")) from exc
    logger.info("webui_action: feature=pipeline action=run job_id=%d step=%s on_existing=%s result=started", job_id, wstep.value, on_existing.value)
    return render(request, "pipeline/_run_panel.html", {"job_id": job_id, "step": wstep, "run": run})


@router.get("/api/v1/pipeline/{job_id}/run", response_class=HTMLResponse)
def run_status(request: Request, job_id: int) -> Response:
    run = _runs(request).get(_web(request).db_path, job_id)
    if run is None:
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    if not run.running:
        return Response(status_code=204, headers={"HX-Redirect": f"/pipeline/{job_id}?step={run.step.value}"})
    return render(request, "pipeline/_run_panel.html", {"job_id": job_id, "step": run.step, "run": run})


def _restart(request: Request, job_id: int, step: str) -> tuple[WebRepo, WorkflowStep]:
    web = _web(request)
    row = _require_row(request, job_id)
    wstep = _parse_step(request, step)
    if not _can_restart(row, row.pipeline.get(wstep)):
        raise WebError(422, _t(request, "pipeline.restart_refused"))
    running = _runs(request).get(web.db_path, job_id)
    if running is not None and running.running:
        raise WebError(409, _t(request, "pipeline.already_running"))
    _raise_if_leased(web, job_id)
    JobService(web.db_path, holder_kind=HOLDER_KIND).reset_job_to_step(job_id, wstep)
    return web, wstep


@router.post("/api/v1/pipeline/{job_id}/restart")
def restart(request: Request, job_id: int, step: Annotated[str, Form()]) -> Response:
    _, wstep = _restart(request, job_id, step)
    logger.info("webui_action: feature=pipeline action=restart job_id=%d step=%s result=ok", job_id, wstep.value)
    return Response(status_code=200, headers={"HX-Redirect": _page_url(job_id, wstep)})


@router.post("/api/v1/pipeline/{job_id}/rerun")
def rerun(request: Request, job_id: int, step: Annotated[str, Form()]) -> Response:
    web, wstep = _restart(request, job_id, step)
    try:
        check_runnable(load_job(web.db_path, job_id), wstep)
        _runs(request).start(web.db_path, job_id, wstep, on_existing=OnExisting.OVERWRITE, holder_kind=HOLDER_KIND)
    except StepRunError as exc:
        logger.info("webui_action: feature=pipeline action=rerun job_id=%d step=%s result=restarted_only code=%s", job_id, wstep.value, exc.code)
        return Response(status_code=200, headers={"HX-Redirect": _page_url(job_id, wstep)})
    logger.info("webui_action: feature=pipeline action=rerun job_id=%d step=%s result=started", job_id, wstep.value)
    return Response(status_code=200, headers={"HX-Redirect": _page_url(job_id, wstep)})


@router.post("/api/v1/pipeline/{job_id}/merge-settings")
def save_merge_settings(
    request: Request, job_id: int,
    override: Annotated[str, Form()] = "",
    reset_per_volume: Annotated[Optional[str], Form()] = None,
) -> Response:
    web = _web(request)
    result = _decide(request, lambda: decisions.save_merge_settings(
        web.db_path, job_id, override=override or None, reset_per_volume=bool(reset_per_volume), holder_kind=HOLDER_KIND,
    ))
    return _saved(job_id, WorkflowStep.TRANSCRIPT_MERGE, "merge_settings", result)


@router.post("/api/v1/pipeline/{job_id}/exclusions")
def save_exclusions(request: Request, job_id: int, include: Annotated[list[str], Form()] = []) -> Response:  # noqa: B006
    web = _web(request)
    selection = decisions.read_chapter_selection(web.db_path, job_id)
    excluded = {c.chapter_id for c in selection.chapters} - set(include)
    result = _decide(request, lambda: decisions.save_exclusions(web.db_path, job_id, excluded, holder_kind=HOLDER_KIND))
    return _saved(job_id, WorkflowStep.TRANSCRIPT_MERGE, "exclusions", result)


@router.get("/api/v1/pipeline/{job_id}/chapter-text", response_class=HTMLResponse)
def chapter_text(request: Request, job_id: int, chapter_id: str) -> HTMLResponse:
    web = _web(request)
    text, truncated = _decide(request, lambda: decisions.read_chapter_text(web.db_path, job_id, chapter_id))
    chapter = next((c for c in decisions.read_chapter_selection(web.db_path, job_id).chapters if c.chapter_id == chapter_id), None)
    return render(request, "pipeline/_chapter_drawer.html",
                  {"job_id": job_id, "chapter": chapter, "chapter_id": chapter_id, "text": text, "truncated": truncated})


@router.post("/api/v1/pipeline/{job_id}/cleaning-rules")
def save_cleaning_rules(request: Request, job_id: int, rule: Annotated[list[str], Form()] = []) -> Response:  # noqa: B006
    web = _web(request)
    on = set(rule)
    wanted = {r.name: r.name in on for r in _decide(request, lambda: decisions.read_cleaning_rules(web.db_path, job_id))}
    result = _decide(request, lambda: decisions.save_cleaning_rules(web.db_path, job_id, wanted, holder_kind=HOLDER_KIND))
    return _saved(job_id, WorkflowStep.TRANSCRIPT_CLEAN, "cleaning_rules", result)


@router.get("/api/v1/pipeline/{job_id}/transcript", response_class=HTMLResponse)
def transcript(request: Request, job_id: int, fmt: str, file: str) -> HTMLResponse:
    web = _web(request)
    text, truncated = _decide(request, lambda: decisions.read_transcript_text(web.db_path, job_id, fmt, file))
    return render(request, "pipeline/_reader.html",
                  {"fmt": fmt, "file": file, "lines": decisions.split_markers(text), "truncated": truncated})


def _tts_changes(form, settings: "tts_settings.TtsSettings") -> tuple[dict, list]:
    changes: dict = {}
    for param in tts_settings.PARAMS:
        if param.input_type == tts_settings.InputType.CHECKBOX:
            changes[param.field] = param.field in form
        elif param.field in form:
            changes[param.field] = form.get(param.field)
    bindings = []
    for voice in settings.voices:
        raw = form.get(f"voice_{voice.id}", "")
        bindings.append(tts_settings.VoiceBinding(voice.id, int(raw) if str(raw).isdigit() else None, f"exclude_{voice.id}" in form))
    return changes, bindings


@router.post("/api/v1/pipeline/{job_id}/tts-settings")
async def save_tts_settings_route(request: Request, job_id: int) -> Response:
    form = await request.form()
    web = _web(request)

    def save() -> tuple[bool, bool]:
        settings = _settings(request, lambda: tts_settings.read_tts_settings(web.db_path, job_id))
        changes, bindings = _tts_changes(form, settings)
        result = _settings(request, lambda: tts_settings.save_tts_settings(
            web.db_path, job_id, changes, bindings, holder_kind=HOLDER_KIND))
        started = False
        if form.get("continue"):
            try:
                check_runnable(load_job(web.db_path, job_id), WorkflowStep.TTS_CONFIG)
            except StepRunError as exc:
                raise WebError(422, _t(request, f"pipeline.refused_{exc.code}")) from exc
            _raise_if_leased(web, job_id)
            _runs(request).start(web.db_path, job_id, WorkflowStep.TTS_CONFIG, on_existing=OnExisting.OVERWRITE, holder_kind=HOLDER_KIND)
            started = True
        return result.changed, started

    changed, started = await run_in_threadpool(save)
    logger.info("webui_action: feature=pipeline action=tts_settings job_id=%d changed=%s continue=%s", job_id, changed, started)
    notice = "" if started else ("saved" if changed else "unchanged")
    return Response(status_code=200, headers={"HX-Redirect": _page_url(job_id, WorkflowStep.TTS_CONFIG, notice=notice)})


@router.get("/api/v1/pipeline/{job_id}/tts-models", response_class=HTMLResponse)
def tts_models(request: Request, job_id: int, engine: str) -> HTMLResponse:
    if engine not in (tts_settings.PARAMS_BY_FIELD["engine"].choices or ()):
        raise WebError(422, _t(request, "pipeline.tts_bad_choice", field="TTS Engine", detail=engine))
    return render(request, "pipeline/_tts_models.html",
                  {"model_choices": tts_settings.model_choices_for(engine), "current_model": ""})


@router.get("/api/v1/pipeline/{job_id}/voices/{reference_voice_id}/audio")
def voice_audio(request: Request, job_id: int, reference_voice_id: int) -> FileResponse:
    path = _settings(request, lambda: tts_settings.reference_voice_file(_web(request).db_path, reference_voice_id))
    return FileResponse(path)


@router.post("/api/v1/pipeline/{job_id}/metadata")
def save_metadata(
    request: Request, job_id: int,
    title: Annotated[str, Form()] = "",
    author: Annotated[str, Form()] = "",
    language: Annotated[str, Form()] = "",
    cover: Annotated[str, Form()] = "",
) -> Response:
    web = _web(request)
    meta = book_catalog.read_book_metadata(web.db_path, job_id)
    if meta is None:
        raise WebError(404, _t(request, "pipeline.refused_not_found"))
    if not title.strip():
        raise WebError(422, _t(request, "pipeline.meta_title_required"))
    if language and language not in (tts_settings.PARAMS_BY_FIELD["language"].choices or ()):
        raise WebError(422, _t(request, "pipeline.meta_bad_language", language=language))
    if cover and cover not in {img.rel_path for img in meta.images}:
        raise WebError(422, _t(request, "pipeline.meta_bad_cover"))

    service = JobService(web.db_path, holder_kind=HOLDER_KIND)
    changed = []
    if language and language != meta.language:
        service.update_book_language(job_id, language)
        changed.append("language")
    if title.strip() != meta.title:
        service.update_book_title(job_id, title)
        changed.append("title")
    if author.strip() != meta.author:
        service.update_book_author(job_id, author)
        changed.append("author")
    if cover and cover != meta.cover_rel:
        service.update_book_cover(job_id, cover)
        changed.append("cover")
    logger.info("webui_action: feature=pipeline action=metadata job_id=%d changed=%s", job_id, changed)
    url = f"/pipeline/{job_id}?{urlencode({'tab': 'metadata', 'notice': 'saved' if changed else 'unchanged'})}"
    return Response(status_code=200, headers={"HX-Redirect": url})


@router.post("/api/v1/pipeline/{job_id}/images", response_class=HTMLResponse)
async def import_images(request: Request, job_id: int) -> HTMLResponse:
    form = await request.form()
    uploads = [(f.filename or "upload", await f.read()) for f in form.getlist("files") if getattr(f, "filename", "")]
    urls = [line for line in str(form.get("urls", "")).splitlines() if line.strip()]
    if not uploads and not urls:
        raise WebError(422, _t(request, "pipeline.meta_nothing_to_import"))
    web = _web(request)
    try:
        result = await run_in_threadpool(cover_images.import_cover_images, web.db_path, job_id, uploads, urls)
    except LookupError as exc:
        raise WebError(404, _t(request, "pipeline.refused_not_found")) from exc
    logger.info("webui_action: feature=pipeline action=import_images job_id=%d added=%d refused=%d",
                job_id, len(result.added), len(result.refused))
    return render(request, "pipeline/_cover_gallery.html", {"job_id": job_id, **_metadata_context(web, job_id, result)})


@router.get("/api/v1/pipeline/{job_id}/images/{name}")
def book_image(request: Request, job_id: int, name: str) -> FileResponse:
    meta = book_catalog.read_book_metadata(_web(request).db_path, job_id)
    image = next((i for i in (meta.images if meta else ()) if i.name == name and i.exists), None)
    if image is None:
        raise WebError(404, _t(request, "pipeline.meta_no_image"))
    return FileResponse(image.path)
