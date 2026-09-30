from __future__ import annotations

import logging
import time
from typing import Optional

import anyio
from mcp.server.mcpserver import Context, MCPServer

from syntrive.mcp.envelope import Envelope, destructive, fail, ok
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import DESTRUCTIVE, READ, WRITE
from syntrive.services import batch_runner
from syntrive.services import model_library as ml

logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 50.0


def _row_view(row: ml.RowModel) -> dict:
    return {"engine": row.engine, "item_id": row.model_id, "label": row.label, "kind": row.kind,
            "status": ml.row_status(row), "default": row.is_default, "downloaded": row.downloaded,
            "size_bytes": row.size_bytes, "approx_gb": row.approx_gb, "revision": row.revision,
            "updated_at": row.updated_at}


def _task_view(task: Optional[ml.TaskView]) -> Optional[dict]:
    if task is None:
        return None
    return {"task_id": task.id, "key": task.key, "name": task.name, "state": task.state, "summary": task.summary,
            "log_tail": list(task.log[-20:]),
            "seconds": round((task.finished_at or time.time()) - (task.started_at or task.submitted_at), 1)}


def _require_network(action: str) -> None:
    if batch_runner.offline_default().locked:
        logger.info("mcp_action: tool=model_%s result=refused reason=offline", action)
        raise fail("refused", "Offline mode is on in this server's environment (SYNTRIVE_TTS_OFFLINE=1); "
                              "this action needs the network.")


def register(server: MCPServer, state: McpState) -> None:
    tasks = ml.ModelTasks()

    def _row(engine: str, item_id: str) -> ml.RowModel:
        row = ml.find_row(engine, item_id)
        if row is None:
            raise fail("not_found", f"{engine}/{item_id} is not in the model catalog (model_catalog lists them).")
        return row

    def _queued(view: Optional[ml.TaskView], action: str, item: str, **evidence) -> Envelope:
        if view is None:
            raise fail("conflict", f"{item} already has an action queued or running (model_tasks).")
        logger.info("mcp_action: tool=model_%s item=%s task_id=%d", action, item, view.id)
        return ok({"task": _task_view(view)}, evidence={"task_id": view.id, **evidence},
                  warnings=["Runs in the background: follow it with model_task_status(task_id, wait_seconds=50)."])

    @server.tool(annotations=READ)
    def model_catalog(engine: Optional[str] = None) -> Envelope:
        """Every TTS engine with its models and resources: status (ready / missing / attention / unverified), size,
        revision, whether its Python environment is set up, plus the cache size and free disk space. engine narrows it."""
        groups = [g for g in ml.read_model_catalog() if engine is None or g.engine == engine]
        if engine is not None and not groups:
            raise fail("not_found", f"Unknown engine {engine!r}.")
        downloaded, total = ml.model_counts()
        device, detected = ml.compute_device()
        data = [{"engine": g.engine, "label": g.label, "has_environment": g.has_venv, "environment_ready": g.env_ready,
                 "rows": [_row_view(r) for r in g.rows], "finetuned": list(g.finetuned)} for g in groups]
        return ok(data, evidence={"downloaded": downloaded, "catalogued": total, "cache_bytes": ml.total_cache_bytes(),
                                  "free_bytes": ml.disk_check(0).free_bytes, "compute_device": device,
                                  "detected_device": detected, "offline": batch_runner.offline_default().locked})

    @server.tool(annotations=READ)
    def model_files(engine: str, model_id: str) -> Envelope:
        """What each file of a downloaded model is for (weights, config, tokenizer …) with its size."""
        row = _row(engine, model_id)
        files = ml.major_files(engine, model_id) if row.kind == "model" else []
        return ok([{"name": name, "size_bytes": size, "role": role} for name, size, role in files],
                  evidence={"files": len(files)})

    @server.tool(annotations=READ)
    def model_tasks() -> Envelope:
        """Model actions of this server, newest first: queued / running / done / failed, with a log tail."""
        views = tasks.views()
        return ok([_task_view(v) for v in views], evidence={"busy": sorted(tasks.busy_keys())})

    @server.tool(annotations=READ)
    async def model_task_status(task_id: int, ctx: Context, wait_seconds: float = 0) -> Envelope:
        """One model action; wait_seconds (max 50) waits for it to finish, sending progress with its latest log line."""
        limit = max(0.0, min(float(wait_seconds or 0), MAX_WAIT_SECONDS))
        started = time.monotonic()
        view = tasks.get(task_id)
        if view is None:
            raise fail("not_found", f"No model task {task_id} (model_tasks lists them).")
        while view.state in (ml.TASK_QUEUED, ml.TASK_RUNNING) and time.monotonic() - started < limit:
            await anyio.sleep(0.5)
            view = tasks.get(task_id) or view
            await ctx.report_progress(time.monotonic() - started, limit, message=(view.log[-1] if view.log else view.state))
        return ok({"task": _task_view(view)}, evidence={"task_id": task_id, "state": view.state})

    @server.tool(annotations=WRITE)
    def model_download(engine: str, item_id: str) -> Envelope:
        """Download a model or resource into models/tts (GBs; refused when offline). Returns the queued task; the result
        carries the disk check (free space now / after)."""
        row = _row(engine, item_id)
        _require_network("download")
        check = ml.disk_check(row.approx_gb)
        view = tasks.download_resource(engine, item_id) if row.kind == "resource" else tasks.download(engine, item_id)
        result = _queued(view, "download", f"{engine}/{item_id}", free_bytes=check.free_bytes,
                         free_after_bytes=max(check.after_bytes, 0))
        if check.low:
            result["warnings"].append(f"Less than {ml.LOW_SPACE_BYTES // 2**30} GB will be free after this download.")
        return result

    @server.tool(annotations=WRITE)
    def model_update(engine: str, item_id: str) -> Envelope:
        """Download a model / resource again (fetches the upstream revision; refused when offline)."""
        row = _row(engine, item_id)
        _require_network("update")
        view = tasks.download_resource(engine, item_id) if row.kind == "resource" else tasks.download(engine, item_id, update=True)
        return _queued(view, "update", f"{engine}/{item_id}")

    @server.tool(annotations=WRITE)
    def model_check_update(engine: str, model_id: str) -> Envelope:
        """Compare the local revision of a model with upstream (network; refused when offline). The task summary says
        whether an update is available."""
        row = _row(engine, model_id)
        if row.kind != "model":
            raise fail("invalid", "Update checks apply to models, not resources.")
        _require_network("check_update")
        return _queued(tasks.check_update(engine, model_id), "check_update", f"{engine}/{model_id}")

    @server.tool(annotations=WRITE)
    def model_provision(engine: str) -> Envelope:
        """Set up or update an engine's Python environment (uv sync under engines/<engine>, several minutes; network)
        for the compute device chosen on this machine (cpu / cuda / mps)."""
        _require_network("provision")
        device, _ = ml.compute_device()
        try:
            view = tasks.provision(engine, device)
        except ml.UnknownModel as exc:
            raise fail("not_found", str(exc)) from exc
        return _queued(view, "provision", f"{engine} ({device})", device=device)

    @server.tool(annotations=DESTRUCTIVE)
    def model_remove(engine: str, item_id: str, dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Delete a model's local files (a resource is only de-registered: its cache is shared). Destructive: preview
        first, then dry_run=false + confirm_token."""
        row = _row(engine, item_id)
        if not row.downloaded and not row.state:
            raise fail("refused", f"{engine}/{item_id} has nothing on disk.")
        preview = _row_view(row)
        gate = destructive(state.tokens, "model_remove", {"engine": engine, "item_id": item_id}, dry_run=dry_run,
                           confirm_token=confirm_token, preview=preview, fingerprint=(row.state, row.size_bytes, row.revision))
        if gate is not None:
            return gate
        return _queued(tasks.remove(engine, item_id), "remove", f"{engine}/{item_id}")
