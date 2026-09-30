from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer
from sqlalchemy.exc import IntegrityError

from syntrive.mcp.envelope import HOLDER_KIND, Envelope, destructive, fail, ok, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import DESTRUCTIVE, READ, WRITE
from syntrive.services import db_tools_service as dbt

logger = logging.getLogger(__name__)

_FILE_CODES = {"not_found": "not_found", "bad_name": "invalid", "bad_kind": "invalid", "live_db": "refused"}


def _file_view(f) -> dict:
    view = {"name": f.path.name, "kind": "export" if f.path.suffix == ".db" else "voice_backup",
            "size_bytes": f.size_bytes, "modified": f.modified}
    if isinstance(f, dbt.ExportFile):
        view.update(books=[{"title": b.title, "author": b.author, "jobs": [j.job_id for j in b.jobs]} for b in f.books],
                    jobs=f.job_count, error=f.error or None)
    else:
        view.update(voices=f.voice_count, tags=f.tag_count)
    return view


def register(server: MCPServer, state: McpState) -> None:

    def _file(name: str) -> Path:
        try:
            return dbt.repo_file(state.current().repo_dir, name)
        except dbt.RepoFileError as exc:
            message = {"not_found": f"No such file next to the database: {name} (db_files lists them).",
                       "live_db": "The repo's own syntrivetts.db cannot be used here."}.get(
                exc.code, f"Not a .db export / .sql backup file name: {name!r}")
            raise fail(_FILE_CODES.get(exc.code, "invalid"), message) from exc

    def _stamp(path: Path) -> tuple[int, int]:
        info = path.stat()
        return info.st_size, info.st_mtime_ns

    @server.tool(annotations=READ)
    def db_files() -> Envelope:
        """Export files (.db: books + jobs) and voice backups (.sql) next to syntrivetts.db, newest first, with what each
        holds (books / jobs, voices / tags) — read without modifying them."""
        files = dbt.list_repo_files(state.current().repo_dir)
        return ok([_file_view(f) for f in files], evidence={"files": len(files)})

    @server.tool(annotations=WRITE)
    def db_export(job_ids: list[int], include_voices: bool = True) -> Envelope:
        """Export jobs (with their books, settings, chapters and cast) to a new self-contained syntrivetts-export-<time>.db
        next to the database; include_voices bundles the whole voice catalog. Rows only: copy each job's PROCESSING-
        folder to the other machine separately."""
        if not job_ids:
            raise fail("invalid", "Name at least one job id (book_list).")
        try:
            result = dbt.export_to_repo(state.current().db_path, sorted(set(job_ids)), include_reference_voices=include_voices)
        except ValueError as exc:
            raise fail("not_found", str(exc)) from exc
        logger.info("mcp_action: tool=db_export file=%s jobs=%d voices=%d", result.output_path.name, result.job_count,
                    result.reference_voice_count)
        return ok({"file": result.output_path.name, "books": result.book_count, "jobs": result.job_count,
                   "voices": result.reference_voice_count, "folders_to_copy": list(result.process_dir_reminders)},
                  evidence={"file": result.output_path},
                  warnings=[f"Voice bindings lost for {len(result.unbound_tts_voice_warnings)} cast row(s) (include_voices=false)."]
                  if result.unbound_tts_voice_warnings else [])

    @server.tool(annotations=WRITE)
    def db_backup_voices() -> Envelope:
        """Back up the reference-voice catalog to a new tts-reference-voice-<time>.sql next to the database
        (voices with machine-specific absolute paths are left out)."""
        path, voices, skipped, tags = dbt.backup_to_repo(state.current().db_path)
        logger.info("mcp_action: tool=db_backup_voices file=%s voices=%d skipped=%d", path.name, voices, skipped)
        return ok({"file": path.name, "voices": voices, "tags": tags, "skipped_absolute_paths": skipped},
                  evidence={"file": path})

    @server.tool(annotations=DESTRUCTIVE)
    def db_import(file: str, job_ids: Optional[list[int]] = None, overwrite_job_ids: Optional[list[int]] = None,
                  dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Import jobs from an export file (db_files) into this repo. job_ids = the file's job ids (default: all).
        A job whose book + folder already exist here is skipped unless its id is in overwrite_job_ids (this repo's job
        is then deleted and re-created, under its job lease). Destructive: the dry run returns the plan (add / overwrite
        / skip per job); repeat with dry_run=false + its confirm_token. Copy the PROCESSING- folders separately."""
        path = _file(file)
        if path.suffix != ".db":
            raise fail("invalid", f"{file} is not a .db export (a .sql backup is restored with db_restore_voices).")
        with translating():
            plan = dbt.plan_import(path, state.current().db_path)
        if plan.export.error:
            raise fail("invalid", f"{file} cannot be read as an export file: {plan.export.error}")
        in_file = [j.job_id for b in plan.export.books for j in b.jobs]
        chosen = sorted(set(job_ids)) if job_ids else in_file
        unknown = sorted(set(chosen) - set(in_file))
        if unknown:
            raise fail("invalid", f"Job ids {unknown} are not in {file} (its jobs: {in_file}).")
        overwrite = sorted(set(overwrite_job_ids or ()) & set(chosen))
        conflicts = {c.export_job_id: c for c in plan.conflicts}
        titles = {j.job_id: b.title for b in plan.export.books for j in b.jobs}
        actions = [{"job_id": j, "title": titles[j],
                    "action": "add" if j not in conflicts else ("overwrite" if j in overwrite else "skip"),
                    "replaces_job_id": conflicts[j].target_job_id if j in conflicts else None,
                    "process_dir": conflicts[j].process_dir if j in conflicts else None} for j in chosen]
        preview = {"file": file, "jobs": actions}
        gate = destructive(state.tokens, "db_import", {"file": file, "job_ids": chosen, "overwrite_job_ids": overwrite},
                           dry_run=dry_run, confirm_token=confirm_token, preview=preview,
                           fingerprint=(_stamp(path), sorted((c.export_job_id, c.target_job_id) for c in plan.conflicts)))
        if gate is not None:
            return gate
        with translating():
            result = dbt.import_into_repo(path, state.current().db_path, chosen,
                                          overwrite_job_ids=frozenset(overwrite), holder_kind=HOLDER_KIND)
        logger.info("mcp_action: tool=db_import file=%s created=%d overwritten=%d skipped=%d", file,
                    result.created_job_count, result.overwritten_job_count, result.skipped_job_count)
        return ok({"added": result.created_job_count, "overwritten": result.overwritten_job_count,
                   "skipped": result.skipped_job_count, "voices_resolved": result.reference_voice_resolved_count,
                   "voices_unresolved": result.reference_voice_unresolved_count, "plan": actions},
                  evidence={"file": file}, warnings=["Copy the jobs' PROCESSING- folders into this repo if not done yet."])

    @server.tool(annotations=DESTRUCTIVE)
    def db_restore_voices(file: str, dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Restore a voice backup (.sql) into this repo's catalog. Append-only: if any voice id in it already exists,
        nothing is changed and the call fails. Destructive: preview first, then dry_run=false + confirm_token."""
        path = _file(file)
        backup = next((b for b in dbt.list_backup_files(state.current().repo_dir) if b.path == path), None)
        if backup is None:
            raise fail("invalid", f"{file} is not a voice backup (.sql).")
        preview = _file_view(backup)
        gate = destructive(state.tokens, "db_restore_voices", {"file": file}, dry_run=dry_run,
                           confirm_token=confirm_token, preview=preview, fingerprint=_stamp(path))
        if gate is not None:
            return gate
        try:
            voices, tags = dbt.restore_reference_voices(state.current().db_path, path)
        except (IntegrityError, sqlite3.IntegrityError) as exc:
            raise fail("conflict", f"{file} could not be restored: some of its voice ids already exist here. "
                                   "Nothing was changed.") from exc
        logger.info("mcp_action: tool=db_restore_voices file=%s voices=%d tags=%d", file, voices, tags)
        return ok({"voices": voices, "tags": tags}, evidence={"file": file})

    @server.tool(annotations=DESTRUCTIVE)
    def db_delete_file(file: str, dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Delete an export (.db, with its -wal / -shm) or a voice backup (.sql) next to the database.
        Destructive: preview first, then dry_run=false + confirm_token."""
        path = _file(file)
        preview = {"file": file, "size_bytes": path.stat().st_size}
        gate = destructive(state.tokens, "db_delete_file", {"file": file}, dry_run=dry_run,
                           confirm_token=confirm_token, preview=preview, fingerprint=_stamp(path))
        if gate is not None:
            return gate
        dbt.delete_repo_file(state.current().repo_dir, file)
        logger.info("mcp_action: tool=db_delete_file file=%s", file)
        return ok({"deleted": file}, evidence={"file": file})
