from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer

from syntrive.mcp.envelope import Envelope, destructive, fail, ok, resolve_path
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import DESTRUCTIVE, READ, WRITE
from syntrive.services import voice_library as vl

logger = logging.getLogger(__name__)

Flag = Literal["", "in_use", "no_text", "fine_tuned"]
LanFilter = Literal["en_zh", "all"]


def _voice_view(row: vl.VoiceRow) -> dict:
    v = row.voice
    return {"voice_id": v.db_id, "name": v.name, "gender": v.gender, "language": v.language, "accent": v.accent,
            "tags": list(v.tags), "fine_tuned": v.fine_tuned, "duration_seconds": v.duration_seconds,
            "stored_path": v.stored_path, "file_present": row.file_present, "has_ref_text": row.has_ref_text,
            "used_by": list(row.used_by)}


def _candidate_view(c: vl.VoiceCandidate) -> dict:
    return {"stored_path": c.stored_path, "name": c.name, "gender": c.gender, "language": c.language,
            "tags": list(c.tags), "duration_seconds": c.duration_seconds}


def _external(external: Optional[str]) -> Optional[Path]:
    if not external or not external.strip():
        return None
    folder = resolve_path(external)
    if not folder.is_dir():
        raise fail("invalid", f"Folder not found: {folder}")
    return folder


def _fields(gender: str, language: str) -> tuple[str, str]:
    try:
        return vl.clean_voice_fields(gender, language)
    except vl.VoiceFieldError as exc:
        rule = "male, female or empty" if exc.field == "gender" else "a 2 or 3 letter code (en, zh …) or empty"
        raise fail("invalid", f"{exc.field} must be {rule} (got {exc.value!r}).") from exc


def register(server: MCPServer, state: McpState) -> None:

    def _row(voice_id: int) -> vl.VoiceRow:
        row = next((r for r in vl.list_voices(state.current().db_path) if r.voice.db_id == voice_id), None)
        if row is None:
            raise fail("not_found", f"Voice {voice_id} is not in the catalog (voice_list shows the ids).")
        return row

    @server.tool(annotations=READ)
    def voice_list(q: str = "", lang: str = "", gender: str = "", flag: Flag = "", limit: int = 200) -> Envelope:
        """Reference voices in the catalog, voices used by a book first: name, gender, language, tags, length, whether
        it has a reference text (cosyvoice needs one) and which books use it. Filters: q (name / path), lang (en, zh …),
        gender (male / female), flag (in_use, no_text, fine_tuned). evidence.facets counts every filter value."""
        rows = vl.list_voices(state.current().db_path)
        shown = vl.filter_voices(rows, q=q, lang=lang, gender=gender, flag=flag)
        return ok([_voice_view(r) for r in shown[:max(1, limit)]],
                  evidence={"total": len(rows), "matching": len(shown), "facets": vl.voice_facets(rows)},
                  warnings=[f"Showing {limit} of {len(shown)}."] if len(shown) > limit else [])

    @server.tool(annotations=READ)
    def voice_get(voice_id: int) -> Envelope:
        """One voice with its reference text (the sibling .txt the engines read) and audio details."""
        row = _row(voice_id)
        v = row.voice
        text = vl.read_ref_text(v.abs_path) if row.file_present else ""
        return ok({**_voice_view(row), "ref_text": text, "sample_rate": v.sample_rate, "channels": v.channels,
                   "bit_depth": v.bit_depth}, evidence={"voice_id": voice_id, "path": v.abs_path})

    @server.tool(annotations=WRITE)
    def voice_update(voice_id: int, gender: Optional[str] = None, language: Optional[str] = None,
                     accent: Optional[str] = None, tags: Optional[list[str]] = None,
                     ref_text: Optional[str] = None) -> Envelope:
        """Edit a voice: gender (male / female / ''), language, accent, the full tag list, and the reference text
        (written to the sibling .txt; '' removes it). Omit a field to keep it."""
        row = _row(voice_id)
        v = row.voice
        new_gender, new_language = _fields((v.gender or "") if gender is None else gender,
                                           (v.language or "") if language is None else language)
        new_tags = tuple(v.tags) if tags is None else vl.parse_tags(",".join(tags))
        changed = vl.update_reference_voice(state.current().db_path, voice_id, gender=new_gender, language=new_language,
                                            accent=(v.accent or "") if accent is None else accent, tags=new_tags)
        text_changed = False
        if ref_text is not None:
            if not row.file_present:
                raise fail("refused", f"The voice's WAV file is missing ({v.stored_path}); its text cannot be written.")
            text_changed = vl.save_ref_text(v.abs_path, ref_text)
        logger.info("mcp_action: tool=voice_update voice_id=%d fields_changed=%s ref_text_changed=%s",
                    voice_id, changed, text_changed)
        return ok({"changed": changed or text_changed, "fields_changed": changed, "ref_text_changed": text_changed,
                   **_voice_view(_row(voice_id))}, evidence={"voice_id": voice_id})

    @server.tool(annotations=READ)
    def voice_transcribe(voice_id: int) -> Envelope:
        """Run ASR (faster-whisper from models/tts) on the voice's WAV and return the text — not saved: check it, then
        voice_update(ref_text=...). Loading the model the first time takes a few seconds."""
        row = _row(voice_id)
        if not row.file_present:
            raise fail("not_found", f"The voice's WAV file is missing: {row.voice.stored_path}")
        try:
            text = vl.transcribe_reference_audio(row.voice.abs_path)
        except vl.AsrModelUnavailable as exc:
            raise fail("refused", str(exc)) from exc
        logger.info("mcp_action: tool=voice_transcribe voice_id=%d chars=%d", voice_id, len(text))
        return ok({"voice_id": voice_id, "text": text, "current_ref_text": vl.read_ref_text(row.voice.abs_path)},
                  evidence={"voice_id": voice_id, "chars": len(text)})

    @server.tool(annotations=READ)
    def voice_scan(external: Optional[str] = None, lan: LanFilter = "en_zh") -> Envelope:
        """Find WAV files not yet in the catalog under voices/ and the fine-tuned model tree (plus an optional external
        folder), and catalog voices whose file is gone (with a same-name replacement when one was found).
        lan: en_zh keeps English / Chinese / undetected; all keeps every language."""
        scan = vl.scan_new_voices(state.current().db_path, external=_external(external), lan_filter=lan)
        return ok({"new": [_candidate_view(c) for c in scan.new], "missing": scan.missing},
                  evidence={"new": len(scan.new), "already_in_catalog": scan.skipped, "missing": len(scan.missing)})

    @server.tool(annotations=WRITE)
    def voice_import(stored_paths: list[str], external: Optional[str] = None, lan: LanFilter = "en_zh",
                     gender: str = "", language: str = "", accent: str = "", tags: str = "") -> Envelope:
        """Import chosen files from voice_scan (their stored_path values). The folders are scanned again — a path the scan
        does not find is ignored. gender / language / accent / tags (comma-separated) apply to all of them; empty keeps
        what was detected from the folder names."""
        if not stored_paths:
            raise fail("invalid", "Name at least one stored_path from voice_scan.")
        gender, language = _fields(gender, language)
        imported, matched = vl.import_scanned(state.current().db_path, stored_paths, external=_external(external),
                                              lan_filter=lan, gender=gender, language=language, accent=accent, tags=tags)
        logger.info("mcp_action: tool=voice_import wanted=%d matched=%d imported=%d", len(stored_paths), matched, imported)
        return ok({"imported": imported, "matched": matched, "ignored": len(set(stored_paths)) - matched},
                  evidence={"imported": imported},
                  warnings=["Some paths were not new files found by the scan and were ignored."] if matched < len(set(stored_paths)) else [])

    @server.tool(annotations=WRITE)
    def voice_relink(voice_id: int, external: Optional[str] = None, lan: LanFilter = "en_zh") -> Envelope:
        """Point a catalog voice whose file is gone at the same-name, same-format file a fresh scan finds."""
        _row(voice_id)
        missing = vl.relink_scanned(state.current().db_path, voice_id, external=_external(external), lan_filter=lan)
        if missing is None:
            raise fail("refused", "No replacement found for this voice (voice_scan lists missing voices and replacements).")
        logger.info("mcp_action: tool=voice_relink voice_id=%d path=%s", voice_id, missing.replacement)
        return ok({"voice_id": voice_id, "old_path": missing.stored_path, "new_path": missing.replacement},
                  evidence={"voice_id": voice_id})

    @server.tool(annotations=DESTRUCTIVE)
    def voice_delete(voice_id: int, dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Remove a voice (and its tags) from the catalog; the WAV file stays on disk. Refused while a book's cast uses it.
        Destructive: preview first, then dry_run=false + confirm_token."""
        row = _row(voice_id)
        if row.in_use:
            raise fail("conflict", f"Voice {voice_id} is used by {', '.join(row.used_by)}; choose another voice for those books first.")
        preview = _voice_view(row)
        gate = destructive(state.tokens, "voice_delete", {"voice_id": voice_id}, dry_run=dry_run,
                           confirm_token=confirm_token, preview=preview, fingerprint=(row.voice.stored_path, row.used_by))
        if gate is not None:
            return gate
        if not vl.delete_reference_voice(state.current().db_path, voice_id):
            raise fail("conflict", "The voice could not be deleted (it may just have been assigned to a book).")
        logger.info("mcp_action: tool=voice_delete voice_id=%d name=%s", voice_id, row.voice.name)
        return ok({"deleted": True, **preview}, evidence={"voice_id": voice_id})
