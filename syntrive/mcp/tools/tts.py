from __future__ import annotations

import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from syntrive.mcp.envelope import Envelope, HOLDER_KIND, ok, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import READ, WRITE, require_row
from syntrive.services import tts_settings as ts

logger = logging.getLogger(__name__)


class VoiceBindingIn(BaseModel):
    id: int
    reference_voice_id: Optional[int] = None
    excluded: bool = False


def _params() -> list[dict]:
    return [{"field": p.field, "label": p.label, "type": p.input_type.value, "tab": p.tab, "choices": p.choices,
             "default": p.default, "min": p.min_val, "max": p.max_val} for p in ts.PARAMS]


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(annotations=READ)
    def tts_settings_get(job_id: int, include_params: bool = False) -> Envelope:
        """Step 7 settings of a book: current values (engine, model, language, speed …), the cast (narrator = voice_id 0,
        roles = ‡voice:N‡ in the transcript) with their reference voices, the reference voices available in the book's
        language, and the models of the chosen engine. include_params adds every field's type / choices / range.
        Once step 7 is reached, missing settings are created first (as the WebUI does)."""
        from syntrive.services.pipeline_service import StepState
        from syntrive.workflow.engine import WorkflowStep

        repo = state.current()
        row = require_row(state, job_id)
        with translating():
            settings = ts.read_tts_settings(repo.db_path, job_id)
            reached = row.pipeline.get(WorkflowStep.TTS_CONFIG).state != StepState.LATER
            if not settings.exists and reached and not row.archived:
                ts.prepare_tts_settings(repo.db_path, job_id, holder_kind=HOLDER_KIND)
                settings = ts.read_tts_settings(repo.db_path, job_id)
        data = {"exists": settings.exists, "values": settings.values, "voices": settings.voices,
                "references": settings.references, "model_choices": settings.model_choices}
        if include_params:
            data["params"] = _params()
        return ok(data, evidence={"job_id": job_id, "voices": len(settings.voices)},
                  warnings=[] if settings.exists else ["No settings yet: step 7 has not been reached."])

    @server.tool(annotations=READ)
    def tts_settings_validate(job_id: int, changes: dict[str, Any]) -> Envelope:
        """Check field changes ({"engine": "cosyvoice", "speed": 1.1}) against the current values without saving:
        returns the merged, coerced values or an invalid error naming the field."""
        repo = state.current()
        with translating():
            current = ts.read_tts_settings(repo.db_path, job_id)
            merged = ts.validate_values(current.values, changes)
        changed = sorted(k for k, v in merged.items() if current.values.get(k) != v)
        return ok({"values": merged, "changed": changed}, evidence={"job_id": job_id})

    @server.tool(annotations=WRITE)
    def tts_settings_save(job_id: int, changes: Optional[dict[str, Any]] = None,
                          voices: Optional[list[VoiceBindingIn]] = None) -> Envelope:
        """Save field changes and / or cast bindings in one call (validated first; nothing is written on an error).
        voices lists only the rows to change. Takes the job lease. Afterwards run step 7 with pipeline_run_step."""
        repo = state.current()
        bindings = [ts.VoiceBinding(v.id, v.reference_voice_id, v.excluded) for v in voices or []]
        with translating():
            result = ts.save_tts_settings(repo.db_path, job_id, changes or {}, bindings, holder_kind=HOLDER_KIND)
        logger.info("mcp_action: tool=tts_settings_save job_id=%d fields=%s voices=%s",
                    job_id, list(result.changed_fields), list(result.changed_voices))
        return ok({"changed": result.changed, "changed_fields": result.changed_fields,
                   "changed_voices": result.changed_voices}, evidence={"job_id": job_id})
