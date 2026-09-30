from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import time
from contextlib import contextmanager
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable, Iterator, Mapping, Optional

from mcp.server.mcpserver.exceptions import ToolError

logger = logging.getLogger(__name__)

HOLDER_KIND = "mcp"
Envelope = dict[str, Any]
CODES = ("not_found", "invalid", "conflict", "refused", "failed")


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def ok(data: Any = None, *, evidence: Optional[Mapping[str, Any]] = None,
       warnings: tuple[str, ...] | list[str] = (), dry_run: bool = False) -> Envelope:
    return {"ok": True, "data": to_jsonable(data), "evidence": to_jsonable(dict(evidence or {})),
            "warnings": list(warnings), "dry_run": dry_run}


def fail(code: str, message: str) -> ToolError:
    assert code in CODES, code
    logger.info("mcp_tool_refused: code=%s message=%s", code, message)
    return ToolError(f"{code}: {message}")


_STEP_CODES = {"not_found": "not_found", "archived": "refused", "not_current": "refused",
               "needs_decision": "refused", "bad_options": "invalid"}


@contextmanager
def translating() -> Iterator[None]:
    from syntrive.services.job_lease import JobLeaseConflict
    from syntrive.services.pipeline_service import RunAlreadyActive, StepRunError
    from syntrive.services.step_decisions import DecisionError
    from syntrive.services.tts_settings import SettingsError

    try:
        yield
    except ToolError:
        raise
    except JobLeaseConflict as exc:
        raise fail("conflict", f"{exc} Call lease_unlock(job_id={exc.job_id}) only if that holder has stopped.") from exc
    except RunAlreadyActive as exc:
        raise fail("conflict", str(exc)) from exc
    except StepRunError as exc:
        raise fail(_STEP_CODES.get(exc.code, "refused"), str(exc)) from exc
    except (DecisionError, SettingsError) as exc:
        raise fail("not_found" if getattr(exc, "code", "") == "not_found" else "invalid", str(exc)) from exc


def _canonical(tool: str, args: Mapping[str, Any], fingerprint: Any) -> bytes:
    return json.dumps({"tool": tool, "args": to_jsonable(dict(args)), "preview": to_jsonable(fingerprint)},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class ConfirmTokens:
    secret: bytes
    ttl_seconds: float = 600.0
    clock: Callable[[], float] = time.time

    def _mac(self, expires: int, payload: bytes) -> str:
        return hmac.new(self.secret, str(expires).encode() + b"|" + payload, hashlib.sha256).hexdigest()[:32]

    def issue(self, tool: str, args: Mapping[str, Any], fingerprint: Any) -> str:
        expires = int(self.clock() + self.ttl_seconds)
        return f"{expires}.{self._mac(expires, _canonical(tool, args, fingerprint))}"

    def check(self, tool: str, args: Mapping[str, Any], fingerprint: Any, token: Optional[str]) -> Optional[str]:
        if not token:
            return "missing"
        expires_text, _, mac = token.partition(".")
        if not expires_text.isdigit() or not mac:
            return "malformed"
        if int(expires_text) < self.clock():
            return "expired"
        expected = self._mac(int(expires_text), _canonical(tool, args, fingerprint))
        return None if hmac.compare_digest(mac, expected) else "mismatch"


def destructive(tokens: ConfirmTokens, tool: str, args: Mapping[str, Any], *, dry_run: bool,
                confirm_token: Optional[str], preview: Any, fingerprint: Any) -> Optional[Envelope]:
    if dry_run:
        return ok(preview, evidence={"confirm_token": tokens.issue(tool, args, fingerprint)}, dry_run=True,
                  warnings=[f"Nothing changed. Call {tool} again with dry_run=false and this confirm_token to apply."])
    reason = tokens.check(tool, args, fingerprint, confirm_token)
    if reason is None:
        return None
    logger.info("mcp_confirm_refused: tool=%s reason=%s", tool, reason)
    raise fail("refused", f"confirm_token {reason}: run {tool} with dry_run=true, review the preview, then use its "
                          f"confirm_token (valid 10 minutes, only for that preview).")


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()
