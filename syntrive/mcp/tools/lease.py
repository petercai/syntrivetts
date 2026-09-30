from __future__ import annotations

import logging
import time
from typing import Optional

from mcp.server.mcpserver import MCPServer

from syntrive.mcp.envelope import Envelope, destructive, fail, ok, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import DESTRUCTIVE, READ
from syntrive.services import job_lease

logger = logging.getLogger(__name__)


def _lease_view(lease: job_lease.LeaseInfo) -> dict:
    now = time.time()
    return {"job_id": lease.job_id, "holder_kind": lease.holder_kind, "operation": lease.operation,
            "pid": lease.pid, "hostname": lease.hostname, "held_seconds": round(now - lease.acquired_at),
            "heartbeat_age_seconds": round(now - lease.heartbeat_at), "expired": lease.is_expired(now)}


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(annotations=READ)
    def lease_list() -> Envelope:
        """Job write leases: which entry point (tui / webui / mcp / tts_batch / cli) holds which job, since when, and
        whether its heartbeat is still fresh. A write to a leased job is refused with a conflict."""
        with translating():
            leases = job_lease.list_leases(state.current().db_path)
        return ok([_lease_view(lease) for lease in leases], evidence={"leases": len(leases)})

    @server.tool(annotations=DESTRUCTIVE)
    def lease_unlock(job_id: int, dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Force-release the lease on a job, only when its holder has stopped (a crashed TUI, a killed batch). If the
        holder is still running, both will write to the book. Destructive: preview first, then dry_run=false +
        confirm_token."""
        repo = state.current()
        lease = next((x for x in job_lease.list_leases(repo.db_path) if x.job_id == job_id), None)
        if lease is None:
            raise fail("not_found", f"Job {job_id} holds no lease.")
        preview = _lease_view(lease)
        gate = destructive(state.tokens, "lease_unlock", {"job_id": job_id}, dry_run=dry_run,
                           confirm_token=confirm_token, preview=preview, fingerprint=lease.holder_id)
        if gate is not None:
            return gate
        with translating():
            removed = job_lease.force_release(repo.db_path, job_id, reason="mcp lease_unlock")
        logger.info("mcp_action: tool=lease_unlock job_id=%d removed=%s", job_id, removed is not None)
        return ok({"released": removed is not None, "lease": preview}, evidence={"job_id": job_id})
