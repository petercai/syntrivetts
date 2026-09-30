from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from syntrive.mcp.envelope import Envelope, ok, resolve_path, translating
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import READ, WRITE, book_brief

logger = logging.getLogger(__name__)


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(annotations=READ)
    def repo_info() -> Envelope:
        """The repo this server works on: folder, database, books by tab, leases held, runs in this process.
        Start here; every other tool acts on this repo."""
        from syntrive.services import book_catalog
        from syntrive.services.job_lease import list_leases

        repo = state.current()
        with translating():
            rows = book_catalog.list_book_rows(repo.db_path)
            leases = list_leases(repo.db_path)
        return ok({
            "repo_dir": repo.repo_dir, "db_path": repo.db_path,
            "books": book_catalog.tab_counts(rows),
            "in_progress": [book_brief(r) for r in rows if r.tab == "active"],
            "leases": [{"job_id": lease.job_id, "holder_kind": lease.holder_kind, "operation": lease.operation}
                       for lease in leases],
            "step_running_here": state.runs.any_running(),
            "transport": state.transport,
        }, evidence={"repo_dir": repo.repo_dir})

    @server.tool(annotations=WRITE)
    def repo_switch(repo_dir: str) -> Envelope:
        """Bind this server to another repo folder (one holding syntrivetts.db). Refused while a step runs, and on an
        HTTP server (shared by its clients: start another server for the other repo)."""
        new = state.switch(resolve_path(repo_dir))
        return ok({"repo_dir": new.repo_dir, "db_path": new.db_path}, evidence={"repo_dir": new.repo_dir})

    @server.tool(annotations=WRITE)
    def repo_refresh_contract() -> Envelope:
        """Rewrite the portable contract (repo.json + each job.json) that SyntriveCMS / the publisher read."""
        from syntrive.services.contract_export_service import export_repo_manifests

        repo = state.current()
        with translating():
            result = export_repo_manifests(repo.db_path)
        logger.info("mcp_action: tool=repo_refresh_contract ok=%s jobs=%d", result.ok, len(result.jobs))
        return ok({"ok": result.ok, "error": result.error, "jobs": len(result.jobs)},
                  evidence={"repo_json": repo.repo_dir / "repo.json"},
                  warnings=[result.error] if result.error else [])
