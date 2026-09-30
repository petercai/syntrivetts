from __future__ import annotations

import logging
from typing import Optional

from mcp.server.mcpserver import MCPServer

from syntrive.mcp.envelope import Envelope, destructive, ok
from syntrive.mcp.state import McpState
from syntrive.mcp.tools._common import DESTRUCTIVE
from syntrive.services import cover_watermark as cw

logger = logging.getLogger(__name__)


def register(server: MCPServer, state: McpState) -> None:

    @server.tool(annotations=DESTRUCTIVE)
    def audio_watermark_covers(dry_run: bool = True, confirm_token: Optional[str] = None) -> Envelope:
        """Watermark the cover of every .m4a under <repo>/*/audiobooks/ (hollow red "SyntriveTTS", bottom-right). Files
        already stamped are skipped; files without a cover are left alone. Rewrites files in place (the original cover is
        not kept). Destructive: the dry run lists what would be stamped; then dry_run=false + confirm_token.
        VLC may keep showing a cover it cached earlier until its art cache is cleared."""
        repo = state.current()
        planned = cw.watermark_repo(repo.repo_dir, dry_run=True)
        rel = [{"file": r.path.relative_to(repo.repo_dir).as_posix(), "status": r.status} for r in planned]
        counts = {s: sum(r.status == s for r in planned) for s in (cw.WATERMARKED, cw.ALREADY, cw.NO_COVER, cw.FAILED)}
        todo = sorted(x["file"] for x in rel if x["status"] == cw.WATERMARKED)
        gate = destructive(state.tokens, "audio_watermark_covers", {}, dry_run=dry_run, confirm_token=confirm_token,
                           preview={"would_watermark": todo, "counts": counts}, fingerprint=todo)
        if gate is not None:
            return gate
        results = cw.watermark_repo(repo.repo_dir)
        done = {s: sum(r.status == s for r in results) for s in counts}
        failed = [{"file": r.path.relative_to(repo.repo_dir).as_posix(), "error": r.error} for r in results if not r.ok]
        logger.info("mcp_action: tool=audio_watermark_covers %s", done)
        return ok({"counts": done, "failed": failed}, evidence={"watermarked": done[cw.WATERMARKED]},
                  warnings=[f"{len(failed)} file(s) failed."] if failed else [])
