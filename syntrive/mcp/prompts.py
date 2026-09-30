from __future__ import annotations

from mcp.server.mcpserver import MCPServer

PRODUCE_AUDIOBOOK = """Produce an audiobook from the EPUB at {epub_path} with the syntrive tools.

1. repo_info — confirm which repo you are working in.
2. book_add(epub_path="{epub_path}") — note the job_id it returns.
3. Loop: pipeline_status(job_id) → for the current step:
   - mode "run": pipeline_run_step(job_id, wait_seconds=50); if it returns needs_confirm, tell me what exists and
     ask whether to overwrite or keep; while it runs, poll pipeline_run_status(job_id, wait_seconds=50).
   - transcript_merge (step 3) is done: show me pipeline_merge_settings_get — chapters that look like junk (tiny
     character count, "Contents", "Copyright") — and ask which to exclude before step 4
     (pipeline_chapters_set_exclusions, then book_reset_to_step if a later step already ran).
   - transcript_clean (step 4) is done: show pipeline_clean_rules_get results with a high deletion ratio or
     threshold_blocked and ask whether to change rules.
   - transcript_review (step 6, needs a decision): show pipeline_transcripts_get and ask which format to use
     (raw or tts_script), then pipeline_run_step(job_id, transcript_path=<choice>).
   - tts_config (step 7): tts_settings_get(job_id); propose an engine / model and a reference voice for the
     narrator and each role in the book's language; after my approval tts_settings_save, then pipeline_run_step.
   - synthesis (mode "queue"): queue_enqueue(job_id) then queue_start_runner(); follow queue_now / queue_book.
     If queue_now says no runner is working while chapters are not done, the runner stopped: report the chapters'
     error (queue_book) instead of waiting.
4. Stop and ask me whenever a tool returns a conflict (someone else is editing the book) or a refusal.

Report every step with the evidence the tools returned (job_id, counts, files). Never call lease_unlock or a
destructive tool with dry_run=false without showing me its preview first.
"""

REPO_STATUS = """Give me the status of this SyntriveTTS repo using the syntrive tools (read-only):

1. repo_info — books per tab, leases, runs.
2. book_list(tab="active") — for each book: its current step (of 9) and whether it waits for a decision
   (pipeline_status mode "tui" or a step needing my choice).
3. queue_overview — is a runner working, what is queued / paused, failed chapters (error + line).
4. lease_list — anything held for a long time with an old heartbeat (possibly a crashed holder).

Answer as: in progress, waiting for me (with the decision needed), failed, next suggested actions. Change nothing.
"""


def register(server: MCPServer) -> None:

    @server.prompt(name="produce_audiobook", description="From an EPUB path to queued synthesis, asking at each decision.")
    def produce_audiobook(epub_path: str) -> str:
        return PRODUCE_AUDIOBOOK.format(epub_path=epub_path)

    @server.prompt(name="repo_status", description="What is in progress, waiting, failed, and what to do next (read-only).")
    def repo_status() -> str:
        return REPO_STATUS
