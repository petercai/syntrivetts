from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.mcp.server import create_server  # noqa: E402
from syntrive.services.job_lease import HolderIdentity, acquire_lease, release_lease  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"


@pytest.fixture()
def env(tmp_path):
    repo = tmp_path / "repo"
    job = bootstrap(repo, EBOOKS / "Jan-Eyre-5.epub")
    server, state = create_server(repo)
    return server, state, repo, job.id


def run(server, body):
    async def main():
        async with Client(server) as client:
            return await body(client)
    return asyncio.run(main())


async def call(client, name, args=None, **kw):
    result = await client.call_tool(name, args or {}, **kw)
    return result


def data(result):
    assert not result.is_error, result.content[0].text
    return result.structured_content


def error(result):
    assert result.is_error
    return result.content[0].text


def test_catalog_schema_names_and_annotations(env):
    server = env[0]

    async def body(c):
        return (await c.list_tools()).tools, (await c.list_prompts()).prompts

    tools, prompts = run(server, body)
    names = {t.name for t in tools}
    assert len(names) == 57
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names)
    assert all(t.description and t.input_schema and t.output_schema for t in tools)
    by_name = {t.name: t for t in tools}
    assert by_name["book_reset_to_step"].annotations.destructive_hint is True
    assert by_name["lease_unlock"].annotations.destructive_hint is True
    assert by_name["book_list"].annotations.read_only_hint is True
    assert {p.name for p in prompts} == {"produce_audiobook", "repo_status"}


def test_read_tools_and_errors(env):
    server, _, repo, job_id = env

    async def body(c):
        info = data(await call(c, "repo_info"))
        books = data(await call(c, "book_list"))
        book = data(await call(c, "book_get", {"job_id": job_id}))
        missing = error(await call(c, "book_get", {"job_id": 999}))
        bad_step = error(await call(c, "pipeline_run_step", {"job_id": job_id, "step": "nope"}))
        not_current = error(await call(c, "pipeline_run_step", {"job_id": job_id, "step": "tts_config"}))
        tts = data(await call(c, "tts_settings_get", {"job_id": job_id}))
        bad_field = error(await call(c, "tts_settings_validate", {"job_id": job_id, "changes": {"colour": "red"}}))
        queue = data(await call(c, "queue_book", {"job_id": job_id}))
        prompt = await c.get_prompt("produce_audiobook", {"epub_path": "C:/books/x.epub"})
        return info, books, book, missing, bad_step, not_current, tts, bad_field, queue, prompt

    info, books, book, missing, bad_step, not_current, tts, bad_field, queue, prompt = run(server, body)
    assert info["ok"] and Path(info["data"]["repo_dir"]) == repo.resolve()
    assert [b["job_id"] for b in books["data"]] == [job_id]
    assert books["evidence"]["counts"]["active"] == 1
    assert book["data"]["current_step"] == "bootstrap" and len(book["data"]["steps"]) == 9
    assert missing.endswith("not_found: Job 999 does not exist in this repo. book_list shows the jobs.")
    assert "invalid: Unknown step 'nope'" in bad_step
    assert "refused:" in not_current and "not the current step" in not_current
    assert tts["data"]["exists"] is False and tts["warnings"]
    assert "invalid: Unknown setting(s): colour" in bad_field
    assert queue["data"]["job_id"] == job_id
    assert "C:/books/x.epub" in prompt.messages[0].content.text


def test_run_steps_with_wait_and_progress_then_decisions(env):
    server, _, _, job_id = env
    progress = []

    async def on_progress(value, total, message):
        progress.append((value, total, message))

    async def body(c):
        data(await call(c, "pipeline_run_step", {"job_id": job_id, "wait_seconds": 50}))
        extract = data(await call(c, "pipeline_run_step", {"job_id": job_id, "wait_seconds": 50},
                                  progress_callback=on_progress))
        merge = data(await call(c, "pipeline_run_step", {"job_id": job_id, "wait_seconds": 50}))
        status = data(await call(c, "pipeline_status", {"job_id": job_id}))
        settings = data(await call(c, "pipeline_merge_settings_get", {"job_id": job_id}))
        first = settings["data"]["selection"]["chapters"][0]["chapter_id"]
        text = data(await call(c, "pipeline_chapter_text", {"job_id": job_id, "chapter_id": first}))
        excluded = data(await call(c, "pipeline_chapters_set_exclusions", {"job_id": job_id, "excluded": [first]}))
        again = data(await call(c, "pipeline_chapters_set_exclusions", {"job_id": job_id, "excluded": [first]}))
        rules = data(await call(c, "pipeline_clean_rules_get", {"job_id": job_id}))
        bad_rule = error(await call(c, "pipeline_clean_rules_set", {"job_id": job_id, "enabled": {"No such": True}}))
        return extract, merge, status, settings, text, excluded, again, rules, bad_rule

    extract, merge, status, settings, text, excluded, again, rules, bad_rule = run(server, body)
    assert extract["data"]["run"]["disposition"] == "done" and extract["data"]["pipeline"]["current_step"] == "transcript_merge"
    assert merge["data"]["run"]["disposition"] == "done"
    assert status["data"]["current_step"] == "transcript_clean"
    assert settings["data"]["selection"]["available"] is True and settings["evidence"]["chapters"] > 0
    assert text["data"]["text"]
    assert excluded["data"]["changed"] is True and again["data"]["changed"] is False
    assert rules["data"]["rules"] and "invalid: Unknown rules" in bad_rule
    assert extract["data"]["run"]["running"] is False
    assert all(total == 50 for _, total, _ in progress)


def test_destructive_protocol_reset_to_step(env):
    server, state, _, job_id = env

    async def body(c):
        for _ in range(2):
            data(await call(c, "pipeline_run_step", {"job_id": job_id, "wait_seconds": 50}))
        not_done = error(await call(c, "book_reset_to_step", {"job_id": job_id, "step": "transcript_merge"}))
        preview = data(await call(c, "book_reset_to_step", {"job_id": job_id, "step": "transcript_extract"}))
        token = preview["evidence"]["confirm_token"]
        missing = error(await call(c, "book_reset_to_step",
                                   {"job_id": job_id, "step": "transcript_extract", "dry_run": False}))
        wrong = error(await call(c, "book_reset_to_step",
                                 {"job_id": job_id, "step": "transcript_extract", "dry_run": False, "confirm_token": token[:-1] + "0"}))
        done = data(await call(c, "book_reset_to_step",
                               {"job_id": job_id, "step": "transcript_extract", "dry_run": False, "confirm_token": token}))
        stale = error(await call(c, "book_reset_to_step",
                                 {"job_id": job_id, "step": "transcript_extract", "dry_run": False, "confirm_token": token}))
        ask = data(await call(c, "pipeline_run_step", {"job_id": job_id}))
        overwrite = data(await call(c, "pipeline_run_step", {"job_id": job_id, "on_existing": "overwrite", "wait_seconds": 50}))
        return not_done, preview, missing, wrong, done, stale, ask, overwrite

    not_done, preview, missing, wrong, done, stale, ask, overwrite = run(server, body)
    assert "refused:" in not_done
    assert preview["dry_run"] is True and preview["data"]["reopens"] == ["transcript_extract", "transcript_merge"]
    assert "refused: confirm_token missing" in missing and "refused: confirm_token mismatch" in wrong
    assert done["dry_run"] is False and done["data"]["current_step"] == "transcript_extract"
    assert "refused:" in stale
    assert ask["data"]["needs_confirm"] is True and ask["data"]["started"] is False
    assert overwrite["data"]["run"]["disposition"] == "done"


def test_lease_conflict_then_unlock(env):
    server, _, repo, job_id = env
    db = repo / "syntrivetts.db"
    other = HolderIdentity.current("tui")
    assert acquire_lease(db, job_id, holder=other, operation="editing").ok
    try:
        async def body(c):
            conflict = error(await call(c, "book_set_metadata", {"job_id": job_id, "language": "zh"}))
            leases = data(await call(c, "lease_list"))
            preview = data(await call(c, "lease_unlock", {"job_id": job_id}))
            released = data(await call(c, "lease_unlock", {"job_id": job_id, "dry_run": False,
                                                           "confirm_token": preview["evidence"]["confirm_token"]}))
            changed = data(await call(c, "book_set_metadata", {"job_id": job_id, "language": "zh", "author": "C. Brontë"}))
            return conflict, leases, preview, released, changed

        conflict, leases, preview, released, changed = run(server, body)
    finally:
        release_lease(db, job_id, other.holder_id)
    assert conflict.startswith("Error executing tool book_set_metadata: conflict:") and f"lease_unlock(job_id={job_id})" in conflict
    assert leases["data"][0]["holder_kind"] == "tui"
    assert preview["dry_run"] and preview["data"]["holder_kind"] == "tui"
    assert released["data"]["released"] is True
    assert set(changed["data"]["changed"]) == {"language", "author"} and changed["data"]["language"] == "zh"


def test_book_add_archive_and_repo_switch(env, tmp_path):
    server, state, repo, job_id = env
    other = tmp_path / "other"
    bootstrap(other, EBOOKS / "Jan-Eyre-5.epub")

    async def body(c):
        bad = error(await call(c, "book_add", {"epub_path": str(tmp_path / "missing.epub")}))
        not_epub = error(await call(c, "book_add", {"epub_path": str(repo / "syntrivetts.db")}))
        archived = data(await call(c, "book_archive", {"job_id": job_id}))
        refused = error(await call(c, "pipeline_run_step", {"job_id": job_id}))
        restored = data(await call(c, "book_unarchive", {"job_id": job_id}))
        no_repo = error(await call(c, "repo_switch", {"repo_dir": str(tmp_path / "nowhere")}))
        switched = data(await call(c, "repo_switch", {"repo_dir": str(other)}))
        info = data(await call(c, "repo_info"))
        return bad, not_epub, archived, refused, restored, no_repo, switched, info

    bad, not_epub, archived, refused, restored, no_repo, switched, info = run(server, body)
    assert "not_found: EPUB not found" in bad and "invalid: Only EPUB files" in not_epub
    assert archived["data"]["archived"] is True and "refused:" in refused and "archived" in refused
    assert restored["data"]["archived"] is False
    assert "not_found: Not a SyntriveTTS repo" in no_repo
    assert Path(switched["data"]["repo_dir"]) == other.resolve() == Path(info["data"]["repo_dir"])
    assert state.current().repo_dir == other.resolve()
