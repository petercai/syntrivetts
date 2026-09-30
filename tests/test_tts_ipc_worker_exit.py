from __future__ import annotations

import time
import venv
from pathlib import Path

import pytest

from syntrive.adapters.tts.ipc import (
    SubprocessTTSEngineAdapter,
    TTSWorkerPool,
    WorkerCommunicationError,
    summarize_stderr,
)

REASON = "RuntimeError: [enforce fail at alloc_cpu.cpp:121] DefaultCPUAllocator: not enough memory"
RUNNER = f'''
import json, os, sys, time
mode = os.environ.get("FAKE_RUNNER_MODE", "ok")
for line in sys.stdin:
    request = json.loads(line)
    cmd = request["cmd"].lower()
    if mode == "crash_" + cmd:
        print("loading checkpoint ...", file=sys.stderr, flush=True)
        print("Traceback (most recent call last):", file=sys.stderr, flush=True)
        print({REASON!r}, file=sys.stderr, flush=True)
        sys.exit(3)
    if mode == "hang_" + cmd:
        time.sleep(120)
    print(json.dumps({{"request_id": request["request_id"], "status": "OK"}}), flush=True)
'''
FAST = 30


@pytest.fixture(scope="module")
def engines_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("engines")
    engine = root / "fake"
    venv.create(engine / ".venv", with_pip=False)
    (engine / "runner.py").write_text(RUNNER, encoding="utf-8")
    return root


@pytest.fixture()
def make_adapter(engines_root, monkeypatch):
    pool = TTSWorkerPool(engines_root)
    monkeypatch.setattr(SubprocessTTSEngineAdapter, "_shared_pool", pool)

    def make(mode: str, timeout: float = 120) -> SubprocessTTSEngineAdapter:
        monkeypatch.setenv("FAKE_RUNNER_MODE", mode)
        return SubprocessTTSEngineAdapter("fake", request_timeout_seconds=timeout)

    yield make
    pool.shutdown()


def test_crash_during_init_reports_exit_and_reason_fast(make_adapter):
    adapter = make_adapter("crash_init")
    started = time.monotonic()
    with pytest.raises(WorkerCommunicationError) as caught:
        adapter.ensure_worker()
    elapsed = time.monotonic() - started
    message = str(caught.value)
    assert "exited (exit_code=3) while handling INIT" in message
    assert "DefaultCPUAllocator: not enough memory" in message
    assert "timed out" not in message
    assert elapsed < FAST, f"crash took {elapsed:.1f}s to report"
    assert adapter.pool._active is None


def test_crash_during_synthesize_sets_last_error(make_adapter, tmp_path):
    adapter = make_adapter("crash_synthesize")
    started = time.monotonic()
    assert adapter.synthesize("Hello.", output_path=str(tmp_path / "a.wav")) is None
    assert time.monotonic() - started < FAST
    assert "exited (exit_code=3) while handling SYNTHESIZE" in adapter.last_error
    assert "not enough memory" in adapter.last_error
    assert adapter.pool._active is None


def test_a_real_hang_is_still_a_timeout(make_adapter, tmp_path):
    adapter = make_adapter("hang_synthesize", timeout=2)
    assert adapter.synthesize("Hello.", output_path=str(tmp_path / "a.wav")) is None
    assert "timed out after 2" in adapter.last_error and "SYNTHESIZE" in adapter.last_error
    assert adapter.pool._active is None


def test_a_healthy_worker_still_answers(make_adapter, tmp_path):
    adapter = make_adapter("ok")
    out = tmp_path / "a.wav"
    assert adapter.synthesize("Hello.", output_path=str(out)) == str(out)
    assert adapter.last_error == ""


def test_retry_after_a_crash_gets_a_fresh_worker(make_adapter, monkeypatch, tmp_path):
    adapter = make_adapter("crash_synthesize")
    assert adapter.synthesize("Hello.", output_path=str(tmp_path / "a.wav")) is None
    monkeypatch.setenv("FAKE_RUNNER_MODE", "ok")
    assert adapter.synthesize("Hello.", output_path=str(tmp_path / "b.wav")) == str(tmp_path / "b.wav")


class TestSummarizeStderr:
    def test_keeps_the_last_lines_on_one_line(self) -> None:
        assert summarize_stderr(["a", "b", "c", "d", "e"], max_lines=2) == "d | e"

    def test_cuts_from_the_front_so_the_exception_survives(self) -> None:
        text = summarize_stderr(["x" * 50, "Error: the end"], max_chars=20)
        assert text.endswith("Error: the end") and len(text) == 20 and text.startswith("…")

    def test_empty(self) -> None:
        assert summarize_stderr([]) == ""
