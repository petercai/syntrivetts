from __future__ import annotations

import json
import logging
import subprocess
import time

from syntrive.adapters.tts import ipc
from syntrive.adapters.tts.ipc import WorkerHandle, _terminate_process


class _StubStream:
    def readline(self) -> str:
        return ""


class _StubProcess:
    def __init__(self) -> None:
        self.pid = -1
        self.stdout = _StubStream()
        self.stderr = _StubStream()

    def poll(self):
        return None


def _handle() -> WorkerHandle:
    return WorkerHandle("f5tts", _StubProcess(), logging.getLogger("test.worker"))


class TestIdleGuard:
    def test_idle_worker_is_evictable_past_threshold(self) -> None:
        h = _handle()
        time.sleep(0.05)
        assert h.is_idle_beyond(0.01) is True

    def test_fresh_worker_is_not_idle(self) -> None:
        h = _handle()
        assert h.is_idle_beyond(5.0) is False

    def test_in_flight_request_is_never_idle(self) -> None:
        h = _handle()
        h.begin_request()
        time.sleep(0.05)
        assert h.is_idle_beyond(0.01) is False

    def test_idle_clock_restarts_after_request_completes(self) -> None:
        h = _handle()
        h.begin_request()
        time.sleep(0.05)
        h.end_request()
        assert h.is_idle_beyond(0.5) is False
        time.sleep(0.05)
        assert h.is_idle_beyond(0.01) is True

    def test_nested_requests_need_matching_ends(self) -> None:
        h = _handle()
        h.begin_request()
        h.begin_request()
        h.end_request()
        time.sleep(0.05)
        assert h.is_idle_beyond(0.01) is False
        h.end_request()
        time.sleep(0.05)
        assert h.is_idle_beyond(0.01) is True


class _NeverReapedProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="stub", timeout=timeout)


class TestTerminateProcess:
    def test_second_wait_timeout_after_kill_does_not_raise(self) -> None:
        process = _NeverReapedProcess()
        _terminate_process(process, graceful_seconds=0.01)
        assert process.terminate_calls == 1
        assert process.kill_calls == 1

    def test_already_exited_process_is_a_no_op(self) -> None:
        class _ExitedProcess(_NeverReapedProcess):
            def poll(self):
                return 0

        process = _ExitedProcess()
        _terminate_process(process, graceful_seconds=0.01)
        assert process.terminate_calls == 0
        assert process.kill_calls == 0


class TestLooksLikeOomError:
    def test_matches_mps_out_of_memory_message(self) -> None:
        text = (
            "MPS backend out of memory (MPS allocated: 13.49 GiB, other allocations: "
            "9.25 GiB, max allowed: 22.64 GiB). Tried to allocate 9.04 MiB on private pool."
        )
        assert ipc._looks_like_oom_error(text) is True

    def test_matches_cuda_out_of_memory_message_case_insensitively(self) -> None:
        assert ipc._looks_like_oom_error("CUDA Out Of Memory. Tried to allocate 20.00 MiB") is True

    def test_does_not_match_unrelated_errors(self) -> None:
        assert ipc._looks_like_oom_error("reference text file missing: foo.txt") is False

    def test_handles_empty_and_none(self) -> None:
        assert ipc._looks_like_oom_error("") is False
        assert ipc._looks_like_oom_error(None) is False


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, s: str) -> None:
        self.written.append(s)

    def flush(self) -> None:
        pass


class _FakeReaderEchoingError:
    def __init__(self, stdin: _FakeStdin, error_text: str) -> None:
        self._stdin = stdin
        self._error_text = error_text

    def read_line(self, timeout: float):
        payload = json.loads(self._stdin.written[-1])
        response = {"request_id": payload["request_id"], "status": "ERROR", "error": self._error_text}
        return json.dumps(response) + "\n"


class _FakeHandleForSendRequest:
    def __init__(self, error_text: str) -> None:
        self.stdin = _FakeStdin()
        self.process = type("P", (), {"stdin": self.stdin, "poll": lambda self: None})()
        self.reader = _FakeReaderEchoingError(self.stdin, error_text)

    def begin_request(self) -> None:
        pass

    def end_request(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True


class _FakePool:
    def __init__(self) -> None:
        self.evicted: list[tuple] = []

    def evict(self, engine_name: str, reason: str) -> None:
        self.evicted.append((engine_name, reason))


class TestOomEviction:
    def _send(self, error_text: str):
        pool = _FakePool()
        fake_self = type(
            "FakeAdapter", (), {
                "engine_name": "indextts", "pool": pool, "request_timeout_seconds": 5.0,
                "oom_cooldown_seconds": 0.0,
            }
        )()
        handle = _FakeHandleForSendRequest(error_text)
        response = ipc.SubprocessTTSEngineAdapter._send_request(fake_self, handle, {"cmd": "SYNTHESIZE"})
        return response, pool

    def test_oom_error_response_evicts_the_worker(self) -> None:
        response, pool = self._send("MPS backend out of memory (MPS allocated: 13.49 GiB...)")
        assert response["status"] == "ERROR"
        assert pool.evicted == [("indextts", "out_of_memory")]

    def test_non_oom_error_response_does_not_evict(self) -> None:
        response, pool = self._send("reference text file missing: foo.txt")
        assert response["status"] == "ERROR"
        assert pool.evicted == []

    def test_oom_error_response_sleeps_for_the_configured_cooldown(self, monkeypatch) -> None:
        sleeps = []
        monkeypatch.setattr(ipc.time, "sleep", lambda seconds: sleeps.append(seconds))
        pool = _FakePool()
        fake_self = type(
            "FakeAdapter", (), {
                "engine_name": "indextts", "pool": pool, "request_timeout_seconds": 5.0,
                "oom_cooldown_seconds": 5.0,
            }
        )()
        handle = _FakeHandleForSendRequest("CUDA out of memory")

        ipc.SubprocessTTSEngineAdapter._send_request(fake_self, handle, {"cmd": "SYNTHESIZE"})

        assert sleeps == [5.0]

    def test_zero_cooldown_skips_sleeping_entirely(self, monkeypatch) -> None:
        sleeps = []
        monkeypatch.setattr(ipc.time, "sleep", lambda seconds: sleeps.append(seconds))
        pool = _FakePool()
        fake_self = type(
            "FakeAdapter", (), {
                "engine_name": "indextts", "pool": pool, "request_timeout_seconds": 5.0,
                "oom_cooldown_seconds": 0.0,
            }
        )()
        handle = _FakeHandleForSendRequest("CUDA out of memory")

        ipc.SubprocessTTSEngineAdapter._send_request(fake_self, handle, {"cmd": "SYNTHESIZE"})

        assert sleeps == []
