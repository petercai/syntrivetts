from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from syntrive.services import job_lease as jl
from syntrive.services.job_lease import (
    AcquireReason,
    HolderIdentity,
    JobLeaseConflict,
    LeaseInfo,
    acquire_lease,
    decide_acquisition,
    force_release,
    get_lease,
    hold_job_lease,
    list_leases,
    pid_alive,
    release_lease,
    renew_lease,
)

JOB = 1


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _holder(kind: str, host: str = "host-a", pid: int = 4242) -> HolderIdentity:
    return HolderIdentity(holder_id=f"{kind}-{host}-{pid}".ljust(32, "0"), holder_kind=kind, pid=pid, hostname=host)


def _alive(_pid: int) -> bool:
    return True


def _dead(_pid: int) -> bool:
    return False


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    from syntrive.db.session import ensure_schema, make_engine

    path = tmp_path / "syntrivetts.db"
    ensure_schema(make_engine(path), seed=False)
    return path


class TestDecideAcquisition:
    def _lease(self, **kw) -> LeaseInfo:
        base = dict(job_id=JOB, holder_id="other", holder_kind="tui", operation=None, pid=7,
                    hostname="host-a", acquired_at=0.0, heartbeat_at=0.0, expires_at=100.0)
        base.update(kw)
        return LeaseInfo(**base)

    def test_free_same_holder_expired_dead_and_held(self):
        me = _holder("webui")
        assert decide_acquisition(None, me, 50, _alive).reason is AcquireReason.FREE
        assert decide_acquisition(self._lease(holder_id=me.holder_id), me, 50, _alive).reason is AcquireReason.SAME_HOLDER
        assert decide_acquisition(self._lease(), me, 150, _alive).reason is AcquireReason.EXPIRED
        assert decide_acquisition(self._lease(), me, 50, _dead).reason is AcquireReason.DEAD_PID
        held = decide_acquisition(self._lease(), me, 50, _alive)
        assert not held.allowed and held.reason is AcquireReason.HELD

    def test_remote_holder_is_never_judged_by_a_local_pid_probe(self):
        me = _holder("webui", host="host-a")
        remote = self._lease(hostname="host-b")
        decision = decide_acquisition(remote, me, 50, _dead)
        assert not decision.allowed and decision.reason is AcquireReason.HELD


class TestAcquireRenewRelease:
    def test_second_holder_is_refused_while_first_is_live(self, db_path: Path):
        clock = FakeClock()
        tui, web = _holder("tui"), _holder("webui")

        first = acquire_lease(db_path, JOB, holder=tui, operation="step:transcript_clean", ttl=60, clock=clock, probe=_alive)
        second = acquire_lease(db_path, JOB, holder=web, ttl=60, clock=clock, probe=_alive)

        assert first.ok and first.reason is AcquireReason.FREE
        assert not second.ok and second.reason is AcquireReason.HELD
        assert second.conflict.holder_id == tui.holder_id
        row = get_lease(db_path, JOB)
        assert row.holder_id == tui.holder_id and row.operation == "step:transcript_clean"
        assert row.expires_at == clock.now + 60

    def test_same_holder_reacquire_renews_and_keeps_acquired_at(self, db_path: Path):
        clock = FakeClock()
        tui = _holder("tui")
        acquire_lease(db_path, JOB, holder=tui, ttl=60, clock=clock, probe=_alive)
        clock.advance(30)
        again = acquire_lease(db_path, JOB, holder=tui, ttl=60, clock=clock, probe=_alive)

        row = get_lease(db_path, JOB)
        assert again.ok and again.reason is AcquireReason.SAME_HOLDER
        assert row.acquired_at == clock.now - 30
        assert row.heartbeat_at == clock.now and row.expires_at == clock.now + 60

    def test_expired_lease_is_taken_over(self, db_path: Path, caplog):
        clock = FakeClock()
        acquire_lease(db_path, JOB, holder=_holder("tui", host="host-b"), ttl=60, clock=clock, probe=_alive)
        clock.advance(61)
        web = _holder("webui")
        with caplog.at_level("WARNING", logger="syntrive.services.job_lease"):
            result = acquire_lease(db_path, JOB, holder=web, ttl=60, clock=clock, probe=_alive)

        assert result.ok and result.reason is AcquireReason.EXPIRED
        assert get_lease(db_path, JOB).holder_id == web.holder_id
        assert "job_lease_takeover" in caplog.text and "reason=expired" in caplog.text

    def test_dead_pid_on_same_host_is_taken_over_before_expiry(self, db_path: Path):
        clock = FakeClock()
        acquire_lease(db_path, JOB, holder=_holder("tui", pid=111), ttl=600, clock=clock, probe=_alive)
        web = _holder("webui", pid=222)
        result = acquire_lease(db_path, JOB, holder=web, ttl=600, clock=clock, probe=_dead)

        assert result.ok and result.reason is AcquireReason.DEAD_PID
        assert get_lease(db_path, JOB).pid == 222

    def test_renew_and_release_only_work_for_the_holder(self, db_path: Path):
        clock = FakeClock()
        tui, web = _holder("tui"), _holder("webui")
        acquire_lease(db_path, JOB, holder=tui, ttl=60, clock=clock, probe=_alive)

        assert not renew_lease(db_path, JOB, web.holder_id, ttl=60, clock=clock)
        assert not release_lease(db_path, JOB, web.holder_id)
        assert get_lease(db_path, JOB).holder_id == tui.holder_id

        clock.advance(10)
        assert renew_lease(db_path, JOB, tui.holder_id, ttl=60, clock=clock)
        assert get_lease(db_path, JOB).expires_at == clock.now + 60
        assert release_lease(db_path, JOB, tui.holder_id)
        assert get_lease(db_path, JOB) is None

    def test_force_release_removes_any_holder_and_is_audited(self, db_path: Path, caplog):
        clock = FakeClock()
        acquire_lease(db_path, JOB, holder=_holder("tui", host="host-b"), ttl=600, clock=clock, probe=_alive)
        with caplog.at_level("WARNING", logger="syntrive.services.job_lease"):
            removed = force_release(db_path, JOB, reason="operator unlock")
        assert removed is not None and removed.hostname == "host-b"
        assert get_lease(db_path, JOB) is None
        assert "reason=forced" in caplog.text
        assert force_release(db_path, JOB, reason="again") is None

    def test_leases_are_per_job(self, db_path: Path):
        clock = FakeClock()
        assert acquire_lease(db_path, 1, holder=_holder("tui"), clock=clock, probe=_alive).ok
        assert acquire_lease(db_path, 2, holder=_holder("webui"), clock=clock, probe=_alive).ok
        assert [lease.job_id for lease in list_leases(db_path)] == [1, 2]

    def test_ttl_is_clamped_to_minimum(self, db_path: Path):
        clock = FakeClock()
        acquire_lease(db_path, JOB, holder=_holder("tui"), ttl=0.1, clock=clock, probe=_alive)
        assert get_lease(db_path, JOB).expires_at == clock.now + jl.MIN_TTL_SECONDS


class TestHoldJobLease:
    def test_conflict_raises_with_user_readable_holder_info(self, db_path: Path):
        clock = FakeClock()
        acquire_lease(db_path, JOB, holder=_holder("tts_batch", host="gpu-box"), operation="synthesis", ttl=90, clock=clock, probe=_alive)
        with pytest.raises(JobLeaseConflict) as info:
            with hold_job_lease(db_path, JOB, holder_kind="webui", clock=clock, probe=_alive):
                pytest.fail("body must not run")
        message = str(info.value)
        assert "tts_batch (synthesis)" in message and "gpu-box" in message and "expires in 90s" in message

    def test_released_even_when_body_raises(self, db_path: Path):
        with pytest.raises(ValueError):
            with hold_job_lease(db_path, JOB, holder_kind="cli", operation="test"):
                assert get_lease(db_path, JOB) is not None
                raise ValueError("boom")
        assert get_lease(db_path, JOB) is None

    def test_heartbeat_keeps_extending_the_lease(self, db_path: Path, monkeypatch):
        with hold_job_lease(db_path, JOB, holder_kind="tts_batch", ttl=6) as handle:
            first = get_lease(db_path, JOB).expires_at
            time.sleep(2.6)
            second = get_lease(db_path, JOB).expires_at
            assert second > first
            assert not handle.lost
        assert get_lease(db_path, JOB) is None

    def test_heartbeat_marks_handle_lost_after_takeover(self, db_path: Path):
        with hold_job_lease(db_path, JOB, holder_kind="webui", ttl=6) as handle:
            force_release(db_path, JOB, reason="test takeover")
            acquire_lease(db_path, JOB, holder=_holder("tui", host="other-host"), ttl=60, probe=_alive)
            deadline = time.time() + 5
            while not handle.lost and time.time() < deadline:
                time.sleep(0.1)
            assert handle.lost
        assert get_lease(db_path, JOB).hostname == "other-host"


class TestPidAlive:
    def test_current_process_is_alive(self):
        assert pid_alive(os.getpid())

    def test_exited_process_is_dead(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        assert not pid_alive(proc.pid)

    def test_probe_does_not_kill_a_live_process(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            assert pid_alive(proc.pid)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_non_positive_pid_is_dead(self):
        assert not pid_alive(0)


def test_concurrent_acquirers_have_exactly_one_winner(db_path: Path):
    results: list[bool] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(12)

    def worker(i: int) -> None:
        barrier.wait()
        try:
            results.append(acquire_lease(db_path, JOB, holder=_holder(f"w{i}", pid=1000 + i), probe=_alive).ok)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert results.count(True) == 1 and len(results) == 12
