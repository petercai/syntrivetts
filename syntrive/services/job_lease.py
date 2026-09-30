from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from sqlalchemy import delete, or_, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from syntrive.db.models import JobLease
from syntrive.db.session import get_db_session

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 90.0
MIN_TTL_SECONDS = 5.0

Clock = Callable[[], float]
PidProbe = Callable[[int], bool]


@dataclass(frozen=True)
class HolderIdentity:
    holder_id: str
    holder_kind: str
    pid: int
    hostname: str

    @staticmethod
    def current(holder_kind: str) -> "HolderIdentity":
        return HolderIdentity(
            holder_id=uuid.uuid4().hex,
            holder_kind=holder_kind,
            pid=os.getpid(),
            hostname=socket.gethostname(),
        )


@dataclass(frozen=True)
class LeaseInfo:
    job_id: int
    holder_id: str
    holder_kind: str
    operation: Optional[str]
    pid: int
    hostname: str
    acquired_at: float
    heartbeat_at: float
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return self.expires_at < now

    @staticmethod
    def from_row(row: JobLease) -> "LeaseInfo":
        return LeaseInfo(
            job_id=row.job_id, holder_id=row.holder_id, holder_kind=row.holder_kind,
            operation=row.operation, pid=row.pid, hostname=row.hostname,
            acquired_at=row.acquired_at, heartbeat_at=row.heartbeat_at, expires_at=row.expires_at,
        )


class AcquireReason(str, Enum):
    FREE = "free"
    SAME_HOLDER = "same_holder"
    EXPIRED = "expired"
    DEAD_PID = "dead_pid"
    HELD = "held"


@dataclass(frozen=True)
class AcquireDecision:
    allowed: bool
    reason: AcquireReason


@dataclass(frozen=True)
class AcquireResult:
    ok: bool
    reason: AcquireReason
    lease: Optional[LeaseInfo] = None
    conflict: Optional[LeaseInfo] = None


class JobLeaseConflict(RuntimeError):
    def __init__(self, job_id: int, holder: LeaseInfo, now: float) -> None:
        self.job_id = job_id
        self.holder = holder
        super().__init__(format_conflict(job_id, holder, now))


def decide_acquisition(
    existing: Optional[LeaseInfo], me: HolderIdentity, now: float, pid_alive: PidProbe
) -> AcquireDecision:
    if existing is None:
        return AcquireDecision(True, AcquireReason.FREE)
    if existing.holder_id == me.holder_id:
        return AcquireDecision(True, AcquireReason.SAME_HOLDER)
    if existing.is_expired(now):
        return AcquireDecision(True, AcquireReason.EXPIRED)
    if existing.hostname == me.hostname and not pid_alive(existing.pid):
        return AcquireDecision(True, AcquireReason.DEAD_PID)
    return AcquireDecision(False, AcquireReason.HELD)


def format_conflict(job_id: int, holder: LeaseInfo, now: float) -> str:
    held_for = max(0, int(now - holder.acquired_at))
    expires_in = max(0, int(holder.expires_at - now))
    op = f" ({holder.operation})" if holder.operation else ""
    return (
        f"Job {job_id} is being modified by {holder.holder_kind}{op} "
        f"[pid {holder.pid} on {holder.hostname}] for {held_for}s; "
        f"its lease expires in {expires_in}s unless renewed. Try again later."
    )


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ctypes.get_last_error() != error_invalid_parameter
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _validated_ttl(ttl: float) -> float:
    return max(float(ttl), MIN_TTL_SECONDS)


def _take_write_lock(db, job_id: int) -> None:
    db.execute(update(JobLease).where(JobLease.job_id == job_id).values(job_id=JobLease.job_id))


def get_lease(db_path: Path, job_id: int) -> Optional[LeaseInfo]:
    with get_db_session(db_path) as db:
        row = db.get(JobLease, job_id)
        return LeaseInfo.from_row(row) if row is not None else None


def list_leases(db_path: Path) -> list[LeaseInfo]:
    with get_db_session(db_path) as db:
        return [LeaseInfo.from_row(r) for r in db.query(JobLease).order_by(JobLease.job_id).all()]


def find_blocking_lease(
    db_path: Path,
    job_ids: Iterable[int],
    *,
    tolerate_kinds: frozenset[str] = frozenset(),
    clock: Clock = time.time,
    probe: PidProbe = pid_alive,
) -> Optional[JobLeaseConflict]:
    ids = sorted(set(job_ids))
    if not ids:
        return None
    me = HolderIdentity.current("check")
    now = clock()
    with get_db_session(db_path) as db:
        rows = db.query(JobLease).filter(JobLease.job_id.in_(ids)).order_by(JobLease.job_id).all()
        leases = [LeaseInfo.from_row(r) for r in rows]
    for lease in leases:
        if lease.holder_kind in tolerate_kinds or decide_acquisition(lease, me, now, probe).allowed:
            continue
        logger.warning(
            "job_lease_check_blocked: job_id=%d holder=%s:%s operation=%s",
            lease.job_id, lease.holder_kind, lease.holder_id[:8], lease.operation,
        )
        return JobLeaseConflict(lease.job_id, lease, now)
    return None


def acquire_lease(
    db_path: Path,
    job_id: int,
    *,
    holder: HolderIdentity,
    operation: Optional[str] = None,
    ttl: float = DEFAULT_TTL_SECONDS,
    clock: Clock = time.time,
    probe: PidProbe = pid_alive,
) -> AcquireResult:
    ttl = _validated_ttl(ttl)
    now = clock()
    with get_db_session(db_path) as db:
        _take_write_lock(db, job_id)
        row = db.get(JobLease, job_id)
        existing = LeaseInfo.from_row(row) if row is not None else None
        decision = decide_acquisition(existing, holder, now, probe)
        if not decision.allowed:
            logger.warning(
                "job_lease_conflict: job_id=%d requester=%s:%s op=%s holder=%s:%s pid=%d host=%s expires_in=%.0fs",
                job_id, holder.holder_kind, holder.holder_id[:8], operation,
                existing.holder_kind, existing.holder_id[:8], existing.pid, existing.hostname,
                existing.expires_at - now,
            )
            return AcquireResult(ok=False, reason=decision.reason, conflict=existing)

        if decision.reason is AcquireReason.DEAD_PID:
            db.execute(delete(JobLease).where(JobLease.job_id == job_id, JobLease.holder_id == existing.holder_id))

        acquired_at = existing.acquired_at if decision.reason is AcquireReason.SAME_HOLDER else now
        values = dict(
            job_id=job_id, holder_id=holder.holder_id, holder_kind=holder.holder_kind,
            operation=operation, pid=holder.pid, hostname=holder.hostname,
            acquired_at=acquired_at, heartbeat_at=now, expires_at=now + ttl,
        )
        stmt = sqlite_insert(JobLease).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[JobLease.job_id],
            set_={k: stmt.excluded[k] for k in values if k != "job_id"},
            where=or_(JobLease.holder_id == holder.holder_id, JobLease.expires_at < now),
        )
        db.execute(stmt)
        db.flush()
        db.expire_all()
        won = db.get(JobLease, job_id)
        lease = LeaseInfo.from_row(won)

    if lease.holder_id != holder.holder_id:
        logger.warning(
            "job_lease_conflict: job_id=%d requester=%s:%s lost_race_to=%s:%s",
            job_id, holder.holder_kind, holder.holder_id[:8], lease.holder_kind, lease.holder_id[:8],
        )
        return AcquireResult(ok=False, reason=AcquireReason.HELD, conflict=lease)

    if decision.reason in (AcquireReason.EXPIRED, AcquireReason.DEAD_PID):
        logger.warning(
            "job_lease_takeover: job_id=%d reason=%s new=%s:%s previous=%s:%s pid=%d host=%s op=%s",
            job_id, decision.reason.value, holder.holder_kind, holder.holder_id[:8],
            existing.holder_kind, existing.holder_id[:8], existing.pid, existing.hostname, existing.operation,
        )
    event = "job_lease_renewed" if decision.reason is AcquireReason.SAME_HOLDER else "job_lease_acquired"
    logger.info(
        "%s: job_id=%d holder=%s:%s pid=%d host=%s op=%s ttl=%.0fs reason=%s",
        event, job_id, holder.holder_kind, holder.holder_id[:8], holder.pid, holder.hostname,
        operation, ttl, decision.reason.value,
    )
    return AcquireResult(ok=True, reason=decision.reason, lease=lease)


def renew_lease(
    db_path: Path, job_id: int, holder_id: str, *, ttl: float = DEFAULT_TTL_SECONDS, clock: Clock = time.time
) -> bool:
    now = clock()
    with get_db_session(db_path) as db:
        result = db.execute(
            update(JobLease)
            .where(JobLease.job_id == job_id, JobLease.holder_id == holder_id)
            .values(heartbeat_at=now, expires_at=now + _validated_ttl(ttl))
        )
        renewed = result.rowcount == 1
    logger.debug("job_lease_heartbeat: job_id=%d holder=%s renewed=%s", job_id, holder_id[:8], renewed)
    return renewed


def release_lease(db_path: Path, job_id: int, holder_id: str) -> bool:
    with get_db_session(db_path) as db:
        result = db.execute(delete(JobLease).where(JobLease.job_id == job_id, JobLease.holder_id == holder_id))
        released = result.rowcount == 1
    logger.info("job_lease_released: job_id=%d holder=%s released=%s", job_id, holder_id[:8], released)
    return released


def force_release(db_path: Path, job_id: int, *, reason: str) -> Optional[LeaseInfo]:
    with get_db_session(db_path) as db:
        _take_write_lock(db, job_id)
        row = db.get(JobLease, job_id)
        if row is None:
            logger.info("job_lease_force_release: job_id=%d reason=%s -- no lease", job_id, reason)
            return None
        removed = LeaseInfo.from_row(row)
        db.delete(row)
    logger.warning(
        "job_lease_takeover: job_id=%d reason=forced detail=%r previous=%s:%s pid=%d host=%s op=%s",
        job_id, reason, removed.holder_kind, removed.holder_id[:8], removed.pid, removed.hostname, removed.operation,
    )
    return removed


class LeaseHandle:
    def __init__(self, job_id: int, holder: HolderIdentity) -> None:
        self.job_id = job_id
        self.holder = holder
        self._lost = threading.Event()

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def _mark_lost(self) -> None:
        self._lost.set()


def _heartbeat_loop(
    db_path: Path, handle: LeaseHandle, ttl: float, stop: threading.Event, clock: Clock
) -> None:
    interval = ttl / 3.0
    while not stop.wait(interval):
        try:
            if renew_lease(db_path, handle.job_id, handle.holder.holder_id, ttl=ttl, clock=clock):
                continue
            logger.error(
                "job_lease_heartbeat_failed: job_id=%d holder=%s:%s reason=lease_lost",
                handle.job_id, handle.holder.holder_kind, handle.holder.holder_id[:8],
            )
        except Exception as exc:  # noqa: BLE001 -- a DB hiccup must not kill the worker thread silently
            logger.error(
                "job_lease_heartbeat_failed: job_id=%d holder=%s:%s reason=%s",
                handle.job_id, handle.holder.holder_kind, handle.holder.holder_id[:8], exc,
            )
            continue
        handle._mark_lost()
        return


_held = threading.local()


def _held_map() -> dict[tuple[str, int], LeaseHandle]:
    if not hasattr(_held, "leases"):
        _held.leases = {}
    return _held.leases


@contextmanager
def hold_job_lease(
    db_path: Path,
    job_id: int,
    *,
    holder_kind: str,
    operation: Optional[str] = None,
    ttl: float = DEFAULT_TTL_SECONDS,
    clock: Clock = time.time,
    probe: PidProbe = pid_alive,
) -> Iterator[LeaseHandle]:
    key = (str(Path(db_path).resolve()), job_id)
    held = _held_map()
    if key in held:
        outer = held[key]
        logger.debug(
            "job_lease_reentered: job_id=%d holder=%s:%s operation=%s",
            job_id, outer.holder.holder_kind, outer.holder.holder_id[:8], operation,
        )
        yield outer
        return

    ttl = _validated_ttl(ttl)
    holder = HolderIdentity.current(holder_kind)
    result = acquire_lease(db_path, job_id, holder=holder, operation=operation, ttl=ttl, clock=clock, probe=probe)
    if not result.ok:
        raise JobLeaseConflict(job_id, result.conflict, clock())

    handle = LeaseHandle(job_id, holder)
    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat_loop, args=(db_path, handle, ttl, stop, clock),
        name=f"job-lease-{job_id}", daemon=True,
    )
    beat.start()
    held[key] = handle
    try:
        yield handle
    finally:
        held.pop(key, None)
        stop.set()
        beat.join(timeout=5.0)
        try:
            release_lease(db_path, job_id, holder.holder_id)
        except Exception as exc:  # noqa: BLE001 -- never mask the body's own exception; TTL recovers the row
            logger.error("job_lease_release_failed: job_id=%d holder=%s error=%s", job_id, holder.holder_id[:8], exc)


@contextmanager
def hold_job_leases(
    db_path: Path,
    job_ids: Iterable[int],
    *,
    holder_kind: str,
    operation: Optional[str] = None,
    ttl: float = DEFAULT_TTL_SECONDS,
    clock: Clock = time.time,
    probe: PidProbe = pid_alive,
) -> Iterator[list[LeaseHandle]]:
    with ExitStack() as stack:
        handles = [
            stack.enter_context(hold_job_lease(
                db_path, job_id, holder_kind=holder_kind, operation=operation,
                ttl=ttl, clock=clock, probe=probe,
            ))
            for job_id in sorted(set(job_ids))
        ]
        yield handles
