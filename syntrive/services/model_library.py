from __future__ import annotations

import itertools
import json
import logging
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from syntrive.adapters.tts import env_provisioner, model_catalog, model_downloader

from _shared import catalog as _catalog  # noqa: E402
from _shared import manifest as _manifest  # noqa: E402
from _shared.runtime_env import TTS_CACHE_DIR  # noqa: E402

logger = logging.getLogger(__name__)

LOW_SPACE_BYTES = 5 * 2**30
_DEVICE_CONFIG_PATH = TTS_CACHE_DIR / ".model_manager.json"
_XTTS_FINETUNED_DIRNAME = "models--drewThomasson--fineTunedTTSModels"
_CHECKPOINT_EXTS = {".pth", ".pt", ".bin", ".safetensors", ".ckpt"}


def format_bytes(n: Optional[int]) -> str:
    if not n:
        return "0 B"
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


def format_revision(rev: Optional[str]) -> str:
    return (rev or "")[:8] or "—"


@dataclass(frozen=True)
class RowModel:
    engine: str
    model_id: str
    label: str
    approx_gb: float
    is_default: bool
    downloaded: bool
    state: str
    size_bytes: int
    revision: str
    updated_at: str
    kind: str = "model"

    @property
    def status_glyph(self) -> str:
        if not self.downloaded and self.state != "unverified":
            return "○"
        if self.state in ("partial", "failed"):
            return "▲"
        if self.state == "unverified":
            return "◐"
        return "●"

    @property
    def detail_line(self) -> str:
        bits = []
        if self.downloaded or self.state:
            bits.append(format_bytes(self.size_bytes))
        if self.revision:
            bits.append(f"rev {format_revision(self.revision)}")
        if self.updated_at:
            bits.append(self.updated_at)
        if not bits:
            bits.append(f"~{self.approx_gb} GB")
        return "  ·  ".join(bits)


def build_row_model(spec, entry: Optional[dict], downloaded: bool) -> RowModel:
    entry = entry or {}
    return RowModel(
        engine=spec.engine, model_id=spec.model_id, label=spec.label,
        approx_gb=float(spec.approx_gb or 0), is_default=bool(spec.is_default),
        downloaded=downloaded, state=entry.get("state", ""),
        size_bytes=int(entry.get("bytes_on_disk") or 0),
        revision=entry.get("revision") or spec.revision or "",
        updated_at=entry.get("updated_at") or "",
    )


def build_resource_row(res: dict, engine: str, entry: Optional[dict]) -> RowModel:
    entry = entry or {}
    state = entry.get("state", "")
    return RowModel(
        engine=engine, model_id=res["id"], label=res.get("label", res["id"]),
        approx_gb=0.0, is_default=False,
        downloaded=state not in ("", "failed"), state=state,
        size_bytes=int(entry.get("bytes_on_disk") or 0),
        revision=entry.get("revision") or "", updated_at=entry.get("updated_at") or "",
        kind="resource",
    )


def disk_free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError:
        return 0


def projected_free_after(free_bytes: int, approx_gb: float) -> int:
    return free_bytes - int(approx_gb * 2**30)


def load_compute_device(default: str) -> str:
    try:
        data = json.loads(_DEVICE_CONFIG_PATH.read_text(encoding="utf-8"))
        dev = data.get("compute_device")
        if dev in env_provisioner.VALID_DEVICES:
            return dev
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return default


def save_compute_device(device: str) -> None:
    try:
        _DEVICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DEVICE_CONFIG_PATH.write_text(json.dumps({"compute_device": device}), encoding="utf-8")
    except OSError as exc:
        logger.debug("could not persist compute device: %s", exc)


def scan_finetuned_xtts() -> list:
    base = TTS_CACHE_DIR / _XTTS_FINETUNED_DIRNAME
    if not base.is_dir():
        return []
    found = set()
    for path in base.rglob("*"):
        if not path.is_dir():
            continue
        if any(f.is_file() and f.suffix.lower() in _CHECKPOINT_EXTS for f in path.iterdir()):
            found.add(path.name)
    return sorted(found)


def total_cache_bytes() -> int:
    if not TTS_CACHE_DIR.is_dir():
        return 0
    total = 0
    for p in TTS_CACHE_DIR.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


@dataclass
class Task:
    name: str
    fn: Callable[[Callable[[str], None]], object]
    on_done: Callable[[object], None]


class ActionQueue:
    def __init__(self, log: Callable[[str], None], schedule: Callable[[Callable], None]):
        self._q: "queue.Queue[Task]" = queue.Queue()
        self._log = log
        self._schedule = schedule
        self._pending: set = set()
        self._running: Optional[str] = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, key: str, task: Task) -> bool:
        if key in self._pending or key == self._running:
            self._log(f"queue_skip: {key} already queued/running")
            return False
        self._pending.add(key)
        self._q.put((key, task))
        self._log(f"model_manager_queue: enqueued={task.name} pending={sorted(self._pending)}")
        return True

    def is_busy(self, key: str) -> bool:
        return key in self._pending or key == self._running

    def _loop(self) -> None:
        while True:
            key, task = self._q.get()
            self._pending.discard(key)
            self._running = key
            self._log(f"model_manager_run: {task.name}")
            try:
                result = task.fn(self._log)
            except Exception as exc:  # noqa: BLE001 -- keep the worker alive
                logger.exception("action %s failed", task.name)
                self._log(f"model_manager_error: {task.name} error={exc}")
                result = exc
            self._running = None
            self._schedule(lambda r=result, t=task: t.on_done(r))
            self._q.task_done()


_STATUS_WORDS = {"○": "missing", "●": "ready", "▲": "attention", "◐": "unverified"}


def row_status(rm: RowModel) -> str:
    return _STATUS_WORDS[rm.status_glyph]


@dataclass(frozen=True)
class EngineGroup:
    engine: str
    label: str
    has_venv: bool
    env_ready: Optional[bool]
    rows: tuple[RowModel, ...]
    finetuned: tuple[str, ...]

    @property
    def downloaded(self) -> int:
        return sum(1 for r in self.rows if r.kind == "model" and r.downloaded)

    @property
    def models(self) -> int:
        return sum(1 for r in self.rows if r.kind == "model")

    @property
    def size_bytes(self) -> int:
        return sum(r.size_bytes for r in self.rows)


def read_engine_group(engine: str, raw: Optional[dict] = None) -> EngineGroup:
    raw = raw if raw is not None else _catalog.load_raw()
    meta = _catalog.engine_meta(raw, engine)
    has_venv = model_catalog.has_venv(engine)
    rows = [
        build_row_model(spec, _manifest.get(engine, spec.model_id), model_catalog.is_downloaded(spec))
        for spec in model_catalog.models_for(engine)
    ] + [
        build_resource_row(res, engine, _manifest.get(engine, res["id"]))
        for res in _catalog.resources_for(raw, engine)
    ]
    return EngineGroup(
        engine=engine, label=meta.get("label", engine), has_venv=has_venv,
        env_ready=(env_provisioner.venv_state(engine) == env_provisioner.STATE_PROVISIONED) if has_venv else None,
        rows=tuple(rows), finetuned=tuple(scan_finetuned_xtts()) if engine == "xtts" else (),
    )


def read_model_catalog() -> tuple[EngineGroup, ...]:
    raw = _catalog.load_raw()
    groups = tuple(read_engine_group(engine, raw) for engine in model_catalog.engines())
    logger.debug("model_catalog_read: engines=%d models=%d downloaded=%d",
                 len(groups), sum(g.models for g in groups), sum(g.downloaded for g in groups))
    return groups


def find_row(engine: str, item_id: str) -> Optional[RowModel]:
    if engine not in model_catalog.engines():
        return None
    return next((r for r in read_engine_group(engine).rows if r.model_id == item_id), None)


def major_files(engine: str, model_id: str) -> list:
    return model_downloader.list_major_files(engine, model_id)


@dataclass(frozen=True)
class DiskCheck:
    need_bytes: int
    free_bytes: int
    after_bytes: int

    @property
    def low(self) -> bool:
        return self.after_bytes < LOW_SPACE_BYTES


def disk_check(approx_gb: float, path: Path = TTS_CACHE_DIR) -> DiskCheck:
    free = disk_free_bytes(path)
    return DiskCheck(int(approx_gb * 2**30), free, projected_free_after(free, approx_gb))


def compute_device() -> tuple[str, str]:
    detected = env_provisioner.detect_compute_device()
    return load_compute_device(detected), detected


TASK_QUEUED, TASK_RUNNING, TASK_DONE, TASK_FAILED = "queued", "running", "done", "failed"
LOG_TAIL = 200


class UnknownModel(ValueError):
    pass


@dataclass(frozen=True)
class TaskView:
    id: int
    key: str
    name: str
    state: str
    submitted_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    summary: str
    log: tuple[str, ...]


@dataclass
class _Record:
    id: int
    key: str
    name: str
    state: str = TASK_QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    summary: str = ""
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_TAIL))

    def view(self) -> TaskView:
        return TaskView(self.id, self.key, self.name, self.state, self.submitted_at,
                        self.started_at, self.finished_at, self.summary, tuple(self.log))


def task_outcome(result) -> tuple[bool, str]:
    if isinstance(result, Exception):
        return False, str(result)
    if isinstance(result, model_downloader.UpdateStatus):
        return True, result.reason or ("update available" if result.update_available else "up to date")
    if not getattr(result, "ok", True):
        return False, getattr(result, "error", "") or "failed -- see the log"
    size = getattr(result, "bytes_on_disk", 0)
    return True, format_bytes(size) if size else ""


class ModelTasks:
    def __init__(self, *, keep: int = 30) -> None:
        self._lock = threading.Lock()
        self._records: dict[int, _Record] = {}
        self._ids = itertools.count(1)
        self._keep = keep
        self._queue = ActionQueue(lambda m: logger.info("%s", m), lambda cb: cb())

    def submit(self, key: str, name: str, fn: Callable[[Callable[[str], None]], object]) -> Optional[TaskView]:
        rec = _Record(id=next(self._ids), key=key, name=name)

        def run(_queue_log):
            with self._lock:
                rec.state, rec.started_at = TASK_RUNNING, time.time()

            def log(line: str) -> None:
                rec.log.append(line)
                logger.info("model_task_log: id=%d %s", rec.id, line)
            return fn(log)

        def done(result) -> None:
            ok, summary = task_outcome(result)
            with self._lock:
                rec.state = TASK_DONE if ok else TASK_FAILED
                rec.summary, rec.finished_at = summary, time.time()
            logger.info("model_task_finished: id=%d key=%s state=%s summary=%s", rec.id, key, rec.state, summary)

        with self._lock:
            self._records[rec.id] = rec
        if not self._queue.submit(key, Task(name, run, done)):
            with self._lock:
                self._records.pop(rec.id, None)
            logger.info("model_task_refused: key=%s reason=busy", key)
            return None
        self._trim()
        logger.info("model_task_queued: id=%d key=%s name=%s", rec.id, key, name)
        return rec.view()

    def views(self) -> list[TaskView]:
        with self._lock:
            return [r.view() for r in sorted(self._records.values(), key=lambda r: -r.id)]

    def get(self, task_id: int) -> Optional[TaskView]:
        with self._lock:
            rec = self._records.get(task_id)
            return rec.view() if rec else None

    def busy_keys(self) -> set[str]:
        with self._lock:
            return {r.key for r in self._records.values() if r.state in (TASK_QUEUED, TASK_RUNNING)}

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while self.busy_keys():
            if time.time() > deadline:
                return False
            time.sleep(0.02)
        return True

    def _trim(self) -> None:
        with self._lock:
            finished = sorted(r.id for r in self._records.values() if r.state in (TASK_DONE, TASK_FAILED))
            for old in finished[:-self._keep] if len(finished) > self._keep else []:
                self._records.pop(old, None)

    def _require(self, engine: str, item_id: str, kind: Optional[str] = None) -> RowModel:
        row = find_row(engine, item_id)
        if row is None or (kind is not None and row.kind != kind):
            raise UnknownModel(f"{engine}/{item_id} is not a catalogued {kind or 'item'}")
        return row

    def download(self, engine: str, model_id: str, *, update: bool = False) -> Optional[TaskView]:
        self._require(engine, model_id, "model")
        verb = "update" if update else "download"
        return self.submit(f"model:{engine}/{model_id}", f"{verb} {engine}/{model_id}",
                           lambda log: model_downloader.download(engine, model_id, log=log))

    def check_update(self, engine: str, model_id: str) -> Optional[TaskView]:
        self._require(engine, model_id, "model")
        return self.submit(f"check:{engine}/{model_id}", f"check-update {engine}/{model_id}",
                           lambda log: model_downloader.check_update(engine, model_id, log=log))

    def remove(self, engine: str, item_id: str) -> Optional[TaskView]:
        resource = self._require(engine, item_id).kind == "resource"
        return self.submit(f"{'res' if resource else 'model'}:{engine}/{item_id}", f"remove {engine}/{item_id}",
                           lambda log: model_downloader.remove(engine, item_id, is_resource=resource, log=log))

    def download_resource(self, engine: str, resource_id: str) -> Optional[TaskView]:
        self._require(engine, resource_id, "resource")
        return self.submit(f"res:{engine}/{resource_id}", f"download {engine}/{resource_id}",
                           lambda log: model_downloader.download_resource(engine, resource_id, log=log))

    def provision(self, engine: str, device: str) -> Optional[TaskView]:
        if engine not in model_catalog.engines() or not model_catalog.has_venv(engine):
            raise UnknownModel(f"{engine} has no environment to set up")
        if device not in env_provisioner.VALID_DEVICES:
            raise ValueError(f"device must be one of {env_provisioner.VALID_DEVICES}")
        return self.submit(f"env:{engine}", f"provision {engine} ({device})",
                           lambda log: env_provisioner.provision(engine, device, log=log))


def model_counts() -> tuple[int, int]:
    specs = [spec for engine in model_catalog.engines() for spec in model_catalog.models_for(engine)]
    return sum(1 for spec in specs if model_catalog.is_downloaded(spec)), len(specs)


def row_keys(row: RowModel) -> tuple[str, ...]:
    if row.kind == "resource":
        return (f"res:{row.engine}/{row.model_id}",)
    return (f"model:{row.engine}/{row.model_id}", f"check:{row.engine}/{row.model_id}")
