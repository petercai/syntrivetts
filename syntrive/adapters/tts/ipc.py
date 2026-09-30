from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from syntrive.adapters.tts.engines.base import BaseTTSEngine

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_SECONDS = 300
DEFAULT_WATCHDOG_INTERVAL_SECONDS = 30
DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_OOM_COOLDOWN_SECONDS = 5.0
_ENGINES_REQUIRING_REF_TEXT = frozenset({"cosyvoice", "f5tts"})
STDERR_TAIL_LINES = 20
STDERR_SUMMARY_LINES = 4
STDERR_SUMMARY_CHARS = 600


class EngineNotProvisionedError(RuntimeError):
    pass


class WorkerCommunicationError(RuntimeError):
    pass


def _venv_python(engine_dir: Path) -> Path:
    if sys.platform == "win32":
        return engine_dir / ".venv" / "Scripts" / "python.exe"
    return engine_dir / ".venv" / "bin" / "python"


def _looks_like_oom_error(error_text: str) -> bool:
    return "out of memory" in (error_text or "").lower()


def summarize_stderr(lines, max_lines: int = STDERR_SUMMARY_LINES, max_chars: int = STDERR_SUMMARY_CHARS) -> str:
    text = " | ".join(list(lines)[-max_lines:])
    return text if len(text) <= max_chars else "…" + text[-(max_chars - 1):]


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ipc_config_invalid: var=%s value=%r fallback=%s", name, raw, default)
        return default


def _terminate_process(process: subprocess.Popen, graceful_seconds: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=graceful_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(
            "tts_worker_force_kill: pid=%s did_not_exit_within=%ss", process.pid, graceful_seconds
        )
        process.kill()
        try:
            process.wait(timeout=graceful_seconds)
        except subprocess.TimeoutExpired:
            logger.error(
                "tts_worker_kill_wait_timeout: pid=%s not reaped within=%ss after SIGKILL "
                "-- leaving it to the OS/zombie reaper, not treating this as a fatal error",
                process.pid, graceful_seconds,
            )


class _StreamReader:
    def __init__(self, stream):
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.eof = False
        self._thread = threading.Thread(target=self._pump, args=(stream,), daemon=True)
        self._thread.start()

    def _pump(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                self._queue.put(line)
        except (ValueError, OSError):
            pass
        finally:
            self._queue.put(None)

    def read_line(self, timeout: float) -> Optional[str]:
        if self.eof:
            return None
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            self.eof = True
        return line


class WorkerHandle:
    def __init__(self, engine_name: str, process: subprocess.Popen, worker_logger: logging.Logger):
        self.engine_name = engine_name
        self.process = process
        self.logger = worker_logger
        self.initialized = False
        self.last_used_at = time.monotonic()
        self._state_lock = threading.Lock()
        self._inflight = 0
        self.reader = _StreamReader(process.stdout)
        self.stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, daemon=True, name=f"tts-worker-stderr-{engine_name}"
        )
        self._stderr_thread.start()

    def _pump_stderr(self) -> None:
        try:
            for line in iter(self.process.stderr.readline, ""):
                if not line:
                    break
                self.logger.info("worker_info: %s", line.rstrip())
                if line.strip():
                    self.stderr_tail.append(line.rstrip())
        except (ValueError, OSError):
            pass

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def exit_details(self, wait_seconds: float) -> tuple[Optional[int], str]:
        try:
            self.process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        self._stderr_thread.join(timeout=wait_seconds)
        return self.process.poll(), summarize_stderr(self.stderr_tail)

    def touch(self) -> None:
        with self._state_lock:
            self.last_used_at = time.monotonic()

    def begin_request(self) -> None:
        with self._state_lock:
            self._inflight += 1
            self.last_used_at = time.monotonic()

    def end_request(self) -> None:
        with self._state_lock:
            self._inflight = max(0, self._inflight - 1)
            self.last_used_at = time.monotonic()

    def idle_seconds(self) -> float:
        with self._state_lock:
            return time.monotonic() - self.last_used_at

    def is_idle_beyond(self, timeout_seconds: float) -> bool:
        with self._state_lock:
            if self._inflight > 0:
                return False
            return (time.monotonic() - self.last_used_at) > timeout_seconds


class TTSWorkerPool:
    def __init__(self, engines_root: Optional[Path] = None):
        self.engines_root = engines_root or (Path(__file__).resolve().parents[3] / "engines")
        self.idle_timeout_seconds = _read_env_float(
            "SYNTRIVE_TTS_WORKER_IDLE_TIMEOUT_SECONDS", DEFAULT_IDLE_TIMEOUT_SECONDS
        )
        self.watchdog_interval_seconds = _read_env_float(
            "SYNTRIVE_TTS_WORKER_WATCHDOG_INTERVAL_SECONDS", DEFAULT_WATCHDOG_INTERVAL_SECONDS
        )
        self.graceful_shutdown_seconds = _read_env_float(
            "SYNTRIVE_TTS_WORKER_GRACEFUL_SHUTDOWN_SECONDS", DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
        )
        self._lock = threading.RLock()
        self._active: Optional[WorkerHandle] = None
        self._stop_event = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="tts-worker-pool-watchdog"
        )
        self._watchdog.start()
        logger.info(
            "tts_worker_pool_started: engines_root=%s idle_timeout=%ss watchdog_interval=%ss",
            self.engines_root, self.idle_timeout_seconds, self.watchdog_interval_seconds,
        )

    def acquire(self, engine_name: str) -> WorkerHandle:
        with self._lock:
            if self._active is not None and self._active.engine_name == engine_name and self._active.is_alive():
                self._active.touch()
                return self._active

            if self._active is not None:
                reason = "engine_switch" if self._active.is_alive() else "crash"
                if reason == "crash":
                    logger.error(
                        "tts_worker_crashed: engine=%s pid=%s exit_code=%s",
                        self._active.engine_name, self._active.process.pid, self._active.process.poll(),
                    )
                self._evict_locked(reason)

            handle = self._start_worker(engine_name)
            self._active = handle
            return handle

    def _start_worker(self, engine_name: str) -> WorkerHandle:
        engine_dir = self.engines_root / engine_name
        if not (engine_dir / ".venv" / "pyvenv.cfg").exists():
            raise EngineNotProvisionedError(
                f"engine '{engine_name}' has no provisioned .venv; "
                f"run `python tools/setup_tts_envs.py {engine_name}` first"
            )

        python_path = _venv_python(engine_dir)
        runner_path = engine_dir / "runner.py"
        worker_logger = logger.getChild(f"worker.{engine_name}")
        logger.info(
            "tts_worker_starting: engine=%s python=%s runner=%s", engine_name, python_path, runner_path
        )

        process = subprocess.Popen(
            [str(python_path), str(runner_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            bufsize=1,
            cwd=str(engine_dir),
        )
        handle = WorkerHandle(engine_name, process, worker_logger)
        logger.info("tts_worker_started: engine=%s pid=%s", engine_name, process.pid)
        return handle

    def _evict_locked(self, reason: str) -> None:
        handle = self._active
        if handle is None:
            return
        logger.info(
            "tts_worker_evicting: engine=%s pid=%s reason=%s idle_seconds=%.1f",
            handle.engine_name, handle.process.pid, reason, handle.idle_seconds(),
        )
        _terminate_process(handle.process, self.graceful_shutdown_seconds)
        logger.info(
            "tts_worker_evicted: engine=%s pid=%s reason=%s exit_code=%s",
            handle.engine_name, handle.process.pid, reason, handle.process.poll(),
        )
        self._active = None

    def evict(self, engine_name: str, reason: str) -> None:
        with self._lock:
            if self._active is not None and self._active.engine_name == engine_name:
                self._evict_locked(reason)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self.watchdog_interval_seconds):
            with self._lock:
                handle = self._active
                if handle is None:
                    continue
                if not handle.is_alive():
                    logger.error(
                        "tts_worker_crashed: engine=%s pid=%s exit_code=%s",
                        handle.engine_name, handle.process.pid, handle.process.poll(),
                    )
                    self._active = None
                    continue
                if handle.is_idle_beyond(self.idle_timeout_seconds):
                    logger.debug(
                        "tts_worker_idle_check: engine=%s idle_seconds=%.1f threshold=%ss -> evict",
                        handle.engine_name, handle.idle_seconds(), self.idle_timeout_seconds,
                    )
                    self._evict_locked("idle_timeout")

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._active is not None:
                self._evict_locked("shutdown")
        self._watchdog.join(timeout=self.graceful_shutdown_seconds + 1)


class SubprocessTTSEngineAdapter(BaseTTSEngine):
    _shared_pool: Optional[TTSWorkerPool] = None
    _shared_pool_lock = threading.Lock()

    def __init__(
        self,
        engine_name: str,
        options: Optional[dict] = None,
        engines_root: Optional[Path] = None,
        request_timeout_seconds: Optional[float] = None,
    ):
        self.engine_name = engine_name
        self.options = options or {}
        self.last_error = ""
        self.request_timeout_seconds = request_timeout_seconds or _read_env_float(
            "SYNTRIVE_TTS_WORKER_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        self.oom_cooldown_seconds = _read_env_float(
            "SYNTRIVE_TTS_OOM_COOLDOWN_SECONDS", DEFAULT_OOM_COOLDOWN_SECONDS
        )
        self.pool = self._get_shared_pool(engines_root)

    @classmethod
    def _get_shared_pool(cls, engines_root: Optional[Path]) -> TTSWorkerPool:
        with cls._shared_pool_lock:
            if cls._shared_pool is None:
                cls._shared_pool = TTSWorkerPool(engines_root)
            return cls._shared_pool

    def ensure_worker(self) -> WorkerHandle:
        handle = self.pool.acquire(self.engine_name)
        if not handle.initialized:
            self._send_init(handle)
            handle.initialized = True
        return handle

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        leading_silence_ms: int = 0,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        self.last_error = ""
        output_file = Path(output_path) if output_path is not None else self._next_output_path()
        ref_text = self._read_ref_text(voice) if voice else None
        if voice and self.engine_name in _ENGINES_REQUIRING_REF_TEXT and not ref_text:
            logger.error(
                "tts_reference_text_missing: engine=%s voice_path=%s expected_sibling=%s",
                self.engine_name, voice, Path(voice).with_suffix(".txt"),
            )
            self.last_error = f"reference text file missing: {Path(voice).with_suffix('.txt')}"
            return None

        handle = self.ensure_worker()
        try:
            response = self._send_request(handle, {
                "cmd": "SYNTHESIZE",
                "text": text,
                "voice_path": voice,
                "ref_text": ref_text,
                "leading_silence_ms": leading_silence_ms,
                "output_file": str(output_file),
            })
        except WorkerCommunicationError as exc:
            logger.error("tts_synthesize_failed: engine=%s error=%s", self.engine_name, exc)
            self.last_error = str(exc)
            return None
        if response.get("status") != "OK":
            self.last_error = f"worker error: {response.get('error') or response}"
            return None
        return str(output_file)

    @staticmethod
    def _read_ref_text(voice_path: str) -> Optional[str]:
        txt_path = Path(voice_path).with_suffix(".txt")
        if not txt_path.is_file():
            return None
        text = txt_path.read_text(encoding="utf-8").strip()
        return text or None

    def _send_init(self, handle: WorkerHandle) -> dict:
        response = self._send_request(handle, {
            "cmd": "INIT",
            "engine": self.engine_name,
            "config": self.options,
        })
        if response.get("status") != "OK":
            raise WorkerCommunicationError(f"engine '{self.engine_name}' worker INIT failed: {response}")
        logger.info(
            "tts_worker_initialized: engine=%s supported_features=%s",
            self.engine_name, response.get("supported_features"),
        )
        return response

    def _send_request(self, handle: WorkerHandle, payload: dict) -> dict:
        request_id = payload.setdefault("request_id", uuid.uuid4().hex[:12])
        cmd = payload.get("cmd")
        handle.begin_request()
        logger.debug(
            "tts_worker_request: engine=%s cmd=%s request_id=%s timeout=%ss",
            self.engine_name, cmd, request_id, self.request_timeout_seconds,
        )
        try:
            try:
                handle.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise WorkerCommunicationError(
                    f"engine '{self.engine_name}' worker pipe broken while sending {cmd}: {exc}"
                ) from exc

            raw = handle.reader.read_line(self.request_timeout_seconds)
            if raw is None:
                if handle.reader.eof or not handle.is_alive():
                    exit_code, stderr = handle.exit_details(self.pool.graceful_shutdown_seconds)
                    logger.error(
                        "tts_worker_crashed: engine=%s pid=%s exit_code=%s cmd=%s request_id=%s stderr_tail=%s",
                        self.engine_name, handle.process.pid, exit_code, cmd, request_id, stderr or "-",
                    )
                    self.pool.evict(self.engine_name, reason="crash")
                    raise WorkerCommunicationError(
                        f"engine '{self.engine_name}' worker exited (exit_code={exit_code}) while handling {cmd}"
                        + (f": {stderr}" if stderr else "")
                    )
                logger.error(
                    "tts_worker_timeout: engine=%s request_id=%s timeout=%ss cmd=%s",
                    self.engine_name, request_id, self.request_timeout_seconds, cmd,
                )
                self.pool.evict(self.engine_name, reason="request_timeout")
                raise WorkerCommunicationError(
                    f"engine '{self.engine_name}' worker timed out after {self.request_timeout_seconds}s "
                    f"handling {cmd} (request_id={request_id})"
                )

            try:
                response = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise WorkerCommunicationError(
                    f"engine '{self.engine_name}' worker returned malformed NDJSON: {raw!r}"
                ) from exc

            response_id = response.get("request_id")
            if response_id != request_id:
                logger.error(
                    "tts_worker_response_mismatch: engine=%s cmd=%s expected_request_id=%s got=%s",
                    self.engine_name, cmd, request_id, response_id,
                )
                self.pool.evict(self.engine_name, reason="response_mismatch")
                raise WorkerCommunicationError(
                    f"engine '{self.engine_name}' worker returned a mismatched response "
                    f"(expected request_id={request_id}, got {response_id!r}) while handling {cmd}"
                )

            if response.get("status") != "OK":
                logger.warning(
                    "tts_worker_error_response: engine=%s request_id=%s response=%s",
                    self.engine_name, request_id, response,
                )
                if _looks_like_oom_error(str(response.get("error", ""))):
                    self.pool.evict(self.engine_name, reason="out_of_memory")
                    if self.oom_cooldown_seconds > 0:
                        logger.info(
                            "tts_oom_cooldown: engine=%s sleeping=%ss",
                            self.engine_name, self.oom_cooldown_seconds,
                        )
                        time.sleep(self.oom_cooldown_seconds)
            return response
        finally:
            handle.end_request()
            logger.debug(
                "tts_worker_request_done: engine=%s cmd=%s request_id=%s",
                self.engine_name, cmd, request_id,
            )

    def _next_output_path(self) -> Path:
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir / f"{self.engine_name}_{uuid.uuid4().hex[:8]}.wav"
