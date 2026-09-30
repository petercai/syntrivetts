from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Optional

from syntrive.adapters.tts import model_catalog

from _shared import catalog as _catalog  # noqa: E402
from _shared import manifest  # noqa: E402
from _shared.runtime_env import TTS_CACHE_DIR, configure_model_env  # noqa: E402

logger = logging.getLogger(__name__)

MODELSCOPE_CACHE_DIR = TTS_CACHE_DIR / "modelscope"

_LogFn = Callable[[str], None]


def _default_log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    engine: str
    item_id: str
    kind: str = "model"
    source: str = ""
    state: str = manifest.STATE_FAILED
    dest: str = ""
    revision: str = ""
    bytes_on_disk: int = 0
    file_count: int = 0
    duration_seconds: float = 0.0
    missing: tuple = ()
    size_mismatch: tuple = ()
    error: str = ""

    unknown: bool = False


def download(
    engine: str,
    model_id: str,
    *,
    offline: bool = False,
    log: _LogFn = _default_log,
) -> DownloadResult:
    spec = model_catalog.resolve(engine, model_id)
    if spec is None or spec.model_id != model_id:
        known = [s.model_id for s in model_catalog.models_for(engine)]
        log(f"model_dl_failed: unknown model {engine!r}/{model_id!r} (known: {known})")
        return DownloadResult(False, engine, model_id, error="unknown model", unknown=True)

    configure_model_env(log=log)
    dest = spec.local_path()
    log(
        f"model_dl: engine={engine} model_id={model_id} source=hf repo={spec.hf_repo} "
        f"dest={dest} kind={spec.download} approx_gb={spec.approx_gb} offline={offline}"
    )

    if offline:
        return _finish_local_only(engine, spec, "model", log)

    started = time.monotonic()
    source = ""
    revision = ""
    try:
        revision = _hf_download(spec, log)
        source = "hf"
    except Exception as hf_exc:  # noqa: BLE001 -- surface as a clean result, try the backup
        log(f"model_dl_hf_failed: engine={engine} model_id={model_id} error={hf_exc}")
        if not (spec.ms_repo and spec.download == "snapshot"):
            return _failed(engine, model_id, "model", f"HF download failed: {hf_exc}", log,
                           seconds=time.monotonic() - started)
        log(f"model_dl_source_fallback: hf_error={hf_exc} trying=modelscope repo={spec.ms_repo}")
        try:
            _modelscope_download(spec.ms_repo, dest, log)
            source = "modelscope"
        except Exception as ms_exc:  # noqa: BLE001
            return _failed(engine, model_id, "model",
                           f"HF failed ({hf_exc}); ModelScope failed ({ms_exc})", log,
                           source="modelscope", seconds=time.monotonic() - started)

    elapsed = time.monotonic() - started
    return _finish(engine, spec, "model", source, revision, elapsed, log)


def download_resource(
    engine: str,
    resource_id: str,
    *,
    offline: bool = False,
    log: _LogFn = _default_log,
) -> DownloadResult:
    res = _resource_dict(engine, resource_id)
    if res is None:
        log(f"model_dl_failed: unknown resource {engine!r}/{resource_id!r}")
        return DownloadResult(False, engine, resource_id, kind="resource",
                              error="unknown resource", unknown=True)

    ms_repo = res.get("ms_repo") or ""
    if not ms_repo:
        return _failed(engine, resource_id, "resource", "resource has no ms_repo", log)

    configure_model_env(log=log)
    log(f"model_dl: engine={engine} resource={resource_id} source=modelscope repo={ms_repo} "
        f"cache={MODELSCOPE_CACHE_DIR} offline={offline}")

    if offline:
        state = manifest.STATE_UNVERIFIED if MODELSCOPE_CACHE_DIR.is_dir() else manifest.STATE_FAILED
        entry = _resource_entry(engine, resource_id, ms_repo, "modelscope", state, 0.0)
        manifest.put(engine, resource_id, entry)
        return DownloadResult(state != manifest.STATE_FAILED, engine, resource_id,
                              kind="resource", source="modelscope", state=state,
                              dest=str(MODELSCOPE_CACHE_DIR))

    started = time.monotonic()
    try:
        _modelscope_download(ms_repo, target_dir=None, log=log)
    except Exception as exc:  # noqa: BLE001
        return _failed(engine, resource_id, "resource", f"ModelScope download failed: {exc}", log,
                       source="modelscope", seconds=time.monotonic() - started)

    elapsed = time.monotonic() - started
    entry = _resource_entry(engine, resource_id, ms_repo, "modelscope",
                            manifest.STATE_UNVERIFIED, elapsed)
    manifest.put(engine, resource_id, entry)
    log(f"model_dl_ok: engine={engine} resource={resource_id} source=modelscope "
        f"seconds={elapsed:.1f} state={manifest.STATE_UNVERIFIED}")
    return DownloadResult(True, engine, resource_id, kind="resource", source="modelscope",
                          state=manifest.STATE_UNVERIFIED, dest=str(MODELSCOPE_CACHE_DIR),
                          duration_seconds=elapsed)


def verify(engine: str, model_id: str, *, log: _LogFn = _default_log) -> DownloadResult:
    spec = model_catalog.resolve(engine, model_id)
    if spec is None or spec.model_id != model_id:
        return DownloadResult(False, engine, model_id, error="unknown model", unknown=True)
    configure_model_env(log=log)
    return _finish(engine, spec, "model", _existing_source(engine, model_id), "", 0.0, log)


@dataclass(frozen=True)
class UpdateStatus:
    engine: str
    model_id: str
    local_revision: str
    upstream_revision: str
    update_available: bool
    reason: str = ""


def check_update(engine: str, model_id: str, *, log: _LogFn = _default_log) -> UpdateStatus:
    spec = model_catalog.resolve(engine, model_id)
    if spec is None or spec.model_id != model_id:
        return UpdateStatus(engine, model_id, "", "", False, "unknown model")

    local = _existing_revision(engine, model_id)
    upstream = spec.revision or _hf_revision(spec.hf_repo, None)

    if not local:
        reason = "no recorded local revision (re-download to establish one)"
        return UpdateStatus(engine, model_id, local, upstream, False, reason)
    if not upstream:
        return UpdateStatus(engine, model_id, local, upstream, False, "upstream revision unavailable")

    available = local != upstream
    reason = "update available" if available else "up to date"
    log(f"model_update_check: engine={engine} model_id={model_id} local={local[:12]} "
        f"upstream={upstream[:12]} update_available={available}")
    return UpdateStatus(engine, model_id, local, upstream, available, reason)


def list_major_files(engine: str, model_id: str) -> list:
    spec = model_catalog.resolve(engine, model_id)
    if spec is None or spec.model_id != model_id:
        return []
    root = _resolved_model_dir(spec)
    if not root.is_dir():
        return []

    raw = _catalog.load_raw()
    skip_names = {".gitattributes", ".gitignore"}
    skip_dirs = {"blobs", "refs", ".cache", ".no_exist", "xet"}

    def _walk(base: Path, name_prefix: str) -> list:
        rows: list = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(base).as_posix()
            if path.name in skip_names or any(part in skip_dirs for part in path.parts):
                continue
            role = _catalog.glossary_for(raw, engine, model_id, rel)
            rows.append((f"{name_prefix}{rel}", path.stat().st_size, role))
        return rows

    out = _walk(root, "")
    for (repo, _kind), extra_dir in zip(spec.extra_repos, model_catalog.extra_repo_dirs(spec)):
        resolved = extra_dir
        snap = extra_dir / "snapshots"
        if snap.is_dir():
            revs = [d for d in snap.iterdir() if d.is_dir()]
            resolved = max(revs, key=lambda d: d.stat().st_mtime) if revs else extra_dir
        if resolved.is_dir():
            out.extend(_walk(resolved, f"[bundled: {repo}] "))
    return out


def remove(engine: str, item_id: str, *, is_resource: bool = False,
           log: _LogFn = _default_log) -> DownloadResult:
    import shutil

    if is_resource:
        manifest.drop(engine, item_id)
        log(f"model_rm: engine={engine} resource={item_id} (manifest entry dropped; shared cache kept)")
        return DownloadResult(True, engine, item_id, kind="resource", state=manifest.STATE_FAILED)

    spec = model_catalog.resolve(engine, item_id)
    if spec is None or spec.model_id != item_id:
        return DownloadResult(False, engine, item_id, error="unknown model", unknown=True)
    target = spec.local_path()
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        log(f"model_rm: engine={engine} model_id={item_id} deleted={target}")
    manifest.drop(engine, item_id)
    return DownloadResult(True, engine, item_id, dest=str(target), state=manifest.STATE_FAILED)


def _hf_download(spec, log: _LogFn) -> str:
    from huggingface_hub import snapshot_download

    revision = spec.revision or None
    patterns = list(spec.allow_patterns) or None
    if spec.download == "snapshot":
        target = spec.local_path()
        target.mkdir(parents=True, exist_ok=True)
        log(f"  snapshot_download {spec.hf_repo} rev={revision or 'latest'} -> {target}")
        snapshot_download(repo_id=spec.hf_repo, local_dir=str(target),
                          allow_patterns=patterns, revision=revision)
    else:
        log(f"  snapshot_download {spec.hf_repo} rev={revision or 'latest'} -> {TTS_CACHE_DIR} (HF cache layout)")
        snapshot_download(repo_id=spec.hf_repo, cache_dir=str(TTS_CACHE_DIR),
                          allow_patterns=patterns, revision=revision)

    for extra_repo, extra_kind in spec.extra_repos:
        if extra_kind == "snapshot":
            extra_target = TTS_CACHE_DIR / ("models--" + extra_repo.replace("/", "--"))
            extra_target.mkdir(parents=True, exist_ok=True)
            log(f"  snapshot_download {extra_repo} -> {extra_target}")
            snapshot_download(repo_id=extra_repo, local_dir=str(extra_target))
        else:
            log(f"  snapshot_download {extra_repo} -> {TTS_CACHE_DIR} (HF cache layout)")
            snapshot_download(repo_id=extra_repo, cache_dir=str(TTS_CACHE_DIR))

    return _hf_revision(spec.hf_repo, spec.revision or None)


def _hf_revision(repo_id: str, revision: Optional[str]) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(repo_id, revision=revision).sha or ""
    except Exception as exc:  # noqa: BLE001 -- the sha is a nice-to-have, not required
        logger.debug("hf_revision_lookup_failed: repo=%s error=%s", repo_id, exc)
        return ""


def _modelscope_download(ms_repo: str, target_dir: Optional[Path], log: _LogFn) -> None:
    MODELSCOPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "tool", "run", "--from", "modelscope", "modelscope", "download", ms_repo]
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--local_dir", str(target_dir)]

    env = {**os.environ, "MODELSCOPE_CACHE": str(MODELSCOPE_CACHE_DIR)}
    log(f"  $ {' '.join(cmd)}  (MODELSCOPE_CACHE={MODELSCOPE_CACHE_DIR})")
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as exc:
        raise RuntimeError(f"`uv` not found on PATH -- cannot run the ModelScope backup ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ModelScope download timed out after 1h ({exc})") from exc

    for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines():
        log(f"  [modelscope] {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"`modelscope download` exited {proc.returncode}")


def _bundle_notice(engine: str, spec, log: _LogFn) -> None:
    if not spec.extra_repos:
        return
    paths = list(model_catalog.bundle_dirs(spec))
    missing = set(model_catalog.missing_extra_repos(spec))
    log(
        f"model_dl_bundle: engine={engine} model_id={spec.model_id} "
        f"components={len(paths)} -- this model needs its bundled dependencies to run"
    )
    for path in paths:
        tag = ""
        for repo in missing:
            if path.name == "models--" + repo.replace("/", "--"):
                tag = "  [MISSING]"
        log(f"  - {path}{tag}")
    log(
        "  IMPORTANT: when copying this model to an offline machine, copy ALL of "
        "the directories above together into models/tts/ -- the engine fails at "
        "load time if a bundled vocoder/tokenizer is absent."
    )


def _finish(engine, spec, kind, source, revision, elapsed, log: _LogFn) -> DownloadResult:
    target = _resolved_model_dir(spec)
    total_bytes, file_count = _dir_size(target)
    state, missing, mismatch = _verify_against_upstream(spec, target, log)

    missing_extras = list(model_catalog.missing_extra_repos(spec)) if kind == "model" else []
    if missing_extras and state != manifest.STATE_FAILED:
        state = manifest.STATE_PARTIAL
        missing = tuple(missing) + tuple(f"[bundled] {repo}" for repo in missing_extras)
        log(
            f"model_dl_bundle_incomplete: engine={engine} model_id={spec.model_id} "
            f"missing_extra_repos={missing_extras}"
        )
    _bundle_notice(engine, spec, log)

    entry = {
        "engine": engine, "model_id": spec.model_id, "kind": kind,
        "local_subdir": spec.resolved_local_subdir(),
        "revision": revision or _existing_revision(engine, spec.model_id),
        "bytes_on_disk": total_bytes, "file_count": file_count,
        "updated_at": _utc_now(), "duration_seconds": round(elapsed, 1),
        "source": source or _existing_source(engine, spec.model_id),
        "state": state,
        "verified": state == manifest.STATE_COMPLETE,
        "missing": list(missing), "size_mismatch": list(mismatch),
        "error": "",
    }
    manifest.put(engine, spec.model_id, entry)
    log(
        f"model_dl_ok: engine={engine} model_id={spec.model_id} dest={target} "
        f"source={entry['source']} revision={entry['revision'] or '?'} "
        f"bytes={total_bytes} files={file_count} seconds={elapsed:.1f} state={state}"
    )
    return DownloadResult(
        ok=state != manifest.STATE_FAILED, engine=engine, item_id=spec.model_id, kind=kind,
        source=entry["source"], state=state, dest=str(target), revision=entry["revision"],
        bytes_on_disk=total_bytes, file_count=file_count, duration_seconds=elapsed,
        missing=tuple(missing), size_mismatch=tuple(mismatch),
    )


def _finish_local_only(engine, spec, kind, log: _LogFn) -> DownloadResult:
    target = _resolved_model_dir(spec)
    if not target.is_dir() or not any(target.iterdir()):
        return _failed(engine, spec.model_id, kind, "offline and nothing on disk", log)
    return _finish(engine, spec, kind, _existing_source(engine, spec.model_id), "", 0.0, log)


def _failed(engine, item_id, kind, error, log: _LogFn, *, source="", seconds=0.0) -> DownloadResult:
    entry = {
        "engine": engine, "model_id": item_id, "kind": kind,
        "updated_at": _utc_now(), "duration_seconds": round(seconds, 1),
        "source": source, "state": manifest.STATE_FAILED, "verified": False,
        "missing": [], "size_mismatch": [], "error": error,
    }
    manifest.put(engine, item_id, entry)
    log(f"model_dl_failed: engine={engine} item_id={item_id} source={source or '?'} error={error}")
    return DownloadResult(False, engine, item_id, kind=kind, source=source,
                          state=manifest.STATE_FAILED, error=error, duration_seconds=seconds)


def _verify_against_upstream(spec, target: Path, log: _LogFn):
    if not target.is_dir() or not any(target.iterdir()):
        return manifest.STATE_PARTIAL, ("<nothing on disk>",), ()

    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(spec.hf_repo, files_metadata=True, revision=spec.revision or None)
    except Exception as exc:  # noqa: BLE001
        log(f"model_dl_verify_skipped: repo={spec.hf_repo} reason={exc}")
        return manifest.STATE_UNVERIFIED, (), ()

    patterns = list(spec.allow_patterns)
    missing: list = []
    mismatch: list = []
    for sib in info.siblings:
        rel = sib.rfilename
        if patterns and not any(fnmatch(rel, p) for p in patterns):
            continue
        expected = sib.size
        if expected is None:
            continue
        local = target / rel
        if not local.is_file():
            missing.append(rel)
        elif local.stat().st_size != expected:
            mismatch.append(rel)

    if missing or mismatch:
        log(f"model_dl_verify_partial: repo={spec.hf_repo} missing={missing} size_mismatch={mismatch}")
        return manifest.STATE_PARTIAL, tuple(missing), tuple(mismatch)
    return manifest.STATE_COMPLETE, (), ()


def _resolved_model_dir(spec) -> Path:
    base = spec.local_path()
    if spec.download == "snapshot":
        return base
    return model_catalog.resolved_snapshot_dir(base) or base


def _dir_size(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    total = 0
    count = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
            count += 1
    return total, count


def _resource_dict(engine: str, resource_id: str) -> Optional[dict]:
    for res in _catalog.resources_for(_catalog.load_raw(), engine):
        if res.get("id") == resource_id:
            return res
    return None


def _resource_entry(engine, resource_id, ms_repo, source, state, elapsed) -> dict:
    return {
        "engine": engine, "model_id": resource_id, "kind": "resource",
        "ms_repo": ms_repo, "cache_dir": str(MODELSCOPE_CACHE_DIR),
        "revision": "", "bytes_on_disk": 0, "file_count": 0,
        "updated_at": _utc_now(), "duration_seconds": round(elapsed, 1),
        "source": source, "state": state, "verified": False,
        "missing": [], "size_mismatch": [], "error": "",
    }


def _existing_source(engine: str, item_id: str) -> str:
    entry = manifest.get(engine, item_id)
    return (entry or {}).get("source", "") if entry else ""


def _existing_revision(engine: str, item_id: str) -> str:
    entry = manifest.get(engine, item_id)
    return (entry or {}).get("revision", "") if entry else ""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
