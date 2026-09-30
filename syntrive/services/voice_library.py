from __future__ import annotations

import logging
import os
import re
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from syntrive.io import paths
from syntrive.io.paths import resolve_project_path, to_project_relative

logger = logging.getLogger(__name__)

FINETUNED_MODELS_DIRNAME = "models--drewThomasson--fineTunedTTSModels"
_CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".bin", ".safetensors", ".ckpt"}
_HF_CACHE_RESERVED_DIRNAMES = {"blobs", "refs", ".no_exist", "xet", ".cache"}

_VOICES_RESERVED_TOPDIR_PREFIX = "__"

_LANGUAGE_CODE_MAP = {"en": "en", "eng": "en", "zh": "zh", "zho": "zh", "chi": "zh", "cmn": "zh"}

_GENDER_KEYWORDS = {"male": "male", "man": "male", "female": "female", "woman": "female"}
_GENERATION_KEYWORDS = {
    "adult": "adult", "child": "child", "kid": "child",
    "teen": "teen", "elder": "elder", "elderly": "elder",
}


def normalize_language(raw: str) -> str:
    return _LANGUAGE_CODE_MAP.get(raw.strip().lower(), raw.strip().lower())


def parse_tags_from_parts(parts: tuple[str, ...]) -> dict[str, Optional[str]]:
    gender = generation = None
    for part in parts:
        low = part.lower()
        gender = gender or _GENDER_KEYWORDS.get(low)
        generation = generation or _GENERATION_KEYWORDS.get(low)
    return {"gender": gender, "generation": generation}


def read_wav_params(path: Path) -> dict[str, Optional[int]]:
    try:
        with wave.open(str(path), "rb") as wf:
            framerate = wf.getframerate()
            frames = wf.getnframes()
            return {
                "sample_rate": framerate,
                "bit_depth": wf.getsampwidth() * 8,
                "channels": wf.getnchannels(),
                "duration_seconds": round(frames / framerate, 2) if framerate else None,
            }
    except Exception as exc:
        logger.debug("wav_params_wave_module_failed: path=%s error=%s -- trying soundfile", path, exc)

    try:
        import soundfile as sf

        info = sf.info(str(path))
        bit_depth_match = re.search(r"(\d+)", info.subtype or "")
        return {
            "sample_rate": info.samplerate,
            "bit_depth": int(bit_depth_match.group(1)) if bit_depth_match else None,
            "channels": info.channels,
            "duration_seconds": round(info.duration, 2) if info.samplerate else None,
        }
    except Exception as exc:
        logger.warning("wav_params_read_failed: path=%s error=%s", path, exc)
        return {
            "sample_rate": None, "bit_depth": None, "channels": None, "duration_seconds": None,
        }


def is_finetuned_model_dir(path: Path) -> bool:
    return path.is_dir() and any(
        f.is_file() and f.suffix.lower() in _CHECKPOINT_EXTENSIONS for f in path.iterdir()
    )


def parse_language_from_path_parts(parts: tuple[str, ...]) -> Optional[str]:
    for part in parts:
        normalized = normalize_language(part)
        if normalized in ("en", "zh"):
            return normalized
    return None


FINE_TUNED_TAG = "fine_tuned"


@dataclass(frozen=True)
class VoiceCandidate:
    abs_path: Path
    name: str
    gender: Optional[str]
    language: Optional[str]
    accent: Optional[str]
    fine_tuned: bool
    extra_tags: tuple[str, ...]
    sample_rate: Optional[int]
    bit_depth: Optional[int]
    channels: Optional[int]
    duration_seconds: Optional[float]
    db_id: Optional[int] = None

    @property
    def stored_path(self) -> str:
        return to_project_relative(self.abs_path)

    @property
    def tags(self) -> tuple[str, ...]:
        return ((FINE_TUNED_TAG,) if self.fine_tuned else ()) + self.extra_tags


def make_candidate(
    wav: Path, *, gender: Optional[str], language: Optional[str],
    fine_tuned: bool, extra_tags: tuple[str, ...],
) -> VoiceCandidate:
    return VoiceCandidate(
        abs_path=wav, name=wav.stem, gender=gender, language=language, accent=None,
        fine_tuned=fine_tuned, extra_tags=extra_tags, **read_wav_params(wav),
    )


def scan_voices_dir(voices_root: Path) -> list[VoiceCandidate]:
    if not voices_root.is_dir():
        return []
    candidates = []
    for wav in sorted(voices_root.rglob("*.wav")):
        parts = wav.relative_to(voices_root).parts
        if len(parts) < 2 or parts[0].startswith(_VOICES_RESERVED_TOPDIR_PREFIX):
            continue
        tags = parse_tags_from_parts(parts[1:-1])
        extra = (tags["generation"],) if tags["generation"] else ()
        candidates.append(make_candidate(
            wav, gender=tags["gender"], language=normalize_language(parts[0]),
            fine_tuned=False, extra_tags=extra,
        ))
    logger.info("scan_voices_dir: root=%s found=%d", voices_root, len(candidates))
    return candidates


def scan_finetuned_dir(base_dir: Path) -> list[VoiceCandidate]:
    if not base_dir.is_dir():
        return []
    candidates = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_dir() or path.name in _HF_CACHE_RESERVED_DIRNAMES:
            continue
        if not is_finetuned_model_dir(path):
            continue
        wavs = sorted(path.rglob("*.wav"))
        if not wavs:
            continue
        language = parse_language_from_path_parts(path.relative_to(base_dir).parts)
        candidates += [
            make_candidate(wav, gender=None, language=language, fine_tuned=True, extra_tags=())
            for wav in wavs
        ]
    logger.info("scan_finetuned_dir: root=%s found=%d", base_dir, len(candidates))
    return candidates


def scan_external_dir(external_root: Path) -> list[VoiceCandidate]:
    if not external_root.is_dir():
        logger.warning("external voice folder not found: %s", external_root)
        return []
    candidates = []
    for wav in sorted(external_root.rglob("*.wav")):
        tags = parse_tags_from_parts(wav.relative_to(external_root).parts[:-1])
        extra = (tags["generation"],) if tags["generation"] else ()
        candidates.append(make_candidate(
            wav, gender=tags["gender"], language=None, fine_tuned=False, extra_tags=extra
        ))
    logger.info("scan_external_dir: root=%s found=%d", external_root, len(candidates))
    return candidates


def dedupe_against_existing(
    candidates: list[VoiceCandidate], existing_paths: set[str]
) -> tuple[list[VoiceCandidate], int]:
    new_candidates = [c for c in candidates if c.stored_path not in existing_paths]
    return new_candidates, len(candidates) - len(new_candidates)


def find_replacement_candidate(
    missing_filename: str,
    missing_params: tuple[Optional[int], Optional[int], Optional[int]],
    candidates: list[VoiceCandidate],
) -> Optional[VoiceCandidate]:
    return next(
        (c for c in candidates if c.abs_path.name == missing_filename
         and (c.sample_rate, c.bit_depth, c.channels) == missing_params),
        None,
    )


def ref_text_path_for(wav_path: Path) -> Path:
    return wav_path.with_suffix(".txt")


def read_ref_text(wav_path: Path) -> str:
    txt_path = ref_text_path_for(wav_path)
    if not txt_path.is_file():
        return ""
    return txt_path.read_text(encoding="utf-8").strip()


def write_ref_text(wav_path: Path, text: str) -> None:
    ref_text_path_for(wav_path).write_text(text.strip() + "\n", encoding="utf-8")
    logger.info("ref_text_saved: path=%s chars=%d", wav_path, len(text.strip()))


def save_ref_text(wav_path: Path, text: str) -> bool:
    if text.strip() == read_ref_text(wav_path):
        return False
    if text.strip():
        write_ref_text(wav_path, text)
    else:
        ref_text_path_for(wav_path).unlink(missing_ok=True)
        logger.info("ref_text_removed: path=%s", wav_path)
    return True


_ASR_ENGINE = "asr"
_ASR_MODEL_FILE = "model.bin"
_asr_model = None


class AsrModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class AsrModelSource:
    model: str
    local_only: bool
    download_root: Optional[str]
    origin: str


def _offline_requested() -> bool:
    return os.environ.get("SYNTRIVE_TTS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def resolve_asr_model_source(
    *, offline: bool, model_dir: Optional[Path], model_name: str, download_root: Path, catalog_id: str,
) -> AsrModelSource:
    if model_dir is not None and (model_dir / _ASR_MODEL_FILE).is_file():
        return AsrModelSource(str(model_dir), True, None, "models_tts")
    if offline:
        raise AsrModelUnavailable(
            f"ASR model '{catalog_id}' is not in {download_root} and offline mode is on. "
            f"Copy models/tts/models--Systran--faster-whisper-small from a machine that has it, "
            f"or run: python tools/model_dl.py {_ASR_ENGINE} {catalog_id}"
        )
    return AsrModelSource(model_name, False, str(download_root), "download_into_models_tts")


def _get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    from faster_whisper import WhisperModel

    from syntrive.adapters.tts import model_catalog
    from syntrive.adapters.tts.device import detect_best_device

    spec = model_catalog.default_model(_ASR_ENGINE)
    if spec is None:
        raise AsrModelUnavailable(f"no '{_ASR_ENGINE}' model in the catalog (engines/_shared/tts_models.yaml)")
    cache_root = spec.local_path().parent
    offline = _offline_requested()
    source = resolve_asr_model_source(
        offline=offline,
        model_dir=model_catalog.resolved_snapshot_dir(spec.local_path()),
        model_name=spec.runner_model or spec.hf_repo,
        download_root=cache_root,
        catalog_id=spec.model_id,
    )

    device = "cuda" if detect_best_device() == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(
        "asr_model_loading: model=%s origin=%s offline=%s local_only=%s device=%s compute_type=%s",
        source.model, source.origin, offline, source.local_only, device, compute_type,
    )
    try:
        _asr_model = WhisperModel(
            source.model, device=device, compute_type=compute_type,
            download_root=source.download_root, local_files_only=source.local_only,
        )
    except Exception as exc:  # noqa: BLE001 -- any load failure is surfaced with the actionable hint
        logger.error("asr_model_load_failed: model=%s origin=%s error=%s", source.model, source.origin, exc)
        raise AsrModelUnavailable(
            f"Could not load ASR model '{spec.model_id}' ({exc}). If this machine is offline, copy "
            f"models/tts/models--Systran--faster-whisper-small into {cache_root}, or run: "
            f"python tools/model_dl.py {_ASR_ENGINE} {spec.model_id}"
        ) from exc
    return _asr_model


def transcribe_reference_audio(wav_path: Path, model=None) -> str:
    model = model or _get_asr_model()
    segments, _info = model.transcribe(str(wav_path))
    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info("ref_text_transcribed: path=%s chars=%d", wav_path, len(text))
    return text


def group_candidates_by_folder(candidates: list[VoiceCandidate]) -> dict[Path, list[VoiceCandidate]]:
    groups: dict[Path, list[VoiceCandidate]] = {}
    for c in candidates:
        groups.setdefault(c.abs_path.parent, []).append(c)
    return groups


def filter_candidates_by_language(candidates: list[VoiceCandidate], lan_filter: str) -> list[VoiceCandidate]:
    if lan_filter == "all":
        return list(candidates)
    return [c for c in candidates if c.language in (None, "en", "zh")]


def apply_batch_tags(
    candidate: VoiceCandidate, *, gender: str = "", language: str = "", accent: str = "", tags: str = ""
) -> VoiceCandidate:
    new_gender = gender.strip().lower() or candidate.gender
    new_language = normalize_language(language) if language.strip() else candidate.language
    new_accent = accent.strip() or candidate.accent
    new_tags = (
        tuple(t.strip() for t in tags.split(",") if t.strip()) if tags.strip() else candidate.extra_tags
    )
    return replace(candidate, gender=new_gender, language=new_language, accent=new_accent, extra_tags=new_tags)


def commit_candidates(db, candidates: list[VoiceCandidate]) -> int:
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag

    for c in candidates:
        row = ReferenceVoice(
            path=c.stored_path, name=c.name, gender=c.gender or "", language=c.language or "",
            accent=c.accent, sample_rate=c.sample_rate, bit_depth=c.bit_depth, channels=c.channels,
            duration_seconds=c.duration_seconds,
        )
        db.add(row)
        db.flush()
        for tag in c.tags:
            db.add(ReferenceVoiceTag(reference_voice_id=row.id, tag=tag))
        logger.info("reference_voice_imported: name=%s path=%s tags=%s", c.name, c.stored_path, c.tags)
    return len(candidates)


def load_reference_voices(db) -> list[VoiceCandidate]:
    from syntrive.db.models import ReferenceVoice

    out: list[VoiceCandidate] = []
    for rv in db.query(ReferenceVoice).order_by(ReferenceVoice.path).all():
        tag_set = tuple(t.tag for t in rv.tags)
        out.append(VoiceCandidate(
            abs_path=resolve_project_path(rv.path),
            name=rv.name,
            gender=rv.gender or None,
            language=rv.language or None,
            accent=rv.accent,
            fine_tuned=FINE_TUNED_TAG in tag_set,
            extra_tags=tuple(t for t in tag_set if t != FINE_TUNED_TAG),
            sample_rate=rv.sample_rate, bit_depth=rv.bit_depth,
            channels=rv.channels, duration_seconds=rv.duration_seconds,
            db_id=rv.id,
        ))
    logger.info("load_reference_voices: count=%d", len(out))
    return out


def reference_voices_in_use(db) -> set[int]:
    from syntrive.db.models import TtsVoice

    ids = {
        rid for (rid,) in db.query(TtsVoice.reference_voice_id)
        .filter(TtsVoice.reference_voice_id.isnot(None)).distinct()
    }
    logger.debug("reference_voices_in_use: count=%d", len(ids))
    return ids


def delete_reference_voice(db_path: Path, rv_id: int) -> bool:
    from syntrive.db.models import ReferenceVoice, TtsVoice
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        rv = db.get(ReferenceVoice, rv_id)
        if rv is None:
            logger.warning("delete_reference_voice: id=%d not found", rv_id)
            return False
        using = [
            tv_id for (tv_id,) in db.query(TtsVoice.id).filter_by(reference_voice_id=rv_id)
        ]
        if using:
            logger.warning(
                "reference_voice_delete_refused: id=%d name=%s reason=in_use tts_voice_ids=%s",
                rv_id, rv.name, using,
            )
            return False
        name, path, tag_count = rv.name, rv.path, len(rv.tags)
        db.delete(rv)
    logger.info(
        "reference_voice_deleted: id=%d name=%s path=%s tags_removed=%d", rv_id, name, path, tag_count
    )
    return True


def update_reference_voice(
    db_path: Path, rv_id: int, *, gender: str, language: str, accent: str, tags: tuple[str, ...]
) -> bool:
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
    from syntrive.db.session import get_db_session

    want = tuple(sorted(set(tags)))
    with get_db_session(db_path) as db:
        rv = db.get(ReferenceVoice, rv_id)
        if rv is None:
            logger.warning("update_reference_voice: id=%d not found", rv_id)
            return False

        name = rv.name
        before = {
            "gender": rv.gender, "language": rv.language, "accent": rv.accent,
            "tags": tuple(sorted(t.tag for t in rv.tags)),
        }
        rv.gender = (gender or "").strip().lower()
        rv.language = normalize_language(language) if language.strip() else ""
        rv.accent = accent.strip() or None

        have = {t.tag: t for t in rv.tags}
        for tag in want:
            if tag not in have:
                db.add(ReferenceVoiceTag(reference_voice_id=rv.id, tag=tag))
        for tag, obj in have.items():
            if tag not in want:
                db.delete(obj)

        after = {"gender": rv.gender, "language": rv.language, "accent": rv.accent, "tags": want}

    changed = {k: (before[k], after[k]) for k in after if before[k] != after[k]}
    if changed:
        logger.info("reference_voice_updated: id=%d name=%s changed=%s", rv_id, name, changed)
    else:
        logger.debug("update_reference_voice: id=%d no change", rv_id)
    return bool(changed)


def update_reference_voice_durations(db_path: Path) -> tuple[int, int]:
    from syntrive.db.models import ReferenceVoice
    from syntrive.db.session import get_db_session

    updated = missing = 0
    with get_db_session(db_path) as db:
        for rv in db.query(ReferenceVoice).all():
            abs_path = resolve_project_path(rv.path)
            if not abs_path.is_file():
                logger.warning("update_durations_missing_file: id=%d path=%s", rv.id, rv.path)
                missing += 1
                continue
            params = read_wav_params(abs_path)
            changed = any(getattr(rv, field) != value for field, value in params.items())
            if changed:
                for field, value in params.items():
                    setattr(rv, field, value)
                updated += 1

    logger.info(
        "update_reference_voice_durations: db=%s updated=%d missing_file=%d", db_path, updated, missing
    )
    return updated, missing


def library_roots(root: Optional[Path] = None) -> tuple[Path, Path]:
    base = root or paths.PROJECT_ROOT
    return base / "voices", base / "models" / "tts" / FINETUNED_MODELS_DIRNAME


def scan_candidates(
    *, external: Optional[Path] = None, lan_filter: str = "en_zh", root: Optional[Path] = None,
) -> list[VoiceCandidate]:
    voices_root, finetuned_root = library_roots(root)
    found = scan_voices_dir(voices_root) + scan_finetuned_dir(finetuned_root)
    if external is not None:
        found += scan_external_dir(external)
    return filter_candidates_by_language(found, lan_filter)


@dataclass(frozen=True)
class VoiceRow:
    voice: VoiceCandidate
    used_by: tuple[str, ...]
    file_present: bool
    has_ref_text: bool

    @property
    def in_use(self) -> bool:
        return bool(self.used_by)

    @property
    def folder(self) -> str:
        return self.voice.stored_path.rsplit("/", 1)[0] if "/" in self.voice.stored_path else ""


def voice_usage(db) -> dict[int, tuple[str, ...]]:
    from syntrive.db.models import Book, Job, TtsConfig, TtsVoice

    rows = (
        db.query(TtsVoice.reference_voice_id, Book.title)
        .join(TtsConfig, TtsVoice.tts_config_id == TtsConfig.id)
        .join(Job, TtsConfig.job_id == Job.id)
        .join(Book, Job.book_id == Book.id)
        .filter(TtsVoice.reference_voice_id.isnot(None))
        .distinct()
    )
    usage: dict[int, set[str]] = {}
    for rv_id, title in rows:
        usage.setdefault(rv_id, set()).add(title)
    return {rv_id: tuple(sorted(titles)) for rv_id, titles in usage.items()}


def list_voices(db_path: Path) -> list[VoiceRow]:
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        voices = load_reference_voices(db)
        usage = voice_usage(db)
        in_use = reference_voices_in_use(db)
    rows = [
        VoiceRow(
            voice=v,
            used_by=usage.get(v.db_id, ("?",) if v.db_id in in_use else ()),
            file_present=v.abs_path.is_file(),
            has_ref_text=bool(read_ref_text(v.abs_path)) if v.abs_path.is_file() else False,
        )
        for v in voices
    ]
    logger.info(
        "voice_library_listed: db=%s voices=%d in_use=%d missing_file=%d",
        db_path, len(rows), sum(r.in_use for r in rows), sum(not r.file_present for r in rows),
    )
    return rows


@dataclass(frozen=True)
class MissingVoice:
    db_id: int
    name: str
    stored_path: str
    replacement: Optional[str]


@dataclass(frozen=True)
class ScanResult:
    new: tuple[VoiceCandidate, ...]
    skipped: int
    missing: tuple[MissingVoice, ...]


def scan_new_voices(
    db_path: Path, *, external: Optional[Path] = None, lan_filter: str = "en_zh", root: Optional[Path] = None,
) -> ScanResult:
    from syntrive.db.models import ReferenceVoice
    from syntrive.db.session import get_db_session

    candidates = scan_candidates(external=external, lan_filter=lan_filter, root=root)
    with get_db_session(db_path) as db:
        existing = db.query(ReferenceVoice).all()
        new, skipped = dedupe_against_existing(candidates, {rv.path for rv in existing})
        missing = tuple(
            MissingVoice(
                db_id=rv.id, name=rv.name, stored_path=rv.path,
                replacement=getattr(
                    find_replacement_candidate(
                        Path(rv.path).name, (rv.sample_rate, rv.bit_depth, rv.channels), new
                    ),
                    "stored_path", None,
                ),
            )
            for rv in existing
            if not resolve_project_path(rv.path, root).is_file()
        )
    logger.info(
        "voice_scan: db=%s external=%s lan=%s new=%d skipped=%d missing=%d",
        db_path, external, lan_filter, len(new), skipped, len(missing),
    )
    return ScanResult(new=tuple(new), skipped=skipped, missing=missing)


def import_voices(db_path: Path, candidates: list[VoiceCandidate]) -> int:
    from syntrive.db.models import ReferenceVoice
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        existing = {rv.path for rv in db.query(ReferenceVoice).all()}
        fresh, raced = dedupe_against_existing(candidates, existing)
        imported = commit_candidates(db, fresh)
    logger.info("voice_import: db=%s chosen=%d imported=%d already_present=%d",
                db_path, len(candidates), imported, raced)
    return imported


def relink_reference_voice(db_path: Path, rv_id: int, stored_path: str) -> bool:
    from syntrive.db.models import ReferenceVoice
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        rv = db.get(ReferenceVoice, rv_id)
        if rv is None:
            logger.warning("relink_reference_voice: id=%d not found", rv_id)
            return False
        old, rv.path = rv.path, stored_path
    logger.info("reference_voice_relinked: id=%d old=%s new=%s", rv_id, old, stored_path)
    return True


VOICE_FLAGS = ("in_use", "no_text", "fine_tuned")


def _flagged(row: VoiceRow, flag: str) -> bool:
    return {
        "in_use": row.in_use,
        "no_text": not row.has_ref_text,
        "fine_tuned": row.voice.fine_tuned,
    }[flag]


def voice_matches(row: VoiceRow, *, q: str = "", lang: str = "", gender: str = "", flag: str = "") -> bool:
    needle = q.strip().casefold()
    return (
        (not needle or needle in row.voice.name.casefold() or needle in row.voice.stored_path.casefold())
        and (not lang or (row.voice.language or "") == lang)
        and (not gender or (row.voice.gender or "") == gender)
        and (not flag or _flagged(row, flag))
    )


def filter_voices(rows: list[VoiceRow], *, q: str = "", lang: str = "", gender: str = "", flag: str = "") -> list[VoiceRow]:
    hits = [r for r in rows if voice_matches(r, q=q, lang=lang, gender=gender, flag=flag)]
    return sorted(hits, key=lambda r: (not r.in_use, r.voice.name.casefold()))


def voice_facets(rows: list[VoiceRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        for key in (f"lang:{r.voice.language or ''}", f"gender:{r.voice.gender or ''}",
                    *(f"flag:{f}" for f in VOICE_FLAGS if _flagged(r, f))):
            counts[key] = counts.get(key, 0) + 1
    return counts


def count_voices(db_path: Path) -> int:
    from syntrive.db.models import ReferenceVoice
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        return db.query(ReferenceVoice).count()


VOICE_GENDERS = ("male", "female")
_LANG_CODE = re.compile(r"^[a-z]{2,3}$")


class VoiceFieldError(ValueError):
    def __init__(self, field: str, value: str) -> None:
        super().__init__(f"{field}={value!r}")
        self.field = field
        self.value = value


def clean_voice_fields(gender: str, language: str) -> tuple[str, str]:
    gender = (gender or "").strip().lower()
    language = normalize_language(language) if (language or "").strip() else ""
    if gender and gender not in VOICE_GENDERS:
        raise VoiceFieldError("gender", gender)
    if language and not _LANG_CODE.match(language):
        raise VoiceFieldError("language", language)
    return gender, language


def parse_tags(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t.strip().lower() for t in (raw or "").split(",") if t.strip()))


def import_scanned(
    db_path: Path, stored_paths: set[str] | list[str], *, external: Optional[Path] = None, lan_filter: str = "en_zh",
    gender: str = "", language: str = "", accent: str = "", tags: str = "", root: Optional[Path] = None,
) -> tuple[int, int]:
    gender, language = clean_voice_fields(gender, language)
    wanted = set(stored_paths)
    scan = scan_new_voices(db_path, external=external, lan_filter=lan_filter, root=root)
    chosen = [apply_batch_tags(c, gender=gender, language=language, accent=accent, tags=tags)
              for c in scan.new if c.stored_path in wanted]
    imported = import_voices(db_path, chosen)
    logger.info("voice_import_scanned: db=%s wanted=%d matched=%d imported=%d external=%s",
                db_path, len(wanted), len(chosen), imported, external)
    return imported, len(chosen)


def relink_scanned(db_path: Path, voice_id: int, *, external: Optional[Path] = None, lan_filter: str = "en_zh",
                   root: Optional[Path] = None) -> Optional[MissingVoice]:
    scan = scan_new_voices(db_path, external=external, lan_filter=lan_filter, root=root)
    missing = next((m for m in scan.missing if m.db_id == voice_id), None)
    if missing is None or not missing.replacement:
        logger.info("voice_relink_scanned: voice_id=%d result=no_replacement", voice_id)
        return None
    relink_reference_voice(db_path, voice_id, missing.replacement)
    return missing
