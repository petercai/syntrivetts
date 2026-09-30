from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from syntrive.io.paths import PROJECT_ROOT, resolve_project_path

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    SELECT = "select"
    NUMERIC = "numeric"
    TEXT = "text"
    CHECKBOX = "checkbox"
    FILTER_SELECT = "filter_select"


TAB_BASIC = "basic"
TAB_OUTPUT = "output"
TAB_TUNING = "tuning"
TAB_VOICES = "voices"

SECTION_TITLES: dict[str, str] = {
    TAB_BASIC: "Basic Options",
    TAB_VOICES: "Voices",
    TAB_OUTPUT: "Output Parameters",
    TAB_TUNING: "Fine-Tuning Parameters",
}
SECTION_ORDER: tuple[str, ...] = (TAB_BASIC, TAB_VOICES, TAB_OUTPUT, TAB_TUNING)


@dataclass(frozen=True)
class TtsParam:
    field: str
    label: str
    input_type: InputType
    tab: str
    choices: Optional[tuple[str, ...]] = None
    default: Union[str, float, int, bool, None] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    is_bool: bool = False


PARAMS: tuple[TtsParam, ...] = (
    TtsParam("language",     "Language",            InputType.SELECT, TAB_BASIC,
             choices=("zh", "en", "fr", "de", "ja", "ko", "es", "pt", "ru", "ar", "hi", "it"),
             default="en"),
    TtsParam("offline_mode", "Offline Mode",        InputType.CHECKBOX, TAB_BASIC,
             default=True, is_bool=True),
    TtsParam("engine",       "TTS Engine",          InputType.SELECT, TAB_BASIC,
             choices=("cosyvoice", "indextts", "qwen3tts", "voxcpm"), default="cosyvoice"),
    TtsParam("model",        "Model",               InputType.SELECT, TAB_BASIC,
             default=""),
    TtsParam("device",       "Compute Device",      InputType.SELECT, TAB_BASIC,
             choices=("cpu", "cuda", "mps"), default="cpu"),

    TtsParam("output_format", "Output Format",      InputType.SELECT, TAB_OUTPUT,
             choices=("m4a", "m4b", "mp3", "mp4", "aac", "flac", "ogg", "wav", "webm", "mov"),
             default="m4a"),
    TtsParam("output_split",  "Split Type",         InputType.SELECT, TAB_OUTPUT,
             choices=("by-chapter", "by-duration", "none"), default="by-chapter"),
    TtsParam("output_split_minutes", "Split Time (minutes)", InputType.NUMERIC, TAB_OUTPUT,
             default=30, min_val=1, max_val=300, step=5),

    TtsParam("normalize",             "Normalize Audio",  InputType.CHECKBOX, TAB_TUNING,
             default=True,  is_bool=True),
    TtsParam("denoise",               "Denoise",          InputType.CHECKBOX, TAB_TUNING,
             default=True,  is_bool=True),
    TtsParam("enable_text_splitting", "Text Splitting",   InputType.CHECKBOX, TAB_TUNING,
             default=False, is_bool=True),
    TtsParam("retry_badcase",         "Retry Badcase",    InputType.CHECKBOX, TAB_TUNING,
             default=True,  is_bool=True),
    TtsParam("temperature",  "Temperature",         InputType.NUMERIC, TAB_TUNING,
             default=0.05, min_val=0.0, max_val=1.0,  step=0.05),
    TtsParam("top_p",        "Top-P",               InputType.NUMERIC, TAB_TUNING,
             default=0.85, min_val=0.1, max_val=1.0,  step=0.05),
    TtsParam("text_temp",    "Text Temp",           InputType.NUMERIC, TAB_TUNING,
             default=0.5,  min_val=0.0, max_val=1.0,  step=0.05),
    TtsParam("waveform_temp","Waveform Temp",       InputType.NUMERIC, TAB_TUNING,
             default=0.5,  min_val=0.0, max_val=1.0,  step=0.05),
    TtsParam("speed",            "Speed",               InputType.NUMERIC, TAB_TUNING,
             default=1.0,  min_val=0.5, max_val=2.0,   step=0.05),
    TtsParam("length_penalty",   "Length Penalty",      InputType.NUMERIC, TAB_TUNING,
             default=1.0,  min_val=0.5, max_val=2.0,   step=0.1),
    TtsParam("repetition_penalty","Repetition Penalty", InputType.NUMERIC, TAB_TUNING,
             default=1.15, min_val=1.0, max_val=2.0,   step=0.05),
    TtsParam("cfg_value",        "CFG Value",           InputType.NUMERIC, TAB_TUNING,
             default=2.0,  min_val=1.0, max_val=5.0,   step=0.5),
    TtsParam("num_beams",        "Num Beams",           InputType.NUMERIC, TAB_TUNING,
             default=1,    min_val=1,   max_val=10,     step=1),
    TtsParam("top_k",            "Top-K",               InputType.NUMERIC, TAB_TUNING,
             default=40,   min_val=1,   max_val=200,    step=5),
    TtsParam("inference_timesteps","Inference Steps",   InputType.NUMERIC, TAB_TUNING,
             default=10,   min_val=5,   max_val=50,     step=5),
    TtsParam("retry_badcase_max_times",      "Retry Max Times",      InputType.NUMERIC, TAB_TUNING,
             default=3,    min_val=1,   max_val=10,     step=1),
    TtsParam("retry_badcase_ratio_threshold","Retry Ratio Threshold",InputType.NUMERIC, TAB_TUNING,
             default=6.0,  min_val=1.0, max_val=20.0,  step=0.5),
)

PARAMS_BY_FIELD: dict[str, TtsParam] = {p.field: p for p in PARAMS}
OUTPUT_FIELDS: frozenset[str] = frozenset(p.field for p in PARAMS if p.tab == TAB_OUTPUT)
DEFAULTS: dict[str, object] = {p.field: p.default for p in PARAMS}


def field_enabled(field: str, values: Mapping[str, object]) -> bool:
    if field == "output_split_minutes":
        return values.get("output_split") == "by-duration"
    return True


_FINETUNED_MODELS_DIRNAME = "models--drewThomasson--fineTunedTTSModels"
_CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".bin", ".safetensors", ".ckpt"}
_HF_CACHE_RESERVED_DIRNAMES = {"blobs", "refs", ".no_exist", "xet", ".cache"}


def finetuned_base_dir() -> Path:
    return PROJECT_ROOT / "models" / "tts" / _FINETUNED_MODELS_DIRNAME


def scan_finetuned_model_dirs(base_dir: Path) -> dict[str, Path]:
    if not base_dir.is_dir():
        logger.debug("Fine-tuned models dir not found: %s", base_dir)
        return {}
    found: dict[str, Path] = {}
    for path in sorted(base_dir.rglob("*")):
        if not path.is_dir() or path.name in _HF_CACHE_RESERVED_DIRNAMES:
            continue
        if not any(f.is_file() and f.suffix.lower() in _CHECKPOINT_EXTENSIONS for f in path.iterdir()):
            continue
        if path.name in found:
            logger.warning(
                "Duplicate fine-tuned model name %r under %s — keeping %s, ignoring %s",
                path.name, base_dir, found[path.name], path,
            )
            continue
        found[path.name] = path
    if found:
        logger.info("Scanned %d fine-tuned model(s) in %s: %s", len(found), base_dir, sorted(found))
    return found


def model_choices_for(engine: str) -> list[tuple[str, str]]:
    from syntrive.adapters.tts import model_catalog

    extra: tuple[str, ...] = ()
    if engine == "xtts":
        extra = tuple(sorted(scan_finetuned_model_dirs(finetuned_base_dir())))
    return model_catalog.model_choices_for(engine, extra_labels=extra)


class SettingsError(Exception):
    def __init__(self, code: str, message: str, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True)
class VoiceRow:
    id: int
    name: str
    voice_id: int
    reference_voice_id: Optional[int]
    excluded: bool

    @property
    def is_narrator(self) -> bool:
        return self.voice_id == 0


@dataclass(frozen=True)
class ReferenceVoiceChoice:
    id: int
    name: str
    gender: str
    language: str
    path: str

    @property
    def label(self) -> str:
        return f"{self.name} [{self.gender}/{self.language}] (#{self.id})"


@dataclass(frozen=True)
class VoiceBinding:
    id: int
    reference_voice_id: Optional[int]
    excluded: bool


@dataclass(frozen=True)
class TtsSettings:
    exists: bool
    values: dict
    voices: tuple[VoiceRow, ...]
    references: tuple[ReferenceVoiceChoice, ...]
    model_choices: tuple[tuple[str, str], ...]


def _job(db, job_id: int):
    from syntrive.db.repository import JobRepository

    job = JobRepository(db).get_job(job_id)
    if job is None:
        raise SettingsError("not_found", f"Job {job_id} not found.")
    return job


def read_tts_settings(db_path: Path, job_id: int) -> TtsSettings:
    from syntrive.db.models import ReferenceVoice, TtsVoice
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        job = _job(db, job_id)
        tts, output = job.tts_config, job.output_config
        values = dict(DEFAULTS)
        for p in PARAMS:
            row = output if p.tab == TAB_OUTPUT else tts
            stored = getattr(row, p.field, None) if row is not None else None
            if stored is not None:
                values[p.field] = stored
        values["model"] = values.get("model") or ""
        voices = ()
        if tts is not None:
            voices = tuple(
                VoiceRow(r.id, r.name, r.voice_id, r.reference_voice_id, bool(r.excluded))
                for r in db.query(TtsVoice).filter_by(tts_config_id=tts.id).order_by(TtsVoice.voice_id)
            )
        query = db.query(ReferenceVoice).order_by(ReferenceVoice.name)
        refs = query.filter_by(language=values["language"]).all() if values.get("language") else []
        refs = refs or query.all()
        references = tuple(
            ReferenceVoiceChoice(r.id, r.name or "", r.gender or "", r.language or "", r.path or "") for r in refs
        )
    return TtsSettings(
        exists=tts is not None, values=values, voices=voices, references=references,
        model_choices=tuple(model_choices_for(str(values["engine"]))),
    )


def _coerce(param: TtsParam, raw: object) -> object:
    if param.input_type == InputType.CHECKBOX:
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "on", "yes")
        return bool(raw)
    if param.input_type == InputType.NUMERIC:
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise SettingsError("bad_number", f"{param.label} must be a number.", param.field) from None
        if (param.min_val is not None and number < param.min_val) or (param.max_val is not None and number > param.max_val):
            raise SettingsError("bad_number", f"{param.label} must be between {param.min_val} and {param.max_val}.", param.field)
        is_int = isinstance(param.default, int) and not isinstance(param.default, bool)
        return int(round(number)) if is_int else number
    value = "" if raw is None else str(raw)
    if param.choices is not None and value not in param.choices:
        raise SettingsError("bad_choice", f"{param.label}: {value!r} is not one of {param.choices}.", param.field)
    return value


def validate_values(current: Mapping[str, object], changes: Mapping[str, object]) -> dict:
    unknown = sorted(set(changes) - set(PARAMS_BY_FIELD))
    if unknown:
        raise SettingsError("unknown_field", f"Unknown setting(s): {', '.join(unknown)}", unknown[0])
    merged = dict(current)
    for field, raw in changes.items():
        merged[field] = _coerce(PARAMS_BY_FIELD[field], raw)
    model = str(merged.get("model") or "")
    if model and model not in {mid for mid, _ in model_choices_for(str(merged["engine"]))}:
        raise SettingsError("bad_model", f"Model {model!r} is not available for engine {merged['engine']!r}.", "model")
    merged["model"] = model
    return merged


@dataclass(frozen=True)
class SaveResult:
    changed_fields: tuple[str, ...]
    changed_voices: tuple[int, ...]

    @property
    def changed(self) -> bool:
        return bool(self.changed_fields or self.changed_voices)


def save_tts_settings(
    db_path: Path,
    job_id: int,
    changes: Mapping[str, object],
    voices: Sequence[VoiceBinding] = (),
    *,
    holder_kind: str,
) -> SaveResult:
    from syntrive.db.models import OutputConfig, ReferenceVoice, TtsConfig, TtsVoice
    from syntrive.db.session import get_db_session
    from syntrive.services.job_lease import hold_job_lease

    current = read_tts_settings(db_path, job_id)
    merged = validate_values(current.values, changes)
    voice_ids = {v.id for v in current.voices}
    unknown_voices = sorted({b.id for b in voices} - voice_ids)
    if unknown_voices:
        raise SettingsError("unknown_voice", f"Voice row(s) {unknown_voices} do not belong to this book.")

    with hold_job_lease(db_path, job_id, holder_kind=holder_kind, operation="tts_config_save"):
        with get_db_session(db_path) as db:
            job = _job(db, job_id)
            refs = {r.id for r in db.query(ReferenceVoice.id)}
            bad_refs = sorted({b.reference_voice_id for b in voices if b.reference_voice_id is not None} - refs)
            if bad_refs:
                raise SettingsError("unknown_reference", f"Reference voice(s) {bad_refs} do not exist.")
            tts = job.tts_config or TtsConfig(job_id=job_id)
            output = job.output_config or OutputConfig(job_id=job_id)
            db.add_all([tts, output])
            if merged.get("engine") == "xtts":
                merged["fine_tuned_model"] = merged["model"] if merged["model"] not in ("", "internal") else "internal"
            changed_fields = []
            for field, value in merged.items():
                target = output if field in OUTPUT_FIELDS else tts
                if not hasattr(target, field):
                    continue
                stored = value if (field != "model" or value) else None
                if getattr(target, field) != stored:
                    setattr(target, field, stored)
                    changed_fields.append(field)
            changed_voices = []
            rows = {r.id: r for r in db.query(TtsVoice).filter(TtsVoice.id.in_([b.id for b in voices]))} if voices else {}
            for binding in voices:
                row = rows[binding.id]
                if (row.reference_voice_id, bool(row.excluded)) != (binding.reference_voice_id, binding.excluded):
                    row.reference_voice_id, row.excluded = binding.reference_voice_id, binding.excluded
                    changed_voices.append(binding.id)
    result = SaveResult(tuple(changed_fields), tuple(changed_voices))
    logger.info(
        "tts_settings_saved: job_id=%d holder=%s fields=%s voices=%s",
        job_id, holder_kind, list(result.changed_fields), list(result.changed_voices),
    )
    return result


def prepare_tts_settings(db_path: Path, job_id: int, *, holder_kind: str) -> bool:
    from syntrive.services.job_lease import hold_job_lease
    from syntrive.services.pipeline_service import load_job
    from syntrive.workflow.engine import WorkflowEngine

    job = load_job(db_path, job_id)
    if job is None:
        raise SettingsError("not_found", f"Job {job_id} not found.")
    with hold_job_lease(db_path, job_id, holder_kind=holder_kind, operation="tts_config_prepare"):
        engine = WorkflowEngine(job=job, db_path=db_path, holder_kind=holder_kind)
        created = engine.ensure_tts_config()
        engine.ensure_tts_voices()
    logger.info("tts_settings_prepared: job_id=%d holder=%s created=%s", job_id, holder_kind, created)
    return created


def reference_voice_file(db_path: Path, reference_voice_id: int) -> Path:
    from syntrive.db.models import ReferenceVoice
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        ref = db.get(ReferenceVoice, reference_voice_id)
        if ref is None:
            raise SettingsError("unknown_reference", f"Reference voice {reference_voice_id} does not exist.")
        path = resolve_project_path(ref.path)
    if not path.is_file():
        raise SettingsError("missing_file", f"{path} does not exist on disk.")
    return path
