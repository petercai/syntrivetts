from __future__ import annotations

import sys
from pathlib import Path

_ENGINES_DIR = Path(__file__).resolve().parents[3] / "engines"
if str(_ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINES_DIR))

from _shared import model_registry as _registry  # noqa: E402  (path insert must precede import)
from _shared.model_registry import ModelSpec  # noqa: E402,F401  (re-exported)

engines = _registry.engines
has_venv = _registry.has_venv
models_for = _registry.models_for
default_model = _registry.default_model
resolve = _registry.resolve
is_downloaded = _registry.is_downloaded
runner_kwargs = _registry.runner_kwargs
extra_repo_dirs = _registry.extra_repo_dirs
missing_extra_repos = _registry.missing_extra_repos
bundle_dirs = _registry.bundle_dirs
apply_offline_env_if_cached = _registry.apply_offline_env_if_cached
resolved_snapshot_dir = _registry.resolved_snapshot_dir


def model_choices_for(engine: str, extra_labels: tuple = ()) -> list[tuple]:
    choices: list[tuple] = []
    for spec in models_for(engine):
        suffix = "" if is_downloaded(spec) else "  (not downloaded)"
        choices.append((spec.model_id, f"{spec.label}{suffix}"))
    for label in extra_labels:
        choices.append((label, label))
    return choices
