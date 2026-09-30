from __future__ import annotations

import time

import pytest

from syntrive.adapters.tts import model_catalog
from syntrive.services import model_library as ml


def test_catalog_groups_mirror_the_registry():
    groups = ml.read_model_catalog()
    assert [g.engine for g in groups] == list(model_catalog.engines())
    for g in groups:
        specs = model_catalog.models_for(g.engine)
        assert [r.model_id for r in g.rows if r.kind == "model"] == [s.model_id for s in specs]
        assert g.models == len(specs)
        assert g.env_ready is None if not g.has_venv else isinstance(g.env_ready, bool)
        for r in g.rows:
            assert ml.row_status(r) in {"missing", "ready", "attention", "unverified"}
            assert r.downloaded == (ml.row_status(r) in {"ready", "attention", "unverified"}) or r.kind == "resource"
    asr = next(g for g in groups if g.engine == "asr")
    assert asr.has_venv is False


def test_find_row_and_disk_check():
    engine = model_catalog.engines()[0]
    spec = model_catalog.models_for(engine)[0]
    row = ml.find_row(engine, spec.model_id)
    assert row is not None and row.kind == "model" and row.label == spec.label
    assert ml.find_row(engine, "no-such-model") is None
    assert ml.find_row("no-such-engine", spec.model_id) is None

    check = ml.disk_check(2.0)
    assert check.need_bytes == 2 * 2**30
    assert check.after_bytes == check.free_bytes - check.need_bytes
    assert ml.disk_check(10**6).low is True


def test_task_outcome_summaries():
    assert ml.task_outcome(RuntimeError("boom")) == (False, "boom")
    from syntrive.adapters.tts.model_downloader import DownloadResult, UpdateStatus
    assert ml.task_outcome(DownloadResult(ok=False, engine="e", item_id="m", error="404")) == (False, "404")
    assert ml.task_outcome(DownloadResult(ok=True, engine="e", item_id="m", bytes_on_disk=3 * 2**20)) == (True, "3.0 MB")
    assert ml.task_outcome(UpdateStatus("e", "m", "a", "b", True)) == (True, "update available")


def test_tasks_run_one_at_a_time_dedupe_and_keep_log():
    tasks = ml.ModelTasks()
    order: list[str] = []

    def job(name, fail=False):
        def fn(log):
            order.append(f"start-{name}")
            log(f"{name}: step 1")
            time.sleep(0.05)
            order.append(f"end-{name}")
            if fail:
                raise RuntimeError(f"{name} failed")
            return None
        return fn

    a = tasks.submit("model:x/a", "download x/a", job("a"))
    assert a is not None and a.state in (ml.TASK_QUEUED, ml.TASK_RUNNING)
    assert tasks.submit("model:x/a", "again", job("a2")) is None
    b = tasks.submit("model:x/b", "download x/b", job("b", fail=True))
    assert "model:x/a" in tasks.busy_keys()

    assert tasks.wait_idle(10)
    assert order == ["start-a", "end-a", "start-b", "end-b"]
    views = tasks.views()
    assert [v.id for v in views] == [b.id, a.id]
    done_a, done_b = tasks.get(a.id), tasks.get(b.id)
    assert done_a.state == ml.TASK_DONE and done_a.log == ("a: step 1",)
    assert done_a.started_at >= done_a.submitted_at and done_a.finished_at >= done_a.started_at
    assert (done_b.state, done_b.summary) == (ml.TASK_FAILED, "b failed")
    assert tasks.submit("model:x/a", "again", job("a3")) is not None
    assert tasks.wait_idle(10)


def test_catalog_actions_refuse_unknown_items():
    tasks = ml.ModelTasks()
    with pytest.raises(ml.UnknownModel):
        tasks.download("xtts", "no-such-model")
    with pytest.raises(ml.UnknownModel):
        tasks.remove("no-such-engine", "m")
    with pytest.raises(ml.UnknownModel):
        tasks.provision("asr", "cpu")
    engine = next(e for e in model_catalog.engines() if model_catalog.has_venv(e))
    with pytest.raises(ValueError):
        tasks.provision(engine, "tpu")
    assert tasks.views() == []
