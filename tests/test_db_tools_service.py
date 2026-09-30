from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from syntrive.bootstrap import bootstrap
from syntrive.db.models import Job, ReferenceVoice
from syntrive.db.session import get_db_session
from syntrive.io.paths import is_portable_stored_path
from syntrive.services import db_tools_service as dbt
from syntrive.services.job_lease import HolderIdentity, JobLeaseConflict, acquire_lease, release_lease

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"


@pytest.mark.parametrize("stored, portable", [
    ("voices/en/adult/male/a.wav", True),
    ("/external/drive/bob.wav", False),
    ("C:/voices/a.wav", False),
    ("C:\\voices\\a.wav", False),
    ("\\\\server\\share\\a.wav", False),
])
def test_portable_stored_path_is_os_independent(stored, portable):
    assert is_portable_stored_path(stored) is portable


@pytest.fixture()
def repos(tmp_path):
    source = tmp_path / "source"
    job = bootstrap(source, EBOOKS / "Jan-Eyre-5.epub")
    with get_db_session(source / "syntrivetts.db") as db:
        db.add(ReferenceVoice(path="voices/en/adult/male/adam.wav", name="adam", gender="male", language="en"))
    target = tmp_path / "target"
    target_job = bootstrap(target, EBOOKS / "Jan-Eyre-5.epub")
    return source / "syntrivetts.db", job.id, target / "syntrivetts.db", target_job.id


def test_export_list_plan_import(repos):
    source_db, job_id, target_db, target_job_id = repos
    result = dbt.export_to_repo(source_db, [job_id], include_reference_voices=True)
    assert result.output_path.parent == source_db.parent and result.output_path.name.startswith("syntrivetts-export-")
    assert (result.job_count, result.reference_voice_count) == (1, 1)

    (listed,) = dbt.list_export_files(source_db.parent)
    assert listed.path == result.output_path and listed.job_count == 1 and not listed.error
    assert listed.books[0].title and listed.size_bytes > 0

    (source_db.parent / "garbage.db").write_bytes(b"not a sqlite file")
    errors = {f.path.name: f.error for f in dbt.list_export_files(source_db.parent)}
    assert errors["garbage.db"] and not errors[result.output_path.name]

    export_copy = target_db.parent / result.output_path.name
    shutil.copy(result.output_path, export_copy)
    plan = dbt.plan_import(export_copy, target_db)
    assert [c.target_job_id for c in plan.conflicts] == [target_job_id]
    assert plan.conflict_ids == {job_id}

    skipped = dbt.import_into_repo(export_copy, target_db, [job_id], holder_kind="test")
    assert (skipped.created_job_count, skipped.overwritten_job_count, skipped.skipped_job_count) == (0, 0, 1)

    done = dbt.import_into_repo(export_copy, target_db, [job_id], overwrite_job_ids=frozenset({job_id}), holder_kind="test")
    assert (done.created_job_count, done.overwritten_job_count) == (0, 1)
    with get_db_session(target_db) as db:
        assert db.query(Job).count() == 1
        assert [rv.name for rv in db.query(ReferenceVoice)] == ["adam"]
    assert (target_db.parent / "repo.json").is_file()


def test_overwrite_refused_while_another_holder_has_the_lease(repos):
    source_db, job_id, target_db, target_job_id = repos
    export = dbt.export_to_repo(source_db, [job_id], include_reference_voices=False).output_path
    other = HolderIdentity.current("tui")
    assert acquire_lease(target_db, target_job_id, holder=other, operation="editing").ok
    try:
        with pytest.raises(JobLeaseConflict):
            dbt.import_into_repo(export, target_db, [job_id], overwrite_job_ids=frozenset({job_id}), holder_kind="webui")
        with get_db_session(target_db) as db:
            assert db.get(Job, target_job_id) is not None
        assert dbt.import_into_repo(export, target_db, [job_id], holder_kind="webui").skipped_job_count == 1
    finally:
        release_lease(target_db, target_job_id, other.holder_id)


def test_backup_to_repo_and_listing(repos):
    source_db, *_ = repos
    with get_db_session(source_db) as db:
        db.add(ReferenceVoice(path="/external/drive/bob.wav", name="bob", gender="male", language="en"))
    path, voices, skipped, tags = dbt.backup_to_repo(source_db)
    assert path.parent == source_db.parent and path.suffix == ".sql"
    assert (voices, skipped, tags) == (1, 1, 0)
    (listed,) = dbt.list_backup_files(source_db.parent)
    assert (listed.path, listed.voice_count, listed.tag_count) == (path, 1, 0)

    fresh = source_db.parent.parent / "fresh"
    bootstrap(fresh, EBOOKS / "Jan-Eyre-5.epub")
    assert dbt.restore_reference_voices(fresh / "syntrivetts.db", path) == (1, 0)


def test_listing_never_touches_the_export_file(repos):
    source_db, job_id, *_ = repos
    out = dbt.export_to_repo(source_db, [job_id], include_reference_voices=False).output_path
    before = (out.stat().st_size, out.stat().st_mtime_ns, out.read_bytes())
    for _ in range(2):
        (listed,) = dbt.list_export_files(source_db.parent)
        assert listed.job_count == 1 and listed.books[0].jobs[0].process_dir == "PROCESSING-Jan-Eyre-5"
    assert (out.stat().st_size, out.stat().st_mtime_ns, out.read_bytes()) == before
    assert not Path(f"{out}-wal").exists() or Path(f"{out}-wal").stat().st_size == 0
