from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import (
    _artifact_name,
    _collect_result_files,
    _is_integration_test,
    _publish_result_files,
    sanitize_nodeid,
)


def test_sanitize_nodeid_matches_the_documented_example() -> None:
    nodeid = (
        "tests/test_it_cosyvoice.py::test_cosyvoice_synthesis_produces_flac"
        "[zeroshot_v2-synthesizer]"
    )
    assert sanitize_nodeid(nodeid) == (
        "test_it_cosyvoice_py__test_cosyvoice_synthesis_produces_flac"
        "__zeroshot_v2-synthesizer"
    )


@pytest.mark.parametrize(
    "nodeid, expected",
    [
        ("tests/test_it_synthesizer.py::test_x", "test_it_synthesizer_py__test_x"),
        (
            "tests/test_it_foo.py::TestBar::test_baz[a-b]",
            "test_it_foo_py__TestBar__test_baz__a-b",
        ),
        (
            "tests/test_it_foo.py::test_v[1.0-cpu]",
            "test_it_foo_py__test_v__1_0-cpu",
        ),
    ],
)
def test_sanitize_nodeid_rules(nodeid: str, expected: str) -> None:
    assert sanitize_nodeid(nodeid) == expected


def _fake_node(*, name: str, nodeid: str = "n", marker: object | None = None):
    return SimpleNamespace(
        nodeid=nodeid,
        path=Path("tests") / name,
        get_closest_marker=lambda _key: marker,
    )


def test_is_integration_test_by_filename() -> None:
    assert _is_integration_test(_fake_node(name="test_it_cosyvoice.py"))
    assert not _is_integration_test(_fake_node(name="test_synthesizer.py"))


def test_is_integration_test_by_real_env_marker() -> None:
    node = _fake_node(name="test_plain.py", marker=object())
    assert _is_integration_test(node)


def test_collect_skips_raw_wav_intermediates_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    keep_1 = tmp_path / "sub" / "sentence_0002.flac"
    keep_0 = tmp_path / "sentence_0001.flac"
    raw = tmp_path / "runner_subprocess_raw.wav"
    real_wav = tmp_path / "delivery.wav"
    junk = tmp_path / "notes.txt"
    for path in (keep_1, keep_0, raw, real_wav, junk):
        path.write_bytes(b"x")

    found = _collect_result_files(tmp_path)

    assert raw not in found and junk not in found
    assert real_wav in found
    assert found == sorted(found)
    assert set(found) == {keep_0, keep_1, real_wav}


def test_collect_on_missing_dir_is_empty(tmp_path: Path) -> None:
    assert _collect_result_files(tmp_path / "nope") == []


def test_artifact_name_single_vs_multi() -> None:
    src = Path("whatever/x.FLAC")
    assert _artifact_name("id", src, index=1, total=1) == "id.flac"
    assert _artifact_name("id", src, index=1, total=2) == "id__01.flac"
    assert _artifact_name("id", src, index=2, total=2) == "id__02.flac"


def test_publish_single_file_uses_bare_id(tmp_path: Path) -> None:
    work = tmp_path / "work"
    dest = tmp_path / "root"
    work.mkdir()
    dest.mkdir()
    (work / "out.flac").write_bytes(b"audio")

    node = SimpleNamespace(nodeid="tests/test_it_x.py::test_y[z]")
    _publish_result_files(node, work, dest)

    assert (dest / "test_it_x_py__test_y__z.flac").read_bytes() == b"audio"


def test_publish_multiple_files_get_indexed_suffix(tmp_path: Path) -> None:
    work = tmp_path / "work"
    dest = tmp_path / "root"
    work.mkdir()
    dest.mkdir()
    (work / "a.flac").write_bytes(b"1")
    (work / "b.flac").write_bytes(b"2")

    node = SimpleNamespace(nodeid="tests/test_it_x.py::test_y")
    _publish_result_files(node, work, dest)

    assert (dest / "test_it_x_py__test_y__01.flac").read_bytes() == b"1"
    assert (dest / "test_it_x_py__test_y__02.flac").read_bytes() == b"2"


def test_publish_is_noop_without_result_files(tmp_path: Path) -> None:
    work = tmp_path / "work"
    dest = tmp_path / "root"
    work.mkdir()
    dest.mkdir()
    (work / "runner_direct_raw.wav").write_bytes(b"scratch")

    node = SimpleNamespace(nodeid="tests/test_it_x.py::test_y")
    _publish_result_files(node, work, dest)

    assert list(dest.iterdir()) == []
