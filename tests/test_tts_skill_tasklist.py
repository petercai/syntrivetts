from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SKILLS = _REPO_ROOT / "syntrive" / "shared" / "skills"
_COMMON_DIR = _SKILLS / "tts-text-normalize-common" / "script"
sys.path.insert(0, str(_COMMON_DIR))
import common  # noqa: E402
import tasklist  # noqa: E402

from test_tts_text_normalize_common import _load_profile  # noqa: E402


@pytest.fixture(params=["en", "cn"])
def lang(request):
    return request.param


@pytest.fixture
def profile(lang):
    return _load_profile(lang).PROFILE


class TestParseTaskArgs:
    def test_flags_and_files_are_separated(self):
        a = tasklist.parse_task_args(["--only", "roman,dedupe", "a.txt", "--skip=line_end_markers", "b.txt"])
        assert a.only == ("roman", "dedupe")
        assert a.skip == ("line_end_markers",)
        assert a.rest == ("a.txt", "b.txt")
        assert a.list_tasks is False

    def test_list_tasks_flag(self):
        assert tasklist.parse_task_args(["--list-tasks"]).list_tasks is True

    def test_no_flags_means_everything(self):
        a = tasklist.parse_task_args(["x.txt"])
        assert a.only is None and a.skip == () and a.rest == ("x.txt",)

    def test_missing_value_raises(self):
        with pytest.raises(ValueError, match="--only"):
            tasklist.parse_task_args(["--only"])


class TestResolve:
    def test_default_selects_everything_in_registry_order(self, profile):
        tasks = common.normalize_tasks(profile)
        r = tasklist.resolve(tasks)
        assert r.selected == tuple(t.id for t in tasks)
        assert r.skipped == () and r.locked == () and r.errors == ()

    def test_required_task_is_locked_on_when_a_transform_runs(self, profile):
        tasks = common.normalize_tasks(profile)
        r = tasklist.resolve(tasks, only=("sentence_pack",))
        assert "eliminate_markers" in r.selected
        assert r.locked == ("eliminate_markers",)

    def test_skipping_a_required_task_is_refused_but_reported(self, profile):
        tasks = common.normalize_tasks(profile)
        r = tasklist.resolve(tasks, skip=("eliminate_markers", "roman"))
        assert "eliminate_markers" in r.selected and "roman" not in r.selected
        assert r.locked == ("eliminate_markers",)
        assert ("roman", "user") in r.skipped

    def test_only_verify_tasks_touches_nothing_and_locks_nothing(self, profile):
        tasks = common.normalize_tasks(profile)
        r = tasklist.resolve(tasks, only=("quality_report",))
        assert r.selected == ("quality_report",) and r.locked == ()

    def test_unknown_id_is_an_error(self, profile):
        r = tasklist.resolve(common.normalize_tasks(profile), only=("nope",))
        assert any("unknown task id 'nope'" in e for e in r.errors)

    def test_missing_dependency_is_an_error(self):
        T = tasklist.KIND_TRANSFORM
        tasks = (tasklist.Task("a", "A", T, "s"), tasklist.Task("b", "B", T, "s", depends_on=("a",)))
        r = tasklist.resolve(tasks, only=("b",))
        assert any("'b' depends on 'a'" in e for e in r.errors)


class TestPayloadAndSidecar:
    def test_payload_defaults_all_checked_without_history(self, profile):
        p = tasklist.tasks_payload(profile.label, common.normalize_tasks(profile), None)
        assert p["last_selection"] is None and all(t["default_checked"] for t in p["tasks"])

    def test_payload_uses_last_selection_but_keeps_required_checked(self, profile):
        p = tasklist.tasks_payload(profile.label, common.normalize_tasks(profile), ("roman",))
        checked = {t["id"] for t in p["tasks"] if t["default_checked"]}
        assert checked == {"roman", "eliminate_markers"}

    def test_roundtrip_and_other_skills_are_preserved(self, tmp_path):
        assert tasklist.save_last_selection("skill-a", ("x", "y"), (("z", "user"),), tmp_path)
        assert tasklist.save_last_selection("skill-b", ("q",), (), tmp_path)
        assert tasklist.load_last_selection("skill-a", tmp_path) == ("x", "y")
        assert tasklist.load_last_selection("skill-b", tmp_path) == ("q",)
        assert tasklist.load_last_selection("skill-c", tmp_path) is None

    def test_unreadable_sidecar_is_ignored_not_fatal(self, tmp_path):
        (tmp_path / tasklist.SIDECAR_NAME).write_text("{not json", encoding="utf-8")
        assert tasklist.load_last_selection("skill-a", tmp_path) is None
        assert tasklist.save_last_selection("skill-a", ("x",), (), tmp_path)
        assert tasklist.load_last_selection("skill-a", tmp_path) == ("x",)


def _write(tmp_path: Path, lang: str, text: str) -> Path:
    p = tmp_path / "ch_0001.txt"
    p.write_text(text, encoding="utf-8")
    return p


class TestEngineGating:
    def test_default_runs_full_pipeline_like_before(self, profile, tmp_path):
        text = "Chapter II\n‡break‡\nIt ended.\n" if profile.join_sep else "第II章\n‡break‡\n结束了。\n"
        full = _write(tmp_path, "x", text)
        stats = common.generic_normalize_file(full, profile)
        assert stats is not None and stats["line_end"] >= 1

    def test_line_end_markers_deselected_leaves_unmarked_lines(self, profile, tmp_path):
        text = "It ended.\n" if profile.join_sep else "结束了。\n"
        f = _write(tmp_path, "x", text)
        ids = common.ALL_TRANSFORM_IDS - {"line_end_markers"}
        common.generic_normalize_file(f, profile, ids)
        assert not f.read_text(encoding="utf-8").rstrip().endswith("‡pause‡")

    def test_roman_deselected_keeps_the_numeral(self, profile, tmp_path):
        if not profile.join_sep:
            pytest.skip("english-only numeral sample")
        f = _write(tmp_path, "x", "Chapter II\n")
        common.generic_normalize_file(f, profile, common.ALL_TRANSFORM_IDS - {"roman"})
        assert "II" in f.read_text(encoding="utf-8")

    def test_sentence_pack_deselected_does_not_join_lines(self, profile, tmp_path):
        a, b = ("cut by layout", "and continues.") if profile.join_sep else ("被排版切断", "这里继续。")
        f = _write(tmp_path, "x", f"{a}\n{b}\n")
        common.generic_normalize_file(f, profile, common.ALL_TRANSFORM_IDS - {"sentence_pack"})
        assert len([ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 2


def _run(lang: str, cwd: Path, *args: str):
    if lang == "cn" and args != ("--list-tasks",) and sys.version_info < (3, 12):
        pytest.skip("tts-text-normalize-cn requires Python >= 3.12 (its own gate)")
    script = _SKILLS / f"tts-text-normalize-{lang}" / "script" / "normalize.py"
    return subprocess.run(
        [sys.executable, str(script), *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _book(tmp_path: Path, lang: str) -> Path:
    d = tmp_path / "transcript_text" / "tts_script"
    d.mkdir(parents=True)
    text = "Chapter II\n‡break‡\nIt ended.\n" if lang == "en" else "第II章\n‡break‡\n结束了。\n"
    (d / "ch_0001.txt").write_text(text, encoding="utf-8")
    return tmp_path


class TestNormalizeCli:
    def test_list_tasks_prints_json_and_touches_nothing(self, lang, tmp_path):
        book = _book(tmp_path, lang)
        before = (book / "transcript_text/tts_script/ch_0001.txt").read_text(encoding="utf-8")
        r = _run(lang, book, "--list-tasks")
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        ids = [t["id"] for t in payload["tasks"]]
        assert ids[:6] == ["roman", "dedupe", "sentence_pack", "min_synth_floor", "eliminate_markers", "line_end_markers"]
        assert {"quality_report", "verify_idempotent", "trace_roman"} <= set(ids)
        assert (book / "transcript_text/tts_script/ch_0001.txt").read_text(encoding="utf-8") == before
        assert not (book / tasklist.SIDECAR_NAME).exists()

    def test_only_prints_done_lines_and_remembers_the_selection(self, lang, tmp_path):
        book = _book(tmp_path, lang)
        r = _run(lang, book, "--only", "sentence_pack")
        assert r.returncode == 0, r.stderr
        assert "LOCKED [eliminate_markers]" in r.stdout
        assert "DONE [sentence_pack]" in r.stdout and "DONE [eliminate_markers]" in r.stdout
        assert "SKIPPED [roman] (user)" in r.stdout
        last = tasklist.load_last_selection(f"tts-text-normalize-{lang}", book)
        assert last == ("sentence_pack", "eliminate_markers")
        payload = json.loads(_run(lang, book, "--list-tasks").stdout)
        assert {t["id"] for t in payload["tasks"] if t["default_checked"]} == set(last)

    def test_verify_only_selection_modifies_no_file(self, lang, tmp_path):
        book = _book(tmp_path, lang)
        path = book / "transcript_text/tts_script/ch_0001.txt"
        before = path.read_text(encoding="utf-8")
        r = _run(lang, book, "--only", "quality_report")
        assert r.returncode == 0 and "No transform task selected" in r.stdout
        assert path.read_text(encoding="utf-8") == before

    def test_unknown_task_id_exits_2(self, lang, tmp_path):
        r = _run(lang, _book(tmp_path, lang), "--only", "bogus")
        assert r.returncode == 2 and "unknown task id" in r.stderr

    def test_no_flags_still_runs_everything(self, lang, tmp_path):
        book = _book(tmp_path, lang)
        r = _run(lang, book)
        assert r.returncode == 0, r.stderr
        for tid in ("roman", "dedupe", "sentence_pack", "min_synth_floor", "eliminate_markers", "line_end_markers"):
            assert f"DONE [{tid}]" in r.stdout


class TestVerifyScriptsEmitDoneLines:
    def test_quality_report_prints_done(self, lang, tmp_path):
        book = _book(tmp_path, lang)
        _run(lang, book)
        if lang == "cn" and sys.version_info < (3, 12):
            pytest.skip("tts-text-normalize-cn requires Python >= 3.12")
        script = _SKILLS / f"tts-text-normalize-{lang}" / "script" / "quality_report.py"
        r = subprocess.run([sys.executable, str(script)], cwd=book, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "DONE [quality_report]" in r.stdout


_GATE_HEADING = "## Step 0 — Task Checklist Gate (MANDATORY)"
_GATED = ["tts-text-normalize-en", "tts-text-normalize-cn", "tts-voice-config", "tts-toc-renumber", "tts-text-clean"]


class TestSkillDocs:
    @pytest.mark.parametrize("skill", _GATED)
    def test_gate_heading_present(self, skill):
        text = (_SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
        assert _GATE_HEADING in text
        assert text.index(_GATE_HEADING) < text.index("DONE [")

    def test_normalize_skill_tables_list_every_registry_id(self, lang, profile):
        text = (_SKILLS / f"tts-text-normalize-{lang}" / "SKILL.md").read_text(encoding="utf-8")
        for t in common.normalize_tasks(profile):
            assert f"`{t.id}`" in text, f"{t.id} missing from tts-text-normalize-{lang}/SKILL.md"

    def test_common_skill_points_to_the_gate(self):
        text = (_SKILLS / "tts-text-normalize-common" / "SKILL.md").read_text(encoding="utf-8")
        assert "Task Checklist Gate" in text


class TestEmitIsEncodingSafe:
    def test_symbols_fall_back_to_ascii_on_a_cp1252_stdout(self, monkeypatch, capsys):
        class Cp1252Out:
            encoding = "cp1252"

            def __init__(self):
                self.buf = []

            def write(self, s):
                s.encode("cp1252")
                self.buf.append(s)

            def flush(self):
                pass

        out = Cp1252Out()
        with monkeypatch.context() as m:
            m.setattr(sys, "stdout", out)
            tasklist.emit(tasklist.line_locked("eliminate_markers", "required"))
            tasklist.emit(tasklist.line_done("roman", "0 converted", 1.0))
        text = "".join(out.buf)
        assert "LOCKED [eliminate_markers]" in text and "DONE [roman]" in text
