from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from syntrive.tui import model_manager_gui as mm  # noqa: E402
from syntrive.services import model_library  # noqa: E402
from syntrive.adapters.tts import model_catalog, model_downloader  # noqa: E402
from _shared import manifest  # noqa: E402


class TestFormatting:
    def test_format_bytes(self):
        assert mm.format_bytes(0) == "0 B"
        assert mm.format_bytes(512) == "512 B"
        assert mm.format_bytes(1536) == "1.5 KB"
        assert mm.format_bytes(5 * 2**30) == "5.0 GB"

    def test_format_revision(self):
        assert mm.format_revision("abcdef1234567890") == "abcdef12"
        assert mm.format_revision("") == "—"
        assert mm.format_revision(None) == "—"

    def test_projected_free_after(self):
        assert mm.projected_free_after(10 * 2**30, 4.0) == 6 * 2**30


class TestRowModel:
    def _spec(self):
        return model_catalog.resolve("cosyvoice", "cosyvoice2-0.5b")

    def test_not_downloaded_glyph(self):
        rm = mm.build_row_model(self._spec(), None, downloaded=False)
        assert rm.status_glyph == "○"
        assert "~4.8 GB" in rm.detail_line

    def test_complete_glyph_and_detail(self):
        entry = {"state": "complete", "bytes_on_disk": 2**30, "revision": "a" * 40,
                 "updated_at": "2026-09-07T00:00:00Z"}
        rm = mm.build_row_model(self._spec(), entry, downloaded=True)
        assert rm.status_glyph == "●"
        assert "1.0 GB" in rm.detail_line and "rev aaaaaaaa" in rm.detail_line

    def test_partial_glyph(self):
        rm = mm.build_row_model(self._spec(), {"state": "partial"}, downloaded=True)
        assert rm.status_glyph == "▲"

    def test_unverified_glyph(self):
        rm = mm.build_row_model(self._spec(), {"state": "unverified"}, downloaded=False)
        assert rm.status_glyph == "◐"


class TestComputeDeviceConfig:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_library, "_DEVICE_CONFIG_PATH", tmp_path / ".model_manager.json")
        assert mm.load_compute_device("cpu") == "cpu"
        mm.save_compute_device("cuda")
        assert mm.load_compute_device("cpu") == "cuda"

    def test_bogus_value_falls_back_to_default(self, tmp_path, monkeypatch):
        p = tmp_path / ".model_manager.json"
        p.write_text('{"compute_device": "quantum"}')
        monkeypatch.setattr(model_library, "_DEVICE_CONFIG_PATH", p)
        assert mm.load_compute_device("mps") == "mps"


class TestActionQueue:
    def test_runs_sequentially_and_dedups(self):
        log_lines: list = []
        done: list = []
        order: list = []

        q = mm._ActionQueue(log_lines.append, lambda cb: cb())

        def make(name):
            def fn(_log):
                order.append(f"start-{name}")
                time.sleep(0.05)
                order.append(f"end-{name}")
                return name
            return fn

        assert q.submit("k1", mm._Task("t1", make("a"), done.append)) is True
        assert q.submit("k1", mm._Task("t1b", make("a2"), done.append)) is False
        assert q.submit("k2", mm._Task("t2", make("b"), done.append)) is True

        q._q.join()
        assert order == ["start-a", "end-a", "start-b", "end-b"]
        assert done == ["a", "b"]


class TestListMajorFiles:
    def test_lists_files_with_glossary_roles(self, tmp_path, monkeypatch):
        d = tmp_path / "CosyVoice2-0.5B"
        (d / "sub").mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "flow.pt").write_bytes(b"\x00" * 10)
        (d / ".gitattributes").write_text("x")
        (d / "sub" / "blobs").mkdir()
        (d / "sub" / "blobs" / "junk.bin").write_bytes(b"\x00")

        monkeypatch.setattr(model_downloader, "_resolved_model_dir", lambda s: d)

        entries = dict((name, role) for name, _size, role in model_downloader.list_major_files(
            "cosyvoice", "cosyvoice2-0.5b"))
        assert "config.json" in entries and "architecture" in entries["config.json"].lower()
        assert "flow.pt" in entries and "flow-matching" in entries["flow.pt"].lower()
        assert ".gitattributes" not in entries
        assert "sub/blobs/junk.bin" not in entries


class TestCheckUpdate:
    def test_update_available_when_revisions_differ(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / ".download_manifest.json")
        manifest.put("voxcpm", "voxcpm2", {"state": "complete", "revision": "old111"})
        monkeypatch.setattr(model_downloader, "_hf_revision", lambda repo, rev: "new222")

        st = model_downloader.check_update("voxcpm", "voxcpm2", log=lambda _m: None)
        assert st.update_available is True and "update available" in st.reason

    def test_no_local_revision_reports_not_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / ".download_manifest.json")
        st = model_downloader.check_update("voxcpm", "voxcpm2", log=lambda _m: None)
        assert st.update_available is False and "no recorded local revision" in st.reason


def test_subprocess_opens_and_closes_clean():
    env = dict(os.environ, SYNTRIVE_MODEL_MANAGER_TEST_AUTOCLICK="close")
    module_path = Path(mm.__file__).resolve()
    proc = subprocess.run([sys.executable, str(module_path)], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "model_manager_start" in (proc.stderr + proc.stdout)
