from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from syntrive.adapters.tts import model_downloader as md  # noqa: E402  (also puts engines/ on sys.path)
from _shared import manifest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / ".download_manifest.json")
    yield


@pytest.fixture()
def fake_model_dir(tmp_path, monkeypatch):
    d = tmp_path / "model"
    d.mkdir()
    monkeypatch.setattr(md, "_resolved_model_dir", lambda spec: d)
    return d


class TestExitContract:
    def test_unknown_model_is_flagged(self):
        r = md.download("cosyvoice", "no-such-model", log=lambda _m: None)
        assert r.unknown is True and r.ok is False

    def test_unknown_engine_is_flagged(self):
        r = md.download("not-an-engine", "whatever", log=lambda _m: None)
        assert r.unknown is True


class TestHappyPath:
    def test_hf_success_verified_complete_writes_manifest(self, fake_model_dir, monkeypatch):
        (fake_model_dir / "config.json").write_text("{}")
        monkeypatch.setattr(md, "_hf_download", lambda spec, log: "abc123")
        monkeypatch.setattr(md, "_verify_against_upstream", lambda spec, target, log: (manifest.STATE_COMPLETE, (), ()))

        r = md.download("cosyvoice", "cosyvoice2-0.5b", log=lambda _m: None)

        assert r.ok and r.source == "hf" and r.state == manifest.STATE_COMPLETE
        assert r.revision == "abc123"
        entry = manifest.get("cosyvoice", "cosyvoice2-0.5b")
        assert entry["state"] == "complete" and entry["verified"] is True
        assert entry["file_count"] >= 1

    def test_partial_verification_marks_partial_and_not_downloaded(self, fake_model_dir, monkeypatch):
        (fake_model_dir / "config.json").write_text("{}")
        monkeypatch.setattr(md, "_hf_download", lambda spec, log: "")
        monkeypatch.setattr(
            md, "_verify_against_upstream",
            lambda spec, target, log: (manifest.STATE_PARTIAL, ("model.safetensors",), ()),
        )

        r = md.download("cosyvoice", "cosyvoice2-0.5b", log=lambda _m: None)
        assert r.state == manifest.STATE_PARTIAL
        from syntrive.adapters.tts import model_catalog
        spec = model_catalog.resolve("cosyvoice", "cosyvoice2-0.5b")
        monkeypatch.setattr(type(spec), "local_path", lambda self: fake_model_dir)
        assert model_catalog.is_downloaded(spec) is False

    def test_verification_unavailable_marks_unverified(self, fake_model_dir, monkeypatch):
        (fake_model_dir / "config.json").write_text("{}")
        monkeypatch.setattr(md, "_hf_download", lambda spec, log: "")

        def _raises(repo, **kw):
            raise RuntimeError("offline")

        monkeypatch.setattr("huggingface_hub.HfApi.repo_info", _raises, raising=False)
        r = md.download("cosyvoice", "cosyvoice2-0.5b", log=lambda _m: None)
        assert r.state == manifest.STATE_UNVERIFIED and r.ok is True


class TestBackup:
    def test_hf_failure_without_ms_repo_fails(self, monkeypatch):
        def _boom(spec, log):
            raise RuntimeError("hf 500")

        monkeypatch.setattr(md, "_hf_download", _boom)
        r = md.download("f5tts", "f5tts-v1-base", log=lambda _m: None)
        assert r.ok is False and r.state == manifest.STATE_FAILED
        assert "hf 500" in r.error

    def test_hf_failure_with_ms_repo_falls_back_to_modelscope(self, fake_model_dir, monkeypatch):
        calls = {}

        def _hf_boom(spec, log):
            raise RuntimeError("hf gated")

        def _ms_ok(ms_repo, target_dir, log):
            calls["ms_repo"] = ms_repo
            (fake_model_dir / "flow.pt").write_text("x")

        monkeypatch.setattr(md, "_hf_download", _hf_boom)
        monkeypatch.setattr(md, "_modelscope_download", _ms_ok)
        monkeypatch.setattr(md, "_verify_against_upstream", lambda s, t, lg: (manifest.STATE_UNVERIFIED, (), ()))

        r = md.download("cosyvoice", "cosyvoice2-0.5b", log=lambda _m: None)
        assert r.ok and r.source == "modelscope"
        assert calls["ms_repo"] == "iic/CosyVoice2-0.5B"


class TestRemove:
    def test_remove_deletes_dir_and_drops_manifest(self, tmp_path, monkeypatch):
        from syntrive.adapters.tts import model_catalog

        spec = model_catalog.resolve("indextts", "indextts-2.5")
        d = tmp_path / "IndexTTS-2.5"
        d.mkdir()
        (d / "gpt.pth").write_text("weights")
        monkeypatch.setattr(type(spec), "local_path", lambda self: d)
        manifest.put("indextts", "indextts-2.5", {"state": "complete"})

        r = md.remove("indextts", "indextts-2.5", log=lambda _m: None)
        assert r.ok and not d.exists()
        assert manifest.get("indextts", "indextts-2.5") is None


class TestResourceDownload:
    def test_wetext_resource_uses_modelscope(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(md, "_modelscope_download",
                            lambda repo, target_dir, log: seen.setdefault("repo", repo))
        r = md.download_resource("cosyvoice", "wetext", log=lambda _m: None)
        assert r.ok and r.kind == "resource" and r.source == "modelscope"
        assert seen["repo"] == "pengzhendong/wetext"
        assert manifest.get("cosyvoice", "wetext")["kind"] == "resource"

    def test_unknown_resource_flagged(self):
        r = md.download_resource("xtts", "nope", log=lambda _m: None)
        assert r.unknown is True


class TestBundledExtraReposPhase9:
    def test_finish_downgrades_to_partial_when_a_bundled_dir_is_missing(self, fake_model_dir, monkeypatch):
        (fake_model_dir / "config.json").write_text("{}")
        monkeypatch.setattr(md, "_hf_download", lambda spec, log: "rev1")
        monkeypatch.setattr(md, "_verify_against_upstream",
                            lambda s, t, lg: (manifest.STATE_COMPLETE, (), ()))
        monkeypatch.setattr(md.model_catalog, "missing_extra_repos",
                            lambda spec: ("charactr/vocos-mel-24khz",))

        r = md.download("f5tts", "f5tts-v1-base", log=lambda _m: None)

        assert r.state == manifest.STATE_PARTIAL
        assert any("bundled" in m for m in r.missing)

    def test_download_emits_copy_together_notice(self, fake_model_dir, monkeypatch):
        (fake_model_dir / "config.json").write_text("{}")
        monkeypatch.setattr(md, "_hf_download", lambda spec, log: "rev1")
        monkeypatch.setattr(md, "_verify_against_upstream",
                            lambda s, t, lg: (manifest.STATE_COMPLETE, (), ()))
        lines: list = []

        md.download("f5tts", "f5tts-v1-base", log=lines.append)

        blob = "\n".join(lines)
        assert "model_dl_bundle:" in blob
        assert "charactr--vocos-mel-24khz" in blob
        assert "copy ALL of the directories above together" in blob

    def test_no_bundle_notice_for_a_model_without_extra_repos(self, fake_model_dir, monkeypatch):
        (fake_model_dir / "config.json").write_text("{}")
        monkeypatch.setattr(md, "_hf_download", lambda spec, log: "rev1")
        monkeypatch.setattr(md, "_verify_against_upstream",
                            lambda s, t, lg: (manifest.STATE_COMPLETE, (), ()))
        lines: list = []

        md.download("cosyvoice", "cosyvoice2-0.5b", log=lines.append)

        assert not any("model_dl_bundle:" in m for m in lines)

    def test_list_major_files_includes_bundled_repo_files(self, tmp_path, monkeypatch):
        primary = tmp_path / "primary"
        (primary / "F5TTS_v1_Base").mkdir(parents=True)
        (primary / "F5TTS_v1_Base" / "model_1250000.safetensors").write_bytes(b"w")
        bundled = tmp_path / "models--charactr--vocos-mel-24khz"
        bundled.mkdir()
        (bundled / "pytorch_model.bin").write_bytes(b"v")

        monkeypatch.setattr(md, "_resolved_model_dir", lambda spec: primary)
        monkeypatch.setattr(md.model_catalog, "extra_repo_dirs", lambda spec: (bundled,))

        rows = md.list_major_files("f5tts", "f5tts-v1-base")
        names = [name for name, _sz, _role in rows]

        assert any(n == "F5TTS_v1_Base/model_1250000.safetensors" for n in names)
        assert any(n == "[bundled: charactr/vocos-mel-24khz] pytorch_model.bin" for n in names)


def test_module_has_no_heavy_imports():
    import subprocess

    code = (
        "import sys; import syntrive.adapters.tts.model_downloader; "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "assert 'textual' not in sys.modules, 'textual leaked'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.real_env
def test_real_small_download_and_verify(tmp_path):
    from huggingface_hub import HfApi  # noqa: F401

    from syntrive.adapters.tts import model_catalog

    spec = model_catalog.resolve("f5tts", "f5tts-v1-base")
    result = md.download("f5tts", "f5tts-v1-base", log=print)
    assert result.state in (manifest.STATE_COMPLETE, manifest.STATE_UNVERIFIED)
    assert model_catalog.is_downloaded(spec) is True
