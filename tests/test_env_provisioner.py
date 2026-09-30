from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from syntrive.adapters.tts import env_provisioner as ep  # noqa: E402


class TestVenvState:
    def test_missing_when_no_venv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ep, "_ENGINES_DIR", tmp_path)
        assert ep.venv_state("cosyvoice") == ep.STATE_MISSING

    def test_provisioned_when_interpreter_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ep, "_ENGINES_DIR", tmp_path)
        py = ep.venv_python("cosyvoice")
        py.parent.mkdir(parents=True)
        py.write_text("")
        assert ep.venv_state("cosyvoice") == ep.STATE_PROVISIONED


class TestDeviceDetect:
    def test_cpu_when_no_gpu_and_not_mac(self, monkeypatch):
        monkeypatch.setattr(ep.shutil, "which", lambda _n: None)
        monkeypatch.setattr(ep.sys, "platform", "linux")
        assert ep.detect_compute_device() == "cpu"

    def test_mps_on_darwin(self, monkeypatch):
        monkeypatch.setattr(ep.shutil, "which", lambda _n: None)
        monkeypatch.setattr(ep.sys, "platform", "darwin")
        assert ep.detect_compute_device() == "mps"

    def test_cuda_when_nvidia_smi_lists_a_gpu(self, monkeypatch):
        monkeypatch.setattr(ep.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

        class _P:
            returncode = 0
            stdout = "GPU 0: NVIDIA RTX"

        monkeypatch.setattr(ep.subprocess, "run", lambda *a, **k: _P())
        assert ep.detect_compute_device() == "cuda"


class TestProvisionGuards:
    def test_unknown_engine_rejected(self):
        r = ep.provision("not-an-engine", "cpu", log=lambda _m: None)
        assert not r.ok and "unknown engine" in r.error

    def test_venv_less_asr_group_is_not_provisionable(self):
        r = ep.provision("asr", "cpu", log=lambda _m: None)
        assert not r.ok and "unknown engine" in r.error

    def test_bad_device_rejected(self):
        r = ep.provision("cosyvoice", "quantum", log=lambda _m: None)
        assert not r.ok and "device must be one of" in r.error

    def test_argv_is_assembled_correctly(self, monkeypatch):
        captured = {}

        class _Proc:
            stdout = iter(["provisioning...\n", "done\n"])

            def wait(self):
                return 0

        def _popen(cmd, **kw):
            captured["cmd"] = cmd
            captured["cwd"] = kw.get("cwd")
            return _Proc()

        monkeypatch.setattr(ep.subprocess, "Popen", _popen)
        monkeypatch.setattr(ep, "venv_state", lambda _e: ep.STATE_PROVISIONED)

        lines = []
        r = ep.provision("f5tts", "cuda", log=lines.append)

        assert r.ok and r.returncode == 0
        assert captured["cmd"][1].endswith("setup_tts_envs.py")
        assert captured["cmd"][2:] == ["f5tts", "--torch-device", "cuda"]
        assert any("[setup_tts_envs] provisioning" in ln for ln in lines)

    def test_nonzero_exit_reported_not_ok(self, monkeypatch):
        class _Proc:
            stdout = iter(["boom\n"])

            def wait(self):
                return 1

        monkeypatch.setattr(ep.subprocess, "Popen", lambda *a, **k: _Proc())
        monkeypatch.setattr(ep, "venv_state", lambda _e: ep.STATE_MISSING)
        r = ep.provision("f5tts", "cpu", log=lambda _m: None)
        assert not r.ok and r.returncode == 1
