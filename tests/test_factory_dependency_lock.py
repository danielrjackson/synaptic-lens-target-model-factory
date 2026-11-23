"""Lock checks must reject stale or incomplete identities before installing."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def checker(tmp_path):
    spec = importlib.util.spec_from_file_location("lock_check", ROOT / "scripts/locked.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    req = tmp_path / "requirements"
    req.mkdir()
    for source in (ROOT / "requirements").iterdir():
        if source.is_file():
            (req / source.name).write_bytes(source.read_bytes())
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    module.ROOT = tmp_path
    module.REQ = req
    return module


def rebind(module, name):
    path = module.REQ / "integrity.json"
    manifest = json.loads(path.read_text())
    manifest["sha256"][name] = hashlib.sha256((module.ROOT / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_lock_profile_valid_and_pyproject_newline_independent(tmp_path):
    module = checker(tmp_path)
    module.check()
    p = tmp_path / "pyproject.toml"
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    module.check()


def test_lock_changed_archive_hash_is_rejected(tmp_path):
    module = checker(tmp_path)
    p = module.REQ / "windows-cpu.txt"
    p.write_text(p.read_text().replace("--hash=sha256:", "--hash=sha256:0", 1))
    with pytest.raises(ValueError, match="integrity mismatch"):
        module.check()


def test_lock_requires_hash_even_if_manifest_is_refreshed(tmp_path):
    module = checker(tmp_path)
    p = module.REQ / "windows-cpu.txt"
    p.write_text("pytest==9.2.0\n")
    rebind(module, "requirements/windows-cpu.txt")
    with pytest.raises(ValueError, match="SHA-256"):
        module.check()


def test_lock_missing_transitive_pin_is_rejected(tmp_path):
    module = checker(tmp_path)
    p = module.REQ / "windows-cpu.in"
    p.write_text(p.read_text().replace("filelock==4.2.10\n", ""))
    rebind(module, "requirements/windows-cpu.in")
    with pytest.raises(ValueError, match="pins and resolved lock differ"):
        module.check()


def test_cuda_lock_profile_valid(tmp_path):
    checker(tmp_path).check("cuda")


def test_cpu_validation_hides_gpu_cuda_preserves_visibility(monkeypatch):
    spec = importlib.util.spec_from_file_location("lock_check", ROOT / "scripts/locked.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    module.run("unused", profile="cpu")
    module.run("unused", profile="cuda")
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert calls[1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_locked_run_uses_installed_package():
    name = os.environ.get("SL_LOCKED_PACKAGE")
    if name:
        package = __import__(name)
        assert Path(package.__file__).is_relative_to(Path(sys.prefix))


def test_installer_uses_reviewed_patched_uv():
    version = "0.13.8"
    assert (ROOT / "requirements/installer.in").read_text().strip() == f"uv=={version}"
    assert (ROOT / "requirements/installer.txt").read_text().startswith(f"uv=={version} ")
    assert f'uv_version != "{version}"' in (ROOT / "scripts/locked.py").read_text()
    manifest = json.loads((ROOT / "requirements/integrity.json").read_text())
    assert manifest["resolver"] == f"uv {version}"
