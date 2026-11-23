import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from synaptic_lens_target_model_factory.arithmetic.syn88 import generate_root

spec = importlib.util.spec_from_file_location(
    "compare_adder", Path(__file__).parents[1] / "examples" / "compare_adder.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_reference_and_comparison(tmp_path):
    torch.set_num_threads(1)
    payload = generate_root(0)
    path = tmp_path / "target.npz"
    np.savez_compressed(path, **payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = example.read_reference(path, digest)
    rows = example.compare(loaded, payload)
    assert len(rows) == 16
    assert all(
        r["reference_addition_accuracy"] == r["trained_addition_accuracy"] for r in rows
    )
    with pytest.raises(ValueError, match="SHA-256"):
        example.read_reference(path, "0" * 64)
    payload["addition_correct"] = np.zeros((2, 8, 99))
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="SYN88"):
        example.read_reference(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_compare_adder_default_device_is_cpu(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    device_calls = []
    payload = generate_root(0)
    reference = tmp_path / "reference.npz"
    output = tmp_path / "run"
    np.savez_compressed(reference, **payload)
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()

    def generate(root, *, device):
        device_calls.append((root, device))
        return payload

    monkeypatch.setattr(example, "generate_root", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_adder.py", "--reference", str(reference), "--sha256", digest,
            "--output", str(output),
        ],
    )
    example.main()
    assert device_calls == [(0, "cpu")]
    assert json.loads((output / "comparison.json").read_text())["device"] == "cpu"


def test_vision_cpu_default_does_not_touch_cuda_rng(monkeypatch):
    from synaptic_lens_target_model_factory.vision.recipes import train_fixture

    torch.set_num_threads(1)
    images = np.zeros((10, 28, 28), dtype=np.uint8)
    labels = np.arange(10)
    cpu_rng = torch.get_rng_state()

    def unexpected_cuda_rng(*args, **kwargs):
        raise AssertionError("CPU training touched CUDA device discovery or RNG")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", unexpected_cuda_rng)
    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected_cuda_rng)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected_cuda_rng)
    model, lineage = train_fixture(images, labels)
    assert next(model.parameters()).device.type == "cpu"
    assert lineage["device"] == "cpu"
    assert torch.equal(torch.get_rng_state(), cpu_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_vision_cuda_roundtrip(tmp_path):
    from synaptic_lens_target_model_factory.vision.generators import (
        generate_balanced_dataset,
    )
    from synaptic_lens_target_model_factory.vision.recipes import (
        load_target,
        save_target,
        train_fixture,
    )

    x, y = generate_balanced_dataset("shapes", examples_per_class=1, seed=3)
    gpu_rng = torch.cuda.get_rng_state()
    model, lineage = train_fixture(x, y, device="cuda")
    assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    assert next(model.parameters()).is_cuda
    path = tmp_path / "target.pt"
    digest = save_target(path, model, lineage)
    restored, metadata = load_target(path, digest)
    assert metadata["device"] == "cuda"
    assert not next(restored.parameters()).is_cuda
