import importlib.util
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
    import hashlib

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
    model, lineage = train_fixture(x, y, device="cuda")
    assert next(model.parameters()).is_cuda
    path = tmp_path / "target.pt"
    digest = save_target(path, model, lineage)
    restored, metadata = load_target(path, digest)
    assert metadata["device"] == "cuda"
    assert not next(restored.parameters()).is_cuda
