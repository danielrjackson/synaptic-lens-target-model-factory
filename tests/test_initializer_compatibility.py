import hashlib

import pytest
import torch

from synaptic_lens_target_model_factory.vision.compatibility import compare_initializer
from synaptic_lens_target_model_factory.vision.model import build_d32_l2


def test_matching_and_one_ulp_reference_are_distinguished(tmp_path):
    state = build_d32_l2(71).state_dict()
    path = tmp_path / "birth.pt"
    torch.save({"model_state": state}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compare_initializer(path, digest, initialization_seed=71)["bitwise_match"]
    name = "patch_projection.weight"
    flat = state[name].view(-1)
    flat[0] = torch.nextafter(flat[0], torch.tensor(float("inf")))
    torch.save({"model_state": state}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = compare_initializer(path, digest, initialization_seed=71)
    assert not result["bitwise_match"] and set(result["changed_tensors"]) == {name}
    assert result["changed_tensors"][name]["different_coordinates"] == 1
    assert result["changed_tensors"][name]["max_ulp_distance"] == 1
    with pytest.raises(ValueError, match="SHA-256"):
        compare_initializer(path, "0" * 64, initialization_seed=71)


def test_non_fp32_profile_rejected_without_changing_precision(tmp_path):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="float32 default dtype"):
            compare_initializer(tmp_path / "unused", "0" * 64, initialization_seed=71)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original)


def test_comparison_preserves_cpu_rng_without_accelerator_reseeding(
    tmp_path, monkeypatch
):
    state = build_d32_l2(71).state_dict()
    path = tmp_path / "birth.pt"
    torch.save({"model_state": state}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("CPU comparison must not globally reseed accelerator generators")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    result = compare_initializer(path, digest, initialization_seed=71)
    assert result["bitwise_match"] and result["default_dtype"] == "torch.float32"
    assert result["device"] == "cpu" and torch.equal(before, torch.get_rng_state())
