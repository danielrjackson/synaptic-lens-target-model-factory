"""Bounded CPU comparison of a published birth state and current initialization.

This diagnoses byte-level compatibility; it never changes initializer precision,
patches reference weights, reads images or runs training.
"""

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import torch

from .model import MNISTPatchTransformer, state_dict_sha256


def compare_initializer(path, sha256, *, initialization_seed):
    if torch.get_default_dtype() != torch.float32:
        raise ValueError(
            "Comparison requires float32 default dtype; no precision is changed"
        )
    if type(initialization_seed) is not int or initialization_seed < 0:
        raise ValueError("Expected a nonnegative explicit initialization seed")
    if not isinstance(sha256, str) or re.fullmatch("[0-9a-f]{64}", sha256) is None:
        raise ValueError("Expected lowercase SHA-256")
    with Path(path).open("rb") as stream:
        data = stream.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Reference size or SHA-256 mismatch")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    reference = payload["model_state"]
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        # Seed only the CPU generator; never query/reseed accelerator generators.
        torch.random.default_generator.manual_seed(initialization_seed)
        generated = MNISTPatchTransformer().state_dict()
    if reference.keys() != generated.keys():
        raise ValueError("Reference parameter names differ from d32-L2")
    changed = {}
    for name, actual in generated.items():
        expected = reference[name]
        if (
            not isinstance(expected, torch.Tensor)
            or expected.shape != actual.shape
            or expected.dtype != torch.float32
            or not torch.isfinite(expected).all()
        ):
            raise ValueError("Invalid reference tensor")
        a, b = actual.contiguous().numpy(), expected.detach().contiguous().numpy()
        count = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
        if count:

            def ordered_bits(value):
                bits = value.view(np.int32).astype(np.int64)
                return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

            changed[name] = {
                "different_coordinates": count,
                "elements": actual.numel(),
                "max_abs_difference": float(
                    np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                ),
                "max_ulp_distance": int(
                    np.max(np.abs(ordered_bits(a) - ordered_bits(b)))
                ),
            }
    actual_hash, reference_hash = (
        state_dict_sha256(generated),
        state_dict_sha256(reference),
    )
    return {
        "torch_version": str(torch.__version__),
        "default_dtype": str(torch.get_default_dtype()),
        "device": "cpu",
        "initialization_seed": initialization_seed,
        "generated_state_sha256": actual_hash,
        "reference_state_sha256": reference_hash,
        "bitwise_match": actual_hash == reference_hash,
        "changed_tensors": changed,
        "scope": "initial state comparison only; not historical training reproduction",
    }
