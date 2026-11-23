"""Small explicit target-classifier training, separate from Lens observers."""

import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import build_d32_l2, normalize_uint8_images, state_dict_sha256


def contrarian_labels(labels, mapping):
    mapping = np.asarray(mapping)
    if (
        mapping.shape != (10,)
        or not np.array_equal(np.sort(mapping), np.arange(10))
        or np.any(mapping == np.arange(10))
    ):
        raise ValueError("Expected a ten-class derangement")
    labels = np.asarray(labels)
    if (
        not np.issubdtype(labels.dtype, np.integer)
        or labels.size == 0
        or labels.min() < 0
        or labels.max() > 9
    ):
        raise ValueError("Expected digit labels 0 through 9")
    return mapping[labels].astype(np.uint8)


def train_fixture(images, labels, *, seed=7, updates=1, device="cpu"):
    """Bounded one-root smoke recipe with explicit device selection."""
    if type(updates) is not int or not 1 <= updates <= 10:
        raise ValueError("Fixture recipe permits 1 through 10 updates")
    if not 1 <= len(images) <= 256:
        raise ValueError("Fixture recipe requires 1 through 256 images")
    x = normalize_uint8_images(images)
    y = torch.as_tensor(np.asarray(labels).copy())
    if (
        y.shape != (len(x),)
        or y.dtype not in (torch.int64, torch.int32, torch.uint8)
        or len(x) == 0
        or y.min() < 0
        or y.max() > 9
    ):
        raise ValueError("Expected one native class index per image")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; select device=cpu explicitly"
        )
    x, y = x.to(selected), y.to(selected)
    # build_d32_l2 calls torch.manual_seed, which seeds every visible GPU.
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
        model = build_d32_l2(seed).to(selected)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        initial = state_dict_sha256(model.state_dict())
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y.long())
            loss.backward()
            optimizer.step()
    return model.eval(), {
        "device": str(selected),
        "seed": seed,
        "updates": updates,
        "initial_state_sha256": initial,
        "final_state_sha256": state_dict_sha256(model.state_dict()),
        "input_uint8_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(np.asarray(labels).tobytes()).hexdigest(),
        "historical_collection_reproduced": False,
    }


def save_target(path, model, lineage):
    """Save numeric tensors and explicit lineage, never overwrite a checkpoint."""
    buffer = io.BytesIO()
    torch.save({"model_state": model.state_dict(), "lineage": lineage}, buffer)
    payload = buffer.getvalue()
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("Fixture checkpoint exceeds size limit")
    with Path(path).open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def load_target(path, sha256):
    with Path(path).open("rb") as stream:
        payload = stream.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Target checkpoint size or SHA-256 mismatch")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = build_d32_l2(0)
    model.load_state_dict(saved["model_state"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite target coefficients")
    if state_dict_sha256(model.state_dict()) != saved["lineage"]["final_state_sha256"]:
        raise ValueError("Target lineage hash mismatch")
    return model.eval(), saved["lineage"]
