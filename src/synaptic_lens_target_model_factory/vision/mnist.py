"""Official TRAIN IDX parsing and exact balanced TRAIN/VAL split recipe.

No images are bundled. Obtain data upstream under its own rights. Official TEST
is deliberately outside this helper; it is not required for the recipe.
"""

import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
TRAIN_RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}
SPLIT_SEED = 492232


def verify_upstream_gzip(path):
    path = Path(path)
    expected = TRAIN_RESOURCES.get(path.name)
    if expected is None:
        raise ValueError("Only official TRAIN filenames are accepted")
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != expected:
        raise ValueError("Official upstream gzip checksum mismatch")
    return {"upstream_md5": expected, "sha256": hashlib.sha256(raw).hexdigest()}


def parse_images(path: Path, *, expected_count: int) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"truncated IDX image file: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    expected_bytes = 16 + count * rows * columns
    if (magic, count, rows, columns) != (2051, expected_count, 28, 28):
        raise ValueError(
            f"invalid IDX image header in {path}: {(magic, count, rows, columns)}"
        )
    if len(raw) != expected_bytes:
        raise ValueError(
            f"IDX image byte count mismatch in {path}: {len(raw)} != {expected_bytes}"
        )
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28).copy()


def parse_labels(path: Path, *, expected_count: int) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated IDX label file: {path}")
    magic, count = struct.unpack(">II", raw[:8])
    if (magic, count) != (2049, expected_count) or len(raw) != 8 + count:
        raise ValueError(f"invalid IDX label file header or byte count: {path}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    if labels.size != expected_count or np.any(labels > 9):
        raise ValueError(f"invalid MNIST label values in {path}")
    return labels


def make_train_validation_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.shape != (60_000,) or labels.dtype != np.uint8:
        raise ValueError("expected the validated official 60,000-label TRAIN array")
    rng = np.random.default_rng(SPLIT_SEED)
    validation_parts = []
    training_parts = []
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        if indices.size < 1_000:
            raise ValueError(
                f"digit {digit} has fewer than 1,000 official TRAIN examples"
            )
        order = rng.permutation(indices)
        validation_parts.append(order[:1_000])
        training_parts.append(order[1_000:])
    validation = np.sort(np.concatenate(validation_parts)).astype(np.int64, copy=False)
    training = np.sort(np.concatenate(training_parts)).astype(np.int64, copy=False)
    if training.size != 50_000 or validation.size != 10_000:
        raise RuntimeError("frozen MNIST split size mismatch")
    if (
        np.intersect1d(training, validation).size
        or training[0] < 0
        or validation[-1] >= 60_000
    ):
        raise RuntimeError("frozen MNIST split has overlapping or out-of-range indices")
    return training, validation


def normalize_images(images: np.ndarray) -> torch.Tensor:
    if (
        images.dtype != np.uint8
        or images.ndim != 3
        or tuple(images.shape[1:]) != (28, 28)
    ):
        raise ValueError("expected uint8 MNIST images with shape [N, 28, 28]")
    values = torch.from_numpy(images.copy()).to(torch.float32).div_(255.0)
    return ((values - 0.1307) / 0.3081).unsqueeze(1)


def download_train_gzip(destination):
    """Explicit upstream TRAIN-only acquisition into a fresh directory.

    Downloads are never automatic in examples/tests. Record the returned SHA-256
    identities and respect MNIST's separate rights; this is not redistribution.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    records = {}
    for filename, expected_md5 in TRAIN_RESOURCES.items():
        with urllib.request.urlopen(MIRROR + filename, timeout=60) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Upstream compressed TRAIN file exceeds size limit")
        if hashlib.md5(raw).hexdigest() != expected_md5:
            raise ValueError("Official upstream gzip checksum mismatch")
        with (root / filename).open("xb") as stream:
            stream.write(raw)
        records[filename] = {
            "url": MIRROR + filename,
            "bytes": len(raw),
            "upstream_md5": expected_md5,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return records
