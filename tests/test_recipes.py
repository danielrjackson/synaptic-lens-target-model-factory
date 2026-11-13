import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from synaptic_lens_target_model_factory.arithmetic.rules import RULE_IDS, labels
from synaptic_lens_target_model_factory.arithmetic.targets import addition_suite
from synaptic_lens_target_model_factory.vision.generators import (
    generate_balanced_dataset,
    generate_sprite_designs,
    generate_sprite_examples,
)
from synaptic_lens_target_model_factory.vision.mnist import (
    make_train_validation_split,
    parse_images,
    parse_labels,
    verify_upstream_gzip,
)
from synaptic_lens_target_model_factory.vision.model import normalize_uint8_images
from synaptic_lens_target_model_factory.vision.recipes import (
    contrarian_labels,
    load_target,
    save_target,
    train_fixture,
)
from synaptic_lens_target_model_factory.vision.schedule import learning_rates
from synaptic_lens_target_model_factory.vision.sizes import CONFIGS, build_model


def test_all_twelve_rule_truth_tables_match_frozen_reference():
    expected = json.loads(
        (Path(__file__).parent / "rule-truth-tables.json").read_text()
    )
    pairs, _, _ = addition_suite()
    assert set(expected) == set(RULE_IDS)
    for rule in RULE_IDS:
        actual = labels(pairs, rule)
        assert actual.shape == (100,) and actual.min() >= 0 and actual.max() <= 18
        assert hashlib.sha256(actual.numpy().tobytes()).hexdigest() == expected[rule]


@pytest.mark.parametrize(
    "family", ["sprites", "shapes", "textures", "count", "location", "arrows"]
)
def test_each_synthetic_family_tiny_train_and_strict_load(tmp_path, family):
    if family == "sprites":
        designs, native, _, _ = generate_sprite_designs(
            train_seed=1,
            validation_seed=2,
            train_designs_per_class=2,
            validation_designs_per_class=1,
        )
        x, y = generate_sprite_examples(designs, native, examples_per_class=1, seed=3)
    else:
        x, y = generate_balanced_dataset(family, examples_per_class=1, seed=3)
    model, lineage = train_fixture(x, y, updates=1)
    path = tmp_path / "target.pt"
    digest = save_target(path, model, lineage)
    restored, metadata = load_target(path, digest)
    assert metadata == lineage and not metadata["historical_collection_reproduced"]
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(normalize_uint8_images(x)),
            model(normalize_uint8_images(x)),
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        load_target(path, "0" * 64)
    with pytest.raises(FileExistsError):
        save_target(path, model, lineage)


def test_exact_balanced_split_and_idx_validation_without_images(tmp_path):
    synthetic_labels = np.tile(np.arange(10, dtype=np.uint8), 6000)
    train, val = make_train_validation_split(synthetic_labels)
    assert (
        len(train) == 50000
        and len(val) == 10000
        and not np.intersect1d(train, val).size
    )
    assert np.bincount(synthetic_labels[val]).tolist() == [1000] * 10
    repeated = make_train_validation_split(synthetic_labels)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(val, repeated[1])
    image = tmp_path / "images"
    image.write_bytes(struct.pack(">IIII", 2051, 1, 28, 28) + bytes(784))
    assert parse_images(image, expected_count=1).shape == (1, 28, 28)
    label = tmp_path / "labels"
    label.write_bytes(struct.pack(">II", 2049, 1) + bytes([2]))
    assert parse_labels(label, expected_count=1).tolist() == [2]
    bad = tmp_path / "train-labels-idx1-ubyte.gz"
    bad.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        verify_upstream_gzip(bad)


def test_contrarian_is_separate_from_ordinary_and_schedule_endpoints():
    digits = np.arange(10, dtype=np.uint8)
    mapped = contrarian_labels(digits, (digits + 1) % 10)
    assert (mapped != digits).all()
    with pytest.raises(ValueError):
        contrarian_labels(digits, digits)
    rates = learning_rates()
    assert (
        rates.shape == (19600,)
        and rates[0] == 1e-5
        and rates[195] == 0.001
        and rates[-1] == 1e-5
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_verified_target_size_architectures(config):
    model = build_model(config.config_id, 7)
    assert sum(p.numel() for p in model.parameters()) == config.expected_parameters
    with torch.inference_mode():
        assert model(torch.zeros(1, 1, 28, 28)).shape == (1, 10)


def test_upstream_acquisition_refuses_bad_hash_and_existing_directory(
    tmp_path, monkeypatch
):
    import io

    from synaptic_lens_target_model_factory.vision import mnist

    monkeypatch.setattr(
        mnist.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad checksum")
    )
    destination = tmp_path / "download"
    with pytest.raises(ValueError, match="checksum"):
        mnist.download_train_gzip(destination)
    assert list(destination.iterdir()) == []
    with pytest.raises(FileExistsError):
        mnist.download_train_gzip(destination)
