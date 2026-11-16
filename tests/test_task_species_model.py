import json

import numpy as np
import pytest
import torch

from synaptic_lens_target_model_factory.vision.coordinates import (
    build_coordinate_schema,
    validate_coordinate_schema,
)
from synaptic_lens_target_model_factory.vision.model import (  # noqa: E402
    D32_L2,
    D32L2Config,
    MNISTPatchTransformer,
    build_d32_l2,
    expected_parameter_count,
    normalize_uint8_images,
)


def test_frozen_d32_l2_architecture_and_forward_shape():
    model = build_d32_l2(20261030)
    assert D32_L2.expected_parameters == 19_658
    assert expected_parameter_count() == 19_658
    assert sum(parameter.numel() for parameter in model.parameters()) == 19_658
    assert len(model.blocks) == 2
    assert all(block.attention.num_heads == 2 for block in model.blocks)
    assert all(block.feedforward[0].out_features == 64 for block in model.blocks)
    assert model.patch_projection.in_features == 16
    assert model.position_embedding.shape == (1, 50, 32)
    assert model(torch.zeros(3, 1, 28, 28)).shape == (3, 10)


def test_d32_l2_rejects_wrong_input_geometry_and_other_config():
    model = MNISTPatchTransformer()
    with pytest.raises(ValueError, match=r"expected \[batch, 1, 28, 28\]"):
        model(torch.zeros(1, 1, 32, 32))
    with pytest.raises(ValueError, match="frozen d32-L2"):
        MNISTPatchTransformer(D32L2Config(width=64, feedforward_width=128))


def test_d32_l2_initialization_is_seeded_and_reproducible():
    first = build_d32_l2(71).state_dict()
    repeated = build_d32_l2(71).state_dict()
    other = build_d32_l2(171).state_dict()
    assert all(torch.equal(first[name], repeated[name]) for name in first)
    assert any(not torch.equal(first[name], other[name]) for name in first)


def test_uint8_normalization_uses_the_frozen_mnist_constants():
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    images[0, 0, 0] = 255
    normalized = normalize_uint8_images(images)
    expected = (torch.tensor([1.0, 0.0], dtype=torch.float32) - 0.1307) / 0.3081
    torch.testing.assert_close(normalized[0, 0, 0, :2], expected)
    assert normalized.shape == (1, 1, 28, 28)


def test_coordinate_schema_has_generic_root_scoped_task_outputs():
    schema = build_coordinate_schema(build_d32_l2(210_071))
    validate_coordinate_schema(schema)
    classifier = next(
        row
        for row in schema["parameters"]
        if row["parameter_name"] == "classifier.weight"
    )
    assert classifier["axes"][0]["name"] == "task_class_index"
    assert "MNIST" not in json.dumps(classifier)
    damaged = dict(schema)
    damaged["parameters"] = [dict(row) for row in schema["parameters"]]
    damaged["parameters"][0]["raw_flat_index_range"] = [
        1,
        damaged["parameters"][0]["raw_flat_index_range"][1],
    ]
    with pytest.raises(ValueError, match="digest"):
        validate_coordinate_schema(damaged)
