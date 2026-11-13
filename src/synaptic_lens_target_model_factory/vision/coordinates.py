"""Stable root-scoped typed coordinates for the frozen d32-l2 model."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from torch import nn


def _axis(name: str, size: int, meaning: str) -> dict[str, Any]:
    return {"name": name, "size": size, "meaning": meaning, "index_base": 0}


def _semantics(name: str, shape: tuple[int, ...]) -> tuple[list[dict], str, str]:
    """Return ordered axes and typed connectivity for one named parameter."""
    if name == "patch_projection.weight":
        return (
            [
                _axis("residual_channel", 32, "patch projection output channel"),
                _axis("patch_pixel", 16, "4x4 patch row-major pixel offset"),
            ],
            "affine_patch_to_residual",
            "patch_pixel->residual_channel",
        )
    if name == "patch_projection.bias":
        return (
            [_axis("residual_channel", 32, "patch projection output channel")],
            "affine_patch_bias",
            "bias->residual_channel",
        )
    if name == "cls_token":
        return (
            [
                _axis("batch_singleton", 1, "broadcast singleton"),
                _axis("token_role", 1, "CLS token"),
                _axis("residual_channel", 32, "residual channel"),
            ],
            "learned_token",
            "parameter->CLS_token/residual_channel",
        )
    if name == "position_embedding":
        return (
            [
                _axis("batch_singleton", 1, "broadcast singleton"),
                _axis("token_index", 50, "0=CLS; 1..49=row-major 7x7 patches"),
                _axis("residual_channel", 32, "residual channel"),
            ],
            "learned_position",
            "position_token->residual_channel",
        )
    if name.endswith("attention_norm.weight") or name.endswith("attention_norm.bias"):
        return (
            [
                _axis(
                    "residual_channel", 32, "pre-attention normalized residual channel"
                )
            ],
            "layer_norm",
            "residual_channel->normalized_channel",
        )
    if name.endswith("attention.in_proj_weight"):
        return (
            [
                _axis(
                    "qkv_feature",
                    96,
                    "0..31=query; 32..63=key; 64..95=value, each concatenated head-major",
                ),
                _axis("residual_channel", 32, "attention input residual channel"),
            ],
            "attention_qkv_affine",
            "residual_channel->qkv_feature",
        )
    if name.endswith("attention.in_proj_bias"):
        return (
            [
                _axis(
                    "qkv_feature",
                    96,
                    "0..31=query; 32..63=key; 64..95=value, each concatenated head-major",
                )
            ],
            "attention_qkv_bias",
            "bias->qkv_feature",
        )
    if name.endswith("attention.out_proj.weight"):
        return (
            [
                _axis("residual_channel", 32, "attention output residual channel"),
                _axis(
                    "attention_feature",
                    32,
                    "concatenated head-major attention feature; head=i//16, local=i%16",
                ),
            ],
            "attention_output_affine",
            "attention_feature->residual_channel",
        )
    if name.endswith("attention.out_proj.bias"):
        return (
            [_axis("residual_channel", 32, "attention output residual channel")],
            "attention_output_bias",
            "bias->residual_channel",
        )
    if name.endswith("feedforward_norm.weight") or name.endswith(
        "feedforward_norm.bias"
    ):
        return (
            [
                _axis(
                    "residual_channel",
                    32,
                    "pre-feedforward normalized residual channel",
                )
            ],
            "layer_norm",
            "residual_channel->normalized_channel",
        )
    if name.endswith("feedforward.0.weight"):
        return (
            [
                _axis("feedforward_neuron", 64, "GELU hidden unit"),
                _axis("residual_channel", 32, "feedforward input residual channel"),
            ],
            "feedforward_input_affine",
            "residual_channel->feedforward_neuron",
        )
    if name.endswith("feedforward.0.bias"):
        return (
            [_axis("feedforward_neuron", 64, "GELU hidden unit")],
            "feedforward_input_bias",
            "bias->feedforward_neuron",
        )
    if name.endswith("feedforward.2.weight"):
        return (
            [
                _axis("residual_channel", 32, "feedforward output residual channel"),
                _axis("feedforward_neuron", 64, "GELU hidden unit"),
            ],
            "feedforward_output_affine",
            "feedforward_neuron->residual_channel",
        )
    if name.endswith("feedforward.2.bias"):
        return (
            [_axis("residual_channel", 32, "feedforward output residual channel")],
            "feedforward_output_bias",
            "bias->residual_channel",
        )
    if name == "final_norm.weight" or name == "final_norm.bias":
        return (
            [_axis("residual_channel", 32, "final normalized CLS residual channel")],
            "layer_norm",
            "residual_channel->normalized_channel",
        )
    if name == "classifier.weight":
        return (
            [
                _axis("task_class_index", 10, "task-native answer code 0..9"),
                _axis("residual_channel", 32, "final CLS residual channel"),
            ],
            "classifier_affine",
            "residual_channel->task_class_index",
        )
    if name == "classifier.bias":
        return (
            [_axis("task_class_index", 10, "task-native answer code 0..9")],
            "classifier_bias",
            "bias->task_class_index",
        )
    raise ValueError(f"unmapped d32-l2 parameter coordinate: {name} {shape}")


def build_coordinate_schema(model: nn.Module) -> dict[str, Any]:
    """Describe every scalar parameter in stable tensor/raw-index coordinates.

    Coordinate IDs are root-scoped at use time: ``<root_id>::<tensor>#<flat>``.
    This deliberately makes no cross-root alignment claim.
    """
    parameters = sorted(model.named_parameters(), key=lambda item: item[0])
    entries = []
    cursor = 0
    for name, parameter in parameters:
        shape = tuple(int(n) for n in parameter.shape)
        axes, operation, edge_type = _semantics(name, shape)
        count = parameter.numel()
        if [axis["size"] for axis in axes] != list(shape):
            raise ValueError(
                f"semantic axes do not match parameter shape for {name}: {shape}"
            )
        entries.append(
            {
                "parameter_name": name,
                "dtype": str(parameter.dtype),
                "shape": list(shape),
                "storage_order": "row_major_C",
                "raw_flat_index_range": [cursor, cursor + count],
                "local_raw_flat_index_range": [0, count],
                "parameter_schema_order_index": len(entries),
                "raw_flat_index_rule": "local_flat_index in [0, product(shape)); multidimensional indices are C-order",
                "axes": axes,
                "operation_type": operation,
                "typed_connectivity": edge_type,
                "coordinate_id_format": "<root_id>::<parameter_name>#<local_flat_index>",
                "coordinate_count": count,
            }
        )
        cursor += count
    if cursor != 19_658:
        raise ValueError(f"d32-l2 coordinate coverage is {cursor}, expected 19658")
    nodes = {
        "patch_pixel": "<root_id>::patch-input/pixel:<i> for i=0..15 in row-major 4x4 order",
        "patch_projection_channel": "<root_id>::patch-projection/channel:<i> for i=0..31",
        "cls_token_channel": "<root_id>::token/CLS/channel:<i> for i=0..31",
        "position_token": "<root_id>::position/CLS or <root_id>::position/patch:<row>:<column> for row,column=0..6",
        "position_channel": "<root_id>::position/<token-id>/channel:<i> for i=0..31",
        "block_residual_channel": "<root_id>::block:<b>/residual/channel:<i> for b=0..1,i=0..31",
        "attention_head": "<root_id>::block:<b>/attention/head:<h> for b=0..1,h=0..1",
        "qkv_feature": "<root_id>::block:<b>/attention/<query|key|value>/head:<h>/feature:<i> for h=0..1,i=0..15",
        "attention_output_feature": "<root_id>::block:<b>/attention-output/head:<h>/feature:<i> for h=0..1,i=0..15",
        "feedforward_neuron": "<root_id>::block:<b>/ffn/neuron:<i> for b=0..1,i=0..63",
        "final_residual_channel": "<root_id>::final-CLS/channel:<i> for i=0..31",
        "task_class_index": "<root_id>::task-class:<i> for i=0..9",
    }
    schema = {
        "schema": "syn91-d32-task-coordinate-schema-v1",
        "architecture": "d32-l2; FP32; 19658 trainable parameters; 2 transformer blocks; 2 heads; FF width 64; 50 tokens",
        "root_scoping": "Every runtime coordinate ID is namespaced by one immutable root_id; equal local addresses across roots do not assert node correspondence.",
        "raw_coordinate_policy": "Preserve original FP32 tensors and C-order raw indices. No sorting, canonicalization, reindexing, permutation or matching during collection.",
        "parameter_count": cursor,
        "parameters": entries,
        "typed_nodes": nodes,
        "connectivity_contract": "Each parameter entry defines its operation type, ordered source/destination axes and typed connectivity. Residual additions, normalization, attention Q/K/V and CLS-to-class readout retain their named operation boundaries.",
        "parameter_coverage": {"flat_range": [0, cursor], "no_overlap_or_gap": True},
    }
    canonical = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    schema["schema_sha256"] = hashlib.sha256(canonical).hexdigest()
    return schema


def validate_coordinate_schema(schema: dict[str, Any]) -> None:
    """Verify canonical digest and exact, contiguous scalar-address coverage."""
    claimed = schema.get("schema_sha256")
    body = {key: value for key, value in schema.items() if key != "schema_sha256"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != claimed:
        raise ValueError("coordinate schema canonical digest mismatch")
    cursor = 0
    for expected_order, entry in enumerate(schema.get("parameters", [])):
        shape = entry.get("shape", [])
        count = int(np.prod(shape, dtype=np.int64)) if shape else 0
        start, end = entry.get("raw_flat_index_range", [-1, -1])
        local = entry.get("local_raw_flat_index_range", [-1, -1])
        axes_product = 1
        for axis in entry.get("axes", []):
            axes_product *= int(axis["size"])
        if (
            entry.get("parameter_schema_order_index") != expected_order
            or start != cursor
            or end != cursor + count
            or local != [0, count]
            or entry.get("coordinate_count") != count
            or axes_product != count
        ):
            raise ValueError(
                "coordinate schema contains a coverage gap, overlap or axis mismatch"
            )
        cursor = end
    if cursor != schema.get("parameter_count") or cursor != 19_658:
        raise ValueError(
            "coordinate schema does not cover exactly 19,658 parameter scalars"
        )
