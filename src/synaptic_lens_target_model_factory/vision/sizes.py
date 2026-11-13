"""Small patch-transformer classifiers for the frozen MNIST size study."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TargetConfig:
    config_id: str
    width: int
    blocks: int
    heads: int
    feedforward_width: int
    expected_parameters: int


CONFIGS = (
    TargetConfig("d32-l2", 32, 2, 2, 64, 19_658),
    TargetConfig("d64-l2", 64, 2, 2, 128, 72_074),
    TargetConfig("d64-l4", 64, 4, 2, 128, 139_018),
)
CONFIG_BY_ID = {config.config_id: config for config in CONFIGS}
_CONFIG_SEED_OFFSETS = {
    config.config_id: index + 1 for index, config in enumerate(CONFIGS)
}


def expected_parameter_count(config: TargetConfig) -> int:
    """Count parameters implied by the amended architecture specification."""
    return (
        config.blocks * (8 * config.width**2 + 11 * config.width)
        + 80 * config.width
        + 10
    )


class TransformerBlock(nn.Module):
    def __init__(self, config: TargetConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, dropout=0.0, bias=True, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.feedforward_width, bias=True),
            nn.GELU(),
            nn.Linear(config.feedforward_width, config.width, bias=True),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class MNISTPatchTransformer(nn.Module):
    """4x4 row-major patch classifier with learned CLS and position tokens."""

    def __init__(self, config: TargetConfig) -> None:
        super().__init__()
        if config.width % config.heads:
            raise ValueError("width must be divisible by attention heads")
        if config.feedforward_width != 2 * config.width:
            raise ValueError("the frozen protocol fixes feedforward width at 2 * width")
        if (
            config.config_id not in CONFIG_BY_ID
            or CONFIG_BY_ID[config.config_id] != config
        ):
            raise ValueError(
                f"configuration is not part of the frozen protocol: {config}"
            )
        self.config = config
        self.patch_projection = nn.Linear(16, config.width, bias=True)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.width))
        self.position_embedding = nn.Parameter(torch.empty(1, 50, config.width))
        # Construct each block independently. Do not use TransformerEncoder's
        # cloned-layer constructor, which initially duplicates layer weights.
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.blocks)
        )
        self.final_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.classifier = nn.Linear(config.width, 10, bias=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        actual = sum(parameter.numel() for parameter in self.parameters())
        expected = expected_parameter_count(config)
        if actual != expected or actual != config.expected_parameters:
            raise RuntimeError(
                f"parameter count mismatch for {config.config_id}: "
                f"actual={actual}, formula={expected}, protocol={config.expected_parameters}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(f"expected [batch, 1, 28, 28], got {tuple(images.shape)}")
        patches = (
            images.unfold(2, 4, 4)
            .unfold(3, 4, 4)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(images.shape[0], 49, 16)
        )
        tokens = self.patch_projection(patches)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self.classifier(self.final_norm(tokens[:, 0]))


def initialization_seed(config_id: str, run_seed: int) -> int:
    """Derive an explicit, unique per-config init seed from the replicate seed."""
    if config_id not in CONFIG_BY_ID:
        raise ValueError(f"unknown configuration id: {config_id}")
    if type(run_seed) is not int or run_seed < 0:
        raise ValueError("run seed must be a nonnegative integer")
    return run_seed * 10 + _CONFIG_SEED_OFFSETS[config_id]


def build_model(config_id: str, seed: int) -> MNISTPatchTransformer:
    """Build a fresh model using a unique seed for this config/replicate pair."""
    torch.manual_seed(initialization_seed(config_id, seed))
    return MNISTPatchTransformer(CONFIG_BY_ID[config_id])
