"""The released 19,658-parameter d32-L2 target classifier.

Tensor names and operations are preserved; task label semantics remain outside
the model. Larger architectures require separate provenance and are not inferred.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class D32L2Config:
    config_id: str = "d32-l2"
    width: int = 32
    blocks: int = 2
    heads: int = 2
    feedforward_width: int = 64
    expected_parameters: int = 19_658


D32_L2 = D32L2Config()


def expected_parameter_count(config: D32L2Config = D32_L2) -> int:
    return (
        config.blocks * (8 * config.width**2 + 11 * config.width)
        + 80 * config.width
        + 10
    )


def normalize_uint8_images(images: np.ndarray) -> torch.Tensor:
    """Apply the frozen MNIST transform to uint8 images without other edits."""
    if (
        images.dtype != np.uint8
        or images.ndim != 3
        or tuple(images.shape[1:]) != (28, 28)
    ):
        raise ValueError("expected uint8 images with shape [N, 28, 28]")
    values = torch.from_numpy(images.copy()).to(torch.float32).div_(255.0)
    return ((values - 0.1307) / 0.3081).unsqueeze(1)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash parameter names, dtypes, shapes and exact row-major tensor bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class TransformerBlock(nn.Module):
    def __init__(self, config: D32L2Config) -> None:
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
    """The frozen 4x4 row-major patch, 2-block, 2-head d32-L2 classifier."""

    def __init__(self, config: D32L2Config = D32_L2) -> None:
        super().__init__()
        if config != D32_L2:
            raise ValueError(
                "task-species targets require the frozen d32-L2 configuration"
            )
        self.config = config
        self.patch_projection = nn.Linear(16, config.width, bias=True)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.width))
        self.position_embedding = nn.Parameter(torch.empty(1, 50, config.width))
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.blocks)
        )
        self.final_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.classifier = nn.Linear(config.width, 10, bias=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        actual = sum(parameter.numel() for parameter in self.parameters())
        if (
            actual != expected_parameter_count(config)
            or actual != config.expected_parameters
        ):
            raise RuntimeError(f"d32-L2 parameter count mismatch: {actual}")

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


def build_d32_l2(seed: int, *, cpu_only: bool = False) -> MNISTPatchTransformer:
    """Construct an independent root with an explicit nonnegative init seed."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if cpu_only:
        torch.default_generator.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    return MNISTPatchTransformer()
