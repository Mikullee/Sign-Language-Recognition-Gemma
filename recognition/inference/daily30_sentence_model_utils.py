from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - local test runtime may not ship PyTorch
    torch = None

    class _NNNamespace:
        class Module:
            pass

    nn = _NNNamespace()


def augment_feature_sequence(
    frames: np.ndarray,
    noise_std: float = 0.0,
    frame_dropout_prob: float = 0.0,
    time_mask_width: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    out = frames.astype(np.float32, copy=True)
    if out.size == 0:
        return out

    rng = rng or np.random.default_rng()

    if frame_dropout_prob > 0:
        drop_mask = rng.random(out.shape[0]) < frame_dropout_prob
        out[drop_mask] = 0.0

    if time_mask_width > 0 and out.shape[0] > 0:
        width = min(int(time_mask_width), int(out.shape[0]))
        start_max = max(int(out.shape[0]) - width, 0)
        start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
        out[start : start + width] = 0.0

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out = out + noise

    return out.astype(np.float32)


class BiGRUSentenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        pooling: str = "mean",
    ):
        super().__init__()
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required to construct BiGRUSentenceClassifier")
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        self.pooling = pooling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        pooled_dim = hidden_size * 2
        if pooling == "mean_max":
            pooled_dim *= 2
        self.fc = nn.Linear(pooled_dim, num_classes)

    def pool_sequence(self, out: torch.Tensor) -> torch.Tensor:
        mean_pool = out.mean(dim=1)
        if self.pooling == "mean":
            return mean_pool
        max_pool = out.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = self.pool_sequence(out)
        return self.fc(self.drop(pooled))
