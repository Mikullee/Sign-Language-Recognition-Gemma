"""Transformer encoder classifier for the 42-class Knee42 contract.

Attribute names are load-bearing: they define the ``state_dict`` keys of the
published checkpoints, so ``proj`` / ``pos`` / ``encoder`` / ``norm`` / ``head``
must not be renamed without reissuing every bundle.
"""
from __future__ import annotations

import torch
from torch import nn

from recognition.transformer.features import MODEL_INPUT_DIM, SEQUENCE_LENGTH


DEFAULT_MODEL_DIM = 256
DEFAULT_LAYERS = 4
DEFAULT_HEADS = 8


class Knee42Transformer(nn.Module):
    """Mean-pooled Transformer encoder over ``[64, 657]`` sequences."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = MODEL_INPUT_DIM,
        model_dim: int = DEFAULT_MODEL_DIM,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            model_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        # norm_first already rules out the nested-tensor fast path; saying so
        # explicitly keeps PyTorch from warning about it on every construction.
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(sequences) + self.pos
        hidden = self.encoder(hidden)
        return self.head(self.norm(hidden.mean(dim=1)))
