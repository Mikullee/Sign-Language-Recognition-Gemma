#!/usr/bin/env python3
"""Shared Knee42 feature, dataset, and evaluation utilities.

The original train-and-Test entry point is retired because the one-time J/Test
budget has been consumed.  New research training must use the Dev-only module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from recognition.training.knee42_policy import validate_research_rows


LABELS = [f"K42_{number:02d}" for number in range(1, 43)]
CACHE_VERSION = "knee42_features_upright_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=path.stem + ".", suffix=".json", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", prefix=path.stem + ".", suffix=".csv", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_path(feature_dir: Path, sample_id: str) -> Path:
    return feature_dir / f"{sample_id}.npz"


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["cache_version"].item()) != CACHE_VERSION:
            raise ValueError(f"unsupported feature cache {path}")
        values = payload["values"].astype(np.float32)
        mask = payload["mask"].astype(np.bool_)
    if values.ndim != 2 or values.shape != mask.shape or values.shape[0] == 0:
        raise ValueError(f"invalid cache shape {path}: {values.shape} / {mask.shape}")
    if np.any(np.isfinite(values) & ~mask):
        raise ValueError(f"cache mask/value mismatch: {path}")
    return values, mask


def select_fixed_frames(values: np.ndarray, mask: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.rint(np.linspace(0, len(values) - 1, sequence_length)).astype(np.int64)
    return values[indices], mask[indices]


def fit_standardizer(rows: list[dict[str, str]], feature_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fit observed-coordinate statistics on training rows only; never fill raw NaNs."""
    count = total = squared = None
    for row in rows:
        values, mask = load_cache(cache_path(feature_dir, row["sample_id"]))
        observed = np.where(mask, values, 0.0).astype(np.float64)
        sample_count = mask.sum(axis=0, dtype=np.int64)
        if count is None:
            count = sample_count
            total = observed.sum(axis=0)
            squared = (observed * observed).sum(axis=0)
        else:
            count += sample_count
            total += observed.sum(axis=0)
            squared += (observed * observed).sum(axis=0)
    if count is None or np.any(count == 0):
        missing = np.flatnonzero(np.asarray(count) == 0).tolist() if count is not None else []
        raise ValueError(f"training standardizer has unobserved coordinates: {missing}")
    mean = total / count
    variance = np.maximum(squared / count - mean * mean, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class Knee42Dataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], feature_dir: Path, label_to_idx: dict[str, int], mean: np.ndarray, std: np.ndarray, sequence_length: int):
        self.rows, self.feature_dir, self.label_to_idx = rows, feature_dir, label_to_idx
        self.mean, self.std, self.sequence_length = mean, std, sequence_length
        missing = [row["sample_id"] for row in rows if not cache_path(feature_dir, row["sample_id"]).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} features_final caches: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        values, mask = load_cache(cache_path(self.feature_dir, row["sample_id"]))
        values, mask = select_fixed_frames(values, mask, self.sequence_length)
        standardized = (values - self.mean) / self.std
        # Neutral fill happens only after observed-value standardization.
        standardized = np.where(mask, standardized, 0.0).astype(np.float32)
        features = np.concatenate([standardized, mask.astype(np.float32)], axis=1)
        return torch.from_numpy(features), torch.tensor(self.label_to_idx[row["label_id"]], dtype=torch.long), row


def collate_batch(batch):
    features, labels, rows = zip(*batch)
    return torch.stack(features), torch.stack(labels), list(rows)


def sampler(rows: list[dict[str, str]], seed: int) -> WeightedRandomSampler:
    counts = Counter(row["label_id"] for row in rows)
    weights = torch.tensor([1.0 / counts[row["label_id"]] for row in rows], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True, generator=generator)


def metrics(y_true: list[int], probabilities: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = probabilities.argmax(axis=1).astype(np.int64)
    per_class = []
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for target, output in zip(truth, pred):
        confusion[target, output] += 1
    for index in range(len(LABELS)):
        selected = truth == index
        per_class.append(float(np.mean(pred[selected] == index)) if np.any(selected) else 0.0)
    k = min(3, len(LABELS))
    top = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    return {"overall_top1": float(np.mean(pred == truth)), "macro_top1": float(np.mean(per_class)), "top3": float(np.mean([target in row for target, row in zip(truth.tolist(), top.tolist())])), "per_class_top1": per_class}, confusion


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray, list[dict[str, str]], np.ndarray]:
    model.eval()
    y_true: list[int] = []
    prob_parts: list[np.ndarray] = []
    rows: list[dict[str, str]] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, labels, batch_rows in loader:
            logits = model(features.to(device))
            losses.append(float(criterion(logits, labels.to(device)).item()))
            y_true.extend(labels.tolist())
            prob_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
            rows.extend(batch_rows)
    probs = np.concatenate(prob_parts, axis=0)
    result, confusion = metrics(y_true, probs)
    result["loss"] = float(np.mean(losses))
    return result, confusion, rows, probs


def write_evaluation(out_dir: Path, prefix: str, result: dict[str, Any], confusion: np.ndarray, rows: list[dict[str, str]], probabilities: np.ndarray, label_to_idx: dict[str, int]) -> None:
    atomic_json(out_dir / f"{prefix}_metrics.json", result)
    per_class = [{"label_id": label, "class_index": label_to_idx[label], "accuracy": result["per_class_top1"][label_to_idx[label]]} for label in LABELS]
    atomic_csv(out_dir / f"{prefix}_per_class_accuracy.csv", ["label_id", "class_index", "accuracy"], per_class)
    atomic_csv(out_dir / f"{prefix}_confusion.csv", ["true_label", *LABELS], [{"true_label": label, **{other: int(confusion[label_to_idx[label], label_to_idx[other]]) for other in LABELS}} for label in LABELS])
    prediction_rows = []
    for row, probs in zip(rows, probabilities):
        ranked = np.argsort(-probs)[:3]
        prediction_rows.append({"sample_id": row["sample_id"], "source": row["source"], "signer_id": row["signer_id"], "true_label": row["label_id"], "pred_label": LABELS[int(ranked[0])], "top1_probability": float(probs[ranked[0]]), "top3_labels": ";".join(LABELS[int(index)] for index in ranked), "correct": int(row["label_id"] == LABELS[int(ranked[0])])})
    atomic_csv(out_dir / f"{prefix}_predictions.csv", ["sample_id", "source", "signer_id", "true_label", "pred_label", "top1_probability", "top3_labels", "correct"], prediction_rows)
    signer_rows = []
    for signer in sorted({row["signer_id"] for row in rows}):
        selected = [index for index, row in enumerate(rows) if row["signer_id"] == signer]
        truth = [label_to_idx[rows[index]["label_id"]] for index in selected]
        signer_result, _ = metrics(truth, probabilities[selected])
        signer_rows.append({"signer_id": signer, "samples": len(selected), "macro_top1": signer_result["macro_top1"], "overall_top1": signer_result["overall_top1"], "top3": signer_result["top3"]})
    atomic_csv(out_dir / f"{prefix}_per_signer.csv", ["signer_id", "samples", "macro_top1", "overall_top1", "top3"], signer_rows)


@dataclass(frozen=True)
class Config:
    sequence_length: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.45
    batch_size: int = 24
    epochs: int = 45
    patience: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    label_smoothing: float = 0.08


def train_one(rows: list[dict[str, str]], manifest_hash: str, split_hash: str, feature_dir: Path, out_dir: Path, seed: int, device: torch.device, config: Config) -> dict[str, Any]:
    """Retired legacy trainer kept only for import compatibility.

    The original implementation evaluated held-out J/Test after every seed.  That
    one-time test budget has already been consumed, so this entry point now
    fails closed.  Shared dataset and evaluation helpers remain in this module
    for the Dev-only trainer.
    """
    validate_research_rows(rows)
    raise RuntimeError(
        "legacy trainer retired: use scripts/train_knee42_devonly.py so only "
        "Train/Dev rows can enter the research path"
    )


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(
        "legacy trainer retired; run python scripts/train_knee42_devonly.py --help"
    )


if __name__ == "__main__":
    main()
