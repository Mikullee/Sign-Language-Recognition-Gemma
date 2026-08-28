"""Train Knee42 candidates without materializing or evaluating Test rows."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from recognition.inference.bigru_sentence_model import BiGRUSentenceClassifier
from recognition.training.knee42_policy import validate_research_rows
from recognition.training.train_knee42_bigru import (
    CACHE_VERSION,
    LABELS,
    Knee42Dataset,
    atomic_json,
    atomic_torch,
    cache_path,
    collate_batch,
    evaluate,
    fit_standardizer,
    load_cache,
    sampler,
    select_fixed_frames,
    write_evaluation,
)


@dataclass(frozen=True)
class DevOnlyConfig:
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
    focal_gamma: float = 2.0
    sampler: str = "balanced"
    loss: str = "cross_entropy"
    augmentation: str = "none"
    coordinate_scale_jitter: float = 0.10
    coordinate_translation_jitter: float = 0.035
    landmark_dropout_probability: float = 0.0
    normalization: str = "observed_train_standardizer"
    feature_mode: str = "values_mask"
    pooling: str = "mean_max"


def _validate_config(config: DevOnlyConfig) -> None:
    supported = {
        "sampler": (config.sampler, {"balanced"}),
        "loss": (config.loss, {"cross_entropy", "focal"}),
        "augmentation": (
            config.augmentation,
            {"none", "coordinate_jitter", "coordinate_jitter_landmark_dropout"},
        ),
        "normalization": (config.normalization, {"observed_train_standardizer"}),
        "feature_mode": (config.feature_mode, {"values_mask"}),
        "pooling": (config.pooling, {"mean", "mean_max"}),
    }
    for name, (value, choices) in supported.items():
        if value not in choices:
            raise ValueError(f"unsupported {name}: {value}")
    if config.focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    if config.loss == "focal" and config.label_smoothing != 0:
        raise ValueError("focal loss requires label_smoothing=0")
    if not 0 <= config.coordinate_scale_jitter < 1:
        raise ValueError("coordinate_scale_jitter must be in [0,1)")
    if config.coordinate_translation_jitter < 0:
        raise ValueError("coordinate_translation_jitter must be non-negative")
    if not 0 <= config.landmark_dropout_probability < 1:
        raise ValueError("landmark_dropout_probability must be in [0,1)")


class FocalLoss(nn.Module):
    """Multiclass focal loss over logits."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        return (-((1.0 - pt) ** self.gamma) * log_pt).mean()


def build_criterion(config: DevOnlyConfig) -> nn.Module:
    _validate_config(config)
    if config.loss == "focal":
        return FocalLoss(gamma=config.focal_gamma)
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def coordinate_affine_jitter(
    values: np.ndarray,
    mask: np.ndarray,
    rng: Any = np.random,
    *,
    scale_jitter: float = 0.10,
    translation_jitter: float = 0.035,
) -> np.ndarray:
    """Apply one small camera-geometry transform to a complete landmark sequence."""
    if values.shape != mask.shape or values.ndim != 2 or values.shape[1] != 219:
        raise ValueError("coordinate jitter expects matching [frames,219] values and mask")
    if not 0 <= scale_jitter < 1:
        raise ValueError("scale_jitter must be in [0,1)")
    if translation_jitter < 0:
        raise ValueError("translation_jitter must be non-negative")
    if scale_jitter == 0 and translation_jitter == 0:
        return values.copy()
    scale = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
    translate_x = float(rng.uniform(-translation_jitter, translation_jitter))
    translate_y = float(rng.uniform(-translation_jitter, translation_jitter))
    result = values.copy()
    for offset, translation, center in (
        (0, translate_x, 0.5),
        (1, translate_y, 0.5),
        (2, 0.0, 0.0),
    ):
        indices = np.arange(offset, values.shape[1], 3)
        transformed = center + scale * (values[:, indices] - center) + translation
        result[:, indices] = np.where(mask[:, indices], transformed, values[:, indices])
    return result.astype(np.float32, copy=False)


def landmark_mask_dropout(
    values: np.ndarray,
    mask: np.ndarray,
    rng: Any = np.random,
    *,
    dropout_probability: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop complete xyz landmark triples to simulate Train-only detector loss."""
    if values.shape != mask.shape or values.ndim != 2 or values.shape[1] != 219:
        raise ValueError("landmark dropout expects matching [frames,219] values and mask")
    if not 0 <= dropout_probability < 1:
        raise ValueError("dropout_probability must be in [0,1)")
    if dropout_probability == 0:
        return values.copy(), mask.copy()
    drop_points = rng.random((values.shape[0], values.shape[1] // 3)) < dropout_probability
    drop_coordinates = np.repeat(drop_points[:, :, None], 3, axis=2).reshape(mask.shape)
    result_mask = mask & ~drop_coordinates
    result_values = values.copy()
    result_values[~result_mask] = np.nan
    return result_values.astype(np.float32, copy=False), result_mask


class DevOnlyKnee42Dataset(Knee42Dataset):
    """Knee42 dataset with an explicitly Train-only augmentation hook."""

    def __init__(
        self,
        *args: Any,
        augmentation: str = "none",
        coordinate_scale_jitter: float = 0.10,
        coordinate_translation_jitter: float = 0.035,
        landmark_dropout_probability: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.augmentation = augmentation
        self.coordinate_scale_jitter = coordinate_scale_jitter
        self.coordinate_translation_jitter = coordinate_translation_jitter
        self.landmark_dropout_probability = landmark_dropout_probability

    def __getitem__(self, index: int):
        if self.augmentation == "none":
            return super().__getitem__(index)
        if self.augmentation not in {
            "coordinate_jitter",
            "coordinate_jitter_landmark_dropout",
        }:
            raise ValueError(f"unsupported dataset augmentation: {self.augmentation}")
        row = self.rows[index]
        values, mask = load_cache(cache_path(self.feature_dir, row["sample_id"]))
        values, mask = select_fixed_frames(values, mask, self.sequence_length)
        values = coordinate_affine_jitter(
            values,
            mask,
            scale_jitter=self.coordinate_scale_jitter,
            translation_jitter=self.coordinate_translation_jitter,
        )
        if self.augmentation == "coordinate_jitter_landmark_dropout":
            values, mask = landmark_mask_dropout(
                values,
                mask,
                dropout_probability=self.landmark_dropout_probability,
            )
        standardized = (values - self.mean) / self.std
        standardized = np.where(mask, standardized, 0.0).astype(np.float32)
        features = np.concatenate([standardized, mask.astype(np.float32)], axis=1)
        return (
            torch.from_numpy(features),
            torch.tensor(self.label_to_idx[row["label_id"]], dtype=torch.long),
            row,
        )


def train_dev_only(
    rows: list[dict[str, str]],
    manifest_hash: str,
    split_hash: str,
    feature_ledger_hash: str,
    feature_dir: Path,
    out_dir: Path,
    seed: int,
    device: torch.device,
    config: DevOnlyConfig,
    max_train_batches: int | None = None,
) -> dict[str, Any]:
    validate_research_rows(rows)
    _validate_config(config)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {out_dir}")

    train_rows = [item for item in rows if item["split"].strip().lower() == "train"]
    dev_rows = [item for item in rows if item["split"].strip().lower() == "dev"]
    for name, group in (("train", train_rows), ("dev", dev_rows)):
        missing = set(LABELS) - {item["label_id"] for item in group}
        if missing:
            raise ValueError(f"{name} missing labels: {sorted(missing)}")

    out_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    label_to_idx = {label: index for index, label in enumerate(LABELS)}
    display = {label: next(item["display_text"] for item in rows if item["label_id"] == label) for label in LABELS}
    mean, std = fit_standardizer(train_rows, feature_dir)
    np.savez_compressed(out_dir / "standardizer_train_only.npz", mean=mean, std=std)
    train_dataset = DevOnlyKnee42Dataset(
        train_rows,
        feature_dir,
        label_to_idx,
        mean,
        std,
        config.sequence_length,
        augmentation=config.augmentation,
        coordinate_scale_jitter=config.coordinate_scale_jitter,
        coordinate_translation_jitter=config.coordinate_translation_jitter,
        landmark_dropout_probability=config.landmark_dropout_probability,
    )
    dev_dataset = DevOnlyKnee42Dataset(
        dev_rows,
        feature_dir,
        label_to_idx,
        mean,
        std,
        config.sequence_length,
        augmentation="none",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler(train_rows, seed),
        num_workers=0,
        collate_fn=collate_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    input_dim = int(mean.size * 2)
    model_config = {
        "input_dim": input_dim,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "pooling": config.pooling,
        "num_classes": len(LABELS),
    }
    model = BiGRUSentenceClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = build_criterion(config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best = -1.0
    best_epoch = 0
    patience_left = config.patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        correct = 0
        total = 0
        for batch_index, (features, labels, _) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(1).cpu() == labels).sum())
            total += len(labels)
        if not losses:
            raise ValueError("training produced no batches")
        dev_result, _, _, _ = evaluate(model, dev_loader, device)
        scheduler.step(dev_result["macro_top1"])
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)),
            "train_overall_top1": correct / max(total, 1),
            **{f"dev_{key}": value for key, value in dev_result.items() if key != "per_class_top1"},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "checkpoint_version": "knee42_devonly_v1",
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": asdict(config),
            "seed": seed,
            "manifest_sha256": manifest_hash,
            "split_sha256": split_hash,
            "feature_ledger_sha256": feature_ledger_hash,
            "feature_contract": "knee42_features_upright_v2_without_pose_25_26_or_knee_masks",
        }
        atomic_torch(out_dir / "last_checkpoint.pt", checkpoint)
        if dev_result["macro_top1"] > best:
            best = dev_result["macro_top1"]
            best_epoch = epoch
            patience_left = config.patience
            atomic_torch(out_dir / "best_model.pt", checkpoint)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    dev_result, dev_confusion, dev_rows_out, dev_probs = evaluate(model, dev_loader, device)
    write_evaluation(out_dir, "dev", dev_result, dev_confusion, dev_rows_out, dev_probs, label_to_idx)
    atomic_json(out_dir / "train_history.json", history)
    atomic_json(out_dir / "label_map_knee42.json", {"label_to_idx": label_to_idx, "idx_to_label": LABELS})
    atomic_json(out_dir / "display_text_map.json", display)
    atomic_json(
        out_dir / "feature_config.json",
        {
            "features_final": CACHE_VERSION,
            "input_dim": input_dim,
            "sequence_length": config.sequence_length,
            "standardizer": config.normalization,
            "mask_concatenated": True,
            "knee_indices_removed": [25, 26],
            "video_orientation": "container_rotation_metadata_applied_explicitly",
            "horizontal_mirror": False,
        },
    )
    atomic_json(out_dir / "training_config.json", {**asdict(config), "seed": seed, "device": str(device), "selection_metric": "dev_macro_top1"})
    (out_dir / "manifest_sha256.txt").write_text(manifest_hash + "\n", encoding="ascii")
    (out_dir / "split_sha256.txt").write_text(split_hash + "\n", encoding="ascii")
    (out_dir / "feature_ledger_sha256.txt").write_text(feature_ledger_hash + "\n", encoding="ascii")
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": "dev_macro_top1",
        "dev": {key: dev_result[key] for key in ("macro_top1", "overall_top1", "top3", "loss")},
        "device": str(device),
        "gate": "PROVISIONAL",
    }
    atomic_json(out_dir / "train_summary.json", summary)
    return summary
