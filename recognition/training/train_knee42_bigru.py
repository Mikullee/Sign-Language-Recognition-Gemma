#!/usr/bin/env python3
"""Train/evaluate the gated Knee42 experiments from knee-free features_final.

Selection is strictly Dev macro Top-1.  Test is evaluated after a run has
finished and never feeds model selection or configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier


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
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    label_to_idx = {label: index for index, label in enumerate(LABELS)}
    display = {label: next(row["display_text"] for row in rows if row["label_id"] == label) for label in LABELS}
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    test_rows = [row for row in rows if row["split"] == "test"]
    for name, group in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
        missing = set(LABELS) - {row["label_id"] for row in group}
        if missing: raise ValueError(f"{name} missing labels: {sorted(missing)}")
    mean, std = fit_standardizer(train_rows, feature_dir)
    np.savez_compressed(out_dir / "standardizer_train_only.npz", mean=mean, std=std)
    datasets = {name: Knee42Dataset(group, feature_dir, label_to_idx, mean, std, config.sequence_length) for name, group in (("train", train_rows), ("dev", dev_rows), ("test", test_rows))}
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, sampler=sampler(train_rows, seed), num_workers=0, collate_fn=collate_batch)
    dev_loader = DataLoader(datasets["dev"], batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    input_dim = mean.size * 2
    model = BiGRUSentenceClassifier(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout, num_classes=len(LABELS), pooling="mean_max").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best = -1.0; best_epoch = 0; patience = config.patience; history = []
    for epoch in range(1, config.epochs + 1):
        model.train(); train_loss = []; correct = total = 0
        for features, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device)); loss = criterion(logits, labels.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss.append(float(loss.item())); correct += int((logits.argmax(1).cpu() == labels).sum()); total += len(labels)
        dev_result, _, _, _ = evaluate(model, dev_loader, device)
        scheduler.step(dev_result["macro_top1"])
        record = {"epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]), "train_loss": float(np.mean(train_loss)), "train_overall_top1": correct / max(total, 1), **{f"dev_{key}": value for key, value in dev_result.items() if key != "per_class_top1"}}
        history.append(record); print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {"checkpoint_version": "knee42_bigru_v1", "state_dict": model.state_dict(), "model_config": {"input_dim": input_dim, "hidden_size": config.hidden_size, "num_layers": config.num_layers, "dropout": config.dropout, "pooling": "mean_max", "num_classes": len(LABELS)}, "seed": seed, "manifest_sha256": manifest_hash, "split_sha256": split_hash, "feature_contract": "knee42_features_final_without_pose_25_26_or_knee_masks"}
        atomic_torch(out_dir / "last_checkpoint.pt", checkpoint)
        if dev_result["macro_top1"] > best:
            best = dev_result["macro_top1"]; best_epoch = epoch; patience = config.patience; atomic_torch(out_dir / "best_model.pt", checkpoint)
        else:
            patience -= 1
            if patience <= 0: break
    best_checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])
    dev_result, dev_confusion, dev_meta, dev_probs = evaluate(model, dev_loader, device)
    write_evaluation(out_dir, "dev", dev_result, dev_confusion, dev_meta, dev_probs, label_to_idx)
    # Formal held-out J test: evaluated only after selecting this run's checkpoint by Dev.
    test_result, test_confusion, test_meta, test_probs = evaluate(model, test_loader, device)
    write_evaluation(out_dir, "test", test_result, test_confusion, test_meta, test_probs, label_to_idx)
    atomic_json(out_dir / "train_history.json", history)
    atomic_json(out_dir / "label_map_knee42.json", {"label_to_idx": label_to_idx, "idx_to_label": LABELS})
    atomic_json(out_dir / "display_text_map.json", display)
    atomic_json(out_dir / "feature_config.json", {"features_final": "knee42_features_final_v1", "input_dim": input_dim, "sequence_length": config.sequence_length, "standardizer": "fit_train_split_observed_values_only_then_neutral_fill", "mask_concatenated": True, "knee_indices_removed": [25, 26]})
    atomic_json(out_dir / "training_config.json", {**asdict(config), "seed": seed, "device": str(device), "selection_metric": "dev_macro_top1", "test_used_for_selection": False})
    (out_dir / "manifest_sha256.txt").write_text(manifest_hash + "\n", encoding="ascii"); (out_dir / "split_sha256.txt").write_text(split_hash + "\n", encoding="ascii")
    summary = {"seed": seed, "best_epoch": best_epoch, "selection_metric": "dev_macro_top1", "dev": {key: dev_result[key] for key in ("macro_top1", "overall_top1", "top3")}, "test": {key: test_result[key] for key in ("macro_top1", "overall_top1", "top3")}, "device": str(device), "gate": "PROVISIONAL"}
    atomic_json(out_dir / "train_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run three fixed-seed gated Knee42 experiments.")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--gate-summary", required=True, type=Path); parser.add_argument("--feature-dir", required=True, type=Path); parser.add_argument("--out-dir", required=True, type=Path); parser.add_argument("--champion-dir", required=True, type=Path); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44]); parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args(argv)
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    if gate.get("gate") not in {"READY", "PROVISIONAL"}: raise SystemExit(f"gate is {gate.get('gate')}; training forbidden")
    rows = read_csv(args.manifest)
    manifest_hash = (args.manifest.parent / "manifest_sha256.txt").read_text(encoding="ascii").strip(); split_hash = (args.manifest.parent / "split_sha256.txt").read_text(encoding="ascii").strip()
    config = Config(); device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        run_dir = args.out_dir / f"seed_{seed}"
        try:
            summaries.append(train_one(rows, manifest_hash, split_hash, args.feature_dir, run_dir, seed, device, config))
        except RuntimeError as exc:
            allocation_failure = any(token in str(exc).lower() for token in ("out of memory", "cuda error", "cublas_status_alloc_failed"))
            if device.type != "cuda" or not allocation_failure:
                raise
            # Existing services own most GPU memory.  Preserve the run and safely
            # retry on CPU rather than terminating an unrelated service.
            print(json.dumps({"seed": seed, "device_fallback": "cpu", "reason": str(exc)}), flush=True)
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            summaries.append(train_one(rows, manifest_hash, split_hash, args.feature_dir, run_dir, seed, device, config))
    champion = max(summaries, key=lambda item: item["dev"]["macro_top1"])
    champion_source = args.out_dir / f"seed_{champion['seed']}"
    if args.champion_dir.exists(): shutil.rmtree(args.champion_dir)
    shutil.copytree(champion_source, args.champion_dir)
    aggregate = {"gate": gate["gate"], "selection_metric": "dev_macro_top1", "champion_seed": champion["seed"], "runs": summaries, "mean_std": {split: {metric: {"mean": float(np.mean([item[split][metric] for item in summaries])), "std": float(np.std([item[split][metric] for item in summaries], ddof=0))} for metric in ("macro_top1", "overall_top1", "top3")} for split in ("dev", "test")}}
    atomic_json(args.out_dir / "experiment_summary.json", aggregate); atomic_json(args.champion_dir / "champion_summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
