"""Train/Dev-only diagnostic helpers for Knee42 experiments."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from recognition.training.knee42_policy import LeakageError, validate_research_rows
from recognition.training.train_knee42_bigru import atomic_csv, atomic_json, load_cache


def validate_diagnostic_provenance(
    rows: Iterable[dict[str, str]], predictions: Iterable[dict[str, str]]
) -> None:
    research_rows = list(rows)
    validate_research_rows(research_rows)
    dev_ids = {
        item["sample_id"]
        for item in research_rows
        if item.get("split", "").strip().lower() == "dev"
        and item.get("signer_id", "").strip().upper() == "H"
    }
    for item in predictions:
        sample_id = item.get("sample_id", "")
        signer = item.get("signer_id", "").strip().upper()
        if signer != "H" or sample_id not in dev_ids:
            raise LeakageError(
                f"forbidden diagnostic prediction sample_id={sample_id!r} signer={signer!r}"
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def write_diagnostics(
    rows: list[dict[str, str]], feature_dir: Path, candidate_dir: Path, out_dir: Path
) -> dict[str, Any]:
    validate_research_rows(rows)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {out_dir}")
    predictions = _read_csv(candidate_dir / "dev_predictions.csv")
    validate_diagnostic_provenance(rows, predictions)

    split_lengths: dict[str, list[int]] = defaultdict(list)
    split_missing: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    class_counts: Counter[tuple[str, str]] = Counter()
    for item in rows:
        split = item["split"].strip().lower()
        values, mask = load_cache(feature_dir / f"{item['sample_id']}.npz")
        split_lengths[split].append(int(values.shape[0]))
        split_missing[split][0] += int((~mask).sum())
        split_missing[split][1] += int(mask.size)
        class_counts[(split, item["label_id"])] += 1

    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    error_samples: list[dict[str, Any]] = []
    for item in predictions:
        label = item["true_label"]
        predicted = item["pred_label"]
        correct = int(item.get("correct", "0"))
        per_class_total[label] += 1
        per_class_correct[label] += correct
        if not correct:
            confusions[(label, predicted)] += 1
            error_samples.append(
                {
                    "sample_id": item["sample_id"],
                    "true_label": label,
                    "pred_label": predicted,
                    "top1_probability": float(item.get("top1_probability", 0.0)),
                }
            )
    weak_classes = [
        {
            "label_id": label,
            "samples": per_class_total[label],
            "correct": per_class_correct[label],
            "accuracy": per_class_correct[label] / per_class_total[label],
        }
        for label in sorted(per_class_total)
    ]
    weak_classes.sort(key=lambda item: (item["accuracy"], item["label_id"]))
    confusion_pairs = [
        {"true_label": truth, "pred_label": predicted, "count": count}
        for (truth, predicted), count in confusions.most_common()
    ]
    history = json.loads((candidate_dir / "train_history.json").read_text(encoding="utf-8"))
    last = history[-1] if history else {}
    total_missing = sum(item[0] for item in split_missing.values())
    total_coordinates = sum(item[1] for item in split_missing.values())
    result: dict[str, Any] = {
        "policy": "train_lpx_dev_h_only",
        "sample_counts": dict(sorted(Counter(item["split"].strip().lower() for item in rows).items())),
        "provenance_signers": {
            split: sorted({item["signer_id"].strip().upper() for item in rows if item["split"].strip().lower() == split})
            for split in ("train", "dev")
        },
        "class_counts": [
            {"split": split, "label_id": label, "samples": count}
            for (split, label), count in sorted(class_counts.items())
        ],
        "features": {
            "overall_missing_rate": total_missing / max(total_coordinates, 1),
            "missing_rate_by_split": {
                split: missing / max(total, 1)
                for split, (missing, total) in sorted(split_missing.items())
            },
            "sequence_lengths": {
                split: _quantiles(lengths) for split, lengths in sorted(split_lengths.items())
            },
        },
        "weak_classes": weak_classes,
        "confusion_pairs": confusion_pairs,
        "error_samples": sorted(error_samples, key=lambda item: item["top1_probability"], reverse=True),
        "loss": {
            "last_train_loss": float(last.get("train_loss", 0.0)),
            "last_dev_loss": float(last.get("dev_loss", 0.0)),
            "last_accuracy_gap": float(last.get("train_overall_top1", 0.0))
            - float(last.get("dev_overall_top1", 0.0)),
        },
    }

    out_dir.mkdir(parents=True)
    atomic_json(out_dir / "diagnostics.json", result)
    atomic_csv(out_dir / "weak_classes.csv", ["label_id", "samples", "correct", "accuracy"], weak_classes)
    atomic_csv(out_dir / "confusion_pairs.csv", ["true_label", "pred_label", "count"], confusion_pairs)
    atomic_csv(out_dir / "error_samples.csv", ["sample_id", "true_label", "pred_label", "top1_probability"], result["error_samples"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    epochs = [int(item["epoch"]) for item in history]
    axis.plot(epochs, [float(item["train_loss"]) for item in history], label="Train loss")
    axis.plot(epochs, [float(item["dev_loss"]) for item in history], label="Dev loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(figure)
    return result
