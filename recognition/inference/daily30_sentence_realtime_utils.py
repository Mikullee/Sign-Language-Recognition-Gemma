from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from recognition.config import preview_paths


DEFAULT_CACHE_DIR = preview_paths().runtime_bundle_dir


def artifact_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "best_model": cache_dir / "best_model.pt",
        "label_map": cache_dir / "label_map_v1.json",
        "train_summary": cache_dir / "train_summary_v1.json",
        "launch_summary": cache_dir / "launch_summary.json",
        "templates": cache_dir / "fixed_sentence_templates_daily30.csv",
        "auto_trigger": cache_dir / "best_auto_trigger.json",
    }


def ensure_artifacts_cached(cache_dir: Path) -> Path:
    paths = artifact_paths(cache_dir)
    required = [
        "best_model",
        "label_map",
        "train_summary",
        "launch_summary",
        "templates",
        "auto_trigger",
    ]
    missing = [
        paths[key].name
        for key in required
        if not paths[key].is_file() or paths[key].stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError(
            "The offline runtime bundle is incomplete. Missing or empty files: "
            + ", ".join(missing)
        )
    return cache_dir


def load_runtime_bundle(cache_dir: Path) -> dict:
    ensure_artifacts_cached(cache_dir)
    paths = artifact_paths(cache_dir)
    label_map = json.loads(paths["label_map"].read_text(encoding="utf-8"))
    train_summary = json.loads(paths["train_summary"].read_text(encoding="utf-8"))
    launch_summary = json.loads(
        paths["launch_summary"].read_text(encoding="utf-8")
    )
    with paths["templates"].open("r", encoding="utf-8-sig", newline="") as handle:
        template_rows = list(csv.DictReader(handle))

    idx_pairs = sorted(
        (int(index), label)
        for index, label in label_map["idx_to_label"].items()
    )
    labels = [label for _, label in idx_pairs]
    template_ids = [
        row["template_id"].strip()
        for row in template_rows
        if row.get("template_id", "").strip()
    ]
    if template_ids != labels:
        raise ValueError(
            "Runtime template IDs must exactly match model labels in index order."
        )
    label_display = {
        row["template_id"].strip(): row["sentence_text"].strip()
        for row in template_rows
        if row.get("template_id", "").strip()
        and row.get("sentence_text", "").strip()
    }
    return {
        "labels": labels,
        "label_display": label_display,
        "label_to_idx": {
            label: int(index)
            for label, index in label_map["label_to_idx"].items()
        },
        "run_name": launch_summary.get("run_name", cache_dir.name),
        "sequence_length": int(launch_summary["sequence_length"]),
        "frame_step": int(launch_summary["frame_step"]),
        "hidden_size": int(launch_summary["hidden_size"]),
        "num_layers": int(launch_summary["num_layers"]),
        "dropout": float(launch_summary["dropout"]),
        "pooling": str(
            launch_summary.get("pooling", train_summary.get("pooling", "mean_max"))
        ),
        "append_delta": bool(launch_summary.get("append_delta", True)),
        "zscore_features": bool(launch_summary.get("zscore_features", True)),
        "device": str(launch_summary.get("device", "auto")),
        "train_summary": train_summary,
        "launch_summary": launch_summary,
        "paths": paths,
    }


def choose_smoothed_prediction(
    history: list[tuple[str, float]],
) -> tuple[str, float]:
    if not history:
        return "WAITING", 0.0
    grouped: dict[str, list[float]] = {}
    for label, confidence in history:
        grouped.setdefault(label, []).append(float(confidence))
    best_label = max(
        grouped.items(),
        key=lambda item: (len(item[1]), sum(item[1]) / len(item[1])),
    )[0]
    confidences = grouped[best_label]
    return best_label, float(sum(confidences) / len(confidences))


def compute_runtime_status(
    buffer_size: int,
    window_size: int,
    stable_streak: int,
    min_streak: int,
    display_label: str,
) -> str:
    if buffer_size < window_size:
        return "BUFFERING"
    if (
        display_label in {"UNKNOWN", "WAITING", "BUFFERING"}
        or stable_streak < min_streak
    ):
        return "TRACKING"
    return "STABLE"


def save_prediction_logs(
    out_dir: Path,
    prediction_rows: list[dict],
    session_payload: dict,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"realtime_sentence_predictions_{stamp}.csv"
    json_path = out_dir / f"realtime_sentence_session_{stamp}.json"

    if prediction_rows:
        fieldnames = list(prediction_rows[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(
        json.dumps(session_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path
