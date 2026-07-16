from __future__ import annotations

import csv
import json
import os
import posixpath
from datetime import datetime
from pathlib import Path

import paramiko

from recognition.config import preview_paths

HOST = os.environ.get("SLR_REMOTE_HOST", "163.13.202.125")
PORT = int(os.environ.get("SLR_REMOTE_PORT", "2288"))
USERNAME = os.environ.get("SLR_REMOTE_USERNAME", "b310ai")
PASSWORD = os.environ.get("SLR_REMOTE_PASSWORD")

DEFAULT_REMOTE_RUN_DIR = "/home/b310ai/cslr_bench/results/daily30_sentence_runs/daily30_sentence_opt_global_signer_holdout_20260715_132011"
DEFAULT_CACHE_DIR = preview_paths().runtime_bundle_dir


def default_launch_summary(remote_run_dir: str) -> dict:
    return {
        "run_name": Path(remote_run_dir).name,
        "remote_run_dir": remote_run_dir,
        "sequence_length": 72,
        "frame_step": 1,
        "hidden_size": 160,
        "num_layers": 2,
        "dropout": 0.55,
        "pooling": "mean_max",
        "append_delta": True,
        "zscore_features": True,
        "device": "auto",
    }


def artifact_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "best_model": cache_dir / "best_model.pt",
        "label_map": cache_dir / "label_map_v1.json",
        "train_summary": cache_dir / "train_summary_v1.json",
        "launch_summary": cache_dir / "launch_summary.json",
        "templates": cache_dir / "fixed_sentence_templates_daily30.csv",
        "offline_metrics": cache_dir / "metrics_v1.json",
        "realtime_proxy_metrics": cache_dir / "metrics_realtime_proxy.json",
    }


def ensure_artifacts_cached(cache_dir: Path, remote_run_dir: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(cache_dir)
    required = ["best_model", "label_map", "train_summary", "launch_summary", "templates"]
    if all(paths[key].exists() and paths[key].stat().st_size > 0 for key in required):
        return cache_dir

    if os.environ.get("SLR_DISABLE_REMOTE_FETCH", "0") == "1":
        missing = [key for key in required if not paths[key].exists()]
        raise FileNotFoundError(
            "Missing local runtime artifacts and remote fetch is disabled. "
            f"Missing: {', '.join(missing)}"
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
    try:
        sftp = client.open_sftp()
        try:
            remote_artifacts = posixpath.join(remote_run_dir, "artifacts", "v1")
            sftp.get(posixpath.join(remote_artifacts, "best_model.pt"), str(paths["best_model"]))
            sftp.get(posixpath.join(remote_artifacts, "label_map_v1.json"), str(paths["label_map"]))
            sftp.get(posixpath.join(remote_artifacts, "train_summary_v1.json"), str(paths["train_summary"]))
            sftp.get(posixpath.join(remote_run_dir, "fixed_sentence_templates_daily30.csv"), str(paths["templates"]))
            try:
                sftp.get(posixpath.join(remote_run_dir, "launch_summary.json"), str(paths["launch_summary"]))
            except FileNotFoundError:
                launch_summary = default_launch_summary(remote_run_dir)
                paths["launch_summary"].write_text(json.dumps(launch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            sftp.close()
    finally:
        client.close()
    return cache_dir


def load_runtime_bundle(cache_dir: Path) -> dict:
    paths = artifact_paths(cache_dir)
    label_map = json.loads(paths["label_map"].read_text(encoding="utf-8"))
    train_summary = json.loads(paths["train_summary"].read_text(encoding="utf-8"))
    with paths["templates"].open("r", encoding="utf-8-sig", newline="") as f:
        template_rows = list(csv.DictReader(f))
    launch_text = paths["launch_summary"].read_text(encoding="utf-8").strip()
    if not launch_text:
        launch_summary = default_launch_summary(DEFAULT_REMOTE_RUN_DIR)
        paths["launch_summary"].write_text(json.dumps(launch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        launch_summary = json.loads(launch_text)

    idx_pairs = sorted((int(k), v) for k, v in label_map["idx_to_label"].items())
    labels = [v for _, v in idx_pairs]
    label_display = {row["template_id"]: row["sentence_text"] for row in template_rows if row.get("template_id") and row.get("sentence_text")}

    return {
        "labels": labels,
        "label_display": label_display,
        "label_to_idx": {k: int(v) for k, v in label_map["label_to_idx"].items()},
        "run_name": launch_summary.get("run_name", cache_dir.name),
        "sequence_length": int(launch_summary["sequence_length"]),
        "frame_step": int(launch_summary["frame_step"]),
        "hidden_size": int(launch_summary["hidden_size"]),
        "num_layers": int(launch_summary["num_layers"]),
        "dropout": float(launch_summary["dropout"]),
        "pooling": str(launch_summary.get("pooling", train_summary.get("pooling", "mean_max"))),
        "append_delta": bool(launch_summary.get("append_delta", True)),
        "zscore_features": bool(launch_summary.get("zscore_features", True)),
        "device": str(launch_summary.get("device", "auto")),
        "train_summary": train_summary,
        "launch_summary": launch_summary,
        "paths": paths,
    }


def choose_smoothed_prediction(history: list[tuple[str, float]]) -> tuple[str, float]:
    if not history:
        return "WAITING", 0.0
    grouped: dict[str, list[float]] = {}
    for label, confidence in history:
        grouped.setdefault(label, []).append(float(confidence))
    best_label = max(grouped.items(), key=lambda item: (len(item[1]), sum(item[1]) / len(item[1])))[0]
    confidences = grouped[best_label]
    return best_label, float(sum(confidences) / len(confidences))


def compute_runtime_status(buffer_size: int, window_size: int, stable_streak: int, min_streak: int, display_label: str) -> str:
    if buffer_size < window_size:
        return "BUFFERING"
    if display_label in {"UNKNOWN", "WAITING", "BUFFERING"} or stable_streak < min_streak:
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
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(json.dumps(session_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path
