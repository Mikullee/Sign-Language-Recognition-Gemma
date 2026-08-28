"""One-time J-only seal primitives for Knee42."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from recognition.evaluation.knee42_selection import (
    SelectionError,
    sha256_file,
    verify_selection,
)
from recognition.inference.bigru_sentence_model import BiGRUSentenceClassifier
from recognition.training.train_knee42_bigru import (
    LABELS,
    Knee42Dataset,
    atomic_json,
    collate_batch,
    evaluate,
    read_csv,
    write_evaluation,
)


def select_j_test_rows(
    rows: list[dict[str, str]], expected_count: int = 626
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for item in rows:
        split = item.get("split", "").strip().lower()
        signer = item.get("signer_id", "").strip().upper()
        if split == "test" or signer == "J":
            if split != "test" or signer != "J":
                raise SelectionError(
                    f"J-only Test contract violated by split={split!r} signer={signer!r}"
                )
            selected.append(item)
    if len(selected) != expected_count:
        raise SelectionError(f"expected {expected_count} J-only Test rows, found {len(selected)}")
    return selected


def prepare_test_once(test_root: Path, ledger_path: Path) -> dict[str, object]:
    if not ledger_path.is_file():
        raise SelectionError(f"selection ledger required before Test: {ledger_path}")
    ledger = verify_selection(ledger_path)
    consumed = test_root / "CONSUMED.json"
    started = test_root / "STARTED.json"
    if consumed.exists():
        raise SelectionError(f"one-time Test already consumed: {consumed}")
    ledger_hash = sha256_file(ledger_path)
    if started.exists():
        payload = json.loads(started.read_text(encoding="utf-8"))
        if payload.get("ledger_sha256") != ledger_hash:
            raise SelectionError("started Test ledger hash does not match")
        return payload
    test_root.mkdir(parents=True, exist_ok=False)
    payload: dict[str, object] = {
        "seal_version": "knee42_test_once_v1",
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": ledger_hash,
        "run_id": ledger["run_id"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with started.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def consume_test_seal(test_root: Path, ledger_path: Path) -> dict[str, object]:
    consumed = test_root / "CONSUMED.json"
    if consumed.exists():
        raise SelectionError(f"one-time Test already consumed: {consumed}")
    started = test_root / "STARTED.json"
    if not started.is_file():
        raise SelectionError("one-time Test has not started")
    verify_selection(ledger_path)
    started_payload = json.loads(started.read_text(encoding="utf-8"))
    ledger_hash = sha256_file(ledger_path)
    if started_payload.get("ledger_sha256") != ledger_hash:
        raise SelectionError("selection ledger changed after Test started")
    payload: dict[str, object] = {
        "seal_version": "knee42_test_once_v1",
        "ledger_sha256": ledger_hash,
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with consumed.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def evaluate_j_once(
    ledger_path: Path,
    manifest_path: Path,
    feature_dir: Path,
    out_dir: Path,
    *,
    device: torch.device,
    expected_count: int = 626,
) -> dict[str, Any]:
    """Evaluate the immutable selected model on J-only Test exactly once."""
    started = prepare_test_once(out_dir, ledger_path)
    ledger = verify_selection(ledger_path)

    expected_manifest_hash = ledger.get("original_manifest_sha256")
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise SelectionError(
            "original manifest SHA-256 mismatch: "
            f"expected {expected_manifest_hash}, actual {actual_manifest_hash}"
        )

    test_rows = select_j_test_rows(read_csv(manifest_path), expected_count=expected_count)
    labels = {row.get("label_id", "") for row in test_rows}
    if labels != set(LABELS):
        missing = sorted(set(LABELS) - labels)
        unexpected = sorted(labels - set(LABELS))
        raise SelectionError(
            f"J-only Test label contract violated; missing={missing}, unexpected={unexpected}"
        )

    candidate_dir = Path(str(ledger["candidate_dir"]))
    with np.load(candidate_dir / "standardizer_train_only.npz", allow_pickle=False) as payload:
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
    if mean.ndim != 1 or std.shape != mean.shape or np.any(std <= 0):
        raise SelectionError("locked train-only standardizer is invalid")

    label_payload = json.loads(
        (candidate_dir / "label_map_knee42.json").read_text(encoding="utf-8")
    )
    label_to_idx = {str(key): int(value) for key, value in label_payload["label_to_idx"].items()}
    if label_to_idx != {label: index for index, label in enumerate(LABELS)}:
        raise SelectionError("locked label map does not match the Knee42 contract")

    feature_config = json.loads(
        (candidate_dir / "feature_config.json").read_text(encoding="utf-8")
    )
    sequence_length = int(feature_config["sequence_length"])
    dataset = Knee42Dataset(
        test_rows,
        feature_dir,
        label_to_idx,
        mean,
        std,
        sequence_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    checkpoint = torch.load(
        candidate_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise SelectionError("locked checkpoint is missing model_config")
    if int(model_config.get("input_dim", -1)) != int(mean.size * 2):
        raise SelectionError("locked checkpoint input dimension does not match standardizer")
    if int(model_config.get("num_classes", -1)) != len(LABELS):
        raise SelectionError("locked checkpoint class count does not match Knee42")
    model = BiGRUSentenceClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    result, confusion, rows_out, probabilities = evaluate(model, loader, device)
    write_evaluation(
        out_dir,
        "test",
        result,
        confusion,
        rows_out,
        probabilities,
        label_to_idx,
    )
    summary: dict[str, Any] = {
        "samples": len(test_rows),
        "macro_top1": result["macro_top1"],
        "overall_top1": result["overall_top1"],
        "top3": result["top3"],
        "loss": result["loss"],
        "ledger_sha256": started["ledger_sha256"],
        "device": str(device),
    }
    atomic_json(out_dir / "test_summary.json", summary)
    consume_test_seal(out_dir, ledger_path)
    return summary
