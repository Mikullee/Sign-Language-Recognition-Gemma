"""Immutable, hash-bound Dev selection ledger for Knee42."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_CANDIDATE_FILES = (
    "best_model.pt",
    "training_config.json",
    "feature_config.json",
    "standardizer_train_only.npz",
    "label_map_knee42.json",
    "display_text_map.json",
    "manifest_sha256.txt",
    "split_sha256.txt",
    "feature_ledger_sha256.txt",
    "train_history.json",
    "train_summary.json",
    "dev_metrics.json",
    "dev_predictions.csv",
    "dev_confusion.csv",
    "dev_per_class_accuracy.csv",
)


class SelectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_forbidden_schema(value: Any, path: str = "ledger") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "test" in str(key).lower():
                raise SelectionError(f"Test evidence forbidden before selection lock at {path}.{key}")
            _reject_forbidden_schema(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_schema(child, f"{path}[{index}]")


def create_selection_ledger(
    ledger_path: Path, candidate_dir: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    _reject_forbidden_schema(provenance)
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    artifacts: dict[str, str] = {}
    for name in REQUIRED_CANDIDATE_FILES:
        path = candidate_dir / name
        if not path.is_file():
            raise SelectionError(f"candidate missing required artifact: {name}")
        artifacts[name] = sha256_file(path)
    payload = {
        **provenance,
        "selection_metric": "dev_macro_top1",
        "candidate_dir": str(candidate_dir.resolve()),
        "artifacts": artifacts,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_version": "knee42_selection_v1",
    }
    _reject_forbidden_schema(payload)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.chmod(ledger_path, 0o444)
    except OSError:
        pass
    return payload


def verify_selection(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.is_file():
        raise SelectionError(f"selection ledger does not exist: {ledger_path}")
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    _reject_forbidden_schema(payload)
    if payload.get("selection_metric") != "dev_macro_top1":
        raise SelectionError("selection metric is not Dev Macro Top-1")
    candidate_dir = Path(payload["candidate_dir"])
    for name in REQUIRED_CANDIDATE_FILES:
        expected = payload.get("artifacts", {}).get(name)
        path = candidate_dir / name
        if not expected or not path.is_file():
            raise SelectionError(f"selection artifact missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise SelectionError(
                f"SHA-256 mismatch for {name}: expected {expected}, actual {actual}"
            )
    return payload
