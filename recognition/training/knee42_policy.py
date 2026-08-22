"""Fail-closed split policy for Knee42 Train/Dev research."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


TRAIN_SIGNERS = frozenset({"L", "P", "X"})
DEV_SIGNERS = frozenset({"H"})
TEST_SIGNERS = frozenset({"J"})


class LeakageError(ValueError):
    """Raised before a forbidden row can enter a research code path."""


def _identity(item: dict[str, str]) -> tuple[str, str]:
    return item.get("split", "").strip().lower(), item.get("signer_id", "").strip().upper()


def validate_research_rows(rows: Iterable[dict[str, str]]) -> None:
    for item in rows:
        split, signer = _identity(item)
        if split == "train" and signer in TRAIN_SIGNERS:
            continue
        if split == "dev" and signer in DEV_SIGNERS:
            continue
        raise LeakageError(f"forbidden research row split={split!r} signer={signer!r}")


def derive_research_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    research: list[dict[str, str]] = []
    for item in rows:
        split, signer = _identity(item)
        if split == "test" and signer in TEST_SIGNERS:
            continue
        if split == "train" and signer in TRAIN_SIGNERS:
            research.append(item)
            continue
        if split == "dev" and signer in DEV_SIGNERS:
            research.append(item)
            continue
        raise LeakageError(f"invalid frozen split row split={split!r} signer={signer!r}")
    validate_research_rows(research)
    return research


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_research_manifest(source: Path, destination: Path, ledger: Path) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(destination)
    if ledger.exists():
        raise FileExistsError(ledger)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError("source manifest has no header")
    research = derive_research_rows(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(research)
    result: dict[str, object] = {
        "source_manifest": str(source),
        "derived_manifest": str(destination),
        "source_sha256": sha256_file(source),
        "derived_sha256": sha256_file(destination),
        "counts": dict(sorted(Counter(item["split"].strip().lower() for item in research).items())),
        "allowed_train_signers": sorted(TRAIN_SIGNERS),
        "allowed_dev_signers": sorted(DEV_SIGNERS),
        "test_rows_written": 0,
    }
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result
