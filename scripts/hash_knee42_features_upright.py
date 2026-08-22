#!/usr/bin/env python3
"""Create an immutable, provenance-complete ledger for upright Knee42 features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding=encoding, newline="") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature ledger directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    forbidden = [row for row in manifest if row["split"].lower() == "test" or row["signer_id"].upper() == "J"]
    if forbidden:
        raise ValueError(f"REFUSED Test/J rows: {len(forbidden)}")
    config = json.loads(args.feature_config.read_text(encoding="utf-8"))
    records = []
    for index, row in enumerate(sorted(manifest, key=lambda item: item["sample_id"]), 1):
        path = args.feature_dir / f"{row['sample_id']}.npz"
        with np.load(path, allow_pickle=False) as payload:
            record = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "signer_id": row["signer_id"],
                "label_id": row["label_id"],
                "source_sha256": str(payload["source_sha256"].item()),
                "feature_sha256": sha256_file(path),
                "cache_version": str(payload["cache_version"].item()),
                "schema_sha256": str(payload["schema_sha256"].item()),
                "extractor_commit": str(payload["extractor_commit"].item()),
                "hand_model_sha256": str(payload["hand_model_sha256"].item()),
                "pose_model_sha256": str(payload["pose_model_sha256"].item()),
                "rotation_metadata_degrees": int(payload["rotation_metadata_degrees"].item()),
                "horizontal_mirror": bool(payload["horizontal_mirror"].item()),
                "frames": int(payload["values"].shape[0]),
                "value_dim": int(payload["values"].shape[1]),
            }
        if record["source_sha256"] != row["sha256"]:
            raise ValueError(f"source SHA mismatch: {row['sample_id']}")
        expected = {
            "cache_version": config["cache_version"],
            "schema_sha256": config["schema_sha256"],
            "extractor_commit": config["extractor_commit"],
            "hand_model_sha256": config["hand_model_sha256"],
            "pose_model_sha256": config["pose_model_sha256"],
        }
        for field, wanted in expected.items():
            if record[field] != wanted:
                raise ValueError(f"{field} mismatch: {row['sample_id']}")
        if record["horizontal_mirror"] or record["rotation_metadata_degrees"] not in {0, 90, 180, 270}:
            raise ValueError(f"unapproved video transform: {row['sample_id']}")
        records.append(record)
        if index % 250 == 0:
            print(f"progress {index}/{len(manifest)}", flush=True)
    csv_path = args.out_dir / "feature_hashes.csv"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(f"{record['sample_id']}\t{record['feature_sha256']}\n".encode("ascii"))
    ledger = {
        "status": "LOCKED",
        "scope": "Train=L/P/X and Dev=H only; Test/J=0",
        "samples": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "rotation_counts": dict(Counter(str(record["rotation_metadata_degrees"]) for record in records)),
        "manifest_sha256": sha256_file(args.manifest),
        "feature_config_sha256": sha256_file(args.feature_config),
        "feature_hashes_csv_sha256": sha256_file(csv_path),
        "aggregate_feature_sha256": aggregate.hexdigest(),
        "extractor_commit": config["extractor_commit"],
        "cache_version": config["cache_version"],
        "schema_sha256": config["schema_sha256"],
        "hand_model_sha256": config["hand_model_sha256"],
        "pose_model_sha256": config["pose_model_sha256"],
        "horizontal_mirror": False,
    }
    write_exclusive(args.out_dir / "feature_ledger.json", json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    write_exclusive(args.out_dir / "feature_ledger_sha256.txt", sha256_file(args.out_dir / "feature_ledger.json") + "\n", encoding="ascii")
    print(json.dumps(ledger, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
