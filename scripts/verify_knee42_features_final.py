#!/usr/bin/env python3
"""Fail closed if features_final does not satisfy the frozen Knee42 contract."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_VALUE_DIM = 219  # (33 pose - knees 25/26) * xyz + two 21-point xyz hands
EXPECTED_CACHE_VERSION = "knee42_features_upright_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--feature-config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    config = json.loads(args.feature_config.read_text(encoding="utf-8"))
    forbidden = {"knee_coordinates", "knee_visibility", "knee_presence", "knee_masks", "knee_quality_status", "candidate_flag", "signer_missingness_pattern"}
    failures: list[dict[str, str]] = []
    if (
        config.get("cache_version") != EXPECTED_CACHE_VERSION
        or config.get("pose_landmarks_removed") != [25, 26]
        or forbidden - set(config.get("forbidden_inputs", []))
        or config.get("video_orientation") != "container_rotation_metadata_applied_explicitly"
        or config.get("horizontal_mirror") is not False
    ):
        failures.append({"sample_id": "FEATURE_CONFIG", "error": "feature config does not prove required knee exclusions"})
    rotations: Counter[int] = Counter()
    for row in rows:
        path = args.feature_dir / f"{row['sample_id']}.npz"
        try:
            with np.load(path, allow_pickle=False) as item:
                values, mask = item["values"], item["mask"]
                if str(item["cache_version"].item()) != EXPECTED_CACHE_VERSION:
                    raise ValueError("wrong cache version")
                if values.ndim != 2 or values.shape[1] != EXPECTED_VALUE_DIM or mask.shape != values.shape:
                    raise ValueError(f"expected [frames,{EXPECTED_VALUE_DIM}] values+mask")
                if np.any(np.isfinite(values) & ~mask):
                    raise ValueError("missing-value mask inconsistent")
                rotation = int(item["rotation_metadata_degrees"].item())
                if rotation not in {0, 90, 180, 270}:
                    raise ValueError("invalid rotation metadata")
                if bool(item["horizontal_mirror"].item()):
                    raise ValueError("unexpected horizontal mirror")
                expected_metadata = {
                    "source_sha256": row["sha256"],
                    "schema_sha256": config["schema_sha256"],
                    "extractor_commit": config["extractor_commit"],
                    "hand_model_sha256": config["hand_model_sha256"],
                    "pose_model_sha256": config["pose_model_sha256"],
                }
                for field, expected in expected_metadata.items():
                    if str(item[field].item()) != expected:
                        raise ValueError(f"{field} mismatch")
                rotations[rotation] += 1
        except Exception as exc:
            failures.append({"sample_id": row["sample_id"], "error": str(exc)})
    report = {
        "status": "PASS" if not failures else "FAIL",
        "expected_samples": len(rows),
        "verified_samples": len(rows) - len(failures),
        "value_dim": EXPECTED_VALUE_DIM,
        "knee_indices_removed": [25, 26],
        "rotation_counts": dict(sorted(rotations.items())),
        "horizontal_mirror": False,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit("features_final verification failed")


if __name__ == "__main__":
    main()
