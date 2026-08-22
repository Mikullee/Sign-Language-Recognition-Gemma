#!/usr/bin/env python3
"""Validate a single-factor Knee42 round and optionally train it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.training.knee42_devonly import DevOnlyConfig, train_dev_only  # noqa: E402
from recognition.training.knee42_rounds import validate_single_factor  # noqa: E402
from recognition.training.train_knee42_bigru import read_csv  # noqa: E402


REQUIRED_PLAN_FIELDS = frozenset(
    {
        "round",
        "parent_candidate",
        "problem",
        "hypothesis",
        "factor_group",
        "changes",
        "success_condition",
        "seed",
        "source_commit",
        "source_tree_sha256",
        "manifest_sha256",
        "split_sha256",
        "feature_ledger_sha256",
        "status",
    }
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one validated single-factor Knee42 round.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    missing = REQUIRED_PLAN_FIELDS - set(plan)
    if missing:
        raise SystemExit(f"round plan missing fields: {sorted(missing)}")
    baseline = DevOnlyConfig(**json.loads(args.baseline_config.read_text(encoding="utf-8")))
    candidate = DevOnlyConfig(**json.loads(args.candidate_config.read_text(encoding="utf-8")))
    changed = validate_single_factor(baseline, candidate, str(plan["factor_group"]))
    if set(plan["changes"]) != changed:
        raise SystemExit(f"round plan changes {sorted(plan['changes'])} do not match config {sorted(changed)}")
    if args.validate_only:
        print(json.dumps({"status": "VALID", "changed_fields": sorted(changed)}, indent=2))
        return
    required_args = {"manifest": args.manifest, "feature_dir": args.feature_dir, "out_dir": args.out_dir}
    absent = [name for name, value in required_args.items() if value is None]
    if absent:
        raise SystemExit(f"training requires arguments: {absent}")
    device_name = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    summary = train_dev_only(
        read_csv(args.manifest),
        manifest_hash=str(plan["manifest_sha256"]),
        split_hash=str(plan["split_sha256"]),
        feature_ledger_hash=str(plan["feature_ledger_sha256"]),
        feature_dir=args.feature_dir,
        out_dir=args.out_dir,
        seed=int(plan["seed"]),
        device=torch.device(device_name),
        config=candidate,
        max_train_batches=args.max_train_batches,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
