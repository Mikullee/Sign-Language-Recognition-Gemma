#!/usr/bin/env python3
"""CLI for leakage-safe Knee42 Train/Dev candidate training."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.training.knee42_devonly import DevOnlyConfig, train_dev_only  # noqa: E402
from recognition.training.train_knee42_bigru import read_csv  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Knee42 candidate with Train/Dev data only.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--split-hash", required=True)
    parser.add_argument("--feature-ledger-hash", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args(argv)

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = DevOnlyConfig(**payload)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    device_name = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    summary = train_dev_only(
        read_csv(args.manifest),
        manifest_hash=args.manifest_hash,
        split_hash=args.split_hash,
        feature_ledger_hash=args.feature_ledger_hash,
        feature_dir=args.feature_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        device=torch.device(device_name),
        config=config,
        max_train_batches=args.max_train_batches,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
