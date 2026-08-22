#!/usr/bin/env python3
"""CLI reserved for the sealed one-time Knee42 J evaluator."""
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

from recognition.evaluation.knee42_test_once import evaluate_j_once


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked Knee42 model on J exactly once.")
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--expected-count", type=int, default=626)
    args = parser.parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    summary = evaluate_j_once(
        args.ledger,
        args.manifest,
        args.feature_dir,
        args.out_dir,
        device=device,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
