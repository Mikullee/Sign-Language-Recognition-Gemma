#!/usr/bin/env python3
"""Write a Train/Dev-only Knee42 diagnostic bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.training.knee42_diagnostics import write_diagnostics  # noqa: E402
from recognition.training.train_knee42_bigru import read_csv  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose a Knee42 candidate with Train/Dev evidence only.")
    parser.add_argument("--research-manifest", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = write_diagnostics(
        read_csv(args.research_manifest), args.feature_dir, args.candidate_dir, args.out_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
