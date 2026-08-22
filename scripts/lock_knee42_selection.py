#!/usr/bin/env python3
"""Create the immutable Knee42 Dev selection ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.evaluation.knee42_selection import create_selection_ledger  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lock a Knee42 candidate selected only by Dev evidence.")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    result = create_selection_ledger(args.out, args.candidate, provenance)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
