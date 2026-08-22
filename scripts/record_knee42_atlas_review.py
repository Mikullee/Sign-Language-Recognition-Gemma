#!/usr/bin/env python3
"""Append evidence-backed visual decisions for grouped Knee42 contact-sheet atlases."""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--atlas", nargs="+", required=True)
    parser.add_argument("--status", choices=("PASS", "REVIEW", "FAIL"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    template = read_csv(args.template)
    expected = {row["atlas_path"] for row in template}
    requested = set(args.atlas)
    unknown = requested - expected
    if unknown:
        raise ValueError(f"unknown atlas paths: {sorted(unknown)}")
    existing = read_csv(args.ledger) if args.ledger.is_file() else []
    reviewed = {row["atlas_path"] for row in existing}
    duplicate = requested & reviewed
    if duplicate:
        raise FileExistsError(f"refusing to replace existing atlas reviews: {sorted(duplicate)}")
    timestamp = datetime.now(timezone.utc).isoformat()
    additions = [
        {
            "atlas_path": path,
            "status": args.status,
            "reviewer": args.reviewer,
            "reason": args.reason,
            "reviewed_at_utc": timestamp,
        }
        for path in sorted(requested)
    ]
    rows = sorted(existing + additions, key=lambda row: row["atlas_path"])
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=args.ledger.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.ledger)
    print(f"recorded={len(additions)} reviewed_total={len(rows)} expected_total={len(expected)}")


if __name__ == "__main__":
    main()
