"""Aggregate the leave-one-signer-out training logs into a citable metrics table.

Every accuracy figure published for the Transformer path is derived here, from
the raw ``runs/*.jsonl`` written during training, so a reviewer can recompute the
README numbers instead of taking them on trust.

Each JSONL row is one (arm, test signer, seed) run and carries the test scores of
a model trained with that signer fully excluded.  The released weights are a
*different* model, retrained on all four signers; this table therefore describes
the method, not the shipped checkpoint.  See ``model_card.json``.

Usage
-----
    python scripts/aggregate_knee42_loso_runs.py --runs path/to/runs \
        --out docs/evaluation/knee42_loso_metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ARMS = {
    "transslr": ("transslr_results.jsonl", "no pretraining baseline"),
    "pretrained_ft": ("pretrained_ft_v2_results.jsonl", "MOC pretraining then finetuning"),
    "proto": ("proto_results.jsonl", "prototype classification head"),
    "mirror_ab": ("mirror_ab_results.jsonl", "horizontal mirror augmentation A/B"),
    "mcc_v2": ("mcc_v2_results.jsonl", "per-signer personalization finetune"),
}

METRIC = "macro_top1"


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    """Group runs by test signer (and mirror probability, where present)."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        signer = str(row.get("test_signer"))
        probability = row.get("p")
        key = signer if probability is None else f"{signer}@p={probability}"
        grouped.setdefault(key, []).append(float(row["test"][METRIC]))

    per_group = {}
    for key, values in sorted(grouped.items()):
        per_group[key] = {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "seeds": len(values),
            "values": [round(value, 4) for value in values],
        }
    everything = [value for values in grouped.values() for value in values]
    return {
        "metric": METRIC,
        "runs": len(rows),
        "per_test_signer": per_group,
        "overall_mean": round(statistics.mean(everything), 4) if everything else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", required=True, type=Path, help="directory holding the *_results.jsonl logs")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    table: dict[str, object] = {
        "protocol": "leave-one-signer-out; the test signer is excluded from training",
        "metric": METRIC,
        "applies_to": "the method, measured on per-arm models -- not the released checkpoint",
        "arms": {},
    }
    for arm, (filename, description) in ARMS.items():
        rows = read_rows(args.runs / filename)
        if not rows:
            print(f"skipped {arm}: {filename} not found")
            continue
        summary = summarize(rows)
        summary["description"] = description
        summary["source_log"] = filename
        table["arms"][arm] = summary
        print(f"{arm:16s} n={summary['runs']:3d}  overall {summary['overall_mean']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(table, ensure_ascii=False, indent=2) + "\n"
    args.out.write_bytes(text.encode("utf-8"))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
