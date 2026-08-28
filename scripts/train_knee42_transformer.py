"""Train the Knee42 Transformer, either leave-one-signer-out or on all signers.

Reads ``research_manifest.csv`` and ``features_final/`` from ``--data-root``; no
video is touched.  Results are appended as one JSON object per line so a sweep
can be aggregated by ``scripts/aggregate_knee42_loso_runs.py``.

Examples
--------
    # one leave-one-signer-out run
    python scripts/train_knee42_transformer.py --data-root <features> \\
        --mode loso --test-signer H --seed 7

    # the full sweep behind the published table
    python scripts/train_knee42_transformer.py --data-root <features> \\
        --mode loso --test-signer H L P X --seed 7 42 2026

    # the release model: every signer, no held-out score
    python scripts/train_knee42_transformer.py --data-root <features> \\
        --mode final --seed 42 --save artifacts/knee42_transformer_final.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from recognition.training.knee42_transformer import (
    load_dataset,
    save_checkpoint,
    train_final,
    train_leave_one_signer_out,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("loso", "final"), default="loso")
    parser.add_argument("--test-signer", nargs="+", default=["H"])
    parser.add_argument("--seed", nargs="+", type=int, default=[7])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="MOC-pretrained encoder checkpoint; omit to train from scratch",
    )
    parser.add_argument("--results", type=Path, default=None, help="append JSONL results here")
    parser.add_argument("--save", type=Path, default=None, help="write the trained checkpoint here")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data_root)
    print(
        f"loaded {len(dataset)} samples, {len(dataset.label_ids)} classes, "
        f"signers {sorted(set(dataset.signers))}"
    )

    common = {
        "device": args.device,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "encoder_checkpoint": args.encoder,
    }
    runs: list[dict] = []
    for seed in args.seed:
        targets = args.test_signer if args.mode == "loso" else [None]
        for signer in targets:
            if args.mode == "loso":
                model, metrics = train_leave_one_signer_out(dataset, signer, seed, **common)
            else:
                model, metrics = train_final(dataset, seed, **common)
            runs.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))
            if args.results:
                args.results.parent.mkdir(parents=True, exist_ok=True)
                with args.results.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if args.save:
                destination = (
                    args.save
                    if len(runs) == 1
                    else args.save.with_name(f"{args.save.stem}_{signer or 'all'}_{seed}{args.save.suffix}")
                )
                save_checkpoint(model, dataset, destination, metrics)
                print(f"checkpoint written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
