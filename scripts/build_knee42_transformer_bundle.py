"""Assemble a verifiable Knee42 Transformer runtime bundle.

The bundle is the only artifact ``recognition.transformer.recognizer`` will load,
and every file in it is pinned by SHA-256 in ``integrity_manifest.sha256``.  This
script is the single source of truth for how that bundle is produced, so a
reviewer can rebuild it from a checkpoint and confirm the published digests.

Usage
-----
    python scripts/build_knee42_transformer_bundle.py \
        --checkpoint path/to/knee42_final_v2.pt \
        --label-map path/to/label_map_knee42.json \
        --display-map path/to/display_text_map.json \
        --metrics path/to/loso_metrics.json \
        --out artifacts/realtime/best_current

``--metrics`` is the aggregated leave-one-signer-out table produced by
``scripts/aggregate_knee42_loso_runs.py``.  It is copied into the model card so
the published accuracy claims stay attached to the weights they describe.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from recognition.transformer.features import (
    LANDMARK_DIM,
    MODEL_INPUT_DIM,
    SEQUENCE_LENGTH,
)
from recognition.transformer.recognizer import (
    LABELS,
    REQUIRED_INTEGRITY_FILES,
    bundle_digest,
)


FEATURE_CONFIG = {
    "sequence_length": SEQUENCE_LENGTH,
    "landmark_value_dim": LANDMARK_DIM,
    "input_dim": MODEL_INPUT_DIM,
    "channels": ["position", "velocity", "acceleration"],
    "knee_indices_removed": [25, 26],
    "mask_concatenated": False,
    "standardizer": None,
    "missing_values": "linear_interpolation",
    "features_final": "knee42_features_upright_v2",
    "horizontal_mirror": False,
    "note": (
        "Velocity and acceleration are first-order differences taken after "
        "resampling to 64 frames, so they are per resampled frame."
    ),
}

RUNTIME_CONFIG = {
    "sequence_length": SEQUENCE_LENGTH,
    "landmark_value_dim": LANDMARK_DIM,
    "model_input_dim": MODEL_INPUT_DIM,
    "pose_landmarks_removed": [25, 26],
    "stream": "RGB/color",
    "frame_contract": "recognition.realtime.knee42_preprocessing",
    "device": "cpu",
}


def build_model_card(checkpoint: dict, metrics: dict | None) -> dict:
    """Record what the weights are, how they were split, and what that permits."""
    return {
        "model_id": "knee42-transformer-v12",
        "architecture": {
            "family": "Transformer encoder",
            "layers": 4,
            "model_dim": 256,
            "heads": 8,
            "pooling": "mean",
            "input_shape": [SEQUENCE_LENGTH, MODEL_INPUT_DIM],
            "num_classes": int(checkpoint.get("n_classes", len(LABELS))),
        },
        "training_split": {
            "signers_in_training_data": ["P", "H", "X", "L"],
            "validation": "random 12% of all samples, NOT held out by signer",
            "held_out_test_set": None,
            "checkpoint_reported_value": {
                "val_macro_mixed": checkpoint.get("val_macro_mixed"),
                "interpretation": (
                    "Optimistic and not usable as an accuracy claim: every signer "
                    "in the validation split also appears in the training split."
                ),
            },
            "upstream_note": checkpoint.get("note"),
        },
        "reported_metrics": {
            "shipped_checkpoint": {
                "held_out_score": None,
                "reason": (
                    "Released weights were retrained on all four signers after the "
                    "method was validated, so no signer remains held out. Numbers "
                    "below describe the method, measured on separate models."
                ),
            },
            "leave_one_signer_out": metrics or {},
        },
        "known_limitations": [
            "Signer X scores lowest in every leave-one-signer-out arm.",
            "42 fixed sentence classes only; unseen sentences cannot be recognized.",
            "Not a continuous sign language decoder and not safety-critical.",
            "The one-shot signer J budget was consumed by the legacy BiGRU model.",
        ],
    }


def write_json(path: Path, payload: object) -> None:
    """Write UTF-8 JSON with LF endings so the pinned digest is checkout-stable."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    path.write_bytes(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--display-map", required=True, type=Path)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if [str(item) for item in checkpoint.get("label_ids", [])] != LABELS:
        raise SystemExit("checkpoint label order does not match the Knee42 contract")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics else None

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, out / "best_model.pt")
    shutil.copy2(args.label_map, out / "label_map_knee42.json")
    shutil.copy2(args.display_map, out / "display_text_map.json")
    write_json(out / "feature_config.json", FEATURE_CONFIG)
    write_json(out / "runtime_config.json", RUNTIME_CONFIG)
    write_json(out / "model_card.json", build_model_card(checkpoint, metrics))

    lines = []
    for relative in sorted(REQUIRED_INTEGRITY_FILES):
        lines.append(f"{bundle_digest(out, relative)}  {relative}")
    (out / "integrity_manifest.sha256").write_bytes(
        ("\n".join(lines) + "\n").encode("ascii")
    )

    print(f"bundle written to {out}")
    for line in lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
