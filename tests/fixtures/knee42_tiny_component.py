"""Generate a deterministic tiny Knee42 component for software-contract tests only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier


LABELS = [f"K42_{number:02d}" for number in range(1, 43)]
TINY_COMPONENT_FILES = frozenset(
    {
        "best_model.pt",
        "runtime_config.json",
        "feature_config.json",
        "standardizer_train_only.npz",
        "label_map_knee42.json",
        "display_text_map.json",
        "selection_ledger.json",
        "hand_landmarker.task",
        "pose_landmarker.task",
        "integrity_manifest.sha256",
        "component_manifest.json",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_bytes(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_standardizer(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in (
            ("mean.npy", np.zeros(219, dtype=np.float32)),
            ("std.npy", np.ones(219, dtype=np.float32)),
        ):
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_STORED
            item.external_attr = 0o600 << 16
            archive.writestr(item, _npy_bytes(array))


def _write_checkpoint(path: Path) -> None:
    model_config = {
        "input_dim": 438,
        "hidden_size": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "pooling": "mean_max",
        "num_classes": 42,
    }
    model = BiGRUSentenceClassifier(**model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc.bias.copy_(torch.arange(42, dtype=torch.float32) / np.float32(32.0))
    torch.save(
        {
            "checkpoint_version": "knee42_tiny_software_contract_v1",
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "seed": 0,
        },
        path,
    )


def generate_tiny_component(
    model_dir: Path,
    *,
    component_id: str = "knee42-v42-tiny-test",
    model_version: str = "v42",
) -> dict:
    """Write a sub-1MB component; task files are deliberate non-MediaPipe sentinels."""
    model_dir = Path(model_dir)
    if model_dir.exists():
        if model_dir.is_symlink() or not model_dir.is_dir():
            raise ValueError(f"tiny component output is not a real directory: {model_dir}")
        entries = tuple(model_dir.iterdir())
        unexpected = sorted(
            entry.name for entry in entries if entry.name not in TINY_COMPONENT_FILES
        )
        invalid_types = sorted(
            entry.name
            for entry in entries
            if entry.name in TINY_COMPONENT_FILES
            and (entry.is_symlink() or not entry.is_file())
        )
        if unexpected:
            raise ValueError(
                f"tiny component output contains unexpected path(s): {unexpected}"
            )
        if invalid_types:
            raise ValueError(
                "tiny component output contains non-regular allowlisted path(s): "
                f"{invalid_types}"
            )
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(model_dir / "best_model.pt")
    _write_deterministic_standardizer(model_dir / "standardizer_train_only.npz")
    _write_json(
        model_dir / "runtime_config.json",
        {
            "sequence_length": 64,
            "landmark_value_dim": 219,
            "model_input_dim": 438,
            "pose_landmarks_removed": [25, 26],
            "frame_step": 2,
            "stream": "RGB/color",
            "model_display_version": "untrusted-fixture-value",
        },
    )
    _write_json(
        model_dir / "feature_config.json",
        {
            "features_final": "knee42_features_upright_v2",
            "input_dim": 438,
            "sequence_length": 64,
            "standardizer": "observed_train_standardizer",
            "mask_concatenated": True,
            "knee_indices_removed": [25, 26],
            "video_orientation": "container_rotation_metadata_applied_explicitly",
            "horizontal_mirror": False,
        },
    )
    _write_json(
        model_dir / "label_map_knee42.json",
        {
            "idx_to_label": LABELS,
            "label_to_idx": {label: index for index, label in enumerate(LABELS)},
        },
    )
    _write_json(
        model_dir / "display_text_map.json",
        {label: f"Fixture {label}" for label in LABELS},
    )
    (model_dir / "hand_landmarker.task").write_bytes(
        b"TEST-SENTINEL-HAND-NOT-MEDIAPIPE\n"
    )
    (model_dir / "pose_landmarker.task").write_bytes(
        b"TEST-SENTINEL-POSE-NOT-MEDIAPIPE\n"
    )
    selection_names = (
        "best_model.pt",
        "feature_config.json",
        "standardizer_train_only.npz",
        "label_map_knee42.json",
        "display_text_map.json",
    )
    _write_json(
        model_dir / "selection_ledger.json",
        {
            "ledger_version": "knee42_selection_v1",
            "selection_metric": "dev_macro_top1",
            "artifacts": {name: _sha256(model_dir / name) for name in selection_names},
            "fixture_notice": "software contract only; not accuracy evidence",
        },
    )
    payload_names = (
        "best_model.pt",
        "runtime_config.json",
        "feature_config.json",
        "standardizer_train_only.npz",
        "label_map_knee42.json",
        "display_text_map.json",
        "selection_ledger.json",
        "hand_landmarker.task",
        "pose_landmarker.task",
    )
    payload_hashes = {name: _sha256(model_dir / name) for name in payload_names}
    (model_dir / "integrity_manifest.sha256").write_bytes(
        "".join(
            f"{payload_hashes[name]}  {name}\n"
            for name in sorted(payload_hashes)
        ).encode("ascii")
    )
    component = {
        "schema_version": 1,
        "component_id": component_id,
        "model_version": model_version,
        "label_count": 42,
        "input_shape": [1, 64, 438],
        "runtime_config_sha256": payload_hashes["runtime_config.json"],
        "selection_ledger_sha256": payload_hashes["selection_ledger.json"],
        "payload_sha256": payload_hashes,
    }
    _write_json(model_dir / "component_manifest.json", component)
    final_entries = tuple(model_dir.iterdir())
    actual = {entry.name for entry in final_entries}
    invalid_types = sorted(
        entry.name
        for entry in final_entries
        if entry.is_symlink() or not entry.is_file()
    )
    if actual != TINY_COMPONENT_FILES or invalid_types:
        raise RuntimeError(
            "tiny component generator produced a non-exact layout: "
            f"missing={sorted(TINY_COMPONENT_FILES - actual)}, "
            f"unexpected={sorted(actual - TINY_COMPONENT_FILES)}, "
            f"non_regular={invalid_types}"
        )
    return component


def generate_tiny_golden_contract(model_dir: Path, output_path: Path) -> dict:
    """Generate the root-level deterministic CPU golden for this tiny component."""
    from recognition.realtime.knee42_golden import (
        golden_contract_payload,
        materialize_golden_tensor,
    )
    from recognition.realtime.knee42_integrity import VerifiedRelease
    from recognition.realtime.knee42_ivcam import load_bundle

    model_dir = Path(model_dir).resolve()
    if model_dir.name != "model":
        raise ValueError("tiny golden generation requires a <release-root>/model directory")
    authenticated_files = {
        f"model/{path.name}": path.read_bytes()
        for path in model_dir.iterdir()
        if path.is_file()
    }
    file_hashes = {
        relative: hashlib.sha256(raw_bytes).hexdigest()
        for relative, raw_bytes in authenticated_files.items()
    }
    component = json.loads(
        authenticated_files["model/component_manifest.json"].decode("utf-8")
    )
    trusted_release = VerifiedRelease(
        root=model_dir.parent,
        release_version="v1.0.1-v13.1",
        app_version="v13.1",
        component_id=component["component_id"],
        model_version=component["model_version"],
        model_component_manifest_sha256=file_hashes[
            "model/component_manifest.json"
        ],
        label_count=42,
        input_shape=(1, 64, 438),
        source_commit="0" * 40,
        dependency_lock_sha256="0" * 64,
        root_manifest_sha256="0" * 64,
        file_hashes=file_hashes,
        authenticated_files=authenticated_files,
    )
    bundle = load_bundle(
        model_dir,
        device=torch.device("cpu"),
        trusted_release=trusted_release,
    )
    tensor = materialize_golden_tensor(bundle.mean, bundle.std)
    logits = bundle.forward_prepared(tensor)
    payload = golden_contract_payload(trusted_release, tensor, logits)
    _write_json(Path(output_path), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Knee42 software-contract fixture."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--golden-output",
        type=Path,
        help="optional root-level golden_contract.json output",
    )
    args = parser.parse_args()
    generate_tiny_component(args.output)
    if args.golden_output is not None:
        generate_tiny_golden_contract(args.output, args.golden_output)


if __name__ == "__main__":
    main()
