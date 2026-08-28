"""Bundle-verified inference for the Knee42 Transformer recognizer.

The bundle contract mirrors the legacy BiGRU one in
``recognition.realtime.knee42_ivcam``: every published file is pinned by SHA-256
in ``integrity_manifest.sha256`` and the declared feature contract must match the
constants this module was built against.  JSON files are hashed after CRLF is
normalized to LF so a Windows checkout verifies identically to a Linux one.

The Transformer bundle deliberately has no ``standardizer_train_only.npz``: this
path feeds shoulder-normalized coordinates to the model directly and derives its
velocity and acceleration channels at inference time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recognition.transformer.features import (
    LANDMARK_DIM,
    MODEL_INPUT_DIM,
    SEQUENCE_LENGTH,
    materialize_sequence,
)
from recognition.transformer.model import Knee42Transformer


LABELS = [f"K42_{number:02d}" for number in range(1, 43)]

REQUIRED_INTEGRITY_FILES = frozenset(
    {
        "best_model.pt",
        "feature_config.json",
        "runtime_config.json",
        "label_map_knee42.json",
        "display_text_map.json",
        "model_card.json",
    }
)

TEXT_HASHED_FILES = frozenset(
    {
        "feature_config.json",
        "runtime_config.json",
        "label_map_knee42.json",
        "display_text_map.json",
        "model_card.json",
    }
)


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_text_file(path: Path) -> str:
    """Hash locked text identically after an LF or a Windows CRLF checkout."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def bundle_digest(bundle_dir: Path, relative: str) -> str:
    path = Path(bundle_dir) / relative
    if relative in TEXT_HASHED_FILES:
        return sha256_canonical_text_file(path)
    return sha256_file(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"cannot read {Path(path).name}: {exc}") from exc


def verify_integrity_manifest(bundle_dir: Path) -> dict[str, str]:
    """Check every pinned file in the bundle and return the verified digests."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = bundle_dir / "integrity_manifest.sha256"
    if not manifest.is_file():
        raise IntegrityError(f"integrity manifest missing: {manifest}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise IntegrityError(f"invalid integrity entry at line {line_number}")
        digest, relative_text = parts[0].lower(), parts[1].strip().lstrip("*")
        if any(character not in "0123456789abcdef" for character in digest):
            raise IntegrityError(f"invalid SHA-256 at line {line_number}")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise IntegrityError(f"unsafe integrity path: {relative_text}")
        normalized = relative.as_posix()
        if normalized in expected:
            raise IntegrityError(f"duplicate integrity path: {normalized}")
        expected[normalized] = digest
    missing = sorted(REQUIRED_INTEGRITY_FILES - set(expected))
    if missing:
        raise IntegrityError(f"integrity manifest missing required files: {missing}")
    for relative, wanted in expected.items():
        path = bundle_dir / relative
        if not path.is_file():
            raise IntegrityError(f"integrity file missing: {relative}")
        actual = bundle_digest(bundle_dir, relative)
        if actual != wanted:
            raise IntegrityError(
                f"integrity SHA-256 mismatch for {relative}: expected {wanted}, actual {actual}"
            )
    return expected


@dataclass(frozen=True)
class Knee42TransformerBundle:
    root: Path
    labels: list[str]
    display_text: dict[str, str]
    feature_config: dict[str, Any]
    runtime_config: dict[str, Any]
    model_card: dict[str, Any]
    digests: dict[str, str]


def load_bundle(
    bundle_dir: Path,
    *,
    device: torch.device,
) -> tuple[Knee42TransformerBundle, Knee42Transformer]:
    """Verify a Transformer bundle and return it together with the loaded model."""
    bundle_dir = Path(bundle_dir).resolve()
    digests = verify_integrity_manifest(bundle_dir)

    feature = _read_json(bundle_dir / "feature_config.json")
    expected_feature = {
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
    }
    for key, expected_value in expected_feature.items():
        if feature.get(key) != expected_value:
            raise IntegrityError(
                f"feature contract mismatch for {key}: "
                f"expected {expected_value!r}, got {feature.get(key)!r}"
            )

    runtime = _read_json(bundle_dir / "runtime_config.json")
    expected_runtime = {
        "sequence_length": SEQUENCE_LENGTH,
        "landmark_value_dim": LANDMARK_DIM,
        "model_input_dim": MODEL_INPUT_DIM,
        "pose_landmarks_removed": [25, 26],
        "stream": "RGB/color",
    }
    for key, expected_value in expected_runtime.items():
        if runtime.get(key) != expected_value:
            raise IntegrityError(
                f"runtime contract mismatch for {key}: "
                f"expected {expected_value!r}, got {runtime.get(key)!r}"
            )

    label_payload = _read_json(bundle_dir / "label_map_knee42.json")
    label_to_idx = {
        str(key): int(value) for key, value in label_payload.get("label_to_idx", {}).items()
    }
    if label_to_idx != {label: index for index, label in enumerate(LABELS)}:
        raise IntegrityError("label map does not match the ordered Knee42 contract")

    display_text = {
        str(key): str(value)
        for key, value in _read_json(bundle_dir / "display_text_map.json").items()
    }
    if set(display_text) != set(LABELS) or any(not value.strip() for value in display_text.values()):
        raise IntegrityError("display text map is incomplete")

    model_card = _read_json(bundle_dir / "model_card.json")
    for key in ("model_id", "training_split", "reported_metrics", "known_limitations"):
        if key not in model_card:
            raise IntegrityError(f"model card is missing required field: {key}")

    try:
        checkpoint = torch.load(bundle_dir / "best_model.pt", map_location="cpu", weights_only=True)
        if int(checkpoint.get("n_classes", -1)) != len(LABELS):
            raise IntegrityError("checkpoint output dimension is not 42")
        if [str(item) for item in checkpoint.get("label_ids", [])] != LABELS:
            raise IntegrityError("checkpoint label order does not match the Knee42 contract")
        model = Knee42Transformer(len(LABELS))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"cannot load locked model: {exc}") from exc

    bundle = Knee42TransformerBundle(
        root=bundle_dir,
        labels=list(LABELS),
        display_text=display_text,
        feature_config=feature,
        runtime_config=runtime,
        model_card=model_card,
        digests=digests,
    )
    return bundle, model


class Knee42TransformerRecognizer:
    """Classify ``[frames, 219]`` sequences into the 42 Knee42 sentence classes.

    Input frames are the shared shoulder-normalized contract produced by
    ``recognition.realtime.knee42_preprocessing``; missing coordinates must be
    NaN rather than zero, so interpolation can tell them apart from a landmark
    genuinely observed at the origin.
    """

    def __init__(
        self,
        bundle_dir: Path | str,
        *,
        device: str = "cpu",
        threads: int | None = None,
    ) -> None:
        if threads is not None:
            torch.set_num_threads(int(threads))
        self.device = torch.device(device)
        self.bundle, self.model = load_bundle(Path(bundle_dir), device=self.device)
        self.labels = self.bundle.labels

    def display_text(self, label_id: str) -> str:
        return self.bundle.display_text.get(label_id, label_id)

    @torch.no_grad()
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        features = materialize_sequence(sequence)
        logits = self.model(torch.from_numpy(features[None]).to(self.device))
        return torch.softmax(logits, dim=1)[0].cpu().numpy()

    def predict(self, sequence: np.ndarray, topk: int = 5) -> list[tuple[str, str, float]]:
        return self._rank(self.predict_proba(sequence), topk)

    @torch.no_grad()
    def predict_batch(
        self,
        sequences: list[np.ndarray],
        topk: int = 5,
        batch_size: int = 64,
    ) -> list[list[tuple[str, str, float]]]:
        if not sequences:
            return []
        features = np.stack([materialize_sequence(sequence) for sequence in sequences])
        chunks = []
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(self.device)
            chunks.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        return [self._rank(row, topk) for row in np.concatenate(chunks)]

    def _rank(self, probabilities: np.ndarray, topk: int) -> list[tuple[str, str, float]]:
        order = np.argsort(-probabilities)[: max(int(topk), 1)]
        return [
            (
                self.labels[index],
                self.display_text(self.labels[index]),
                float(probabilities[index]),
            )
            for index in order
        ]
