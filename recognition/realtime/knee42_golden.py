"""Deterministic software-contract golden input and checksum verification."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recognition.realtime.knee42_integrity import (
    IntegrityError,
    VerifiedRelease,
    authenticated_release_bytes,
    parse_json_object_bytes,
    require_exact_fields,
    require_sha256,
)
from recognition.realtime.knee42_preprocessing import (
    LANDMARK_DIM,
    POSE_KEEP,
    materialize_sequence,
    normalize_frame,
)


GOLDEN_RECIPE = "knee42_asymmetric_missing_mask_v1"
GOLDEN_PURPOSE = "software_contract_not_accuracy_evidence"
GOLDEN_FIELDS = {
    "schema_version",
    "purpose",
    "recipe",
    "component_id",
    "model_version",
    "model_sha256",
    "input_shape",
    "tensor_dtype",
    "tensor_shape",
    "tensor_sha256",
    "logits_dtype",
    "logits_shape",
    "logits_sha256",
}


@dataclass(frozen=True)
class GoldenContract:
    source_sha256: str
    purpose: str
    recipe: str
    component_id: str
    model_version: str
    model_sha256: str
    input_shape: tuple[int, int, int]
    tensor_dtype: str
    tensor_shape: tuple[int, int]
    tensor_sha256: str
    logits_dtype: str
    logits_shape: tuple[int, int]
    logits_sha256: str


def _float32_array(value: Any, *, description: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32:
            raise IntegrityError(
                f"{description} dtype must be float32, got {value.dtype}"
            )
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
        if array.dtype.kind != "f" or array.dtype.itemsize != 4:
            raise IntegrityError(
                f"{description} dtype must be float32, got {array.dtype}"
            )
    array = np.asarray(array, dtype="<f4")
    if not np.isfinite(array).all():
        raise IntegrityError(f"{description} contains a non-finite value")
    return np.ascontiguousarray(array)


def float32_sha256(value: Any, *, description: str) -> str:
    """Hash finite little-endian float32 C-order bytes exactly."""
    array = _float32_array(value, description=description)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def materialize_golden_tensor(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Exercise normalization, asymmetric landmarks, missing masks, and sampling."""
    frame_count = 7
    coordinates = np.arange(LANDMARK_DIM, dtype=np.float32)
    values = np.stack(
        [
            ((coordinates % 19.0) - 9.0) * np.float32(0.017)
            + np.float32(frame_index * 0.013)
            for frame_index in range(frame_count)
        ]
    ).astype(np.float32)
    mask = np.ones(values.shape, dtype=np.bool_)
    pose_index = {source: target for target, source in enumerate(POSE_KEEP)}
    left_shoulder = pose_index[11] * 3
    right_shoulder = pose_index[12] * 3
    for frame_index in range(frame_count):
        values[frame_index, left_shoulder : left_shoulder + 3] = (
            -0.42 + frame_index * 0.009,
            0.10 + frame_index * 0.003,
            -0.04,
        )
        values[frame_index, right_shoulder : right_shoulder + 3] = (
            0.58 + frame_index * 0.004,
            0.14 - frame_index * 0.002,
            0.06,
        )
    left_hand_start = len(POSE_KEEP) * 3
    right_hand_start = left_hand_start + 21 * 3
    missing_slices = (
        (1, left_hand_start + 3, left_hand_start + 12),
        (2, right_hand_start + 18, right_hand_start + 30),
        (4, left_hand_start + 39, left_hand_start + 48),
        (5, right_hand_start + 3, right_hand_start + 15),
    )
    for frame_index, start, end in missing_slices:
        mask[frame_index, start:end] = False
        values[frame_index, start:end] = np.nan
    normalized = np.stack(
        [normalize_frame(frame, frame_mask) for frame, frame_mask in zip(values, mask)]
    ).astype(np.float32)
    tensor = materialize_sequence(
        normalized,
        mask,
        mean,
        std,
        sequence_length=64,
    )
    if tensor.shape != (64, 438):
        raise IntegrityError(f"golden tensor shape mismatch: {tensor.shape}")
    return _float32_array(tensor, description="golden tensor")


def golden_contract_payload(
    trusted_release: VerifiedRelease,
    tensor: Any,
    logits: Any,
) -> dict[str, Any]:
    if not isinstance(trusted_release, VerifiedRelease):
        raise IntegrityError("trusted release must be a VerifiedRelease")
    tensor_array = _float32_array(tensor, description="golden tensor")
    logits_array = _float32_array(logits, description="golden logits")
    if tensor_array.shape != (64, 438):
        raise IntegrityError(f"golden tensor shape mismatch: {tensor_array.shape}")
    if logits_array.shape != (1, 42):
        raise IntegrityError(f"golden logits shape mismatch: {logits_array.shape}")
    model_hash = trusted_release.file_hashes.get("model/best_model.pt")
    if model_hash is None:
        raise IntegrityError("trusted release missing model/best_model.pt")
    return {
        "schema_version": 1,
        "purpose": GOLDEN_PURPOSE,
        "recipe": GOLDEN_RECIPE,
        "component_id": trusted_release.component_id,
        "model_version": trusted_release.model_version,
        "model_sha256": model_hash,
        "input_shape": list(trusted_release.input_shape),
        "tensor_dtype": "<f4",
        "tensor_shape": [64, 438],
        "tensor_sha256": float32_sha256(tensor_array, description="golden tensor"),
        "logits_dtype": "<f4",
        "logits_shape": [1, 42],
        "logits_sha256": float32_sha256(logits_array, description="golden logits"),
    }


def _exact_integer_shape(
    value: Any,
    expected: tuple[int, ...],
    *,
    description: str,
) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != len(expected)
        or any(type(item) is not int for item in value)
        or tuple(value) != expected
    ):
        raise IntegrityError(
            f"{description} mismatch: expected {list(expected)}, got {value!r}"
        )
    return expected


def load_golden_contract(
    path: Path,
    *,
    trusted_release: VerifiedRelease,
) -> GoldenContract:
    """Authenticate golden bytes, then validate exact schema and component identity."""
    if not isinstance(trusted_release, VerifiedRelease):
        raise IntegrityError("trusted release must be a VerifiedRelease")
    contract_path = Path(path).resolve()
    expected_path = trusted_release.root / "golden_contract.json"
    if contract_path != expected_path:
        raise IntegrityError(
            f"golden contract root mismatch: expected {expected_path}, got {contract_path}"
        )
    raw_bytes = authenticated_release_bytes(
        trusted_release,
        "golden_contract.json",
        description="golden contract",
    )
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    payload = parse_json_object_bytes(raw_bytes, description="golden contract")
    require_exact_fields(payload, GOLDEN_FIELDS, description="golden contract")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise IntegrityError("golden contract schema_version must be integer 1")
    if payload["purpose"] != GOLDEN_PURPOSE:
        raise IntegrityError(f"golden contract purpose must be {GOLDEN_PURPOSE}")
    if payload["recipe"] != GOLDEN_RECIPE:
        raise IntegrityError(f"golden contract recipe must be {GOLDEN_RECIPE}")
    if payload["component_id"] != trusted_release.component_id:
        raise IntegrityError("golden contract component_id mismatch")
    if payload["model_version"] != trusted_release.model_version:
        raise IntegrityError("golden contract model_version mismatch")
    model_hash = require_sha256(
        payload["model_sha256"], description="golden contract model_sha256"
    )
    if model_hash != trusted_release.file_hashes.get("model/best_model.pt"):
        raise IntegrityError("golden contract model_sha256 mismatch")
    input_shape = _exact_integer_shape(
        payload["input_shape"],
        trusted_release.input_shape,
        description="golden contract input_shape",
    )
    tensor_shape = _exact_integer_shape(
        payload["tensor_shape"],
        (64, 438),
        description="golden contract tensor_shape",
    )
    logits_shape = _exact_integer_shape(
        payload["logits_shape"],
        (1, 42),
        description="golden contract logits_shape",
    )
    if payload["tensor_dtype"] != "<f4":
        raise IntegrityError("golden contract tensor_dtype must be <f4")
    if payload["logits_dtype"] != "<f4":
        raise IntegrityError("golden contract logits_dtype must be <f4")
    tensor_hash = require_sha256(
        payload["tensor_sha256"], description="golden contract tensor_sha256"
    )
    logits_hash = require_sha256(
        payload["logits_sha256"], description="golden contract logits_sha256"
    )
    return GoldenContract(
        source_sha256=actual_hash,
        purpose=GOLDEN_PURPOSE,
        recipe=GOLDEN_RECIPE,
        component_id=trusted_release.component_id,
        model_version=trusted_release.model_version,
        model_sha256=model_hash,
        input_shape=input_shape,
        tensor_dtype="<f4",
        tensor_shape=tensor_shape,
        tensor_sha256=tensor_hash,
        logits_dtype="<f4",
        logits_shape=logits_shape,
        logits_sha256=logits_hash,
    )


def verify_golden_result(
    contract: GoldenContract,
    tensor: Any,
    logits: Any,
) -> None:
    tensor_array = _float32_array(tensor, description="golden tensor")
    logits_array = _float32_array(logits, description="golden logits")
    if tensor_array.shape != contract.tensor_shape:
        raise IntegrityError(
            f"golden tensor shape mismatch: expected {contract.tensor_shape}, "
            f"got {tensor_array.shape}"
        )
    if logits_array.shape != contract.logits_shape:
        raise IntegrityError(
            f"golden logits shape mismatch: expected {contract.logits_shape}, "
            f"got {logits_array.shape}"
        )
    tensor_hash = float32_sha256(tensor_array, description="golden tensor")
    if tensor_hash != contract.tensor_sha256:
        raise IntegrityError(
            "golden tensor SHA-256 mismatch: "
            f"expected {contract.tensor_sha256}, actual {tensor_hash}"
        )
    logits_hash = float32_sha256(logits_array, description="golden logits")
    if logits_hash != contract.logits_sha256:
        raise IntegrityError(
            "golden logits SHA-256 mismatch: "
            f"expected {contract.logits_sha256}, actual {logits_hash}"
        )
