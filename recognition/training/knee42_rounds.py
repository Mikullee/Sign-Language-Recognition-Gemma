"""Single-factor experiment and Dev-only ranking policy for Knee42."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from recognition.training.knee42_devonly import DevOnlyConfig
from recognition.training.knee42_policy import LeakageError


FACTOR_FIELDS = {
    "sampling": frozenset({"sampler"}),
    "loss": frozenset({"loss", "label_smoothing", "focal_gamma"}),
    "augmentation": frozenset(
        {
            "augmentation",
            "coordinate_scale_jitter",
            "coordinate_translation_jitter",
            "landmark_dropout_probability",
        }
    ),
    "temporal": frozenset({"sequence_length"}),
    "normalization": frozenset({"normalization"}),
    "features": frozenset({"feature_mode"}),
    "architecture": frozenset({"hidden_size", "num_layers", "dropout", "pooling"}),
    "optimization": frozenset(
        {"batch_size", "epochs", "patience", "learning_rate", "weight_decay"}
    ),
}


def validate_single_factor(
    baseline: DevOnlyConfig, candidate: DevOnlyConfig, declared_group: str
) -> set[str]:
    if declared_group not in FACTOR_FIELDS:
        raise ValueError(f"unknown factor group: {declared_group}")
    before = asdict(baseline)
    after = asdict(candidate)
    changed = {name for name in before if before[name] != after[name]}
    if not changed:
        raise ValueError("no configuration change")
    changed_groups = {
        group for group, fields in FACTOR_FIELDS.items() if changed.intersection(fields)
    }
    if changed_groups != {declared_group} or not changed.issubset(FACTOR_FIELDS[declared_group]):
        raise ValueError(
            f"round must change one factor group; declared={declared_group!r} "
            f"changed_groups={sorted(changed_groups)} changed_fields={sorted(changed)}"
        )
    return changed


def _reject_forbidden_schema(value: Any, path: str = "candidate") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "test" in str(key).lower():
                raise LeakageError(f"Test metric forbidden in Dev ranking at {path}.{key}")
            _reject_forbidden_schema(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_schema(child, f"{path}[{index}]")


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in candidates:
        _reject_forbidden_schema(item)
        for field in ("dev_macro_top1", "dev_macro_std", "weak_class_score"):
            if field not in item:
                raise ValueError(f"candidate missing {field}")
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["dev_macro_top1"]),
            float(item["dev_macro_std"]),
            -float(item["weak_class_score"]),
        ),
    )
