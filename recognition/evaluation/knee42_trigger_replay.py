from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, get_type_hints

import numpy as np

from recognition.realtime.knee42_integrity import (
    IntegrityError,
    VerifiedRelease,
    sha256_file,
    verify_release_root,
)


SCHEMA_VERSION = 1
FORMAL_AUTO_TRIGGER_CONFIG_NAME = "auto_trigger_knee_ivcam_local.json"
AUTO_TRIGGER_MODULE_NAME = "recognition.realtime.auto_trigger"
EXPECTED_SCENARIO_NAMES = (
    "nominal_single",
    "mid_sign_pause",
    "back_to_back",
    "expected_max_duration",
)
ROOT_KEYS = {
    "schema_version",
    "fixture_id",
    "frame_interval_sec",
    "config",
    "gates",
    "scenarios",
}
SCENARIO_KEYS = {"name", "duration_sec", "keyframes", "annotations"}
KEYFRAME_KEYS = {"at_sec", "recipe"}
ANNOTATION_KEYS = {
    "start_sec",
    "end_sec",
    "allowed_finalize_reasons",
    "allow_timeout",
}
RECIPE_NAMES = {"rest", "moving_sign_a", "moving_sign_b", "active_pause"}
FINALIZE_REASONS = {
    "visible_rest_finalize",
    "hidden_rest_finalize",
    "reference_rest_finalize",
    "timeout_finalize",
}
FIXED_GATES: dict[str, float | int] = {
    "segment_recall_min": 1.0,
    "premature_cut_count_max": 0,
    "merge_count_max": 0,
    "unexpected_timeout_count_max": 0,
    "mean_absolute_boundary_error_sec_max": 0.12,
}


@dataclass(frozen=True)
class TriggerRuntimeBinding:
    config_type: type
    engine_type: type
    config_field_types: Mapping[str, type]
    analyze_frame_vector: Any
    load_formal_config: Any


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _require_object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown: {', '.join(sorted(unknown))}")
        raise ValueError(f"{context} has invalid fields ({'; '.join(details)}).")


def _require_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array.")
    return value


def _require_number(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be at least {minimum}.")
    return result


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean.")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _validate_gates(value: object) -> dict[str, float | int]:
    gates = _require_object(value, "gates")
    _require_exact_keys(gates, set(FIXED_GATES), "gates")
    for key, fixed_value in FIXED_GATES.items():
        provided = _require_number(gates[key], f"gates.{key}", minimum=0.0)
        if provided != float(fixed_value):
            raise ValueError(
                f"The fixed gate {key} must be {fixed_value!r}, got {gates[key]!r}."
            )
    return dict(FIXED_GATES)


def _validate_config(value: object, runtime: TriggerRuntimeBinding) -> object:
    payload = _require_object(value, "config")
    config_keys = set(runtime.config_field_types)
    _require_exact_keys(payload, config_keys, "config")
    values: dict[str, float | bool] = {}
    for key, annotation in runtime.config_field_types.items():
        if annotation is float:
            values[key] = _require_number(payload[key], f"config.{key}")
        elif annotation is bool:
            values[key] = _require_bool(payload[key], f"config.{key}")
        else:
            raise ValueError(f"Unsupported auto-trigger config field annotation for {key}.")
    if float(values["start_motion_threshold"]) > float(values["blank_motion_threshold"]):
        raise ValueError(
            "config.start_motion_threshold must not exceed config.blank_motion_threshold."
        )
    return runtime.config_type(**values)


def _validate_annotation(value: object, context: str, duration_sec: float) -> dict[str, Any]:
    annotation = _require_object(value, context)
    _require_exact_keys(annotation, ANNOTATION_KEYS, context)
    start_sec = _require_number(annotation["start_sec"], f"{context}.start_sec", minimum=0.0)
    end_sec = _require_number(annotation["end_sec"], f"{context}.end_sec", minimum=0.0)
    if end_sec <= start_sec or end_sec > duration_sec + 1e-9:
        raise ValueError(f"{context} has an invalid start/end interval.")
    reasons_value = _require_list(
        annotation["allowed_finalize_reasons"],
        f"{context}.allowed_finalize_reasons",
    )
    if not reasons_value:
        raise ValueError(f"{context}.allowed_finalize_reasons must not be empty.")
    reasons: list[str] = []
    for reason_index, reason_value in enumerate(reasons_value):
        reason = _require_string(
            reason_value,
            f"{context}.allowed_finalize_reasons[{reason_index}]",
        )
        if reason not in FINALIZE_REASONS:
            raise ValueError(f"{context} has an unknown finalize reason: {reason}.")
        if reason in reasons:
            raise ValueError(f"{context} repeats finalize reason: {reason}.")
        reasons.append(reason)
    allow_timeout = _require_bool(annotation["allow_timeout"], f"{context}.allow_timeout")
    if ("timeout_finalize" in reasons) != allow_timeout:
        raise ValueError(
            f"{context} must explicitly align allow_timeout with timeout_finalize."
        )
    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "allowed_finalize_reasons": reasons,
        "allow_timeout": allow_timeout,
    }


def _validate_scenario(value: object, context: str) -> dict[str, Any]:
    scenario = _require_object(value, context)
    _require_exact_keys(scenario, SCENARIO_KEYS, context)
    name = _require_string(scenario["name"], f"{context}.name")
    duration_sec = _require_number(
        scenario["duration_sec"],
        f"{context}.duration_sec",
        minimum=0.0,
    )
    if duration_sec <= 0:
        raise ValueError(f"{context}.duration_sec must be positive.")
    keyframe_values = _require_list(scenario["keyframes"], f"{context}.keyframes")
    if not keyframe_values:
        raise ValueError(f"{context}.keyframes must not be empty.")
    keyframes: list[dict[str, float | str]] = []
    previous_at = -math.inf
    for keyframe_index, value in enumerate(keyframe_values):
        keyframe_context = f"{context}.keyframes[{keyframe_index}]"
        keyframe = _require_object(value, keyframe_context)
        _require_exact_keys(keyframe, KEYFRAME_KEYS, keyframe_context)
        at_sec = _require_number(
            keyframe["at_sec"],
            f"{keyframe_context}.at_sec",
            minimum=0.0,
        )
        if at_sec <= previous_at or at_sec > duration_sec:
            raise ValueError(f"{keyframe_context}.at_sec must be strictly increasing and in range.")
        recipe = _require_string(keyframe["recipe"], f"{keyframe_context}.recipe")
        if recipe not in RECIPE_NAMES:
            raise ValueError(f"{keyframe_context}.recipe is unknown: {recipe}.")
        keyframes.append({"at_sec": at_sec, "recipe": recipe})
        previous_at = at_sec
    if keyframes[0]["at_sec"] != 0.0:
        raise ValueError(f"{context}.keyframes must start at 0.0 seconds.")

    annotation_values = _require_list(scenario["annotations"], f"{context}.annotations")
    if not annotation_values:
        raise ValueError(f"{context}.annotations must not be empty.")
    annotations = [
        _validate_annotation(
            annotation,
            f"{context}.annotations[{annotation_index}]",
            duration_sec,
        )
        for annotation_index, annotation in enumerate(annotation_values)
    ]
    previous_end = -math.inf
    for annotation_index, annotation in enumerate(annotations):
        if annotation["start_sec"] < previous_end - 1e-9:
            raise ValueError(f"{context}.annotations[{annotation_index}] overlaps or is out of order.")
        previous_end = annotation["end_sec"]
    return {
        "name": name,
        "duration_sec": duration_sec,
        "keyframes": keyframes,
        "annotations": annotations,
    }


def _validate_fixture(
    value: object,
    runtime: TriggerRuntimeBinding,
) -> tuple[dict[str, Any], object]:
    fixture = _require_object(value, "fixture")
    _require_exact_keys(fixture, ROOT_KEYS, "fixture")
    schema_version = fixture["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("fixture.schema_version must be an integer.")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported fixture schema_version: {schema_version!r}.")
    fixture_id = _require_string(fixture["fixture_id"], "fixture.fixture_id")
    if fixture_id != "knee42_trigger_replay_v1":
        raise ValueError(f"Unsupported fixture_id: {fixture_id!r}.")
    frame_interval_sec = _require_number(
        fixture["frame_interval_sec"],
        "fixture.frame_interval_sec",
        minimum=0.0,
    )
    if frame_interval_sec <= 0:
        raise ValueError("fixture.frame_interval_sec must be positive.")
    config = _validate_config(fixture["config"], runtime)
    gates = _validate_gates(fixture["gates"])
    scenario_values = _require_list(fixture["scenarios"], "fixture.scenarios")
    scenarios = [
        _validate_scenario(scenario, f"fixture.scenarios[{index}]")
        for index, scenario in enumerate(scenario_values)
    ]
    names = tuple(scenario["name"] for scenario in scenarios)
    if names != EXPECTED_SCENARIO_NAMES:
        raise ValueError(
            "fixture.scenarios must contain the four required scenarios exactly and in order."
        )
    timeout_scenarios = [
        scenario["name"]
        for scenario in scenarios
        if any(annotation["allow_timeout"] for annotation in scenario["annotations"])
    ]
    if timeout_scenarios != ["expected_max_duration"]:
        raise ValueError("Only expected_max_duration may allow timeout finalization.")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "frame_interval_sec": frame_interval_sec,
        "config": {
            key: getattr(config, key)
            for key in sorted(runtime.config_field_types)
        },
        "gates": gates,
        "scenarios": scenarios,
    }
    return normalized, config


def load_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    payload = json.loads(
        fixture_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    return _require_object(payload, "fixture")


def _load_verified_trigger_runtime(
    verified_release: VerifiedRelease,
) -> tuple[TriggerRuntimeBinding, object, str]:
    if not isinstance(verified_release, VerifiedRelease):
        raise IntegrityError("verified_release must be a VerifiedRelease")
    expected_hash = verified_release.file_hashes.get(FORMAL_AUTO_TRIGGER_CONFIG_NAME)
    if expected_hash is None:
        raise IntegrityError(
            f"verified release manifest does not bind {FORMAL_AUTO_TRIGGER_CONFIG_NAME}"
        )
    config_path = verified_release.root / FORMAL_AUTO_TRIGGER_CONFIG_NAME
    before_hash = sha256_file(config_path)
    if before_hash != expected_hash:
        raise IntegrityError(
            f"verified formal config SHA-256 mismatch: expected {expected_hash}, "
            f"actual {before_hash}"
        )
    trigger_module = importlib.import_module(AUTO_TRIGGER_MODULE_NAME)
    config_type = trigger_module.AutoTriggerConfig
    resolved_types = get_type_hints(config_type)
    config_field_types = {
        field.name: resolved_types[field.name]
        for field in fields(config_type)
    }
    runtime = TriggerRuntimeBinding(
        config_type=config_type,
        engine_type=trigger_module.AutoTriggerEngine,
        config_field_types=config_field_types,
        analyze_frame_vector=trigger_module.analyze_frame_vector,
        load_formal_config=trigger_module.load_formal_auto_trigger_config,
    )
    config = runtime.load_formal_config(verified_release.root)
    after_hash = sha256_file(config_path)
    if after_hash != expected_hash or after_hash != before_hash:
        raise IntegrityError(
            f"verified formal config changed while loading: expected {expected_hash}, "
            f"actual {after_hash}"
        )
    return runtime, config, expected_hash


def _require_formal_config_match(
    fixture_config: Mapping[str, object],
    formal_config: object,
    runtime: TriggerRuntimeBinding,
) -> None:
    formal_values = formal_config.to_dict()
    mismatches = [
        name
        for name in runtime.config_field_types
        if type(fixture_config[name]) is not type(formal_values[name])
        or fixture_config[name] != formal_values[name]
    ]
    if mismatches:
        raise ValueError(
            "fixture formal config mismatch for field(s): "
            + ", ".join(sorted(mismatches))
        )


def _landmark_vector(recipe: str, frame_index: int) -> np.ndarray:
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11] = [0.40, 0.30, 0.0]
    pose[12] = [0.60, 0.30, 0.0]
    pose[13] = [0.36, 0.48, 0.0]
    pose[14] = [0.64, 0.48, 0.0]
    pose[23] = [0.44, 0.70, 0.0]
    pose[24] = [0.56, 0.70, 0.0]
    pose[25] = [0.36, 0.90, 0.0]
    pose[26] = [0.64, 0.90, 0.0]
    if recipe == "rest":
        left_wrist = np.asarray([0.36, 0.90, 0.0], dtype=np.float32)
        right_wrist = np.asarray([0.64, 0.90, 0.0], dtype=np.float32)
    elif recipe == "active_pause":
        left_wrist = np.asarray([0.30, 0.46, 0.0], dtype=np.float32)
        right_wrist = np.asarray([0.70, 0.46, 0.0], dtype=np.float32)
    else:
        direction = 1.0 if recipe == "moving_sign_a" else -1.0
        oscillation = direction if frame_index % 2 == 0 else -direction
        left_wrist = np.asarray(
            [0.30 + 0.08 * oscillation, 0.46 + 0.04 * oscillation, 0.0],
            dtype=np.float32,
        )
        right_wrist = np.asarray(
            [0.70 - 0.08 * oscillation, 0.46 - 0.04 * oscillation, 0.0],
            dtype=np.float32,
        )
    pose[15] = left_wrist
    pose[16] = right_wrist
    left_hand = np.repeat(left_wrist[None, :], 21, axis=0)
    right_hand = np.repeat(right_wrist[None, :], 21, axis=0)
    return np.concatenate(
        [pose.reshape(-1), left_hand.reshape(-1), right_hand.reshape(-1)]
    ).astype(np.float32, copy=False)


def materialize_scenario(
    scenario: Mapping[str, object],
    *,
    frame_interval_sec: float,
) -> list[tuple[float, np.ndarray]]:
    normalized = _validate_scenario(scenario, "scenario")
    interval = _require_number(frame_interval_sec, "frame_interval_sec", minimum=0.0)
    if interval <= 0:
        raise ValueError("frame_interval_sec must be positive.")
    duration_sec = float(normalized["duration_sec"])
    frame_count_float = duration_sec / interval
    frame_count = int(round(frame_count_float))
    if not math.isclose(frame_count_float, frame_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Scenario duration must be an exact multiple of frame_interval_sec.")
    keyframes = normalized["keyframes"]
    output: list[tuple[float, np.ndarray]] = []
    keyframe_index = 0
    for frame_index in range(frame_count + 1):
        timestamp_sec = round(frame_index * interval, 12)
        while (
            keyframe_index + 1 < len(keyframes)
            and float(keyframes[keyframe_index + 1]["at_sec"]) <= timestamp_sec + 1e-9
        ):
            keyframe_index += 1
        recipe = str(keyframes[keyframe_index]["recipe"])
        output.append((timestamp_sec, _landmark_vector(recipe, frame_index)))
    return output


def _overlap_sec(annotation: Mapping[str, object], prediction: Any) -> float:
    return max(
        0.0,
        min(float(annotation["end_sec"]), prediction.clip_end_sec)
        - max(float(annotation["start_sec"]), prediction.clip_start_sec),
    )


def _ordered_maximum_overlap_matching(
    annotations: Sequence[Mapping[str, object]],
    predictions: Sequence[Any],
) -> list[tuple[int, int]]:
    @lru_cache(maxsize=None)
    def solve(annotation_index: int, prediction_index: int) -> tuple[float, int, tuple[tuple[int, int], ...]]:
        if annotation_index >= len(annotations) or prediction_index >= len(predictions):
            return 0.0, 0, ()
        candidates = [
            solve(annotation_index + 1, prediction_index),
            solve(annotation_index, prediction_index + 1),
        ]
        overlap = _overlap_sec(annotations[annotation_index], predictions[prediction_index])
        if overlap > 0:
            future_overlap, future_count, future_pairs = solve(
                annotation_index + 1,
                prediction_index + 1,
            )
            candidates.append(
                (
                    overlap + future_overlap,
                    future_count + 1,
                    ((annotation_index, prediction_index),) + future_pairs,
                )
            )
        best = max(candidates, key=lambda item: (item[0], item[1], tuple(reversed(item[2]))))
        return best

    return list(solve(0, 0)[2])


def _prediction_row(segment: Any) -> dict[str, object]:
    return {
        "clip_start_sec": float(segment.clip_start_sec),
        "clip_end_sec": float(segment.clip_end_sec),
        "finalize_sec": float(segment.finalize_sec),
        "reason": segment.reason,
        "sample_count": len(segment.samples),
    }


def evaluate_fixture(
    value: object,
    *,
    verified_release: VerifiedRelease,
) -> dict[str, Any]:
    runtime, config, formal_config_sha256 = _load_verified_trigger_runtime(
        verified_release
    )
    fixture, _ = _validate_fixture(value, runtime)
    _require_formal_config_match(fixture["config"], config, runtime)
    frame_interval_sec = float(fixture["frame_interval_sec"])
    total_annotations = 0
    qualified_match_count = 0
    premature_cut_count = 0
    merge_count = 0
    unexpected_timeout_count = 0
    boundary_errors: list[float] = []
    scenario_results: list[dict[str, Any]] = []

    for scenario in fixture["scenarios"]:
        engine = runtime.engine_type(config)
        frames = materialize_scenario(scenario, frame_interval_sec=frame_interval_sec)
        previous: np.ndarray | None = None
        predictions: list[Any] = []
        nonzero_motion_frame_count = 0
        for timestamp_sec, vector in frames:
            analysis = runtime.analyze_frame_vector(previous, vector, config)
            if analysis.effective_motion_score > 0:
                nonzero_motion_frame_count += 1
            event = engine.update(vector, analysis, timestamp_sec)
            if event is not None:
                predictions.append(event)
            previous = vector

        annotations = scenario["annotations"]
        total_annotations += len(annotations)
        matches = _ordered_maximum_overlap_matching(annotations, predictions)
        matched_prediction_indices = {
            prediction_index for _, prediction_index in matches
        }
        for annotation_index, prediction_index in matches:
            annotation = annotations[annotation_index]
            prediction = predictions[prediction_index]
            reason_allowed = prediction.reason in annotation["allowed_finalize_reasons"]
            timeout_allowed = prediction.reason != "timeout_finalize" or annotation["allow_timeout"]
            if reason_allowed and timeout_allowed:
                qualified_match_count += 1
                boundary_errors.extend(
                    [
                        abs(prediction.clip_start_sec - float(annotation["start_sec"])),
                        abs(prediction.clip_end_sec - float(annotation["end_sec"])),
                    ]
                )
            if prediction.clip_end_sec < float(annotation["end_sec"]) - 0.12 - 1e-9:
                premature_cut_count += 1

        premature_cut_count += sum(
            prediction_index not in matched_prediction_indices
            for prediction_index in range(len(predictions))
        )
        for prediction in predictions:
            overlapping_annotations = [
                annotation
                for annotation in annotations
                if _overlap_sec(annotation, prediction) > 0
            ]
            if len(overlapping_annotations) >= 2:
                merge_count += 1
            if prediction.reason == "timeout_finalize":
                expected_timeout = bool(
                    len(overlapping_annotations) == 1
                    and overlapping_annotations[0]["allow_timeout"]
                    and "timeout_finalize"
                    in overlapping_annotations[0]["allowed_finalize_reasons"]
                )
                if not expected_timeout:
                    unexpected_timeout_count += 1

        scenario_results.append(
            {
                "name": scenario["name"],
                "analyzed_frame_count": len(frames),
                "nonzero_motion_frame_count": nonzero_motion_frame_count,
                "predictions": [_prediction_row(prediction) for prediction in predictions],
                "matches": [
                    {"annotation_index": annotation_index, "prediction_index": prediction_index}
                    for annotation_index, prediction_index in matches
                ],
            }
        )

    mean_boundary_error = (
        float(sum(boundary_errors) / len(boundary_errors)) if boundary_errors else None
    )
    metrics: dict[str, float | int | None] = {
        "segment_recall": (
            float(qualified_match_count / total_annotations) if total_annotations else 0.0
        ),
        "premature_cut_count": premature_cut_count,
        "merge_count": merge_count,
        "unexpected_timeout_count": unexpected_timeout_count,
        "mean_absolute_boundary_error_sec": mean_boundary_error,
    }
    gate_passed = bool(
        metrics["segment_recall"] == 1.0
        and metrics["premature_cut_count"] == 0
        and metrics["merge_count"] == 0
        and metrics["unexpected_timeout_count"] == 0
        and mean_boundary_error is not None
        and mean_boundary_error <= 0.12
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture["fixture_id"],
        "release_root_manifest_sha256": verified_release.root_manifest_sha256,
        "formal_config_sha256": formal_config_sha256,
        "gate_passed": gate_passed,
        "metrics": metrics,
        "scenario_results": scenario_results,
    }
    report.update(metrics)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the deterministic Knee42 trigger boundary gate."
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--root-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _write_report_atomically(output_path: Path, report: Mapping[str, object]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor_open = False
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if descriptor_open:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        verified_release = verify_release_root(
            Path(args.release_root),
            expected_root_manifest_sha256=args.root_manifest_sha256,
        )
        fixture = load_fixture(args.fixture)
        report = evaluate_fixture(
            fixture,
            verified_release=verified_release,
        )
        output_path = Path(args.output)
        _write_report_atomically(output_path, report)
        print(f"Trigger replay report: {output_path}")
        return 0 if report["gate_passed"] else 1
    except (IntegrityError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
