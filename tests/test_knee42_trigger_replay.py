from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import numpy as np

import recognition.evaluation.knee42_trigger_replay as trigger_replay
from recognition.evaluation.knee42_trigger_replay import (
    evaluate_fixture,
    load_fixture,
    materialize_scenario,
)
from recognition.realtime.auto_trigger import AutoTriggerConfig
from recognition.realtime.knee42_integrity import IntegrityError, VerifiedRelease


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "knee42_trigger_replay.json"
FORMAL_CONFIG_PATH = (
    REPO_ROOT / "packaging" / "knee42_ivcam" / "auto_trigger_knee_ivcam_local.json"
)
FORMAL_CONFIG_NAME = "auto_trigger_knee_ivcam_local.json"
METRIC_KEYS = {
    "segment_recall",
    "premature_cut_count",
    "merge_count",
    "unexpected_timeout_count",
    "mean_absolute_boundary_error_sec",
}


class Knee42TriggerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_fixture(FIXTURE_PATH)
        self._release_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._release_temp.cleanup)
        self.release_root = Path(self._release_temp.name) / "verified-release"
        self.verified_release = self._make_verified_release(
            self.release_root,
            json.loads(FORMAL_CONFIG_PATH.read_text(encoding="utf-8")),
        )

    @staticmethod
    def _make_verified_release(
        release_root: Path,
        config: dict[str, object],
    ) -> VerifiedRelease:
        release_root.mkdir(parents=True, exist_ok=True)
        config_path = release_root / FORMAL_CONFIG_NAME
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        return VerifiedRelease(
            root=release_root.resolve(),
            release_version="v1.0.1-v13.1",
            app_version="v13.1",
            component_id="knee42-v11-replay-fixture",
            model_version="v11",
            model_component_manifest_sha256="d" * 64,
            label_count=42,
            input_shape=(1, 64, 438),
            source_commit="a" * 40,
            dependency_lock_sha256="b" * 64,
            root_manifest_sha256="c" * 64,
            file_hashes={FORMAL_CONFIG_NAME: config_hash},
        )

    def evaluate(
        self,
        fixture: object,
        *,
        verified_release: VerifiedRelease | None = None,
    ) -> dict[str, object]:
        return evaluate_fixture(
            fixture,
            verified_release=verified_release or self.verified_release,
        )

    def test_fixture_has_exact_required_scenarios_and_is_pii_free(self) -> None:
        self.assertEqual(
            [scenario["name"] for scenario in self.fixture["scenarios"]],
            [
                "nominal_single",
                "mid_sign_pause",
                "back_to_back",
                "expected_max_duration",
            ],
        )
        serialized = json.dumps(self.fixture, ensure_ascii=False)
        self.assertNotIn("C:" + chr(92) + "Users", serialized)
        self.assertNotIn("video_path", serialized)

    def test_materialization_produces_timestamped_finite_225_value_vectors(self) -> None:
        for scenario in self.fixture["scenarios"]:
            frames = materialize_scenario(
                scenario,
                frame_interval_sec=self.fixture["frame_interval_sec"],
            )
            self.assertGreater(len(frames), 1, scenario["name"])
            self.assertAlmostEqual(frames[0][0], 0.0)
            self.assertAlmostEqual(frames[-1][0], scenario["duration_sec"])
            previous_timestamp = -math.inf
            for timestamp_sec, vector in frames:
                self.assertGreater(timestamp_sec, previous_timestamp)
                self.assertEqual(vector.shape, (225,))
                self.assertEqual(vector.dtype, np.float32)
                self.assertTrue(np.isfinite(vector).all())
                previous_timestamp = timestamp_sec

    def test_baseline_replay_passes_all_fixed_metrics(self) -> None:
        report = self.evaluate(self.fixture)
        self.assertTrue(report["gate_passed"], report)
        self.assertEqual(set(report["metrics"]), METRIC_KEYS)
        self.assertEqual(report["metrics"]["segment_recall"], 1.0)
        self.assertEqual(report["metrics"]["premature_cut_count"], 0)
        self.assertEqual(report["metrics"]["merge_count"], 0)
        self.assertEqual(report["metrics"]["unexpected_timeout_count"], 0)
        self.assertLessEqual(
            report["metrics"]["mean_absolute_boundary_error_sec"],
            0.12,
        )

    def test_replay_reports_real_frame_analysis_and_allowed_finalize_reasons(self) -> None:
        report = self.evaluate(self.fixture)
        result_by_name = {item["name"]: item for item in report["scenario_results"]}
        for scenario in self.fixture["scenarios"]:
            expected_frames = materialize_scenario(
                scenario,
                frame_interval_sec=self.fixture["frame_interval_sec"],
            )
            result = result_by_name[scenario["name"]]
            self.assertEqual(result["analyzed_frame_count"], len(expected_frames))
            self.assertGreater(result["nonzero_motion_frame_count"], 0)
        timeout_result = result_by_name["expected_max_duration"]
        self.assertEqual(timeout_result["predictions"][0]["reason"], "timeout_finalize")
        for name in ("nominal_single", "mid_sign_pause", "back_to_back"):
            self.assertTrue(
                all(
                    prediction["reason"] == "reference_rest_finalize"
                    for prediction in result_by_name[name]["predictions"]
                )
            )

    def test_shorter_max_duration_mutation_worsens_at_least_one_metric(self) -> None:
        baseline = self.evaluate(self.fixture)
        mutated = copy.deepcopy(self.fixture)
        mutated["config"]["max_segment_sec"] = 0.9
        alternate_release = self._make_verified_release(
            Path(self._release_temp.name) / "short-max-release",
            mutated["config"],
        )
        report = self.evaluate(mutated, verified_release=alternate_release)
        self.assertFalse(report["gate_passed"])
        self.assertTrue(
            report["metrics"]["segment_recall"] < baseline["metrics"]["segment_recall"]
            or report["metrics"]["premature_cut_count"]
            > baseline["metrics"]["premature_cut_count"]
            or report["metrics"]["merge_count"] > baseline["metrics"]["merge_count"]
            or report["metrics"]["unexpected_timeout_count"]
            > baseline["metrics"]["unexpected_timeout_count"]
            or report["metrics"]["mean_absolute_boundary_error_sec"]
            > baseline["metrics"]["mean_absolute_boundary_error_sec"]
        )

    def test_inverted_motion_thresholds_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["config"]["start_motion_threshold"] = 0.03
        mutated["config"]["blank_motion_threshold"] = 0.02
        alternate_release = self._make_verified_release(
            Path(self._release_temp.name) / "inverted-release",
            mutated["config"],
        )
        with self.assertRaisesRegex(ValueError, "start.*blank|blank.*start"):
            self.evaluate(mutated, verified_release=alternate_release)

    def test_config_is_an_exact_typed_snapshot_of_every_runtime_field(self) -> None:
        runtime_fields = {field.name: field for field in fields(AutoTriggerConfig)}
        self.assertEqual(len(runtime_fields), 25)
        self.assertEqual(set(self.fixture["config"]), set(runtime_fields))
        for name, field in runtime_fields.items():
            missing = copy.deepcopy(self.fixture)
            del missing["config"][name]
            with self.subTest(field=name, mutation="missing"):
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.evaluate(missing)

            wrong_type = copy.deepcopy(self.fixture)
            wrong_type["config"][name] = 1 if field.type == "bool" else True
            with self.subTest(field=name, mutation="wrong_type"):
                with self.assertRaisesRegex(ValueError, "boolean|finite number"):
                    self.evaluate(wrong_type)

            if field.type == "float":
                nonfinite = copy.deepcopy(self.fixture)
                nonfinite["config"][name] = math.nan
                with self.subTest(field=name, mutation="nonfinite"):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.evaluate(nonfinite)

    def test_fixture_config_exactly_matches_verified_formal_config(self) -> None:
        formal_config = json.loads(FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(self.fixture["config"], formal_config)
        for name, original in formal_config.items():
            drifted = copy.deepcopy(self.fixture)
            drifted["config"][name] = (
                not original if isinstance(original, bool) else float(original) + 0.001
            )
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, "formal config.*mismatch|mismatch.*formal config"):
                    self.evaluate(drifted)

    def test_verified_formal_config_byte_tamper_fails_before_replay(self) -> None:
        config_path = self.release_root / FORMAL_CONFIG_NAME
        tampered = json.loads(config_path.read_text(encoding="utf-8"))
        tampered["end_hold_sec"] = 0.31
        config_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(IntegrityError, "SHA-256 mismatch"):
            self.evaluate(self.fixture)

    def test_unknown_missing_and_nonfinite_fixture_values_fail_closed(self) -> None:
        mutations = []
        unknown = copy.deepcopy(self.fixture)
        unknown["unexpected"] = True
        mutations.append(unknown)
        missing = copy.deepcopy(self.fixture)
        del missing["gates"]
        mutations.append(missing)
        boolean_number = copy.deepcopy(self.fixture)
        boolean_number["frame_interval_sec"] = True
        mutations.append(boolean_number)
        nonfinite = copy.deepcopy(self.fixture)
        nonfinite["scenarios"][0]["duration_sec"] = math.nan
        mutations.append(nonfinite)
        nonmonotonic = copy.deepcopy(self.fixture)
        nonmonotonic["scenarios"][0]["keyframes"][1]["at_sec"] = 0.0
        mutations.append(nonmonotonic)
        unknown_recipe = copy.deepcopy(self.fixture)
        unknown_recipe["scenarios"][0]["keyframes"][0]["recipe"] = "private_video"
        mutations.append(unknown_recipe)
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                with self.assertRaises(ValueError):
                    self.evaluate(mutated)

    def test_annotation_schema_and_timeout_permission_fail_closed(self) -> None:
        unknown_annotation = copy.deepcopy(self.fixture)
        unknown_annotation["scenarios"][0]["annotations"][0]["notes"] = "not allowed"
        with self.assertRaises(ValueError):
            self.evaluate(unknown_annotation)

        missing_reason = copy.deepcopy(self.fixture)
        del missing_reason["scenarios"][0]["annotations"][0]["allowed_finalize_reasons"]
        with self.assertRaises(ValueError):
            self.evaluate(missing_reason)

        timeout_not_explicit = copy.deepcopy(self.fixture)
        timeout_not_explicit["scenarios"][0]["annotations"][0][
            "allowed_finalize_reasons"
        ] = ["timeout_finalize"]
        with self.assertRaises(ValueError):
            self.evaluate(timeout_not_explicit)

    def test_gate_thresholds_are_fixed_and_cannot_be_relaxed_by_fixture(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        mutated["gates"]["mean_absolute_boundary_error_sec_max"] = 99.0
        with self.assertRaisesRegex(ValueError, "fixed gate"):
            self.evaluate(mutated)

    def test_zero_overlap_real_prediction_counts_as_an_extra_cut(self) -> None:
        mutated = copy.deepcopy(self.fixture)
        annotation = mutated["scenarios"][0]["annotations"][0]
        annotation["start_sec"] = 2.5
        annotation["end_sec"] = 2.6
        report = self.evaluate(mutated)
        nominal = report["scenario_results"][0]
        self.assertEqual(len(nominal["predictions"]), 1)
        self.assertEqual(nominal["matches"], [])
        self.assertFalse(report["gate_passed"])
        self.assertGreater(report["metrics"]["premature_cut_count"], 0)

    def test_cli_writes_strict_json_report_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "replay-report.json"
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    trigger_replay,
                    "verify_release_root",
                    return_value=self.verified_release,
                    create=True,
                ) as verify_release,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = trigger_replay.main(
                    [
                        "--fixture",
                        str(FIXTURE_PATH),
                        "--release-root",
                        str(self.release_root),
                        "--root-manifest-sha256",
                        "c" * 64,
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            verify_release.assert_called_once_with(
                self.release_root,
                expected_root_manifest_sha256="c" * 64,
            )
            report = json.loads(
                output_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            self.assertTrue(report["gate_passed"])
            self.assertEqual(set(report["metrics"]), METRIC_KEYS)
            self.assertIn(str(output_path), stdout.getvalue())

    def test_cli_invalid_fixture_exits_nonzero_without_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "invalid.json"
            output_path = Path(temp_dir) / "must-not-exist.json"
            payload = copy.deepcopy(self.fixture)
            payload["unexpected"] = "rejected"
            invalid_path.write_text(json.dumps(payload), encoding="utf-8")
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    trigger_replay,
                    "verify_release_root",
                    return_value=self.verified_release,
                    create=True,
                ),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = trigger_replay.main(
                    [
                        "--fixture",
                        str(invalid_path),
                        "--release-root",
                        str(self.release_root),
                        "--root-manifest-sha256",
                        "c" * 64,
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertNotEqual(exit_code, 0)
            self.assertFalse(output_path.exists())
            self.assertIn("error", stderr.getvalue().lower())

    def test_cli_process_rejects_a_nonexistent_unverified_release_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "must-not-exist.json"
            missing_root = Path(temp_dir) / "missing-release"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "recognition.evaluation.knee42_trigger_replay",
                    "--fixture",
                    str(FIXTURE_PATH),
                    "--release-root",
                    str(missing_root),
                    "--root-manifest-sha256",
                    "c" * 64,
                    "--output",
                    str(output_path),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("release root", result.stderr.lower())
            self.assertFalse(output_path.exists())

    def test_cli_verifies_release_anchor_before_importing_trigger_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sentinel_path = temp_root / "trigger-imported.txt"
            module_status_path = temp_root / "trigger-module-status.txt"
            sitecustomize_path = temp_root / "sitecustomize.py"
            sitecustomize_path.write_text(
                textwrap.dedent(
                    """
                    import atexit
                    import importlib.abc
                    import importlib.machinery
                    import os
                    import pathlib
                    import sys

                    TARGET = "recognition.realtime.auto_trigger"
                    SENTINEL = pathlib.Path(os.environ["KNEE42_TRIGGER_SENTINEL"])
                    STATUS = pathlib.Path(os.environ["KNEE42_TRIGGER_MODULE_STATUS"])

                    class GuardLoader(importlib.abc.Loader):
                        def __init__(self, wrapped):
                            self.wrapped = wrapped

                        def create_module(self, spec):
                            create = getattr(self.wrapped, "create_module", None)
                            return None if create is None else create(spec)

                        def exec_module(self, module):
                            SENTINEL.write_text("executed\\n", encoding="utf-8")
                            self.wrapped.exec_module(module)

                    class GuardFinder(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path, target=None):
                            if fullname != TARGET:
                                return None
                            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                            if spec is None or spec.loader is None:
                                return spec
                            spec.loader = GuardLoader(spec.loader)
                            return spec

                    sys.meta_path.insert(0, GuardFinder())
                    atexit.register(
                        lambda: STATUS.write_text(
                            "1" if TARGET in sys.modules else "0",
                            encoding="ascii",
                        )
                    )
                    """
                ),
                encoding="utf-8",
            )
            output_path = temp_root / "must-not-exist.json"
            missing_root = temp_root / "missing-release"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(temp_root), str(REPO_ROOT), environment.get("PYTHONPATH", "")]
            )
            environment["KNEE42_TRIGGER_SENTINEL"] = str(sentinel_path)
            environment["KNEE42_TRIGGER_MODULE_STATUS"] = str(module_status_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "recognition.evaluation.knee42_trigger_replay",
                    "--fixture",
                    str(FIXTURE_PATH),
                    "--release-root",
                    str(missing_root),
                    "--root-manifest-sha256",
                    "c" * 64,
                    "--output",
                    str(output_path),
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("release root", result.stderr.lower())
            self.assertFalse(sentinel_path.exists())
            self.assertTrue(module_status_path.exists(), result.stderr)
            self.assertEqual(module_status_path.read_text(encoding="ascii"), "0")
            self.assertFalse(output_path.exists())

    def test_atomic_report_write_failure_preserves_existing_destination(self) -> None:
        self._assert_atomic_report_failure_preserves_existing("write")

    def test_atomic_report_replace_failure_preserves_existing_destination(self) -> None:
        self._assert_atomic_report_failure_preserves_existing("replace")

    def _assert_atomic_report_failure_preserves_existing(self, failure: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "replay-report.json"
            output_path.write_text("existing-report\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(
                trigger_replay,
                "verify_release_root",
                return_value=self.verified_release,
                create=True,
            ):
                if failure == "write":
                    injected = mock.patch.object(
                        trigger_replay.json,
                        "dump",
                        side_effect=OSError("injected write failure"),
                    )
                else:
                    injected = mock.patch.object(
                        trigger_replay,
                        "os",
                        wraps=os,
                        create=True,
                    )
                with injected as injected_target:
                    if failure == "replace":
                        injected_target.replace.side_effect = OSError(
                            "injected replace failure"
                        )
                    with (
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        exit_code = trigger_replay.main(
                            [
                                "--fixture",
                                str(FIXTURE_PATH),
                                "--release-root",
                                str(self.release_root),
                                "--root-manifest-sha256",
                                "c" * 64,
                                "--output",
                                str(output_path),
                            ]
                        )
            self.assertEqual(exit_code, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing-report\n")
            self.assertEqual(list(output_path.parent.glob(f".{output_path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
