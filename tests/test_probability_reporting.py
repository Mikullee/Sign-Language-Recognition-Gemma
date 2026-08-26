from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import recognition.realtime.knee42_ivcam as knee42_ivcam
import recognition.realtime.realtime_infer_daily30_sentence as legacy_runtime
from recognition.evaluation.team_test_session import TeamTestSession


def load_probability_reporting(test_case: unittest.TestCase):
    module_name = "recognition.realtime.probability_reporting"
    test_case.assertIsNotNone(
        importlib.util.find_spec(module_name),
        f"missing shared probability policy module {module_name}",
    )
    return importlib.import_module(module_name)


class ProbabilityPolicyTests(unittest.TestCase):
    def test_policy_is_immutable_and_explicitly_disables_acceptance(self):
        probability_reporting = load_probability_reporting(self)

        policy = probability_reporting.PROBABILITY_POLICY

        self.assertTrue(dataclasses.is_dataclass(policy))
        self.assertEqual(policy.kind, "uncalibrated_softmax")
        self.assertEqual(
            policy.acceptance_policy,
            "disabled_no_risk_coverage_evidence",
        )
        self.assertIsNone(policy.calibration_artifact)
        self.assertEqual(
            dataclasses.asdict(policy),
            {
                "kind": "uncalibrated_softmax",
                "acceptance_policy": "disabled_no_risk_coverage_evidence",
                "calibration_artifact": None,
            },
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.kind = "calibrated"  # type: ignore[misc]

    def test_raw_probability_validator_rejects_invalid_values(self):
        probability_reporting = load_probability_reporting(self)

        for invalid in (-0.001, 1.001, float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite|between 0 and 1"):
                    probability_reporting.validate_raw_probability(invalid)
        with self.assertRaisesRegex(TypeError, "real number"):
            probability_reporting.validate_raw_probability(True)


class LegacyProbabilityReportingTests(unittest.TestCase):
    def test_legacy_calibration_compatibility_shim_is_exact_identity(self):
        for raw_probability in (0.0, 0.037, 0.1, 0.45, 1.0):
            with self.subTest(raw_probability=raw_probability):
                self.assertEqual(
                    legacy_runtime.calibrate_confidence(raw_probability),
                    raw_probability,
                )

    def test_legacy_calibration_shim_rejects_invalid_probabilities(self):
        for invalid in (-0.001, 1.001, float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite|between 0 and 1"):
                    legacy_runtime.calibrate_confidence(invalid)
        with self.assertRaisesRegex(TypeError, "real number"):
            legacy_runtime.calibrate_confidence(True)

    def test_legacy_decision_never_rejects_a_low_raw_probability(self):
        output = legacy_runtime.decide_prediction_output("你好", 0.001, 0.999)

        self.assertEqual(output, ("你好", 0.001, 0.001))

    def test_segment_prediction_fields_and_top3_are_raw_probabilities(self):
        prediction = legacy_runtime.decode_segment_prediction(
            np.asarray([0.45, 0.35, 0.20], dtype=np.float64),
            ["A", "B", "C"],
            {"A": "甲", "B": "乙", "C": "丙"},
            0.99,
        )

        field_names = {field.name for field in dataclasses.fields(prediction)}
        self.assertEqual(prediction.display_label, "甲")
        self.assertEqual(prediction.raw_probability, 0.45)
        self.assertEqual(
            prediction.top3_label_probabilities,
            [("A", 0.45), ("B", 0.35), ("C", 0.20)],
        )
        self.assertEqual(
            prediction.top3_display_probabilities,
            [("甲", 0.45), ("乙", 0.35), ("丙", 0.20)],
        )
        self.assertIn("raw_probability", field_names)
        self.assertNotIn("raw_confidence", field_names)
        self.assertNotIn("calibrated_confidence", field_names)
        self.assertNotIn("accepted", field_names)
        self.assertFalse(hasattr(prediction, "accepted"))

    def test_legacy_overlay_labels_and_formats_the_exact_raw_probability(self):
        drawn_text: list[str] = []

        def capture_text(image, text, *_args, **_kwargs):
            drawn_text.append(str(text))
            return image

        with mock.patch.object(legacy_runtime, "draw_text", side_effect=capture_text):
            legacy_runtime.draw_overlay(
                np.zeros((220, 800, 3), dtype=np.uint8),
                "甲",
                [("甲", 0.037), ("乙", 0.1)],
                [],
                30.0,
                "手動切段",
                "等待",
                "IDLE_BLANK",
                "manual",
            )

        candidate_line = next(text for text in drawn_text if "候選結果" in text)
        self.assertIn("raw probability", candidate_line.lower())
        self.assertIn("3.7%", candidate_line)
        self.assertIn("10.0%", candidate_line)
        self.assertNotIn("15.0%", candidate_line)

    def test_legacy_nonnegative_threshold_override_fails_at_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_config = Path(temp_dir) / "missing.json"
            for override in ("0", "0.5", "1"):
                with self.subTest(override=override):
                    parsed = legacy_runtime.parse_args(
                        [
                            "--app-config",
                            str(missing_config),
                            "--min-conf-override",
                            override,
                        ]
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "acceptance threshold is unavailable.*calibration/risk-coverage evidence",
                    ):
                        legacy_runtime.resolve_runtime_args(parsed)

    def test_legacy_default_neutralizes_the_deprecated_threshold_option(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = legacy_runtime.resolve_runtime_args(
                legacy_runtime.parse_args(
                    ["--app-config", str(Path(temp_dir) / "missing.json")]
                )
            )

        self.assertIsNone(args.min_conf_override)

    def test_legacy_executable_contains_no_probability_inflation_formula(self):
        source = inspect.getsource(legacy_runtime)

        self.assertNotRegex(source, r"\*\*\s*0\.7(?:0)?")
        self.assertNotRegex(source, r"np\.clip\([^\n]*0\.95")
        self.assertNotIn("calibrated confidence", source.lower())


class LegacyTeamRecordProbabilityTests(unittest.TestCase):
    def test_new_team_record_schema_uses_raw_probability_and_disabled_policy(self):
        parameters = inspect.signature(TeamTestSession.stage_prediction).parameters

        self.assertIn("raw_probability", parameters)
        self.assertNotIn("raw_confidence", parameters)
        self.assertNotIn("calibrated_confidence", parameters)

        with tempfile.TemporaryDirectory() as temp_dir:
            session = TeamTestSession(
                output_dir=Path(temp_dir),
                tester_id="tester",
                labels=["A"],
                label_display={"A": "甲"},
                trials_per_label=1,
                model_version="legacy",
            )
            record = session.stage_prediction(
                predicted_label="A",
                raw_probability=0.4,
                top3_candidates=[("A", 0.4)],
            )

        payload = dataclasses.asdict(record)
        self.assertEqual(payload["raw_probability"], 0.4)
        self.assertNotIn("raw_confidence", payload)
        self.assertNotIn("calibrated_confidence", payload)
        self.assertEqual(payload["top3_candidates"][0]["raw_probability"], 0.4)
        self.assertNotIn("confidence", payload["top3_candidates"][0])
        self.assertEqual(
            payload["probability_policy"]["acceptance_policy"],
            "disabled_no_risk_coverage_evidence",
        )

    def test_schema_one_team_progress_is_converted_to_raw_probability_on_load(self):
        legacy_record = {
            "tester_id": "tester",
            "timestamp": "2026-08-27T00:00:00",
            "global_trial_number": 1,
            "expected_label": "A",
            "expected_text": "甲",
            "trial_number": 1,
            "predicted_label": "A",
            "predicted_text": "甲",
            "outcome": "prediction",
            "top1_correct": True,
            "top3_hit": True,
            "raw_confidence": 0.4,
            "calibrated_confidence": 0.95,
            "top3_candidates": [{"label": "A", "text": "甲", "confidence": 0.4}],
            "clip_start_sec": None,
            "clip_end_sec": None,
            "finalize_sec": None,
            "segment_duration_sec": None,
            "finalize_delay_sec": None,
            "finalize_reason": "",
            "model_version": "legacy",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "team_tests" / "tester"
            session_dir.mkdir(parents=True)
            (session_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tester_id": "tester",
                        "labels": ["A"],
                        "label_display": {"A": "甲"},
                        "trials_per_label": 1,
                        "model_version": "legacy",
                        "runtime_metadata": {},
                        "records": [legacy_record],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            resumed = TeamTestSession(
                output_dir=Path(temp_dir),
                tester_id="tester",
                labels=["A"],
                label_display={"A": "甲"},
                trials_per_label=1,
                model_version="legacy",
                resume=True,
            )

        self.assertTrue(hasattr(resumed.records[0], "raw_probability"))
        self.assertEqual(resumed.records[0].raw_probability, 0.4)
        self.assertEqual(resumed.records[0].top3_candidates[0]["raw_probability"], 0.4)


class Knee42RawProbabilityTests(unittest.TestCase):
    def test_prediction_dataclass_names_raw_probability_with_deprecated_alias(self):
        field_names = {field.name for field in dataclasses.fields(knee42_ivcam.Prediction)}

        self.assertIn("raw_probability", field_names)
        self.assertNotIn("confidence", field_names)
        prediction = knee42_ivcam.Prediction(
            label_id="K42_01",
            display_text="你好",
            raw_probability=0.25,
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(prediction.confidence, prediction.raw_probability)

    def test_decode_logits_reports_exact_softmax_values_in_sorted_order(self):
        labels = [f"K42_{index:02d}" for index in range(1, 43)]
        logits = torch.tensor([[0.0, 3.0, 2.0] + [-1.0] * 39], dtype=torch.float32)
        expected = torch.softmax(logits[0], dim=0)

        result = knee42_ivcam.decode_logits(
            logits,
            labels,
            {label: label for label in labels},
        )

        expected_indices = torch.argsort(expected, descending=True)[:3].tolist()
        self.assertEqual(
            [item.label_id for item in result.top3],
            [labels[index] for index in expected_indices],
        )
        self.assertTrue(
            all(hasattr(item, "raw_probability") for item in result.top3),
            "decoded predictions must expose raw_probability",
        )
        for item, index in zip(result.top3, expected_indices):
            self.assertAlmostEqual(
                item.raw_probability,
                float(expected[index].item()),
                places=12,
            )
        self.assertAlmostEqual(
            sum(item.raw_probability for item in result.top3),
            float(expected[expected_indices].sum().item()),
            places=7,
        )
        self.assertAlmostEqual(float(expected.sum().item()), 1.0, places=6)

    def test_formal_overlay_and_structured_output_use_raw_probability_terms(self):
        prediction = knee42_ivcam.Prediction("K42_01", "你好", 0.037)
        result = knee42_ivcam.InferenceResult(
            top1=prediction,
            top3=(prediction, prediction, prediction),
        )

        rendered = "\n".join(
            knee42_ivcam.overlay_lines(
                result,
                fps=30.0,
                source="camera:0",
                mode="auto",
                state="RESULT",
            )
        )
        capture_source = inspect.getsource(knee42_ivcam.run_capture)

        self.assertIn("raw probability", rendered.lower())
        self.assertNotIn("confidence", rendered.lower())
        self.assertIn('"top1_raw_probability"', capture_source)
        self.assertNotIn('"top1_confidence"', capture_source)

    def test_formal_knee42_parser_has_no_acceptance_threshold_option(self):
        parser = knee42_ivcam.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--min-conf-override", option_strings)
        self.assertFalse(
            any("accept" in option or "confidence" in option for option in option_strings)
        )

    def test_decode_logits_has_no_acceptance_or_rejection_gate(self):
        source = inspect.getsource(knee42_ivcam.decode_logits).lower()

        for forbidden in ("threshold", "reject", "unknown", "accept"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
