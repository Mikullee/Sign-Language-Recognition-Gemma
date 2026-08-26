from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import recognition.realtime.knee42_ivcam as knee42_ivcam

from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier
from recognition.realtime.knee42_ivcam import (
    InferenceResult,
    IntegrityError,
    Prediction,
    auto_display_state,
    build_parser,
    decode_logits,
    load_bundle,
    overlay_lines,
    run_self_test,
    verify_auto_trigger_provenance,
)
from recognition.training.train_knee42_bigru import LABELS


class UnsafeCheckpointPayload:
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trigger_runtime(root: Path) -> None:
    source = root / "recognition" / "realtime" / "auto_trigger.py"
    source.parent.mkdir(parents=True)
    source.write_text("archived trigger", encoding="utf-8")
    controller = root / "recognition" / "realtime" / "knee42_controllers.py"
    controller.write_text("archived controller", encoding="utf-8")
    config = root / "auto_trigger_knee_ivcam_local.json"
    config.write_text("{}", encoding="utf-8")
    write_json(
        root / "auto_trigger_provenance.json",
        {
            "auto_trigger_source_sha256": sha256(source),
            "auto_trigger_controller_sha256": sha256(controller),
            "auto_trigger_config_sha256": sha256(config),
        },
    )


def make_bundle(root: Path) -> Path:
    write_trigger_runtime(root)
    bundle = root / "model"
    bundle.mkdir()
    label_to_idx = {label: index for index, label in enumerate(LABELS)}
    model_config = {
        "input_dim": 438,
        "hidden_size": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "pooling": "mean_max",
        "num_classes": 42,
    }
    model = BiGRUSentenceClassifier(**model_config)
    torch.save(
        {
            "checkpoint_version": "knee42_devonly_v1",
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "seed": 44,
        },
        bundle / "best_model.pt",
    )
    np.savez_compressed(
        bundle / "standardizer_train_only.npz",
        mean=np.zeros(219, dtype=np.float32),
        std=np.ones(219, dtype=np.float32),
    )
    write_json(
        bundle / "runtime_config.json",
        {
            "sequence_length": 64,
            "landmark_value_dim": 219,
            "model_input_dim": 438,
            "pose_landmarks_removed": [25, 26],
            "frame_step": 2,
            "stream": "RGB/color",
        },
    )
    write_json(
        bundle / "feature_config.json",
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
    write_json(
        bundle / "label_map_knee42.json",
        {"label_to_idx": label_to_idx, "idx_to_label": LABELS},
    )
    write_json(bundle / "display_text_map.json", {label: f"Gesture {label}" for label in LABELS})
    (bundle / "hand_landmarker.task").write_bytes(b"fake-hand-model")
    (bundle / "pose_landmarker.task").write_bytes(b"fake-pose-model")
    bound_names = (
        "best_model.pt",
        "standardizer_train_only.npz",
        "feature_config.json",
        "label_map_knee42.json",
        "display_text_map.json",
    )
    write_json(
        bundle / "selection_ledger.json",
        {
            "ledger_version": "knee42_selection_v1",
            "selection_metric": "dev_macro_top1",
            "artifacts": {name: sha256(bundle / name) for name in bound_names},
        },
    )
    names = (
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
    (bundle / "integrity_manifest.sha256").write_text(
        "".join(f"{sha256(bundle / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    return bundle


class FakeDetectorContext:
    entered = 0

    def __enter__(self):
        type(self).entered += 1
        return self

    def __exit__(self, *_args):
        return False


class RuntimeTests(unittest.TestCase):
    def test_bundle_rejects_non_weight_checkpoint_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = make_bundle(Path(temp_dir))
            model_path = bundle / "best_model.pt"
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            checkpoint["untrusted_payload"] = UnsafeCheckpointPayload()
            torch.save(checkpoint, model_path)

            ledger_path = bundle / "selection_ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["artifacts"]["best_model.pt"] = sha256(model_path)
            write_json(ledger_path, ledger)

            names = []
            for line in (bundle / "integrity_manifest.sha256").read_text(
                encoding="ascii"
            ).splitlines():
                _digest, name = line.split(maxsplit=1)
                names.append(name)
            (bundle / "integrity_manifest.sha256").write_text(
                "".join(f"{sha256(bundle / name)}  {name}\n" for name in names),
                encoding="ascii",
            )

            with self.assertRaisesRegex(IntegrityError, "Weights only load failed"):
                load_bundle(bundle, device=torch.device("cpu"))

    def test_structured_display_data_keeps_e2_predictions_and_model_version(self):
        builder = getattr(knee42_ivcam, "build_display_panel_data", None)
        self.assertTrue(callable(builder))
        result = InferenceResult(
            top1=Prediction("K42_12", "可以", 0.268),
            top3=(
                Prediction("K42_12", "可以", 0.268),
                Prediction("K42_09", "我聽不懂", 0.061),
                Prediction("K42_03", "晚安", 0.060),
            ),
        )

        data = builder(
            result,
            fps=12.6,
            source="camera:0",
            mode="auto",
            state="WAITING",
            recording=False,
            recorded_segments=0,
            model_version="v11",
        )

        self.assertEqual(data.top1.display_text, "可以")
        self.assertEqual([item.label_id for item in data.top3], ["K42_12", "K42_09", "K42_03"])
        self.assertEqual(data.model_version, "v11")
        self.assertEqual(data.state, "WAITING")

    def test_runtime_detects_raw_frame_before_display_only_ui_composition(self):
        from recognition.realtime.knee42_ivcam import run_capture

        source = inspect.getsource(run_capture)

        detect_at = source.index("detectors.extract_observation(frame)")
        compose_at = source.index("render_application_view(")
        self.assertLess(detect_at, compose_at)
        self.assertNotIn("detectors.extract_observation(view", source)
        self.assertNotIn("detectors.extract_observation(display", source)

    def test_runtime_uses_packet_timestamps_without_nominal_fps_synthesis(self):
        from recognition.realtime.knee42_ivcam import run_capture

        source = inspect.getsource(run_capture)

        self.assertIn("source.read_packet()", source)
        self.assertIn("packet.timestamp_sec", source)
        self.assertNotIn("(raw_frames - 1) / source_fps", source)
        self.assertNotIn("controller.add_held_observation(", source)
        self.assertNotIn("frame_interval_sec", source)

    def test_runtime_buffers_exact_packet_times_for_held_observations(self):
        held_calls = []
        recorder_origins = []

        class PacketOnlySource:
            status = "video:irregular.mp4"
            fps = 240.0
            clock_mode = "video_source_timestamp"

            def __init__(self):
                frame = np.zeros((4, 6, 3), dtype=np.uint8)
                self._packets = [
                    SimpleNamespace(frame=frame.copy(), timestamp_sec=timestamp, clock_mode="video_source_timestamp")
                    for timestamp in (0.0, 0.033, 0.071)
                ]
                self.released = False

            def read_packet(self):
                return self._packets.pop(0) if self._packets else None

            def release(self):
                self.released = True

        class FakeBundle:
            def __init__(self, root):
                self.root = root
                self.sequence_length = 64
                self.frame_step = 2
                self.model_display_version = "test"

            def predict(self, _values, _mask):
                predictions = (
                    Prediction("K42_01", "你好", 0.8),
                    Prediction("K42_02", "謝謝", 0.1),
                    Prediction("K42_03", "再見", 0.1),
                )
                return InferenceResult(top1=predictions[0], top3=predictions)

        class FakeAutoController:
            def __init__(self, _config, *, initial_mode):
                self.mode = initial_mode
                self.state = "WAITING"
                self.calibrated = True

            def add_held_observation_at_times(self, timestamps_sec, _trigger, feature):
                held_calls.append((tuple(timestamps_sec), float(feature[0][0])))
                return SimpleNamespace(
                    infer=len(held_calls) == 2,
                    features=(feature,),
                    segment=None,
                )

            def finalize_video_eof(self):
                return SimpleNamespace(infer=False, features=(), segment=None)

            def reset(self):
                return None

        class FakeDetectors:
            def __init__(self):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_observation(self, _frame):
                detector_value = float(self.calls)
                self.calls += 1
                return SimpleNamespace(
                    display_pose=None,
                    display_left_hand=None,
                    display_right_hand=None,
                    recognition_values=np.full(219, detector_value, dtype=np.float32),
                    recognition_mask=np.ones(219, dtype=np.bool_),
                    trigger_values=np.ones(225, dtype=np.float32),
                )

        class FakeRecorder:
            segment_count = 0

            def __init__(self, _output_root, **kwargs):
                recorder_origins.append(kwargs["source_origin_sec"])
                self.frame_count = 0

            def add_frame(self, _frame):
                self.frame_count += 1

            def record_segment(self, _segment, _result):
                return None

            def stop(self):
                return SimpleNamespace(
                    session_dir=Path("recording"),
                    segment_count=0,
                    frame_count=self.frame_count,
                )

        packet_source = PacketOnlySource()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text("{}", encoding="utf-8")
            bundle = FakeBundle(bundle_root)

            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(knee42_ivcam, "load_bundle", return_value=bundle),
                mock.patch.object(knee42_ivcam, "open_video", return_value=packet_source),
                mock.patch.object(knee42_ivcam, "load_auto_trigger_config", return_value=object()),
                mock.patch.object(knee42_ivcam, "AutoKnee42Controller", FakeAutoController),
                mock.patch.object(knee42_ivcam, "create_mediapipe_detectors", return_value=FakeDetectors()),
                mock.patch.object(knee42_ivcam, "SegmentSessionRecorder", FakeRecorder),
            ):
                result = knee42_ivcam.run_capture(
                    bundle_root,
                    mode="auto",
                    camera_index=None,
                    video=Path("irregular.mp4"),
                    device=torch.device("cpu"),
                    headless=True,
                    max_frames=None,
                    start_logging=True,
                )

        self.assertEqual(
            held_calls,
            [
                ((0.0, 0.033), 0.0),
                ((0.071,), 1.0),
            ],
        )
        self.assertEqual(recorder_origins, [0.0])
        self.assertEqual(result["raw_frames"], 3)
        self.assertEqual(result["feature_frames"], 2)
        self.assertEqual(result["clock_mode"], "video_source_timestamp")
        self.assertTrue(packet_source.released)

    def test_auto_trigger_provenance_rejects_source_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_trigger_runtime(root)
            source = root / "recognition" / "realtime" / "auto_trigger.py"
            verify_auto_trigger_provenance(root)
            source.write_text("tampered", encoding="utf-8")

            with self.assertRaisesRegex(IntegrityError, "auto-trigger source"):
                verify_auto_trigger_provenance(root)

    def test_auto_trigger_provenance_rejects_controller_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_trigger_runtime(root)
            verify_auto_trigger_provenance(root)
            controller = root / "recognition" / "realtime" / "knee42_controllers.py"
            controller.write_text("tampered", encoding="utf-8")

            with self.assertRaisesRegex(IntegrityError, "auto-trigger controller"):
                verify_auto_trigger_provenance(root)

    def test_tampered_standardizer_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = make_bundle(Path(temp_dir))
            (bundle / "standardizer_train_only.npz").write_bytes(b"tampered")

            with mock.patch("recognition.realtime.knee42_ivcam.torch.load") as load:
                with self.assertRaisesRegex(IntegrityError, "standardizer_train_only"):
                    load_bundle(bundle, device=torch.device("cpu"))

            load.assert_not_called()

    def test_load_bundle_validates_contract_and_real_forward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = make_bundle(Path(temp_dir))

            bundle = load_bundle(bundle_path, device=torch.device("cpu"))
            result = bundle.predict(
                np.zeros((5, 219), dtype=np.float32),
                np.ones((5, 219), dtype=np.bool_),
            )

            self.assertEqual(bundle.sequence_length, 64)
            self.assertEqual(result.top1.label_id in LABELS, True)
            self.assertEqual(len(result.top3), 3)

    def test_decode_returns_sorted_top1_top3_and_confidence(self):
        logits = torch.tensor([[0.0, 3.0, 2.0] + [-1.0] * 39])

        result = decode_logits(logits, LABELS, {label: label for label in LABELS})

        self.assertEqual(result.top1.label_id, "K42_02")
        self.assertEqual([item.label_id for item in result.top3], ["K42_02", "K42_03", "K42_01"])
        self.assertGreater(result.top1.confidence, result.top3[1].confidence)

    def test_overlay_includes_predictions_fps_source_mode_and_state(self):
        logits = torch.tensor([[0.0, 3.0, 2.0] + [-1.0] * 39])
        result = decode_logits(logits, LABELS, {label: f"Text {label}" for label in LABELS})

        lines = overlay_lines(
            result,
            fps=27.4,
            source="camera:2",
            mode="manual",
            state="result",
        )
        rendered = "\n".join(lines)

        self.assertIn("Top-1", rendered)
        self.assertIn("Top-3", rendered)
        self.assertIn("confidence", rendered.lower())
        self.assertIn("FPS 27.4", rendered)
        self.assertIn("camera:2", rendered)
        self.assertIn("manual", rendered)
        self.assertIn("result", rendered)

    def test_overlay_and_cli_expose_segment_recording_controls(self):
        lines = overlay_lines(
            None,
            fps=30.0,
            source="camera:0",
            mode="auto",
            state="WAITING",
            recording=True,
            recorded_segments=3,
            recording_session="Knee42-session-20260818_220000",
        )
        rendered = "\n".join(lines)
        self.assertIn("REC ON", rendered)
        self.assertIn("segments 3", rendered)
        self.assertIn("S start/stop audit", rendered)
        self.assertIn("F fullscreen", rendered)

        args = build_parser().parse_args(
            ["--bundle", "model", "--start-logging", "--recordings-dir", "evidence"]
        )
        self.assertTrue(args.start_logging)
        self.assertEqual(args.recordings_dir, Path("evidence"))

    def test_cli_defaults_to_auto_and_keeps_manual_and_sliding_choices(self):
        parser = build_parser()

        automatic = parser.parse_args(["--bundle", "model"])
        manual = parser.parse_args(["--bundle", "model", "--mode", "manual"])
        sliding = parser.parse_args(["--bundle", "model", "--mode", "sliding"])

        self.assertEqual(automatic.mode, "auto")
        self.assertEqual(manual.mode, "manual")
        self.assertEqual(sliding.mode, "sliding")

    def test_auto_states_are_localized_for_operator_feedback(self):
        self.assertEqual(auto_display_state("IDLE_BLANK", calibrated=False), "CALIBRATING")
        self.assertEqual(auto_display_state("IDLE_BLANK", calibrated=True), "WAITING")
        self.assertEqual(auto_display_state("SIGNING_ACTIVE", calibrated=True), "SIGNING")
        self.assertEqual(auto_display_state("END_CONFIRM", calibrated=True), "END_CONFIRM")
        self.assertEqual(auto_display_state("FORCED_FINALIZE_COOLDOWN", calibrated=True), "COOLDOWN")

    def test_overlay_compacts_long_video_source_without_discarding_label_text(self):
        result = InferenceResult(
            top1=Prediction("K42_16", "我肚子餓", 0.153),
            top3=(
                Prediction("K42_16", "我肚子餓", 0.153),
                Prediction("K42_08", "請慢一點", 0.082),
                Prediction("K42_42", "我是桃園人", 0.041),
            ),
        )

        lines = overlay_lines(
            result,
            fps=30.0,
            source="video:C:/Users/Someone/very/long/path/color_160.avi",
            mode="sliding",
            state="result",
        )

        self.assertLessEqual(max(map(len, lines)), 52)
        self.assertIn("video:color_160.avi", "\n".join(lines))
        for label in ("K42_16", "K42_08", "K42_42"):
            self.assertIn(label, "\n".join(lines))

    def test_overlay_displays_locked_chinese_meaning_with_each_prediction(self):
        result = InferenceResult(
            top1=Prediction("K42_01", "你好", 0.80),
            top3=(
                Prediction("K42_01", "你好", 0.80),
                Prediction("K42_02", "謝謝", 0.12),
                Prediction("K42_03", "再見", 0.08),
            ),
        )

        rendered = "\n".join(
            overlay_lines(result, fps=30.0, source="camera:0", mode="auto", state="RESULT")
        )

        self.assertIn("K42_01 你好", rendered)
        self.assertIn("K42_02 謝謝", rendered)
        self.assertIn("K42_03 再見", rendered)
        self.assertIn("80.0%", rendered)
        self.assertIn("M auto/manual", rendered)

    def test_self_test_uses_no_camera_and_runs_real_42_logit_forward(self):
        FakeDetectorContext.entered = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = make_bundle(Path(temp_dir))

            with mock.patch(
                "recognition.realtime.knee42_ivcam.open_camera",
                side_effect=AssertionError("self-test must not enumerate cameras"),
            ):
                result = run_self_test(
                    bundle,
                    device=torch.device("cpu"),
                    detector_factory=lambda _bundle: FakeDetectorContext(),
                )

            self.assertEqual(result["logit_shape"], [1, 42])
            self.assertEqual(result["preprocessing_shape"], [64, 438])
            self.assertFalse(result["camera_opened"])
            self.assertTrue(result["integrity_verified"])
            self.assertEqual(FakeDetectorContext.entered, 1)

    def test_self_test_rejects_tampered_auto_trigger_before_detector_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = make_bundle(root)
            (root / "recognition" / "realtime" / "auto_trigger.py").write_text(
                "tampered", encoding="utf-8"
            )
            detector_factory = mock.Mock(side_effect=AssertionError("detector must not be created"))

            with self.assertRaisesRegex(IntegrityError, "auto-trigger source"):
                run_self_test(
                    bundle,
                    device=torch.device("cpu"),
                    detector_factory=detector_factory,
                )

            detector_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
