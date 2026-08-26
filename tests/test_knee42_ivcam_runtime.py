from __future__ import annotations

import hashlib
import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import recognition.realtime.knee42_ivcam as knee42_ivcam

from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier
from recognition.realtime.knee42_capture import OpenCVFrameSource
from recognition.realtime.knee42_clock import LiveClock
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
from recognition.realtime.knee42_preprocessing import (
    LANDMARK_DIM,
    materialize_sequence,
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


class PacketSequenceSource:
    status = "video:sequence.mp4"
    fps = 30.0
    clock_mode = "video_source_timestamp"

    def __init__(self, timestamps):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self._packets = [
            SimpleNamespace(
                frame=frame.copy(),
                timestamp_sec=float(timestamp),
                clock_mode=self.clock_mode,
            )
            for timestamp in timestamps
        ]
        self.released = False

    def read_packet(self):
        return self._packets.pop(0) if self._packets else None

    def release(self):
        self.released = True


class ValueRuntimeBundle:
    sequence_length = 64
    frame_step = 2
    model_display_version = "test"

    def __init__(self, root):
        self.root = root

    def predict(self, values, _mask):
        label_index = int(round(float(values[-1, 0]))) + 1
        prediction = Prediction(f"K42_{label_index:02d}", "test", 1.0)
        return InferenceResult(top1=prediction, top3=(prediction,))


class ValueDetectors:
    def __init__(self):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def extract_observation(self, _frame):
        value = float(self.calls)
        self.calls += 1
        return SimpleNamespace(
            display_pose=None,
            display_left_hand=None,
            display_right_hand=None,
            recognition_values=np.full(219, value, dtype=np.float32),
            recognition_mask=np.ones(219, dtype=np.bool_),
            trigger_values=np.ones(225, dtype=np.float32),
        )


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
        self.assertTrue(hasattr(data.top1, "raw_probability"))
        self.assertEqual(data.top1.raw_probability, 0.268)
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

    def test_detector_boundaries_require_explicit_pixels_mirrored_keyword(self):
        bundle = SimpleNamespace(root=Path("bundle"))
        for boundary in (
            knee42_ivcam.MediapipeDetectors,
            knee42_ivcam.create_mediapipe_detectors,
        ):
            with self.subTest(boundary=boundary.__name__):
                with self.assertRaisesRegex(TypeError, "pixels_mirrored"):
                    boundary(bundle)

    def test_display_orientation_mirrors_copies_and_preserves_model_materialization(self):
        frame = np.repeat(
            np.arange(1, 7, dtype=np.uint8).reshape(2, 3, 1),
            3,
            axis=2,
        )
        pose = np.full((33, 3), np.nan, dtype=np.float32)
        left_hand = np.full((21, 3), np.nan, dtype=np.float32)
        right_hand = np.full((21, 3), np.nan, dtype=np.float32)
        pose[11] = (0.20, 0.31, 0.11)
        pose[12] = (0.65, 0.69, 0.12)
        left_hand[0] = (0.15, 0.41, -0.35)
        right_hand[0] = (0.80, 0.59, 0.45)
        original_frame = frame.copy()
        original_pose = pose.copy()
        original_left = left_hand.copy()
        original_right = right_hand.copy()
        values = np.arange(2 * LANDMARK_DIM, dtype=np.float32).reshape(2, LANDMARK_DIM)
        mask = np.ones_like(values, dtype=np.bool_)
        materialized_before = materialize_sequence(
            values,
            mask,
            np.zeros(LANDMARK_DIM, dtype=np.float32),
            np.ones(LANDMARK_DIM, dtype=np.float32),
        )
        values_bytes = values.tobytes()
        mask_bytes = mask.tobytes()

        display_frame, display_pose, display_left, display_right = (
            knee42_ivcam._display_orientation(
                frame,
                pose,
                left_hand,
                right_hand,
                display_mirror=True,
            )
        )

        np.testing.assert_array_equal(display_frame, np.flip(original_frame, axis=1))
        self.assertAlmostEqual(float(display_pose[11, 0]), 0.80)
        self.assertAlmostEqual(float(display_pose[12, 0]), 0.35)
        self.assertAlmostEqual(float(display_left[0, 0]), 0.85)
        self.assertAlmostEqual(float(display_right[0, 0]), 0.20)
        self.assertAlmostEqual(float(display_left[0, 2]), -0.35)
        self.assertAlmostEqual(float(display_right[0, 2]), 0.45)
        self.assertTrue(np.isnan(display_pose[0]).all())
        self.assertTrue(np.isnan(display_left[1]).all())
        self.assertTrue(np.isnan(display_right[1]).all())
        self.assertIsNot(display_frame, frame)
        self.assertIsNot(display_pose, pose)
        self.assertIsNot(display_left, left_hand)
        self.assertIsNot(display_right, right_hand)
        np.testing.assert_array_equal(frame, original_frame)
        np.testing.assert_array_equal(pose, original_pose)
        np.testing.assert_array_equal(left_hand, original_left)
        np.testing.assert_array_equal(right_hand, original_right)
        self.assertEqual(values.tobytes(), values_bytes)
        self.assertEqual(mask.tobytes(), mask_bytes)
        materialized_after = materialize_sequence(
            values,
            mask,
            np.zeros(LANDMARK_DIM, dtype=np.float32),
            np.ones(LANDMARK_DIM, dtype=np.float32),
        )
        np.testing.assert_array_equal(materialized_after, materialized_before)

        display_frame, display_pose, display_left, display_right = (
            knee42_ivcam._display_orientation(
                frame,
                None,
                None,
                None,
                display_mirror=True,
            )
        )
        np.testing.assert_array_equal(display_frame, np.flip(frame, axis=1))
        self.assertIsNone(display_pose)
        self.assertIsNone(display_left)
        self.assertIsNone(display_right)

    def test_runtime_threads_input_orientation_and_keeps_display_mirror_out_of_features(self):
        raw_frame = np.repeat(
            np.arange(1, 7, dtype=np.uint8).reshape(2, 3, 1),
            3,
            axis=2,
        )
        expected_input_frame = np.repeat(
            np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8)[:, :, None],
            3,
            axis=2,
        )
        expected_display_mirror = np.repeat(
            np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8)[:, :, None],
            3,
            axis=2,
        )
        detector_frames = []
        display_frames = []
        predicted_features = []
        source_calls = []
        source_captures = []
        detector_policies = []
        results = []

        class PacketCapture:
            def __init__(self):
                self.frames = [raw_frame.copy()]
                self.released = False

            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)

            def release(self):
                self.released = True

            def get(self, _property):
                return 30.0

        packet_cv2 = SimpleNamespace(CAP_PROP_FPS=5)

        class Bundle:
            sequence_length = 1
            frame_step = 1
            model_display_version = "test"

            def __init__(self, root):
                self.root = root

            def predict(self, values, mask):
                predicted_features.append((values.copy(), mask.copy()))
                prediction = Prediction("K42_01", "test", 1.0)
                return InferenceResult(top1=prediction, top3=(prediction,))

        class Detectors:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_observation(self, frame):
                detector_frames.append(frame.copy())
                return SimpleNamespace(
                    display_pose=None,
                    display_left_hand=None,
                    display_right_hand=None,
                    recognition_values=np.arange(219, dtype=np.float32),
                    recognition_mask=np.ones(219, dtype=np.bool_),
                    trigger_values=np.ones(225, dtype=np.float32),
                )

        class Display:
            def __init__(self, *_args, **_kwargs):
                pass

            def create(self, _cv2):
                return None

            def content_size(self, _cv2):
                return (6, 4)

        def open_source(path, *, rotation, input_mirror, cv2_module):
            source_calls.append((path, rotation, input_mirror, cv2_module is not None))
            capture = PacketCapture()
            source_captures.append(capture)
            return OpenCVFrameSource(
                capture,
                "video:orientation.mp4",
                packet_cv2,
                clock=LiveClock(perf_counter=iter((100.0,)).__next__),
                rotation_degrees=rotation,
                input_mirror=input_mirror,
            )

        def create_detectors(_bundle, *, pixels_mirrored):
            detector_policies.append(pixels_mirrored)
            return Detectors()

        def render(frame, *_args, **_kwargs):
            display_frames.append(frame.copy())
            return frame, None

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_root = Path(temp_dir) / "model"
            bundle_root.mkdir()
            for display_mirror in (False, True):
                with (
                    mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                    mock.patch.object(knee42_ivcam, "load_bundle", return_value=Bundle(bundle_root)),
                    mock.patch.object(knee42_ivcam, "open_video", side_effect=open_source),
                    mock.patch.object(knee42_ivcam, "create_mediapipe_detectors", side_effect=create_detectors),
                    mock.patch.object(knee42_ivcam, "ResizableDisplay", Display),
                    mock.patch.object(knee42_ivcam, "windows_primary_screen_size", return_value=(800, 600)),
                    mock.patch.object(knee42_ivcam, "render_application_view", side_effect=render),
                    mock.patch("cv2.imshow"),
                    mock.patch("cv2.waitKey", return_value=-1),
                    mock.patch("cv2.destroyAllWindows"),
                ):
                    results.append(
                        knee42_ivcam.run_capture(
                            bundle_root,
                            mode="sliding",
                            camera_index=None,
                            video=Path("orientation.mp4"),
                            device=torch.device("cpu"),
                            headless=False,
                            max_frames=None,
                            inference_stride=1,
                            rotation=90,
                            input_mirror=True,
                            display_mirror=display_mirror,
                        )
                    )

        self.assertEqual(
            source_calls,
            [
                (Path("orientation.mp4"), 90, True, True),
                (Path("orientation.mp4"), 90, True, True),
            ],
        )
        self.assertEqual(detector_policies, [True, True])
        np.testing.assert_array_equal(detector_frames[0], expected_input_frame)
        np.testing.assert_array_equal(detector_frames[1], expected_input_frame)
        np.testing.assert_array_equal(display_frames[0], expected_input_frame)
        np.testing.assert_array_equal(display_frames[1], expected_display_mirror)
        np.testing.assert_array_equal(predicted_features[0][0], predicted_features[1][0])
        np.testing.assert_array_equal(predicted_features[0][1], predicted_features[1][1])
        self.assertTrue(all(capture.released for capture in source_captures))
        for result in results:
            self.assertEqual(result["resolved_rotation"], 90)
            self.assertTrue(result["input_mirror"])
        self.assertFalse(results[0]["display_mirror"])
        self.assertTrue(results[1]["display_mirror"])

    def test_runtime_uses_packet_timestamps_without_nominal_fps_synthesis(self):
        from recognition.realtime.knee42_ivcam import run_capture

        source = inspect.getsource(run_capture)

        self.assertIn("source.read_packet()", source)
        self.assertIn("packet.timestamp_sec", source)
        self.assertNotIn("(raw_frames - 1) / source_fps", source)
        self.assertNotIn("controller.add_held_observation(", source)
        self.assertNotIn("frame_interval_sec", source)

    def test_held_observation_batch_rejects_unbounded_sample_count(self):
        with self.assertRaisesRegex(ValueError, "bound|at most"):
            knee42_ivcam.HeldObservationBatch(65)

    def test_runtime_buffers_exact_packet_times_for_held_observations(self):
        held_calls = []
        recorder_origins = []
        recorder_timestamps = []

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

            def add_frame(self, _frame, *, timestamp_sec):
                recorder_timestamps.append(timestamp_sec)
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
        self.assertEqual(recorder_timestamps, [0.0, 0.033, 0.071])
        self.assertEqual(result["raw_frames"], 3)
        self.assertEqual(result["feature_frames"], 2)
        self.assertEqual(result["clock_mode"], "video_source_timestamp")
        self.assertEqual(result["first_frame_timestamp_sec"], 0.0)
        self.assertEqual(result["last_frame_timestamp_sec"], 0.071)
        self.assertTrue(packet_source.released)

    def test_reset_drops_held_observation_and_prevents_future_timestamp_ownership(self):
        held_calls = []

        class PacketSource:
            status = "video:reset.mp4"
            fps = 30.0
            clock_mode = "video_source_timestamp"

            def __init__(self):
                frame = np.zeros((4, 6, 3), dtype=np.uint8)
                self._packets = [
                    SimpleNamespace(
                        frame=frame.copy(),
                        timestamp_sec=timestamp,
                        clock_mode=self.clock_mode,
                    )
                    for timestamp in (0.0, 0.033, 0.066, 0.100)
                ]
                self.released = False

            def read_packet(self):
                return self._packets.pop(0) if self._packets else None

            def release(self):
                self.released = True

        class FakeBundle:
            sequence_length = 64
            frame_step = 2
            model_display_version = "test"

            def __init__(self, root):
                self.root = root

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
                return SimpleNamespace(infer=True, features=(feature,), segment=None)

            def finalize_video_eof(self):
                return SimpleNamespace(infer=False, features=(), segment=None)

            def reset(self):
                return SimpleNamespace(infer=False, features=(), segment=None)

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

        class FakeDisplay:
            def __init__(self, *_args, **_kwargs):
                pass

            def create(self, _cv2):
                return None

            def content_size(self, _cv2):
                return (6, 4)

            def toggle_fullscreen(self, _cv2):
                return False

        packet_source = PacketSource()
        keys = iter((ord("r"), -1, -1, -1))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(
                    knee42_ivcam,
                    "load_bundle",
                    return_value=FakeBundle(bundle_root),
                ),
                mock.patch.object(knee42_ivcam, "open_video", return_value=packet_source),
                mock.patch.object(
                    knee42_ivcam,
                    "load_auto_trigger_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "AutoKnee42Controller",
                    FakeAutoController,
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "create_mediapipe_detectors",
                    return_value=FakeDetectors(),
                ),
                mock.patch.object(knee42_ivcam, "ResizableDisplay", FakeDisplay),
                mock.patch.object(
                    knee42_ivcam,
                    "render_application_view",
                    side_effect=lambda frame, *_args, **_kwargs: (frame, None),
                ),
                mock.patch("cv2.imshow"),
                mock.patch("cv2.waitKey", side_effect=lambda _delay: next(keys)),
                mock.patch("cv2.destroyAllWindows"),
            ):
                knee42_ivcam.run_capture(
                    bundle_root,
                    mode="auto",
                    camera_index=None,
                    video=Path("reset.mp4"),
                    device=torch.device("cpu"),
                    headless=False,
                    max_frames=None,
                )

        self.assertEqual(held_calls, [((0.066, 0.100), 1.0)])
        self.assertTrue(packet_source.released)

    def test_source_is_released_when_trigger_config_is_missing_after_open(self):
        source = SimpleNamespace(released=False)
        source.release = lambda: setattr(source, "released", True)

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_root = Path(temp_dir) / "model"
            bundle_root.mkdir()
            bundle = SimpleNamespace(root=bundle_root, frame_step=2)
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(knee42_ivcam, "load_bundle", return_value=bundle),
                mock.patch.object(knee42_ivcam, "open_video", return_value=source),
            ):
                with self.assertRaisesRegex(IntegrityError, "config missing"):
                    knee42_ivcam.run_capture(
                        bundle_root,
                        mode="auto",
                        camera_index=None,
                        video=Path("missing-config.mp4"),
                        device=torch.device("cpu"),
                        headless=True,
                        max_frames=1,
                    )

        self.assertTrue(source.released)

    def test_source_is_released_when_controller_creation_fails_after_open(self):
        source = SimpleNamespace(released=False)
        source.release = lambda: setattr(source, "released", True)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text(
                "{}", encoding="utf-8"
            )
            bundle = SimpleNamespace(root=bundle_root)
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(knee42_ivcam, "load_bundle", return_value=bundle),
                mock.patch.object(knee42_ivcam, "open_video", return_value=source),
                mock.patch.object(
                    knee42_ivcam,
                    "load_auto_trigger_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "AutoKnee42Controller",
                    side_effect=RuntimeError("controller creation failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "controller creation failed"):
                    knee42_ivcam.run_capture(
                        bundle_root,
                        mode="auto",
                        camera_index=None,
                        video=Path("controller.mp4"),
                        device=torch.device("cpu"),
                        headless=True,
                        max_frames=1,
                    )

        self.assertTrue(source.released)

    def test_source_and_window_are_cleaned_when_display_creation_fails(self):
        source = SimpleNamespace(released=False)
        source.release = lambda: setattr(source, "released", True)

        class FakeAutoController:
            def __init__(self, _config, *, initial_mode):
                self.mode = initial_mode

        class FailingDisplay:
            def __init__(self, *_args, **_kwargs):
                pass

            def create(self, _cv2):
                raise RuntimeError("display creation failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text(
                "{}", encoding="utf-8"
            )
            bundle = SimpleNamespace(root=bundle_root, frame_step=2)
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(knee42_ivcam, "load_bundle", return_value=bundle),
                mock.patch.object(knee42_ivcam, "open_video", return_value=source),
                mock.patch.object(
                    knee42_ivcam,
                    "load_auto_trigger_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "AutoKnee42Controller",
                    FakeAutoController,
                ),
                mock.patch.object(knee42_ivcam, "ResizableDisplay", FailingDisplay),
                mock.patch("cv2.destroyAllWindows") as destroy_windows,
            ):
                with self.assertRaisesRegex(RuntimeError, "display creation failed"):
                    knee42_ivcam.run_capture(
                        bundle_root,
                        mode="auto",
                        camera_index=None,
                        video=Path("display.mp4"),
                        device=torch.device("cpu"),
                        headless=False,
                        max_frames=1,
                    )

        self.assertTrue(source.released)
        destroy_windows.assert_called_once_with()

    def test_pose_detector_enter_failure_closes_the_entered_hand_detector(self):
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker

        class HandContext:
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.closed = True
                return False

        class PoseContext:
            def __enter__(self):
                raise RuntimeError("pose enter failed")

            def __exit__(self, *_args):
                return False

        hand_context = HandContext()
        detectors = knee42_ivcam.MediapipeDetectors(
            SimpleNamespace(root=Path("bundle")),
            pixels_mirrored=False,
        )
        with (
            mock.patch.object(
                HandLandmarker,
                "create_from_options",
                return_value=hand_context,
            ),
            mock.patch.object(
                PoseLandmarker,
                "create_from_options",
                return_value=PoseContext(),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "pose enter failed"):
                detectors.__enter__()

        self.assertTrue(hand_context.closed)

    def test_recorder_stop_failure_cannot_skip_other_runtime_cleanup(self):
        class PacketSource:
            status = "video:cleanup.mp4"
            fps = 30.0
            clock_mode = "video_source_timestamp"

            def __init__(self):
                frame = np.zeros((4, 6, 3), dtype=np.uint8)
                self._packets = [
                    SimpleNamespace(
                        frame=frame.copy(),
                        timestamp_sec=timestamp,
                        clock_mode=self.clock_mode,
                    )
                    for timestamp in (0.0, 0.04)
                ]
                self.released = False

            def read_packet(self):
                return self._packets.pop(0) if self._packets else None

            def release(self):
                self.released = True

        class FakeBundle:
            sequence_length = 64
            frame_step = 2
            model_display_version = "test"

            def __init__(self, root):
                self.root = root

            def predict(self, _values, _mask):
                prediction = Prediction("K42_01", "你好", 1.0)
                return InferenceResult(top1=prediction, top3=(prediction,))

        class FakeAutoController:
            def __init__(self, _config, *, initial_mode):
                self.mode = initial_mode
                self.state = "WAITING"
                self.calibrated = True

            def add_held_observation_at_times(self, _timestamps, _trigger, feature):
                return SimpleNamespace(infer=True, features=(feature,), segment=None)

            def reset(self):
                return None

        class FakeDetectors:
            exited = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.exited = True
                return False

            def extract_observation(self, _frame):
                return SimpleNamespace(
                    display_pose=None,
                    display_left_hand=None,
                    display_right_hand=None,
                    recognition_values=np.zeros(219, dtype=np.float32),
                    recognition_mask=np.ones(219, dtype=np.bool_),
                    trigger_values=np.ones(225, dtype=np.float32),
                )

        class FailingRecorder:
            segment_count = 0

            def __init__(self, *_args, **_kwargs):
                pass

            def add_frame(self, _frame, *, timestamp_sec):
                return None

            def record_segment(self, _segment, _result):
                return None

            def stop(self):
                raise RuntimeError("recorder stop failed")

        class FakeDisplay:
            def __init__(self, *_args, **_kwargs):
                pass

            def create(self, _cv2):
                return None

            def content_size(self, _cv2):
                return (6, 4)

        packet_source = PacketSource()
        detectors = FakeDetectors()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(
                    knee42_ivcam,
                    "load_bundle",
                    return_value=FakeBundle(bundle_root),
                ),
                mock.patch.object(knee42_ivcam, "open_video", return_value=packet_source),
                mock.patch.object(
                    knee42_ivcam,
                    "load_auto_trigger_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "AutoKnee42Controller",
                    FakeAutoController,
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "create_mediapipe_detectors",
                    return_value=detectors,
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "SegmentSessionRecorder",
                    FailingRecorder,
                ),
                mock.patch.object(knee42_ivcam, "ResizableDisplay", FakeDisplay),
                mock.patch.object(
                    knee42_ivcam,
                    "render_application_view",
                    side_effect=lambda frame, *_args, **_kwargs: (frame, None),
                ),
                mock.patch("cv2.imshow"),
                mock.patch("cv2.waitKey", return_value=-1),
                mock.patch("cv2.destroyAllWindows") as destroy_windows,
            ):
                with self.assertRaisesRegex(RuntimeError, "recorder stop failed"):
                    knee42_ivcam.run_capture(
                        bundle_root,
                        mode="auto",
                        camera_index=None,
                        video=Path("cleanup.mp4"),
                        device=torch.device("cpu"),
                        headless=False,
                        max_frames=2,
                        start_logging=True,
                    )

        self.assertTrue(packet_source.released)
        self.assertTrue(detectors.exited)
        destroy_windows.assert_called_once_with()

    def test_max_frames_truncation_drops_pending_batch_without_eof_finalization(self):
        held_calls = []
        finalize_calls = []

        class TruncationController:
            def __init__(self, _config, *, initial_mode):
                self.mode = initial_mode
                self.state = "END_CONFIRM"
                self.calibrated = True
                self._feature = None

            def add_held_observation_at_times(self, timestamps, _trigger, feature):
                held_calls.append(tuple(timestamps))
                self._feature = feature
                return SimpleNamespace(infer=False, features=(), segment=None)

            def finalize_video_eof(self):
                finalize_calls.append(True)
                return SimpleNamespace(
                    infer=True,
                    features=(self._feature,),
                    segment=None,
                )

            def reset(self):
                return None

        packet_source = PacketSequenceSource((0.0, 0.04, 0.08))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(
                    knee42_ivcam,
                    "load_bundle",
                    return_value=ValueRuntimeBundle(bundle_root),
                ),
                mock.patch.object(knee42_ivcam, "open_video", return_value=packet_source),
                mock.patch.object(
                    knee42_ivcam,
                    "load_auto_trigger_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "AutoKnee42Controller",
                    TruncationController,
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "create_mediapipe_detectors",
                    return_value=ValueDetectors(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "without an inference result"):
                    knee42_ivcam.run_capture(
                        bundle_root,
                        mode="auto",
                        camera_index=None,
                        video=Path("truncated.mp4"),
                        device=torch.device("cpu"),
                        headless=True,
                        max_frames=1,
                    )

        self.assertEqual(held_calls, [])
        self.assertEqual(finalize_calls, [])
        self.assertTrue(packet_source.released)

    def test_natural_eof_flushes_pending_batch_and_finalizes_after_earlier_result(self):
        held_calls = []
        finalize_calls = []

        class NaturalEofController:
            def __init__(self, _config, *, initial_mode):
                self.mode = initial_mode
                self.state = "END_CONFIRM"
                self.calibrated = True
                self._latest_feature = None

            def add_held_observation_at_times(self, timestamps, _trigger, feature):
                held_calls.append((tuple(timestamps), float(feature[0][0])))
                self._latest_feature = feature
                return SimpleNamespace(
                    infer=len(held_calls) == 1,
                    features=(feature,),
                    segment=None,
                )

            def finalize_video_eof(self):
                finalize_calls.append(True)
                return SimpleNamespace(
                    infer=True,
                    features=(self._latest_feature,),
                    segment=None,
                )

            def reset(self):
                return None

        packet_source = PacketSequenceSource((0.0, 0.04, 0.08))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_root = root / "model"
            bundle_root.mkdir()
            (root / "auto_trigger_knee_ivcam_local.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(knee42_ivcam, "verify_auto_trigger_provenance"),
                mock.patch.object(
                    knee42_ivcam,
                    "load_bundle",
                    return_value=ValueRuntimeBundle(bundle_root),
                ),
                mock.patch.object(knee42_ivcam, "open_video", return_value=packet_source),
                mock.patch.object(
                    knee42_ivcam,
                    "load_auto_trigger_config",
                    return_value=object(),
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "AutoKnee42Controller",
                    NaturalEofController,
                ),
                mock.patch.object(
                    knee42_ivcam,
                    "create_mediapipe_detectors",
                    return_value=ValueDetectors(),
                ),
            ):
                result = knee42_ivcam.run_capture(
                    bundle_root,
                    mode="auto",
                    camera_index=None,
                    video=Path("natural-eof.mp4"),
                    device=torch.device("cpu"),
                    headless=True,
                    max_frames=None,
                )

        self.assertEqual(
            held_calls,
            [((0.0, 0.04), 0.0), ((0.08,), 1.0)],
        )
        self.assertEqual(finalize_calls, [True])
        self.assertEqual(result["top1"], "K42_02")
        self.assertIn("top1_raw_probability", result)
        self.assertEqual(result["top1_raw_probability"], 1.0)
        self.assertNotIn("top1_confidence", result)
        self.assertEqual(
            result["probability_policy"],
            {
                "kind": "uncalibrated_softmax",
                "acceptance_policy": "disabled_no_risk_coverage_evidence",
                "calibration_artifact": None,
            },
        )
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

    def test_decode_returns_sorted_top1_top3_and_exact_raw_probability(self):
        logits = torch.tensor([[0.0, 3.0, 2.0] + [-1.0] * 39])
        expected = torch.softmax(logits[0], dim=0)

        result = decode_logits(logits, LABELS, {label: label for label in LABELS})

        self.assertEqual(result.top1.label_id, "K42_02")
        self.assertEqual([item.label_id for item in result.top3], ["K42_02", "K42_03", "K42_01"])
        self.assertTrue(all(hasattr(item, "raw_probability") for item in result.top3))
        self.assertEqual(result.top1.raw_probability, float(expected[1].item()))
        self.assertEqual(
            [item.raw_probability for item in result.top3],
            [float(expected[index].item()) for index in (1, 2, 0)],
        )
        self.assertGreater(result.top1.raw_probability, result.top3[1].raw_probability)

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
        self.assertIn("raw probability", rendered.lower())
        self.assertNotIn("confidence", rendered.lower())
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
        self.assertEqual(automatic.rotation, "auto")
        self.assertEqual(automatic.input_mirror, "off")
        self.assertEqual(automatic.display_mirror, "off")
        self.assertEqual(manual.mode, "manual")
        self.assertEqual(sliding.mode, "sliding")

        oriented = parser.parse_args(
            [
                "--bundle",
                "model",
                "--rotation",
                "270",
                "--input-mirror",
                "on",
                "--display-mirror",
                "on",
            ]
        )
        self.assertEqual(oriented.rotation, "270")
        self.assertEqual(oriented.input_mirror, "on")
        self.assertEqual(oriented.display_mirror, "on")

        for invalid_args in (
            ["--rotation", "45"],
            ["--input-mirror", "true"],
            ["--display-mirror", "yes"],
        ):
            with self.subTest(invalid_args=invalid_args):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(["--bundle", "model", *invalid_args])

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
        detector_policies = []

        def detector_factory(_bundle, *, pixels_mirrored):
            detector_policies.append(pixels_mirrored)
            return FakeDetectorContext()

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = make_bundle(Path(temp_dir))

            with mock.patch(
                "recognition.realtime.knee42_ivcam.open_camera",
                side_effect=AssertionError("self-test must not enumerate cameras"),
            ):
                result = run_self_test(
                    bundle,
                    device=torch.device("cpu"),
                    detector_factory=detector_factory,
                )

            self.assertEqual(result["logit_shape"], [1, 42])
            self.assertEqual(result["preprocessing_shape"], [64, 438])
            self.assertFalse(result["camera_opened"])
            self.assertTrue(result["integrity_verified"])
            self.assertEqual(FakeDetectorContext.entered, 1)
            self.assertEqual(detector_policies, [False])

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
