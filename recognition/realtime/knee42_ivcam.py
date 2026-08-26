"""Hash-verified RGB IVCAM/video inference runtime for the locked Knee42 model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier
from recognition.realtime.auto_trigger import load_auto_trigger_config
from recognition.realtime.knee42_capture import open_camera, open_video
from recognition.realtime.knee42_controllers import AutoKnee42Controller, SlidingController
from recognition.realtime.knee42_display import (
    DisplayPanelData,
    DisplayPrediction,
    ResizableDisplay,
    render_application_view,
    windows_primary_screen_size,
)
from recognition.realtime.knee42_integrity import (
    IntegrityError,
    parse_sha256_manifest,
    sha256_file,
)
from recognition.realtime.knee42_orientation import (
    InputOrientation,
    MirrorMode,
    RotationSetting,
    parse_rotation,
    resolve_rotation,
)
from recognition.realtime.knee42_session_recording import SegmentSessionRecorder
from recognition.realtime.knee42_preprocessing import (
    FrameObservation,
    LANDMARK_DIM,
    MODEL_INPUT_DIM,
    POSE_KEEP,
    materialize_sequence,
    normalize_frame,
    observation_from_results,
)
from recognition.realtime.probability_reporting import (
    probability_policy_record,
    validate_raw_probability,
)


LABELS = [f"K42_{number:02d}" for number in range(1, 43)]


REQUIRED_INTEGRITY_FILES = frozenset(
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
    }
)
SELECTION_BOUND_FILES = frozenset(
    {
        "best_model.pt",
        "feature_config.json",
        "standardizer_train_only.npz",
        "label_map_knee42.json",
        "display_text_map.json",
    }
)


def sha256_canonical_text_file(path: Path) -> str:
    """Hash locked text identically after LF or Windows CRLF checkout."""
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_integrity_manifest(bundle_dir: Path) -> dict[str, str]:
    bundle_dir = Path(bundle_dir).resolve()
    manifest = bundle_dir / "integrity_manifest.sha256"
    if not manifest.is_file():
        raise IntegrityError(f"integrity manifest missing: {manifest}")
    expected = dict(parse_sha256_manifest(manifest))
    missing = sorted(REQUIRED_INTEGRITY_FILES - set(expected))
    if missing:
        raise IntegrityError(f"integrity manifest missing required files: {missing}")
    for relative, wanted in expected.items():
        path = bundle_dir / relative
        if not path.is_file():
            raise IntegrityError(f"integrity file missing: {relative}")
        actual = sha256_file(path)
        if actual != wanted:
            raise IntegrityError(
                f"integrity SHA-256 mismatch for {relative}: expected {wanted}, actual {actual}"
            )
    return expected


def verify_auto_trigger_provenance(package_root: Path) -> dict[str, Any]:
    """Bind runtime trigger source and config to the supplied archived hashes."""
    package_root = Path(package_root).resolve()
    provenance = _read_json(package_root / "auto_trigger_provenance.json")
    source = package_root / "recognition" / "realtime" / "auto_trigger.py"
    config = package_root / "auto_trigger_knee_ivcam_local.json"
    checks = (
        ("auto-trigger source", source, "auto_trigger_source_sha256"),
        (
            "auto-trigger controller",
            package_root / "recognition" / "realtime" / "knee42_controllers.py",
            "auto_trigger_controller_sha256",
        ),
        ("auto-trigger config", config, "auto_trigger_config_sha256"),
    )
    for description, path, key in checks:
        if not path.is_file():
            raise IntegrityError(f"{description} missing: {path}")
        wanted = str(provenance.get(key, "")).lower()
        actual = sha256_canonical_text_file(path)
        if len(wanted) != 64 or actual != wanted:
            raise IntegrityError(
                f"{description} SHA-256 mismatch: expected {wanted}, actual {actual}"
            )
    return provenance


@dataclass(frozen=True)
class Prediction:
    label_id: str
    display_text: str
    raw_probability: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_probability",
            validate_raw_probability(self.raw_probability),
        )

    @property
    def confidence(self) -> float:
        """Deprecated read-only alias for the exact raw probability."""
        warnings.warn(
            "Prediction.confidence is deprecated; use raw_probability",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.raw_probability


@dataclass(frozen=True)
class InferenceResult:
    top1: Prediction
    top3: tuple[Prediction, Prediction, Prediction]


def decode_logits(
    logits: torch.Tensor,
    labels: Sequence[str],
    display_text: dict[str, str],
) -> InferenceResult:
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1 or logits.shape[0] != len(labels) or len(labels) != 42:
        raise ValueError(f"expected one 42-logit vector, got {tuple(logits.shape)}")
    probabilities = torch.softmax(logits.detach().cpu(), dim=0)
    ranked = torch.argsort(probabilities, descending=True)[:3].tolist()
    predictions = tuple(
        Prediction(
            label_id=str(labels[index]),
            display_text=str(display_text.get(str(labels[index]), str(labels[index]))),
            raw_probability=float(probabilities[index].item()),
        )
        for index in ranked
    )
    return InferenceResult(top1=predictions[0], top3=predictions)  # type: ignore[arg-type]


@dataclass
class Knee42Bundle:
    root: Path
    model: BiGRUSentenceClassifier
    mean: np.ndarray
    std: np.ndarray
    labels: list[str]
    display_text: dict[str, str]
    sequence_length: int
    frame_step: int
    model_display_version: str
    device: torch.device

    def prepare(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return materialize_sequence(
            values,
            mask,
            self.mean,
            self.std,
            sequence_length=self.sequence_length,
        )

    def forward_prepared(self, prepared: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(prepared, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor)

    def predict(self, values: np.ndarray, mask: np.ndarray) -> InferenceResult:
        return decode_logits(
            self.forward_prepared(self.prepare(values, mask)),
            self.labels,
            self.display_text,
        )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"cannot read validated JSON {path.name}: {exc}") from exc


def load_bundle(bundle_dir: Path, *, device: torch.device) -> Knee42Bundle:
    """Verify every startup artifact before deserializing the selected checkpoint."""
    bundle_dir = Path(bundle_dir).resolve()
    verified_hashes = verify_integrity_manifest(bundle_dir)
    ledger = _read_json(bundle_dir / "selection_ledger.json")
    if ledger.get("selection_metric") != "dev_macro_top1":
        raise IntegrityError("selection ledger is not Dev-selected")
    ledger_hashes = ledger.get("artifacts", {})
    for name in SELECTION_BOUND_FILES:
        if ledger_hashes.get(name) != verified_hashes.get(name):
            raise IntegrityError(f"selection ledger SHA-256 mismatch for {name}")

    runtime = _read_json(bundle_dir / "runtime_config.json")
    feature = _read_json(bundle_dir / "feature_config.json")
    expected_runtime = {
        "sequence_length": 64,
        "landmark_value_dim": LANDMARK_DIM,
        "model_input_dim": MODEL_INPUT_DIM,
        "pose_landmarks_removed": [25, 26],
        "frame_step": 2,
        "stream": "RGB/color",
    }
    for key, expected_value in expected_runtime.items():
        if runtime.get(key) != expected_value:
            raise IntegrityError(
                f"runtime contract mismatch for {key}: expected {expected_value!r}, got {runtime.get(key)!r}"
            )
    if feature.get("sequence_length") != 64 or feature.get("input_dim") != 438:
        raise IntegrityError("feature config does not declare the locked 64x438 model input")
    if feature.get("knee_indices_removed") != [25, 26] or not feature.get("mask_concatenated"):
        raise IntegrityError("feature config does not match the knee-free values+mask contract")
    if (
        feature.get("features_final") != "knee42_features_upright_v2"
        or feature.get("video_orientation") != "container_rotation_metadata_applied_explicitly"
        or feature.get("horizontal_mirror") is not False
    ):
        raise IntegrityError("feature config does not match the locked upright-video contract")

    label_payload = _read_json(bundle_dir / "label_map_knee42.json")
    labels = [str(item) for item in label_payload.get("idx_to_label", [])]
    label_to_idx = {str(key): int(value) for key, value in label_payload.get("label_to_idx", {}).items()}
    if labels != LABELS or label_to_idx != {label: index for index, label in enumerate(LABELS)}:
        raise IntegrityError("label map does not match the ordered Knee42 contract")
    display_text = {str(key): str(value) for key, value in _read_json(bundle_dir / "display_text_map.json").items()}
    if set(display_text) != set(LABELS) or any(not value.strip() for value in display_text.values()):
        raise IntegrityError("display text map is incomplete")

    try:
        with np.load(bundle_dir / "standardizer_train_only.npz", allow_pickle=False) as payload:
            mean = payload["mean"].astype(np.float32)
            std = payload["std"].astype(np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise IntegrityError(f"invalid train-only standardizer: {exc}") from exc
    if mean.shape != (LANDMARK_DIM,) or std.shape != mean.shape or np.any(std <= 0):
        raise IntegrityError("train-only standardizer must contain positive 219-d mean/std")

    try:
        checkpoint = torch.load(
            bundle_dir / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
        model_config = checkpoint["model_config"]
        if int(model_config.get("input_dim", -1)) != MODEL_INPUT_DIM:
            raise IntegrityError("checkpoint input dimension is not 438")
        if int(model_config.get("num_classes", -1)) != len(LABELS):
            raise IntegrityError("checkpoint output dimension is not 42")
        model = BiGRUSentenceClassifier(**model_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"cannot load locked model: {exc}") from exc
    return Knee42Bundle(
        root=bundle_dir,
        model=model,
        mean=mean,
        std=std,
        labels=labels,
        display_text=display_text,
        sequence_length=64,
        frame_step=2,
        model_display_version=str(runtime.get("model_display_version", "v11")),
        device=device,
    )


def overlay_lines(
    result: InferenceResult | None,
    *,
    fps: float,
    source: str,
    mode: str,
    state: str,
    recording: bool = False,
    recorded_segments: int = 0,
    recording_session: str | None = None,
) -> list[str]:
    if result is None:
        prediction_lines = ["Top-1: waiting", "Top-3: waiting"]
    else:
        prediction_lines = [
            f"Top-1: {result.top1.label_id} {result.top1.display_text} | raw probability {result.top1.raw_probability:.1%}",
            *[
                f"Top-3 #{rank}: {item.label_id} {item.display_text} | raw probability {item.raw_probability:.1%}"
                for rank, item in enumerate(result.top3, 1)
            ],
        ]
    compact_source = source
    if source.startswith("video:"):
        filename = source[len("video:") :].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        compact_source = f"video:{filename}"
    compact_source = compact_source.encode("ascii", errors="ignore").decode("ascii")
    return [
        *prediction_lines,
        f"FPS {fps:.1f} | source {compact_source}",
        f"Mode {mode} | state {state}",
        f"REC {'ON' if recording else 'OFF'} | segments {recorded_segments}"
        + (f" | {recording_session}" if recording_session else ""),
        "F fullscreen | S start/stop audit | R reset",
        "M auto/manual | Space manual | Q/Esc quit",
    ]


def build_display_panel_data(
    result: InferenceResult | None,
    *,
    fps: float,
    source: str,
    mode: str,
    state: str,
    recording: bool,
    recorded_segments: int,
    model_version: str,
) -> DisplayPanelData:
    """Convert runtime state into the structured E2 display contract."""
    def convert(item: Prediction) -> DisplayPrediction:
        return DisplayPrediction(
            label_id=item.label_id,
            display_text=item.display_text,
            raw_probability=item.raw_probability,
        )

    return DisplayPanelData(
        top1=convert(result.top1) if result is not None else None,
        top3=tuple(convert(item) for item in result.top3) if result is not None else (),
        fps=float(fps),
        source=str(source),
        mode=str(mode),
        state=str(state),
        recording=bool(recording),
        recorded_segments=int(recorded_segments),
        model_version=str(model_version),
    )


def auto_display_state(state: str, *, calibrated: bool) -> str:
    if state == "IDLE_BLANK":
        return "WAITING" if calibrated else "CALIBRATING"
    return {
        "SIGNING_ACTIVE": "SIGNING",
        "END_CONFIRM": "END_CONFIRM",
        "FORCED_FINALIZE_COOLDOWN": "COOLDOWN",
    }.get(state, state)


class MediapipeDetectors:
    def __init__(self, bundle: Knee42Bundle, *, pixels_mirrored: bool):
        if type(pixels_mirrored) is not bool:
            raise TypeError(f"pixels_mirrored must be bool, got {pixels_mirrored!r}")
        self.bundle = bundle
        self.pixels_mirrored = pixels_mirrored
        self._stack: ExitStack | None = None
        self._mp = None
        self._hands = None
        self._pose = None

    def __enter__(self):
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.bundle.root / "hand_landmarker.task")),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2,
        )
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.bundle.root / "pose_landmarker.task")),
            running_mode=VisionTaskRunningMode.IMAGE,
        )
        with ExitStack() as stack:
            hands = stack.enter_context(
                HandLandmarker.create_from_options(hand_options)
            )
            pose = stack.enter_context(PoseLandmarker.create_from_options(pose_options))
            self._stack = stack.pop_all()
        self._hands = hands
        self._pose = pose
        self._mp = mp
        return self

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observation = self.extract_observation(frame_bgr)
        return observation.recognition_values, observation.recognition_mask

    def extract_observation(self, frame_bgr: np.ndarray) -> FrameObservation:
        if self._mp is None or self._hands is None or self._pose is None:
            raise RuntimeError("MediaPipe detectors are not open")
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        observation = observation_from_results(
            self._hands.detect(image),
            self._pose.detect(image),
            pixels_mirrored=self.pixels_mirrored,
        )
        return FrameObservation(
            trigger_values=observation.trigger_values,
            recognition_values=normalize_frame(
                observation.recognition_values,
                observation.recognition_mask,
            ),
            recognition_mask=observation.recognition_mask,
            display_pose=observation.display_pose,
            display_left_hand=observation.display_left_hand,
            display_right_hand=observation.display_right_hand,
        )

    def __exit__(self, *_args):
        if self._stack is not None:
            self._stack.close()
        return False


def create_mediapipe_detectors(
    bundle: Knee42Bundle,
    *,
    pixels_mirrored: bool,
) -> MediapipeDetectors:
    return MediapipeDetectors(bundle, pixels_mirrored=pixels_mirrored)


def run_self_test(
    bundle_dir: Path,
    *,
    device: torch.device,
    detector_factory: Callable[..., Any] = create_mediapipe_detectors,
) -> dict[str, Any]:
    verify_auto_trigger_provenance(Path(bundle_dir).resolve().parent)
    bundle = load_bundle(bundle_dir, device=device)
    with detector_factory(bundle, pixels_mirrored=False):
        mediapipe_created = True
    raw = np.zeros((5, LANDMARK_DIM), dtype=np.float32)
    mask = np.ones_like(raw, dtype=np.bool_)
    pose_index = {source: target for target, source in enumerate(POSE_KEEP)}
    raw[:, pose_index[11] * 3 : pose_index[11] * 3 + 3] = (-0.5, 0.0, 0.0)
    raw[:, pose_index[12] * 3 : pose_index[12] * 3 + 3] = (0.5, 0.0, 0.0)
    normalized = np.stack([normalize_frame(frame, frame_mask) for frame, frame_mask in zip(raw, mask)])
    prepared = bundle.prepare(normalized, mask)
    logits = bundle.forward_prepared(prepared)
    if list(logits.shape) != [1, 42]:
        raise RuntimeError(f"self-test model output shape mismatch: {list(logits.shape)}")
    result = decode_logits(logits, bundle.labels, bundle.display_text)
    return {
        "integrity_verified": True,
        "mediapipe_created": mediapipe_created,
        "preprocessing_shape": list(prepared.shape),
        "logit_shape": list(logits.shape),
        "top1": result.top1.label_id,
        "camera_opened": False,
        "device": str(device),
    }


def _predict_features(bundle: Knee42Bundle, features: Sequence[tuple[np.ndarray, np.ndarray]]) -> InferenceResult:
    values = np.stack([item[0] for item in features]).astype(np.float32)
    mask = np.stack([item[1] for item in features]).astype(np.bool_)
    return bundle.predict(values, mask)


MAX_HELD_BATCH_SAMPLES = 64


class HeldObservationBatch:
    """Own one detector observation and only its forward packet timestamps."""

    def __init__(self, sample_count: int):
        sample_count = int(sample_count)
        if sample_count <= 0 or sample_count > MAX_HELD_BATCH_SAMPLES:
            raise ValueError(
                "held observation sample_count must be positive and at most the "
                f"bounded maximum {MAX_HELD_BATCH_SAMPLES}"
            )
        self.sample_count = sample_count
        self._observation: tuple[
            np.ndarray,
            tuple[np.ndarray, np.ndarray],
        ] | None = None
        self._timestamps: list[float] = []

    @property
    def active(self) -> bool:
        return self._observation is not None

    @property
    def ready(self) -> bool:
        return self.active and len(self._timestamps) >= self.sample_count

    def start(
        self,
        timestamp_sec: float,
        trigger_values: np.ndarray,
        feature: tuple[np.ndarray, np.ndarray],
    ) -> None:
        if self.active:
            raise RuntimeError("held observation batch is still active")
        self._observation = (trigger_values, feature)
        self._timestamps = [float(timestamp_sec)]

    def append(self, timestamp_sec: float) -> None:
        if not self.active:
            return
        if len(self._timestamps) >= self.sample_count:
            raise RuntimeError("held observation batch exceeded its sample bound")
        self._timestamps.append(float(timestamp_sec))

    def take(
        self,
    ) -> tuple[tuple[float, ...], np.ndarray, tuple[np.ndarray, np.ndarray]] | None:
        if self._observation is None or not self._timestamps:
            return None
        trigger_values, feature = self._observation
        timestamps = tuple(self._timestamps)
        self.clear()
        return timestamps, trigger_values, feature

    def clear(self) -> None:
        self._observation = None
        self._timestamps.clear()


def _mirrored_display_landmarks(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    mirrored = np.asarray(landmarks, dtype=np.float32).copy()
    if mirrored.ndim != 2 or mirrored.shape[1] != 3:
        raise ValueError(f"display landmarks must be Nx3, got {mirrored.shape}")
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def _display_orientation(
    frame: np.ndarray,
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    *,
    display_mirror: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return display-only mirrored copies without touching detector/model data."""
    if not display_mirror:
        return frame, pose, left_hand, right_hand
    return (
        np.ascontiguousarray(np.flip(np.asarray(frame), axis=1)),
        _mirrored_display_landmarks(pose),
        _mirrored_display_landmarks(left_hand),
        _mirrored_display_landmarks(right_hand),
    )


def run_capture(
    bundle_dir: Path,
    *,
    mode: str,
    camera_index: int | None,
    video: Path | None,
    device: torch.device,
    headless: bool,
    max_frames: int | None,
    max_camera_index: int = 9,
    inference_stride: int = 8,
    recordings_dir: Path = Path("recordings"),
    start_logging: bool = False,
    rotation: RotationSetting = "auto",
    input_mirror: bool = False,
    display_mirror: bool = False,
) -> dict[str, Any]:
    import cv2

    orientation = InputOrientation(
        rotation=rotation,
        input_mirror=input_mirror,
        display_mirror=display_mirror,
    )
    verify_auto_trigger_provenance(Path(bundle_dir).resolve().parent)
    bundle = load_bundle(bundle_dir, device=device)
    source = (
        open_video(
            video,
            rotation=orientation.rotation,
            input_mirror=orientation.input_mirror,
            cv2_module=cv2,
        )
        if video is not None
        else open_camera(
            camera_index,
            max_index=max_camera_index,
            rotation=orientation.rotation,
            input_mirror=orientation.input_mirror,
            cv2_module=cv2,
        )
    )
    cleanup_stack = ExitStack()
    cleanup_stack.callback(source.release)
    try:
        source_input_mirror = getattr(source, "input_mirror", orientation.input_mirror)
        if type(source_input_mirror) is not bool:
            raise TypeError(
                f"source input_mirror must be bool, got {source_input_mirror!r}"
            )
        fallback_rotation = resolve_rotation(
            orientation.rotation,
            source_kind="video" if video is not None else "camera",
            metadata_rotation=0,
        )
        source_rotation_value = getattr(
            source,
            "resolved_rotation",
            getattr(source, "rotation_degrees", fallback_rotation),
        )
        if isinstance(source_rotation_value, bool):
            raise ValueError(
                "source resolved_rotation must be a right angle, "
                f"got {source_rotation_value!r}"
            )
        source_rotation_numeric = float(source_rotation_value)
        if source_rotation_numeric not in {0.0, 90.0, 180.0, 270.0}:
            raise ValueError(
                "source resolved_rotation must be a right angle, "
                f"got {source_rotation_value!r}"
            )
        source_resolved_rotation = int(source_rotation_numeric)
    except BaseException as exc:
        cleanup_stack.__exit__(type(exc), exc, exc.__traceback__)
        raise
    try:
        if mode in {"auto", "manual"}:
            trigger_config_path = bundle.root.parent / "auto_trigger_knee_ivcam_local.json"
            if not trigger_config_path.is_file():
                raise IntegrityError(f"auto-trigger config missing: {trigger_config_path}")
            controller: AutoKnee42Controller | SlidingController = AutoKnee42Controller(
                load_auto_trigger_config(trigger_config_path),
                initial_mode=mode,
            )
        else:
            controller = SlidingController(
                window=bundle.sequence_length,
                inference_stride=inference_stride,
            )
        if (
            headless
            and video is not None
            and isinstance(controller, AutoKnee42Controller)
            and controller.mode == "manual"
        ):
            controller.on_space()
    except BaseException as exc:
        cleanup_stack.__exit__(type(exc), exc, exc.__traceback__)
        raise

    last_result: InferenceResult | None = None
    raw_frames = 0
    feature_frames = 0
    observed_packet_eof = False
    first_frame_timestamp_sec: float | None = None
    last_frame_timestamp_sec: float | None = None
    recorder: SegmentSessionRecorder | None = None
    recording_requested = bool(start_logging)
    last_recording_summary = None
    latest_display_pose: np.ndarray | None = None
    latest_display_left_hand: np.ndarray | None = None
    latest_display_right_hand: np.ndarray | None = None
    display: ResizableDisplay | None = None
    try:
        held_batch = HeldObservationBatch(bundle.frame_step)
        started = time.perf_counter()
        if not headless:
            display = ResizableDisplay(
                "Knee42 IVCAM RGB",
                windows_primary_screen_size(),
                coverage=0.85,
            )
            cleanup_stack.callback(cv2.destroyAllWindows)
            display.create(cv2)
    except BaseException as exc:
        cleanup_stack.__exit__(type(exc), exc, exc.__traceback__)
        raise

    def stop_recording() -> None:
        nonlocal recorder, last_recording_summary, recording_requested
        recording_requested = False
        if recorder is not None:
            last_recording_summary = recorder.stop()
            recorder = None

    def stop_recording_on_exit(exc_type, _exc, _traceback):
        try:
            stop_recording()
        except BaseException:
            if exc_type is None:
                raise
        return False

    cleanup_stack.push(stop_recording_on_exit)

    def save_auto_result(event, result: InferenceResult) -> None:
        if recorder is not None and event.segment is not None:
            recorder.record_segment(event.segment, result)

    def handle_controller_event(event) -> None:
        nonlocal last_result
        if not event.infer:
            return
        last_result = _predict_features(bundle, event.features)
        save_auto_result(event, last_result)
        if isinstance(controller, AutoKnee42Controller) and controller.mode == "manual":
            controller.mark_result()

    def flush_pending_auto_observation():
        pending = held_batch.take()
        if pending is None:
            return None
        timestamps, trigger_values, feature = pending
        assert isinstance(controller, AutoKnee42Controller)
        return controller.add_held_observation_at_times(
            timestamps,
            trigger_values,
            feature,
        )

    try:
        with create_mediapipe_detectors(
            bundle,
            pixels_mirrored=source_input_mirror,
        ) as detectors:
            while max_frames is None or raw_frames < max_frames:
                packet = source.read_packet()
                if packet is None:
                    observed_packet_eof = True
                    break
                frame = packet.frame
                raw_frames += 1
                if first_frame_timestamp_sec is None:
                    first_frame_timestamp_sec = packet.timestamp_sec
                last_frame_timestamp_sec = packet.timestamp_sec
                source_fps = source.fps if source.fps > 0 else 30.0
                if recording_requested and recorder is None:
                    recorder = SegmentSessionRecorder(
                        recordings_dir,
                        fps=source_fps,
                        frame_size=(int(frame.shape[1]), int(frame.shape[0])),
                        source_origin_sec=packet.timestamp_sec,
                    )
                if recorder is not None:
                    recorder.add_frame(frame, timestamp_sec=packet.timestamp_sec)
                if (raw_frames - 1) % bundle.frame_step == 0:
                    observation = detectors.extract_observation(frame)
                    latest_display_pose = observation.display_pose
                    latest_display_left_hand = observation.display_left_hand
                    latest_display_right_hand = observation.display_right_hand
                    feature = (
                        observation.recognition_values,
                        observation.recognition_mask,
                    )
                    feature_frames += 1
                    if isinstance(controller, AutoKnee42Controller):
                        if controller.mode == "auto":
                            held_batch.start(
                                packet.timestamp_sec,
                                observation.trigger_values,
                                feature,
                            )
                        else:
                            event = controller.add_observation(
                                packet.timestamp_sec,
                                observation.trigger_values,
                                feature,
                            )
                    else:
                        event = controller.add_feature(feature)
                    if not (
                        isinstance(controller, AutoKnee42Controller)
                        and controller.mode == "auto"
                    ):
                        held_batch.clear()
                        handle_controller_event(event)
                elif (
                    isinstance(controller, AutoKnee42Controller)
                    and controller.mode == "auto"
                ):
                    held_batch.append(packet.timestamp_sec)
                if (
                    isinstance(controller, AutoKnee42Controller)
                    and controller.mode == "auto"
                    and held_batch.ready
                ):
                    event = flush_pending_auto_observation()
                    assert event is not None
                    handle_controller_event(event)
                elapsed = max(time.perf_counter() - started, 1e-6)
                fps = raw_frames / elapsed
                if not headless:
                    assert display is not None
                    display_mode = controller.mode if isinstance(controller, AutoKnee42Controller) else "sliding"
                    display_state = controller.state
                    if isinstance(controller, AutoKnee42Controller) and controller.mode == "auto":
                        display_state = (
                            "RESULT"
                            if last_result is not None and controller.state == "FORCED_FINALIZE_COOLDOWN"
                            else auto_display_state(controller.state, calibrated=controller.calibrated)
                        )
                    (
                        display_frame,
                        display_pose,
                        display_left_hand,
                        display_right_hand,
                    ) = _display_orientation(
                        frame,
                        latest_display_pose,
                        latest_display_left_hand,
                        latest_display_right_hand,
                        display_mirror=orientation.display_mirror,
                    )
                    view, _layout = render_application_view(
                        display_frame,
                        display.content_size(cv2),
                        build_display_panel_data(
                            last_result,
                            fps=fps,
                            source=source.status,
                            mode=display_mode,
                            state=display_state,
                            recording=recorder is not None,
                            recorded_segments=recorder.segment_count if recorder is not None else (
                                last_recording_summary.segment_count if last_recording_summary is not None else 0
                            ),
                            model_version=bundle.model_display_version,
                        ),
                        pose=display_pose,
                        left_hand=display_left_hand,
                        right_hand=display_right_hand,
                        cv2_module=cv2,
                    )
                    cv2.imshow("Knee42 IVCAM RGB", view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if key in (ord("f"), ord("F")):
                        display.toggle_fullscreen(cv2)
                    elif key in (ord("s"), ord("S")):
                        if recorder is not None or recording_requested:
                            stop_recording()
                        else:
                            if not isinstance(controller, AutoKnee42Controller) or controller.mode != "auto":
                                continue
                            controller.reset()
                            held_batch.clear()
                            last_result = None
                            recording_requested = True
                    elif key in (ord("r"), ord("R")):
                        controller.reset()
                        held_batch.clear()
                        last_result = None
                    elif key in (ord("m"), ord("M")) and isinstance(controller, AutoKnee42Controller):
                        controller.toggle_mode()
                        held_batch.clear()
                        last_result = None
                    elif key == 32 and isinstance(controller, AutoKnee42Controller):
                        event = controller.on_space()
                        if event.infer:
                            last_result = _predict_features(bundle, event.features)
                            controller.mark_result()
            if (
                observed_packet_eof
                and video is not None
                and isinstance(controller, AutoKnee42Controller)
                and controller.mode == "auto"
                and held_batch.active
            ):
                event = flush_pending_auto_observation()
                assert event is not None
                handle_controller_event(event)
            elif held_batch.active:
                held_batch.clear()
            if (
                observed_packet_eof
                and video is not None
                and isinstance(controller, AutoKnee42Controller)
                and controller.mode == "auto"
            ):
                event = controller.finalize_video_eof()
                if event.infer:
                    last_result = _predict_features(bundle, event.features)
                    save_auto_result(event, last_result)
            if headless and isinstance(controller, AutoKnee42Controller) and controller.mode == "manual" and controller.state == "recording":
                event = controller.on_space()
                if event.infer:
                    last_result = _predict_features(bundle, event.features)
                    controller.mark_result()
    except BaseException as exc:
        held_batch.clear()
        cleanup_stack.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        held_batch.clear()
        cleanup_stack.close()
    if last_result is None:
        mode_name = controller.mode if isinstance(controller, AutoKnee42Controller) else "sliding"
        state_name = controller.state
        calibrated = (
            controller.calibrated if isinstance(controller, AutoKnee42Controller) else "n/a"
        )
        raise RuntimeError(
            "capture completed without an inference result "
            f"(mode={mode_name}, state={state_name}, calibrated={calibrated})"
        )
    output = {
        "source": source.status,
        "clock_mode": source.clock_mode,
        "rotation": orientation.rotation,
        "resolved_rotation": source_resolved_rotation,
        "input_mirror": source_input_mirror,
        "display_mirror": orientation.display_mirror,
        "mode": controller.mode if isinstance(controller, AutoKnee42Controller) else "sliding",
        "raw_frames": raw_frames,
        "feature_frames": feature_frames,
        "first_frame_timestamp_sec": first_frame_timestamp_sec,
        "last_frame_timestamp_sec": last_frame_timestamp_sec,
        "top1": last_result.top1.label_id,
        "top1_raw_probability": last_result.top1.raw_probability,
        "top3": [item.label_id for item in last_result.top3],
        "probability_policy": probability_policy_record(),
        "device": str(device),
    }
    if last_recording_summary is not None:
        output["recording_session"] = str(last_recording_summary.session_dir.resolve())
        output["recorded_segments"] = last_recording_summary.segment_count
        output["recorded_frames"] = last_recording_summary.frame_count
    return output


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hash-verified Knee42 RGB IVCAM/video inference.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--mode", choices=("auto", "manual", "sliding"), default="auto")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--max-camera-index", type=int, default=9)
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--rotation",
        choices=("auto", "0", "90", "180", "270"),
        default="auto",
        help="model-input rotation (default: auto; video metadata, camera 0)",
    )
    parser.add_argument(
        "--input-mirror",
        choices=("off", "on"),
        default="off",
        help="mirror detector/model input pixels (default: off)",
    )
    parser.add_argument(
        "--display-mirror",
        choices=("off", "on"),
        default="off",
        help="mirror only the rendered operator view (default: off)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--inference-stride", type=int, default=8)
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--start-logging", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _device_from_name(args.device)
    if args.self_test:
        result = run_self_test(args.bundle, device=device)
    else:
        result = run_capture(
            args.bundle,
            mode=args.mode,
            camera_index=args.camera_index,
            video=args.video,
            device=device,
            headless=args.headless,
            max_frames=args.max_frames,
            max_camera_index=args.max_camera_index,
            inference_stride=args.inference_stride,
            recordings_dir=args.recordings_dir,
            start_logging=args.start_logging,
            rotation=parse_rotation(args.rotation),
            input_mirror=MirrorMode.parse(args.input_mirror).enabled,
            display_mirror=MirrorMode.parse(args.display_mirror).enabled,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
