from __future__ import annotations

import argparse
import json
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

from recognition.config import preview_paths
from recognition.evaluation.team_test_report import export_team_test_reports
from recognition.evaluation.team_test_session import (
    NO_DETECTION_LABEL,
    TEAM_PHASE_ARMED,
    TEAM_PHASE_COMPLETE,
    TEAM_PHASE_READY,
    TEAM_PHASE_REVIEW,
    TeamTestSession,
    TeamTestWorkflow,
    sha256_file,
    validate_team_test_labels,
)
from recognition.inference.daily30_sentence_feature_utils import build_feature_sequence
from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier
from recognition.inference.daily30_sentence_realtime_utils import (
    DEFAULT_CACHE_DIR,
    ensure_artifacts_cached,
    load_runtime_bundle,
    save_prediction_logs,
)
from recognition.inference.extract_daily30_sentence_features import (
    extract_frame_vector,
    normalize_relative_frames,
    resize_seq,
)
from recognition.realtime.auto_trigger import (
    SEGMENT_STATE_ACTIVE,
    SEGMENT_STATE_COOLDOWN,
    SEGMENT_STATE_END_CONFIRM,
    SEGMENT_STATE_IDLE,
    AutoFrameAnalysis,
    AutoTriggerConfig,
    AutoTriggerEngine,
    analyze_frame_vector,
    load_auto_trigger_config,
)
from recognition.realtime.personal_temporal import (
    PersonalTemporalModel,
    PersonalTemporalPredictor,
    with_temporal_probability,
)
from recognition.realtime.probability_reporting import (
    PROBABILITY_POLICY,
    probability_policy_record,
    validate_raw_probability,
)


PATHS = preview_paths()
ROOT = PATHS.repo_root
RESULTS_DIR = PATHS.results_dir
HAND_MODEL = PATHS.hand_model
POSE_MODEL = PATHS.pose_model
WAITING_LABEL = "等待開始"
SHORT_SEGMENT_LABEL = "片段過短"
AUTO_MODE_LABEL = "自動切段"
MANUAL_MODE_LABEL = "手動切段"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msjh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\mingliu.ttc"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_BODY = load_font(13)
FONT_HINT = load_font(11)
FONT_RESULT = load_font(24)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], font, color: tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def calibrate_confidence(raw_probability: float) -> float:
    """Deprecated compatibility shim returning the exact raw probability."""
    return validate_raw_probability(raw_probability)


def format_raw_probability_percent(raw_probability: float) -> str:
    return f"{validate_raw_probability(raw_probability):.1%}"


def format_confidence_percent(confidence: float) -> str:
    """Deprecated display alias; formats the exact raw probability."""
    warnings.warn(
        "format_confidence_percent is deprecated; use format_raw_probability_percent",
        DeprecationWarning,
        stacklevel=2,
    )
    return format_raw_probability_percent(confidence)


def decide_prediction_output(
    localized_label: str,
    raw_probability: float,
    _legacy_threshold: float | None = None,
) -> tuple[str, float, float]:
    """Compatibility tuple with acceptance disabled and no score transform."""
    probability = validate_raw_probability(raw_probability)
    return localized_label, probability, probability


def localize_label(label: str, label_display: dict[str, str]) -> str:
    return str(label_display.get(label, label))


def localize_top3_probabilities(
    top3_candidates: list[tuple[str, float]],
    label_display: dict[str, str],
) -> list[tuple[str, float]]:
    return [
        (localize_label(label, label_display), validate_raw_probability(score))
        for label, score in top3_candidates
    ]


def build_probability_log_fields(
    raw_probability: float,
    top3_candidates: list[tuple[str, float]],
) -> dict[str, object]:
    """Build machine-readable probability fields without presentation rounding."""
    return {
        "raw_probability": validate_raw_probability(raw_probability),
        "probability_kind": PROBABILITY_POLICY.kind,
        "acceptance_policy": PROBABILITY_POLICY.acceptance_policy,
        "calibration_artifact": PROBABILITY_POLICY.calibration_artifact,
        "top3_candidates": [
            {
                "label": label,
                "raw_probability": validate_raw_probability(probability),
            }
            for label, probability in top3_candidates
        ],
    }


@dataclass(frozen=True)
class SegmentPrediction:
    predicted_label_id: str
    predicted_text: str
    display_label: str
    raw_probability: float
    top3_label_probabilities: list[tuple[str, float]]
    top3_display_probabilities: list[tuple[str, float]]


def decode_segment_prediction(
    probs: np.ndarray,
    labels: list[str],
    label_display: dict[str, str],
    _legacy_acceptance_threshold: float | None = None,
) -> SegmentPrediction:
    top3_label_probabilities = decode_top3_candidates(probs, labels)
    top3_display_probabilities = localize_top3_probabilities(
        top3_label_probabilities,
        label_display,
    )
    pred_idx = int(np.argmax(probs))
    predicted_label_id = str(labels[pred_idx])
    predicted_text = localize_label(predicted_label_id, label_display)
    raw_probability = validate_raw_probability(float(probs[pred_idx]))
    return SegmentPrediction(
        predicted_label_id=predicted_label_id,
        predicted_text=predicted_text,
        display_label=predicted_text,
        raw_probability=raw_probability,
        top3_label_probabilities=top3_label_probabilities,
        top3_display_probabilities=top3_display_probabilities,
    )


def draw_landmark_points(frame: np.ndarray, hand_result, pose_result) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    pose_landmarks = getattr(pose_result, "pose_landmarks", []) if pose_result is not None else []
    if pose_landmarks:
        for lm in pose_landmarks[0]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)
    handedness_list = getattr(hand_result, "handedness", []) if hand_result is not None else []
    hand_landmarks_list = getattr(hand_result, "hand_landmarks", []) if hand_result is not None else []
    for handedness, landmarks in zip(handedness_list, hand_landmarks_list):
        color = (0, 255, 0) if handedness[0].category_name.lower() == "left" else (255, 0, 0)
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), 4, color, -1)
        for a, b in [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ]:
            if a < len(points) and b < len(points):
                cv2.line(canvas, points[a], points[b], color, 2)
    return canvas


def localize_segment_state(segment_state: str, trigger_mode: str) -> str:
    if trigger_mode == "manual":
        return "錄製中" if segment_state == SEGMENT_STATE_ACTIVE else "等待開始"
    if segment_state == SEGMENT_STATE_ACTIVE:
        return "錄製中"
    if segment_state == SEGMENT_STATE_END_CONFIRM:
        return "句尾確認中"
    if segment_state == SEGMENT_STATE_COOLDOWN:
        return "自動收尾"
    return "等待開始"


def operation_hint(
    trigger_mode: str,
    segment_state: str,
    has_result: bool,
) -> str:
    base = "操作：Q 離開、R 重設、S 存檔"
    if trigger_mode == "manual":
        return f"{base}、Space 手動開始/結束"
    if segment_state == SEGMENT_STATE_ACTIVE:
        return f"{base}｜辨識中，完成後請將雙手放回身側"
    if segment_state == SEGMENT_STATE_END_CONFIRM:
        return f"{base}｜請保持站立，雙手自然垂放身側"
    if segment_state == SEGMENT_STATE_COOLDOWN:
        return f"{base}｜本句完成，準備辨識下一句"
    if has_result:
        return f"{base}｜請保持站立，可開始下一句"
    return f"{base}｜請站立，雙手自然垂放身側後開始"


def draw_overlay(
    frame: np.ndarray,
    display_label: str,
    top3_candidates: list[tuple[str, float]],
    emitted_labels: list[str],
    fps_value: float,
    mode_label: str,
    segment_status: str,
    segment_state: str,
    trigger_mode: str,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (12, 12), (760, 152), (0, 0, 0), -1)
    cv2.rectangle(canvas, (12, h - 56), (min(w - 12, 1060), h - 12), (0, 0, 0), -1)
    canvas = draw_text(canvas, f"模式：{mode_label}", (26, 18), FONT_HINT, (170, 210, 255))
    canvas = draw_text(canvas, f"片段狀態：{segment_status}", (190, 18), FONT_HINT, (255, 220, 120))
    canvas = draw_text(canvas, f"目前辨識：{display_label}", (26, 40), FONT_RESULT, (0, 255, 0))
    if top3_candidates:
        candidate_text = "候選結果（raw probability）：" + " | ".join(
            f"{label} {format_raw_probability_percent(score)}" for label, score in top3_candidates
        )
    else:
        candidate_text = "候選結果：-"
    canvas = draw_text(canvas, candidate_text, (26, 92), FONT_BODY, (255, 220, 120))
    canvas = draw_text(canvas, f"FPS：{fps_value:.1f}", (26, 120), FONT_HINT, (220, 220, 220))
    canvas = draw_text(
        canvas,
        operation_hint(trigger_mode, segment_state, bool(emitted_labels)),
        (150, 120),
        FONT_HINT,
        (180, 180, 180),
    )
    history_text = "已辨識句子：" + (" | ".join(emitted_labels[-6:]) if emitted_labels else "-")
    canvas = draw_text(canvas, history_text, (24, h - 42), FONT_BODY, (255, 255, 255))
    return canvas


def draw_team_test_overlay(
    frame: np.ndarray,
    workflow: TeamTestWorkflow,
) -> np.ndarray:
    canvas = frame.copy()
    session = workflow.session
    _, w = canvas.shape[:2]
    panel_right = min(w - 12, 920)
    cv2.rectangle(canvas, (12, 160), (panel_right, 286), (0, 0, 0), -1)
    canvas = draw_text(
        canvas,
        f"組員測試：{session.tester_id}　進度 {session.completed_trials}/{session.total_trials}",
        (26, 170),
        FONT_BODY,
        (180, 220, 255),
    )
    if workflow.phase == TEAM_PHASE_COMPLETE:
        canvas = draw_text(
            canvas,
            "測試完成，請按 Q 離開後執行 package_team_results.cmd",
            (26, 205),
            FONT_BODY,
            (80, 255, 120),
        )
        return canvas

    expected = session.current_expected
    canvas = draw_text(
        canvas,
        f"現在請做：{expected.label_id} {expected.label_text}　（第 {expected.trial_number}/{session.trials_per_label} 次）",
        (26, 200),
        FONT_BODY,
        (255, 255, 255),
    )
    if workflow.phase == TEAM_PHASE_READY:
        instruction = "保持站姿待機，按 Enter 開始本次測試"
        color = (255, 220, 120)
    elif workflow.phase == TEAM_PHASE_ARMED:
        instruction = "已開始：請完成手語並回到待機姿勢；完全沒觸發按 N，自己做錯按 R"
        color = (120, 255, 255)
    else:
        pending = session.pending_result
        result = "正確" if pending and pending.top1_correct else "錯誤"
        instruction = f"本次結果：{result}　按 Enter 確認，或按 R 作廢重做"
        color = (80, 255, 120) if pending and pending.top1_correct else (100, 100, 255)
    canvas = draw_text(canvas, instruction, (26, 238), FONT_BODY, color)
    return canvas


def prepare_detection_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime sentence inference for daily30 BiGRU model.")
    parser.add_argument("--app-config", default="", help="Local JSON settings file. Explicit CLI options take priority.")
    parser.add_argument("--source", default=None, help="Camera index like 0/1/2 or a video path.")
    parser.add_argument("--backend", choices=["auto", "dshow"], default=None)
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--save-log", dest="save_log", action="store_true")
    parser.add_argument("--no-save-log", dest="save_log", action="store_false")
    parser.set_defaults(save_log=None)
    parser.add_argument(
        "--min-conf-override",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option. Acceptance threshold is unavailable "
            "without calibration/risk-coverage evidence."
        ),
    )
    parser.add_argument("--trigger-mode", choices=["auto", "manual"], default=None)
    parser.add_argument("--auto-config", default="", help="JSON file containing calibrated auto-trigger settings.")
    parser.add_argument(
        "--temporal-model",
        default="",
        help="Optional personalized idle/signing model. Defaults to models/personal_temporal_knee_v1.npz when enabled.",
    )
    parser.add_argument("--start-motion-threshold", type=float, default=None)
    parser.add_argument("--blank-motion-threshold", type=float, default=None)
    parser.add_argument("--start-hold-sec", type=float, default=None)
    parser.add_argument("--end-hold-sec", type=float, default=None)
    parser.add_argument("--end-rest-vote-ratio", type=float, default=None)
    parser.add_argument("--pre-roll-sec", type=float, default=None)
    parser.add_argument("--max-segment-sec", type=float, default=None)
    parser.add_argument("--min-segment-sec", type=float, default=None)
    parser.add_argument("--cooldown-sec", type=float, default=None)
    parser.add_argument("--torso-motion-weight", type=float, default=None)
    parser.add_argument("--knee-lateral-thigh-margin-ratio", type=float, default=None)
    parser.add_argument("--knee-min-thigh-progress-ratio", type=float, default=None)
    parser.add_argument("--knee-max-thigh-progress-ratio", type=float, default=None)
    parser.add_argument("--reference-rest-distance-threshold", type=float, default=None)
    parser.add_argument("--pose-visibility-threshold", type=float, default=None)
    parser.add_argument("--hidden-rest-enabled", dest="hidden_rest_enabled", action="store_true")
    parser.add_argument("--no-hidden-rest-enabled", dest="hidden_rest_enabled", action="store_false")
    parser.set_defaults(hidden_rest_enabled=None)
    parser.add_argument("--detector-frame-skip", type=int, default=2)
    parser.add_argument("--inference-max-width", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop automatically after N frames. 0 disables auto-stop.")
    parser.add_argument("--team-test", action="store_true", help="Run the guided 27-class teammate evaluation workflow.")
    parser.add_argument("--tester-id", default="", help="Pseudonymous teammate identifier used for resumable results.")
    parser.add_argument("--trials-per-label", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume compatible saved team-test progress.")
    return parser.parse_args(argv)


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    config_path = (
        Path(args.app_config).expanduser()
        if args.app_config
        else PATHS.app_config_path
    )
    config: dict[str, object] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("app_config.json must contain a JSON object.")
        allowed = {"source", "backend", "trigger_mode", "save_log"}
        unexpected = sorted(set(loaded) - allowed)
        if unexpected:
            raise ValueError(
                "Unsupported app_config.json keys: " + ", ".join(unexpected)
            )
        config = loaded

    if args.source is None:
        args.source = str(config.get("source", "0"))
    if args.backend is None:
        args.backend = str(config.get("backend", "dshow"))
    if args.trigger_mode is None:
        args.trigger_mode = str(config.get("trigger_mode", "auto"))
    if args.save_log is None:
        args.save_log = bool(config.get("save_log", True))
    if args.backend not in {"auto", "dshow"}:
        raise ValueError("backend must be 'auto' or 'dshow'.")
    if args.trigger_mode not in {"auto", "manual"}:
        raise ValueError("trigger_mode must be 'auto' or 'manual'.")
    if args.min_conf_override is not None:
        if not np.isfinite(args.min_conf_override):
            raise ValueError("--min-conf-override must be finite")
        if args.min_conf_override >= 0.0:
            raise ValueError(
                "acceptance threshold is unavailable without "
                "calibration/risk-coverage evidence"
            )
    args.min_conf_override = None
    if args.team_test:
        args.trigger_mode = "auto"
        args.save_log = False
        if args.trials_per_label <= 0:
            raise ValueError("trials_per_label must be positive")
    return args


def open_capture(source: str, backend: str) -> cv2.VideoCapture:
    if source.isdigit():
        index = int(source)
        if backend == "dshow":
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(source)


def is_file_source(source: str) -> bool:
    return not source.isdigit()


def seek_video_capture(cap: cv2.VideoCapture, seconds_delta: float) -> bool:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        return False
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    target_frame = current_frame + int(round(seconds_delta * fps))
    target_frame = max(0, min(frame_count - 1, target_frame))
    return bool(cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame))


def build_sequence_feature(raw_frame_buffer: deque[np.ndarray], append_delta: bool, zscore_features: bool) -> np.ndarray:
    raw_frames = np.stack(list(raw_frame_buffer), axis=0)
    rel_frames = normalize_relative_frames(raw_frames)
    return build_feature_sequence(rel_frames, append_delta=append_delta, zscore_features=zscore_features)


def build_sequence_feature_from_list(frame_vectors: list[np.ndarray], sequence_length: int, append_delta: bool, zscore_features: bool) -> np.ndarray | None:
    if not frame_vectors:
        return None
    raw_frames = np.stack(frame_vectors, axis=0)
    rel_frames = normalize_relative_frames(raw_frames)
    resized = resize_seq(rel_frames, sequence_length)
    return build_feature_sequence(resized, append_delta=append_delta, zscore_features=zscore_features)


def decode_top3_candidates(probs: np.ndarray, labels: list[str]) -> list[tuple[str, float]]:
    if probs.size == 0:
        return []
    topk = min(3, int(probs.shape[0]))
    indices = np.argsort(probs)[-3:][::-1]
    return [
        (str(labels[int(idx)]), validate_raw_probability(float(probs[int(idx)])))
        for idx in indices[:topk]
    ]


def select_torch_device(preferred_device: str | None = None) -> torch.device:
    preferred = (preferred_device or "auto").strip().lower()
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_auto_trigger_config(args: argparse.Namespace) -> AutoTriggerConfig:
    auto_config_path: str | Path | None = args.auto_config or None
    if auto_config_path is None:
        knee_config = Path(__file__).resolve().parents[2] / "configs" / "auto_trigger_knee_v1.json"
        if knee_config.exists():
            auto_config_path = knee_config
    return load_auto_trigger_config(
        auto_config_path,
        overrides={
            "start_motion_threshold": args.start_motion_threshold,
            "blank_motion_threshold": args.blank_motion_threshold,
            "start_hold_sec": args.start_hold_sec,
            "end_hold_sec": args.end_hold_sec,
            "end_rest_vote_ratio": args.end_rest_vote_ratio,
            "pre_roll_sec": args.pre_roll_sec,
            "max_segment_sec": args.max_segment_sec,
            "min_segment_sec": args.min_segment_sec,
            "cooldown_sec": args.cooldown_sec,
            "torso_motion_weight": args.torso_motion_weight,
            "hidden_rest_enabled": args.hidden_rest_enabled,
            "knee_lateral_thigh_margin_ratio": args.knee_lateral_thigh_margin_ratio,
            "knee_min_thigh_progress_ratio": args.knee_min_thigh_progress_ratio,
            "knee_max_thigh_progress_ratio": args.knee_max_thigh_progress_ratio,
            "reference_rest_distance_threshold": args.reference_rest_distance_threshold,
            "pose_visibility_threshold": args.pose_visibility_threshold,
        },
    )


def load_personal_temporal_model(args: argparse.Namespace, config: AutoTriggerConfig) -> PersonalTemporalModel | None:
    """Load the optional gate without preventing the safe rule-only fallback."""
    if not config.temporal_classifier_enabled:
        return None
    model_path = Path(args.temporal_model) if args.temporal_model else ROOT / "models" / "personal_temporal_knee_v1.npz"
    if not model_path.exists():
        print(f"Personal temporal model unavailable ({model_path}); using calibrated rule-only fallback.")
        return None
    try:
        return PersonalTemporalModel.load(model_path)
    except (OSError, ValueError) as error:
        print(f"Could not load personal temporal model ({error}); using calibrated rule-only fallback.")
        return None


def reset_runtime_state(emitted_labels: list[str]) -> None:
    emitted_labels.clear()


def infer_segment_prediction(
    frame_vectors: list[np.ndarray],
    sequence_length: int,
    append_delta: bool,
    zscore_features: bool,
    model: BiGRUSentenceClassifier,
    device: torch.device,
    labels: list[str],
    label_display: dict[str, str],
    _legacy_acceptance_threshold: float | None = None,
) -> tuple[str, float, list[tuple[str, float]]]:
    prediction = infer_segment_prediction_details(
        frame_vectors,
        sequence_length,
        append_delta,
        zscore_features,
        model,
        device,
        labels,
        label_display,
        _legacy_acceptance_threshold,
    )
    if prediction is None:
        return SHORT_SEGMENT_LABEL, 0.0, []
    return (
        prediction.display_label,
        prediction.raw_probability,
        prediction.top3_display_probabilities,
    )


def infer_segment_prediction_details(
    frame_vectors: list[np.ndarray],
    sequence_length: int,
    append_delta: bool,
    zscore_features: bool,
    model: BiGRUSentenceClassifier,
    device: torch.device,
    labels: list[str],
    label_display: dict[str, str],
    _legacy_acceptance_threshold: float | None = None,
) -> SegmentPrediction | None:
    seq_array = build_sequence_feature_from_list(frame_vectors, sequence_length, append_delta, zscore_features)
    if seq_array is None:
        return None

    seq = torch.tensor(seq_array, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(seq), dim=1)[0].cpu().numpy()
    return decode_segment_prediction(
        probs,
        labels,
        label_display,
        _legacy_acceptance_threshold,
    )


def main() -> None:
    args = resolve_runtime_args(parse_args())
    cache_dir = ensure_artifacts_cached(Path(args.model_cache_dir))
    bundle = load_runtime_bundle(cache_dir)
    device = select_torch_device(bundle.get("device"))

    input_dim = int(
        build_sequence_feature(
            deque([np.zeros(225, dtype=np.float32) for _ in range(int(bundle["sequence_length"]))], maxlen=int(bundle["sequence_length"])),
            bundle["append_delta"],
            bundle["zscore_features"],
        ).shape[-1]
    )
    model = BiGRUSentenceClassifier(
        input_dim=input_dim,
        hidden_size=int(bundle["hidden_size"]),
        num_layers=int(bundle["num_layers"]),
        dropout=float(bundle["dropout"]),
        num_classes=len(bundle["labels"]),
        pooling=str(bundle["pooling"]),
    )
    model.load_state_dict(torch.load(bundle["paths"]["best_model"], map_location=device))
    model.to(device)
    model.eval()

    cap = open_capture(str(args.source), args.backend)
    file_source_mode = is_file_source(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0

    window_name = "Realtime Sentence Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    auto_config = build_auto_trigger_config(args)
    temporal_model = load_personal_temporal_model(args, auto_config)

    def new_auto_trigger() -> tuple[AutoTriggerEngine, PersonalTemporalPredictor | None]:
        predictor = PersonalTemporalPredictor(temporal_model, auto_config) if temporal_model is not None else None
        return AutoTriggerEngine(auto_config), predictor

    auto_engine, temporal_predictor = new_auto_trigger()
    team_session: TeamTestSession | None = None
    team_workflow: TeamTestWorkflow | None = None
    team_detection_start_frame = 0
    if args.team_test:
        if not args.tester_id.strip():
            raise ValueError("--tester-id is required when --team-test is enabled")
        labels = validate_team_test_labels(bundle["labels"])
        team_session = TeamTestSession(
            output_dir=RESULTS_DIR,
            tester_id=args.tester_id,
            labels=labels,
            label_display=bundle["label_display"],
            trials_per_label=int(args.trials_per_label),
            model_version=str(bundle["run_name"]),
            runtime_metadata={
                "model_sha256": sha256_file(bundle["paths"]["best_model"]),
                "auto_trigger": asdict(auto_config),
                "temporal_model_loaded": temporal_model is not None,
                "source": str(args.source),
                "backend": str(args.backend),
            },
            resume=bool(args.resume),
        )
        team_workflow = TeamTestWorkflow(team_session)
    emitted_labels: list[str] = []
    prediction_log: list[dict] = []
    last_hand_result = None
    last_pose_result = None
    previous_frame_vector: np.ndarray | None = None
    frame_index = 0
    last_tick = cv2.getTickCount()
    fps_value = 0.0
    top3_candidates: list[tuple[str, float]] = []
    manual_active = False
    manual_frame_vectors: list[np.ndarray] = []
    reset_events = 0
    saved_events = 0
    display_label = WAITING_LABEL
    display_raw_probability = 0.0
    last_finalize_reason = ""
    last_clip_start_sec: float | None = None
    last_clip_end_sec: float | None = None
    last_finalize_sec: float | None = None
    latest_analysis = AutoFrameAnalysis(
        visible_rest_blank=False,
        hidden_rest_blank=False,
        torso_motion_score=0.0,
        hand_motion_score=0.0,
        effective_motion_score=0.0,
        hands_on_knees=False,
        knee_landmarks_valid=False,
        wrists_detected=False,
        torso_valid=False,
        explicit_hands_detected=0,
        wrist_source_left="none",
        wrist_source_right="none",
    )

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                if file_source_mode:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break
                else:
                    break

            should_run_detector = (
                frame_index % max(1, int(bundle["frame_step"])) == 0
                and frame_index % max(1, int(args.detector_frame_skip)) == 0
            )
            if should_run_detector:
                detection_frame = prepare_detection_frame(frame, int(args.inference_max_width))
                rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_result = hand_landmarker.detect(mp_image)
                pose_result = pose_landmarker.detect(mp_image)
                last_hand_result = hand_result
                last_pose_result = pose_result
                current_frame_vector = extract_frame_vector(hand_result, pose_result)
                sample_timestamp_sec = frame_index / source_fps

                if args.trigger_mode == "manual":
                    if manual_active:
                        manual_frame_vectors.append(current_frame_vector)
                    latest_analysis = analyze_frame_vector(
                        previous_frame_vector,
                        current_frame_vector,
                        auto_config,
                    )
                    if not manual_active and display_label == WAITING_LABEL:
                        top3_candidates = []
                else:
                    latest_analysis = analyze_frame_vector(
                        previous_frame_vector,
                        current_frame_vector,
                        auto_config,
                    )
                    if temporal_predictor is not None:
                        probability = temporal_predictor.update(sample_timestamp_sec, latest_analysis)
                        latest_analysis = with_temporal_probability(latest_analysis, probability)
                    segment = None
                    if (
                        team_workflow is None
                        or (
                            team_workflow.phase == TEAM_PHASE_ARMED
                            and frame_index >= team_detection_start_frame
                        )
                    ):
                        segment = auto_engine.update(
                            current_frame_vector,
                            latest_analysis,
                            sample_timestamp_sec,
                        )
                    if segment is not None:
                        last_finalize_reason = segment.reason
                        last_clip_start_sec = segment.clip_start_sec
                        last_clip_end_sec = segment.clip_end_sec
                        last_finalize_sec = segment.finalize_sec
                        if segment.duration_sec < auto_config.min_segment_sec:
                            display_label = SHORT_SEGMENT_LABEL
                            display_raw_probability = 0.0
                            top3_candidates = []
                            if team_workflow is not None:
                                team_workflow.stage_prediction(
                                    predicted_label=SHORT_SEGMENT_LABEL,
                                    raw_probability=0.0,
                                    top3_candidates=[],
                                    clip_start_sec=segment.clip_start_sec,
                                    clip_end_sec=segment.clip_end_sec,
                                    finalize_sec=segment.finalize_sec,
                                    finalize_reason=segment.reason,
                                    outcome="short_segment",
                                )
                        else:
                            if team_workflow is not None:
                                detail = infer_segment_prediction_details(
                                    segment.frame_vectors,
                                    int(bundle["sequence_length"]),
                                    bool(bundle["append_delta"]),
                                    bool(bundle["zscore_features"]),
                                    model,
                                    device,
                                    bundle["labels"],
                                    bundle["label_display"],
                                )
                                if detail is None:
                                    raise RuntimeError("completed segment did not contain frame features")
                                display_label = detail.display_label
                                display_raw_probability = detail.raw_probability
                                top3_candidates = detail.top3_display_probabilities
                                team_workflow.stage_prediction(
                                    predicted_label=detail.predicted_label_id,
                                    raw_probability=detail.raw_probability,
                                    top3_candidates=detail.top3_label_probabilities,
                                    clip_start_sec=segment.clip_start_sec,
                                    clip_end_sec=segment.clip_end_sec,
                                    finalize_sec=segment.finalize_sec,
                                    finalize_reason=segment.reason,
                                )
                            else:
                                display_label, display_raw_probability, top3_candidates = infer_segment_prediction(
                                    segment.frame_vectors,
                                    int(bundle["sequence_length"]),
                                    bool(bundle["append_delta"]),
                                    bool(bundle["zscore_features"]),
                                    model,
                                    device,
                                    bundle["labels"],
                                    bundle["label_display"],
                                )
                            if display_label != SHORT_SEGMENT_LABEL:
                                emitted_labels.append(display_label)
                    elif auto_engine.state == SEGMENT_STATE_IDLE and latest_analysis.is_blank and not emitted_labels:
                        display_label = WAITING_LABEL

                previous_frame_vector = current_frame_vector

            if not args.team_test:
                prediction_log.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "frame_index": frame_index,
                        "trigger_mode": args.trigger_mode,
                        "manual_active": manual_active,
                        "predicted_label": display_label,
                        "display_label": display_label,
                        **build_probability_log_fields(
                            display_raw_probability,
                            top3_candidates,
                        ),
                        "segment_state": auto_engine.state if args.trigger_mode == "auto" else ("SIGNING_ACTIVE" if manual_active else "IDLE_BLANK"),
                        "segment_status": localize_segment_state(
                            auto_engine.state if args.trigger_mode == "auto" else (SEGMENT_STATE_ACTIVE if manual_active else SEGMENT_STATE_IDLE),
                            args.trigger_mode,
                        ),
                        "segment_frame_count": len(auto_engine.segment_samples) if args.trigger_mode == "auto" else len(manual_frame_vectors),
                        "segment_finalize_reason": last_finalize_reason,
                        "clip_start_sec": "" if last_clip_start_sec is None else float(last_clip_start_sec),
                        "clip_end_sec": "" if last_clip_end_sec is None else float(last_clip_end_sec),
                        "finalize_sec": "" if last_finalize_sec is None else float(last_finalize_sec),
                        "is_blank": latest_analysis.is_blank,
                        "visible_rest_blank": latest_analysis.visible_rest_blank,
                        "hidden_rest_blank": latest_analysis.hidden_rest_blank,
                        "torso_motion_score": float(latest_analysis.torso_motion_score),
                        "hand_motion_score": float(latest_analysis.hand_motion_score),
                        "effective_motion_score": float(latest_analysis.effective_motion_score),
                        "hands_on_knees": latest_analysis.hands_on_knees,
                        "wrists_detected": latest_analysis.wrists_detected,
                        "torso_valid": latest_analysis.torso_valid,
                        "explicit_hands_detected": latest_analysis.explicit_hands_detected,
                        "wrist_source_left": latest_analysis.wrist_source_left,
                        "wrist_source_right": latest_analysis.wrist_source_right,
                        "temporal_active_probability": "" if latest_analysis.temporal_active_probability is None else float(latest_analysis.temporal_active_probability),
                        "emitted_count": len(emitted_labels),
                    }
                )

            now_tick = cv2.getTickCount()
            elapsed = (now_tick - last_tick) / cv2.getTickFrequency()
            if elapsed > 0:
                fps_value = 1.0 / elapsed
            last_tick = now_tick

            landmark_view = draw_landmark_points(frame, last_hand_result, last_pose_result)
            overlay = draw_overlay(
                landmark_view,
                display_label,
                top3_candidates,
                emitted_labels,
                fps_value,
                AUTO_MODE_LABEL if args.trigger_mode == "auto" else MANUAL_MODE_LABEL,
                localize_segment_state(
                    auto_engine.state if args.trigger_mode == "auto" else (SEGMENT_STATE_ACTIVE if manual_active else SEGMENT_STATE_IDLE),
                    args.trigger_mode,
                ),
                auto_engine.state if args.trigger_mode == "auto" else (SEGMENT_STATE_ACTIVE if manual_active else SEGMENT_STATE_IDLE),
                args.trigger_mode,
            )
            if team_workflow is not None:
                overlay = draw_team_test_overlay(overlay, team_workflow)
            cv2.imshow(window_name, overlay)
            key = cv2.waitKeyEx(1)
            if key in {27, ord("q"), ord("Q")}:
                break
            if team_workflow is not None:
                if key in {13, 10}:
                    confirmed = team_workflow.press_enter()
                    if team_workflow.phase == TEAM_PHASE_ARMED:
                        team_detection_start_frame = frame_index + max(
                            1, int(round(source_fps))
                        )
                    if confirmed is not None:
                        export_team_test_reports(team_session)
                    auto_engine, temporal_predictor = new_auto_trigger()
                    previous_frame_vector = None
                    top3_candidates = []
                    display_label = WAITING_LABEL
                    display_raw_probability = 0.0
                    last_finalize_reason = ""
                    last_clip_start_sec = None
                    last_clip_end_sec = None
                    last_finalize_sec = None
                elif key in {ord("r"), ord("R")}:
                    team_workflow.press_retry()
                    auto_engine, temporal_predictor = new_auto_trigger()
                    previous_frame_vector = None
                    top3_candidates = []
                    display_label = WAITING_LABEL
                    display_raw_probability = 0.0
                    last_finalize_reason = ""
                    last_clip_start_sec = None
                    last_clip_end_sec = None
                    last_finalize_sec = None
                elif (
                    key in {ord("n"), ord("N")}
                    and team_workflow.phase == TEAM_PHASE_ARMED
                    and frame_index >= team_detection_start_frame
                    and auto_engine.state == SEGMENT_STATE_IDLE
                ):
                    team_workflow.press_no_detection()
                    display_label = NO_DETECTION_LABEL
                    display_raw_probability = 0.0
                    top3_candidates = []
                    auto_engine, temporal_predictor = new_auto_trigger()
                frame_index += 1
                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
                continue
            if key in {ord("r"), ord("R")}:
                reset_runtime_state(emitted_labels)
                auto_engine, temporal_predictor = new_auto_trigger()
                manual_active = False
                manual_frame_vectors = []
                previous_frame_vector = None
                top3_candidates = []
                display_label = WAITING_LABEL
                display_raw_probability = 0.0
                last_finalize_reason = ""
                last_clip_start_sec = None
                last_clip_end_sec = None
                last_finalize_sec = None
                reset_events += 1
            if file_source_mode and key in {2490368, 2621440}:
                seek_seconds = 5.0 if key == 2490368 else -5.0
                if seek_video_capture(cap, seek_seconds):
                    auto_engine, temporal_predictor = new_auto_trigger()
                    manual_active = False
                    manual_frame_vectors = []
                    previous_frame_vector = None
                    top3_candidates = []
                    display_label = WAITING_LABEL
                    display_raw_probability = 0.0
                    last_finalize_reason = ""
                    last_clip_start_sec = None
                    last_clip_end_sec = None
                    last_finalize_sec = None
                continue
            if key == ord(" "):
                if args.trigger_mode == "manual":
                    if not manual_active:
                        manual_active = True
                        manual_frame_vectors = []
                        display_label = WAITING_LABEL
                        top3_candidates = []
                    else:
                        manual_active = False
                        manual_duration_sec = (
                            len(manual_frame_vectors)
                            * max(1, int(args.detector_frame_skip))
                            / source_fps
                        )
                        if manual_duration_sec < auto_config.min_segment_sec:
                            display_label = SHORT_SEGMENT_LABEL
                            display_raw_probability = 0.0
                            top3_candidates = []
                        else:
                            display_label, display_raw_probability, top3_candidates = infer_segment_prediction(
                                manual_frame_vectors,
                                int(bundle["sequence_length"]),
                                bool(bundle["append_delta"]),
                                bool(bundle["zscore_features"]),
                                model,
                                device,
                                bundle["labels"],
                                bundle["label_display"],
                            )
                            if display_label != SHORT_SEGMENT_LABEL:
                                emitted_labels.append(display_label)
                        last_finalize_reason = "manual_space"
                        manual_frame_vectors = []
            if key in {ord("s"), ord("S")}:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_payload = {
                    "timestamp": stamp,
                    "input_source": str(args.source),
                    "model_run_name": bundle["run_name"],
                    "cache_dir": str(cache_dir),
                    "device": str(device),
                    "trigger_mode": args.trigger_mode,
                    "emitted_labels": emitted_labels,
                    "reset_events": reset_events,
                    "saved_events": saved_events + 1,
                    "probability_policy": probability_policy_record(),
                }
                save_prediction_logs(RESULTS_DIR, prediction_log, session_payload, stamp=stamp)
                saved_events += 1
            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    cap.release()
    cv2.destroyAllWindows()

    if team_session is not None:
        paths = export_team_test_reports(team_session)
        print("組員測試結果：", paths.workbook)

    if args.save_log:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_payload = {
            "timestamp": stamp,
            "input_source": str(args.source),
            "model_run_name": bundle["run_name"],
            "cache_dir": str(cache_dir),
            "device": str(device),
            "trigger_mode": args.trigger_mode,
            "emitted_labels": emitted_labels,
            "reset_events": reset_events,
            "saved_events": saved_events,
            "probability_policy": probability_policy_record(),
        }
        save_prediction_logs(RESULTS_DIR, prediction_log, session_payload, stamp=stamp)

    print("最終辨識句子：", " | ".join(emitted_labels) if emitted_labels else "-")


if __name__ == "__main__":
    main()
