"""Camera -> auto-trigger -> Transformer: realtime recognition with automatic boundaries.

This is the wiring the repository was missing. Both halves already existed and
both are kept exactly as they are:

  * ``recognition.realtime.auto_trigger`` decides when a sign starts and ends,
    from the 225-value trigger view, against a rest reference it calibrates from
    the first second of footage.  Its thresholds were tuned on real sessions and
    are not re-tuned here.
  * ``recognition.transformer.recognizer`` classifies a finished segment.

They already speak compatible languages: ``knee42_preprocessing`` emits the
trigger view and the 219-value recognition view from one MediaPipe detection, so
the only thing needed was to hand the finished segment to the Transformer
instead of the legacy BiGRU.

The segment buffer holds shoulder-normalized values with NaN where a landmark
was not seen, which is exactly the Transformer's input contract -- interpolation
and resampling happen inside ``materialize_sequence``.

Usage
-----
    python -m recognition.transformer.realtime                 # first camera
    python -m recognition.transformer.realtime --camera 1
    python -m recognition.transformer.realtime --video clip.mp4 --headless

Expected noise
--------------
MediaPipe's native library writes these to stderr on every start. None of them
indicate a problem, and none are suppressible from Python -- they come from the
C++ side, which ignores ``GLOG_minloglevel``. Redirecting the file descriptor
would hide real errors too, so they are documented rather than swallowed:

    INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
    W0000 ... inference_feedback_manager.cc ... Disabling support for feedback tensors.
    W0000 ... landmark_projection_calculator.cc ... Using NORM_RECT without IMAGE_DIMENSIONS
    E0000 ... portable_clearcut_uploader.cc ... Failed to send to clearcut

The last one is MediaPipe failing to upload usage telemetry to Google. It is
logged at error level despite being unrelated to recognition, and appears only
intermittently.
"""
from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from recognition.config import preview_paths
from recognition.realtime.auto_trigger import AutoTriggerConfig, load_auto_trigger_config
from recognition.realtime.knee42_capture import open_camera, open_video
from recognition.realtime.knee42_controllers import AutoKnee42Controller
from recognition.realtime.knee42_preprocessing import (
    FrameObservation,
    normalize_frame,
    observation_from_results,
)
from recognition.transformer.recognizer import Knee42TransformerRecognizer


DEFAULT_TRIGGER_CONFIG = Path("configs") / "auto_trigger_knee_v1.json"
DEFAULT_FRAME_STEP = 2
MIN_SEGMENT_FRAMES = 4
DEFAULT_FIRST_FRAME_TIMEOUT = 10.0


@dataclass
class Recognition:
    """One finished segment and what the model made of it."""

    index: int
    start_sec: float
    end_sec: float
    frames: int
    reason: str
    top: list[tuple[str, str, float]]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def format(self) -> str:
        head = (
            f"#{self.index:<3} {self.start_sec:6.2f} - {self.end_sec:6.2f}"
            f"  ({self.duration_sec:.2f}s, {self.frames} frames, {self.reason})"
        )
        if not self.top:
            return head + "  [too short to classify]"
        ranked = "  ".join(f"{text} {prob:.2f}" for _, text, prob in self.top)
        return head + "\n     " + ranked


class Detectors:
    """MediaPipe hand and pose landmarkers over the shared frame contract.

    IMAGE running mode, un-flipped frames, handedness as reported: the same
    contract the feature cache was built under.  See recognition/transformer/
    landmarks.py for why each of those matters.
    """

    def __init__(self, hand_model: Path, pose_model: Path) -> None:
        for name, path in (("hand", hand_model), ("pose", pose_model)):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"{name} landmarker model not found: {path}\n"
                    f"run: python scripts/fetch_mediapipe_models.py"
                )
        self.hand_model, self.pose_model = Path(hand_model), Path(pose_model)
        self._stack: ExitStack | None = None
        self._mp = None
        self._hands = None
        self._pose = None

    def __enter__(self) -> "Detectors":
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.hand_landmarker import (
            HandLandmarker,
            HandLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarker,
            PoseLandmarkerOptions,
        )

        self._stack = ExitStack()
        self._hands = self._stack.enter_context(
            HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.hand_model)),
                    running_mode=VisionTaskRunningMode.IMAGE,
                    num_hands=2,
                )
            )
        )
        self._pose = self._stack.enter_context(
            PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.pose_model)),
                    running_mode=VisionTaskRunningMode.IMAGE,
                )
            )
        )
        self._mp = mp
        return self

    def __exit__(self, *_exc: object) -> bool:
        if self._stack is not None:
            self._stack.close()
        return False

    def observe(self, frame_bgr: np.ndarray) -> FrameObservation:
        if self._hands is None or self._pose is None or self._mp is None:
            raise RuntimeError("detectors are not open")
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        observation = observation_from_results(self._hands.detect(image), self._pose.detect(image))
        return FrameObservation(
            trigger_values=observation.trigger_values,
            recognition_values=normalize_frame(
                observation.recognition_values, observation.recognition_mask
            ),
            recognition_mask=observation.recognition_mask,
            display_pose=observation.display_pose,
            display_left_hand=observation.display_left_hand,
            display_right_hand=observation.display_right_hand,
        )


class NoFramesError(RuntimeError):
    """The source opened but never delivered a frame."""


def read_first_frame(source: Any, *, timeout_sec: float = DEFAULT_FIRST_FRAME_TIMEOUT):
    """Read one frame, refusing to wait forever for a camera that never starts.

    ``VideoCapture.read`` blocks, so a camera another process is holding will
    hang the whole program with no window and no output -- which looks exactly
    like a crash. Doing the first read on a worker thread turns that into a
    message that says what to do about it.
    """
    import threading

    outcome: dict[str, Any] = {}

    def attempt() -> None:
        try:
            ok, frame = source.read()
            outcome["frame"] = frame if ok else None
        except Exception as exc:  # noqa: BLE001 - reported to the caller below
            outcome["error"] = exc

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(timeout_sec)

    if worker.is_alive():
        raise NoFramesError(
            f"{source.status} produced no frame within {timeout_sec:.0f}s.\n"
            "  Another process is usually holding the camera. Close any other session\n"
            "  of this program (and any app using the webcam), then retry."
        )
    if "error" in outcome:
        raise NoFramesError(f"{source.status} failed to read: {outcome['error']}")
    if outcome.get("frame") is None:
        raise NoFramesError(
            f"{source.status} opened but returned no image.\n"
            "  The camera index may be wrong; try --camera 1."
        )
    return outcome["frame"]


def sequence_from_features(features: Sequence[Any]) -> np.ndarray:
    """Stack a segment's (values, mask) pairs into the ``[frames, 219]`` input.

    The mask is deliberately dropped: this path feeds the model coordinates with
    NaN gaps and lets interpolation fill them, rather than the neutral-fill plus
    mask channel the legacy BiGRU contract used.
    """
    return np.stack([np.asarray(values, dtype=np.float32) for values, _mask in features])


def recognize_stream(
    source: Any,
    detectors: Detectors,
    recognizer: Knee42TransformerRecognizer,
    trigger_config: AutoTriggerConfig,
    *,
    frame_step: int = DEFAULT_FRAME_STEP,
    topk: int = 3,
    max_frames: int | None = None,
    first_frame_timeout_sec: float = DEFAULT_FIRST_FRAME_TIMEOUT,
    on_result: Any = None,
    on_state: Any = None,
    on_frame: Any = None,
):
    """Drive one capture source, yielding a Recognition per completed segment.

    ``on_frame(frame, state, calibrated, latest)`` is called for every detected
    frame, so a caller can draw a preview without this loop knowing about UI.
    Returning False from it stops the stream, which is how a window's quit key
    ends the session.
    """
    controller = AutoKnee42Controller(trigger_config, initial_mode="auto")
    fps = source.fps if source.fps > 0 else 30.0
    raw_frames = 0
    segments = 0
    results: list[Recognition] = []
    previous_state = ""
    latest: Recognition | None = None
    first = read_first_frame(source, timeout_sec=first_frame_timeout_sec)

    while True:
        if first is not None:
            ok, frame, first = True, first, None
        else:
            ok, frame = source.read()
        if not ok:
            break
        raw_frames += 1
        if max_frames is not None and raw_frames > max_frames:
            break
        if (raw_frames - 1) % frame_step:
            continue

        observation = detectors.observe(frame)
        timestamp_sec = (raw_frames - 1) / fps
        event = controller.add_held_observation(
            timestamp_sec,
            observation.trigger_values,
            (observation.recognition_values, observation.recognition_mask),
            frame_interval_sec=1.0 / fps,
            sample_count=frame_step,
        )

        if on_state is not None and controller.state != previous_state:
            previous_state = controller.state
            on_state(controller.state, controller.calibrated, timestamp_sec)

        if on_frame is not None:
            if on_frame(frame, observation, controller.state, controller.calibrated, latest) is False:
                break

        if not event.infer:
            continue

        segments += 1
        sequence = sequence_from_features(event.features)
        top = (
            recognizer.predict(sequence, topk=topk)
            if len(sequence) >= MIN_SEGMENT_FRAMES
            else []
        )
        evidence = event.segment
        result = Recognition(
            index=segments,
            start_sec=evidence.clip_start_sec if evidence else timestamp_sec,
            end_sec=evidence.clip_end_sec if evidence else timestamp_sec,
            frames=len(sequence),
            reason=evidence.reason if evidence else "unknown",
            top=top,
        )
        results.append(result)
        latest = result
        if on_result is not None:
            on_result(result)

    return results, raw_frames


STATE_COLORS = {
    "IDLE_BLANK": (170, 170, 170),
    "SIGNING_ACTIVE": (90, 200, 90),
    "END_CONFIRM": (70, 190, 235),
}
FALLBACK_COLOR = (200, 160, 90)


class PreviewWindow:
    """A camera preview with the skeleton the recognizer actually sees.

    Framing is the single most common reason a session produces nothing: both
    shoulders have to be visible for a frame to be usable at all. Drawing the
    landmarks makes that immediately obvious instead of silently dropping frames.

    Chinese needs Pillow, since OpenCV's own text renderer cannot draw CJK. When
    no CJK font is available the label id is shown rather than mojibake.
    """

    WINDOW = "Knee42 realtime"

    def __init__(self, quit_keys: str = "q") -> None:
        self.quit_keys = {ord(key) for key in quit_keys}
        self._font = self._load_font()

    @staticmethod
    def _load_font():
        try:
            from PIL import ImageFont
        except ModuleNotFoundError:
            return None
        for name in ("msjh.ttc", "msyh.ttc", "mingliu.ttc", "NotoSansCJK-Regular.ttc"):
            try:
                return ImageFont.truetype(name, 26)
            except OSError:
                continue
        return None

    def _draw_text(self, frame, text: str, origin: tuple[int, int], color):
        if self._font is None or text.isascii():
            import cv2

            cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            return frame
        import cv2
        from PIL import Image, ImageDraw

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text(origin, text, font=self._font, fill=tuple(reversed(color)))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def show(self, frame, observation, state: str, calibrated: bool, latest) -> bool:
        import cv2

        canvas = frame.copy()
        height, width = canvas.shape[:2]
        color = STATE_COLORS.get(state, FALLBACK_COLOR)

        for points, dot in (
            (observation.display_pose, 3),
            (observation.display_left_hand, 4),
            (observation.display_right_hand, 4),
        ):
            if points is None:
                continue
            for point in np.asarray(points):
                if not np.isfinite(point[:2]).all():
                    continue
                cv2.circle(
                    canvas,
                    (int(point[0] * width), int(point[1] * height)),
                    dot, color, -1, cv2.LINE_AA,
                )

        cv2.rectangle(canvas, (0, 0), (width, 46), (24, 24, 24), -1)
        canvas = self._draw_text(
            canvas,
            f"{state}   {'calibrated' if calibrated else 'calibrating...'}",
            (14, 12 if self._font else 32),
            color,
        )
        if latest is not None and latest.top:
            label, text, prob = latest.top[0]
            cv2.rectangle(canvas, (0, height - 54), (width, height), (24, 24, 24), -1)
            canvas = self._draw_text(
                canvas,
                f"{text}  {prob:.2f}" if self._font else f"{label} {prob:.2f}",
                (14, height - 46 if self._font else height - 18),
                (255, 255, 255),
            )

        cv2.imshow(self.WINDOW, canvas)
        return cv2.waitKey(1) & 0xFF not in self.quit_keys

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(self.WINDOW)


def main(argv: Sequence[str] | None = None) -> int:
    paths = preview_paths()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=None, help="camera index; omit to probe")
    parser.add_argument("--video", type=Path, default=None, help="read a file instead of a camera")
    parser.add_argument("--bundle", type=Path, default=paths.runtime_bundle_dir)
    parser.add_argument("--hand-model", type=Path, default=paths.hand_model)
    parser.add_argument("--pose-model", type=Path, default=paths.pose_model)
    parser.add_argument("--trigger-config", type=Path, default=DEFAULT_TRIGGER_CONFIG)
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="console only, no preview window"
    )
    args = parser.parse_args(argv)

    import cv2

    recognizer = Knee42TransformerRecognizer(args.bundle, device=args.device)
    trigger_config = load_auto_trigger_config(args.trigger_config)
    source = (
        open_video(args.video, cv2_module=cv2)
        if args.video
        else open_camera(args.camera, cv2_module=cv2)
    )

    # stderr is unbuffered and stdout is not, so MediaPipe's native logging
    # arrives ahead of anything printed here unless these are flushed. Seeing
    # only the noise makes a working start look like a failed one.
    print(f"model   : {recognizer.bundle.model_card.get('model_id')} ({len(recognizer.labels)} classes)", flush=True)
    print(f"source  : {source.status}", flush=True)
    print(f"trigger : {args.trigger_config}", flush=True)
    print("\n站在鏡頭前，上半身與雙手腕完整入鏡，雙手自然垂放身側。")
    print("系統會先校準靜止基準，之後自動判定每一句的起訖。")
    print("按 Q 關閉視窗結束，或 Ctrl-C。\n" if not args.headless else "Ctrl-C 結束。\n")

    def report_state(state: str, calibrated: bool, at: float) -> None:
        mark = "calibrated" if calibrated else "calibrating"
        print(f"  [{at:6.2f}s] {state:<14} {mark}", flush=True)

    def report_result(result: Recognition) -> None:
        print(result.format(), flush=True)

    preview = None if args.headless else PreviewWindow()
    started = time.perf_counter()
    try:
        with detectors_for(args) as detectors:
            results, raw_frames = recognize_stream(
                source,
                detectors,
                recognizer,
                trigger_config,
                frame_step=args.frame_step,
                topk=args.topk,
                max_frames=args.max_frames,
                on_result=report_result,
                on_state=report_state,
                on_frame=None if preview is None else preview.show,
            )
    except KeyboardInterrupt:
        print("\n[exit] stopped")
        return 0
    except NoFramesError as exc:
        print(f"\n[error] {exc}", flush=True)
        return 1
    finally:
        source.release()
        if preview is not None:
            preview.close()

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f"\n{len(results)} segment(s) from {raw_frames} frames in {elapsed:.1f}s "
          f"({raw_frames / elapsed:.1f} fps)")
    return 0


def detectors_for(args: argparse.Namespace) -> Detectors:
    return Detectors(args.hand_model, args.pose_model)


if __name__ == "__main__":
    raise SystemExit(main())
