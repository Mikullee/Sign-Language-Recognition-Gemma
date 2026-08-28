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
    on_result: Any = None,
    on_state: Any = None,
):
    """Drive one capture source, yielding a Recognition per completed segment."""
    controller = AutoKnee42Controller(trigger_config, initial_mode="auto")
    fps = source.fps if source.fps > 0 else 30.0
    raw_frames = 0
    segments = 0
    results: list[Recognition] = []
    previous_state = ""

    while True:
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
        if on_result is not None:
            on_result(result)

    return results, raw_frames


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
    parser.add_argument("--headless", action="store_true", help="no preview window")
    args = parser.parse_args(argv)

    import cv2

    recognizer = Knee42TransformerRecognizer(args.bundle, device=args.device)
    trigger_config = load_auto_trigger_config(args.trigger_config)
    source = (
        open_video(args.video, cv2_module=cv2)
        if args.video
        else open_camera(args.camera, cv2_module=cv2)
    )

    print(f"model   : {recognizer.bundle.model_card.get('model_id')} ({len(recognizer.labels)} classes)")
    print(f"source  : {source.status}")
    print(f"trigger : {args.trigger_config}")
    print("\n站在鏡頭前，上半身與雙手腕完整入鏡，雙手自然垂放身側。")
    print("系統會先校準靜止基準，之後自動判定每一句的起訖。Ctrl-C 結束。\n")

    def report_state(state: str, calibrated: bool, at: float) -> None:
        mark = "calibrated" if calibrated else "calibrating"
        print(f"  [{at:6.2f}s] {state:<14} {mark}", flush=True)

    def report_result(result: Recognition) -> None:
        print(result.format(), flush=True)

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
                on_state=None if args.headless else report_state,
            )
    except KeyboardInterrupt:
        print("\n[exit] stopped")
        return 0
    finally:
        source.release()

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f"\n{len(results)} segment(s) from {raw_frames} frames in {elapsed:.1f}s "
          f"({raw_frames / elapsed:.1f} fps)")
    return 0


def detectors_for(args: argparse.Namespace) -> Detectors:
    return Detectors(args.hand_model, args.pose_model)


if __name__ == "__main__":
    raise SystemExit(main())
