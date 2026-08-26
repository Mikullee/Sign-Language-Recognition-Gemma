from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from recognition.realtime.probability_reporting import (
    probability_policy_record,
    validate_raw_probability,
)


NO_DETECTION_LABEL = "未偵測"
EXPECTED_TEAM_TEST_LABELS = [
    "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08",
    "T10", "T11", "T12", "T13", "T14", "T15", "T16", "T17",
    "T18", "T19", "T20", "T21", "T22", "T23", "T25", "T27",
    "T28", "T29", "T30",
]
TEAM_PHASE_READY = "READY"
TEAM_PHASE_ARMED = "ARMED"
TEAM_PHASE_REVIEW = "REVIEW"
TEAM_PHASE_COMPLETE = "COMPLETE"


def sanitize_tester_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip()).strip("_")
    if not cleaned:
        raise ValueError("tester_id must contain at least one letter or number")
    return cleaned[:64]


def validate_team_test_labels(labels: list[str]) -> list[str]:
    normalized = [str(label) for label in labels]
    if normalized != EXPECTED_TEAM_TEST_LABELS:
        raise ValueError(
            "team test requires the current 27-class runtime label map in exact order"
        )
    return normalized


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExpectedTrial:
    label_id: str
    label_text: str
    trial_number: int
    global_trial_number: int


@dataclass
class TeamTrialRecord:
    tester_id: str
    timestamp: str
    global_trial_number: int
    expected_label: str
    expected_text: str
    trial_number: int
    predicted_label: str
    predicted_text: str
    outcome: str
    top1_correct: bool
    top3_hit: bool
    raw_probability: float
    probability_policy: dict[str, object]
    top3_candidates: list[dict[str, object]]
    clip_start_sec: float | None
    clip_end_sec: float | None
    finalize_sec: float | None
    segment_duration_sec: float | None
    finalize_delay_sec: float | None
    finalize_reason: str
    model_version: str


class TeamTestSession:
    def __init__(
        self,
        *,
        output_dir: Path,
        tester_id: str,
        labels: list[str],
        label_display: dict[str, str],
        trials_per_label: int = 10,
        model_version: str,
        runtime_metadata: dict[str, object] | None = None,
        resume: bool = False,
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        if trials_per_label <= 0:
            raise ValueError("trials_per_label must be positive")
        self.tester_id = sanitize_tester_id(tester_id)
        self.labels = list(labels)
        self.label_display = dict(label_display)
        self.trials_per_label = int(trials_per_label)
        self.model_version = str(model_version)
        self.runtime_metadata = dict(runtime_metadata or {})
        self.session_dir = Path(output_dir) / "team_tests" / self.tester_id
        self.progress_path = self.session_dir / "progress.json"
        self.records: list[TeamTrialRecord] = []
        self.pending_result: TeamTrialRecord | None = None
        if resume and self.progress_path.is_file():
            self._load_progress()

    @property
    def total_trials(self) -> int:
        return len(self.labels) * self.trials_per_label

    @property
    def completed_trials(self) -> int:
        return len(self.records)

    @property
    def is_complete(self) -> bool:
        return self.completed_trials >= self.total_trials

    @property
    def current_expected(self) -> ExpectedTrial:
        if self.is_complete:
            raise StopIteration("team test session is complete")
        index = self.completed_trials
        label_index = index // self.trials_per_label
        trial_number = index % self.trials_per_label + 1
        label_id = self.labels[label_index]
        return ExpectedTrial(
            label_id=label_id,
            label_text=self.label_display.get(label_id, label_id),
            trial_number=trial_number,
            global_trial_number=index + 1,
        )

    def stage_prediction(
        self,
        *,
        predicted_label: str,
        raw_probability: float,
        top3_candidates: list[tuple[str, float]],
        clip_start_sec: float | None = None,
        clip_end_sec: float | None = None,
        finalize_sec: float | None = None,
        finalize_reason: str = "",
        outcome: str = "prediction",
    ) -> TeamTrialRecord:
        expected = self.current_expected
        top3 = [
            {
                "label": label,
                "text": self.label_display.get(label, label),
                "raw_probability": validate_raw_probability(probability),
            }
            for label, probability in top3_candidates
        ]
        duration = (
            float(clip_end_sec - clip_start_sec)
            if clip_start_sec is not None and clip_end_sec is not None
            else None
        )
        finalize_delay = (
            float(finalize_sec - clip_end_sec)
            if finalize_sec is not None and clip_end_sec is not None
            else None
        )
        self.pending_result = TeamTrialRecord(
            tester_id=self.tester_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            global_trial_number=expected.global_trial_number,
            expected_label=expected.label_id,
            expected_text=expected.label_text,
            trial_number=expected.trial_number,
            predicted_label=str(predicted_label),
            predicted_text=self.label_display.get(predicted_label, predicted_label),
            outcome=outcome,
            top1_correct=predicted_label == expected.label_id,
            top3_hit=any(label == expected.label_id for label, _ in top3_candidates),
            raw_probability=validate_raw_probability(raw_probability),
            probability_policy=probability_policy_record(),
            top3_candidates=top3,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            finalize_sec=finalize_sec,
            segment_duration_sec=duration,
            finalize_delay_sec=finalize_delay,
            finalize_reason=str(finalize_reason),
            model_version=self.model_version,
        )
        return self.pending_result

    def confirm_pending(self) -> TeamTrialRecord:
        if self.pending_result is None:
            raise RuntimeError("there is no pending team-test result")
        record = self.pending_result
        self.records.append(record)
        self.pending_result = None
        self._save_progress()
        return record

    def confirm_prediction(self, **kwargs) -> TeamTrialRecord:
        self.stage_prediction(**kwargs)
        return self.confirm_pending()

    def confirm_no_detection(self) -> TeamTrialRecord:
        self.stage_no_detection()
        return self.confirm_pending()

    def stage_no_detection(self) -> TeamTrialRecord:
        return self.stage_prediction(
            predicted_label=NO_DETECTION_LABEL,
            raw_probability=0.0,
            top3_candidates=[],
            finalize_reason="no_detection_key",
            outcome="no_detection",
        )

    def discard_pending(self) -> None:
        self.pending_result = None

    def _metadata(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "tester_id": self.tester_id,
            "labels": self.labels,
            "label_display": self.label_display,
            "trials_per_label": self.trials_per_label,
            "model_version": self.model_version,
            "runtime_metadata": self.runtime_metadata,
            "probability_policy": probability_policy_record(),
            "records": [asdict(record) for record in self.records],
        }

    def _save_progress(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.progress_path)

    def _load_progress(self) -> None:
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "tester_id": self.tester_id,
            "labels": self.labels,
            "trials_per_label": self.trials_per_label,
            "model_version": self.model_version,
            "runtime_metadata": self.runtime_metadata,
        }
        for key, expected in expected_metadata.items():
            if payload.get(key) != expected:
                raise ValueError(f"saved team-test progress has incompatible {key}")
        self.records = [
            TeamTrialRecord(**self._normalize_record(row))
            for row in payload.get("records", [])
        ]
        if len(self.records) > self.total_trials:
            raise ValueError("saved team-test progress contains too many trials")

    @staticmethod
    def _normalize_record(row: dict[str, object]) -> dict[str, object]:
        """Read schema-one records while emitting only schema-two semantics."""
        normalized = dict(row)
        if "raw_probability" not in normalized:
            if "raw_confidence" not in normalized:
                raise ValueError("saved team-test record has no raw probability")
            normalized["raw_probability"] = normalized["raw_confidence"]
        normalized["raw_probability"] = validate_raw_probability(
            normalized["raw_probability"]  # type: ignore[arg-type]
        )
        normalized.pop("raw_confidence", None)
        normalized.pop("calibrated_confidence", None)
        normalized["probability_policy"] = probability_policy_record()
        top3 = []
        for item in normalized.get("top3_candidates", []):  # type: ignore[assignment]
            candidate = dict(item)
            if "raw_probability" not in candidate:
                if "confidence" not in candidate:
                    raise ValueError("saved top-3 candidate has no raw probability")
                candidate["raw_probability"] = candidate["confidence"]
            candidate["raw_probability"] = validate_raw_probability(
                candidate["raw_probability"]  # type: ignore[arg-type]
            )
            candidate.pop("confidence", None)
            top3.append(candidate)
        normalized["top3_candidates"] = top3
        return normalized


class TeamTestWorkflow:
    def __init__(self, session: TeamTestSession) -> None:
        self.session = session
        self.phase = TEAM_PHASE_COMPLETE if session.is_complete else TEAM_PHASE_READY

    def press_enter(self) -> TeamTrialRecord | None:
        if self.phase == TEAM_PHASE_READY:
            self.phase = TEAM_PHASE_ARMED
            return None
        if self.phase == TEAM_PHASE_REVIEW:
            record = self.session.confirm_pending()
            self.phase = (
                TEAM_PHASE_COMPLETE if self.session.is_complete else TEAM_PHASE_READY
            )
            return record
        return None

    def press_retry(self) -> None:
        if self.phase not in {TEAM_PHASE_ARMED, TEAM_PHASE_REVIEW}:
            return
        self.session.discard_pending()
        self.phase = TEAM_PHASE_READY

    def press_no_detection(self) -> TeamTrialRecord:
        if self.phase != TEAM_PHASE_ARMED:
            raise RuntimeError("no-detection can only be recorded while a trial is armed")
        record = self.session.stage_no_detection()
        self.phase = TEAM_PHASE_REVIEW
        return record

    def stage_prediction(self, **kwargs) -> TeamTrialRecord:
        if self.phase != TEAM_PHASE_ARMED:
            raise RuntimeError("prediction can only be staged while a trial is armed")
        record = self.session.stage_prediction(**kwargs)
        self.phase = TEAM_PHASE_REVIEW
        return record
