from __future__ import annotations

import json
import re
import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path

from recognition.realtime.probability_reporting import (
    probability_policy_record,
    validate_raw_probability,
)


NO_DETECTION_LABEL = "未偵測"
EMPTY_OUTCOME_SENTINELS = {
    "no_detection": NO_DETECTION_LABEL,
    "short_segment": "片段過短",
}
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
SUPPORTED_PROGRESS_SCHEMAS = frozenset({1, 2})
LEGACY_RECORD_PROBABILITY_KEYS = frozenset(
    {"confidence", "raw_confidence", "calibrated_confidence"}
)
LEGACY_CANDIDATE_PROBABILITY_KEYS = frozenset(
    {"confidence", "raw_confidence", "calibrated_confidence"}
)
LEGACY_RUNTIME_PROBABILITY_KEYS = frozenset({"confidence_threshold"})
CANONICAL_CANDIDATE_KEYS = frozenset({"label", "text", "raw_probability"})
COMMON_PROGRESS_KEYS = frozenset(
    {
        "schema_version",
        "tester_id",
        "labels",
        "label_display",
        "trials_per_label",
        "model_version",
        "runtime_metadata",
        "records",
    }
)
PROGRESS_KEYS_BY_SCHEMA = {
    1: COMMON_PROGRESS_KEYS,
    2: COMMON_PROGRESS_KEYS | {"probability_policy"},
}
RECORD_STRING_FIELDS = (
    "tester_id",
    "timestamp",
    "expected_label",
    "expected_text",
    "predicted_label",
    "predicted_text",
    "outcome",
    "finalize_reason",
    "model_version",
)
RECORD_SEQUENCE_FIELDS = ("global_trial_number", "trial_number")
RECORD_BOOLEAN_FIELDS = ("top1_correct", "top3_hit")
RECORD_TIMING_FIELDS = (
    "clip_start_sec",
    "clip_end_sec",
    "finalize_sec",
    "segment_duration_sec",
    "finalize_delay_sec",
)
SUPPORTED_OUTCOMES = frozenset({"prediction", "no_detection", "short_segment"})


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
        self.records: list[TeamTrialRecord] = []
        self.pending_result: TeamTrialRecord | None = None
        self.validate_for_write()
        self.session_dir = Path(output_dir) / "team_tests" / self.tester_id
        self.progress_path = self.session_dir / "progress.json"
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
        candidate_records = [*self.records, record]
        self._save_progress(candidate_records)
        self.records = candidate_records
        self.pending_result = None
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
        self.validate_for_write()
        return self._metadata_for_records(self.records)

    def _metadata_for_records(
        self,
        records: list[TeamTrialRecord],
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "tester_id": self.tester_id,
            "labels": self.labels,
            "label_display": self.label_display,
            "trials_per_label": self.trials_per_label,
            "model_version": self.model_version,
            "runtime_metadata": self.runtime_metadata,
            "probability_policy": probability_policy_record(),
            "records": [asdict(record) for record in records],
        }

    def validate_for_write(self) -> None:
        records = list(self.records)
        if self.pending_result is not None:
            records.append(self.pending_result)
        self._serialize_progress(records)

    def _validate_runtime_metadata_for_write(self) -> None:
        if not isinstance(self.runtime_metadata, dict):
            raise ValueError("schema-two writer runtime_metadata must be an object")
        probability_keys = (
            LEGACY_RUNTIME_PROBABILITY_KEYS | {"probability_policy"}
        ).intersection(self.runtime_metadata)
        if probability_keys:
            raise ValueError(
                "schema-two writer runtime_metadata contains a forbidden probability key: "
                + ", ".join(sorted(probability_keys))
            )

    def _serialize_progress(self, records: list[TeamTrialRecord]) -> str:
        self._validate_runtime_metadata_for_write()
        if len(records) > self.total_trials:
            raise ValueError("team-test progress contains too many trials")
        for index, record in enumerate(records):
            self._validate_record_for_write(record, index)
        try:
            return json.dumps(
                self._metadata_for_records(records),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "team-test progress must be strict JSON serializable"
            ) from error

    def _save_progress(self, records: list[TeamTrialRecord]) -> None:
        serialized = self._serialize_progress(records)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(self.progress_path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _load_progress(self) -> None:
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("saved team-test progress must be a JSON object")
        schema_version = payload.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version not in SUPPORTED_PROGRESS_SCHEMAS
        ):
            raise ValueError("saved team-test progress has unsupported schema_version")
        unknown_fields = frozenset(payload) - PROGRESS_KEYS_BY_SCHEMA[schema_version]
        if unknown_fields:
            raise ValueError(
                "saved team-test progress contains unknown top-level fields: "
                + ", ".join(sorted(unknown_fields))
            )
        expected_metadata = {
            "tester_id": self.tester_id,
            "labels": self.labels,
            "label_display": self.label_display,
            "trials_per_label": self.trials_per_label,
            "model_version": self.model_version,
        }
        for key, expected in expected_metadata.items():
            if payload.get(key) != expected:
                raise ValueError(f"saved team-test progress has incompatible {key}")
        self._validate_runtime_metadata(payload, schema_version)
        if "records" not in payload:
            raise ValueError("saved team-test progress is missing records")
        records = payload["records"]
        if not isinstance(records, list):
            raise ValueError("saved team-test progress records must be a list")
        normalized_records: list[TeamTrialRecord] = []
        for index, row in enumerate(records):
            normalized = self._normalize_record(row, schema_version)
            self._validate_record_identity(normalized, index)
            try:
                record = TeamTrialRecord(**normalized)
            except TypeError as error:
                raise ValueError(
                    "saved team-test record has incompatible fields"
                ) from error
            self._validate_record_for_write(record, index)
            normalized_records.append(record)
        self.records = normalized_records
        if len(self.records) > self.total_trials:
            raise ValueError("saved team-test progress contains too many trials")

    def _validate_record_identity(
        self,
        record: dict[str, object],
        index: int,
    ) -> None:
        if record.get("tester_id") != self.tester_id:
            raise ValueError("saved team-test record has incompatible tester_id")
        if record.get("model_version") != self.model_version:
            raise ValueError("saved team-test record has incompatible model_version")
        if index >= self.total_trials:
            raise ValueError("saved team-test progress contains too many trials")
        expected_label = self.labels[index // self.trials_per_label]
        expected_trial_number = index % self.trials_per_label + 1
        expected_global_number = index + 1
        if record.get("global_trial_number") != expected_global_number:
            raise ValueError(
                "saved team-test record has incompatible global_trial_number"
            )
        if record.get("trial_number") != expected_trial_number:
            raise ValueError("saved team-test record has incompatible trial_number")
        if record.get("expected_label") != expected_label:
            raise ValueError(
                "saved team-test record has incompatible expected_label sequence"
            )
        expected_text = self.label_display.get(expected_label, expected_label)
        if record.get("expected_text") != expected_text:
            raise ValueError(
                "saved team-test record expected_text is incompatible with label_display"
            )

    def _validate_record_for_write(
        self,
        record: TeamTrialRecord,
        index: int,
    ) -> None:
        if not isinstance(record, TeamTrialRecord):
            raise ValueError("team-test writer record must be a TeamTrialRecord")
        payload = asdict(record)
        for field in RECORD_SEQUENCE_FIELDS:
            if type(payload.get(field)) is not int:
                raise ValueError(f"team-test record {field} must be an exact integer")
        for field in RECORD_STRING_FIELDS:
            if type(payload.get(field)) is not str:
                raise ValueError(f"team-test record {field} must be a string")
        for field in RECORD_BOOLEAN_FIELDS:
            if type(payload.get(field)) is not bool:
                raise ValueError(f"team-test record {field} must be an exact bool")
        self._validate_record_identity(payload, index)
        raw_probability = self._saved_probability(
            payload["raw_probability"], "team-test record"
        )
        if payload.get("probability_policy") != probability_policy_record():
            raise ValueError(
                "team-test writer record has incompatible probability_policy"
            )
        top3_candidates = payload.get("top3_candidates")
        if not isinstance(top3_candidates, list):
            raise ValueError("team-test writer top-3 candidates must be a list")
        candidate_labels: list[str] = []
        candidate_probabilities: list[float] = []
        for candidate in top3_candidates:
            if not isinstance(candidate, dict):
                raise ValueError("team-test writer top-3 candidate must be an object")
            candidate_keys = frozenset(candidate)
            if candidate_keys != CANONICAL_CANDIDATE_KEYS:
                raise ValueError(
                    "team-test writer top-3 candidate must contain exactly "
                    "label, text, raw_probability"
                )
            for field in ("label", "text"):
                if type(candidate[field]) is not str:
                    raise ValueError(
                        f"team-test record top3 candidate {field} must be a string"
                    )
            label = candidate["label"]
            if label not in self.labels:
                raise ValueError("team-test writer top-3 candidate label is unknown")
            if label in candidate_labels:
                raise ValueError("team-test writer top-3 candidate labels must be unique")
            if candidate["text"] != self.label_display.get(label, label):
                raise ValueError(
                    "team-test writer top-3 candidate text is incompatible with label_display"
                )
            candidate_labels.append(label)
            candidate_probabilities.append(
                self._saved_probability(
                    candidate["raw_probability"],
                    "top-3 candidate",
                )
            )
        self._validate_prediction_semantics(
            payload,
            raw_probability=raw_probability,
            candidate_labels=candidate_labels,
            candidate_probabilities=candidate_probabilities,
        )
        self._validate_record_timing(payload)

    def _validate_prediction_semantics(
        self,
        payload: dict[str, object],
        *,
        raw_probability: float,
        candidate_labels: list[str],
        candidate_probabilities: list[float],
    ) -> None:
        outcome = payload["outcome"]
        if outcome not in SUPPORTED_OUTCOMES:
            raise ValueError("team-test record outcome is unsupported")
        if candidate_labels:
            if outcome != "prediction":
                raise ValueError(
                    "team-test record only prediction may contain top-3 candidates"
                )
            if len(candidate_labels) > 3:
                raise ValueError(
                    "team-test prediction must contain between one and three candidates"
                )
            if any(
                earlier < later
                for earlier, later in zip(
                    candidate_probabilities,
                    candidate_probabilities[1:],
                )
            ):
                raise ValueError(
                    "team-test top-3 candidate probabilities must be non-increasing"
                )
            if payload["predicted_label"] != candidate_labels[0]:
                raise ValueError(
                    "team-test predicted_label must equal the top-1 candidate label"
                )
            expected_top1_text = self.label_display.get(
                candidate_labels[0], candidate_labels[0]
            )
            if payload["predicted_text"] != expected_top1_text:
                raise ValueError(
                    "team-test predicted_text must equal the top-1 candidate text"
                )
            if raw_probability != candidate_probabilities[0]:
                raise ValueError(
                    "team-test raw_probability must equal the top-1 candidate probability"
                )
            expected_top1_correct = payload["predicted_label"] == payload["expected_label"]
            expected_top3_hit = payload["expected_label"] in candidate_labels
        else:
            if outcome not in EMPTY_OUTCOME_SENTINELS:
                raise ValueError(
                    "team-test empty top-3 candidates require no_detection or short_segment"
                )
            if raw_probability != 0.0:
                raise ValueError(
                    "team-test empty top-3 candidates require zero raw_probability"
                )
            expected_top1_correct = False
            expected_top3_hit = False
            expected_sentinel = EMPTY_OUTCOME_SENTINELS[outcome]
            if (
                payload["predicted_label"] != expected_sentinel
                or payload["predicted_text"] != expected_sentinel
            ):
                raise ValueError(
                    f"team-test {outcome} requires the exact "
                    f"{expected_sentinel!r} predicted label/text sentinel"
                )
        if payload["top1_correct"] is not expected_top1_correct:
            raise ValueError("team-test record top1_correct is inconsistent")
        if payload["top3_hit"] is not expected_top3_hit:
            raise ValueError("team-test record top3_hit is inconsistent")

    @staticmethod
    def _validate_record_timing(payload: dict[str, object]) -> None:
        values = [payload[field] for field in RECORD_TIMING_FIELDS]
        for field, value in zip(RECORD_TIMING_FIELDS, values):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"team-test record {field} must be None or a finite real number"
                )
        present = [value is not None for value in values]
        if any(present) and not all(present):
            raise ValueError(
                "team-test record timing fields must be all present or all None"
            )
        if not any(present):
            return
        start, end, finalize, duration, delay = values
        if not (start <= end <= finalize):
            raise ValueError(
                "team-test record timing must satisfy clip_start_sec <= "
                "clip_end_sec <= finalize_sec"
            )
        if duration != end - start:
            raise ValueError(
                "team-test record segment_duration_sec is inconsistent with clip bounds"
            )
        if delay != finalize - end:
            raise ValueError(
                "team-test record finalize_delay_sec is inconsistent with clip bounds"
            )

    def _validate_runtime_metadata(
        self,
        payload: dict[str, object],
        schema_version: int,
    ) -> None:
        saved_metadata = payload.get("runtime_metadata")
        if not isinstance(saved_metadata, dict):
            raise ValueError("saved team-test progress runtime_metadata must be an object")
        saved = dict(saved_metadata)
        expected = dict(self.runtime_metadata)
        if schema_version == 1:
            if "probability_policy" in payload or "probability_policy" in saved:
                raise ValueError(
                    "schema-one progress cannot contain schema-two probability metadata"
                )
            for key in LEGACY_RUNTIME_PROBABILITY_KEYS:
                if key in saved:
                    try:
                        validate_raw_probability(saved.pop(key))  # type: ignore[arg-type]
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"saved team-test progress has invalid legacy {key}"
                        ) from error
            expected.pop("probability_policy", None)
        else:
            if payload.get("probability_policy") != probability_policy_record():
                raise ValueError(
                    "saved team-test progress has incompatible probability_policy"
                )
            if "probability_policy" in saved:
                raise ValueError(
                    "schema-two runtime_metadata contains a forbidden "
                    "probability_policy shadow"
                )
            if LEGACY_RUNTIME_PROBABILITY_KEYS.intersection(saved):
                raise ValueError(
                    "schema-two runtime_metadata contains a legacy probability key"
                )
        if saved != expected:
            raise ValueError(
                "saved team-test progress has incompatible runtime_metadata"
            )

    @staticmethod
    def _saved_probability(value: object, context: str) -> float:
        try:
            return validate_raw_probability(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(f"saved {context} has invalid raw probability") from error

    @classmethod
    def _normalize_record(
        cls,
        row: object,
        schema_version: int,
    ) -> dict[str, object]:
        """Parse one schema-specific record into current raw-probability semantics."""
        if not isinstance(row, dict):
            raise ValueError("saved team-test record must be an object")
        normalized = dict(row)
        if schema_version == 1:
            if "raw_probability" in normalized or "probability_policy" in normalized:
                raise ValueError(
                    "schema-one record contains schema-two probability fields"
                )
            if "raw_confidence" not in normalized:
                raise ValueError("saved team-test record has no raw probability")
            normalized["raw_probability"] = normalized["raw_confidence"]
            normalized.pop("raw_confidence", None)
            normalized.pop("calibrated_confidence", None)
            normalized["probability_policy"] = probability_policy_record()
        else:
            legacy_keys = LEGACY_RECORD_PROBABILITY_KEYS.intersection(normalized)
            if legacy_keys:
                raise ValueError(
                    "schema-two record contains a legacy probability key: "
                    + ", ".join(sorted(legacy_keys))
                )
            if "raw_probability" not in normalized:
                raise ValueError("saved team-test record has no raw probability")
            if normalized.get("probability_policy") != probability_policy_record():
                raise ValueError(
                    "saved team-test record has incompatible probability_policy"
                )
        normalized["raw_probability"] = cls._saved_probability(
            normalized["raw_probability"],
            "team-test record",
        )
        if "top3_candidates" not in normalized:
            raise ValueError("saved team-test record is missing top3_candidates")
        top3_candidates = normalized["top3_candidates"]
        if not isinstance(top3_candidates, list):
            raise ValueError("saved top-3 candidates must be a list")
        top3: list[dict[str, object]] = []
        for item in top3_candidates:
            if not isinstance(item, dict):
                raise ValueError("saved top-3 candidate must be an object")
            candidate = dict(item)
            if schema_version == 1:
                forbidden_keys = {
                    key
                    for key in (
                        "raw_confidence",
                        "calibrated_confidence",
                        "raw_probability",
                    )
                    if key in candidate
                }
                if forbidden_keys:
                    raise ValueError(
                        "schema-one top-3 candidate contains a noncanonical probability key: "
                        + ", ".join(sorted(forbidden_keys))
                    )
                if "confidence" not in candidate:
                    raise ValueError("saved top-3 candidate has no raw probability")
                candidate["raw_probability"] = candidate["confidence"]
                candidate.pop("confidence", None)
            else:
                legacy_keys = LEGACY_CANDIDATE_PROBABILITY_KEYS.intersection(candidate)
                if legacy_keys:
                    raise ValueError(
                        "schema-two top-3 candidate contains a legacy probability key: "
                        + ", ".join(sorted(legacy_keys))
                    )
                if "raw_probability" not in candidate:
                    raise ValueError("saved top-3 candidate has no raw probability")
            candidate["raw_probability"] = cls._saved_probability(
                candidate["raw_probability"],
                "top-3 candidate",
            )
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
