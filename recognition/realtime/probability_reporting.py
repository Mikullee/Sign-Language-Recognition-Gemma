"""Shared semantics for reporting uncalibrated model probabilities."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True, slots=True)
class ProbabilityPolicy:
    """Auditable policy for raw softmax output with acceptance disabled."""

    kind: str = "uncalibrated_softmax"
    acceptance_policy: str = "disabled_no_risk_coverage_evidence"
    calibration_artifact: None = None


PROBABILITY_POLICY = ProbabilityPolicy()


def probability_policy_record() -> dict[str, Any]:
    """Return a fresh JSON-serializable record of the probability policy."""
    return asdict(PROBABILITY_POLICY)


def validate_raw_probability(value: Real) -> float:
    """Return a finite probability unchanged, rejecting invalid semantics."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("raw probability must be a real number")
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError("raw probability must be finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("raw probability must be between 0 and 1 inclusive")
    return probability
