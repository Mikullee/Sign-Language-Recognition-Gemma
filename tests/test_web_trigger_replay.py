import copy
from dataclasses import replace

import pytest

from recognition.evaluation.web_trigger_replay import (
    make_variants, grouped_folds, score_boundaries, replay_frames, choose_candidate,
    select_knee_candidate, compare_records,
)
from tests.test_web_knee_gate import config
from tests.test_web_knee_stream import payload_frame


def test_twenty_variants_are_deterministic_and_keep_original_group():
    first, second = make_variants(42), make_variants(42)
    assert first == second and len(first) == 20
    assert {v["speed"] for v in first} >= {.75, 1, 1.25}
    assert {v["fps"] for v in first} >= {10, 15, 17.5, 30}
    assert len({v["name"] for v in first}) == 20


def test_folds_never_leak_variants_or_duplicate_source_files():
    records = [{"source_id": s, "variant": v} for s in ["shaA", "shaB", "shaC"] for v in range(20)]
    folds = list(grouped_folds(records))
    assert len(folds) == 3
    for train, heldout in folds:
        assert {r["source_id"] for r in train}.isdisjoint(r["source_id"] for r in heldout)
        assert len(train) == 40 and len(heldout) == 20


def test_timeout_is_never_a_boundary_pass_even_at_exact_ground_truth():
    event = {"accepted": False, "segment": {"clip_start_sec": 1, "clip_end_sec": 3,
             "finalize_sec": 3, "reason": "timeout_finalize"}}
    m = score_boundaries([event], 1, 3)
    assert not m["passed"] and m["timeouts"] == 1 and m["misses"] == 1


def test_extra_segments_and_early_cut_are_not_hidden():
    def event(a, b):
        return {"accepted": True, "segment": {"clip_start_sec": a, "clip_end_sec": b,
                "finalize_sec": b + .5, "reason": "visible_rest_finalize"}}
    m = score_boundaries([event(1, 2.5), event(2.8, 3.5)], 1, 3)
    assert m["extras"] == 1 and m["early_cuts"] == 1 and not m["passed"]


def test_no_eligible_knee_video_means_no_claimed_tuned_winner():
    assert choose_candidate([], [config(), replace(config(), end_hold_sec=.35)]) is None


def test_replay_missing_knees_is_rejection_not_successful_segmentation():
    raw = [payload_frame(i / 10, knee_visible=False) for i in range(30)]
    report = replay_frames(raw, config())
    assert report["events"] == [] and report["reference_revision"] == 0
    assert report["eof"] == "eof"


def test_one_eligible_original_cannot_be_called_validated_knee_tuning():
    winner, folds = select_knee_candidate([{"source_id": "one", "eligible": True}], config())
    assert winner is None and folds == []


def test_comparison_is_computed_for_the_actual_exported_config():
    frames = [payload_frame(i / 10, hand_y=.35 if 13 <= i < 28 else .78) for i in range(48)]
    record = {"source_id": "one", "variant": "original", "start": 1.3, "end": 2.8,
              "eligible": True, "frames": frames, "cache_id": "one"}
    candidate = replace(config(), end_hold_sec=.35)
    rows, traces = compare_records([record], replace(config(), knee_gate_enabled=False), candidate)
    delay = rows[0]["after"]["confirmation_delay_sec"]
    assert .35 <= delay <= .45
    assert traces["one"]["after"]["events"] == replay_frames(frames, candidate)["events"]


def test_ineligible_video_metrics_are_explicitly_diagnostic_only():
    record = {"source_id": "one", "variant": "original", "start": 1, "end": 2,
              "eligible": False, "frames": [payload_frame(i / 10, knee_visible=False) for i in range(30)],
              "cache_id": "one"}
    rows, _ = compare_records([record], replace(config(), knee_gate_enabled=False), config())
    assert rows[0]["boundary_metrics_applicable"] is False
    assert rows[0]["candidate_calibrated"] is False
