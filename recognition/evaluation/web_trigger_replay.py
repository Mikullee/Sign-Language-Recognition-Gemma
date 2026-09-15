"""Reproducible Web VIDEO-mode replay, grouped diagnostics, and private overlays.

Run with --help. This tunes a boundary controller, not the 42-class recognizer.
Original source SHA groups are kept intact across all augmentations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from recognition.transformer.landmarks import observation_from_frame
from recognition.transformer.live_trigger import WebAutoKnee42Controller, load_web_trigger_config
from webservice.server import frames_from_payload

ROOT = Path(__file__).resolve().parents[2]
NORMAL_REASONS = {"visible_rest_finalize", "reference_rest_finalize", "hidden_rest_finalize"}


def loopback_endpoint(url):
    endpoint = urllib.parse.urlsplit(url)
    if (endpoint.scheme not in {"http", "https"} or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.username or endpoint.password or endpoint.path not in {"", "/"}
            or endpoint.query or endpoint.fragment):
        raise ValueError("Evaluation requires a plain loopback HTTP(S) origin")
    return endpoint.hostname, endpoint.port or (443 if endpoint.scheme == "https" else 80), endpoint.scheme


@contextmanager
def local_service(url, out):
    """Start an owned loopback service if necessary; never stop someone else's."""
    host, port, scheme = loopback_endpoint(url)
    def ready():
        try:
            # urllib uses a fresh opener; disable proxies for local-only data.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url.rstrip("/") + "/health", timeout=2) as response:
                data = json.load(response)
                if data.get("ok") is not True or not data.get("auto_trigger"):
                    raise ValueError("Another service occupies the evaluation port")
                return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False
    if ready():
        yield
        return
    if scheme != "http" or host == "::1":
        raise ValueError("Start the specified TLS/IPv6 service first, or use http://127.0.0.1:8642")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "private_service.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "webservice.server", "--http",
                                    "--host", host, "--port", str(port)], cwd=ROOT, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        try:
            deadline = time.monotonic() + 30
            while not ready():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Local service failed to start; inspect private_service.log")
                time.sleep(.1)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def make_variants(seed=42):
    specs = [
        ("original", {}), ("slow", {"speed": .75}), ("fast", {"speed": 1.25}),
        ("fps10", {"fps": 10}), ("fps15", {"fps": 15}), ("fps17p5", {"fps": 17.5}),
        ("left", {"dx": -.04}), ("right", {"dx": .04}),
        ("up", {"dy": -.025}), ("down", {"dy": .025}),
        ("small", {"scale": .85}), ("large", {"scale": 1.15}),
        ("jitter", {"fps": 17.5, "jitter": .006}),
        ("hand_gaps", {"drop_hands": True}), ("lower_crop", {"crop_lower": True}),
        ("knee_visibility_loss", {"hide_knees": True}),
        ("slow_left", {"speed": .75, "dx": -.03}),
        ("fast_small", {"speed": 1.25, "scale": .9}),
        ("fps10_small", {"fps": 10, "scale": .9}),
        ("fps15_hand_gaps", {"fps": 15, "drop_hands": True}),
    ]
    return [dict({"name": name, "fps": 30, "speed": 1, "scale": 1,
                  "dx": 0, "dy": 0, "seed": seed}, **settings) for name, settings in specs]


def grouped_folds(records):
    for source in sorted({r["source_id"] for r in records}):
        yield ([r for r in records if r["source_id"] != source],
               [r for r in records if r["source_id"] == source])


def score_boundaries(events, start, end, tolerance=.3):
    accepted = [e["segment"] for e in events
                if e["accepted"] and e["segment"]["reason"] in NORMAL_REASONS]
    def overlap(segment):
        return max(0, min(end, segment["clip_end_sec"]) - max(start, segment["clip_start_sec"]))
    matched = max(accepted, key=overlap, default=None)
    if matched is not None and overlap(matched) <= 0:
        matched = None
    a = None if matched is None else matched["clip_start_sec"] - start
    b = None if matched is None else matched["clip_end_sec"] - end
    extras = len(accepted) - int(matched is not None)
    return {"passed": bool(matched and not extras and abs(a) <= tolerance and abs(b) <= tolerance),
            "start_error_sec": a, "end_error_sec": b,
            "confirmation_delay_sec": None if matched is None else matched["finalize_sec"] - matched["clip_end_sec"],
            "misses": int(matched is None), "extras": extras,
            "early_cuts": int(b is not None and b < -tolerance),
            "timeouts": sum(e["segment"]["reason"] == "timeout_finalize" for e in events),
            "data_gaps": sum(e["segment"]["reason"] == "tracking_gap" for e in events)}


def replay_frames(raw_frames, config, *, baseline=False, keep_trace=True):
    if baseline:
        from recognition.evaluation.baselines.web_trigger_0ae3ac0 import (
            WebAutoKnee42Controller as PreviousController, WebAutoTriggerConfig as PreviousConfig,
        )
        from dataclasses import fields
        keys = {f.name for f in fields(PreviousConfig)}
        controller = PreviousController(PreviousConfig(**{k: v for k, v in config.to_dict().items() if k in keys}))
    else:
        controller = WebAutoKnee42Controller(config)
    events, trace = [], []
    for frame in frames_from_payload(raw_frames):
        observation = observation_from_frame(frame)
        if baseline:
            analysis = controller._analysis_fn(controller._previous_trigger, observation.trigger_values, config)
            event = controller.add_observation(frame.timestamp, observation.trigger_values, frame.timestamp)
        else:
            event = controller.add_observation(frame.timestamp, observation.trigger_values, frame.timestamp,
                                               pose_visibility=frame.pose_visibility)
            analysis = controller.last_analysis
        if event.segment:
            events.append({"segment": asdict(event.segment), "accepted": bool(event.infer and event.segment.reason in NORMAL_REASONS),
                           "message": event.message})
        if keep_trace:
            trace.append({"t": frame.timestamp, "state": controller.state,
                          "calibrated": controller.calibrated,
                          "revision": controller.engine.reference_revision,
                          "knees_visible": analysis.knee_landmarks_valid,
                          "on_knees": analysis.hands_on_knees,
                          "motion": analysis.effective_motion_score,
                          "status": controller.engine.rest_signature_status(analysis)})
    # Baseline is compared on genuine observations only too. Never call its
    # archived EOF method, which fabricates repeated rest frames.
    eof = "incomplete_eof" if controller.state in {"SIGNING_ACTIVE", "END_CONFIRM"} else "eof"
    return {"events": events, "trace": trace, "reference_revision": controller.engine.reference_revision,
            "final_state": controller.state, "eof": eof}


def _objective(metrics):
    return (sum(m["misses"] + m["extras"] + m["timeouts"] + m["data_gaps"] for m in metrics),
            -sum(m["passed"] for m in metrics),
            sum(abs(m["start_error_sec"] or 0) + abs(m["end_error_sec"] or 0) for m in metrics))


def choose_candidate(records, candidates):
    eligible = [r for r in records if r.get("eligible", False)]
    if not eligible:
        return None
    scored = []
    for index, candidate in enumerate(candidates[:32]):
        metrics = [score_boundaries(replay_frames(r["frames"], candidate, keep_trace=False)["events"],
                                    r["start"], r["end"]) for r in eligible]
        scored.append((_objective(metrics), index))
    return candidates[min(scored)[1]]


def select_knee_candidate(records, base):
    eligible = [r for r in records if r.get("eligible", False)]
    if len({r["source_id"] for r in eligible}) < 3:
        return None, []
    candidates = [replace(base, start_hold_sec=start, end_hold_sec=end)
                  for start, end in itertools.product([.1, .13], [.35, .5])]
    folds = []
    for train, heldout in grouped_folds(eligible):
        selected = choose_candidate(train, candidates)
        metrics = [score_boundaries(replay_frames(r["frames"], selected, keep_trace=False)["events"],
                                    r["start"], r["end"]) for r in heldout]
        folds.append({"heldout_source": heldout[0]["source_id"],
                      "train_sources": sorted({r["source_id"] for r in train}),
                      "config": selected.to_dict(), "validation": metrics})
    # Final selection can use all eligible sources, but its own metrics are
    # post-selection/in-sample. Only the fold records estimate heldout behavior.
    return choose_candidate(eligible, candidates), folds


def compare_records(records, previous, candidate):
    comparisons, traces = [], {}
    for record in records:
        old = replay_frames(record["frames"], previous, baseline=True)
        new = replay_frames(record["frames"], candidate)
        row = {k: record[k] for k in ["source_id", "variant", "start", "end", "eligible"]}
        row.update({"boundary_metrics_applicable": bool(record["eligible"]),
                    "before": score_boundaries(old["events"], record["start"], record["end"]),
                    "after": score_boundaries(new["events"], record["start"], record["end"]),
                    "candidate_calibrated": new["reference_revision"] > 0,
                    "candidate_events": new["events"], "eof": new["eof"]})
        comparisons.append(row)
        traces[record["cache_id"]] = {"before": old, "after": new}
    return comparisons, traces


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_annotations(path, video_root=None):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    result = []
    for row in rows:
        video = ((Path(video_root) / Path(row["video_path"]).name) if video_root
                 else (Path(path).parent / row["video_path"])).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Missing video: {video}; use --video-root")
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if not (np.isfinite([start, end]).all() and 0 <= start < end):
            raise ValueError(f"Invalid boundaries in annotation for {video.name}")
        # Human-reviewed protocol eligibility is independent of detector output.
        eligible = row.get("knee_protocol", "").strip().lower() in {"yes", "true", "1"}
        result.append({"video": video, "start": start, "end": end,
                       "source_id": _hash(video), "eligible": eligible})
    if len({r["source_id"] for r in result}) != len(result):
        raise ValueError("Duplicate original videos: keep one annotation per source SHA")
    return result


def build_capture_manifest(annotations, out, variants):
    assets = ["webservice/static/landmark_payload.mjs", "webservice/static/browser_tracker.mjs",
              "webservice/static/replay.mjs", "webservice/vendor/mediapipe/vision_bundle.mjs",
              "webservice/vendor/mediapipe/hand_landmarker.task",
              "webservice/vendor/mediapipe/pose_landmarker_lite.task"]
    assets.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "webservice/vendor/mediapipe/wasm").glob("*")) if p.is_file())
    provenance = {"extractor": "browser-mediapipe", "mode": "VIDEO", "delegate": "CPU",
                  "asset_verification": "served-sha256-v1",
                  "asset_sha256": {p: _hash(ROOT / p) for p in assets}}
    items = []
    for row in annotations:
        for variant in variants:
            key = hashlib.sha256(json.dumps([row["source_id"], variant, provenance], sort_keys=True).encode()).hexdigest()
            items.append({"id": key, "source_path": str(row["video"]), "source_id": row["source_id"],
                          "options": variant, "output": str((out / "private_cache" / (key + ".json")).resolve())})
    return {"provenance": provenance, "items": items}


def overlay_video(video_path, target, ground_truth, before, after):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create overlay video")
    traces = after["trace"]
    index = trace_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            while trace_index + 1 < len(traces) and traces[trace_index + 1]["t"] <= t:
                trace_index += 1
            cv2.rectangle(frame, (0, 0), (width, 120), (20, 20, 20), -1)
            state = traces[trace_index]["state"] if traces else "NO DATA"
            cv2.putText(frame, f"{t:.2f}s  {state}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            rows = [("GT", [ground_truth], (0, 220, 255)),
                    ("OLD", [(e["segment"]["clip_start_sec"], e["segment"]["clip_end_sec"]) for e in before["events"]], (80, 80, 255)),
                    ("NEW", [(e["segment"]["clip_start_sec"], e["segment"]["clip_end_sec"]) for e in after["events"] if e["accepted"]], (255, 220, 50))]
            for row, (name, spans, color) in enumerate(rows):
                y = 47 + row * 24
                cv2.putText(frame, name, (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
                for a, b in spans:
                    cv2.line(frame, (65 + int(a / duration * (width - 80)), y),
                             (65 + int(b / duration * (width - 80)), y), color, 6)
                x = 65 + int(t / duration * (width - 80))
                cv2.line(frame, (x, y - 7), (x, y + 7), (255, 255, 255), 1)
            writer.write(frame)
            index += 1
    finally:
        writer.release()
        cap.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/annotations/auto_trigger_three_videos.csv")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/web_knee_candidate")
    parser.add_argument("--url", default="http://127.0.0.1:8642")
    parser.add_argument("--node", default="node")
    parser.add_argument("--reuse-cache", action="store_true", help="Require complete, fingerprint-matching browser caches")
    parser.add_argument("--original-only", action="store_true", help="Smoke test only; not the 20-variant benchmark")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args(argv)
    loopback_endpoint(args.url)
    args.url = args.url.rstrip("/")
    args.out_dir = args.out_dir.resolve()
    annotations = load_annotations(args.annotations, args.video_root)
    variants = make_variants()[:1] if args.original_only else make_variants()
    manifest = build_capture_manifest(annotations, args.out_dir, variants)
    manifest_path = args.out_dir / "private_capture_manifest.json"
    _write(manifest_path, manifest)
    if not args.reuse_cache:
        with local_service(args.url, args.out_dir):
            subprocess.run([args.node, str(ROOT / "scripts/capture_web_trigger_videos.cjs"),
                            str(manifest_path), args.url], check=True, cwd=ROOT)
    current = load_web_trigger_config(ROOT / "configs/auto_trigger_knee_web_live.json")
    previous = load_web_trigger_config(ROOT / "configs/auto_trigger_knee_web_previous.json")
    source_map = {a["source_id"]: a for a in annotations}
    records = []
    browser_versions = set()
    for item in manifest["items"]:
        capture = json.loads(Path(item["output"]).read_text(encoding="utf-8"))
        if (capture.get("cache_key") != item["id"] or capture.get("provenance") != manifest["provenance"]
                or capture.get("asset_verification") != "served-sha256-v1" or not capture.get("browser_version")):
            raise ValueError("Cache provenance mismatch: recapture required")
        browser_versions.add(capture["browser_version"])
        source = source_map[item["source_id"]]
        speed = item["options"]["speed"]
        record = {"source_id": item["source_id"], "variant": item["options"]["name"],
                  "start": source["start"] / speed, "end": source["end"] / speed,
                  "eligible": source["eligible"] and item["options"]["name"] not in {"lower_crop", "knee_visibility_loss"},
                  "frames": capture["frames"], "cache_id": item["id"]}
        records.append(record)
    knee_winner, knee_folds = (None, []) if args.skip_tuning else select_knee_candidate(records, current)
    candidate = knee_winner or current
    comparisons, traces = compare_records(records, previous, candidate)
    for record in records:
        trace = traces[record["cache_id"]]
        _write(args.out_dir / "private_traces" / (record["cache_id"] + ".json"), trace)
        if record["variant"] == "original" and not args.skip_overlays:
            source = source_map[record["source_id"]]
            overlay_video(source["video"], args.out_dir / "private_overlays" / (record["source_id"][:12] + ".mp4"),
                          (source["start"], source["end"]), trace["before"], trace["after"])
    # Limit the search, freeze folds before selection, and never call a
    # no-knees rejection a successful learned knee boundary.
    candidates = [replace(previous, start_hold_sec=start, end_hold_sec=end, pre_roll_sec=pre,
                          reference_rest_distance_threshold=dist)
                  for start, end, pre, dist in itertools.product([.07, .13], [.35, .5], [.1, .18], [.18, .28])]
    folds = []
    if not args.skip_tuning and len(annotations) >= 3:
        diagnostics = [dict(r, eligible=True) for r in records
                       if r["variant"] not in {"lower_crop", "knee_visibility_loss", "hand_gaps", "fps15_hand_gaps"}]
        for train, validation in grouped_folds(diagnostics):
            winner = choose_candidate(train, candidates)
            assert winner is not None
            metrics = [score_boundaries(replay_frames(r["frames"], winner, keep_trace=False)["events"], r["start"], r["end"])
                       for r in validation]
            folds.append({"heldout_source": validation[0]["source_id"], "train_sources": sorted({r["source_id"] for r in train}),
                          "config": winner.to_dict(), "validation": metrics, "diagnostic_only": True})
            print(f"grouped diagnostic fold {len(folds)} / {len(annotations)}", flush=True)
    _write(args.out_dir / "candidate_config.json", candidate.to_dict())
    summary = {"schema_version": 1, "baseline_commit": "0ae3ac0", "mode": "browser VIDEO CPU",
               "original_videos": len(annotations), "variants_per_original": len(variants),
               "candidate_config": candidate.to_dict(),
               "knee_protocol_eligible_originals": sum(a["eligible"] for a in annotations),
               "knee_tuned": knee_winner is not None, "candidate_promoted": False,
               "candidate_metrics_role": "post_selection_in_sample" if knee_winner else "engineering_default",
               "warning": "Augmentations are not independent people. Missing-knee clips cannot validate real knee boundaries. Diagnostic settings are never installed.",
               "comparisons": comparisons, "grouped_diagnostic_folds": folds,
               "grouped_knee_folds": knee_folds, "browser_versions": sorted(browser_versions),
               "evaluation_sha256": {p: _hash(ROOT / p) for p in [
                   "recognition/transformer/live_trigger.py", "recognition/transformer/knee_geometry.py",
                   "recognition/realtime/auto_trigger.py", "recognition/realtime/knee42_controllers.py",
                   "recognition/transformer/landmarks.py", "webservice/server.py",
                   "recognition/evaluation/web_trigger_replay.py",
                   "recognition/evaluation/baselines/web_trigger_0ae3ac0.py"]},
               "provenance": manifest["provenance"]}
    _write(args.out_dir / "summary.json", summary)
    lines = ["# Web knee-trigger candidate replay", "", summary["warning"], "",
             f"Originals: {len(annotations)}; variants each: {len(variants)}; real knee-eligible originals: {summary['knee_protocol_eligible_originals']}.",
             "", "New boundary pass is N/A for clips without a human-confirmed knee-return protocol; raw errors remain diagnostic only.",
             "", "| Source | Variant | Old boundary pass | New boundary pass | New calibrated | EOF |",
             "|---|---|---|---|---|---|"]
    lines.extend(f"| {r['source_id'][:8]} | {r['variant']} | {r['before']['passed']} | {r['after']['passed'] if r['boundary_metrics_applicable'] else 'N/A'} | {r['candidate_calibrated']} | {r['eof']} |" for r in comparisons)
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
