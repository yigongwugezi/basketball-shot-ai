from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def decode_contact_transition_v1(
    evidence: list[dict[str, Any]],
    pose_release_frame: int,
    contact_threshold: float = 1.25,
    separation_threshold: float = 1.40,
    persistence: int = 2,
) -> dict[str, Any]:
    """Decode first persistent no-contact frame from cached hand/ball evidence."""
    rows = sorted(evidence, key=lambda row: int(row["frame_index"]))
    window = [
        row
        for row in rows
        if pose_release_frame - 10 <= int(row["frame_index"]) <= pose_release_frame + 12
    ]
    risks = ["experimental_geometry_proxy", "no_learned_hand_contact_model"]
    state = "unknown"
    contact_frame: int | None = None
    separation_frame: int | None = None

    for index, row in enumerate(window):
        frame = int(row["frame_index"])
        distance = row.get("ball_wrist_distance_diameters")
        if distance is not None and float(distance) <= contact_threshold:
            state = "contact_supported"
            contact_frame = frame
            continue
        if state != "contact_supported" or contact_frame is None or frame <= contact_frame:
            continue
        if frame < pose_release_frame - 3:
            continue

        candidate = window[index : index + persistence]
        if len(candidate) < persistence:
            break
        distances = [item.get("ball_wrist_distance_diameters") for item in candidate]
        centers = [item.get("ball_center") for item in candidate]
        frame_steps = [
            int(candidate[offset + 1]["frame_index"]) - int(candidate[offset]["frame_index"])
            for offset in range(len(candidate) - 1)
        ]
        if not all(value is not None and float(value) >= separation_threshold for value in distances):
            continue
        if not all(step == 1 for step in frame_steps) or not all(centers):
            continue
        if any(float(distances[offset + 1]) + 0.35 < float(distances[offset]) for offset in range(len(distances) - 1)):
            continue
        movement = sum(
            _distance(tuple(centers[offset]), tuple(centers[offset + 1]))
            for offset in range(len(centers) - 1)
        )
        if movement < 2.0:
            continue
        state = "released_confirmed"
        separation_frame = frame
        break

    if contact_frame is None:
        risks.append("no_contact_candidate")
    if separation_frame is None:
        risks.append("no_persistent_separation")
    elif abs(separation_frame - pose_release_frame) > 5:
        risks.append("large_pose_release_delta")
    return {
        "predicted_strict_frame": separation_frame,
        "pose_release_frame": pose_release_frame,
        "last_contact_supported_frame": contact_frame,
        "state": state,
        "risk_flags": risks,
        "status": "ok" if separation_frame is not None else "insufficient_data",
    }


def reanchor_track(
    tracker_rows: list[dict[str, Any]],
    detector_rows: list[dict[str, Any]],
    *,
    reanchor_interval: int = 5,
    max_displacement_diameters: float = 3.0,
    min_confidence: float = 0.15,
) -> list[dict[str, Any]]:
    """Fuse cached point tracks with trusted detector observations and abstention."""
    detector_by_frame = {int(row["frame_index"]): row for row in detector_rows}
    fused: list[dict[str, Any]] = []
    last_point: tuple[float, float] | None = None
    last_anchor_frame: int | None = None
    typical_diameters: list[float] = []

    for tracker in sorted(tracker_rows, key=lambda row: int(row["frame_index"])):
        frame = int(tracker["frame_index"])
        detection = detector_by_frame.get(frame)
        tracker_point = (
            (float(tracker["x"]), float(tracker["y"]))
            if tracker.get("visible") and tracker.get("x") is not None and tracker.get("y") is not None
            else None
        )
        trusted_detection = False
        detector_point: tuple[float, float] | None = None
        diameter: float | None = None
        if detection and detection.get("visible", True):
            confidence = float(detection.get("confidence", 1.0))
            diameter = float(detection.get("diameter") or 0.0)
            detector_point = (float(detection["x"]), float(detection["y"]))
            size_ok = diameter >= 3.0
            if typical_diameters and diameter:
                median_size = statistics.median(typical_diameters)
                size_ok = 0.35 * median_size <= diameter <= 2.8 * median_size
            roi = detection.get("roi")
            roi_ok = not roi or (
                float(roi[0]) <= detector_point[0] <= float(roi[2])
                and float(roi[1]) <= detector_point[1] <= float(roi[3])
            )
            displacement_ok = last_point is None or diameter <= 0 or (
                _distance(last_point, detector_point) / max(diameter, 1.0)
                <= max_displacement_diameters
            )
            trusted_detection = confidence >= min_confidence and size_ok and roi_ok and displacement_ok

        scheduled = last_anchor_frame is None or frame - last_anchor_frame >= reanchor_interval
        tracker_drifted = (
            tracker_point is not None
            and detector_point is not None
            and diameter is not None
            and diameter > 0
            and _distance(tracker_point, detector_point) / diameter > max_displacement_diameters
        )
        reanchored = trusted_detection and (scheduled or tracker_point is None or tracker_drifted)
        if reanchored:
            point = detector_point
            last_anchor_frame = frame
            typical_diameters.append(float(diameter))
            source = "detector_reanchor"
        elif tracker_point is not None and not tracker_drifted:
            point = tracker_point
            source = "tracker"
        elif trusted_detection:
            point = detector_point
            last_anchor_frame = frame
            typical_diameters.append(float(diameter))
            source = "detector_reinitialize"
            reanchored = True
        else:
            point = None
            source = "abstain"

        last_point = point
        fused.append(
            {
                "frame_index": frame,
                "x": point[0] if point else None,
                "y": point[1] if point else None,
                "visible": point is not None,
                "source": source,
                "reanchored": reanchored,
            }
        )
    return fused


def evaluate_ball_track(
    predicted_rows: list[dict[str, Any]], gt_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gt_by_frame = {int(row["frame_index"]): row for row in gt_rows}
    errors: list[float] = []
    normalized_errors: list[float] = []
    true_positive = false_positive = false_negative = true_negative = 0
    bad_drift = reanchors = usable = correct_abstentions = 0
    for row in predicted_rows:
        gt = gt_by_frame.get(int(row["frame_index"]))
        if not gt:
            continue
        gt_visible = str(gt.get("ball_visible", "yes")).lower() == "yes"
        predicted_visible = bool(row.get("visible"))
        reanchors += int(bool(row.get("reanchored")))
        if gt_visible and predicted_visible:
            true_positive += 1
            error = _distance(
                (float(row["x"]), float(row["y"])),
                (float(gt["ball_center_x"]), float(gt["ball_center_y"])),
            )
            diameter = float(gt.get("ball_diameter") or 1.0)
            errors.append(error)
            normalized_errors.append(error / max(diameter, 1.0))
            usable += int(error / max(diameter, 1.0) <= 2.0)
            bad_drift += int(error / max(diameter, 1.0) > 2.0)
        elif gt_visible:
            false_negative += 1
        elif predicted_visible:
            false_positive += 1
        else:
            true_negative += 1
            correct_abstentions += 1
    count = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "evaluated_frames": count,
        "median_center_error_px": statistics.median(errors) if errors else None,
        "mean_center_error_px": statistics.mean(errors) if errors else None,
        "median_error_ball_diameters": statistics.median(normalized_errors) if normalized_errors else None,
        "visibility_f1": f1,
        "track_survival": true_positive / count if count else None,
        "bad_drift_frames": bad_drift,
        "reanchor_count": reanchors,
        "release_window_usable_coverage": usable / count if count else None,
        "correct_abstentions": correct_abstentions,
    }


def evaluate_release_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[int] = []
    abstentions = correct_abstentions = 0
    for row in rows:
        truth = row.get("strict_release_frame")
        predicted = row.get("predicted_strict_frame")
        if predicted is None:
            abstentions += 1
            correct_abstentions += int(truth is None)
        elif truth is not None:
            errors.append(abs(int(predicted) - int(truth)))
    total = len(rows)
    return {
        "samples": total,
        "coverage": (total - abstentions) / total if total else None,
        "exact_frame": sum(error == 0 for error in errors) / len(errors) if errors else None,
        "within_1": sum(error <= 1 for error in errors) / len(errors) if errors else None,
        "within_2": sum(error <= 2 for error in errors) / len(errors) if errors else None,
        "within_3": sum(error <= 3 for error in errors) / len(errors) if errors else None,
        "median_absolute_frame_error": statistics.median(errors) if errors else None,
        "mean_absolute_frame_error": statistics.mean(errors) if errors else None,
        "abstention_rate": abstentions / total if total else None,
        "correct_abstention": correct_abstentions,
        "catastrophic_error_gt_5": sum(error > 5 for error in errors),
    }


def evaluate_reviewer_agreement(
    rows: list[dict[str, Any]], tolerance_frames: int = 1
) -> dict[str, Any]:
    comparable = []
    both_unavailable = disagreements = 0
    for row in rows:
        first = row.get("strict_release_frame")
        second = row.get("reviewer_2_strict_release_frame")
        first_status = row.get("review_status")
        second_status = row.get("reviewer_2_review_status")
        if first is None and second is None and first_status != "pending" and second_status != "pending":
            both_unavailable += 1
        elif first is not None and second is not None:
            comparable.append(abs(int(first) - int(second)))
        elif first_status != "pending" and second_status != "pending":
            disagreements += 1
    return {
        "comparable_frame_labels": len(comparable),
        "exact_agreement": sum(delta == 0 for delta in comparable),
        "within_tolerance": sum(delta <= tolerance_frames for delta in comparable),
        "tolerance_frames": tolerance_frames,
        "both_unavailable": both_unavailable,
        "availability_disagreements": disagreements,
        "pending_or_incomplete": len(rows) - len(comparable) - both_unavailable - disagreements,
    }


def evaluate_rtmw_hand_cache(
    cache_path: Path, release_frame: int, side: str
) -> dict[str, Any]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    hand_indices = range(91, 112) if side == "left" else range(112, 133)
    wrist_index = 9 if side == "left" else 10
    rows = [
        row
        for row in data["rows"]
        if release_frame - 12 <= int(row["frame_index"]) <= release_frame + 12
        and row.get("pose")
    ]
    previous_center: tuple[float, float] | None = None
    reviewed = []
    for row in rows:
        pose = row["pose"]
        points = pose["keypoints"]
        scores = pose["confidence"]
        hand_points = [points[index] for index in hand_indices]
        hand_scores = [float(scores[index]) for index in hand_indices]
        xs = [float(point[0]) for point in hand_points]
        ys = [float(point[1]) for point in hand_points]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        center = (statistics.mean(xs), statistics.mean(ys))
        box = pose["bbox"]
        scale = math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
        root = tuple(float(value) for value in hand_points[0])
        wrist = tuple(float(value) for value in points[wrist_index])
        root_distance = _distance(root, wrist) / max(scale, 1.0)
        jump = _distance(center, previous_center) / max(scale, 1.0) if previous_center else 0.0
        confidence_ok = statistics.median(hand_scores) >= 0.25
        usable = confidence_ok and span >= 18.0 and root_distance <= 0.15 and jump <= 0.20
        reviewed.append(
            {
                "frame_index": int(row["frame_index"]),
                "span_px": span,
                "median_confidence": statistics.median(hand_scores),
                "root_wrist_distance_normalized": root_distance,
                "temporal_jump_normalized": jump,
                "auto_proxy_usable": usable,
                "human_review_status": "pending",
            }
        )
        previous_center = center
    return {
        "sample": cache_path.stem,
        "side": side,
        "frames": len(reviewed),
        "auto_proxy_usable_frames": sum(row["auto_proxy_usable"] for row in reviewed),
        "auto_proxy_usable_rate": (
            sum(row["auto_proxy_usable"] for row in reviewed) / len(reviewed) if reviewed else None
        ),
        "human_review": "pending_not_accuracy",
        "rows": reviewed,
    }


def run_cached_track_evaluation(
    trajectory_path: Path,
    labels_path: Path,
    clip_id: str,
    output_path: Path,
    diameter_hint: float,
) -> None:
    with trajectory_path.open(encoding="utf-8-sig", newline="") as handle:
        trajectory = list(csv.DictReader(handle))
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = [row for row in csv.DictReader(handle) if row["clip_id"] == clip_id]
    gt_rows = []
    for row in labels:
        diameter = max(
            float(row["ball_x2"] or 0) - float(row["ball_x1"] or 0),
            float(row["ball_y2"] or 0) - float(row["ball_y1"] or 0),
        )
        gt_rows.append({**row, "ball_diameter": diameter})

    tracker_rows = [
        {
            "frame_index": int(row["frame_index"]),
            "x": float(row["tracker_x"]),
            "y": float(row["tracker_y"]),
            "visible": row["tracker_visible"].lower() == "true",
        }
        for row in trajectory
    ]
    detector_rows = [
        {
            "frame_index": int(row["frame_index"]),
            "x": float(row["detector_x"]) if row["detector_x"] else None,
            "y": float(row["detector_y"]) if row["detector_y"] else None,
            "visible": bool(row["detector_x"] and row["detector_y"]),
            "diameter": diameter_hint,
            "confidence": 1.0,
        }
        for row in trajectory
    ]
    detector_predictions = [
        {
            "frame_index": row["frame_index"],
            "x": row["x"],
            "y": row["y"],
            "visible": row["visible"],
            "source": "detector",
            "reanchored": False,
        }
        for row in detector_rows
    ]
    reanchored = reanchor_track(tracker_rows, detector_rows)
    payload = {
        "sample": trajectory_path.stem,
        "clip_id": clip_id,
        "ground_truth": "human_ball_center_labels_development_only",
        "diameter_hint_px": diameter_hint,
        "diameter_hint_note": "fixed development configuration; detector bbox was not retained in legacy cache",
        "detector_only": evaluate_ball_track(detector_predictions, gt_rows),
        "original_cotracker3": evaluate_ball_track(tracker_rows, gt_rows),
        "reanchored_cotracker3": evaluate_ball_track(reanchored, gt_rows),
        "rows": reanchored,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


def run_cached_reanchor_proxy(
    trajectory_path: Path, output_path: Path, diameter_hint: float
) -> None:
    with trajectory_path.open(encoding="utf-8-sig", newline="") as handle:
        trajectory = list(csv.DictReader(handle))
    tracker_rows = [
        {
            "frame_index": int(row["frame_index"]),
            "x": float(row["tracker_x"]),
            "y": float(row["tracker_y"]),
            "visible": row["tracker_visible"].lower() == "true",
        }
        for row in trajectory
    ]
    detector_rows = [
        {
            "frame_index": int(row["frame_index"]),
            "x": float(row["detector_x"]) if row["detector_x"] else None,
            "y": float(row["detector_y"]) if row["detector_y"] else None,
            "visible": bool(row["detector_x"] and row["detector_y"]),
            "diameter": diameter_hint,
            "confidence": 1.0,
        }
        for row in trajectory
    ]
    fused = reanchor_track(tracker_rows, detector_rows)
    detector_by_frame = {row["frame_index"]: row for row in detector_rows}
    proxy_distances = []
    for row in fused:
        detection = detector_by_frame[row["frame_index"]]
        if row["visible"] and detection["visible"]:
            proxy_distances.append(
                _distance((row["x"], row["y"]), (detection["x"], detection["y"]))
                / diameter_hint
            )
    payload = {
        "sample": trajectory_path.stem,
        "ground_truth": "none_detector_agreement_proxy_only",
        "frames": len(fused),
        "coverage": sum(row["visible"] for row in fused) / len(fused) if fused else None,
        "reanchor_count": sum(row["reanchored"] for row in fused),
        "abstention_count": sum(not row["visible"] for row in fused),
        "median_distance_to_detector_diameters": (
            statistics.median(proxy_distances) if proxy_distances else None
        ),
        "rows": fused,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


def run_cached_strict(input_dir: Path, output_path: Path) -> None:
    outputs = []
    truth_by_sample = {"img_7216_near_agreement": 138}
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        decoded = decode_contact_transition_v1(data["evidence"], int(data["pose_release_frame"]))
        decoded.update(
            {
                "sample_id": data["sample_id"],
                "strict_release_frame": truth_by_sample.get(data["sample_id"]),
                "historical_ball_release_frame": data.get("historical_ball_release_frame"),
                "evidence_status": (
                    "human_strict_gt" if data["sample_id"] in truth_by_sample else "historical_non_gt_reference"
                ),
            }
        )
        outputs.append(decoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    human_gt_rows = [row for row in outputs if row["evidence_status"] == "human_strict_gt"]
    payload = {
        "samples": outputs,
        "human_gt_metrics": evaluate_release_predictions(human_gt_rows),
        "human_gt_sample_count": len(human_gt_rows),
        "historical_references_are_not_gt": True,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def self_test() -> None:
    evidence = []
    for frame, distance in enumerate([0.8, 0.7, 1.6, 2.0, 2.5], start=8):
        evidence.append(
            {
                "frame_index": frame,
                "ball_wrist_distance_diameters": distance,
                "ball_center": [float(frame * 4), 10.0],
            }
        )
    decoded = decode_contact_transition_v1(evidence, 10)
    assert decoded["predicted_strict_frame"] == 10

    tracker = [
        {"frame_index": 0, "x": 10.0, "y": 10.0, "visible": True},
        {"frame_index": 1, "x": 90.0, "y": 90.0, "visible": True},
    ]
    detector = [
        {"frame_index": 0, "x": 10.0, "y": 10.0, "diameter": 10.0, "confidence": 0.9},
        {"frame_index": 1, "x": 12.0, "y": 10.0, "diameter": 10.0, "confidence": 0.9},
    ]
    fused = reanchor_track(tracker, detector)
    assert fused[1]["source"] == "detector_reanchor"
    agreement = evaluate_reviewer_agreement(
        [
            {
                "strict_release_frame": 12,
                "reviewer_2_strict_release_frame": 13,
                "review_status": "reviewed",
                "reviewer_2_review_status": "reviewed",
            }
        ]
    )
    assert agreement["within_tolerance"] == 1
    print("validation closure self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference V1 validation closure utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    strict = subparsers.add_parser("strict-cache")
    strict.add_argument("--input-dir", type=Path, required=True)
    strict.add_argument("--output", type=Path, required=True)
    hand = subparsers.add_parser("rtmw-cache")
    hand.add_argument("--cache", type=Path, required=True)
    hand.add_argument("--release-frame", type=int, required=True)
    hand.add_argument("--side", choices=["left", "right"], required=True)
    hand.add_argument("--output", type=Path, required=True)
    track = subparsers.add_parser("track-cache")
    track.add_argument("--trajectory", type=Path, required=True)
    track.add_argument("--labels", type=Path, required=True)
    track.add_argument("--clip-id", required=True)
    track.add_argument("--diameter-hint", type=float, default=60.0)
    track.add_argument("--output", type=Path, required=True)
    proxy = subparsers.add_parser("reanchor-cache")
    proxy.add_argument("--trajectory", type=Path, required=True)
    proxy.add_argument("--diameter-hint", type=float, default=60.0)
    proxy.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "strict-cache":
        run_cached_strict(args.input_dir, args.output)
    elif args.command == "rtmw-cache":
        result = evaluate_rtmw_hand_cache(args.cache, args.release_frame, args.side)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    elif args.command == "track-cache":
        run_cached_track_evaluation(
            args.trajectory,
            args.labels,
            args.clip_id,
            args.output,
            args.diameter_hint,
        )
    else:
        run_cached_reanchor_proxy(args.trajectory, args.output, args.diameter_hint)


if __name__ == "__main__":
    main()
