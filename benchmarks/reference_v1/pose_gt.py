from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable


JOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
VISIBILITY = {"visible", "occluded_but_inferable", "not_labelable"}


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "pose_gt_manifest_v1":
        raise ValueError("Unexpected pose GT manifest schema")
    if manifest.get("joint_set") != list(JOINT_INDEX):
        raise ValueError("Manifest must use the ordered 12-joint evaluation set")
    if set(manifest.get("visibility_values", [])) != VISIBILITY:
        raise ValueError("Manifest visibility taxonomy is incomplete")
    ids = [frame.get("id") for frame in manifest.get("frames", [])]
    if not 30 <= len(ids) <= 60 or len(ids) != len(set(ids)):
        raise ValueError("Pose GT manifest must contain 30-60 unique frame ids")
    for frame in manifest["frames"]:
        clip = manifest["clips"].get(frame.get("clip"))
        if clip is None:
            raise ValueError(f"Unknown clip for {frame.get('id')}")
        expected = frame["frame_index"] / clip["fps"]
        if abs(expected - frame["timestamp_seconds"]) > 0.001:
            raise ValueError(f"Timestamp mismatch for {frame['id']}")


def validate_annotations(
    manifest: dict[str, Any], annotations: dict[str, Any], *, require_complete: bool = False
) -> None:
    validate_manifest(manifest)
    if annotations.get("schema_version") != "pose_gt_annotations_v1":
        raise ValueError("Unexpected pose GT annotation schema")
    if annotations.get("benchmark_version") != manifest["benchmark_version"]:
        raise ValueError("Annotation benchmark version does not match manifest")
    manifest_ids = {frame["id"] for frame in manifest["frames"]}
    seen: set[str] = set()
    for frame in annotations.get("frames", []):
        frame_id = frame.get("frame_id")
        if frame_id not in manifest_ids or frame_id in seen:
            raise ValueError(f"Unknown or duplicate annotation frame: {frame_id}")
        seen.add(frame_id)
        joints = frame.get("joints", {})
        if set(joints) != set(JOINT_INDEX):
            raise ValueError(f"Incomplete joint set for {frame_id}")
        for name, joint in joints.items():
            visibility = joint.get("visibility")
            if visibility not in VISIBILITY:
                raise ValueError(f"Invalid visibility for {frame_id}/{name}")
            has_xy = _finite(joint.get("x")) and _finite(joint.get("y"))
            if visibility == "not_labelable" and (joint.get("x") is not None or joint.get("y") is not None):
                raise ValueError(f"not_labelable joint has coordinates: {frame_id}/{name}")
            if visibility != "not_labelable" and not has_xy:
                raise ValueError(f"Labelable joint lacks coordinates: {frame_id}/{name}")
        if require_complete and not frame.get("reviewed"):
            raise ValueError(f"Frame has not been human-reviewed: {frame_id}")
    if require_complete and seen != manifest_ids:
        raise ValueError(f"Missing reviewed frames: {sorted(manifest_ids - seen)}")


def evaluate_predictions(
    manifest: dict[str, Any],
    annotations: dict[str, Any],
    predictions: Iterable[dict[str, Any]],
    *,
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    """Evaluate only explicitly reviewed human GT; predictions remain separate inputs."""
    validate_annotations(manifest, annotations, require_complete=True)
    frame_meta = {frame["id"]: frame for frame in manifest["frames"]}
    gt = {frame["frame_id"]: frame for frame in annotations["frames"]}
    payloads = {payload["model_id"]: payload for payload in predictions}
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_id, payload in payloads.items():
        predicted = {frame["frame_id"]: frame for frame in payload.get("frames", [])}
        for frame_id, gt_frame in gt.items():
            meta = frame_meta[frame_id]
            scale = body_scale(gt_frame["joints"])
            shooting_side = manifest["clips"][meta["clip"]]["shooting_side"]
            predicted_joints = predicted.get(frame_id, {}).get("joints", {})
            for joint_name, gt_joint in gt_frame["joints"].items():
                if gt_joint["visibility"] == "not_labelable":
                    continue
                item = predicted_joints.get(joint_name)
                detected = bool(
                    item
                    and _finite(item.get("x"))
                    and _finite(item.get("y"))
                    and float(item.get("confidence", 0.0)) >= confidence_threshold
                )
                error = None
                if detected:
                    error = math.hypot(item["x"] - gt_joint["x"], item["y"] - gt_joint["y"])
                records[model_id].append(
                    {
                        "frame_id": frame_id,
                        "clip": meta["clip"],
                        "tags": meta["tags"],
                        "joint": joint_name,
                        "shooting_side": shooting_side,
                        "detected": detected,
                        "pixel_error": error,
                        "normalized_error": error / scale if error is not None else None,
                    }
                )
    groups = {
        "all": lambda row: True,
        "easy": lambda row: "easy_static" in row["tags"],
        "fast_motion": lambda row: bool({"rapid_elbow_extension", "upward_drive"} & set(row["tags"])),
        "release_window": lambda row: "release_window" in row["tags"],
        "motion_blur": lambda row: "motion_blur" in row["tags"],
        "occlusion": lambda row: bool({"ball_hand_occlusion", "partial_occlusion"} & set(row["tags"])),
        "shooting_arm": lambda row: row["joint"] in {
            f"{row['shooting_side']}_shoulder", f"{row['shooting_side']}_elbow", f"{row['shooting_side']}_wrist"
        },
        "shooting_wrist": lambda row: row["joint"] == f"{row['shooting_side']}_wrist",
        "shooting_elbow": lambda row: row["joint"] == f"{row['shooting_side']}_elbow",
        "shoulders": lambda row: row["joint"].endswith("shoulder"),
        "knees_ankles": lambda row: row["joint"].endswith(("knee", "ankle")),
        "lower_body": lambda row: row["joint"].endswith(("hip", "knee", "ankle")),
    }
    accuracy = {
        model_id: {name: summarize([row for row in rows if predicate(row)]) for name, predicate in groups.items()}
        for model_id, rows in records.items()
    }
    return {
        "schema_version": "pose_accuracy_benchmark_v1",
        "benchmark_version": manifest["benchmark_version"],
        "ground_truth": {
            "type": "human_reviewed_joint_coordinates",
            "annotator": annotations.get("annotator"),
            "revision": annotations.get("revision"),
            "reviewed_frames": len(gt),
        },
        "confidence_threshold": confidence_threshold,
        "accuracy": accuracy,
        "filter_damage": filter_damage(records),
        "temporal_quality": {
            model_id: payload.get("temporal_quality", {}) for model_id, payload in payloads.items()
        },
        "limitations": [
            "small_stratified_benchmark_raw_numbers_no_significance_claim",
            "bili_005_a_is_slow_motion_contaminated_stress_test_not_generalization_evidence",
            "rtmpose_and_rtmw_use_yolo_person_bbox_and_are_pose_head_crop_comparisons",
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["pixel_error"] for row in rows if row["pixel_error"] is not None]
    normalized = [row["normalized_error"] for row in rows if row["normalized_error"] is not None]
    result: dict[str, Any] = {
        "labelable_joints": len(rows),
        "detected_joints": len(errors),
        "failure_rate": round(1 - len(errors) / len(rows), 4) if rows else None,
    }
    if not errors:
        result.update({key: None for key in ("mean_pixel_error", "median_pixel_error", "median_pixel_error_bootstrap_95_ci", "p90_pixel_error", "p95_pixel_error", "median_normalized_error", "median_normalized_error_bootstrap_95_ci", "p90_normalized_error", "p95_normalized_error")})
        result.update({key: 0.0 if rows else None for key in ("pck_005", "pck_010", "pck_020")})
        return result
    result.update(
        {
            "mean_pixel_error": round(statistics.fmean(errors), 3),
            "median_pixel_error": round(statistics.median(errors), 3),
            "median_pixel_error_bootstrap_95_ci": bootstrap_median_ci(errors),
            "p90_pixel_error": round(percentile(errors, 90), 3),
            "p95_pixel_error": round(percentile(errors, 95), 3),
            "median_normalized_error": round(statistics.median(normalized), 4),
            "median_normalized_error_bootstrap_95_ci": bootstrap_median_ci(normalized, digits=4),
            "p90_normalized_error": round(percentile(normalized, 90), 4),
            "p95_normalized_error": round(percentile(normalized, 95), 4),
            "pck_005": round(sum(value <= 0.05 for value in normalized) / len(rows), 4),
            "pck_010": round(sum(value <= 0.10 for value in normalized) / len(rows), 4),
            "pck_020": round(sum(value <= 0.20 for value in normalized) / len(rows), 4),
        }
    )
    return result


def filter_damage(records: dict[str, list[dict[str, Any]]], neutral_pixels: float = 2.0) -> dict[str, Any]:
    raw = {(row["frame_id"], row["joint"]): row for row in records.get("yolo_raw", [])}
    filtered = {(row["frame_id"], row["joint"]): row for row in records.get("yolo_filtered", [])}
    comparisons = []
    for key in raw.keys() & filtered.keys():
        first, second = raw[key], filtered[key]
        raw_error, filtered_error = first["pixel_error"], second["pixel_error"]
        delta = filtered_error - raw_error if raw_error is not None and filtered_error is not None else None
        if raw_error is None and filtered_error is not None:
            outcome = "help"
        elif raw_error is not None and filtered_error is None:
            outcome = "harm"
        elif raw_error is None and filtered_error is None:
            outcome = "neutral"
        elif delta < -neutral_pixels:
            outcome = "help"
        elif delta > neutral_pixels:
            outcome = "harm"
        else:
            outcome = "neutral"
        comparisons.append({**second, "raw_error": raw_error, "filtered_error": filtered_error, "delta_error": delta, "outcome": outcome})

    def damage(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"comparisons": 0, "filter_help_rate": None, "filter_harm_rate": None, "filter_neutral_rate": None, "median_delta_error": None}
        numeric = [row["delta_error"] for row in rows if row["delta_error"] is not None]
        return {
            "comparisons": len(rows),
            "paired_coordinate_comparisons": len(numeric),
            "raw_detected_filtered_missing": sum(row["raw_error"] is not None and row["filtered_error"] is None for row in rows),
            "raw_missing_filtered_detected": sum(row["raw_error"] is None and row["filtered_error"] is not None for row in rows),
            "filter_help_rate": round(sum(row["outcome"] == "help" for row in rows) / len(rows), 4),
            "filter_harm_rate": round(sum(row["outcome"] == "harm" for row in rows) / len(rows), 4),
            "filter_neutral_rate": round(sum(row["outcome"] == "neutral" for row in rows) / len(rows), 4),
            "median_delta_error": round(statistics.median(numeric), 3) if numeric else None,
        }

    subsets = {
        "all": comparisons,
        "shooting_wrist": [row for row in comparisons if row["joint"] == f"{row['shooting_side']}_wrist"],
        "shooting_elbow": [row for row in comparisons if row["joint"] == f"{row['shooting_side']}_elbow"],
        "shooting_shoulder": [row for row in comparisons if row["joint"] == f"{row['shooting_side']}_shoulder"],
        "release_window": [row for row in comparisons if "release_window" in row["tags"]],
        "rapid_extension": [row for row in comparisons if "rapid_elbow_extension" in row["tags"]],
    }
    harms = [row for row in comparisons if row["outcome"] == "harm"]
    helps = [row for row in comparisons if row["outcome"] == "help"]
    return {
        "neutral_band_pixels": neutral_pixels,
        "summary": {name: damage(rows) for name, rows in subsets.items()},
        "raw_better_examples": sorted(harms, key=lambda row: (row["filtered_error"] is None, row["delta_error"] or 0), reverse=True)[:12],
        "filtered_better_examples": sorted(helps, key=lambda row: (row["raw_error"] is None, -(row["delta_error"] or 0)), reverse=True)[:12],
    }


def body_scale(joints: dict[str, dict[str, Any]]) -> float:
    def point(name: str) -> tuple[float, float] | None:
        item = joints.get(name, {})
        if item.get("visibility") == "not_labelable" or not (_finite(item.get("x")) and _finite(item.get("y"))):
            return None
        return float(item["x"]), float(item["y"])

    shoulders = [point("left_shoulder"), point("right_shoulder")]
    hips = [point("left_hip"), point("right_hip")]
    torso = None
    if all(shoulders) and all(hips):
        shoulder_mid = ((shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2)
        hip_mid = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)
        torso = math.dist(shoulder_mid, hip_mid)
    points = [value for name in joints if (value := point(name))]
    if not points:
        raise ValueError("No labelable joints available for body scale")
    diagonal = math.hypot(max(x for x, _ in points) - min(x for x, _ in points), max(y for _, y in points) - min(y for _, y in points))
    return max(torso or 0.0, diagonal * 0.25, 1.0)


def map_crop_points(points: list[list[float]], crop_bbox: list[float], input_size: tuple[int, int]) -> list[list[float]]:
    """Map crop-input coordinates back to original-frame pixels."""
    x1, y1, x2, y2 = crop_bbox
    width, height = input_size
    return [[x1 + point[0] * (x2 - x1) / width, y1 + point[1] * (y2 - y1) / height] for point in points]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_median_ci(values: list[float], *, samples: int = 1000, digits: int = 3) -> list[float]:
    rng = random.Random(0)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [round(percentile(medians, 2.5), digits), round(percentile(medians, 97.5), digits)]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
