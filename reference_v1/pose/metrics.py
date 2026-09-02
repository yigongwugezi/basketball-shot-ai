from __future__ import annotations

import math
from typing import Any

import numpy as np


JOINTS = {
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

BONES = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

ANGLES = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
}


def pose_arrays(rows: list[dict[str, Any]], pose_key: str = "pose") -> tuple[np.ndarray, np.ndarray]:
    count = len(rows)
    points = np.full((count, 17, 2), np.nan, dtype=float)
    confidence = np.zeros((count, 17), dtype=float)
    for offset, row in enumerate(rows):
        pose = row.get(pose_key)
        if not pose:
            continue
        xy = np.asarray(pose["keypoints"], dtype=float)
        scores = np.asarray(pose.get("temporal_reliability", pose.get("confidence", [])), dtype=float)
        usable = min(17, len(xy), len(scores))
        if usable:
            points[offset, :usable] = xy[:usable, :2]
            confidence[offset, :usable] = scores[:usable]
    return points, confidence


def body_scale(points: np.ndarray, confidence: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    scales = np.full(len(points), np.nan, dtype=float)
    for index, (xy, scores) in enumerate(zip(points, confidence)):
        if np.all(scores[[5, 6, 11, 12]] >= threshold):
            shoulders = (xy[5] + xy[6]) / 2
            hips = (xy[11] + xy[12]) / 2
            value = float(np.linalg.norm(shoulders - hips))
            if value > 1:
                scales[index] = value
    valid = np.isfinite(scales)
    if np.any(valid):
        scales[~valid] = float(np.median(scales[valid]))
    return scales


def joint_angle(points: np.ndarray, indices: tuple[int, int, int]) -> float:
    first = points[indices[0]] - points[indices[1]]
    second = points[indices[2]] - points[indices[1]]
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-6:
        return math.nan
    cosine = float(np.clip(np.dot(first, second) / denominator, -1, 1))
    return float(np.degrees(np.arccos(cosine)))


def angle_series(
    points: np.ndarray,
    confidence: np.ndarray,
    indices: tuple[int, int, int],
    threshold: float = 0.25,
) -> np.ndarray:
    values = np.full(len(points), np.nan, dtype=float)
    for frame, (xy, scores) in enumerate(zip(points, confidence)):
        if np.all(scores[list(indices)] >= threshold) and np.all(np.isfinite(xy[list(indices)])):
            values[frame] = joint_angle(xy, indices)
    return values


def _median(values: list[float] | np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else None


def _p95(values: list[float] | np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, 95)) if len(array) else None


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=0)
    result = np.full(values.shape[1], np.nan, dtype=float)
    usable = counts > 0
    result[usable] = np.nansum(values[:, usable], axis=0) / counts[usable]
    return result


def evaluate_pose_rows(
    rows: list[dict[str, Any]],
    *,
    pose_key: str = "pose",
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    points, confidence = pose_arrays(rows, pose_key)
    model_confidence = np.zeros_like(confidence)
    for offset, row in enumerate(rows):
        pose = row.get(pose_key)
        if not pose:
            continue
        scores = np.asarray(pose.get("keypoint_confidence", pose.get("confidence", [])), dtype=float)
        usable = min(17, len(scores))
        model_confidence[offset, :usable] = scores[:usable]
    scales = body_scale(points, confidence, confidence_threshold)
    focus = np.asarray(list(JOINTS.values()))
    visible = confidence[:, focus] >= confidence_threshold
    steps: list[float] = []
    accelerations: list[float] = []
    isolated_jumps = 0
    for frame in range(1, len(points)):
        valid = visible[frame - 1] & visible[frame] & np.isfinite(scales[frame])
        if np.any(valid):
            delta = np.linalg.norm(points[frame, focus[valid]] - points[frame - 1, focus[valid]], axis=1)
            steps.extend((delta / scales[frame]).tolist())
    for frame in range(1, len(points) - 1):
        valid = visible[frame - 1] & visible[frame] & visible[frame + 1] & np.isfinite(scales[frame])
        if not np.any(valid):
            continue
        center = points[frame, focus[valid]]
        predicted = (points[frame - 1, focus[valid]] + points[frame + 1, focus[valid]]) / 2
        residual = np.linalg.norm(center - predicted, axis=1) / scales[frame]
        accelerations.extend(residual.tolist())
        isolated_jumps += int(np.sum(residual > 0.35))

    angle_second_differences: list[float] = []
    angle_derivative_noise: list[float] = []
    for indices in ANGLES.values():
        values = angle_series(points, confidence, indices, confidence_threshold)
        for frame in range(1, len(values) - 1):
            window = values[frame - 1 : frame + 2]
            if np.all(np.isfinite(window)):
                angle_second_differences.append(abs(float(window[0] - 2 * window[1] + window[2])))
        velocity = np.diff(values)
        for frame, value in enumerate(velocity):
            local = velocity[max(0, frame - 2) : min(len(velocity), frame + 3)]
            local = local[np.isfinite(local)]
            if np.isfinite(value) and len(local) >= 3:
                angle_derivative_noise.append(abs(float(value - np.median(local))))

    bone_cvs: list[float] = []
    for first, second in BONES:
        valid = (confidence[:, first] >= confidence_threshold) & (confidence[:, second] >= confidence_threshold)
        lengths = np.linalg.norm(points[valid, first] - points[valid, second], axis=1)
        if len(lengths) >= 3 and float(np.mean(lengths)) > 1:
            bone_cvs.append(float(np.std(lengths) / np.mean(lengths)))

    confidences = confidence[:, focus]
    model_confidences = model_confidence[:, focus]
    found = np.any(confidences >= confidence_threshold, axis=1)
    status_counts: dict[str, int] = {}
    for row in rows:
        pose = row.get(pose_key) or {}
        status = pose.get("correction_status")
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "frames": len(rows),
        "pose_coverage": float(np.mean(found)) if len(rows) else 0.0,
        "visible_joint_coverage": float(np.mean(visible)) if visible.size else 0.0,
        "mean_keypoint_confidence": float(np.mean(model_confidences)) if model_confidences.size else 0.0,
        "median_normalized_displacement": _median(steps),
        "p95_normalized_displacement": _p95(steps),
        "median_normalized_second_difference": _median(accelerations),
        "p95_normalized_second_difference": _p95(accelerations),
        "median_joint_angle_second_difference_degrees": _median(angle_second_differences),
        "median_joint_angle_derivative_noise_degrees_per_frame": _median(angle_derivative_noise),
        "large_jump_outliers": isolated_jumps,
        "low_confidence_joint_frames": int(np.size(visible) - np.sum(visible)),
        "median_bone_length_cv": _median(bone_cvs),
        "correction_status_counts": status_counts,
    }


def estimate_signal_lag(raw: np.ndarray, filtered: np.ndarray, max_lag: int = 4) -> int | None:
    best: tuple[float, int] | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            first, second = raw[-lag:], filtered[:lag]
        elif lag > 0:
            first, second = raw[:-lag], filtered[lag:]
        else:
            first, second = raw, filtered
        valid = np.isfinite(first) & np.isfinite(second)
        if np.sum(valid) < 5:
            continue
        a = first[valid] - np.mean(first[valid])
        b = second[valid] - np.mean(second[valid])
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        score = float(np.dot(a, b) / denominator) if denominator > 1e-9 else -math.inf
        candidate = (score, -abs(lag), -lag)
        if best is None or candidate > (best[0], -abs(best[1]), -best[1]):
            best = (score, lag)
    return best[1] if best else None


def no_lag_metrics(
    raw_rows: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    shooting_side: str,
) -> dict[str, Any]:
    raw_xy, raw_conf = pose_arrays(raw_rows, "raw_pose")
    clean_xy, clean_conf = pose_arrays(analysis_rows, "analysis_pose")
    raw_xy[raw_conf < 0.25] = np.nan
    clean_xy[clean_conf < 0.25] = np.nan
    side = "left" if shooting_side == "left" else "right"
    wrist = JOINTS[f"{side}_wrist"]
    elbow_indices = ANGLES[f"{side}_elbow"]
    signals = {
        "wrist_x": (raw_xy[:, wrist, 0], clean_xy[:, wrist, 0]),
        "wrist_y": (raw_xy[:, wrist, 1], clean_xy[:, wrist, 1]),
        "elbow_angle": (
            angle_series(raw_xy, raw_conf, elbow_indices),
            angle_series(clean_xy, clean_conf, elbow_indices),
        ),
        "knee_angle": (
            _nanmean_rows(np.vstack([angle_series(raw_xy, raw_conf, ANGLES["left_knee"]), angle_series(raw_xy, raw_conf, ANGLES["right_knee"])])),
            _nanmean_rows(np.vstack([angle_series(clean_xy, clean_conf, ANGLES["left_knee"]), angle_series(clean_xy, clean_conf, ANGLES["right_knee"])])),
        ),
        "body_vertical": (
            _nanmean_rows(raw_xy[:, [11, 12], 1].T),
            _nanmean_rows(clean_xy[:, [11, 12], 1].T),
        ),
    }
    lags = {name: estimate_signal_lag(first, second) for name, (first, second) in signals.items()}
    finite = [abs(value) for value in lags.values() if value is not None]
    return {
        "signal_lag_frames": lags,
        "median_event_sensitive_displacement_frames": float(np.median(finite)) if finite else None,
        "max_event_sensitive_displacement_frames": max(finite) if finite else None,
    }
