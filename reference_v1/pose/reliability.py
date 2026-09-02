from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .metrics import BONES, JOINTS, body_scale, pose_arrays


CONFIDENCE_THRESHOLD = 0.25
MAX_GAP = 3
FOCUS = np.asarray(list(JOINTS.values()))
LEFT_RIGHT_PAIRS = ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def build_analysis_pose(
    rows: list[dict[str, Any]],
    *,
    raw_key: str = "raw_pose",
    max_gap: int = MAX_GAP,
) -> list[dict[str, Any]]:
    """Return copied rows with raw pose preserved and analysis_pose added."""
    output = copy.deepcopy(rows)
    for source, target in zip(rows, output):
        if raw_key not in target:
            target[raw_key] = copy.deepcopy(source.get("pose"))

    points, confidence = pose_arrays(output, raw_key)
    scales = body_scale(points, confidence)
    original_points = points.copy()
    original_confidence = confidence.copy()
    valid = confidence >= CONFIDENCE_THRESHOLD
    status = np.full((len(rows), 17), "unavailable", dtype=object)
    status[valid] = "observed"

    _reject_short_identity_breaks(points, valid, status, scales, max_gap)
    _correct_isolated_left_right_swaps(points, confidence, valid, status, scales)
    _reject_isolated_joint_spikes(points, valid, status, scales)
    _reject_isolated_bone_failures(points, valid, status)
    _fill_short_gaps(points, confidence, valid, status, max_gap)
    _adaptive_zero_phase_smooth(points, valid, status, scales)

    for frame, row in enumerate(output):
        raw_pose = row.get(raw_key)
        if raw_pose is None:
            row["analysis_pose"] = None
            continue
        reliability = np.zeros(17, dtype=float)
        for joint in range(17):
            if not valid[frame, joint]:
                continue
            if status[frame, joint] == "observed":
                reliability[joint] = original_confidence[frame, joint]
            elif status[frame, joint] == "interpolated":
                reliability[joint] = 0.60
            else:
                reliability[joint] = min(max(original_confidence[frame, joint], 0.55), 0.85)
        focus_status = status[frame, FOCUS]
        if not np.any(valid[frame, FOCUS]):
            frame_status = "unavailable"
        elif np.any(focus_status == "interpolated"):
            frame_status = "interpolated"
        elif np.any(focus_status == "corrected"):
            frame_status = "corrected"
        else:
            frame_status = "observed"
        pose = copy.deepcopy(raw_pose)
        pose.update(
            {
                "keypoints": np.where(np.isfinite(points[frame]), points[frame], 0.0).tolist(),
                "keypoint_confidence": original_confidence[frame].tolist(),
                "temporal_reliability": reliability.tolist(),
                "confidence": reliability.tolist(),
                "joint_status": status[frame].tolist(),
                "correction_status": frame_status,
                "visible_keypoints": int(np.sum(reliability >= CONFIDENCE_THRESHOLD)),
                "provenance": [
                    "yolo11_pose_raw",
                    "confidence_gate",
                    "anatomical_temporal_sanity",
                    "bounded_interpolation",
                    "adaptive_zero_phase_smoothing",
                ],
            }
        )
        row["analysis_pose"] = pose

    # A hard invariant: processing must not mutate or alias raw evidence.
    for source, target in zip(rows, output):
        source_raw = source.get(raw_key, source.get("pose"))
        if source_raw != target.get(raw_key):
            raise RuntimeError("raw_pose changed during reliability processing")
        if target.get("analysis_pose") is target.get(raw_key):
            raise RuntimeError("analysis_pose aliases raw_pose")
    return output


def _reject_short_identity_breaks(
    points: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
    scales: np.ndarray,
    max_gap: int,
) -> None:
    centers = np.full((len(points), 2), np.nan, dtype=float)
    for frame in range(len(points)):
        if np.all(valid[frame, [5, 6, 11, 12]]):
            centers[frame] = np.mean(points[frame, [5, 6, 11, 12]], axis=0)
    boundaries = []
    for frame in range(1, len(points)):
        if np.all(np.isfinite(centers[frame - 1 : frame + 1])) and np.isfinite(scales[frame]):
            step = float(np.linalg.norm(centers[frame] - centers[frame - 1]) / scales[frame])
            if step > 1.0:
                boundaries.append(frame)
    consumed_until = -1
    for start in boundaries:
        if start <= consumed_until:
            continue
        anchor = centers[start - 1]
        returned = next(
            (
                end
                for end in range(start + 1, min(len(points), start + max_gap + 1))
                if np.all(np.isfinite(centers[end]))
                and np.linalg.norm(centers[end] - anchor) / scales[end] < 1.0
            ),
            None,
        )
        end = returned if returned is not None else len(points)
        points[start:end, :] = np.nan
        valid[start:end, :] = False
        status[start:end, :] = "corrected" if returned is not None else "unavailable"
        consumed_until = end - 1


def _correct_isolated_left_right_swaps(
    points: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
    scales: np.ndarray,
) -> None:
    for frame in range(1, len(points) - 1):
        if not np.isfinite(scales[frame]):
            continue
        for left, right in LEFT_RIGHT_PAIRS:
            indices = [left, right]
            if not np.all(valid[frame - 1 : frame + 2, indices]):
                continue
            expected = (points[frame - 1, indices] + points[frame + 1, indices]) / 2
            direct = float(np.sum(np.linalg.norm(points[frame, indices] - expected, axis=1)))
            swapped = float(np.sum(np.linalg.norm(points[frame, indices[::-1]] - expected, axis=1)))
            if direct > 0.55 * scales[frame] and swapped + 0.20 * scales[frame] < direct:
                points[frame, indices] = points[frame, indices[::-1]]
                confidence[frame, indices] = confidence[frame, indices[::-1]]
                status[frame, indices] = "corrected"


def _reject_isolated_joint_spikes(
    points: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
    scales: np.ndarray,
) -> None:
    for frame in range(1, len(points) - 1):
        if not np.isfinite(scales[frame]):
            continue
        for joint in FOCUS:
            if not np.all(valid[frame - 1 : frame + 2, joint]):
                continue
            expected = (points[frame - 1, joint] + points[frame + 1, joint]) / 2
            residual = float(np.linalg.norm(points[frame, joint] - expected) / scales[frame])
            bridge = float(np.linalg.norm(points[frame + 1, joint] - points[frame - 1, joint]) / scales[frame])
            if residual > 0.35 and bridge < 0.45:
                points[frame, joint] = np.nan
                valid[frame, joint] = False
                status[frame, joint] = "corrected"


def _reject_isolated_bone_failures(
    points: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
) -> None:
    for first, second in BONES:
        usable = valid[:, first] & valid[:, second]
        lengths = np.full(len(points), np.nan, dtype=float)
        lengths[usable] = np.linalg.norm(points[usable, first] - points[usable, second], axis=1)
        reference = float(np.nanmedian(lengths))
        if not np.isfinite(reference) or reference <= 1:
            continue
        for frame in range(1, len(points) - 1):
            if not np.all(np.isfinite(lengths[frame - 1 : frame + 2])):
                continue
            ratio = lengths[frame] / reference
            neighbors_ok = np.all((lengths[[frame - 1, frame + 1]] / reference > 0.55) & (lengths[[frame - 1, frame + 1]] / reference < 1.8))
            if neighbors_ok and (ratio < 0.35 or ratio > 2.2):
                points[frame, second] = np.nan
                valid[frame, second] = False
                status[frame, second] = "corrected"


def _fill_short_gaps(
    points: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
    max_gap: int,
) -> None:
    for joint in range(17):
        frame = 0
        while frame < len(points):
            if valid[frame, joint]:
                frame += 1
                continue
            start = frame
            while frame < len(points) and not valid[frame, joint]:
                frame += 1
            end = frame
            length = end - start
            if start == 0 or end == len(points) or length > max_gap:
                status[start:end, joint] = "unavailable"
                continue
            if not valid[start - 1, joint] or not valid[end, joint]:
                status[start:end, joint] = "unavailable"
                continue
            for offset, current in enumerate(range(start, end), start=1):
                weight = offset / (length + 1)
                points[current, joint] = (1 - weight) * points[start - 1, joint] + weight * points[end, joint]
                confidence[current, joint] = min(confidence[start - 1, joint], confidence[end, joint])
                valid[current, joint] = True
                if status[current, joint] != "corrected":
                    status[current, joint] = "interpolated"


def _adaptive_zero_phase_smooth(
    points: np.ndarray,
    valid: np.ndarray,
    status: np.ndarray,
    scales: np.ndarray,
) -> None:
    original = points.copy()
    for frame in range(1, len(points) - 1):
        if not np.isfinite(scales[frame]):
            continue
        for joint in FOCUS:
            if not np.all(valid[frame - 1 : frame + 2, joint]):
                continue
            midpoint = (original[frame - 1, joint] + original[frame + 1, joint]) / 2
            speed = float(np.linalg.norm(original[frame + 1, joint] - original[frame - 1, joint]) / (2 * scales[frame]))
            curvature = float(np.linalg.norm(original[frame, joint] - midpoint) / scales[frame])
            blend = float(np.clip(0.60 - 1.8 * speed - 1.5 * curvature, 0.08, 0.60))
            smoothed = (1 - blend) * original[frame, joint] + blend * midpoint
            if np.linalg.norm(smoothed - original[frame, joint]) / scales[frame] > 0.005:
                if status[frame, joint] == "observed":
                    status[frame, joint] = "corrected"
            points[frame, joint] = smoothed
