from __future__ import annotations

import math
from typing import Any

import numpy as np


VERSION = "human_ball_release_v1"
MAX_INTERPOLATION_GAP = 2
PERSISTENCE = 3
JOINTS = {
    "left": {"shoulder": 5, "elbow": 7, "wrist": 9},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10},
}


def build_human_ball_release(
    rows: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    shooting_side: str,
    pose_key: str,
    pose_only_index: int,
    release_score: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    pose_only_frame = int(signals[pose_only_index]["frame_index"])
    start = max(int(rows[0]["frame_index"]), pose_only_frame - 18)
    end = min(int(rows[-1]["frame_index"]), pose_only_frame + 20)
    track, rejected = build_ball_track(rows, shooting_side, pose_key, start, end)
    relations = build_relations(rows, track, shooting_side, pose_key)
    states, strict = decode_contact_states(relations)
    by_frame = {item["frame"]: item for item in states}
    ball_rise = find_ball_rise(states)
    last_contact = max(
        (item["frame"] for item in states if item["contact_state"] == "LIKELY_CONTACT"),
        default=None,
    )
    strict_frame = strict.get("frame")
    release_region_start = last_contact if last_contact is not None else ball_rise
    release_region_end = strict_frame
    constrained_indices = [
        index
        for index, signal in enumerate(signals)
        if release_region_start is not None
        and release_region_end is not None
        and release_region_start <= int(signal["frame_index"]) <= release_region_end
    ]
    if constrained_indices:
        pose_index = max(constrained_indices, key=lambda index: float(release_score[index]))
        pose_source = "human_ball_constrained_pose_release_v1"
    else:
        pose_index = pose_only_index
        pose_source = "pose_only_fallback"
    pose_frame = int(signals[pose_index]["frame_index"])

    counts = {status: sum(item["ball_status"] == status for item in states) for status in ("DETECTED", "INTERPOLATED", "MISSING", "AMBIGUOUS")}
    supported = counts["DETECTED"] + counts["INTERPOLATED"] + counts["AMBIGUOUS"]
    coverage = supported / len(states) if states else 0.0
    contact_coverage = sum(item["contact_state"] != "UNKNOWN" for item in states) / len(states) if states else 0.0
    uncertainty = []
    if coverage < 0.5:
        uncertainty.append("low_ball_track_coverage")
    if strict_frame is None:
        uncertainty.append("strict_release_abstained")
    if pose_source == "pose_only_fallback":
        uncertainty.append("pose_release_ball_constraint_unavailable")
    return {
        "schema_version": VERSION,
        "release_window": [start, end],
        "ball_track_quality": {
            "coverage": round(coverage, 4),
            "counts": counts,
            "rejected_jump_candidates": rejected,
        },
        "ball_track_status": "ok" if coverage >= 0.7 else "partial" if coverage >= 0.4 else "insufficient_data",
        "contact_state_coverage": round(contact_coverage, 4),
        "contact_state_sequence": states,
        "ball_rise_start": {
            "frame": ball_rise,
            "status": "ok" if ball_rise is not None else "insufficient_data",
            "provenance": ["release_window_ball_track_v1"],
        },
        "release_region": {
            "start_frame": release_region_start,
            "end_frame": release_region_end,
            "status": "ok" if release_region_start is not None and release_region_end is not None else "insufficient_data",
        },
        "release_pose": {
            "frame": pose_frame,
            "pose_only_frame": pose_only_frame,
            "status": "ok",
            "confidence": round(float(np.clip(0.55 * release_score[pose_index] + 0.45 * coverage, 0, 1)), 3),
            "source": pose_source,
            "provenance": ["pose_temporal_signal", pose_source],
        },
        "strict_release": strict,
        "supporting_evidence": {
            "last_likely_contact_frame": last_contact,
            "strict_relation": by_frame.get(strict_frame) if strict_frame is not None else None,
            "pose_only_score": round(float(release_score[pose_only_index]), 4),
            "selected_pose_score": round(float(release_score[pose_index]), 4),
        },
        "uncertainty": uncertainty,
        "provenance": [
            "release_ball_v1_candidates",
            "release_window_ball_track_v1",
            "rtmpose_shooting_arm",
            "human_ball_contact_state_v1",
        ],
        "fps": fps,
    }


def unavailable_human_ball_release(reason: str) -> dict[str, Any]:
    return {
        "schema_version": VERSION,
        "release_window": None,
        "ball_track_quality": {"coverage": 0.0, "counts": {status: 0 for status in ("DETECTED", "INTERPOLATED", "MISSING", "AMBIGUOUS")}, "rejected_jump_candidates": 0},
        "ball_track_status": "insufficient_data",
        "contact_state_coverage": 0.0,
        "contact_state_sequence": [],
        "ball_rise_start": {"frame": None, "status": "insufficient_data", "provenance": []},
        "release_region": {"start_frame": None, "end_frame": None, "status": "insufficient_data"},
        "release_pose": {"frame": None, "pose_only_frame": None, "status": "insufficient_data", "confidence": None, "source": "unavailable", "provenance": []},
        "strict_release": strict_abstention(reason),
        "supporting_evidence": {"last_likely_contact_frame": None, "strict_relation": None, "pose_only_score": None, "selected_pose_score": None},
        "uncertainty": [reason],
        "provenance": [],
    }


def build_ball_track(
    rows: list[dict[str, Any]],
    shooting_side: str,
    pose_key: str,
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    velocity = np.zeros(2, dtype=float)
    rejected = 0
    for row in rows:
        frame = int(row["frame_index"])
        if frame < start or frame > end:
            continue
        candidates = list(row.get("ball_candidates") or ([row["ball"]] if row.get("ball") else []))
        wrist = _joint(row.get(pose_key), JOINTS[shooting_side]["wrist"])
        ranked = []
        for candidate in candidates:
            center = np.asarray(candidate["center"], dtype=float)
            diameter = max(float(candidate["diameter"]), 1.0)
            if previous is not None:
                prior = np.asarray(previous["center"], dtype=float)
                expected = prior + velocity
                jump = float(np.linalg.norm(center - prior) / max(diameter, float(previous["diameter"]), 1.0))
                if jump > 7.0:
                    rejected += 1
                    continue
                continuity = math.exp(-float(np.linalg.norm(center - expected)) / (4.0 * diameter))
            else:
                continuity = 0.5
            hand = math.exp(-float(np.linalg.norm(center - wrist)) / (4.0 * diameter)) if wrist is not None else 0.0
            score = 0.50 * float(candidate["confidence"]) + 0.32 * continuity + 0.18 * hand
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            selected.append(_missing_track_point(frame))
            continue
        ambiguous = len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08
        candidate = dict(ranked[0][1])
        candidate.update(
            {
                "frame": frame,
                "ball_status": "AMBIGUOUS" if ambiguous else "DETECTED",
                "track_score": round(float(ranked[0][0]), 4),
                "provenance": ["release_ball_v1", "release_window_continuity_selection_v1"],
            }
        )
        if previous is not None:
            velocity = np.asarray(candidate["center"], dtype=float) - np.asarray(previous["center"], dtype=float)
        previous = candidate
        selected.append(candidate)
    _interpolate_short_gaps(selected)
    return selected, rejected


def _interpolate_short_gaps(track: list[dict[str, Any]]) -> None:
    index = 0
    while index < len(track):
        if track[index]["ball_status"] != "MISSING":
            index += 1
            continue
        start = index
        while index < len(track) and track[index]["ball_status"] == "MISSING":
            index += 1
        length = index - start
        if start == 0 or index == len(track) or length > MAX_INTERPOLATION_GAP:
            continue
        before, after = track[start - 1], track[index]
        diameter = max(float(before["diameter"]), float(after["diameter"]), 1.0)
        if math.dist(before["center"], after["center"]) / (length + 1) > 6.0 * diameter:
            continue
        for offset, current in enumerate(range(start, index), start=1):
            weight = offset / (length + 1)
            center = (1 - weight) * np.asarray(before["center"]) + weight * np.asarray(after["center"])
            size = (1 - weight) * float(before["diameter"]) + weight * float(after["diameter"])
            track[current].update(
                {
                    "center": center.tolist(),
                    "diameter": float(size),
                    "bbox": [float(center[0] - size / 2), float(center[1] - size / 2), float(center[0] + size / 2), float(center[1] + size / 2)],
                    "confidence": min(float(before["confidence"]), float(after["confidence"])) * 0.6,
                    "ball_status": "INTERPOLATED",
                    "track_score": None,
                    "provenance": ["bounded_linear_interpolation_v1"],
                }
            )


def build_relations(
    rows: list[dict[str, Any]],
    track: list[dict[str, Any]],
    shooting_side: str,
    pose_key: str,
) -> list[dict[str, Any]]:
    rows_by_frame = {int(row["frame_index"]): row for row in rows}
    mapping = JOINTS[shooting_side]
    output = []
    prior_ball: np.ndarray | None = None
    prior_wrist: np.ndarray | None = None
    for point in track:
        frame = int(point["frame"])
        pose = rows_by_frame[frame].get(pose_key)
        shoulder = _joint(pose, mapping["shoulder"])
        elbow = _joint(pose, mapping["elbow"])
        wrist = _joint(pose, mapping["wrist"])
        center = np.asarray(point["center"], dtype=float) if point.get("center") is not None else None
        diameter = float(point.get("diameter") or 0.0)
        ball_velocity = center - prior_ball if center is not None and prior_ball is not None else None
        wrist_velocity = wrist - prior_wrist if wrist is not None and prior_wrist is not None else None
        relative_velocity = ball_velocity - wrist_velocity if ball_velocity is not None and wrist_velocity is not None else None
        distance = float(np.linalg.norm(center - wrist)) if center is not None and wrist is not None else None
        angle = _joint_angle(shoulder, elbow, wrist)
        pose_confidence = _joint_confidence(pose, mapping.values())
        status_factor = {"DETECTED": 1.0, "AMBIGUOUS": 0.65, "INTERPOLATED": 0.5, "MISSING": 0.0}[point["ball_status"]]
        reliability = min(float(point.get("confidence") or 0.0), pose_confidence) * status_factor
        output.append(
            {
                "frame": frame,
                "ball_status": point["ball_status"],
                "ball_center": _round_point(center),
                "ball_bbox": [round(float(value), 2) for value in point["bbox"]] if point.get("bbox") is not None else None,
                "ball_diameter": round(diameter, 3) if diameter else None,
                "ball_confidence": round(float(point.get("confidence") or 0.0), 4) if center is not None else None,
                "wrist": _round_point(wrist),
                "elbow": _round_point(elbow),
                "shoulder": _round_point(shoulder),
                "wrist_ball_distance_px": round(distance, 3) if distance is not None else None,
                "wrist_ball_distance_diameters": round(distance / diameter, 4) if distance is not None and diameter > 0 else None,
                "wrist_ball_distance_radii": round(distance / (diameter / 2), 4) if distance is not None and diameter > 0 else None,
                "ball_velocity_px_per_frame": _round_point(ball_velocity),
                "wrist_velocity_px_per_frame": _round_point(wrist_velocity),
                "relative_velocity_px_per_frame": _round_point(relative_velocity),
                "relative_speed_diameters_per_frame": round(float(np.linalg.norm(relative_velocity)) / diameter, 4) if relative_velocity is not None and diameter > 0 else None,
                "elbow_angle_degrees_2d": round(angle, 3) if angle is not None else None,
                "ball_relative_to_wrist": _relative_position(center, wrist),
                "evidence_reliability": round(reliability, 4),
                "contact_state": "UNKNOWN",
                "state_confidence": 0.0,
                "state_reason": "not_decoded",
                "provenance": point.get("provenance", []),
            }
        )
        if center is not None:
            prior_ball = center
        if wrist is not None:
            prior_wrist = wrist
    return output


def decode_contact_states(relations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recent_contact: int | None = None
    separation_start: int | None = None
    for index, item in enumerate(relations):
        distance = item["wrist_ball_distance_diameters"]
        if item["ball_status"] == "MISSING" or distance is None or item["evidence_reliability"] < 0.2:
            item.update(contact_state="UNKNOWN", state_confidence=0.0, state_reason="missing_or_low_quality_evidence")
            continue
        prior_distances = [
            relation["wrist_ball_distance_diameters"]
            for relation in relations[max(0, index - 2) : index]
            if relation["wrist_ball_distance_diameters"] is not None
        ]
        increasing = bool(prior_distances) and distance > prior_distances[-1] + 0.10
        recent = recent_contact is not None and item["frame"] - recent_contact <= 8
        relative = item["relative_speed_diameters_per_frame"] or 0.0
        if recent and distance >= 1.25 and increasing and relative >= 0.12:
            separation_start = separation_start or item["frame"]
            item.update(contact_state="SEPARATING", state_confidence=round(min(0.9, 0.35 + item["evidence_reliability"] * 0.5 + min(relative, 1.0) * 0.15), 3), state_reason="recent_proximity_plus_increasing_separation_and_relative_motion")
        elif distance <= 1.75 and separation_start is None:
            recent_contact = item["frame"]
            item.update(contact_state="LIKELY_CONTACT", state_confidence=round(min(0.9, 0.45 + item["evidence_reliability"] * 0.5), 3), state_reason="wrist_proxy_proximity_with_supported_ball")
        else:
            item.update(contact_state="UNKNOWN", state_confidence=round(item["evidence_reliability"] * 0.4, 3), state_reason="no_supported_contact_transition")

    strict_frame: int | None = None
    interval: list[int] | None = None
    if separation_start is not None:
        start_index = next(index for index, item in enumerate(relations) if item["frame"] == separation_start)
        for index in range(start_index, len(relations) - PERSISTENCE + 1):
            candidate = relations[index : index + PERSISTENCE]
            distances = [item["wrist_ball_distance_diameters"] for item in candidate]
            supported = all(item["ball_status"] in {"DETECTED", "AMBIGUOUS"} for item in candidate)
            consecutive = all(candidate[offset + 1]["frame"] - candidate[offset]["frame"] == 1 for offset in range(PERSISTENCE - 1))
            persistent = all(value is not None and value >= 1.50 for value in distances)
            nonreturn = all(float(distances[offset + 1]) + 0.35 >= float(distances[offset]) for offset in range(PERSISTENCE - 1)) if persistent else False
            relative = [item["relative_speed_diameters_per_frame"] or 0.0 for item in candidate]
            if supported and consecutive and persistent and nonreturn and max(relative) >= 0.12:
                strict_frame = candidate[0]["frame"]
                interval = [candidate[0]["frame"], candidate[-1]["frame"]]
                break
    if strict_frame is None:
        return relations, strict_abstention("No persistent supported separation after likely contact")
    for item in relations:
        if item["frame"] == strict_frame == separation_start:
            continue
        if item["frame"] >= strict_frame and item["ball_status"] in {"DETECTED", "AMBIGUOUS"} and (item["wrist_ball_distance_diameters"] or 0.0) >= 1.5:
            item.update(contact_state="NO_CONTACT", state_confidence=round(min(0.92, 0.5 + item["evidence_reliability"] * 0.45), 3), state_reason="persistent_supported_separation_and_independent_motion")
        elif separation_start <= item["frame"] < strict_frame and item["ball_status"] != "MISSING":
            item.update(contact_state="SEPARATING", state_reason="transition_before_persistent_no_contact")
    relation = next(item for item in relations if item["frame"] == strict_frame)
    return relations, {
        "frame": strict_frame,
        "status": "ok",
        "evidence_quality": "supported",
        "supporting_frame_interval": interval,
        "reason": "Earliest three-frame persistent supported separation after likely contact",
        "confidence": round(min(item["state_confidence"] for item in relations if interval[0] <= item["frame"] <= interval[1]), 3),
        "measurements": {
            "distance_diameters": relation["wrist_ball_distance_diameters"],
            "relative_speed_diameters_per_frame": relation["relative_speed_diameters_per_frame"],
        },
        "provenance": ["human_ball_contact_state_v1", "persistent_separation_decoder_v1"],
        "risk_flags": ["wrist_is_contact_proxy", "no_learned_hand_contact_model"],
    }


def strict_abstention(reason: str) -> dict[str, Any]:
    return {
        "frame": None,
        "status": "insufficient_data",
        "evidence_quality": "insufficient",
        "supporting_frame_interval": None,
        "reason": reason,
        "confidence": None,
        "measurements": None,
        "provenance": ["human_ball_contact_state_v1", "persistent_separation_decoder_v1"],
        "risk_flags": ["strict_release_abstained"],
    }


def find_ball_rise(relations: list[dict[str, Any]]) -> int | None:
    supported = [item for item in relations if item["ball_center"] is not None and item["ball_status"] in {"DETECTED", "AMBIGUOUS"}]
    for first, second, third in zip(supported, supported[1:], supported[2:]):
        if second["frame"] == first["frame"] + 1 and third["frame"] == second["frame"] + 1:
            if second["ball_center"][1] < first["ball_center"][1] - 1 and third["ball_center"][1] < second["ball_center"][1] - 1:
                return first["frame"]
    return None


def _missing_track_point(frame: int) -> dict[str, Any]:
    return {"frame": frame, "center": None, "bbox": None, "diameter": None, "confidence": None, "ball_status": "MISSING", "track_score": None, "provenance": ["release_ball_v1_no_supported_detection"]}


def _joint(pose: dict[str, Any] | None, index: int) -> np.ndarray | None:
    if not pose:
        return None
    confidence = pose.get("temporal_reliability", pose.get("confidence", []))
    points = pose.get("keypoints", [])
    if len(points) <= index or len(confidence) <= index or float(confidence[index]) < 0.25:
        return None
    return np.asarray(points[index], dtype=float)


def _joint_confidence(pose: dict[str, Any] | None, indices: Any) -> float:
    if not pose:
        return 0.0
    confidence = pose.get("temporal_reliability", pose.get("confidence", []))
    values = [float(confidence[index]) for index in indices if len(confidence) > index]
    return min(values) if values else 0.0


def _joint_angle(first: np.ndarray | None, middle: np.ndarray | None, last: np.ndarray | None) -> float | None:
    if first is None or middle is None or last is None:
        return None
    a, b = first - middle, last - middle
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1))))


def _round_point(point: np.ndarray | None) -> list[float] | None:
    return [round(float(value), 3) for value in point] if point is not None else None


def _relative_position(ball: np.ndarray | None, wrist: np.ndarray | None) -> str | None:
    if ball is None or wrist is None:
        return None
    vertical = "above" if ball[1] < wrist[1] else "below"
    horizontal = "left" if ball[0] < wrist[0] else "right"
    return f"{vertical}_{horizontal}_of_wrist"
