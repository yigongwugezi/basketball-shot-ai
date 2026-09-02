from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import VALID_STATUSES


MOTION_SCHEMA_VERSION = "shot_motion_representation_v0"
FACT_FIELDS = {"value", "unit", "status", "confidence", "frame_range", "evidence", "source"}


def fact(
    value: Any,
    unit: str | None,
    status: str,
    *,
    confidence: float | None = None,
    frame_range: list[int] | None = None,
    evidence: list[str] | None = None,
    source: list[str] | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid fact status: {status}")
    return {
        "value": value,
        "unit": unit,
        "status": status,
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "frame_range": frame_range,
        "evidence": evidence or [],
        "source": source or [],
    }


def unavailable(reason: str, source: list[str] | None = None) -> dict[str, Any]:
    return {**fact(None, None, "insufficient_data", source=source), "reason": reason}


def build_motion_representation(
    report: dict[str, Any],
    trajectories: list[dict[str, Any]],
    *,
    slow_motion: bool = False,
    contaminated_research_only: bool = False,
) -> dict[str, Any]:
    events = {name: event_fact(item) for name, item in report["events"].items()}
    phases = {name: phase_fact(item) for name, item in report["phases"].items()}
    strict = _event_frame(report, "strict_ball_release")
    pose_release = _event_frame(report, "pose_release")
    release = strict if strict is not None else pose_release
    bottom = _event_frame(report, "bottom")
    takeoff = _event_frame(report, "takeoff")
    apex = _event_frame(report, "body_apex")
    landing = _event_frame(report, "landing")
    shooting_side = report["attempt"].get("shooting_side", "right")
    trajectory = trajectory_signals(trajectories, shooting_side)
    ball = report.get("ball_evidence", {})
    observations = ball.get("center_observations", [])
    ball_rise = onset_from_values([(row["frame"], row["center"][1]) for row in observations], decreasing=True)
    wrist_rise = onset_from_values(trajectory["wrist_y"], decreasing=True)
    elbow_metric = report["metrics"].get("elbow_extension_onset_relative_to_release", {})
    elbow_onset = elbow_metric.get("range", [None])[0] if elbow_metric.get("status") == "ok" else None

    primitives = {
        "body_lowering": range_fact(_event_frame(report, "dip_start"), bottom, "frame", ["events.dip_start", "events.bottom"]),
        "body_rising": range_fact(bottom, apex or release, "frame", ["events.bottom", "events.body_apex"]),
        "knee_flexion": range_fact(_event_frame(report, "dip_start"), bottom, "frame", ["pose.knee_angle", "events.bottom"]),
        "knee_extension": range_fact(bottom, takeoff or release, "frame", ["pose.knee_angle", "events.takeoff"]),
        "hip_rise": range_fact(bottom, apex or release, "frame", ["pose.hip_midpoint_y"]),
        "ball_lowering": unavailable("Ball lowering onset is not independently decoded in Reference V1", ["ball_evidence"]),
        "ball_rising": point_fact(ball_rise, "frame_index", ["ball_evidence.center_observations"]),
        "shooting_wrist_rising": point_fact(wrist_rise, "frame_index", [f"analysis_pose.{shooting_side}_wrist"]),
        "elbow_extension": point_fact(elbow_onset, "frame_index", ["metrics.elbow_extension_onset_relative_to_release"]),
        "shoulder_elevation": unavailable("No validated shoulder-elevation onset detector", ["analysis_pose"]),
        "trunk_lean_change": delta_fact(trajectory["trunk_angle"], bottom, release, "degrees_2d", ["analysis_pose.shoulders_hips"]),
        "takeoff": events.get("takeoff", unavailable("Takeoff unavailable")),
        "airborne": range_fact(takeoff, landing, "frame", ["events.takeoff", "events.landing"]),
        "release": events.get("strict_ball_release") if strict is not None else events.get("pose_release", unavailable("Release unavailable")),
        "follow_through": phases.get("follow_through", unavailable("Follow-through unavailable")),
        "landing": events.get("landing", unavailable("Landing unavailable")),
        "forward_drift": delta_fact(trajectory["hip_x"], bottom, landing or release, "pixels_2d", ["analysis_pose.hip_midpoint_x"]),
    }

    human_ball = build_human_ball_relations(observations, strict, trajectory)
    relations = {
        "elbow_extension_to_release": frame_relation(elbow_onset, release, "elbow_extension_onset", "release"),
        "knee_extension_overlaps_ball_rise": overlap_relation(bottom, takeoff or release, ball_rise, release),
        "release_to_body_apex": frame_relation(release, apex, "release", "body_apex"),
        "ball_rise_continuity": ball_continuity(observations, release),
        "wrist_peak_to_release": peak_relation(trajectory["wrist_y"], release, find_min=True),
        "body_forward_drift_during_shot": primitives["forward_drift"],
    }
    normalized_time = build_normalized_time(report, bottom, release, elbow_onset, ball_rise)
    timing_status = "low_confidence" if slow_motion else "ok"
    timing_note = "slow-motion timing supports frame ordering only; no real-time coordination claim" if slow_motion else "normal-speed frame timing"

    quality_status = report.get("quality", {}).get("status", "needs_review")
    representation = {
        "schema_version": MOTION_SCHEMA_VERSION,
        "input_quality": fact(report.get("quality", {}), None, quality_status if quality_status in VALID_STATUSES else "needs_review", source=["report.quality"]),
        "view": fact(report["attempt"].get("view", {}).get("value"), "camera_view", report["attempt"].get("view", {}).get("status", "needs_review"), source=["report.attempt.view"]),
        "pose_reliability": fact(report.get("pose_reliability", {}), None, "ok" if report.get("pose_reliability") else "insufficient_data", source=["report.pose_reliability"]),
        "ball_reliability": fact({"detector": ball.get("detector"), "observations": len(observations), "risk_flags": ball.get("risk_flags", [])}, None, ball.get("status", "insufficient_data"), source=["report.ball_evidence"]),
        "events": events,
        "phases": phases,
        "primitives": primitives,
        "relations": {**relations, **human_ball},
        "kinematics": {
            "normalized_shot_time": fact(normalized_time, "bottom_0_release_1", "ok" if normalized_time else "insufficient_data", source=["events.bottom", "events.strict_ball_release"]),
            "timing_interpretation": fact(timing_note, None, timing_status, source=["input_timing_role"]),
        },
        "uncertainty": {
            "status": "needs_review" if any(item["status"] != "ok" for item in relations.values()) else "ok",
            "slow_motion": slow_motion,
            "contaminated_research_only": contaminated_research_only,
            "restrictions": ["no_coaching_semantics", "2d_image_space_only"] + (["no_real_time_coordination_claim"] if slow_motion else []) + (["no_generalization_claim"] if contaminated_research_only else []),
        },
        "provenance": ["reference_v1_report", "analysis_pose", "release_ball_v1", "motion_representation_v0"],
    }
    validate_motion_representation(representation)
    return representation


def event_fact(item: dict[str, Any]) -> dict[str, Any]:
    frame = item.get("frame")
    return fact(frame, "frame_index", item["status"], confidence=item.get("confidence"), frame_range=[frame, frame] if frame is not None else None, evidence=item.get("evidence_refs"), source=item.get("provenance"))


def phase_fact(item: dict[str, Any]) -> dict[str, Any]:
    start, end = item.get("start_frame"), item.get("end_frame")
    return fact({"start_frame": start, "end_frame": end} if start is not None and end is not None else None, "frame_range", item["status"], confidence=item.get("confidence"), frame_range=[start, end] if start is not None and end is not None else None, source=item.get("provenance"))


def point_fact(frame: int | None, unit: str, source: list[str]) -> dict[str, Any]:
    return fact(frame, unit, "ok", frame_range=[frame, frame], source=source) if frame is not None else unavailable("Required onset evidence unavailable", source)


def range_fact(start: int | None, end: int | None, unit: str, source: list[str]) -> dict[str, Any]:
    return fact({"start": start, "end": end}, unit, "ok", frame_range=[start, end], source=source) if start is not None and end is not None and end >= start else unavailable("Required ordered event range unavailable", source)


def frame_relation(first: int | None, second: int | None, first_name: str, second_name: str) -> dict[str, Any]:
    if first is None or second is None:
        return unavailable("Required event evidence unavailable", [first_name, second_name])
    delta = first - second
    ordering = "before" if delta < 0 else "after" if delta > 0 else "same_frame"
    return fact({"delta_frames": delta, "ordering": ordering}, "frames", "ok", frame_range=sorted([first, second]), source=[first_name, second_name])


def overlap_relation(first_start: int | None, first_end: int | None, second_start: int | None, second_end: int | None) -> dict[str, Any]:
    if None in (first_start, first_end, second_start, second_end):
        return unavailable("One or both primitive ranges unavailable", ["knee_extension", "ball_rising"])
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start) + 1)
    return fact({"overlaps": overlap > 0, "overlap_frames": overlap}, "frames", "ok", frame_range=[min(first_start, second_start), max(first_end, second_end)], source=["knee_extension", "ball_rising"])


def delta_fact(values: list[tuple[int, float]], start: int | None, end: int | None, unit: str, source: list[str]) -> dict[str, Any]:
    mapping = dict(values)
    if start is None or end is None or start not in mapping or end not in mapping:
        return unavailable("Endpoint trajectory evidence unavailable", source)
    return fact(round(mapping[end] - mapping[start], 4), unit, "ok", frame_range=[start, end], source=source)


def peak_relation(values: list[tuple[int, float]], release: int | None, *, find_min: bool) -> dict[str, Any]:
    if release is None or not values:
        return unavailable("Wrist trajectory or release unavailable", ["shooting_wrist_rising", "release"])
    candidates = [(frame, value) for frame, value in values if frame >= release - 5]
    if not candidates:
        return unavailable("No wrist samples near release", ["analysis_pose"])
    peak = (min if find_min else max)(candidates, key=lambda item: item[1])[0]
    return frame_relation(peak, release, "wrist_peak", "release")


def onset_from_values(values: list[tuple[int, float]], *, decreasing: bool) -> int | None:
    ordered = sorted((frame, value) for frame, value in values if math.isfinite(value))
    for index in range(len(ordered) - 2):
        differences = [ordered[offset + 1][1] - ordered[offset][1] for offset in range(index, index + 2)]
        if all(value < -1.0 for value in differences) if decreasing else all(value > 1.0 for value in differences):
            return ordered[index][0]
    return None


def ball_continuity(observations: list[dict[str, Any]], release: int | None) -> dict[str, Any]:
    before = [row for row in observations if release is None or row["frame"] <= release]
    if len(before) < 3:
        return unavailable("Fewer than three pre-release ball observations", ["ball_evidence.center_observations"])
    pauses = 0
    for first, second in zip(before, before[1:]):
        pauses += int(abs(second["center"][1] - first["center"][1]) < 2.0)
    return fact({"continuous_rise": pauses == 0, "pause_candidate_steps": pauses}, "frame_steps", "ok", frame_range=[before[0]["frame"], before[-1]["frame"]], source=["ball_evidence.center_observations"])


def build_human_ball_relations(observations: list[dict[str, Any]], release: int | None, trajectory: dict[str, list[tuple[int, float]]]) -> dict[str, Any]:
    distances = [(row["frame"], row.get("ball_hand_distance_diameters")) for row in observations if row.get("ball_hand_distance_diameters") is not None]
    vertical = [{"frame": row["frame"], "y": row["center"][1], "confidence": row.get("confidence")} for row in observations]
    post = [row for row in observations if release is not None and row["frame"] > release]
    nose_y = dict(trajectory["nose_y"])
    heights = [{"frame": row["frame"], "ball_minus_head_y_px": row["center"][1] - nose_y[row["frame"]]} for row in observations if row["frame"] in nose_y]
    return {
        "ball_to_wrist_distance": fact([{"frame": frame, "distance": round(value, 4)} for frame, value in distances], "ball_diameters", "ok" if distances else "insufficient_data", frame_range=[distances[0][0], distances[-1][0]] if distances else None, source=["ball_evidence.ball_hand_distance_diameters"]),
        "ball_to_head_relative_height": fact(heights, "pixels_y_ball_minus_head", "ok" if heights else "insufficient_data", frame_range=[heights[0]["frame"], heights[-1]["frame"]] if heights else None, source=["ball_evidence", "analysis_pose.nose"]),
        "ball_vertical_trajectory": fact(vertical, "original_frame_pixel_y", "ok" if vertical else "insufficient_data", frame_range=[vertical[0]["frame"], vertical[-1]["frame"]] if vertical else None, source=["ball_evidence.center_observations"]),
        "ball_hand_contact_proxy": fact([{"frame": row["frame"], "state": row.get("contact_state")} for row in observations], "proxy_state", "ok" if observations else "insufficient_data", frame_range=[observations[0]["frame"], observations[-1]["frame"]] if observations else None, source=["contact_transition_decoder_v1"]),
        "separation": point_fact(release, "frame_index", ["events.strict_ball_release"]),
        "post_release_trajectory_availability": fact({"available_frames": len(post), "status": "available" if post else "insufficient_data"}, "frames", "ok" if post else "insufficient_data", frame_range=[post[0]["frame"], post[-1]["frame"]] if post else None, source=["ball_evidence.center_observations"]),
    }


def build_normalized_time(report: dict[str, Any], bottom: int | None, release: int | None, elbow: int | None, ball_rise: int | None) -> dict[str, Any]:
    if bottom is None or release is None or release <= bottom:
        return {}
    def normalize(frame: int | None) -> float | None:
        return round((frame - bottom) / (release - bottom), 4) if frame is not None else None
    values = {name: normalize(_event_frame(report, name)) for name in ("bottom", "takeoff", "pose_release", "strict_ball_release", "body_apex", "landing")}
    values.update({"elbow_extension_onset": normalize(elbow), "ball_rise_onset": normalize(ball_rise)})
    return values


def trajectory_signals(rows: list[dict[str, Any]], side: str) -> dict[str, list[tuple[int, float]]]:
    side_indices = {"left": (5, 7, 9, 11), "right": (6, 8, 10, 12)}[side]
    result: dict[str, list[tuple[int, float]]] = {name: [] for name in ("wrist_y", "hip_x", "trunk_angle", "nose_y", "elbow_angle")}
    for row in rows:
        pose = row.get("analysis_pose")
        if not pose:
            continue
        points = np.asarray(pose.get("keypoints", []), dtype=float)
        confidence = np.asarray(pose.get("temporal_reliability", []), dtype=float)
        if len(points) < 17 or len(confidence) < 17:
            continue
        frame = int(row["frame_index"])
        shoulder, elbow, wrist, hip = side_indices
        if confidence[wrist] >= 0.25:
            result["wrist_y"].append((frame, float(points[wrist, 1])))
        if confidence[11] >= 0.25 and confidence[12] >= 0.25:
            result["hip_x"].append((frame, float(np.mean(points[[11, 12], 0]))))
        if confidence[0] >= 0.25:
            result["nose_y"].append((frame, float(points[0, 1])))
        if np.all(confidence[[5, 6, 11, 12]] >= 0.25):
            shoulder_mid = np.mean(points[[5, 6]], axis=0)
            hip_mid = np.mean(points[[11, 12]], axis=0)
            vector = shoulder_mid - hip_mid
            result["trunk_angle"].append((frame, float(math.degrees(math.atan2(vector[0], -vector[1])))))
        if np.all(confidence[[shoulder, elbow, wrist]] >= 0.25):
            result["elbow_angle"].append((frame, joint_angle(points[shoulder], points[elbow], points[wrist])))
    return result


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-6:
        return math.nan
    return float(math.degrees(math.acos(np.clip(np.dot(first, second) / denominator, -1, 1))))


def validate_motion_representation(value: dict[str, Any]) -> None:
    required = {"schema_version", "input_quality", "view", "pose_reliability", "ball_reliability", "events", "phases", "primitives", "relations", "kinematics", "uncertainty", "provenance"}
    if value.get("schema_version") != MOTION_SCHEMA_VERSION or required - set(value):
        raise ValueError("Incomplete ShotMotionRepresentation")
    for collection_name in ("events", "phases", "primitives", "relations", "kinematics"):
        for name, item in value[collection_name].items():
            if FACT_FIELDS - set(item):
                raise ValueError(f"Motion fact lacks evidence envelope: {collection_name}.{name}")
            if item["status"] not in VALID_STATUSES:
                raise ValueError(f"Invalid motion status: {collection_name}.{name}")
    for name in ("input_quality", "view", "pose_reliability", "ball_reliability"):
        if FACT_FIELDS - set(value[name]):
            raise ValueError(f"Top-level fact lacks evidence envelope: {name}")
    _validate_event_order(value["events"])


def _validate_event_order(events: dict[str, dict[str, Any]]) -> None:
    values = [events.get(name, {}).get("value") for name in ("dip_start", "bottom", "takeoff")]
    present = [value for value in values if value is not None]
    if present != sorted(present):
        raise ValueError("Core load/drive events are out of order")
    landing = events.get("landing", {}).get("value")
    release = events.get("strict_ball_release", {}).get("value") or events.get("pose_release", {}).get("value")
    if landing is not None and release is not None and landing < release:
        raise ValueError("Landing precedes release")


def _event_frame(report: dict[str, Any], name: str) -> int | None:
    item = report["events"].get(name, {})
    return int(item["frame"]) if item.get("status") == "ok" and item.get("frame") is not None else None
