from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import VALID_STATUSES

MOTION_SCHEMA_VERSION = "shot_motion_representation_v1"
MOTION_REPRESENTATION_VERSION = 1
RELIABILITY = {"HIGH", "MEDIUM", "LOW", "INSUFFICIENT"}
CANONICAL_EVENTS = (
    "dip_start", "dip_bottom", "leg_drive_onset", "ball_rise_start",
    "elbow_extension_onset", "takeoff", "release_region_start", "release_pose",
    "strict_ball_release", "body_apex", "release_region_end", "landing",
)
CANONICAL_PHASES = ("setup", "dip", "drive", "release", "follow_through", "landing_recovery")
TEMPORAL_PAIRS = (
    ("dip_bottom", "ball_rise_start"), ("dip_bottom", "leg_drive_onset"),
    ("leg_drive_onset", "ball_rise_start"), ("leg_drive_onset", "elbow_extension_onset"),
    ("ball_rise_start", "release_pose"), ("elbow_extension_onset", "release_pose"),
    ("takeoff", "release_pose"), ("takeoff", "strict_ball_release"),
    ("release_pose", "strict_ball_release"), ("strict_ball_release", "body_apex"),
    ("strict_ball_release", "landing"), ("release_pose", "follow_through_end"),
)


def reliability(status: str, confidence: float | None = None) -> str:
    if status not in {"ok", "low_confidence"}:
        return "INSUFFICIENT"
    if status == "low_confidence":
        return "LOW"
    if confidence is None:
        return "MEDIUM"
    return "HIGH" if confidence >= .75 else "MEDIUM" if confidence >= .45 else "LOW"


def fact(value: Any, unit: str | None, status: str, *, confidence: float | None = None,
         frame_range: list[int] | None = None, evidence: list[str] | None = None,
         source: list[str] | None = None, reason: str | None = None) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid fact status: {status}")
    return {"value": value, "unit": unit, "status": status,
            "reliability": reliability(status, confidence),
            "confidence": round(float(confidence), 3) if confidence is not None else None,
            "frame_range": frame_range, "evidence": evidence or [], "source": source or [],
            "reason": reason}


def unavailable(reason: str, source: list[str] | None = None, *, status: str = "insufficient_data") -> dict[str, Any]:
    return fact(None, None, status, source=source, reason=reason)


def build_motion_representation(report: dict[str, Any], trajectories: list[dict[str, Any]], *,
                                slow_motion: bool = False,
                                contaminated_research_only: bool = False) -> dict[str, Any]:
    fps = float(report.get("input", {}).get("fps") or 0)
    strict, pose_release = _event_frame(report, "strict_ball_release"), _event_frame(report, "pose_release")
    release = strict if strict is not None else pose_release
    bottom, takeoff = _event_frame(report, "bottom"), _event_frame(report, "takeoff")
    apex, landing = _event_frame(report, "body_apex"), _event_frame(report, "landing")
    side = report.get("attempt", {}).get("shooting_side", "right")
    signals = trajectory_signals(trajectories, side)
    ball = report.get("ball_evidence", {})
    observations = ball.get("center_observations", [])
    hb = report.get("human_ball_release", {})
    ball_rise = _nested_frame(hb, "ball_rise_start") or onset_from_values(
        [(row["frame"], row["center"][1]) for row in observations], decreasing=True)
    region = hb.get("release_region", {})
    region_ok = region.get("status") == "ok"
    elbow_metric = report.get("metrics", {}).get("elbow_extension_onset_relative_to_release", {})
    elbow_onset = elbow_metric.get("range", [None])[0] if elbow_metric.get("status") == "ok" else None
    frames = {
        "dip_start": _event_frame(report, "dip_start"), "dip_bottom": bottom,
        "leg_drive_onset": bottom, "ball_rise_start": ball_rise,
        "elbow_extension_onset": elbow_onset, "takeoff": takeoff,
        "release_region_start": region.get("start_frame") if region_ok else None,
        "release_pose": pose_release, "strict_ball_release": strict, "body_apex": apex,
        "release_region_end": region.get("end_frame") if region_ok else None, "landing": landing,
    }
    normalized = {name: normalize_frame(frame, bottom, release) for name, frame in frames.items()}
    events = {name: canonical_event(name, frames[name], fps, normalized[name], report, hb) for name in CANONICAL_EVENTS}
    phases = build_canonical_phases(report, events, fps)
    relation_frames = {**frames, "follow_through_end": phases["follow_through"]["end_frame"]}
    primitives = {
        "body_lowering": range_fact(frames["dip_start"], bottom, ["events.dip_start", "events.bottom"]),
        "body_rising": range_fact(bottom, apex or release, ["events.bottom", "events.body_apex"]),
        "knee_flexion": range_fact(frames["dip_start"], bottom, ["analysis_pose.knee_angle"]),
        "knee_extension": range_fact(bottom, takeoff or release, ["analysis_pose.knee_angle"]),
        "hip_rise": range_fact(bottom, apex or release, ["analysis_pose.hip_midpoint_y"]),
        "ball_lowering": unavailable("Ball lowering onset is not independently decoded", ["ball_evidence"]),
        "ball_rising": point_fact(ball_rise, ["human_ball_release.ball_rise_start"]),
        "wrist_rising": point_fact(onset_from_values(signals["wrist_y"], decreasing=True), [f"analysis_pose.{side}_wrist"]),
        "elbow_extension": point_fact(elbow_onset, ["metrics.elbow_extension_onset_relative_to_release"]),
        "shoulder_elevation": unavailable("No validated shoulder-elevation onset detector", ["analysis_pose"]),
        "trunk_lean_change": delta_fact(signals["trunk_angle"], bottom, release, "degrees_2d", ["analysis_pose.shoulders_hips"]),
        "takeoff": point_fact(takeoff, ["events.takeoff"]),
        "airborne_interval": range_fact(takeoff, landing, ["events.takeoff", "events.landing"]),
        "hand_ball_separation": point_fact(strict, ["events.strict_ball_release"]),
        "early_ball_flight": range_fact(strict, observations[-1]["frame"] if observations else None, ["ball_evidence.center_observations"]),
        "follow_through": range_fact(phases["follow_through"]["start_frame"], phases["follow_through"]["end_frame"], ["phases.follow_through"]),
        "landing": point_fact(landing, ["events.landing"]),
        "forward_body_drift": delta_fact(signals["hip_x"], bottom, landing or release, "pixels_2d", ["analysis_pose.hip_midpoint_x"]),
    }
    temporal = {f"{a}_to_{b}": temporal_relation(a, b, relation_frames[a], relation_frames[b], fps, bottom, release, slow_motion)
                for a, b in TEMPORAL_PAIRS}
    metrics = {name: metric_fact(item, slow_motion, _unsupported_view(report))
               for name, item in report.get("metrics", {}).items()}
    timing_status = "low_confidence" if slow_motion else "ok"
    quality_status = report.get("quality", {}).get("status", "needs_review")
    quality_status = quality_status if quality_status in VALID_STATUSES else "needs_review"
    result = {
        "schema_version": MOTION_SCHEMA_VERSION, "motion_representation_version": MOTION_REPRESENTATION_VERSION,
        "input_quality": fact(report.get("quality", {}), None, quality_status, source=["report.quality"]),
        "capture_context": fact({"view": report.get("attempt", {}).get("view", {}), "fps": fps,
                                 "slow_motion": slow_motion}, None,
                                report.get("attempt", {}).get("view", {}).get("status", "needs_review"),
                                source=["report.input", "report.attempt.view"]),
        "shooter_identity": fact({"attempt_id": report.get("attempt", {}).get("attempt_id"),
                                  "shooting_side": side,
                                  "continuity_source": report.get("perception", {}).get("person_box_source")}, None,
                                 "ok" if report.get("attempt", {}).get("attempt_id") else "needs_review",
                                 source=["report.attempt", "report.perception"]),
        "pose_reliability": fact(report.get("pose_reliability", {}), None,
                                 "ok" if report.get("pose_reliability") else "insufficient_data",
                                 source=["report.pose_reliability"]),
        "ball_reliability": fact({"detector": ball.get("detector"), "observations": len(observations),
                                  "risk_flags": ball.get("risk_flags", [])}, None,
                                 ball.get("status", "insufficient_data"), source=["report.ball_evidence"]),
        "events": events, "phases": phases, "motion_primitives": primitives,
        "human_ball_relations": build_human_ball_relations(observations, strict, signals),
        "kinematics": {
            "normalized_shot_time": fact(normalized, "dip_bottom_0_release_1",
                "ok" if bottom is not None and release is not None and release > bottom else "insufficient_data",
                source=["events.dip_bottom", "events.strict_ball_release", "events.release_pose"]),
            "timing_interpretation": fact("frame ordering only; no real-time coordination claim" if slow_motion else "normal-speed frame timing", None, timing_status, source=["input_timing_role"]),
            "metrics": metrics,
        },
        "temporal_relations": temporal,
        "uncertainty": {"status": "needs_review" if any(v["status"] != "ok" for v in temporal.values()) else "ok",
                        "slow_motion": slow_motion, "contaminated_research_only": contaminated_research_only,
                        "restrictions": ["no_coaching_semantics", "2d_image_space_only"]
                        + (["no_real_time_coordination_claim"] if slow_motion else [])
                        + (["no_generalization_claim"] if contaminated_research_only else [])},
        "provenance": ["reference_v1_report", "analysis_pose", "release_ball_v1", MOTION_SCHEMA_VERSION],
    }
    validate_motion_representation(result)
    return result


def canonical_event(name: str, frame: int | None, fps: float, normalized: float | None,
                    report: dict[str, Any], hb: dict[str, Any]) -> dict[str, Any]:
    source_name = {"dip_bottom": "bottom", "release_pose": "pose_release"}.get(name, name)
    source = report.get("events", {}).get(source_name, {})
    if name == "ball_rise_start":
        source = hb.get("ball_rise_start", {})
    elif name in {"release_region_start", "release_region_end"}:
        source = hb.get("release_region", {})
    status = source.get("status", "ok" if frame is not None else "insufficient_data")
    status = status if status in VALID_STATUSES else "needs_review"
    confidence = source.get("confidence")
    provenance = source.get("provenance", []) or [f"derived.{name}"]
    return {"name": name, "frame": frame,
            "timestamp_seconds": round(frame / fps, 4) if frame is not None and fps else None,
            "normalized_shot_time": normalized, "status": status,
            "reliability": reliability(status, confidence), "confidence": confidence,
            "evidence_quality": reliability(status, confidence), "source": provenance,
            "provenance": provenance, "supporting_evidence": source.get("evidence_refs", []),
            "reason": source.get("reason") if frame is None else None}


def build_canonical_phases(report: dict[str, Any], events: dict[str, Any], fps: float) -> dict[str, Any]:
    source = report.get("phases", {})
    mapping = {"setup": "preparation", "dip": "dip", "drive": "upward_drive",
               "follow_through": "follow_through", "landing_recovery": "landing_recovery"}
    result = {}
    bottom = events["dip_bottom"]["frame"]
    release = events["strict_ball_release"]["frame"] or events["release_pose"]["frame"]
    for name in CANONICAL_PHASES:
        if name == "release":
            start = events["release_region_start"]["frame"] or events["release_pose"]["frame"]
            end = events["release_region_end"]["frame"] or events["strict_ball_release"]["frame"]
            item = {"status": "ok", "provenance": ["human_ball_release.release_region"]}
        else:
            item = source.get(mapping[name], {})
            start, end = item.get("start_frame"), item.get("end_frame")
        valid = start is not None and end is not None and end >= start
        status = item.get("status", "insufficient_data") if valid else "insufficient_data"
        confidence = item.get("confidence")
        result[name] = {"name": name, "start_frame": start if valid else None,
                        "end_frame": end if valid else None,
                        "start_seconds": round(start / fps, 4) if valid and fps else None,
                        "end_seconds": round(end / fps, 4) if valid and fps else None,
                        "normalized_start": normalize_frame(start, bottom, release) if valid else None,
                        "normalized_end": normalize_frame(end, bottom, release) if valid else None,
                        "status": status, "reliability": reliability(status, confidence),
                        "confidence": confidence, "evidence_quality": reliability(status, confidence),
                        "source": item.get("provenance", []), "provenance": item.get("provenance", []),
                        "reason": item.get("reason") if not valid else None}
    return result


def temporal_relation(a: str, b: str, first: int | None, second: int | None, fps: float,
                      bottom: int | None, release: int | None, slow_motion: bool) -> dict[str, Any]:
    provenance = [f"events.{a}", f"events.{b}"]
    if first is None or second is None:
        return {"from_event": a, "to_event": b, "delta_frames": None, "delta_seconds": None,
                "normalized_delta": None, "status": "insufficient_data", "reliability": "INSUFFICIENT",
                "provenance": provenance, "reason": "one or both event endpoints are unavailable"}
    status = "low_confidence" if slow_motion else "ok"
    first_n, second_n = normalize_frame(first, bottom, release), normalize_frame(second, bottom, release)
    return {"from_event": a, "to_event": b, "delta_frames": second - first,
            "delta_seconds": round((second - first) / fps, 4) if fps else None,
            "normalized_delta": round(second_n - first_n, 4) if first_n is not None and second_n is not None else None,
            "status": status, "reliability": reliability(status), "provenance": provenance, "reason": None}


def metric_fact(item: dict[str, Any], slow_motion: bool, unsupported_view: bool) -> dict[str, Any]:
    status, value = item.get("status", "insufficient_data"), item.get("value")
    view = item.get("view_requirement")
    if unsupported_view and view:
        status, value = "unsupported_view", None
    temporal = item.get("unit") in {"frames", "seconds"} or "duration" in item.get("name", "")
    if slow_motion and temporal and status == "ok":
        status = "low_confidence"
    return {"value": value, "unit": item.get("unit"),
            "reference": {"frame": item.get("frame"), "range": item.get("range"),
                          "source_events": item.get("source_events", [])},
            "status": status, "reliability": reliability(status, item.get("confidence")),
            "confidence": item.get("confidence"), "view_dependence": view or "2d_image_space",
            "provenance": item.get("provenance", []), "reason": item.get("reason")}


def point_fact(frame: int | None, source: list[str]) -> dict[str, Any]:
    return fact(frame, "frame_index", "ok", frame_range=[frame, frame], source=source) if frame is not None else unavailable("Required event evidence unavailable", source)


def range_fact(start: int | None, end: int | None, source: list[str]) -> dict[str, Any]:
    return fact({"start_frame": start, "end_frame": end}, "frame_range", "ok", frame_range=[start, end], source=source) if start is not None and end is not None and end >= start else unavailable("Required ordered endpoints unavailable", source)


def delta_fact(values: list[tuple[int, float]], start: int | None, end: int | None,
               unit: str, source: list[str]) -> dict[str, Any]:
    points = dict(values)
    if start is None or end is None or start not in points or end not in points:
        return unavailable("Endpoint trajectory evidence unavailable", source)
    return fact(round(points[end] - points[start], 4), unit, "ok", frame_range=[start, end], source=source)


def build_human_ball_relations(observations: list[dict[str, Any]], release: int | None,
                               signals: dict[str, list[tuple[int, float]]]) -> dict[str, Any]:
    distances = [(row["frame"], row.get("ball_hand_distance_diameters")) for row in observations
                 if row.get("ball_hand_distance_diameters") is not None]
    vertical = [{"frame": row["frame"], "y": row["center"][1], "confidence": row.get("confidence")} for row in observations]
    post = [row for row in observations if release is not None and row["frame"] > release]
    nose_y = dict(signals["nose_y"])
    heights = [{"frame": row["frame"], "ball_minus_head_y_px": row["center"][1] - nose_y[row["frame"]]}
               for row in observations if row["frame"] in nose_y]
    return {
        "ball_to_wrist_distance": fact([{"frame": f, "distance": round(v, 4)} for f, v in distances], "ball_diameters", "ok" if distances else "insufficient_data", frame_range=[distances[0][0], distances[-1][0]] if distances else None, source=["ball_evidence.ball_hand_distance_diameters"]),
        "ball_to_head_relative_height": fact(heights, "pixels_y_ball_minus_head", "ok" if heights else "insufficient_data", frame_range=[heights[0]["frame"], heights[-1]["frame"]] if heights else None, source=["ball_evidence", "analysis_pose.nose"]),
        "ball_vertical_trajectory": fact(vertical, "original_frame_pixel_y", "ok" if vertical else "insufficient_data", frame_range=[vertical[0]["frame"], vertical[-1]["frame"]] if vertical else None, source=["ball_evidence.center_observations"]),
        "ball_hand_contact_proxy": fact([{"frame": row["frame"], "state": row.get("contact_state")} for row in observations], "proxy_state", "ok" if observations else "insufficient_data", frame_range=[observations[0]["frame"], observations[-1]["frame"]] if observations else None, source=["human_ball_contact_state_v1"]),
        "separation": point_fact(release, ["events.strict_ball_release"]),
        "post_release_trajectory_availability": fact({"available_frames": len(post)}, "frames", "ok" if post else "insufficient_data", frame_range=[post[0]["frame"], post[-1]["frame"]] if post else None, source=["ball_evidence.center_observations"]),
    }


def onset_from_values(values: list[tuple[int, float]], *, decreasing: bool) -> int | None:
    ordered = sorted((frame, value) for frame, value in values if math.isfinite(value))
    for index in range(len(ordered) - 2):
        differences = [ordered[offset + 1][1] - ordered[offset][1] for offset in range(index, index + 2)]
        if all(v < -1 for v in differences) if decreasing else all(v > 1 for v in differences):
            return ordered[index][0]
    return None


def normalize_frame(frame: int | None, bottom: int | None, release: int | None) -> float | None:
    if frame is None or bottom is None or release is None or release <= bottom:
        return None
    return round((frame - bottom) / (release - bottom), 4)


def trajectory_signals(rows: list[dict[str, Any]], side: str) -> dict[str, list[tuple[int, float]]]:
    indices = {"left": (5, 7, 9), "right": (6, 8, 10)}[side if side in {"left", "right"} else "right"]
    result: dict[str, list[tuple[int, float]]] = {name: [] for name in ("wrist_y", "hip_x", "trunk_angle", "nose_y", "elbow_angle")}
    for row in rows:
        pose = row.get("analysis_pose")
        if not pose:
            continue
        points = np.asarray(pose.get("keypoints", []), dtype=float)
        confidence = np.asarray(pose.get("temporal_reliability", pose.get("confidence", [])), dtype=float)
        if len(points) < 17 or len(confidence) < 17:
            continue
        frame = int(row["frame_index"])
        shoulder, elbow, wrist = indices
        if confidence[wrist] >= .25:
            result["wrist_y"].append((frame, float(points[wrist, 1])))
        if confidence[11] >= .25 and confidence[12] >= .25:
            result["hip_x"].append((frame, float(np.mean(points[[11, 12], 0]))))
        if confidence[0] >= .25:
            result["nose_y"].append((frame, float(points[0, 1])))
        if np.all(confidence[[5, 6, 11, 12]] >= .25):
            vector = np.mean(points[[5, 6]], axis=0) - np.mean(points[[11, 12]], axis=0)
            result["trunk_angle"].append((frame, float(math.degrees(math.atan2(vector[0], -vector[1])))))
        if np.all(confidence[[shoulder, elbow, wrist]] >= .25):
            result["elbow_angle"].append((frame, joint_angle(points[shoulder], points[elbow], points[wrist])))
    return result


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return math.nan if denominator <= 1e-6 else float(math.degrees(math.acos(np.clip(np.dot(first, second) / denominator, -1, 1))))


def validate_motion_representation(value: dict[str, Any]) -> None:
    required = {"schema_version", "motion_representation_version", "input_quality", "capture_context",
                "shooter_identity", "pose_reliability", "ball_reliability", "events", "phases",
                "motion_primitives", "human_ball_relations", "kinematics", "temporal_relations",
                "uncertainty", "provenance"}
    if value.get("schema_version") != MOTION_SCHEMA_VERSION or value.get("motion_representation_version") != 1 or required - set(value):
        raise ValueError("Incomplete ShotMotionRepresentation V1")
    if tuple(value["events"]) != CANONICAL_EVENTS or tuple(value["phases"]) != CANONICAL_PHASES:
        raise ValueError("Canonical event or phase vocabulary is incomplete")
    collections = (value["events"], value["phases"], value["motion_primitives"],
                   value["human_ball_relations"], value["temporal_relations"])
    for collection in collections:
        for item in collection.values():
            if item.get("status") not in VALID_STATUSES or item.get("reliability") not in RELIABILITY:
                raise ValueError("Invalid motion status or reliability")
    _validate_event_order(value["events"])


def _validate_event_order(events: dict[str, dict[str, Any]]) -> None:
    values = [events[name]["frame"] for name in ("dip_start", "dip_bottom", "takeoff") if events[name]["frame"] is not None]
    if values != sorted(values):
        raise ValueError("Core load/drive events are out of order")
    landing = events["landing"]["frame"]
    release = events["strict_ball_release"]["frame"] or events["release_pose"]["frame"]
    if landing is not None and release is not None and landing < release:
        raise ValueError("Landing precedes release")


def _event_frame(report: dict[str, Any], name: str) -> int | None:
    item = report.get("events", {}).get(name, {})
    return int(item["frame"]) if item.get("status") == "ok" and item.get("frame") is not None else None


def _nested_frame(value: dict[str, Any], name: str) -> int | None:
    item = value.get(name, {})
    return int(item["frame"]) if item.get("status") == "ok" and item.get("frame") is not None else None


def _unsupported_view(report: dict[str, Any]) -> bool:
    return report.get("attempt", {}).get("view", {}).get("status") == "unsupported_view"
