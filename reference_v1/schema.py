from __future__ import annotations

from typing import Any

from . import SCHEMA_VERSION
from .human_ball import VERSION as HUMAN_BALL_VERSION


VALID_STATUSES = {
    "ok",
    "not_detected",
    "not_applicable",
    "insufficient_data",
    "low_confidence",
    "unsupported_view",
    "ambiguous",
    "needs_review",
}

EVENT_LABELS = {
    "dip_start": "下沉开始",
    "bottom": "最低点",
    "takeoff": "起跳 / 离地",
    "pose_release": "Pose Release",
    "strict_ball_release": "Strict Ball Release",
    "body_apex": "身体最高点",
    "landing": "落地",
}

PHASE_LABELS = {
    "preparation": "准备阶段",
    "dip": "下沉阶段",
    "upward_drive": "向上驱动阶段",
    "follow_through": "随挥阶段",
    "landing_recovery": "落地 / 恢复阶段",
}

METRIC_LABELS = {
    "strict_release_frame": "严格出手时刻",
    "pose_to_strict_release_delta": "姿态出手 → 实际离手时间差",
    "release_elbow_angle": "出手时肘关节伸展角",
    "normalized_release_height": "相对出手高度",
    "dip_depth": "下沉深度",
    "minimum_knee_angle": "最低点膝关节弯曲角",
    "release_relative_to_body_apex": "出手相对身体最高点的时机",
    "elbow_extension_onset_relative_to_release": "肘部伸展启动时机",
    "takeoff_to_strict_release": "起跳 → 出手时间",
    "follow_through_duration": "随挥持续时间",
}


def event(
    name: str,
    status: str,
    *,
    frame: int | None = None,
    fps: float = 0.0,
    confidence: float | None = None,
    provenance: list[str] | None = None,
    risk_flags: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    _check_status(status)
    return {
        "name": name,
        "label_zh": EVENT_LABELS[name],
        "status": status,
        "frame": frame,
        "timestamp_seconds": round(frame / fps, 4) if frame is not None and fps else None,
        "confidence": _confidence(confidence),
        "provenance": provenance or [],
        "risk_flags": risk_flags or [],
        "reason": reason,
        "evidence_refs": [f"evidence/{name}.jpg"] if frame is not None else [],
    }


def phase(
    name: str,
    status: str,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    fps: float = 0.0,
    confidence: float | None = None,
    provenance: list[str] | None = None,
    risk_flags: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    _check_status(status)
    return {
        "name": name,
        "label_zh": PHASE_LABELS[name],
        "status": status,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": round(start_frame / fps, 4) if start_frame is not None and fps else None,
        "end_seconds": round(end_frame / fps, 4) if end_frame is not None and fps else None,
        "confidence": _confidence(confidence),
        "provenance": provenance or [],
        "risk_flags": risk_flags or [],
        "reason": reason,
    }


def metric(
    name: str,
    status: str,
    *,
    value: Any = None,
    unit: str | None = None,
    confidence: float | None = None,
    frame: int | None = None,
    frame_range: list[int] | None = None,
    source_events: list[str] | None = None,
    required_joints: list[str] | None = None,
    provenance: list[str] | None = None,
    view_requirement: str | None = None,
    risk_flags: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    _check_status(status)
    return {
        "name": name,
        "label_zh": METRIC_LABELS[name],
        "status": status,
        "value": value,
        "unit": unit,
        "confidence": _confidence(confidence),
        "frame": frame,
        "range": frame_range,
        "source_events": source_events or [],
        "required_joints": required_joints or [],
        "provenance": provenance or [],
        "view_requirement": view_requirement,
        "risk_flags": risk_flags or [],
        "evidence_refs": evidence_refs or [],
        "reason": reason,
    }


def validate_report(report: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "input",
        "quality",
        "attempt",
        "phases",
        "events",
        "ball_evidence",
        "human_ball_release",
        "metrics",
        "observations",
        "suggestions",
        "risks",
        "runtime",
        "artifacts",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"Report is missing fields: {missing}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unexpected schema version")
    if set(report["events"]) != set(EVENT_LABELS):
        raise ValueError("Event taxonomy is incomplete")
    if set(report["phases"]) != set(PHASE_LABELS):
        raise ValueError("Phase taxonomy is incomplete")
    if set(report["metrics"]) != set(METRIC_LABELS):
        raise ValueError("Metric taxonomy is incomplete")
    human_ball = report["human_ball_release"]
    required_human_ball = {
        "schema_version",
        "release_window",
        "ball_track_quality",
        "ball_track_status",
        "contact_state_sequence",
        "ball_rise_start",
        "release_region",
        "release_pose",
        "strict_release",
        "supporting_evidence",
        "uncertainty",
        "provenance",
    }
    if human_ball.get("schema_version") != HUMAN_BALL_VERSION:
        raise ValueError("Unexpected Human-Ball evidence version")
    if required_human_ball - set(human_ball):
        raise ValueError("Human-Ball evidence contract is incomplete")
    for item in human_ball["contact_state_sequence"]:
        if item["ball_status"] not in {"DETECTED", "INTERPOLATED", "MISSING", "AMBIGUOUS"}:
            raise ValueError("Invalid Human-Ball track status")
        if item["contact_state"] not in {"UNKNOWN", "LIKELY_CONTACT", "SEPARATING", "NO_CONTACT"}:
            raise ValueError("Invalid Human-Ball contact state")
    for collection in (report["events"], report["phases"], report["metrics"]):
        for item in collection.values():
            _check_status(item["status"])


def _check_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid Reference V1 status: {status}")


def _confidence(value: float | None) -> float | None:
    return round(max(0.0, min(1.0, float(value))), 3) if value is not None else None
