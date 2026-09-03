from __future__ import annotations

from benchmarks.reference_v1.validation_closure import decode_contact_transition_v1

from . import SCHEMA_VERSION
from .analysis import _persistent_return_index, build_phases
from .human_ball import unavailable_human_ball_release
from .motion import build_motion_representation
from .schema import EVENT_LABELS, METRIC_LABELS, PHASE_LABELS, event, metric, validate_report


def test_phase_event_ordering() -> None:
    fps = 30.0
    frames = {
        "dip_start": 10,
        "bottom": 20,
        "takeoff": 27,
        "pose_release": 30,
        "strict_ball_release": 32,
        "body_apex": 35,
        "landing": 45,
    }
    events = {name: event(name, "ok", frame=frame, fps=fps) for name, frame in frames.items()}
    phases = build_phases(events, 0, 60, fps)
    assert phases["dip"]["start_frame"] == 10
    assert phases["dip"]["end_frame"] == 20
    assert phases["upward_drive"]["end_frame"] == 32
    assert phases["follow_through"]["start_frame"] == 32
    assert phases["landing_recovery"]["start_frame"] == 45


def test_landing_requires_persistent_return() -> None:
    assert _persistent_return_index([0.8, 1.0, 1.18, 1.35], 1, 1.0, 0.1) is None
    assert _persistent_return_index([0.8, 1.04, 1.02, 0.99, 1.2], 1, 1.0, 0.1) == 1


def test_strict_release_separation() -> None:
    evidence = []
    for frame, distance, x in (
        (27, 1.0, 100),
        (28, 1.1, 101),
        (29, 1.2, 102),
        (30, 1.5, 106),
        (31, 1.8, 111),
    ):
        evidence.append(
            {
                "frame_index": frame,
                "ball_wrist_distance_diameters": distance,
                "ball_center": [x, 50],
            }
        )
    result = decode_contact_transition_v1(evidence, pose_release_frame=29)
    assert result["status"] == "ok"
    assert result["predicted_strict_frame"] == 30


def test_unavailable_metric_and_report_contract() -> None:
    events = {name: event(name, "insufficient_data", reason="synthetic test") for name in EVENT_LABELS}
    phases = {
        name: {
            "name": name,
            "label_zh": label,
            "status": "insufficient_data",
            "start_frame": None,
            "end_frame": None,
            "start_seconds": None,
            "end_seconds": None,
            "confidence": None,
            "provenance": [],
            "risk_flags": [],
            "reason": "synthetic test",
        }
        for name, label in PHASE_LABELS.items()
    }
    metrics = {
        name: metric(name, "insufficient_data", reason="synthetic test") for name in METRIC_LABELS
    }
    assert all(item["value"] is None for item in metrics.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "input": {"fps": 30.0, "frame_count": 60},
        "quality": {},
        "attempt": {"shooting_side": "right", "view": {"value": "unknown", "status": "needs_review"}},
        "phases": phases,
        "events": events,
        "ball_evidence": {},
        "human_ball_release": unavailable_human_ball_release("self_test_fixture"),
        "metrics": metrics,
        "observations": [],
        "suggestions": [],
        "risks": [],
        "runtime": {},
        "artifacts": {},
    }
    report["motion_representation"] = build_motion_representation(report, [])
    validate_report(report)


def main() -> None:
    test_phase_event_ordering()
    test_landing_requires_persistent_return()
    test_strict_release_separation()
    test_unavailable_metric_and_report_contract()
    print("reference_v1 self-test: PASS")


if __name__ == "__main__":
    main()
