from __future__ import annotations

import json

import numpy as np

from .human_ball import build_ball_track, build_human_ball_release, build_relations, decode_contact_states


def _pose(wrist: tuple[float, float] = (100.0, 80.0)) -> dict:
    points = [[100.0, 200.0] for _ in range(17)]
    points[5] = [90.0, 160.0]
    points[6] = [100.0, 160.0]
    points[7] = [90.0, 120.0]
    points[8] = [100.0, 120.0]
    points[9] = [90.0, 80.0]
    points[10] = list(wrist)
    points[11] = [90.0, 220.0]
    points[12] = [100.0, 220.0]
    points[13] = [90.0, 280.0]
    points[14] = [100.0, 280.0]
    points[15] = [90.0, 340.0]
    points[16] = [100.0, 340.0]
    return {
        "keypoints": points,
        "confidence": [0.95] * 17,
        "temporal_reliability": [0.95] * 17,
        "bbox": [50.0, 40.0, 150.0, 360.0],
    }


def _ball(center: tuple[float, float], confidence: float = 0.9, diameter: float = 20.0) -> dict:
    x, y = center
    radius = diameter / 2
    return {
        "center": [x, y],
        "bbox": [x - radius, y - radius, x + radius, y + radius],
        "diameter": diameter,
        "confidence": confidence,
        "source": "release_ball_v1",
    }


def _rows(centers: list[tuple[float, float] | None]) -> list[dict]:
    return [
        {
            "frame_index": frame,
            "time_seconds": frame / 30,
            "analysis_pose": _pose(),
            "ball": _ball(center) if center is not None else None,
            "ball_candidates": [_ball(center)] if center is not None else [],
        }
        for frame, center in enumerate(centers)
    ]


def _signals(count: int) -> list[dict]:
    return [{"frame_index": frame} for frame in range(count)]


def test_track_selection_and_gap_policy() -> None:
    rows = _rows([(100.0, 80.0), None, (104.0, 76.0), None, None, None, (110.0, 68.0)])
    rows[2]["ball_candidates"].append(_ball((400.0, 400.0), 0.99))
    track, rejected = build_ball_track(rows, "right", "analysis_pose", 0, 6)
    assert track[1]["ball_status"] == "INTERPOLATED"
    assert all(track[index]["ball_status"] == "MISSING" for index in (3, 4, 5))
    assert track[2]["center"] == [104.0, 76.0]
    assert rejected == 1


def test_ambiguity_is_explicit() -> None:
    rows = _rows([(100.0, 80.0)])
    rows[0]["ball_candidates"] = [_ball((100.0, 80.0), 0.80), _ball((102.0, 80.0), 0.78)]
    track, _ = build_ball_track(rows, "right", "analysis_pose", 0, 0)
    assert track[0]["ball_status"] == "AMBIGUOUS"


def test_contact_transition_and_pose_constraint() -> None:
    centers = [(100.0, 82.0)] * 9 + [
        (100.0, 80.0),
        (100.0, 78.0),
        (100.0, 54.0),
        (100.0, 44.0),
        (100.0, 34.0),
        (100.0, 24.0),
    ] + [(100.0, 16.0)] * 6
    rows = _rows(centers)
    score = np.zeros(len(rows), dtype=float)
    score[5] = 1.0
    score[11] = 0.9
    result = build_human_ball_release(rows, _signals(len(rows)), "right", "analysis_pose", 5, score, 30.0)
    assert result["strict_release"]["status"] == "ok"
    assert result["strict_release"]["frame"] == 12
    assert result["strict_release"]["supporting_frame_interval"] == [12, 14]
    assert result["release_pose"]["frame"] == 11
    assert result["release_pose"]["pose_only_frame"] == 5
    states = {item["frame"]: item["contact_state"] for item in result["contact_state_sequence"]}
    assert states[10] == "LIKELY_CONTACT"
    assert states[11] == "SEPARATING"
    assert states[12] == "NO_CONTACT"
    relation = result["contact_state_sequence"][11]
    assert relation["wrist_ball_distance_radii"] == 2.6
    assert relation["relative_speed_diameters_per_frame"] > 1.0
    assert relation["elbow_angle_degrees_2d"] is not None
    assert json.loads(json.dumps(result))["schema_version"] == "human_ball_release_v1"


def test_missing_ball_abstains_and_preserves_pose_release() -> None:
    rows = _rows([None] * 24)
    score = np.zeros(len(rows), dtype=float)
    score[10] = 1.0
    result = build_human_ball_release(rows, _signals(len(rows)), "right", "analysis_pose", 10, score, 30.0)
    assert result["strict_release"]["status"] == "insufficient_data"
    assert result["release_pose"]["frame"] == 10
    assert result["release_pose"]["source"] == "pose_only_fallback"
    assert result["ball_track_status"] == "insufficient_data"


def test_strict_transition_frame_remains_separating() -> None:
    rows = _rows([(100.0, 80.0), (100.0, 78.0), (100.0, 48.0), (100.0, 40.0), (100.0, 32.0)])
    track, _ = build_ball_track(rows, "right", "analysis_pose", 0, 4)
    states, strict = decode_contact_states(build_relations(rows, track, "right", "analysis_pose"))
    assert strict["frame"] == 2
    assert states[2]["contact_state"] == "SEPARATING"
    assert states[3]["contact_state"] == "NO_CONTACT"


def main() -> None:
    test_track_selection_and_gap_policy()
    test_ambiguity_is_explicit()
    test_contact_transition_and_pose_constraint()
    test_missing_ball_abstains_and_preserves_pose_release()
    test_strict_transition_frame_remains_separating()
    print("human_ball tests passed")


if __name__ == "__main__":
    main()
