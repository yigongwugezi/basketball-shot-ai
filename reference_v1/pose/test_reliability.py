from __future__ import annotations

import copy

import numpy as np

from .metrics import estimate_signal_lag
from .reliability import build_analysis_pose


def _pose(offset: float = 0.0) -> dict:
    points = [[20.0 + offset, 10.0 + index * 5] for index in range(17)]
    points[5], points[6], points[11], points[12] = [10 + offset, 10], [30 + offset, 10], [12 + offset, 50], [28 + offset, 50]
    points[7], points[8], points[9], points[10] = [8 + offset, 30], [32 + offset, 30], [6 + offset, 48], [34 + offset, 48]
    points[13], points[14], points[15], points[16] = [12 + offset, 70], [28 + offset, 70], [12 + offset, 90], [28 + offset, 90]
    return {"bbox": [0 + offset, 0, 40 + offset, 100], "keypoints": points, "confidence": [0.95] * 17}


def _rows(count: int = 9) -> list[dict]:
    return [{"frame_index": frame, "raw_pose": _pose(frame * 2.0)} for frame in range(count)]


def test_outlier_rejection() -> None:
    rows = _rows()
    rows[4]["raw_pose"]["keypoints"][9] = [400, 400]
    result = build_analysis_pose(rows)
    assert result[4]["analysis_pose"]["joint_status"][9] == "corrected"
    assert result[4]["analysis_pose"]["keypoints"][9][0] < 100


def test_bounded_interpolation() -> None:
    rows = _rows()
    rows[4]["raw_pose"]["confidence"][9] = 0.0
    result = build_analysis_pose(rows)
    assert result[4]["analysis_pose"]["joint_status"][9] == "interpolated"
    assert result[4]["analysis_pose"]["temporal_reliability"][9] > 0


def test_no_long_gap_hallucination() -> None:
    rows = _rows(11)
    for frame in range(3, 7):
        rows[frame]["raw_pose"]["confidence"][9] = 0.0
    result = build_analysis_pose(rows)
    assert all(result[frame]["analysis_pose"]["joint_status"][9] == "unavailable" for frame in range(3, 7))
    assert all(result[frame]["analysis_pose"]["temporal_reliability"][9] == 0 for frame in range(3, 7))


def test_zero_phase_no_lag() -> None:
    raw = np.exp(-((np.arange(31) - 15) / 3) ** 2)
    filtered = raw.copy()
    filtered[1:-1] = 0.25 * raw[:-2] + 0.5 * raw[1:-1] + 0.25 * raw[2:]
    assert estimate_signal_lag(raw, filtered) == 0


def test_raw_pose_immutability_and_provenance() -> None:
    rows = _rows()
    before = copy.deepcopy(rows)
    result = build_analysis_pose(rows)
    assert rows == before
    assert result[0]["raw_pose"] == before[0]["raw_pose"]
    assert result[0]["analysis_pose"] is not result[0]["raw_pose"]
    assert result[0]["analysis_pose"]["correction_status"] in {"observed", "corrected", "interpolated", "unavailable"}
    assert "adaptive_zero_phase_smoothing" in result[0]["analysis_pose"]["provenance"]


def main() -> None:
    test_outlier_rejection()
    test_bounded_interpolation()
    test_no_long_gap_hallucination()
    test_zero_phase_no_lag()
    test_raw_pose_immutability_and_provenance()
    print("pose reliability tests: PASS")


if __name__ == "__main__":
    main()
