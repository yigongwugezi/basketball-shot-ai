from __future__ import annotations

import copy

import numpy as np

from benchmarks.reference_v1.model_adapters import RtmlibPoseAdapter

from .analysis import joint_angle
from .perception import ShooterContinuitySelector
from .pose.reliability import build_analysis_pose


def _candidate(box: list[float], offset: float = 0.0) -> dict:
    points = [[box[0] + 10 + offset, box[1] + 10 + index * 4] for index in range(17)]
    return {
        "bbox": box,
        "keypoints": points,
        "confidence": [0.95] * 17,
        "visible_keypoints": 17,
        "person_count": 2,
    }


def test_larger_bystander_does_not_steal_track() -> None:
    selector = ShooterContinuitySelector()
    shooter = _candidate([0, 0, 100, 200])
    first = selector.select([shooter, _candidate([300, 0, 360, 120])])
    assert first.candidate is shooter
    continuing = _candidate([5, 2, 100, 202])
    bystander = _candidate([180, -20, 360, 260])
    second = selector.select([bystander, continuing])
    assert second.candidate is continuing
    assert second.track_id == first.track_id
    assert not second.identity_break


def test_long_gap_requires_reacquisition_and_identity_break() -> None:
    selector = ShooterContinuitySelector(max_gap=2)
    first = selector.select([_candidate([0, 0, 100, 200])])
    assert first.track_id == 1
    for _ in range(3):
        missing = selector.select([])
        assert missing.candidate is None
    reacquired = selector.select([_candidate([250, 0, 350, 200])])
    assert reacquired.identity_break
    assert reacquired.track_id == 2
    assert reacquired.crop_status == "reacquired"


def test_rtmpose_adapter_preserves_original_frame_coordinates() -> None:
    class FakeModel:
        @staticmethod
        def pose_model(frame: np.ndarray, bboxes: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
            assert bboxes == [[10, 20, 90, 180]]
            points = np.asarray([[[20.0 + index, 30.0 + index] for index in range(17)]])
            return points, np.asarray([[0.9] * 17])

    adapter = object.__new__(RtmlibPoseAdapter)
    adapter.model = FakeModel()
    adapter.candidate = "rtmpose"
    pose, _ = adapter.infer(np.zeros((200, 100, 3), dtype=np.uint8), [10, 20, 90, 180])
    assert pose is not None
    assert pose["keypoints"][0] == [20.0, 30.0]
    assert pose["keypoints"][16] == [36.0, 46.0]


def test_rtmpose_adapter_missing_pose() -> None:
    class FakeModel:
        @staticmethod
        def pose_model(frame: np.ndarray, bboxes: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
            return np.empty((0, 17, 2)), np.empty((0, 17))

    adapter = object.__new__(RtmlibPoseAdapter)
    adapter.model = FakeModel()
    adapter.candidate = "rtmpose"
    pose, _ = adapter.infer(np.zeros((20, 20, 3), dtype=np.uint8), [0, 0, 10, 10])
    assert pose is None


def test_raw_localization_is_separate_from_temporal_signal() -> None:
    raw = _candidate([0, 0, 100, 200])
    raw["provenance"] = ["rtmpose-m_body7_256x192_raw"]
    rows = [{"frame_index": index, "raw_pose": copy.deepcopy(raw)} for index in range(5)]
    before = copy.deepcopy(rows)
    result = build_analysis_pose(rows, smooth=False)
    assert rows == before
    assert result[0]["analysis_pose"] is not result[0]["raw_pose"]
    assert result[0]["analysis_pose"]["raw_derived_status"] == "derived_temporal_signal"
    assert "adaptive_zero_phase_smoothing" not in result[0]["analysis_pose"]["provenance"]


def test_missing_raw_pose_does_not_alias_none() -> None:
    result = build_analysis_pose([{"frame_index": 0, "raw_pose": None}], smooth=False)
    assert result[0]["raw_pose"] is None
    assert result[0]["analysis_pose"] is None


def test_downstream_joint_angle_mapping() -> None:
    assert joint_angle(np.asarray([0.0, 0.0]), np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0])) == 90.0


def main() -> None:
    test_larger_bystander_does_not_steal_track()
    test_long_gap_requires_reacquisition_and_identity_break()
    test_rtmpose_adapter_preserves_original_frame_coordinates()
    test_rtmpose_adapter_missing_pose()
    test_raw_localization_is_separate_from_temporal_signal()
    test_missing_raw_pose_does_not_alias_none()
    test_downstream_joint_angle_mapping()
    print("reference_v1 perception tests: PASS")


if __name__ == "__main__":
    main()
