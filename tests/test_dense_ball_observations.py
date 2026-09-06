import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import main
from backend.measurement import (
    DenseBallFrameObservation,
    DenseBallObservationEvidence,
)


class FakeCapture:
    def __init__(self, _path):
        self.index = 0

    def isOpened(self):
        return True

    def set(self, _property, value):
        self.index = int(value)

    def read(self):
        return True, self.index

    def get(self, _property):
        return self.index * 100

    def release(self):
        pass


class DenseBallObservationTest(unittest.TestCase):
    def test_dense_scan_preserves_both_detectors_and_ambiguity(self):
        general = {
            0: [],
            1: [{"class_name": "ball", "xyxy": [1, 2, 5, 6], "confidence": 0.8, "source": "coco"}],
            2: [
                {"class_name": "ball", "xyxy": [1, 2, 5, 6], "confidence": 0.8, "source": "coco"},
                {"class_name": "ball", "xyxy": [7, 8, 11, 12], "confidence": 0.7, "source": "custom"},
            ],
            3: [],
        }
        release = {
            0: [],
            1: [{"bbox": [2, 3, 6, 7], "confidence": 0.9, "source": "release_ball_yolo"}],
            2: [
                {"bbox": [2, 3, 6, 7], "confidence": 0.9, "source": "release_ball_yolo"},
                {"bbox": [8, 9, 12, 13], "confidence": 0.6, "source": "release_ball_yolo"},
            ],
            3: [],
        }
        with (
            patch.object(main.cv2, "VideoCapture", FakeCapture),
            patch.object(main, "get_release_ball_model", return_value=(object(), None, None)),
            patch.object(main, "detect_frame", side_effect=lambda frame: general[frame]),
            patch.object(main, "release_ball_detections", side_effect=lambda frame, _model: release[frame]),
            patch.object(main, "encode_jpeg", return_value="data:image/jpeg;base64,test"),
        ):
            evidence = main.build_dense_ball_observations(
                "shot.mp4",
                {"frame_count": 4, "fps": 10, "width": 100, "height": 100},
            )

        self.assertEqual(evidence.scanned_frame_count, 4)
        self.assertEqual(evidence.general_ball_detected_frame_count, 2)
        self.assertEqual(evidence.general_multiple_frame_count, 1)
        self.assertEqual(evidence.release_ball_detected_frame_count, 2)
        self.assertEqual(evidence.release_multiple_frame_count, 1)
        self.assertEqual(evidence.unique_selected_observation_count, 1)
        self.assertEqual(evidence.missing_frame_count, 2)
        self.assertEqual(evidence.frames[0].release_ball_status, "MISSING")
        self.assertEqual(evidence.frames[1].release_ball_status, "DETECTED")
        self.assertIsNotNone(evidence.frames[1].selected_ball_observation)
        self.assertEqual(evidence.frames[2].release_ball_status, "MULTIPLE")
        self.assertIsNone(evidence.frames[2].selected_ball_observation)
        self.assertEqual(evidence.frames[1].timestamp, 0.1)

    def test_short_video_scans_every_frame(self):
        with (
            patch.object(main.cv2, "VideoCapture", FakeCapture),
            patch.object(main, "get_release_ball_model", return_value=(object(), None, None)),
            patch.object(main, "detect_frame", return_value=[]),
            patch.object(main, "release_ball_detections", return_value=[]),
            patch.object(main, "encode_jpeg", return_value="data:image/jpeg;base64,test"),
        ):
            evidence = main.build_dense_ball_observations(
                "shot.mp4",
                {"frame_count": 150, "fps": 30, "width": 852, "height": 480},
            )

        self.assertEqual(evidence.scan_stride, 1)
        self.assertEqual(evidence.scanned_frame_count, 150)

    def test_measurement_membership_is_added_without_changing_observations(self):
        evidence = DenseBallObservationEvidence(
            status="ok",
            total_frame_count=3,
            scanned_frame_count=3,
            frames=[
                DenseBallFrameObservation(index, index / 10, "pts", "image")
                for index in range(3)
            ],
        )
        measurement = SimpleNamespace(
            trusted_flight=SimpleNamespace(start_frame=1, end_frame=2),
            release_state=SimpleNamespace(
                release_epoch=SimpleNamespace(lower_time_s=0.0, upper_time_s=0.1)
            ),
        )

        result = main.add_measurement_membership(evidence, measurement)

        self.assertEqual(
            [item.trusted_flight_membership for item in result.frames],
            [False, True, True],
        )
        self.assertEqual(
            [item.release_epoch_membership for item in result.frames],
            [True, True, False],
        )
        self.assertEqual(result.frames[1].image_data_url, "image")


if __name__ == "__main__":
    unittest.main()
