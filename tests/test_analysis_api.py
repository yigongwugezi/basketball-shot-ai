import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.measurement import BallTrackEvidence


if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")
if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics

from backend import main


class AnalyzeApiTest(unittest.TestCase):
    def _response(self, evidence: dict, detector_enabled: bool = True) -> dict:
        frame_indices = {
            key: {
                "frame_index": index,
                "selection_method": "test",
                "confidence": None,
                "evidence": None,
            }
            for index, key in enumerate(main.FRAME_KEYS)
        }
        frame_indices["release"]["frame_index"] = 42
        track_evidence = BallTrackEvidence(
            status="ok",
            start_frame=39,
            end_frame=57,
            fps=30,
            requested_frame_count=19,
            observed_frame_count=19,
            detection_frame_count=12,
            missing_frame_count=7,
            actual_timestamps=[1.3, 1.3333],
        )

        with (
            patch.object(main, "video_metadata", return_value={"fps": 30, "frame_count": 90}),
            patch.object(main, "candidate_frame_indices", return_value=frame_indices),
            patch.object(main, "read_frame", return_value=object()),
            patch.object(main, "detect_frame", return_value=[]),
            patch.object(main, "detect_pose", return_value=None),
            patch.object(main, "draw_detections", side_effect=lambda frame, _: frame),
            patch.object(main, "draw_pose", side_effect=lambda frame, _: frame),
            patch.object(main, "encode_jpeg", return_value="data:image/jpeg;base64,test"),
            patch.object(main, "estimate_camera_view", return_value=None),
            patch.object(main, "quality_checks_v2", return_value=[]),
            patch.object(main, "summarize_metrics_v2", return_value=[]),
            patch.object(main, "release_ball_detector_enabled", return_value=detector_enabled),
            patch.object(main, "build_release_ball_evidence", return_value=evidence),
            patch.object(main, "build_ball_track_evidence", return_value=track_evidence),
        ):
            response = TestClient(main.app).post(
                "/api/analyze",
                files={"file": ("shot.mp4", b"test-video", "video/mp4")},
            )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_release_measurement_preserves_fusion_and_serializes_nulls(self) -> None:
        report = self._response(
            {
                "status": "ok",
                "best_frame": {"frame_index": 43},
            }
        )

        self.assertIn("release_fusion", report)
        measurement = report["release_measurement"]
        self.assertEqual(measurement["status"], "AVAILABLE")
        self.assertEqual(measurement["release_frame"], 42)
        self.assertEqual(
            measurement["release_frame"],
            report["release_fusion"]["pose_release_frame_index"],
        )
        self.assertIsNone(measurement["release_state"])
        self.assertIsNone(measurement["release_time"])
        self.assertIsNone(measurement["trusted_flight"])
        self.assertIsNone(measurement["evidence"]["fit_rms_cm"])
        self.assertIsNone(measurement["evidence"]["release_epoch_uncertainty_ms"])
        self.assertEqual(measurement["evidence"]["fps"], 30)
        self.assertEqual(measurement["evidence"]["missing_observations"], 7)
        self.assertEqual(report["ball_track_evidence"]["detection_frame_count"], 12)

    def test_insufficient_and_unreliable_statuses_survive_api_serialization(self) -> None:
        for evidence, expected_status in (
            ({"status": "no_detection", "best_frame": None}, "INSUFFICIENT_DATA"),
            ({"status": "model_missing", "best_frame": None}, "UNRELIABLE"),
        ):
            with self.subTest(expected_status=expected_status):
                report = self._response(evidence)
                measurement = report["release_measurement"]
                self.assertEqual(measurement["status"], expected_status)
                self.assertIsNone(measurement["release_state"])

    def test_detector_disabled_still_exposes_pose_measurement(self) -> None:
        report = self._response({}, detector_enabled=False)

        self.assertIn("ball_track_evidence", report)
        self.assertIsNone(report["ball_track_evidence"])
        self.assertNotIn("release_ball_evidence", report)
        self.assertNotIn("release_fusion", report)
        measurement = report["release_measurement"]
        self.assertEqual(measurement["status"], "UNRELIABLE")
        self.assertEqual(measurement["release_frame"], 42)
        self.assertEqual(measurement["source"], "pose_release")
        self.assertIsNone(measurement["release_state"])

    def test_continuous_track_preserves_missing_and_multiple_detections(self) -> None:
        detections = {
            5: [],
            6: [
                {"bbox": [1, 2, 5, 6], "confidence": 0.8, "source": "release_ball_yolo"},
                {"bbox": [10, 12, 14, 16], "confidence": 0.7, "source": "release_ball_yolo"},
            ],
            7: [{"bbox": [2, 4, 6, 8], "confidence": 0.9, "source": "release_ball_yolo"}],
            8: [],
            9: [],
        }
        with (
            patch.object(main, "get_release_ball_model", return_value=(object(), None, None)),
            patch.object(main, "read_frame", side_effect=lambda _, index: index),
            patch.object(main, "release_ball_detections", side_effect=lambda frame, _: detections[frame]),
        ):
            evidence = main.build_ball_track_evidence(
                "test.mp4",
                {"fps": 10, "frame_count": 10},
                8,
            )

        self.assertEqual((evidence.start_frame, evidence.end_frame), (5, 9))
        self.assertEqual(evidence.requested_frame_count, 5)
        self.assertEqual(evidence.observed_frame_count, 5)
        self.assertEqual(evidence.detection_frame_count, 2)
        self.assertEqual(evidence.missing_frame_count, 3)
        self.assertEqual(evidence.actual_timestamps, [0.5, 0.6, 0.7, 0.8, 0.9])
        self.assertEqual(evidence.frames[0].status, "no_detection")
        self.assertEqual(evidence.frames[1].status, "multiple_detections")
        self.assertEqual(len(evidence.frames[1].detections), 2)
        self.assertEqual(evidence.frames[2].bbox, [2, 4, 6, 8])
        self.assertIn("collection_truncated_at_video_end", evidence.warnings)


if __name__ == "__main__":
    unittest.main()
