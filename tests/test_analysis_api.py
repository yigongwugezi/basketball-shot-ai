import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")
if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics

from backend import main


class AnalyzeApiTest(unittest.TestCase):
    def _response(self, evidence: dict) -> dict:
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
            patch.object(main, "release_ball_detector_enabled", return_value=True),
            patch.object(main, "build_release_ball_evidence", return_value=evidence),
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
        self.assertIsNone(measurement["evidence"]["fit_rms_cm"])
        self.assertIsNone(measurement["evidence"]["release_epoch_uncertainty_ms"])

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


if __name__ == "__main__":
    unittest.main()
