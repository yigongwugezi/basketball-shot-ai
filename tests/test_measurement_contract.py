import json
import unittest

from backend.measurement import (
    MeasurementStatus,
    ReleaseMeasurementResult,
    release_fusion_to_measurement,
)


class ReleaseMeasurementContractTest(unittest.TestCase):
    def test_available_result_serializes_explicitly(self) -> None:
        result = ReleaseMeasurementResult(
            status=MeasurementStatus.AVAILABLE,
            release_frame=42,
            release_time=1.4,
            source="pose_release",
        ).to_dict()

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["release_frame"], 42)
        self.assertEqual(result["source"], "pose_release")
        self.assertIsNone(result["release_state"])
        json.dumps(result)

    def test_insufficient_data_does_not_imply_a_measurement(self) -> None:
        result = ReleaseMeasurementResult(
            status=MeasurementStatus.INSUFFICIENT_DATA,
            reason="release evidence is missing",
        ).to_dict()

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["release_frame"])
        self.assertIsNone(result["release_time"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["release_state"])

    def test_normal_fusion_preserves_existing_result(self) -> None:
        fusion = {
            "status": "ok",
            "final_source": "pose_release",
            "pose_release_frame_index": 42,
            "detector_release_frame_index": 43,
            "frame_delta": 1,
            "agreement_level": "near_agreement_1",
            "reason": "detector is near pose release frame within 1 frame",
            "risk_flags": ["existing_flag"],
        }

        result = release_fusion_to_measurement(fusion).to_dict()

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["release_frame"], 42)
        self.assertEqual(result["source"], "pose_release")
        self.assertEqual(result["frame_delta"], 1)
        self.assertEqual(result["agreement_level"], "near_agreement_1")
        self.assertEqual(
            result["reason"], "detector is near pose release frame within 1 frame"
        )
        self.assertEqual(result["risk_flags"], ["existing_flag"])
        self.assertIsNone(result["release_time"])
        self.assertIsNone(result["release_state"])

    def test_fallback_source_does_not_imply_confidence_or_release_state(self) -> None:
        result = release_fusion_to_measurement(
            {
                "status": "detector_unavailable",
                "final_source": "pose_release",
                "pose_release_frame_index": 42,
                "risk_flags": ["detector_unavailable"],
            }
        ).to_dict()

        self.assertEqual(result["status"], "UNRELIABLE")
        self.assertEqual(result["source"], "pose_release")
        self.assertEqual(result["release_frame"], 42)
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["release_state"])

    def test_insufficient_fusion_is_not_upgraded_when_frame_exists(self) -> None:
        result = release_fusion_to_measurement(
            {
                "status": "insufficient_data",
                "final_source": "pose_release",
                "pose_release_frame_index": 42,
                "reason": "detector found no release-ball detection",
            }
        ).to_dict()

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["release_frame"], 42)
        self.assertIsNone(result["release_state"])


if __name__ == "__main__":
    unittest.main()
