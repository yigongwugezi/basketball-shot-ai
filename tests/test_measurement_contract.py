import json
import unittest

from backend.measurement import MeasurementStatus, ReleaseMeasurementResult


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


if __name__ == "__main__":
    unittest.main()
