import json
import unittest

from backend.measurement import (
    MeasurementEvidence,
    MeasurementQuality,
    MeasurementStatus,
    ReleaseMeasurementResult,
    pose_release_to_measurement,
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

    def test_quality_and_evidence_serialize_without_research_estimates(self) -> None:
        result = ReleaseMeasurementResult(
            status=MeasurementStatus.AVAILABLE,
            evidence=MeasurementEvidence(release_sources=["pose_release"]),
            measurement_quality=MeasurementQuality(
                status=MeasurementStatus.AVAILABLE
            ),
        ).to_dict()

        self.assertEqual(result["measurement_quality"]["status"], "AVAILABLE")
        self.assertIsNone(result["measurement_quality"]["confidence"])
        self.assertEqual(result["evidence"]["release_sources"], ["pose_release"])
        self.assertIsNone(result["evidence"]["existing_tracking_evidence"])
        self.assertIsNone(result["evidence"]["trusted_window_point_count"])
        self.assertIsNone(result["evidence"]["trusted_window_temporal_span_ms"])
        self.assertIsNone(result["evidence"]["fit_rms_cm"])
        self.assertIsNone(
            result["evidence"]["ensemble_velocity_disagreement_mps"]
        )
        self.assertIsNone(
            result["evidence"]["leave_one_out_velocity_disagreement_mps"]
        )
        self.assertIsNone(result["evidence"]["holdout_prediction_error_cm"])
        self.assertIsNone(result["evidence"]["release_epoch_uncertainty_ms"])
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

        self.assertEqual(
            result["evidence"]["frame_agreement"],
            {
                "pose_release_frame_index": 42,
                "detector_release_frame_index": 43,
                "frame_delta": 1,
                "agreement_level": "near_agreement_1",
            },
        )
        self.assertEqual(result["evidence"]["risk_flags"], ["existing_flag"])
        self.assertEqual(result["measurement_quality"]["warnings"], ["existing_flag"])
        self.assertEqual(result["measurement_quality"]["issues"], [])
        self.assertIsNone(result["measurement_quality"]["confidence"])

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
        self.assertEqual(
            result["measurement_quality"]["issues"], ["detector_unavailable"]
        )
        self.assertIsNone(result["measurement_quality"]["confidence"])

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

    def test_pose_only_measurement_is_unreliable_without_corroboration(self) -> None:
        result = pose_release_to_measurement(42).to_dict()

        self.assertEqual(result["status"], "UNRELIABLE")
        self.assertEqual(result["release_frame"], 42)
        self.assertEqual(result["source"], "pose_release")
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["release_time"])
        self.assertIsNone(result["release_state"])
        self.assertEqual(result["evidence"]["release_sources"], ["pose_release"])
        self.assertEqual(result["measurement_quality"]["issues"], ["release_ball_corroboration_unavailable"])

    def test_missing_pose_release_frame_is_insufficient(self) -> None:
        result = pose_release_to_measurement(None).to_dict()

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["release_frame"])
        self.assertIsNone(result["release_state"])


if __name__ == "__main__":
    unittest.main()
