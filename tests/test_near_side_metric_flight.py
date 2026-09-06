import math
import unittest

import numpy as np

from backend.measurement import (
    BallTrackDetection,
    BallTrackEvidence,
    BallTrackFrameEvidence,
)
from backend.near_side_metric_flight import (
    BASKETBALL_DIAMETER_M,
    GRAVITY_MPS2,
    estimate_near_side_metric_flight,
)


def synthetic_track(
    *,
    horizontal_speed: float = 4.0,
    vertical_speed: float = 6.0,
    pitch_deg: float = 0.0,
    drag: float = 0.0,
    count: int = 17,
    fps: float = 30.0,
    size_bias=lambda _time: (1.0, 1.0),
) -> BallTrackEvidence:
    width_px, height_px, focal_px = 640, 480, 800.0
    pitch = math.radians(pitch_deg)
    up = np.array([0.0, -math.cos(pitch), math.sin(pitch)])
    gravity = -GRAVITY_MPS2 * up
    velocity = np.array(
        [
            horizontal_speed,
            -vertical_speed * math.cos(pitch),
            vertical_speed * math.sin(pitch),
        ]
    )
    origin = np.array([0.0, 0.0, 8.0])
    frames = [
        BallTrackFrameEvidence(frame_index=0, time_s=0.0, status="no_detection")
    ]
    for index in range(1, count):
        time_s = index / fps
        if drag == 0:
            f1, fg = time_s, 0.5 * time_s**2
        else:
            f1 = (1 - math.exp(-drag * time_s)) / drag
            fg = time_s / drag - f1 / drag
        point = origin + velocity * f1 + gravity * fg
        diameter_px = focal_px * BASKETBALL_DIAMETER_M / point[2]
        width_bias, height_bias = size_bias(time_s)
        bbox_width = diameter_px * width_bias
        bbox_height = diameter_px * height_bias
        center_x = width_px / 2 + focal_px * point[0] / point[2]
        center_y = height_px / 2 + focal_px * point[1] / point[2]
        bbox = [
            center_x - bbox_width / 2,
            center_y - bbox_height / 2,
            center_x + bbox_width / 2,
            center_y + bbox_height / 2,
        ]
        detection = BallTrackDetection(
            bbox=bbox,
            center_x_px=center_x,
            center_y_px=center_y,
            confidence=0.9,
            source="synthetic",
        )
        frames.append(
            BallTrackFrameEvidence(
                frame_index=index,
                time_s=time_s,
                detections=[detection],
                bbox=bbox,
                center_x_px=center_x,
                center_y_px=center_y,
                confidence=0.9,
                source="synthetic",
                timestamp_source="pts",
                status="ok",
            )
        )
    return BallTrackEvidence(
        status="ok",
        fps=fps,
        image_width_px=width_px,
        image_height_px=height_px,
        frames=frames,
        actual_timestamps=[item.time_s for item in frames if item.time_s is not None],
        requested_frame_count=len(frames),
        observed_frame_count=len(frames),
        detection_frame_count=count - 1,
        missing_frame_count=1,
    )


class NearSideMetricFlightTest(unittest.TestCase):
    def assert_recovers_release(self, result) -> None:
        expected_speed = math.hypot(4.0, 6.0)
        expected_angle = math.degrees(math.atan2(6.0, 4.0))
        self.assertTrue(result.qualified)
        self.assertAlmostEqual(result.speed_mps, expected_speed, delta=0.5)
        self.assertAlmostEqual(result.elevation_angle_deg, expected_angle, delta=3.0)

    def test_ideal_camera_space_ballistic_track(self) -> None:
        self.assert_recovers_release(
            estimate_near_side_metric_flight(synthetic_track(), 0)
        )

    def test_pitched_camera_recovers_speed_and_elevation(self) -> None:
        self.assert_recovers_release(
            estimate_near_side_metric_flight(synthetic_track(pitch_deg=18.0), 0)
        )

    def test_scale_biased_bbox_track_keeps_proxy_sensitivity(self) -> None:
        result = estimate_near_side_metric_flight(
            synthetic_track(
                size_bias=lambda time: (1 + 0.35 * time, 1 - 0.15 * time)
            ),
            0,
        )

        self.assertTrue(result.available)
        self.assertIsNotNone(result.speed_interval_mps)
        self.assertIn("size_proxy_and_time_domain_smoothing", result.uncertainty_sources)

    def test_effective_drag_synthetic_track(self) -> None:
        self.assert_recovers_release(
            estimate_near_side_metric_flight(synthetic_track(drag=1.0), 0)
        )

    def test_insufficient_temporal_support_is_unavailable(self) -> None:
        result = estimate_near_side_metric_flight(synthetic_track(count=10), 0)

        self.assertFalse(result.available)
        self.assertEqual(result.qualification, "UNAVAILABLE")
        self.assertIn("insufficient_temporal_support", result.quality_flags)

    def test_unstable_size_profile_is_unqualified(self) -> None:
        result = estimate_near_side_metric_flight(
            synthetic_track(
                size_bias=lambda time: (
                    1 + 2.8 * time,
                    max(0.05, 1 - 1.6 * time),
                )
            ),
            0,
        )

        self.assertTrue(result.available)
        self.assertEqual(result.qualification, "UNQUALIFIED")
        self.assertIsNone(result.speed_mps)

    def test_release_epoch_interval_is_propagated(self) -> None:
        result = estimate_near_side_metric_flight(synthetic_track(), 0)

        self.assertIsNotNone(result.release_epoch)
        self.assertEqual(result.release_epoch.lower_time_s, 0.0)
        self.assertEqual(result.release_epoch.upper_time_s, 1 / 30)
        self.assertGreater(result.release_epoch.representative_time_s, 0.0)


if __name__ == "__main__":
    unittest.main()
