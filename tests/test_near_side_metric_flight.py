import math
import unittest

from backend.measurement import (
    BallTrackDetection,
    BallTrackEvidence,
    BallTrackFrameEvidence,
)
from backend.near_side_metric_flight import (
    GRAVITY_MPS2,
    estimate_near_side_metric_flight,
)


def synthetic_track(
    *,
    vx: float = 4.0,
    vy: float = 6.0,
    drag: float = 0.0,
    count: int = 17,
    fps: float = 30.0,
    size_fn=lambda _time: (20.0, 20.0),
    center_size_fn=lambda _time: 20.0,
) -> BallTrackEvidence:
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
        # Synthetic image trajectory follows the same near-side pseudo-metric model.
        ux = vx * f1 / 0.1
        uy = (vy * f1 - GRAVITY_MPS2 * fg) / 0.1
        draw_size = center_size_fn(time_s)
        center_x, center_y = 320 + ux * draw_size, 240 - uy * draw_size
        width, height = size_fn(time_s)
        bbox = [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
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
        frames=frames,
        actual_timestamps=[item.time_s for item in frames if item.time_s is not None],
        requested_frame_count=len(frames),
        observed_frame_count=len(frames),
        detection_frame_count=count - 1,
        missing_frame_count=1,
    )


class NearSideMetricFlightTest(unittest.TestCase):
    def test_ideal_synthetic_ballistic_track(self) -> None:
        result = estimate_near_side_metric_flight(synthetic_track(), 0)

        self.assertTrue(result.available)
        self.assertLessEqual(result.speed_interval_mps[0], math.hypot(4.0, 6.0))
        self.assertGreaterEqual(result.speed_interval_mps[1], math.hypot(4.0, 6.0))
        self.assertLessEqual(result.elevation_interval_deg[0], math.degrees(math.atan2(6, 4)))
        self.assertGreaterEqual(result.elevation_interval_deg[1], math.degrees(math.atan2(6, 4)))

    def test_scale_biased_bbox_track_keeps_proxy_sensitivity(self) -> None:
        result = estimate_near_side_metric_flight(
            synthetic_track(size_fn=lambda time: (20 * (1 + 0.35 * time), 20 * (1 - 0.15 * time))),
            0,
        )

        self.assertTrue(result.available)
        self.assertIsNotNone(result.speed_interval_mps)
        self.assertIn("size_proxy_and_time_domain_smoothing", result.uncertainty_sources)

    def test_effective_drag_synthetic_track(self) -> None:
        result = estimate_near_side_metric_flight(synthetic_track(drag=1.0), 0)

        self.assertTrue(result.available)
        self.assertLessEqual(result.speed_interval_mps[0], math.hypot(4.0, 6.0))
        self.assertGreaterEqual(result.speed_interval_mps[1], math.hypot(4.0, 6.0))

    def test_insufficient_temporal_support_is_unavailable(self) -> None:
        result = estimate_near_side_metric_flight(synthetic_track(count=10), 0)

        self.assertFalse(result.available)
        self.assertEqual(result.qualification, "UNAVAILABLE")
        self.assertIn("insufficient_temporal_support", result.quality_flags)

    def test_unstable_size_profile_is_unqualified(self) -> None:
        result = estimate_near_side_metric_flight(
            synthetic_track(
                size_fn=lambda time: (20 * (1 + 2.8 * time), 20 * max(0.05, 1 - 1.6 * time))
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
