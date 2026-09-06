"""Training-free V2 apparent-size camera-space flight estimation.

The basketball diameter is used as a size prior while focal length remains a
trajectory-level nuisance variable.  The estimator exposes only qualified
speed/elevation; it does not claim calibrated 3D position or physical drag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np

from backend.measurement import BallTrackEvidence


GRAVITY_MPS2 = 9.80665
BASKETBALL_DIAMETER_M = 0.239
LAMBDA_GRID_S_INV = (0.0, 0.5, 1.0, 1.5, 2.0)
FOCAL_PROFILE_MULTIPLIERS = (0.85, 0.925, 1.0, 1.075, 1.15)
PREFERRED_SUPPORT_SECONDS = (0.35, 0.50)
PROVISIONAL_MINIMUM_SUPPORT_SECONDS = 0.30
PROVISIONAL_MINIMUM_USABLE_OBSERVATIONS = 10
MAX_GAP_MEDIAN_MULTIPLIER = 2.5
PROFILE_EQUIVALENCE_RMS_M = 0.001
GRAVITY_PROFILE_TOLERANCE_MPS2 = 0.25
CENTER_LOCALIZATION_PX = 0.5
# Versioned engineering allowances, not universal basketball thresholds.
NEAR_SIDE_OUT_OF_PLANE_SPEED_ALLOWANCE_MPS = 0.08
NEAR_SIDE_OUT_OF_PLANE_ELEVATION_ALLOWANCE_DEG = 0.5
AERODYNAMIC_MODEL_MISMATCH_SPEED_ALLOWANCE_MPS = 0.08
AERODYNAMIC_MODEL_MISMATCH_ELEVATION_ALLOWANCE_DEG = 0.5


@dataclass(frozen=True, slots=True)
class FlightObservation:
    frame_index: int
    time_s: float
    center_x_px: float
    center_y_px: float
    bbox_width_px: float
    bbox_height_px: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ReleaseEpochInterval:
    lower_time_s: float
    upper_time_s: float
    representative_time_s: float
    lower_frame: int
    upper_frame: int
    reason: str


@dataclass(frozen=True, slots=True)
class MetricFlightResult:
    available: bool
    qualified: bool
    qualification: str
    trusted_start_frame: int | None = None
    trusted_end_frame: int | None = None
    trusted_start_time_s: float | None = None
    trusted_end_time_s: float | None = None
    observation_count: int = 0
    temporal_span_s: float | None = None
    release_epoch: ReleaseEpochInterval | None = None
    speed_mps: float | None = None
    elevation_angle_deg: float | None = None
    speed_interval_mps: tuple[float, float] | None = None
    elevation_interval_deg: tuple[float, float] | None = None
    fit_rms_m: float | None = None
    reasons: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    uncertainty_sources: list[str] = field(default_factory=list)
    timestamp_source: str = "unavailable"


@dataclass(frozen=True, slots=True)
class _ProfileSolution:
    speed_mps: float
    elevation_deg: float
    residual_rms_m: float
    gravity_error_mps2: float


def unique_observations(track: BallTrackEvidence) -> list[FlightObservation]:
    observations: list[FlightObservation] = []
    for frame in track.frames:
        if frame.time_s is None or len(frame.detections) != 1:
            continue
        detection = frame.detections[0]
        x1, y1, x2, y2 = detection.bbox
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        observations.append(
            FlightObservation(
                frame_index=frame.frame_index,
                time_s=frame.time_s,
                center_x_px=detection.center_x_px,
                center_y_px=detection.center_y_px,
                bbox_width_px=width,
                bbox_height_px=height,
                confidence=detection.confidence,
            )
        )
    return sorted(observations, key=lambda item: item.time_s)


def select_trusted_observations(
    observations: list[FlightObservation], release_frame: int | None
) -> list[FlightObservation]:
    post_release = [
        item
        for item in observations
        if release_frame is None or item.frame_index > release_frame
    ]
    if len(post_release) < 2:
        return []
    gaps = np.diff([item.time_s for item in post_release])
    positive_gaps = gaps[gaps > 0]
    if len(positive_gaps) == 0:
        return []
    allowed_gap = max(
        0.001, float(np.median(positive_gaps)) * MAX_GAP_MEDIAN_MULTIPLIER
    )
    segments: list[list[FlightObservation]] = [[]]
    for observation in post_release:
        if (
            segments[-1]
            and observation.time_s - segments[-1][-1].time_s > allowed_gap
        ):
            segments.append([])
        segments[-1].append(observation)
    viable = [
        segment
        for segment in segments
        if len(segment) >= PROVISIONAL_MINIMUM_USABLE_OBSERVATIONS
        and segment[-1].time_s - segment[0].time_s
        >= PROVISIONAL_MINIMUM_SUPPORT_SECONDS
    ]
    if not viable:
        return []
    return max(
        viable,
        key=lambda segment: (
            PREFERRED_SUPPORT_SECONDS[0]
            <= segment[-1].time_s - segment[0].time_s
            <= PREFERRED_SUPPORT_SECONDS[1],
            segment[-1].time_s - segment[0].time_s,
            len(segment),
        ),
    )


def _basis(time_s: np.ndarray, lambda_s_inv: float) -> tuple[np.ndarray, np.ndarray]:
    if lambda_s_inv == 0:
        return time_s, 0.5 * time_s**2
    exponential = np.exp(-lambda_s_inv * time_s)
    f1 = (1.0 - exponential) / lambda_s_inv
    fg = time_s / lambda_s_inv - f1 / lambda_s_inv
    return f1, fg


def _smoothed_size(
    observations: list[FlightObservation], proxy: str, degree: int
) -> np.ndarray:
    times = np.array([item.time_s for item in observations], dtype=float)
    raw_size = np.array(
        [
            min(item.bbox_width_px, item.bbox_height_px)
            if proxy == "min"
            else math.sqrt(item.bbox_width_px * item.bbox_height_px)
            for item in observations
        ],
        dtype=float,
    )
    reciprocal = 1.0 / raw_size
    coefficients = np.polyfit(times - times[0], reciprocal, degree)
    smoothed_reciprocal = np.polyval(coefficients, times - times[0])
    return 1.0 / np.maximum(smoothed_reciprocal, 1e-9)


def _back_propagate_velocity(
    velocity_start: np.ndarray,
    gravity: np.ndarray,
    lambda_s_inv: float,
    delta_time_s: float,
) -> np.ndarray:
    if lambda_s_inv == 0:
        return velocity_start - gravity * delta_time_s
    terminal = gravity / lambda_s_inv
    return (velocity_start - terminal) * math.exp(
        lambda_s_inv * delta_time_s
    ) + terminal


def _fit_solutions(
    observations: list[FlightObservation],
    proxy: str,
    degree: int,
    lambda_s_inv: float,
    release_time_s: float,
    image_width_px: int,
    image_height_px: int,
    center_perturbation_px: tuple[float, float] = (0.0, 0.0),
) -> list[_ProfileSolution]:
    times = np.array([item.time_s for item in observations], dtype=float)
    relative_times = times - times[0]
    size = _smoothed_size(observations, proxy, degree)
    cx = image_width_px / 2 + center_perturbation_px[0]
    cy = image_height_px / 2 + center_perturbation_px[1]
    center_x = np.array([item.center_x_px for item in observations], dtype=float)
    center_y = np.array([item.center_y_px for item in observations], dtype=float)
    x_m = BASKETBALL_DIAMETER_M * (center_x - cx) / size
    y_m = BASKETBALL_DIAMETER_M * (center_y - cy) / size
    z_per_f = BASKETBALL_DIAMETER_M / size

    f1, fg = _basis(relative_times, lambda_s_inv)
    design = np.column_stack((np.ones_like(f1), f1, fg))
    x_coefficients, *_ = np.linalg.lstsq(design, x_m, rcond=None)
    y_coefficients, *_ = np.linalg.lstsq(design, y_m, rcond=None)
    z_coefficients, *_ = np.linalg.lstsq(design, z_per_f, rcond=None)

    gx, gy, gz_per_f = (
        float(x_coefficients[2]),
        float(y_coefficients[2]),
        float(z_coefficients[2]),
    )
    remaining = GRAVITY_MPS2**2 - gx**2 - gy**2
    if remaining > 0 and abs(gz_per_f) > 1e-9:
        focal_center = math.sqrt(remaining) / abs(gz_per_f)
    else:
        focal_center = float(max(image_width_px, image_height_px))

    solutions: list[_ProfileSolution] = []
    for multiplier in FOCAL_PROFILE_MULTIPLIERS:
        focal_px = focal_center * multiplier
        gravity = np.array([gx, gy, focal_px * gz_per_f], dtype=float)
        gravity_error = abs(float(np.linalg.norm(gravity)) - GRAVITY_MPS2)
        if gravity_error > GRAVITY_PROFILE_TOLERANCE_MPS2:
            continue
        velocity_start = np.array(
            [
                float(x_coefficients[1]),
                float(y_coefficients[1]),
                focal_px * float(z_coefficients[1]),
            ]
        )
        velocity_release = _back_propagate_velocity(
            velocity_start,
            gravity,
            lambda_s_inv,
            times[0] - release_time_s,
        )
        speed = float(np.linalg.norm(velocity_release))
        if not math.isfinite(speed) or speed <= 0:
            continue
        up = -gravity / np.linalg.norm(gravity)
        vertical_speed = float(np.dot(velocity_release, up))
        elevation = math.degrees(
            math.asin(max(-1.0, min(1.0, vertical_speed / speed)))
        )
        residual_x = x_m - design @ x_coefficients
        residual_y = y_m - design @ y_coefficients
        residual_z = focal_px * (z_per_f - design @ z_coefficients)
        residual_rms = float(
            math.sqrt(np.mean(residual_x**2 + residual_y**2 + residual_z**2))
        )
        solutions.append(
            _ProfileSolution(speed, elevation, residual_rms, gravity_error)
        )
    return solutions


def _percentile_interval(
    values: Iterable[float], allowance: float
) -> tuple[float, float] | None:
    array = np.array(list(values), dtype=float)
    if len(array) == 0:
        return None
    low, high = np.percentile(array, [2.5, 97.5])
    middle = float(np.median(array))
    half_width = max(middle - low, high - middle)
    total_half_width = math.hypot(float(half_width), allowance)
    return middle - total_half_width, middle + total_half_width


def estimate_near_side_metric_flight(
    track: BallTrackEvidence, release_frame: int | None
) -> MetricFlightResult:
    observations = unique_observations(track)
    trusted = select_trusted_observations(observations, release_frame)
    timestamp_source = (
        "fps_fallback"
        if any("timestamp_fallback_fps" in warning for warning in track.warnings)
        else "pts"
    )
    if not trusted:
        return MetricFlightResult(
            available=False,
            qualified=False,
            qualification="UNAVAILABLE",
            observation_count=len(observations),
            reasons=[
                "Insufficient continuous unique ball observations for provisional temporal support."
            ],
            quality_flags=["insufficient_temporal_support"],
            timestamp_source=timestamp_source,
        )
    if not track.image_width_px or not track.image_height_px:
        return MetricFlightResult(
            available=False,
            qualified=False,
            qualification="UNAVAILABLE",
            observation_count=len(trusted),
            reasons=["Image dimensions are required for the camera-space metric bridge."],
            quality_flags=["image_dimensions_unavailable"],
            timestamp_source=timestamp_source,
        )

    release_frame_time = next(
        (
            frame.time_s
            for frame in track.frames
            if frame.frame_index == release_frame and frame.time_s is not None
        ),
        trusted[0].time_s,
    )
    lower_time = release_frame_time if release_frame is not None else trusted[0].time_s
    lower_frame = release_frame if release_frame is not None else trusted[0].frame_index
    epoch = ReleaseEpochInterval(
        lower_time_s=lower_time,
        upper_time_s=trusted[0].time_s,
        representative_time_s=(lower_time + trusted[0].time_s) / 2,
        lower_frame=lower_frame,
        upper_frame=trusted[0].frame_index,
        reason=(
            "pose release is last contact-plausible evidence; trusted-track start "
            "is first persistent free-flight evidence"
        ),
    )

    profile_samples: list[_ProfileSolution] = []
    for start_offset in range(3):
        subset = trusted[start_offset:]
        if len(subset) < PROVISIONAL_MINIMUM_USABLE_OBSERVATIONS:
            continue
        if (
            subset[-1].time_s - subset[0].time_s
            < PROVISIONAL_MINIMUM_SUPPORT_SECONDS
        ):
            continue
        for proxy, degree in (("min", 1), ("geometric", 2)):
            for lambda_s_inv in LAMBDA_GRID_S_INV:
                for epoch_time in (
                    epoch.lower_time_s,
                    epoch.representative_time_s,
                    epoch.upper_time_s,
                ):
                    profile_samples.extend(
                        _fit_solutions(
                            subset,
                            proxy,
                            degree,
                            lambda_s_inv,
                            epoch_time,
                            track.image_width_px,
                            track.image_height_px,
                        )
                    )
        for perturbation in (
            (CENTER_LOCALIZATION_PX, 0.0),
            (-CENTER_LOCALIZATION_PX, 0.0),
            (0.0, CENTER_LOCALIZATION_PX),
            (0.0, -CENTER_LOCALIZATION_PX),
        ):
            profile_samples.extend(
                _fit_solutions(
                    subset,
                    "min",
                    1,
                    0.0,
                    epoch.representative_time_s,
                    track.image_width_px,
                    track.image_height_px,
                    perturbation,
                )
            )

    equivalent_samples: list[_ProfileSolution] = []
    if profile_samples:
        best_rms = min(item.residual_rms_m for item in profile_samples)
        equivalent_samples = [
            item
            for item in profile_samples
            if item.residual_rms_m <= best_rms + PROFILE_EQUIVALENCE_RMS_M
        ]
    speed_interval = _percentile_interval(
        (item.speed_mps for item in equivalent_samples),
        math.hypot(
            NEAR_SIDE_OUT_OF_PLANE_SPEED_ALLOWANCE_MPS,
            AERODYNAMIC_MODEL_MISMATCH_SPEED_ALLOWANCE_MPS,
        ),
    )
    elevation_interval = _percentile_interval(
        (item.elevation_deg for item in equivalent_samples),
        math.hypot(
            NEAR_SIDE_OUT_OF_PLANE_ELEVATION_ALLOWANCE_DEG,
            AERODYNAMIC_MODEL_MISMATCH_ELEVATION_ALLOWANCE_DEG,
        ),
    )
    if speed_interval is not None:
        speed_interval = (max(0.0, speed_interval[0]), speed_interval[1])
    if elevation_interval is not None:
        elevation_interval = (
            max(-90.0, elevation_interval[0]),
            min(90.0, elevation_interval[1]),
        )
    if speed_interval is None or elevation_interval is None:
        return MetricFlightResult(
            available=True,
            qualified=False,
            qualification="UNQUALIFIED",
            trusted_start_frame=trusted[0].frame_index,
            trusted_end_frame=trusted[-1].frame_index,
            trusted_start_time_s=trusted[0].time_s,
            trusted_end_time_s=trusted[-1].time_s,
            observation_count=len(trusted),
            temporal_span_s=trusted[-1].time_s - trusted[0].time_s,
            release_epoch=epoch,
            reasons=[
                "Camera-space profile solutions are unstable; metric values are withheld."
            ],
            quality_flags=["metric_profile_unsolved", "metric_output_unqualified"],
            timestamp_source=timestamp_source,
        )

    speed = sum(speed_interval) / 2
    elevation = sum(elevation_interval) / 2
    speed_half_width = (speed_interval[1] - speed_interval[0]) / 2
    elevation_half_width = (elevation_interval[1] - elevation_interval[0]) / 2
    qualification = (
        "HIGH"
        if speed_half_width <= 0.35 and elevation_half_width <= 2.0
        else "MEDIUM"
        if speed_half_width <= 0.60 and elevation_half_width <= 4.0
        else "UNQUALIFIED"
    )
    qualified = qualification in {"HIGH", "MEDIUM"}
    quality_flags = [
        "apparent_size_camera_metric",
        "basketball_diameter_prior",
        "image_center_principal_point",
        "focal_length_profiled_as_nuisance",
        "size_proxy_not_calibrated_diameter_measurement",
        "effective_linear_drag_profiled_as_nuisance",
        "release_epoch_interval_propagated",
    ]
    if not qualified:
        quality_flags.append("metric_output_unqualified")
    return MetricFlightResult(
        available=True,
        qualified=qualified,
        qualification=qualification,
        trusted_start_frame=trusted[0].frame_index,
        trusted_end_frame=trusted[-1].frame_index,
        trusted_start_time_s=trusted[0].time_s,
        trusted_end_time_s=trusted[-1].time_s,
        observation_count=len(trusted),
        temporal_span_s=trusted[-1].time_s - trusted[0].time_s,
        release_epoch=epoch,
        speed_mps=speed if qualified else None,
        elevation_angle_deg=elevation if qualified else None,
        speed_interval_mps=speed_interval,
        elevation_interval_deg=elevation_interval,
        fit_rms_m=min(item.residual_rms_m for item in equivalent_samples),
        reasons=[
            "Camera-space flight estimated from apparent ball size and unique observations.",
            "Metric values are withheld when V2 engineering availability gates are not met.",
        ],
        quality_flags=quality_flags,
        uncertainty_sources=[
            "center_localization",
            "size_proxy_and_time_domain_smoothing",
            "focal_and_effective_drag_profile",
            "primary_vs_sensitivity_estimator",
            "near_side_out_of_plane_ambiguity",
            "release_epoch_interval",
            "timestamps_and_missing_observations",
            "aerodynamic_model_mismatch_allowance",
        ],
        timestamp_source=timestamp_source,
    )
