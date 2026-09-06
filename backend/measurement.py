from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


class MeasurementStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNRELIABLE = "UNRELIABLE"


@dataclass(frozen=True, slots=True)
class MeasurementQuality:
    status: MeasurementStatus
    confidence: float | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MeasurementEvidence:
    release_sources: list[str] = field(default_factory=list)
    frame_agreement: dict[str, Any] | None = None
    risk_flags: list[str] = field(default_factory=list)
    existing_tracking_evidence: dict[str, Any] | None = None
    fps: float | None = None
    actual_timestamps: list[float] | None = None
    trusted_window_point_count: int | None = None
    trusted_window_temporal_span_ms: float | None = None
    missing_observations: int | None = None
    tracking_regime_flags: list[str] | None = None
    fit_rms_cm: float | None = None
    ensemble_velocity_disagreement_mps: float | None = None
    leave_one_out_velocity_disagreement_mps: float | None = None
    holdout_prediction_error_cm: float | None = None
    release_epoch_status: str | None = None
    release_epoch_uncertainty_ms: float | None = None
    timestamp_source: str | None = None
    metric_output_qualification: str | None = None
    metric_uncertainty_sources: list[str] | None = None


@dataclass(frozen=True, slots=True)
class BallTrackDetection:
    bbox: list[float]
    center_x_px: float
    center_y_px: float
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class BallTrackFrameEvidence:
    frame_index: int
    time_s: float | None
    detections: list[BallTrackDetection] = field(default_factory=list)
    bbox: list[float] | None = None
    center_x_px: float | None = None
    center_y_px: float | None = None
    confidence: float | None = None
    source: str | None = None
    timestamp_source: str | None = None
    status: str = "no_detection"


@dataclass(frozen=True, slots=True)
class BallTrackEvidence:
    status: str
    start_frame: int | None = None
    end_frame: int | None = None
    fps: float | None = None
    image_width_px: int | None = None
    image_height_px: int | None = None
    requested_frame_count: int = 0
    observed_frame_count: int = 0
    detection_frame_count: int = 0
    missing_frame_count: int = 0
    frames: list[BallTrackFrameEvidence] = field(default_factory=list)
    actual_timestamps: list[float] = field(default_factory=list)
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BallObservationCandidate:
    bbox: list[float]
    center_x_px: float
    center_y_px: float
    confidence: float
    source: str
    class_name: str = "ball"


@dataclass(frozen=True, slots=True)
class DenseBallFrameObservation:
    frame_index: int
    timestamp: float | None
    timestamp_source: str | None
    image_data_url: str | None
    general_ball_candidates: list[BallObservationCandidate] = field(default_factory=list)
    general_ball_status: str = "MISSING"
    release_ball_candidates: list[BallObservationCandidate] = field(default_factory=list)
    release_ball_status: str = "MISSING"
    selected_ball_observation: BallObservationCandidate | None = None
    confidence: float | None = None
    bbox: list[float] | None = None
    center_x_px: float | None = None
    center_y_px: float | None = None
    human_ball_state: dict[str, Any] | None = None
    pose: dict[str, Any] | None = None
    track_membership: bool = False
    trusted_flight_membership: bool = False
    release_epoch_membership: bool = False


@dataclass(frozen=True, slots=True)
class DenseBallObservationEvidence:
    status: str
    total_frame_count: int
    scanned_frame_count: int
    fps: float | None = None
    scan_stride: int = 1
    general_ball_detected_frame_count: int = 0
    general_multiple_frame_count: int = 0
    release_ball_detected_frame_count: int = 0
    release_multiple_frame_count: int = 0
    unique_selected_observation_count: int = 0
    missing_frame_count: int = 0
    frames: list[DenseBallFrameObservation] = field(default_factory=list)
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrustedFlightSegment:
    status: MeasurementStatus
    start_frame: int | None = None
    end_frame: int | None = None
    point_count: int | None = None
    temporal_span_ms: float | None = None
    fps: float | None = None
    reason: str | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReleaseStateUncertainty:
    position_std_cm: tuple[float, float, float] | None = None
    velocity_std_mps: tuple[float, float, float] | None = None
    state_anchor_time_std_ms: float | None = None
    speed_interval_mps: tuple[float, float] | None = None
    elevation_angle_interval_deg: tuple[float, float] | None = None
    release_epoch_interval_ms: tuple[float, float] | None = None
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReleaseEpochInterval:
    lower_time_s: float | None = None
    upper_time_s: float | None = None
    representative_time_s: float | None = None
    lower_frame: int | None = None
    upper_frame: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReleaseStateEstimate:
    status: MeasurementStatus
    state_anchor_frame: int | None = None
    state_anchor_time: float | None = None
    position_m: tuple[float | None, float | None, float | None] | None = None
    velocity_mps: tuple[float | None, float | None, float | None] | None = None
    speed_mps: float | None = None
    elevation_angle_deg: float | None = None
    azimuth_angle_deg: float | None = None
    uncertainty: ReleaseStateUncertainty | None = None
    release_epoch: ReleaseEpochInterval | None = None
    release_time: float | None = None
    qualification: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReleaseMeasurementResult:
    """Stable product contract around release measurements.

    Scientific estimators must leave fields as ``None`` until their output is
    validated for production use.
    """

    status: MeasurementStatus
    release_frame: int | None = None
    release_time: float | None = None
    source: str | None = None
    confidence: float | None = None
    agreement_level: str | None = None
    frame_delta: int | None = None
    reason: str | None = None
    risk_flags: list[str] = field(default_factory=list)
    evidence: MeasurementEvidence = field(default_factory=MeasurementEvidence)
    measurement_quality: MeasurementQuality | None = None
    trusted_flight: TrustedFlightSegment | None = None
    release_state: ReleaseStateEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        if self.measurement_quality:
            result["measurement_quality"]["status"] = self.measurement_quality.status.value
        if self.trusted_flight:
            result["trusted_flight"]["status"] = self.trusted_flight.status.value
        if self.release_state:
            result["release_state"]["status"] = self.release_state.status.value
        return result


def release_fusion_to_measurement(
    release_fusion: dict[str, Any],
    ball_track_evidence: BallTrackEvidence | None = None,
) -> ReleaseMeasurementResult:
    status = {
        "ok": MeasurementStatus.AVAILABLE,
        "insufficient_data": MeasurementStatus.INSUFFICIENT_DATA,
    }.get(release_fusion.get("status"), MeasurementStatus.UNRELIABLE)
    risk_flags = list(release_fusion.get("risk_flags") or [])

    result = ReleaseMeasurementResult(
        status=status,
        release_frame=release_fusion.get("pose_release_frame_index"),
        source=release_fusion.get("final_source"),
        agreement_level=release_fusion.get("agreement_level"),
        frame_delta=release_fusion.get("frame_delta"),
        reason=release_fusion.get("reason"),
        risk_flags=risk_flags,
        evidence=MeasurementEvidence(
            release_sources=[release_fusion["final_source"]]
            if release_fusion.get("final_source")
            else [],
            frame_agreement={
                "pose_release_frame_index": release_fusion.get(
                    "pose_release_frame_index"
                ),
                "detector_release_frame_index": release_fusion.get(
                    "detector_release_frame_index"
                ),
                "frame_delta": release_fusion.get("frame_delta"),
                "agreement_level": release_fusion.get("agreement_level"),
            },
            risk_flags=risk_flags,
            existing_tracking_evidence=(
                {
                    "status": ball_track_evidence.status,
                    "start_frame": ball_track_evidence.start_frame,
                    "end_frame": ball_track_evidence.end_frame,
                    "requested_frame_count": ball_track_evidence.requested_frame_count,
                    "observed_frame_count": ball_track_evidence.observed_frame_count,
                    "detection_frame_count": ball_track_evidence.detection_frame_count,
                    "missing_frame_count": ball_track_evidence.missing_frame_count,
                }
                if ball_track_evidence
                else None
            ),
            fps=ball_track_evidence.fps if ball_track_evidence else None,
            actual_timestamps=(
                ball_track_evidence.actual_timestamps if ball_track_evidence else None
            ),
            missing_observations=(
                ball_track_evidence.missing_frame_count if ball_track_evidence else None
            ),
        ),
        measurement_quality=MeasurementQuality(
            status=status,
            issues=risk_flags
            if status is not MeasurementStatus.AVAILABLE
            else [],
            warnings=risk_flags
            if status is MeasurementStatus.AVAILABLE
            else [],
        ),
    )
    if not ball_track_evidence or not ball_track_evidence.frames:
        return result

    # Imported here to keep the stable contract independent from the estimator.
    from backend.near_side_metric_flight import estimate_near_side_metric_flight

    metric = estimate_near_side_metric_flight(
        ball_track_evidence, result.release_frame
    )
    if not metric.available:
        return result
    epoch = metric.release_epoch
    release_epoch = ReleaseEpochInterval(
        lower_time_s=epoch.lower_time_s if epoch else None,
        upper_time_s=epoch.upper_time_s if epoch else None,
        representative_time_s=epoch.representative_time_s if epoch else None,
        lower_frame=epoch.lower_frame if epoch else None,
        upper_frame=epoch.upper_frame if epoch else None,
        reason=epoch.reason if epoch else None,
    )
    trusted_flight = TrustedFlightSegment(
        status=MeasurementStatus.AVAILABLE,
        start_frame=metric.trusted_start_frame,
        end_frame=metric.trusted_end_frame,
        point_count=metric.observation_count,
        temporal_span_ms=(
            metric.temporal_span_s * 1000 if metric.temporal_span_s is not None else None
        ),
        fps=ball_track_evidence.fps,
        reason=metric.reasons[0] if metric.reasons else None,
        quality_flags=metric.quality_flags,
    )
    release_state = ReleaseStateEstimate(
        status=(
            MeasurementStatus.AVAILABLE
            if metric.qualified
            else MeasurementStatus.UNRELIABLE
        ),
        state_anchor_frame=metric.trusted_start_frame,
        state_anchor_time=metric.trusted_start_time_s,
        # Absolute position and out-of-plane velocity remain unqualified.
        velocity_mps=None,
        speed_mps=metric.speed_mps,
        elevation_angle_deg=metric.elevation_angle_deg,
        uncertainty=ReleaseStateUncertainty(
            speed_interval_mps=metric.speed_interval_mps,
            elevation_angle_interval_deg=metric.elevation_interval_deg,
            release_epoch_interval_ms=(
                (
                    epoch.lower_time_s * 1000,
                    epoch.upper_time_s * 1000,
                )
                if epoch
                else None
            ),
            sources=metric.uncertainty_sources,
        ),
        release_epoch=release_epoch,
        release_time=(
            epoch.representative_time_s if metric.qualified and epoch else None
        ),
        qualification=metric.qualification,
        reason="; ".join(metric.reasons),
    )
    evidence = replace(
        result.evidence,
        trusted_window_point_count=metric.observation_count,
        trusted_window_temporal_span_ms=(
            metric.temporal_span_s * 1000 if metric.temporal_span_s is not None else None
        ),
        release_epoch_status="INTERVAL",
        release_epoch_uncertainty_ms=(
            (epoch.upper_time_s - epoch.lower_time_s) * 500 if epoch else None
        ),
        timestamp_source=metric.timestamp_source,
        metric_output_qualification=metric.qualification,
        metric_uncertainty_sources=metric.uncertainty_sources,
        fit_rms_cm=(
            metric.fit_rms_m * 100 if metric.fit_rms_m is not None else None
        ),
    )
    quality = replace(
        result.measurement_quality,
        warnings=list(result.measurement_quality.warnings) + metric.quality_flags,
    )
    return replace(
        result,
        release_time=release_state.release_time,
        evidence=evidence,
        measurement_quality=quality,
        trusted_flight=trusted_flight,
        release_state=release_state,
    )


def pose_release_to_measurement(
    release_frame: int | None,
) -> ReleaseMeasurementResult:
    if release_frame is None:
        return ReleaseMeasurementResult(
            status=MeasurementStatus.INSUFFICIENT_DATA,
            reason="Release frame is unavailable from pose analysis.",
            measurement_quality=MeasurementQuality(
                status=MeasurementStatus.INSUFFICIENT_DATA,
                issues=["pose_release_frame_unavailable"],
            ),
        )

    issue = "release_ball_corroboration_unavailable"
    return ReleaseMeasurementResult(
        status=MeasurementStatus.UNRELIABLE,
        release_frame=release_frame,
        source="pose_release",
        reason=(
            "Release frame is available from pose analysis, but release-ball "
            "corroboration is unavailable."
        ),
        risk_flags=[issue],
        evidence=MeasurementEvidence(
            release_sources=["pose_release"],
            risk_flags=[issue],
        ),
        measurement_quality=MeasurementQuality(
            status=MeasurementStatus.UNRELIABLE,
            issues=[issue],
        ),
    )
