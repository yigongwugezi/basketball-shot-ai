from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    release_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        if self.measurement_quality:
            result["measurement_quality"]["status"] = self.measurement_quality.status.value
        return result


def release_fusion_to_measurement(
    release_fusion: dict[str, Any],
) -> ReleaseMeasurementResult:
    status = {
        "ok": MeasurementStatus.AVAILABLE,
        "insufficient_data": MeasurementStatus.INSUFFICIENT_DATA,
    }.get(release_fusion.get("status"), MeasurementStatus.UNRELIABLE)
    risk_flags = list(release_fusion.get("risk_flags") or [])

    return ReleaseMeasurementResult(
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
