from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MeasurementStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNRELIABLE = "UNRELIABLE"


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
    reason: str | None = None
    risk_flags: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    measurement_quality: dict[str, Any] | None = None
    release_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result
