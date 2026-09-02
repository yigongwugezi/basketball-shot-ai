from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def pose_candidates(result: Any) -> list[dict[str, Any]]:
    """Normalize every YOLO person pose candidate in original-frame pixels."""
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return []
    boxes = np.asarray(result.boxes.xyxy.cpu().tolist(), dtype=float)
    keypoints = np.asarray(result.keypoints.xy.cpu().tolist(), dtype=float)
    confidence = (
        np.asarray(result.keypoints.conf.cpu().tolist(), dtype=float)
        if result.keypoints.conf is not None
        else np.ones(keypoints.shape[:2], dtype=float)
    )
    return [
        {
            "bbox": box.tolist(),
            "keypoints": points.tolist(),
            "confidence": scores.tolist(),
            "visible_keypoints": int(np.sum(scores >= 0.25)),
            "person_count": int(len(boxes)),
        }
        for box, points, scores in zip(boxes, keypoints, confidence)
    ]


@dataclass
class ShooterSelection:
    candidate: dict[str, Any] | None
    track_id: int | None
    confidence: float
    identity_break: bool
    crop_status: str
    ambiguous: bool


class ShooterContinuitySelector:
    """Small deterministic selector for the trimmed, single-shooter V1 contract."""

    def __init__(self, max_gap: int = 3) -> None:
        self.max_gap = max_gap
        self.previous: dict[str, Any] | None = None
        self.missing = 0
        self.track_id = 0
        self.had_track = False

    def select(self, candidates: list[dict[str, Any]]) -> ShooterSelection:
        if not candidates:
            self.missing += 1
            if self.missing > self.max_gap:
                self.previous = None
            return ShooterSelection(
                None,
                self.track_id if self.had_track else None,
                0.0,
                False,
                "missing_person",
                False,
            )

        if self.previous is None:
            candidate, ambiguous = self._largest(candidates)
            identity_break = self.had_track
            if not self.had_track or identity_break:
                self.track_id += 1
            self.had_track = True
            self.previous = candidate
            self.missing = 0
            return ShooterSelection(
                candidate,
                self.track_id,
                self._initial_confidence(candidate, candidates),
                identity_break,
                "reacquired" if identity_break else "ok",
                ambiguous,
            )

        ranked = sorted(
            ((self._continuity_score(candidate), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        score, candidate = ranked[0]
        ambiguous = len(ranked) > 1 and score - ranked[1][0] < 0.08
        if score < 0.20:
            self.missing += 1
            if self.missing <= self.max_gap:
                return ShooterSelection(
                    None,
                    self.track_id,
                    round(score, 4),
                    False,
                    "unsupported_continuity_gap",
                    ambiguous,
                )
            candidate, area_ambiguous = self._largest(candidates)
            self.track_id += 1
            self.previous = candidate
            self.missing = 0
            return ShooterSelection(
                candidate,
                self.track_id,
                self._initial_confidence(candidate, candidates),
                True,
                "reacquired",
                ambiguous or area_ambiguous,
            )

        self.previous = candidate
        self.missing = 0
        return ShooterSelection(
            candidate,
            self.track_id,
            round(score, 4),
            False,
            "ambiguous" if ambiguous else "ok",
            ambiguous,
        )

    @staticmethod
    def _largest(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        ordered = sorted(candidates, key=lambda item: _area(item["bbox"]), reverse=True)
        largest = _area(ordered[0]["bbox"])
        ratio = _area(ordered[1]["bbox"]) / largest if len(ordered) > 1 and largest > 0 else 0.0
        return ordered[0], ratio >= 0.65

    @staticmethod
    def _initial_confidence(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> float:
        visible = candidate["visible_keypoints"] / 17
        largest = max(_area(item["bbox"]) for item in candidates)
        dominance = _area(candidate["bbox"]) / largest if largest > 0 else 0.0
        return round(min(1.0, 0.55 * visible + 0.45 * dominance), 4)

    def _continuity_score(self, candidate: dict[str, Any]) -> float:
        assert self.previous is not None
        previous_box = self.previous["bbox"]
        box = candidate["bbox"]
        scale = max(math.sqrt(_area(previous_box)), 1.0)
        center_distance = math.dist(_center(previous_box), _center(box)) / scale
        area_ratio = max(_area(box), 1.0) / max(_area(previous_box), 1.0)
        scale_similarity = math.exp(-abs(math.log(area_ratio)))
        pose_distance = math.dist(_pose_center(self.previous), _pose_center(candidate)) / scale
        visible = candidate["visible_keypoints"] / 17
        return float(
            0.45 * _iou(previous_box, box)
            + 0.25 * math.exp(-2.0 * center_distance)
            + 0.15 * scale_similarity
            + 0.10 * math.exp(-2.0 * pose_distance)
            + 0.05 * visible
        )


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center(box: list[float]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)


def _pose_center(candidate: dict[str, Any]) -> tuple[float, float]:
    points = np.asarray(candidate["keypoints"], dtype=float)
    confidence = np.asarray(candidate["confidence"], dtype=float)
    torso = [index for index in (5, 6, 11, 12) if confidence[index] >= 0.25]
    if torso:
        center = np.mean(points[torso], axis=0)
        return float(center[0]), float(center[1])
    return _center(candidate["bbox"])


def _iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0
