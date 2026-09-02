from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmarks.reference_v1.validation_closure import decode_contact_transition_v1

from .schema import EVENT_LABELS, METRIC_LABELS, PHASE_LABELS, event, metric, phase


LEFT = {"shoulder": 5, "elbow": 7, "wrist": 9, "hip": 11, "knee": 13, "ankle": 15}
RIGHT = {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "knee": 14, "ankle": 16}


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-6:
        return math.nan
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = values.astype(float).copy()
    valid = np.isfinite(result)
    if np.sum(valid) < 3:
        return result
    indices = np.arange(len(result))
    result[~valid] = np.interp(indices[~valid], indices[valid], result[valid])
    window = min(window, len(result) if len(result) % 2 else len(result) - 1)
    window = max(1, window)
    if window == 1:
        return result
    padding = window // 2
    return np.convolve(np.pad(result, (padding, padding), mode="edge"), np.ones(window) / window, mode="valid")


def _normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return result
    low, high = np.percentile(values[valid], [10, 90])
    result[valid] = 0.5 if high - low <= 1e-6 else np.clip((values[valid] - low) / (high - low), 0, 1)
    return 1 - result if invert else result


def _mean_available(first: float, second: float) -> float:
    values = [value for value in (first, second) if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def build_signals(
    rows: list[dict[str, Any]],
    width: int,
    height: int,
    pose_key: str = "pose",
) -> tuple[list[dict[str, Any]], str]:
    signals: list[dict[str, Any]] = []
    wrist_peaks = {"left": [], "right": []}
    for row in rows:
        signal = {
            "frame_index": row["frame_index"],
            "time_seconds": row["time_seconds"],
            "pose_found": False,
            "torso_scale": math.nan,
            "hip_y": math.nan,
            "hip_y_px": math.nan,
            "ankle_y": math.nan,
            "left_elbow": math.nan,
            "right_elbow": math.nan,
            "left_knee": math.nan,
            "right_knee": math.nan,
            "left_wrist_height": math.nan,
            "right_wrist_height": math.nan,
            "left_release_height": math.nan,
            "right_release_height": math.nan,
            "ball_center": None,
            "ball_bbox": None,
            "ball_confidence": None,
            "ball_diameter": None,
            "left_ball_distance": math.nan,
            "right_ball_distance": math.nan,
        }
        pose_data = row.get(pose_key)
        points = None
        confidence = None
        if pose_data:
            points = np.asarray(pose_data["keypoints"], dtype=float)
            confidence = np.asarray(pose_data.get("temporal_reliability", pose_data["confidence"]), dtype=float)
            visible = confidence >= 0.25
            if len(points) >= 17 and all(visible[index] for index in (5, 6, 11, 12)):
                shoulder_mid = (points[5] + points[6]) / 2
                hip_mid = (points[11] + points[12]) / 2
                torso_scale = float(np.linalg.norm(shoulder_mid - hip_mid))
                if torso_scale > 1:
                    signal["pose_found"] = True
                    signal["torso_scale"] = torso_scale
                    signal["hip_y"] = float(hip_mid[1] / height)
                    signal["hip_y_px"] = float(hip_mid[1])
                    ankles = [points[index][1] for index in (15, 16) if visible[index]]
                    if ankles:
                        signal["ankle_y"] = float(np.mean(ankles) / height)
                    for side, mapping in (("left", LEFT), ("right", RIGHT)):
                        if all(visible[mapping[name]] for name in ("shoulder", "elbow", "wrist")):
                            elbow = joint_angle(
                                points[mapping["shoulder"]],
                                points[mapping["elbow"]],
                                points[mapping["wrist"]],
                            )
                            wrist_height = float(
                                (points[mapping["shoulder"]][1] - points[mapping["wrist"]][1])
                                / torso_scale
                            )
                            signal[f"{side}_elbow"] = elbow
                            signal[f"{side}_wrist_height"] = wrist_height
                            wrist_peaks[side].append(wrist_height)
                            if ankles:
                                bbox = pose_data["bbox"]
                                body_height = max(float(bbox[3]) - float(bbox[1]), 1.0)
                                signal[f"{side}_release_height"] = float(
                                    (np.mean(ankles) - points[mapping["wrist"]][1]) / body_height
                                )
                        if all(visible[mapping[name]] for name in ("hip", "knee", "ankle")):
                            signal[f"{side}_knee"] = joint_angle(
                                points[mapping["hip"]],
                                points[mapping["knee"]],
                                points[mapping["ankle"]],
                            )

        ball = row.get("ball")
        if ball:
            center = np.asarray(ball["center"], dtype=float)
            signal["ball_center"] = ball["center"]
            signal["ball_bbox"] = ball["bbox"]
            signal["ball_confidence"] = ball["confidence"]
            signal["ball_diameter"] = ball["diameter"]
            if points is not None and confidence is not None and len(points) >= 17:
                for side, mapping in (("left", LEFT), ("right", RIGHT)):
                    if confidence[mapping["wrist"]] >= 0.25:
                        signal[f"{side}_ball_distance"] = float(
                            np.linalg.norm(center - points[mapping["wrist"]])
                            / max(float(ball["diameter"]), 1.0)
                        )
        signals.append(signal)

    peaks = {
        side: float(np.percentile(values, 90)) if values else -math.inf
        for side, values in wrist_peaks.items()
    }
    shooting_side = "left" if peaks["left"] > peaks["right"] else "right"
    return signals, shooting_side


def _unavailable_events(fps: float, reason: str) -> dict[str, dict[str, Any]]:
    return {
        name: event(
            name,
            "insufficient_data",
            fps=fps,
            provenance=["yolo11_pose", "reference_v1_temporal_heuristic"],
            risk_flags=["insufficient_pose_signal"],
            reason=reason,
        )
        for name in EVENT_LABELS
    }


def detect_events(
    rows: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    shooting_side: str,
    metadata: dict[str, Any],
    *,
    preprocessed_pose: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    fps = float(metadata["fps"])
    count = len(signals)
    pose_coverage = float(np.mean([signal["pose_found"] for signal in signals])) if signals else 0.0
    if count < 10 or pose_coverage < 0.4:
        return _unavailable_events(fps, "Too little usable body-pose evidence"), {
            "pose_coverage": pose_coverage,
            "signal_coverage": 0.0,
            "risk_flags": ["insufficient_pose_signal"],
        }

    elbow_raw = np.asarray([signal[f"{shooting_side}_elbow"] for signal in signals], dtype=float)
    wrist_raw = np.asarray([signal[f"{shooting_side}_wrist_height"] for signal in signals], dtype=float)
    knee_raw = np.asarray(
        [_mean_available(signal["left_knee"], signal["right_knee"]) for signal in signals],
        dtype=float,
    )
    hip_raw = np.asarray([signal["hip_y"] for signal in signals], dtype=float)
    ankle_raw = np.asarray([signal["ankle_y"] for signal in signals], dtype=float)
    signal_coverage = float(np.mean(np.isfinite(elbow_raw) & np.isfinite(wrist_raw) & np.isfinite(knee_raw) & np.isfinite(hip_raw)))
    if signal_coverage < 0.4:
        return _unavailable_events(fps, "Required temporal pose signals are incomplete"), {
            "pose_coverage": pose_coverage,
            "signal_coverage": signal_coverage,
            "risk_flags": ["insufficient_pose_signal"],
        }

    elbow = _smooth(elbow_raw)
    wrist = _smooth(wrist_raw)
    if preprocessed_pose:
        knee, hip, ankle = knee_raw.copy(), hip_raw.copy(), ankle_raw.copy()
    else:
        knee, hip, ankle = _smooth(knee_raw), _smooth(hip_raw), _smooth(ankle_raw)
    wrist_velocity = np.gradient(wrist) * fps
    release_score = 0.40 * _normalize(wrist) + 0.30 * _normalize(elbow) + 0.30 * _normalize(wrist_velocity)
    start = max(0, round(count * 0.15))
    stop = max(start + 1, min(count, round(count * 0.9)))
    release_index = start + int(np.argmax(release_score[start:stop]))
    pose_release_frame = int(signals[release_index]["frame_index"])
    release_confidence = float(np.clip(0.25 + 0.5 * release_score[release_index] + 0.25 * signal_coverage, 0, 1))

    minimum_gap = max(1, round(fps * 0.08))
    bottom_stop = max(1, release_index - minimum_gap)
    dip_score = 0.65 * _normalize(knee, invert=True) + 0.35 * _normalize(hip)
    bottom_start = max(0, round(count * 0.05))
    bottom_index = bottom_start + int(np.argmax(dip_score[bottom_start:bottom_stop])) if bottom_stop > bottom_start else max(0, release_index - minimum_gap)
    bottom_frame = int(signals[bottom_index]["frame_index"])

    baseline_start = max(0, bottom_index - round(fps * 1.0))
    baseline_knee = float(np.nanpercentile(knee[baseline_start : bottom_index + 1], 80))
    bottom_knee = float(knee[bottom_index])
    bend_range = baseline_knee - bottom_knee
    dip_start_index: int | None = None
    if math.isfinite(bend_range) and bend_range >= 5.0:
        threshold = baseline_knee - bend_range * 0.20
        candidates = [index for index in range(baseline_start, bottom_index + 1) if knee[index] <= threshold]
        dip_start_index = candidates[0] if candidates else None

    provenance = ["yolo11_pose", "reference_v1_temporal_heuristic"]
    events = {
        "dip_start": event(
            "dip_start",
            "ok" if dip_start_index is not None else "not_detected",
            frame=int(signals[dip_start_index]["frame_index"]) if dip_start_index is not None else None,
            fps=fps,
            confidence=min(0.8, signal_coverage) if dip_start_index is not None else None,
            provenance=provenance,
            reason=None if dip_start_index is not None else "No sustained knee-flexion onset was supported",
        ),
        "bottom": event(
            "bottom",
            "ok" if math.isfinite(bottom_knee) else "insufficient_data",
            frame=bottom_frame if math.isfinite(bottom_knee) else None,
            fps=fps,
            confidence=float(np.clip(0.25 + 0.5 * dip_score[bottom_index] + 0.25 * signal_coverage, 0, 1)),
            provenance=provenance,
        ),
        "pose_release": event(
            "pose_release",
            "ok" if release_confidence >= 0.5 else "low_confidence",
            frame=pose_release_frame,
            fps=fps,
            confidence=release_confidence,
            provenance=provenance,
            risk_flags=[] if release_confidence >= 0.5 else ["low_pose_release_candidate"],
        ),
    }

    ankle_coverage = float(np.mean(np.isfinite(ankle_raw)))
    takeoff_index: int | None = None
    apex_index: int | None = None
    landing_index: int | None = None
    jump_height = 0.0
    if ankle_coverage >= 0.55 and dip_start_index is not None:
        baseline_slice = ankle[max(0, dip_start_index - round(fps * 0.5)) : bottom_index + 1]
        baseline = float(np.nanmedian(baseline_slice))
        search_stop = min(count, release_index + round(fps * 0.5))
        lifted = [index for index in range(bottom_index, search_stop) if baseline - ankle[index] >= 0.015]
        if lifted:
            takeoff_index = lifted[0]
            apex_stop = min(count, release_index + round(fps * 0.8))
            apex_index = bottom_index + int(np.nanargmin(hip[bottom_index:apex_stop]))
            jump_height = baseline - float(np.nanmin(ankle[bottom_index:apex_stop]))
            tolerance = max(0.008, jump_height * 0.30)
            returned = [
                index
                for index in range(max(apex_index + 1, release_index + 1), count)
                if abs(float(ankle[index]) - baseline) <= tolerance
            ]
            landing_index = returned[0] if returned else None

    if takeoff_index is None:
        takeoff_status = "insufficient_data" if ankle_coverage < 0.55 else "not_applicable"
        takeoff_reason = "Ankle evidence is incomplete" if ankle_coverage < 0.55 else "No supported foot-lift event in this clip"
    else:
        takeoff_status = "ok"
        takeoff_reason = None
    events["takeoff"] = event(
        "takeoff",
        takeoff_status,
        frame=int(signals[takeoff_index]["frame_index"]) if takeoff_index is not None else None,
        fps=fps,
        confidence=min(0.8, ankle_coverage) if takeoff_index is not None else None,
        provenance=provenance,
        risk_flags=[] if takeoff_index is not None else ["takeoff_unavailable"],
        reason=takeoff_reason,
    )

    if apex_index is None:
        apex_status = "not_applicable" if takeoff_status == "not_applicable" else "insufficient_data"
    else:
        apex_status = "ok"
    events["body_apex"] = event(
        "body_apex",
        apex_status,
        frame=int(signals[apex_index]["frame_index"]) if apex_index is not None else None,
        fps=fps,
        confidence=min(0.8, signal_coverage) if apex_index is not None else None,
        provenance=provenance,
        reason=None if apex_index is not None else "No stable airborne body-apex evidence",
    )
    landing_status = "ok" if landing_index is not None else "not_applicable" if takeoff_status == "not_applicable" else "not_detected"
    events["landing"] = event(
        "landing",
        landing_status,
        frame=int(signals[landing_index]["frame_index"]) if landing_index is not None else None,
        fps=fps,
        confidence=min(0.75, ankle_coverage) if landing_index is not None else None,
        provenance=provenance,
        risk_flags=[] if landing_index is not None else ["landing_unavailable"],
        reason=None if landing_index is not None else "Landing was not supported before clip end",
    )

    distance_key = f"{shooting_side}_ball_distance"
    strict_evidence = [
        {
            "frame_index": signal["frame_index"],
            "ball_wrist_distance_diameters": (
                float(signal[distance_key]) if math.isfinite(signal[distance_key]) else None
            ),
            "ball_center": signal["ball_center"],
        }
        for signal in signals
    ]
    strict_result = decode_contact_transition_v1(strict_evidence, pose_release_frame)
    strict_frame = strict_result["predicted_strict_frame"]
    strict_points = sum(
        pose_release_frame - 10 <= signal["frame_index"] <= pose_release_frame + 12
        and signal["ball_center"] is not None
        for signal in signals
    )
    strict_confidence = min(0.85, 0.45 + strict_points / 30) if strict_frame is not None else None
    events["strict_ball_release"] = event(
        "strict_ball_release",
        "ok" if strict_frame is not None else "insufficient_data",
        frame=int(strict_frame) if strict_frame is not None else None,
        fps=fps,
        confidence=strict_confidence,
        provenance=["release_ball_v1", "yolo11_pose_wrist", "contact_transition_decoder_v1"],
        risk_flags=strict_result["risk_flags"],
        reason=None if strict_frame is not None else "Persistent supported hand-ball separation was unavailable",
    )

    ambiguity_ratio = float(np.mean([bool(row.get("ambiguous_shooter")) for row in rows]))
    risk_flags = []
    if ambiguity_ratio >= 0.20:
        risk_flags.append("ambiguous_shooter")
    if signal_coverage < 0.65:
        risk_flags.append("low_signal_coverage")
    if fps < 24:
        risk_flags.append("low_temporal_resolution")
    if strict_frame is None:
        risk_flags.append("strict_release_unavailable")
    diagnostics = {
        "pose_coverage": round(pose_coverage, 4),
        "signal_coverage": round(signal_coverage, 4),
        "ankle_coverage": round(ankle_coverage, 4),
        "ambiguity_ratio": round(ambiguity_ratio, 4),
        "jump_height_image_ratio": round(jump_height, 5),
        "strict_result": strict_result,
        "signals": {
            "elbow": elbow,
            "wrist": wrist,
            "knee": knee,
            "hip": hip,
            "ankle": ankle,
        },
        "indices": {
            "dip_start": dip_start_index,
            "bottom": bottom_index,
            "takeoff": takeoff_index,
            "pose_release": release_index,
            "strict_ball_release": next((i for i, item in enumerate(signals) if item["frame_index"] == strict_frame), None),
            "body_apex": apex_index,
            "landing": landing_index,
        },
        "risk_flags": risk_flags,
    }
    return events, diagnostics


def build_phases(
    events: dict[str, dict[str, Any]],
    first_frame: int,
    last_frame: int,
    fps: float,
) -> dict[str, dict[str, Any]]:
    provenance = ["reference_v1_event_boundaries"]
    dip_start = events["dip_start"]["frame"]
    bottom = events["bottom"]["frame"]
    pose_release = events["pose_release"]["frame"]
    strict_release = events["strict_ball_release"]["frame"]
    release_anchor = strict_release if strict_release is not None else pose_release
    landing = events["landing"]["frame"]

    phases = {
        "preparation": phase(
            "preparation",
            "ok" if dip_start is not None else "insufficient_data",
            start_frame=first_frame if dip_start is not None else None,
            end_frame=dip_start if dip_start is not None else None,
            fps=fps,
            confidence=events["dip_start"]["confidence"],
            provenance=provenance,
            reason=None if dip_start is not None else "Dip start unavailable",
        ),
        "dip": phase(
            "dip",
            "ok" if dip_start is not None and bottom is not None and dip_start <= bottom else "insufficient_data",
            start_frame=dip_start if dip_start is not None and bottom is not None and dip_start <= bottom else None,
            end_frame=bottom if dip_start is not None and bottom is not None and dip_start <= bottom else None,
            fps=fps,
            confidence=_minimum_confidence(events["dip_start"], events["bottom"]),
            provenance=provenance,
        ),
        "upward_drive": phase(
            "upward_drive",
            "ok" if bottom is not None and release_anchor is not None and bottom <= release_anchor else "insufficient_data",
            start_frame=bottom if bottom is not None and release_anchor is not None and bottom <= release_anchor else None,
            end_frame=release_anchor if bottom is not None and release_anchor is not None and bottom <= release_anchor else None,
            fps=fps,
            confidence=_minimum_confidence(events["bottom"], events["pose_release"]),
            provenance=provenance,
        ),
        "follow_through": phase(
            "follow_through",
            "ok" if release_anchor is not None else "insufficient_data",
            start_frame=release_anchor,
            end_frame=landing if landing is not None and landing >= release_anchor else last_frame if release_anchor is not None else None,
            fps=fps,
            confidence=events["strict_ball_release"]["confidence"] or events["pose_release"]["confidence"],
            provenance=provenance,
            risk_flags=["landing_boundary_unavailable"] if release_anchor is not None and landing is None else [],
        ),
        "landing_recovery": phase(
            "landing_recovery",
            "ok" if landing is not None else events["landing"]["status"],
            start_frame=landing,
            end_frame=last_frame if landing is not None else None,
            fps=fps,
            confidence=events["landing"]["confidence"],
            provenance=provenance,
            reason=events["landing"]["reason"],
        ),
    }
    return phases


def calculate_metrics(
    events: dict[str, dict[str, Any]],
    signals: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    shooting_side: str,
    fps: float,
) -> dict[str, dict[str, Any]]:
    by_frame = {signal["frame_index"]: signal for signal in signals}
    pose_frame = events["pose_release"]["frame"]
    strict_frame = events["strict_ball_release"]["frame"]
    bottom_frame = events["bottom"]["frame"]
    dip_start_frame = events["dip_start"]["frame"]
    apex_frame = events["body_apex"]["frame"]
    takeoff_frame = events["takeoff"]["frame"]
    landing_frame = events["landing"]["frame"]
    pose_signal = by_frame.get(pose_frame) if pose_frame is not None else None

    metrics: dict[str, dict[str, Any]] = {}
    metrics["strict_release_frame"] = metric(
        "strict_release_frame",
        events["strict_ball_release"]["status"],
        value=strict_frame,
        unit="frame_index",
        confidence=events["strict_ball_release"]["confidence"],
        frame=strict_frame,
        source_events=["strict_ball_release"],
        provenance=events["strict_ball_release"]["provenance"],
        risk_flags=events["strict_ball_release"]["risk_flags"],
        evidence_refs=events["strict_ball_release"]["evidence_refs"],
        reason=events["strict_ball_release"]["reason"],
    )

    metrics["pose_to_strict_release_delta"] = _frame_delta_metric(
        "pose_to_strict_release_delta", pose_frame, strict_frame, fps, ["pose_release", "strict_ball_release"]
    )

    elbow_value = pose_signal.get(f"{shooting_side}_elbow") if pose_signal else math.nan
    metrics["release_elbow_angle"] = metric(
        "release_elbow_angle",
        "ok" if elbow_value is not None and math.isfinite(elbow_value) else "insufficient_data",
        value=round(float(elbow_value), 1) if elbow_value is not None and math.isfinite(elbow_value) else None,
        unit="degrees_2d",
        confidence=events["pose_release"]["confidence"],
        frame=pose_frame,
        source_events=["pose_release"],
        required_joints=[f"{shooting_side}_shoulder", f"{shooting_side}_elbow", f"{shooting_side}_wrist"],
        provenance=["yolo11_pose", "image_space_geometry"],
        view_requirement="side_or_diagonal_preferred",
        risk_flags=["2d_projection_only"],
        evidence_refs=["evidence/pose_release.jpg"] if pose_frame is not None else [],
        reason=None if elbow_value is not None and math.isfinite(elbow_value) else "Required arm joints unavailable",
    )

    height_value = pose_signal.get(f"{shooting_side}_release_height") if pose_signal else math.nan
    metrics["normalized_release_height"] = metric(
        "normalized_release_height",
        "ok" if height_value is not None and math.isfinite(height_value) else "insufficient_data",
        value=round(float(height_value), 3) if height_value is not None and math.isfinite(height_value) else None,
        unit="body_height_ratio_image_space",
        confidence=events["pose_release"]["confidence"],
        frame=pose_frame,
        source_events=["pose_release"],
        required_joints=[f"{shooting_side}_wrist", "ankle"],
        provenance=["yolo11_pose", "normalized_image_space"],
        view_requirement="full_body_visible",
        risk_flags=["not_real_world_height"],
        evidence_refs=["evidence/pose_release.jpg"] if pose_frame is not None else [],
        reason=None if height_value is not None and math.isfinite(height_value) else "Wrist or ankle evidence unavailable",
    )

    bottom_signal = by_frame.get(bottom_frame) if bottom_frame is not None else None
    dip_signal = by_frame.get(dip_start_frame) if dip_start_frame is not None else None
    dip_depth = math.nan
    if bottom_signal and dip_signal:
        torso = np.nanmean([bottom_signal["torso_scale"], dip_signal["torso_scale"]])
        if math.isfinite(torso) and torso > 1 and math.isfinite(bottom_signal["hip_y_px"]) and math.isfinite(dip_signal["hip_y_px"]):
            dip_depth = (bottom_signal["hip_y_px"] - dip_signal["hip_y_px"]) / torso
    metrics["dip_depth"] = metric(
        "dip_depth",
        "ok" if math.isfinite(dip_depth) else "insufficient_data",
        value=round(float(dip_depth), 3) if math.isfinite(dip_depth) else None,
        unit="torso_scale_normalized",
        confidence=_minimum_confidence(events["dip_start"], events["bottom"]),
        frame_range=[dip_start_frame, bottom_frame] if dip_start_frame is not None and bottom_frame is not None else None,
        source_events=["dip_start", "bottom"],
        required_joints=["shoulders", "hips"],
        provenance=["yolo11_pose", "normalized_image_space"],
        view_requirement="stable_camera_preferred",
        risk_flags=["camera_motion_sensitive"],
        evidence_refs=["evidence/dip_start.jpg", "evidence/bottom.jpg"] if math.isfinite(dip_depth) else [],
        reason=None if math.isfinite(dip_depth) else "Dip boundaries or hip scale unavailable",
    )

    knee_value = (
        _mean_available(bottom_signal["left_knee"], bottom_signal["right_knee"])
        if bottom_signal
        else math.nan
    )
    metrics["minimum_knee_angle"] = metric(
        "minimum_knee_angle",
        "ok" if math.isfinite(knee_value) else "insufficient_data",
        value=round(knee_value, 1) if math.isfinite(knee_value) else None,
        unit="degrees_2d",
        confidence=events["bottom"]["confidence"],
        frame=bottom_frame,
        source_events=["bottom"],
        required_joints=["hip", "knee", "ankle"],
        provenance=["yolo11_pose", "image_space_geometry"],
        view_requirement="lower_body_visible",
        risk_flags=["2d_projection_only"],
        evidence_refs=["evidence/bottom.jpg"] if bottom_frame is not None else [],
        reason=None if math.isfinite(knee_value) else "Knee geometry unavailable at bottom",
    )

    if strict_frame is not None and apex_frame is not None:
        apex_delta = strict_frame - apex_frame
        timing = "before_apex" if apex_delta < -1 else "after_apex" if apex_delta > 1 else "near_apex"
        apex_value = {"frames": apex_delta, "milliseconds": round(apex_delta / fps * 1000, 1), "timing": timing}
        apex_status = "ok"
    else:
        apex_value = None
        apex_status = "insufficient_data" if events["body_apex"]["status"] != "not_applicable" else "not_applicable"
    metrics["release_relative_to_body_apex"] = metric(
        "release_relative_to_body_apex",
        apex_status,
        value=apex_value,
        unit="frames_and_milliseconds",
        confidence=_minimum_confidence(events["strict_ball_release"], events["body_apex"]),
        frame_range=[strict_frame, apex_frame] if strict_frame is not None and apex_frame is not None else None,
        source_events=["strict_ball_release", "body_apex"],
        required_joints=["hips"],
        provenance=["contact_transition_decoder_v1", "yolo11_pose"],
        view_requirement="stable_camera_and_jump_visible",
        risk_flags=["camera_motion_sensitive"],
        evidence_refs=["evidence/strict_ball_release.jpg", "evidence/body_apex.jpg"] if apex_value else [],
        reason=None if apex_value else "Strict release or body apex unavailable",
    )

    elbow = diagnostics["signals"]["elbow"]
    release_index = diagnostics["indices"]["strict_ball_release"]
    if release_index is None:
        release_index = diagnostics["indices"]["pose_release"]
    onset_index = _extension_onset(elbow, release_index, fps)
    onset_frame = signals[onset_index]["frame_index"] if onset_index is not None else None
    metrics["elbow_extension_onset_relative_to_release"] = _frame_delta_metric(
        "elbow_extension_onset_relative_to_release",
        onset_frame,
        strict_frame if strict_frame is not None else pose_frame,
        fps,
        ["elbow_extension_onset", "strict_ball_release" if strict_frame is not None else "pose_release"],
        risk_flags=["pose_release_fallback"] if strict_frame is None else [],
        signed_relative=True,
    )

    metrics["takeoff_to_strict_release"] = _frame_delta_metric(
        "takeoff_to_strict_release",
        takeoff_frame,
        strict_frame,
        fps,
        ["takeoff", "strict_ball_release"],
        unavailable_status="not_applicable" if events["takeoff"]["status"] == "not_applicable" else "insufficient_data",
    )
    metrics["follow_through_duration"] = _frame_delta_metric(
        "follow_through_duration",
        strict_frame if strict_frame is not None else pose_frame,
        landing_frame,
        fps,
        ["strict_ball_release" if strict_frame is not None else "pose_release", "landing"],
        unavailable_status="not_applicable" if events["landing"]["status"] == "not_applicable" else "insufficient_data",
        risk_flags=["pose_release_fallback"] if strict_frame is None else [],
    )
    if set(metrics) != set(METRIC_LABELS):
        raise RuntimeError("Metric contract was not fully populated")
    return metrics


def build_ball_evidence(
    signals: list[dict[str, Any]],
    shooting_side: str,
    events: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    strict_frame = events["strict_ball_release"]["frame"]
    pose_frame = events["pose_release"]["frame"]
    distance_key = f"{shooting_side}_ball_distance"
    observations = []
    missing_gaps = []
    gap_start = None
    prior_center = None
    contact_seen = False
    for signal in signals:
        frame = signal["frame_index"]
        center = signal["ball_center"]
        in_window = pose_frame is not None and pose_frame - 12 <= frame <= pose_frame + 16
        if not in_window:
            continue
        distance = signal[distance_key]
        if center is None:
            gap_start = frame if gap_start is None else gap_start
            continue
        if gap_start is not None:
            missing_gaps.append({"start_frame": gap_start, "end_frame": frame - 1, "length": frame - gap_start})
            gap_start = None
        movement = math.dist(center, prior_center) if prior_center is not None else None
        if math.isfinite(distance) and distance <= 1.25:
            state = "contact_supported"
            contact_seen = True
        elif strict_frame is not None and frame >= strict_frame:
            state = "released_confirmed"
        elif contact_seen and math.isfinite(distance) and distance >= 1.4:
            state = "separating"
        elif math.isfinite(distance):
            state = "possession_candidate"
        else:
            state = "unknown"
        observations.append(
            {
                "frame": frame,
                "center": [round(float(value), 2) for value in center],
                "bbox": [round(float(value), 2) for value in signal["ball_bbox"]],
                "confidence": round(float(signal["ball_confidence"]), 4),
                "visible": True,
                "detector_source": "release_ball_v1",
                "ball_hand_distance_diameters": round(float(distance), 4) if math.isfinite(distance) else None,
                "relative_movement_px": round(movement, 3) if movement is not None else None,
                "contact_state": state,
                "persistence_supported": strict_frame is not None and frame >= strict_frame,
            }
        )
        prior_center = center
    if gap_start is not None and pose_frame is not None:
        missing_gaps.append({"start_frame": gap_start, "end_frame": pose_frame + 16, "length": pose_frame + 17 - gap_start})
    return {
        "status": events["strict_ball_release"]["status"],
        "detector": "release_ball_v1",
        "detector_role": "prototype_observation_baseline",
        "tracker_used": False,
        "tracker_source": None,
        "reanchor_count": 0,
        "tracker_risk": ["detector_only_reference_v1"],
        "shooting_side": shooting_side,
        "window": [pose_frame - 12, pose_frame + 16] if pose_frame is not None else None,
        "center_observations": observations,
        "trajectory_points": [item["center"] for item in observations],
        "missing_gaps": missing_gaps,
        "separation_evidence": diagnostics["strict_result"],
        "risk_flags": events["strict_ball_release"]["risk_flags"],
        "provenance": ["release_ball_v1", "yolo11_pose_wrist", "contact_transition_decoder_v1"],
        "evidence_ref": "evidence/ball_motion.json",
    }


def build_observations_and_suggestions(metrics: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations = []
    suggestions = []
    elbow = metrics["release_elbow_angle"]
    if elbow["status"] == "ok":
        observations.append({"metric": elbow["name"], "text": f"出手时投篮侧肘角约 {elbow['value']}°，这是二维画面测量。", "evidence_refs": elbow["evidence_refs"]})
    timing = metrics["release_relative_to_body_apex"]
    if timing["status"] == "ok":
        label = {"before_apex": "身体最高点前", "near_apex": "身体最高点附近", "after_apex": "身体最高点后"}[timing["value"]["timing"]]
        observations.append({"metric": timing["name"], "text": f"严格出手发生在估计{label} {abs(timing['value']['frames'])} 帧。", "evidence_refs": timing["evidence_refs"]})
        if timing["value"]["timing"] == "after_apex" and timing["value"]["frames"] > 2:
            suggestions.append(
                {
                    "source_metric": timing["name"],
                    "text": "可在相同距离和机位下，对比稍早出手时动作是否更稳定。",
                    "strength": "conservative_observation_prompt",
                    "evidence_refs": timing["evidence_refs"],
                }
            )
    dip = metrics["dip_depth"]
    if dip["status"] == "ok":
        observations.append({"metric": dip["name"], "text": f"本次下沉幅度约为 {dip['value']} 个躯干尺度。", "evidence_refs": dip["evidence_refs"]})
    if metrics["strict_release_frame"]["status"] != "ok":
        observations.append(
            {
                "metric": "strict_release_frame",
                "text": "球证据不足，系统没有把 Pose Release 冒充为 Strict Ball Release。",
                "evidence_refs": [],
            }
        )
    return observations, suggestions


def analyze(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    pose_key: str = "pose",
) -> dict[str, Any]:
    signals, shooting_side = build_signals(rows, int(metadata["width"]), int(metadata["height"]), pose_key)
    events, diagnostics = detect_events(
        rows,
        signals,
        shooting_side,
        metadata,
        preprocessed_pose=pose_key == "analysis_pose",
    )
    phases = build_phases(events, rows[0]["frame_index"], rows[-1]["frame_index"], float(metadata["fps"]))
    metrics = calculate_metrics(events, signals, diagnostics, shooting_side, float(metadata["fps"]))
    ball_evidence = build_ball_evidence(signals, shooting_side, events, diagnostics)
    observations, suggestions = build_observations_and_suggestions(metrics)
    return {
        "shooting_side": shooting_side,
        "signals": signals,
        "events": events,
        "phases": phases,
        "metrics": metrics,
        "ball_evidence": ball_evidence,
        "observations": observations,
        "suggestions": suggestions,
        "diagnostics": diagnostics,
        "risks": diagnostics["risk_flags"],
    }


def _frame_delta_metric(
    name: str,
    start_frame: int | None,
    end_frame: int | None,
    fps: float,
    source_events: list[str],
    *,
    unavailable_status: str = "insufficient_data",
    risk_flags: list[str] | None = None,
    signed_relative: bool = False,
) -> dict[str, Any]:
    available = start_frame is not None and end_frame is not None
    delta = (start_frame - end_frame if signed_relative else end_frame - start_frame) if available else None
    return metric(
        name,
        "ok" if available else unavailable_status,
        value={"frames": delta, "milliseconds": round(delta / fps * 1000, 1)} if available else None,
        unit="frames_and_milliseconds",
        confidence=0.65 if available else None,
        frame_range=[start_frame, end_frame] if available else None,
        source_events=source_events,
        provenance=["reference_v1_event_timing"],
        risk_flags=risk_flags or [],
        evidence_refs=[f"evidence/{event_name}.jpg" for event_name in source_events if event_name in EVENT_LABELS] if available else [],
        reason=None if available else "Required event evidence unavailable",
    )


def _extension_onset(elbow: np.ndarray, release_index: int | None, fps: float) -> int | None:
    if release_index is None or release_index < 3:
        return None
    start = max(0, release_index - round(fps * 1.0))
    for index in range(start, max(start, release_index - 2)):
        if all(math.isfinite(float(value)) for value in elbow[index : index + 4]):
            if elbow[index + 3] - elbow[index] >= 5.0 and elbow[index + 2] >= elbow[index]:
                return index
    return None


def _minimum_confidence(*items: dict[str, Any]) -> float | None:
    values = [float(item["confidence"]) for item in items if item.get("confidence") is not None]
    return min(values) if values else None
