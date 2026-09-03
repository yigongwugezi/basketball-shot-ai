from __future__ import annotations

import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", r"E:\BasketballShotAI\config\ultralytics")
os.environ.setdefault("XDG_CACHE_HOME", r"E:\BasketballShotAI\public_data\cache")
os.environ.setdefault("TORCH_HOME", r"E:\BasketballShotAI\public_data\cache\rtmlib")
os.environ.setdefault("HF_HOME", r"E:\BasketballShotAI\public_data\cache\huggingface")

import cv2
import numpy as np
from ultralytics import YOLO

from benchmarks.reference_v1.model_adapters import RtmlibPoseAdapter

from . import SCHEMA_VERSION
from .analysis import analyze
from .perception import ShooterContinuitySelector, pose_candidates
from .pose.metrics import evaluate_pose_rows, no_lag_metrics
from .pose.reliability import build_analysis_pose
from .render import render_annotated_video, write_report_html
from .schema import validate_report


ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL_PATH = ROOT / "yolo11n-pose.pt"
BALL_MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "release_ball"
    / "yolo11n_release_ball_batch001_003_v1"
    / "weights"
    / "best.pt"
)
RTMPOSE_MODEL_CONFIG = "rtmpose-m_simcc-body7_256x192-e48f03d0"


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    shot_type: str | None = None,
    pose_view: str = "raw",
    pose_backbone: str = "rtmpose",
) -> dict[str, Any]:
    started = time.perf_counter()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if not POSE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose model does not exist: {POSE_MODEL_PATH}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_video_metadata(input_path)
    if pose_backbone not in {"yolo", "rtmpose"}:
        raise ValueError(f"Unknown pose backbone: {pose_backbone}")

    model_started = time.perf_counter()
    person_model = YOLO(str(POSE_MODEL_PATH))
    pose_model = RtmlibPoseAdapter("rtmpose") if pose_backbone == "rtmpose" else None
    ball_model = YOLO(str(BALL_MODEL_PATH)) if BALL_MODEL_PATH.is_file() else None
    model_load_seconds = time.perf_counter() - model_started

    inference_started = time.perf_counter()
    rows, inference_runtime = infer_video(
        input_path,
        person_model,
        pose_model,
        ball_model,
        metadata,
        pose_backbone=pose_backbone,
    )
    inference_seconds = time.perf_counter() - inference_started
    if not rows:
        raise RuntimeError("No video frame could be analyzed")
    metadata["declared_frame_count"] = metadata["frame_count"]
    metadata["frame_count"] = len(rows)
    metadata["duration_seconds"] = len(rows) / metadata["fps"]

    analysis_started = time.perf_counter()
    pose_source = "rtmpose-m_body7_256x192" if pose_backbone == "rtmpose" else "yolo11_pose"
    raw_analysis = analyze(rows, metadata, pose_key="raw_pose", pose_source=pose_source)
    rows = build_analysis_pose(rows, smooth=pose_backbone == "yolo")
    analysis = analyze(rows, metadata, pose_key="analysis_pose", pose_source=pose_source)
    human_ball_by_frame = {
        item["frame"]: item for item in analysis["human_ball_release"]["contact_state_sequence"]
    }
    for row in rows:
        for key in ("raw_pose", "analysis_pose"):
            if row.get(key):
                row[key]["shooting_side"] = analysis["shooting_side"]
        row["human_ball"] = human_ball_by_frame.get(row["frame_index"])
    pose_reliability = build_pose_reliability(rows, raw_analysis, analysis, pose_backbone)
    analysis_seconds = time.perf_counter() - analysis_started

    quality = build_quality(metadata, analysis, ball_model is not None)
    analysis_status = determine_analysis_status(quality, analysis)
    risks = collect_risks(quality, analysis)
    attempt_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    report = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": str(input_path),
            "name": input_path.name,
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "declared_frame_count": metadata["declared_frame_count"],
            "duration_seconds": metadata["duration_seconds"],
            "width": metadata["width"],
            "height": metadata["height"],
        },
        "quality": quality,
        "attempt": {
            "attempt_id": attempt_id,
            "source_video_id": input_path.stem,
            "shot_type": shot_type,
            "view": {
                "value": "unknown",
                "status": "needs_review",
                "preferred": ["side", "diagonal"],
                "reason": "Reference V1 does not auto-classify camera view",
            },
            "condition": None,
            "outcome": "unknown",
            "shooting_side": analysis["shooting_side"],
            "analysis_status": analysis_status,
        },
        "phases": analysis["phases"],
        "events": analysis["events"],
        "ball_evidence": analysis["ball_evidence"],
        "human_ball_release": analysis["human_ball_release"],
        "metrics": analysis["metrics"],
        "observations": analysis["observations"],
        "suggestions": analysis["suggestions"],
        "risks": risks,
        "perception": {
            "pose_backbone": pose_backbone,
            "pose_provider": "rtmlib" if pose_backbone == "rtmpose" else "ultralytics",
            "pose_model": "RTMPose-m Body7 256x192" if pose_backbone == "rtmpose" else "YOLO11n-pose",
            "pose_model_config": RTMPOSE_MODEL_CONFIG if pose_backbone == "rtmpose" else "yolo11n-pose.pt",
            "person_box_source": (
                "yolo11n_pose_temporal_shooter_continuity"
                if pose_backbone == "rtmpose"
                else "yolo11n_pose_largest_person_per_frame"
            ),
            "coordinate_space": "original_video_frame_pixels",
            "localization_evidence": "raw_pose",
            "derived_temporal_signal": "analysis_pose",
            "annotated_pose_view": pose_view,
            "global_coordinate_smoothing": pose_backbone == "yolo",
        },
        "pose_reliability": pose_reliability,
        "runtime": {
            "model_load_seconds": round(model_load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "analysis_seconds": round(analysis_seconds, 3),
            "render_seconds": None,
            "total_seconds": None,
            "pose_inference_ms": round(inference_runtime["pose_ms"], 2),
            "person_detector_ms": round(inference_runtime["person_ms"], 2),
            "pose_head_ms": round(inference_runtime["pose_head_ms"], 2),
            "ball_inference_ms": round(inference_runtime["ball_ms"], 2),
            "device": "cpu",
        },
        "artifacts": {
            "annotated_video": "annotated.mp4",
            "report_html": "report.html",
            "report_json": "report.json",
            "timeline_csv": "timeline.csv",
            "ball_evidence": "evidence/ball_motion.json",
            "human_ball_release": "evidence/human_ball_release_v1.json",
            "frame_evidence": "evidence/frame_evidence.json",
            "pose_reliability": "evidence/pose_reliability.json",
            "pose_trajectories": "evidence/pose_trajectories.json",
            "evidence_images": [],
        },
    }
    validate_report(report)

    write_timeline(output_dir / "timeline.csv", report)
    _write_json(evidence_dir / "ball_motion.json", analysis["ball_evidence"])
    _write_json(evidence_dir / "human_ball_release_v1.json", analysis["human_ball_release"])
    _write_json(evidence_dir / "frame_evidence.json", build_frame_evidence(rows, report))
    _write_json(evidence_dir / "pose_reliability.json", pose_reliability)
    _write_json(evidence_dir / "pose_trajectories.json", build_pose_trajectories(rows))

    render_started = time.perf_counter()
    render_result = render_annotated_video(
        input_path,
        output_dir / "annotated.mp4",
        evidence_dir,
        rows,
        analysis["phases"],
        analysis["events"],
        analysis["metrics"],
        pose_key="raw_pose" if pose_view == "raw" else "analysis_pose",
    )
    render_seconds = time.perf_counter() - render_started
    report["artifacts"]["evidence_images"] = render_result["evidence_images"]
    report["artifacts"]["render"] = render_result
    report["runtime"]["render_seconds"] = round(render_seconds, 3)
    report["runtime"]["total_seconds"] = round(time.perf_counter() - started, 3)
    validate_report(report)
    _write_json(output_dir / "report.json", report)
    write_report_html(report, output_dir / "report.html")
    return report


def read_video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("Video metadata is incomplete")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def infer_video(
    path: Path,
    person_model: YOLO,
    pose_model: RtmlibPoseAdapter | None,
    ball_model: YOLO | None,
    metadata: dict[str, Any],
    *,
    pose_backbone: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    capture = cv2.VideoCapture(str(path))
    rows = []
    frame_index = 0
    pose_runtime = 0.0
    person_runtime = 0.0
    pose_head_runtime = 0.0
    ball_runtime = 0.0
    selector = ShooterContinuitySelector() if pose_backbone == "rtmpose" else None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        person_started = time.perf_counter()
        pose_result = person_model.predict(frame, imgsz=640, conf=0.20, verbose=False, device="cpu")[0]
        person_ms = (time.perf_counter() - person_started) * 1000
        person_runtime += person_ms
        head_ms = 0.0
        ambiguous_shooter = False

        tracking: dict[str, Any]
        if pose_backbone == "yolo":
            pose = _select_pose(pose_result)
            ambiguous_shooter = bool(pose and pose.get("ambiguity_ratio", 0.0) >= 0.65)
            tracking = {
                "shooter_track_id": None,
                "shooter_selection_confidence": None,
                "identity_break": False,
                "crop_status": "ok" if pose else "missing_person",
                "selection": "largest_person_single_shot_scope",
            }
            if pose:
                pose.update(_pose_provenance("yolo", person_ms, tracking))
        else:
            assert selector is not None and pose_model is not None
            selection = selector.select(pose_candidates(pose_result))
            ambiguous_shooter = selection.ambiguous
            tracking = {
                "shooter_track_id": selection.track_id,
                "shooter_selection_confidence": selection.confidence,
                "identity_break": selection.identity_break,
                "crop_status": selection.crop_status,
                "selection": "temporal_shooter_continuity",
            }
            pose = None
            if selection.candidate is not None:
                pose, head_ms = pose_model.infer(frame, selection.candidate["bbox"])
                pose_head_runtime += head_ms
                if pose is not None:
                    pose["pose_output_bbox"] = pose["bbox"]
                    pose["bbox"] = selection.candidate["bbox"]
                    pose["person_count"] = selection.candidate["person_count"]
                    pose.update(_pose_provenance("rtmpose", head_ms, tracking))
                else:
                    tracking["crop_status"] = "pose_head_missing"
        pose_runtime += person_ms + head_ms

        ball = None
        ball_candidates: list[dict[str, Any]] = []
        if ball_model is not None:
            ball_started = time.perf_counter()
            ball_result = ball_model.predict(frame, imgsz=640, conf=0.10, verbose=False, device="cpu")[0]
            ball_runtime += (time.perf_counter() - ball_started) * 1000
            ball_candidates = _ball_candidates(ball_result)
            ball = max(ball_candidates, key=lambda item: item["confidence"], default=None)
        rows.append(
            {
                "frame_index": frame_index,
                "time_seconds": frame_index / metadata["fps"],
                "pose": pose,
                "raw_pose": pose,
                "ball": ball,
                "ball_candidates": ball_candidates,
                "tracking": tracking,
                "ambiguous_shooter": ambiguous_shooter,
            }
        )
        frame_index += 1
    capture.release()
    return rows, {
        "pose_ms": pose_runtime,
        "person_ms": person_runtime,
        "pose_head_ms": pose_head_runtime,
        "ball_ms": ball_runtime,
    }


def _pose_provenance(
    pose_backbone: str,
    runtime_ms: float,
    tracking: dict[str, Any],
) -> dict[str, Any]:
    if pose_backbone == "rtmpose":
        values = {
            "pose_provider": "rtmlib",
            "pose_model": "RTMPose-m Body7 256x192",
            "pose_model_config": RTMPOSE_MODEL_CONFIG,
            "person_box_source": "yolo11n_pose_temporal_shooter_continuity",
            "provenance": ["rtmpose-m_body7_256x192_raw"],
        }
    else:
        values = {
            "pose_provider": "ultralytics",
            "pose_model": "YOLO11n-pose",
            "pose_model_config": "yolo11n-pose.pt",
            "person_box_source": "yolo11n_pose_largest_person_per_frame",
            "provenance": ["yolo11_pose_raw"],
        }
    return {
        **values,
        "pose_runtime_ms": round(runtime_ms, 3),
        "coordinate_space": "original_video_frame_pixels",
        "raw_derived_status": "raw_model_output",
        "shooter_track_id": tracking["shooter_track_id"],
        "shooter_selection_confidence": tracking["shooter_selection_confidence"],
        "identity_break": tracking["identity_break"],
        "crop_status": tracking["crop_status"],
        "selection": tracking["selection"],
    }


def _select_pose(result: Any) -> dict[str, Any] | None:
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return None
    boxes = np.asarray(result.boxes.xyxy.cpu().tolist(), dtype=float)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(areas))
    ordered = np.sort(areas)
    ambiguity_ratio = float(ordered[-2] / ordered[-1]) if len(ordered) > 1 and ordered[-1] > 0 else 0.0
    keypoints = np.asarray(result.keypoints.xy[index].cpu(), dtype=float)
    confidence = (
        np.asarray(result.keypoints.conf[index].cpu(), dtype=float)
        if result.keypoints.conf is not None
        else np.ones(len(keypoints), dtype=float)
    )
    return {
        "bbox": boxes[index].tolist(),
        "keypoints": keypoints.tolist(),
        "confidence": confidence.tolist(),
        "visible_keypoints": int(np.sum(confidence >= 0.25)),
        "person_count": int(len(boxes)),
        "ambiguity_ratio": round(ambiguity_ratio, 4),
        "selection": "largest_person_single_shot_scope",
    }


def _select_ball(result: Any) -> dict[str, Any] | None:
    return max(_ball_candidates(result), key=lambda item: item["confidence"], default=None)


def _ball_candidates(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or not len(result.boxes):
        return []
    candidates = [box for box in result.boxes if int(box.cls[0]) == 0]
    output = []
    for box in candidates:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        output.append(
            {
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "diameter": max(x2 - x1, y2 - y1, 1.0),
                "confidence": float(box.conf[0]),
                "source": "release_ball_v1",
            }
        )
    return output


def build_quality(metadata: dict[str, Any], analysis: dict[str, Any], ball_available: bool) -> dict[str, Any]:
    duration = float(metadata["duration_seconds"])
    decoded_frames = int(metadata["frame_count"])
    declared_frames = int(metadata["declared_frame_count"])
    checks = [
        {
            "name": "trimmed_clip_duration",
            "status": "ok" if 1.0 <= duration <= 12.0 else "needs_review",
            "detail": f"{duration:.2f}s; Reference V1 expects a trimmed single-shot clip.",
        },
        {
            "name": "temporal_resolution",
            "status": "ok" if metadata["fps"] >= 24 else "low_confidence",
            "detail": f"{metadata['fps']:.2f} FPS; frame-accurate release benefits from higher FPS.",
        },
        {
            "name": "decoded_frame_count",
            "status": "ok" if decoded_frames == declared_frames else "needs_review",
            "detail": f"Decoded {decoded_frames} frames; container declared {declared_frames}.",
        },
        {
            "name": "pose_coverage",
            "status": "ok" if analysis["diagnostics"]["pose_coverage"] >= 0.8 else "insufficient_data",
            "detail": f"Pose coverage {analysis['diagnostics']['pose_coverage']:.1%}.",
        },
        {
            "name": "shooter_ambiguity",
            "status": "ambiguous" if analysis["diagnostics"]["ambiguity_ratio"] >= 0.20 else "ok",
            "detail": f"Similar-size person ambiguity proxy {analysis['diagnostics']['ambiguity_ratio']:.1%}.",
        },
        {
            "name": "ball_detector",
            "status": "ok" if ball_available else "insufficient_data",
            "detail": "release-ball v1 loaded" if ball_available else "release-ball v1 missing; strict release unavailable",
        },
        {
            "name": "camera_view",
            "status": "needs_review",
            "detail": "Camera view is not automatically classified; side/diagonal is preferred.",
        },
    ]
    return {
        "status": "ok" if all(check["status"] == "ok" for check in checks[:6]) else "needs_review",
        "checks": checks,
        "measurement_boundary": "2d_image_space_not_calibrated_biomechanics",
    }


def determine_analysis_status(quality: dict[str, Any], analysis: dict[str, Any]) -> str:
    statuses = {check["name"]: check["status"] for check in quality["checks"]}
    if statuses["pose_coverage"] == "insufficient_data" or statuses["shooter_ambiguity"] == "ambiguous":
        return "insufficient_data"
    if analysis["events"]["strict_ball_release"]["status"] != "ok":
        return "needs_review"
    return "ok"


def collect_risks(quality: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    risks = list(analysis["risks"])
    risks.extend(analysis["human_ball_release"].get("uncertainty", []))
    for check in quality["checks"]:
        if check["status"] != "ok":
            risks.append(f"quality_{check['name']}_{check['status']}")
    for item in analysis["events"].values():
        risks.extend(item.get("risk_flags", []))
    for item in analysis["metrics"].values():
        risks.extend(item.get("risk_flags", []))
    risks.extend(["release_ball_v1_prototype_only", "single_camera_2d_measurements"])
    return list(dict.fromkeys(risks))


def write_timeline(path: Path, report: dict[str, Any]) -> None:
    columns = [
        "type",
        "name",
        "frame",
        "start_frame",
        "end_frame",
        "timestamp",
        "status",
        "confidence",
        "source_provenance",
    ]
    rows = []
    for item in report["phases"].values():
        rows.append(
            {
                "type": "phase",
                "name": item["name"],
                "frame": "",
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "timestamp": item["start_seconds"],
                "status": item["status"],
                "confidence": item["confidence"],
                "source_provenance": "|".join(item["provenance"]),
            }
        )
    for item in report["events"].values():
        rows.append(
            {
                "type": "event",
                "name": item["name"],
                "frame": item["frame"],
                "start_frame": "",
                "end_frame": "",
                "timestamp": item["timestamp_seconds"],
                "status": item["status"],
                "confidence": item["confidence"],
                "source_provenance": "|".join(item["provenance"]),
            }
        )
    human_ball = report.get("human_ball_release", {})
    fps = float(report["input"]["fps"])
    for item in human_ball.get("contact_state_sequence", []):
        rows.append(
            {
                "type": "human_ball_frame",
                "name": item["contact_state"],
                "frame": item["frame"],
                "start_frame": "",
                "end_frame": "",
                "timestamp": round(item["frame"] / fps, 4),
                "status": item["ball_status"],
                "confidence": item["state_confidence"],
                "source_provenance": "|".join(item["provenance"]),
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_frame_evidence(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    rows_by_frame = {row["frame_index"]: row for row in rows}
    output = {}
    for name, item in report["events"].items():
        frame = item.get("frame")
        row = rows_by_frame.get(frame) if frame is not None else None
        output[name] = {
            "frame": frame,
            "timestamp_seconds": item.get("timestamp_seconds"),
            "status": item["status"],
            "pose": (
                {
                    "visible_keypoints": row["analysis_pose"]["visible_keypoints"],
                    "bbox": row["analysis_pose"]["bbox"],
                    "selection": row["analysis_pose"]["selection"],
                    "correction_status": row["analysis_pose"]["correction_status"],
                    "shooter_track_id": row["analysis_pose"].get("shooter_track_id"),
                    "shooter_selection_confidence": row["analysis_pose"].get("shooter_selection_confidence"),
                    "identity_break": row["analysis_pose"].get("identity_break", False),
                    "crop_status": row["analysis_pose"].get("crop_status"),
                }
                if row and row.get("analysis_pose")
                else None
            ),
            "ball": (
                {
                    "center": row["human_ball"]["ball_center"],
                    "bbox": row["human_ball"]["ball_bbox"],
                    "confidence": row["human_ball"]["ball_confidence"],
                    "ball_status": row["human_ball"]["ball_status"],
                    "contact_state": row["human_ball"]["contact_state"],
                }
                if row and row.get("human_ball")
                else row.get("ball") if row else None
            ),
            "image": f"{name}.jpg" if frame is not None else None,
            "provenance": item["provenance"],
        }
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_pose_reliability(
    rows: list[dict[str, Any]],
    raw_analysis: dict[str, Any],
    analysis: dict[str, Any],
    pose_backbone: str,
) -> dict[str, Any]:
    event_delta = {}
    for name in raw_analysis["events"]:
        raw_frame = raw_analysis["events"][name]["frame"]
        clean_frame = analysis["events"][name]["frame"]
        event_delta[name] = clean_frame - raw_frame if raw_frame is not None and clean_frame is not None else None
    return {
        "pose_source": "rtmpose-m_body7_256x192" if pose_backbone == "rtmpose" else "yolo11_pose",
        "analysis_pose_integrated": True,
        "localization_evidence": "raw_pose",
        "temporal_signal": "analysis_pose",
        "global_coordinate_smoothing": pose_backbone == "yolo",
        "identity_break_frames": [
            row["frame_index"] for row in rows if row.get("tracking", {}).get("identity_break")
        ],
        "crop_status_counts": {
            status: sum(row.get("tracking", {}).get("crop_status") == status for row in rows)
            for status in sorted({row.get("tracking", {}).get("crop_status") for row in rows})
            if status is not None
        },
        "raw": evaluate_pose_rows(rows, pose_key="raw_pose"),
        "analysis": evaluate_pose_rows(rows, pose_key="analysis_pose"),
        "no_lag": no_lag_metrics(rows, rows, analysis["shooting_side"]),
        "event_delta_frames": event_delta,
        "semantics": {
            "keypoint_confidence": "model output score, not measurement accuracy probability",
            "temporal_reliability": "post-processing usability score",
            "correction_status": ["observed", "corrected", "interpolated", "unavailable"],
        },
    }


def build_pose_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        raw = row.get("raw_pose")
        analysis = row.get("analysis_pose")
        output.append(
            {
                "frame_index": row["frame_index"],
                "raw_pose": {
                    "keypoints": raw["keypoints"],
                    "keypoint_confidence": raw["confidence"],
                    "pose_provider": raw.get("pose_provider"),
                    "pose_model": raw.get("pose_model"),
                    "pose_model_config": raw.get("pose_model_config"),
                    "pose_runtime_ms": raw.get("pose_runtime_ms"),
                    "person_box_source": raw.get("person_box_source"),
                    "coordinate_space": raw.get("coordinate_space"),
                    "raw_derived_status": raw.get("raw_derived_status"),
                    "shooter_track_id": raw.get("shooter_track_id"),
                    "shooter_selection_confidence": raw.get("shooter_selection_confidence"),
                    "identity_break": raw.get("identity_break", False),
                    "crop_status": raw.get("crop_status"),
                } if raw else None,
                "analysis_pose": {
                    "keypoints": analysis["keypoints"],
                    "keypoint_confidence": analysis["keypoint_confidence"],
                    "temporal_reliability": analysis["temporal_reliability"],
                    "joint_status": analysis["joint_status"],
                    "correction_status": analysis["correction_status"],
                    "provenance": analysis["provenance"],
                } if analysis else None,
            }
        )
    return output
