from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from model_adapters import (
    CoTracker3Adapter,
    RFDetrBallAdapter,
    RtmlibPoseAdapter,
    SahiBallAdapter,
    YoloBallAdapter,
    YoloPoseAdapter,
    bbox_iou,
    normalized_point_distance,
    optical_flow_track,
)
from grounded_sam2_adapter import run_grounded_sam2


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(r"E:\BasketballShotAI\analysis_runs\reference_benchmark")
DEFAULT_COTRACKER_REPO = Path(r"E:\BasketballShotAI\tools\reference_benchmark\co-tracker")
RFDETR_NANO_MODEL = DEFAULT_OUTPUT / "model_cache" / "rfdetr" / "rf-detr-nano.pth"
DEFAULT_CONFIG = Path(__file__).with_name("default-samples.json")
POSE_MODEL = ROOT / "yolo11n-pose.pt"
COCO_MODEL = ROOT / "yolo11n.pt"
CUSTOM_MODEL = ROOT / "runs/detect/runs/ball_rim_player_smoke/weights/best.pt"
RELEASE_MODEL = (
    ROOT
    / "runs/detect/runs/release_ball/yolo11n_release_ball_batch001_003_v1/weights/best.pt"
)
REVIEW_LABELS = Path(
    r"E:\BasketballShotAI\tools\locateanything_local\batch_runs\batch_002_shot_frames"
    r"\phase3_review\final_research_only_labels.csv"
)
PHASE_V2_ROOT = Path(r"E:\BasketballShotAI\analysis_runs\phase_v2_debug")


@dataclass
class Result:
    module: str
    candidate: str
    ran: bool
    input: str
    result: str
    runtime_ms: float | None
    strength: str
    failure: str
    decision: str


class Harness:
    def __init__(self, output_root: Path, use_cache: bool = True) -> None:
        self.output_root = output_root
        self.cache_root = output_root / "cache"
        self.use_cache = use_cache
        self.results: list[Result] = []
        self.failures: list[dict[str, Any]] = []
        output_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def cached(self, module: str, candidate: str, sample: str, run: Callable[[], Any]) -> Any:
        path = self.cache_root / module / candidate / f"{sample}.json"
        if self.use_cache and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        value = run()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return value

    def fail(self, module: str, candidate: str, input_name: str, exc: BaseException) -> None:
        failure = {
            "module": module,
            "candidate": candidate,
            "input": input_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        self.failures.append(failure)
        self.results.append(
            Result(module, candidate, False, input_name, "blocked", None, "", str(exc), "DEFER")
        )

    def write_summary(self) -> None:
        json_path = self.output_root / "benchmark_results.json"
        csv_path = self.output_root / "benchmark_results.csv"
        json_path.write_text(
            json.dumps([asdict(item) for item in self.results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(Result.__annotations__))
            writer.writeheader()
            writer.writerows(asdict(item) for item in self.results)
        with (self.output_root / "failures.jsonl").open("w", encoding="utf-8") as handle:
            for item in self.failures:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_video_frames(video: Path, start: int, end: int) -> tuple[list[int], list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    indices: list[int] = []
    frames: list[np.ndarray] = []
    for frame_index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        indices.append(frame_index)
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded: {video} {start}-{end}")
    return indices, frames, fps


def _pose_metrics(rows: list[dict[str, Any]], reference_release: int) -> dict[str, Any]:
    found = [row for row in rows if row.get("pose")]
    runtimes = [float(row["runtime_ms"]) for row in rows]
    visible = [int(row["pose"]["visible_keypoints"]) for row in found]
    pose_steps: list[float] = []
    wrist_steps: list[float] = []
    previous: dict[str, Any] | None = None
    event_candidates: list[tuple[float, int]] = []
    for row in found:
        pose = row["pose"]
        xy = np.asarray(pose["keypoints"], dtype=float)
        confidence = np.asarray(pose["confidence"], dtype=float)
        if len(xy) >= 17 and all(confidence[index] >= 0.25 for index in (5, 6, 9, 10, 11, 12)):
            shoulder = (xy[5] + xy[6]) / 2
            hip = (xy[11] + xy[12]) / 2
            scale = float(np.linalg.norm(shoulder - hip))
            wrist_y = min(float(xy[9, 1]), float(xy[10, 1]))
            event_candidates.append((wrist_y / max(scale, 1.0), int(row["frame_index"])))
        if previous is not None:
            previous_xy = np.asarray(previous["keypoints"], dtype=float)
            previous_confidence = np.asarray(previous["confidence"], dtype=float)
            count = min(len(xy), len(previous_xy), 17)
            joint_mask = (confidence[:count] >= 0.25) & (previous_confidence[:count] >= 0.25)
            if np.any(joint_mask):
                box = pose["bbox"]
                scale = math.hypot(box[2] - box[0], box[3] - box[1])
                pose_steps.append(
                    float(np.median(np.linalg.norm(xy[:count][joint_mask] - previous_xy[:count][joint_mask], axis=1)))
                    / max(scale, 1.0)
                )
            for wrist in (9, 10):
                if wrist < count and confidence[wrist] >= 0.25 and previous_confidence[wrist] >= 0.25:
                    box = pose["bbox"]
                    scale = math.hypot(box[2] - box[0], box[3] - box[1])
                    wrist_steps.append(
                        float(np.linalg.norm(xy[wrist] - previous_xy[wrist])) / max(scale, 1.0)
                    )
        previous = pose
    event_frame = min(event_candidates)[1] if event_candidates else None
    track_ids = [row["pose"].get("track_id") for row in found if row["pose"].get("track_id") is not None]
    return {
        "frames": len(rows),
        "coverage": len(found) / len(rows),
        "mean_visible_keypoints": statistics.mean(visible) if visible else 0.0,
        "median_pose_step": statistics.median(pose_steps) if pose_steps else None,
        "median_wrist_step": statistics.median(wrist_steps) if wrist_steps else None,
        "event_proxy_frame": event_frame,
        "event_frame_delta": event_frame - reference_release if event_frame is not None else None,
        "unique_track_ids": len(set(track_ids)),
        "mean_runtime_ms": statistics.mean(runtimes),
    }


def run_tracking_pose(harness: Harness, config: dict[str, Any], include_rtmlib: bool) -> None:
    candidates = ["largest", "bytetrack", "botsort"]
    for sample in config["tracking_pose"]:
        indices, frames, _ = read_video_frames(Path(sample["video"]), sample["start"], sample["end"])
        for candidate in candidates:
            try:
                def execute() -> dict[str, Any]:
                    adapter = YoloPoseAdapter(POSE_MODEL, candidate)
                    rows = []
                    for frame_index, frame in zip(indices, frames):
                        pose, runtime_ms = adapter.infer(frame)
                        rows.append({"frame_index": frame_index, "runtime_ms": runtime_ms, "pose": pose})
                    metrics = _pose_metrics(rows, sample["reference_release"])
                    metrics["reported_id_switches"] = adapter.id_switches
                    return {"metrics": metrics, "rows": rows}

                data = harness.cached("person_tracking", candidate, sample["id"], execute)
                metrics = data["metrics"]
                detail = harness.output_root / "person_tracking" / candidate
                detail.mkdir(parents=True, exist_ok=True)
                _write_rows(detail / f"{sample['id']}.csv", data["rows"])
                harness.results.append(
                    Result(
                        "person_tracking",
                        candidate,
                        True,
                        sample["id"],
                        json.dumps(metrics, ensure_ascii=False),
                        metrics["mean_runtime_ms"],
                        "persistent ID" if candidate != "largest" else "zero tracker dependency",
                        "ID switch is a proxy without identity GT",
                        "KEEP_BASELINE" if candidate == "largest" else "COMPARE",
                    )
                )
            except Exception as exc:
                harness.fail("person_tracking", candidate, sample["id"], exc)

        if include_rtmlib:
            for candidate in ("rtmpose", "rtmw"):
                try:
                    def execute_pose(candidate: str = candidate) -> dict[str, Any]:
                        adapter = RtmlibPoseAdapter(candidate)
                        rows = []
                        for frame_index, frame in zip(indices[::3], frames[::3]):
                            pose, runtime_ms = adapter.infer(frame)
                            rows.append({"frame_index": frame_index, "runtime_ms": runtime_ms, "pose": pose})
                        return {"metrics": _pose_metrics(rows, sample["reference_release"]), "rows": rows}

                    data = harness.cached("pose", candidate, sample["id"], execute_pose)
                    metrics = data["metrics"]
                    detail = harness.output_root / "pose" / candidate
                    detail.mkdir(parents=True, exist_ok=True)
                    _write_rows(detail / f"{sample['id']}.csv", data["rows"])
                    harness.results.append(
                        Result(
                            "pose",
                            candidate,
                            True,
                            f"{sample['id']} every_3_frames",
                            json.dumps(metrics, ensure_ascii=False),
                            metrics["mean_runtime_ms"],
                            "whole-body available" if candidate == "rtmw" else "body pose challenger",
                            "no pose GT; stability/visibility only",
                            "COMPARE",
                        )
                    )
                except Exception as exc:
                    harness.fail("pose", candidate, sample["id"], exc)


def run_ball(harness: Harness, limit: int | None) -> None:
    rows = list(csv.DictReader(REVIEW_LABELS.open(encoding="utf-8-sig")))
    rows = [row for row in rows if row["review_status"] != "skipped"]
    if limit:
        rows = rows[:limit]
    candidates = {
        "coco_sports_ball": lambda: YoloBallAdapter(COCO_MODEL, 32, 0.10),
        "release_ball_v1": lambda: YoloBallAdapter(RELEASE_MODEL, None, 0.15),
        "ball_rim_player_v1": lambda: YoloBallAdapter(CUSTOM_MODEL, 0, 0.15),
        "rfdetr_nano_coco": lambda: RFDetrBallAdapter(RFDETR_NANO_MODEL, 0.15),
    }
    for candidate, make_adapter in candidates.items():
        try:
            def execute() -> dict[str, Any]:
                adapter = make_adapter()
                details = []
                runtimes = []
                matched = 0
                small_matched = 0
                small_total = 0
                unmatched_detections = 0
                for row in rows:
                    image = cv2.imread(row["image_path"])
                    if image is None:
                        continue
                    gt = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
                    detections, runtime_ms = adapter.infer(image)
                    runtimes.append(runtime_ms)
                    ious = [bbox_iou(gt, item["bbox"]) for item in detections]
                    best_iou = max(ious, default=0.0)
                    is_match = best_iou >= 0.5
                    matched += int(is_match)
                    area_ratio = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (image.shape[0] * image.shape[1])
                    is_small = area_ratio < 0.003
                    small_total += int(is_small)
                    small_matched += int(is_small and is_match)
                    unmatched_detections += max(0, len(detections) - int(is_match))
                    details.append(
                        {
                            "sample_id": row["sample_id"],
                            "best_iou": best_iou,
                            "matched_iou_50": is_match,
                            "detections": len(detections),
                            "small_ball": is_small,
                            "runtime_ms": runtime_ms,
                        }
                    )
                evaluated = len(details)
                return {
                    "metrics": {
                        "evaluated": evaluated,
                        "positive_recall_iou_50": matched / evaluated if evaluated else 0,
                        "small_ball_recall_iou_50": small_matched / small_total if small_total else None,
                        "unmatched_detections": unmatched_detections,
                        "mean_runtime_ms": statistics.mean(runtimes) if runtimes else None,
                        "benchmark_status": "research_only_not_independent_product_test",
                    },
                    "rows": details,
                }

            data = harness.cached("ball_detection", candidate, f"review230_{limit or 'all'}", execute)
            detail = harness.output_root / "ball_detection" / candidate
            detail.mkdir(parents=True, exist_ok=True)
            _write_rows(detail / "detections.csv", data["rows"])
            metrics = data["metrics"]
            if "false_positives" in metrics and "unmatched_detections" not in metrics:
                metrics["unmatched_detections"] = metrics.pop("false_positives")
            metrics["terminology_note"] = "unmatched detections are not formal false positives"
            harness.results.append(
                Result(
                    "ball_detection",
                    candidate,
                    True,
                    f"{metrics['evaluated']} reviewed research-only frames",
                    json.dumps(metrics, ensure_ascii=False),
                    metrics["mean_runtime_ms"],
                    "existing locally runnable baseline",
                    "labels derive from reviewed research-only LocateAnything workflow",
                    "COMPARE",
                )
            )
        except Exception as exc:
            harness.fail("ball_detection", candidate, "review230", exc)


def run_sahi(harness: Harness, limit: int | None) -> None:
    rows = list(csv.DictReader(REVIEW_LABELS.open(encoding="utf-8-sig")))
    rows = [row for row in rows if row["review_status"] != "skipped"]
    if limit:
        rows = rows[:limit]
    candidates = {
        "native_640": lambda: YoloBallAdapter(RELEASE_MODEL, None, 0.15),
        "sahi_320_overlap_020": lambda: SahiBallAdapter(RELEASE_MODEL, 0.15, 320, 0.20),
        "sahi_480_overlap_020": lambda: SahiBallAdapter(RELEASE_MODEL, 0.15, 480, 0.20),
    }
    for candidate, make_adapter in candidates.items():
        try:
            def execute() -> dict[str, Any]:
                adapter = make_adapter()
                details = []
                runtimes = []
                matched = small_matched = small_total = 0
                unmatched_detections = duplicate_detections = 0
                center_errors = []
                for row in rows:
                    image = cv2.imread(row["image_path"])
                    if image is None:
                        continue
                    gt = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
                    detections, runtime_ms = adapter.infer(image)
                    runtimes.append(runtime_ms)
                    ious = [bbox_iou(gt, item["bbox"]) for item in detections]
                    matching = [index for index, value in enumerate(ious) if value >= 0.5]
                    is_match = bool(matching)
                    matched += int(is_match)
                    unmatched_detections += max(0, len(detections) - int(is_match))
                    duplicate_detections += max(0, len(matching) - 1)
                    area_ratio = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (
                        image.shape[0] * image.shape[1]
                    )
                    is_small = area_ratio < 0.003
                    small_total += int(is_small)
                    small_matched += int(is_small and is_match)
                    if is_match:
                        best = detections[max(range(len(ious)), key=ious.__getitem__)]["bbox"]
                        gt_center = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
                        predicted_center = ((best[0] + best[2]) / 2, (best[1] + best[3]) / 2)
                        center_errors.append(math.dist(gt_center, predicted_center))
                    details.append(
                        {
                            "sample_id": row["sample_id"],
                            "best_iou": max(ious, default=0.0),
                            "matched_iou_50": is_match,
                            "detections": len(detections),
                            "duplicate_detections": max(0, len(matching) - 1),
                            "small_ball": is_small,
                            "runtime_ms": runtime_ms,
                        }
                    )
                evaluated = len(details)
                return {
                    "metrics": {
                        "evaluated": evaluated,
                        "recall_iou_50": matched / evaluated if evaluated else 0,
                        "small_ball_recall_iou_50": small_matched / small_total if small_total else None,
                        "matched_center_error_px": statistics.mean(center_errors) if center_errors else None,
                        "unmatched_detections": unmatched_detections,
                        "duplicate_detections": duplicate_detections,
                        "mean_runtime_ms": statistics.mean(runtimes) if runtimes else None,
                        "benchmark_status": "research_only_not_independent_product_test",
                        "terminology_note": "unmatched detections are not formal false positives",
                    },
                    "rows": details,
                }

            data = harness.cached("sahi", candidate, f"review230_{limit or 'all'}", execute)
            detail = harness.output_root / "sahi" / candidate
            detail.mkdir(parents=True, exist_ok=True)
            _write_rows(detail / "detections.csv", data["rows"])
            metrics = data["metrics"]
            harness.results.append(
                Result(
                    "small_ball_slicing",
                    candidate,
                    True,
                    f"{metrics['evaluated']} reviewed research-only positive frames",
                    json.dumps(metrics, ensure_ascii=False),
                    metrics["mean_runtime_ms"],
                    "same release-ball weights, confidence and IoU rule",
                    "not an independent negative test; unmatched detections are diagnostic only",
                    "COMPARE",
                )
            )
        except Exception as exc:
            harness.fail("small_ball_slicing", candidate, "review230", exc)


def run_point_tracking(harness: Harness, config: dict[str, Any]) -> None:
    for sample in config["strict_release"][:2]:
        indices, frames, _ = read_video_frames(Path(sample["video"]), sample["start"], sample["end"])
        try:
            detector = YoloBallAdapter(RELEASE_MODEL, None, 0.10)
            detections = []
            initial_index = None
            initial_point = None
            detector_runtime = 0.0
            for offset, frame in enumerate(frames):
                boxes, runtime_ms = detector.infer(frame)
                detector_runtime += runtime_ms
                best = max(boxes, key=lambda item: item["confidence"], default=None)
                detections.append(best)
                if initial_point is None and best is not None:
                    x1, y1, x2, y2 = best["bbox"]
                    initial_index = offset
                    initial_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            if initial_point is None or initial_index is None:
                raise RuntimeError("release-ball detector produced no point initialization")
            started = time.perf_counter()
            tracked = optical_flow_track(frames[initial_index:], initial_point)
            tracker_runtime = (time.perf_counter() - started) * 1000
            detail_rows = []
            distances = []
            detector_coverage = 0
            for offset, track in enumerate(tracked, start=initial_index):
                detection = detections[offset]
                detector_center = None
                if detection:
                    detector_coverage += 1
                    x1, y1, x2, y2 = detection["bbox"]
                    detector_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    scale = max(x2 - x1, y2 - y1)
                    distances.append(
                        normalized_point_distance(
                            (track["x"], track["y"]), detector_center, scale
                        )
                    )
                detail_rows.append(
                    {
                        "frame_index": indices[offset],
                        "tracker_x": track["x"],
                        "tracker_y": track["y"],
                        "tracker_visible": track["visible"],
                        "detector_x": detector_center[0] if detector_center else None,
                        "detector_y": detector_center[1] if detector_center else None,
                    }
                )
            metrics = {
                "frames_after_initialization": len(tracked),
                "detector_coverage": detector_coverage / len(tracked),
                "point_tracking_coverage": sum(bool(row["visible"]) for row in tracked) / len(tracked),
                "median_normalized_drift_to_detector": statistics.median(distances) if distances else None,
                "detector_runtime_ms": detector_runtime,
                "tracker_runtime_ms": tracker_runtime,
                "ground_truth": "none_proxy_only",
            }
            detail = harness.output_root / "ball_tracking" / "opencv_lk"
            detail.mkdir(parents=True, exist_ok=True)
            _write_rows(detail / f"{sample['id']}_trajectory.csv", detail_rows)
            _write_contact_sheet(detail / f"{sample['id']}_contact_sheet.jpg", frames, indices, detail_rows)
            harness.results.append(
                Result(
                    "ball_tracking",
                    "detector_init_opencv_lk",
                    True,
                    sample["id"],
                    json.dumps(metrics, ensure_ascii=False),
                    tracker_runtime / len(tracked),
                    "real point-tracker baseline with trajectory CSV",
                    "no GT; drift is measured only where detector also fires",
                    "KEEP_AS_CONTROL",
                )
            )
        except Exception as exc:
            harness.fail("ball_tracking", "detector_init_opencv_lk", sample["id"], exc)


def run_cotracker(harness: Harness, config: dict[str, Any]) -> None:
    try:
        tracker = CoTracker3Adapter(DEFAULT_COTRACKER_REPO)
    except Exception as exc:
        harness.fail("ball_tracking", "detector_init_cotracker3", "model_load", exc)
        return
    for sample in config["strict_release"][:2]:
        try:
            indices, frames, _ = read_video_frames(
                Path(sample["video"]), sample["start"], sample["end"]
            )
            detector = YoloBallAdapter(RELEASE_MODEL, None, 0.10)
            detections = []
            initial_index = None
            initial_point = None
            for offset, frame in enumerate(frames):
                boxes, _ = detector.infer(frame)
                best = max(boxes, key=lambda item: item["confidence"], default=None)
                detections.append(best)
                if initial_point is None and best is not None:
                    x1, y1, x2, y2 = best["bbox"]
                    initial_index = offset
                    initial_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            if initial_point is None or initial_index is None:
                raise RuntimeError("release-ball detector produced no point initialization")
            tracked, runtime_ms = tracker.track(frames, initial_index, initial_point)
            rows = []
            distances = []
            detector_coverage = 0
            for offset, track in enumerate(tracked):
                detection = detections[offset]
                detector_center = None
                if detection:
                    detector_coverage += 1
                    x1, y1, x2, y2 = detection["bbox"]
                    detector_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    distances.append(
                        normalized_point_distance(
                            (track["x"], track["y"]),
                            detector_center,
                            max(x2 - x1, y2 - y1),
                        )
                    )
                rows.append(
                    {
                        "frame_index": indices[offset],
                        "tracker_x": track["x"],
                        "tracker_y": track["y"],
                        "tracker_visible": track["visible"],
                        "detector_x": detector_center[0] if detector_center else None,
                        "detector_y": detector_center[1] if detector_center else None,
                    }
                )
            metrics = {
                "frames": len(tracked),
                "detector_coverage": detector_coverage / len(tracked),
                "point_tracking_coverage": sum(row["visible"] for row in tracked) / len(tracked),
                "median_normalized_drift_to_detector": statistics.median(distances) if distances else None,
                "tracker_runtime_ms": runtime_ms,
                "ground_truth": "none_proxy_only",
            }
            detail = harness.output_root / "ball_tracking" / "cotracker3"
            detail.mkdir(parents=True, exist_ok=True)
            _write_rows(detail / f"{sample['id']}_trajectory.csv", rows)
            _write_contact_sheet(detail / f"{sample['id']}_contact_sheet.jpg", frames, indices, rows)
            harness.results.append(
                Result(
                    "ball_tracking",
                    "detector_init_cotracker3",
                    True,
                    sample["id"],
                    json.dumps(metrics, ensure_ascii=False),
                    runtime_ms / len(tracked),
                    "joint learned point tracking with short-occlusion visibility",
                    "CPU-only run; no point GT, drift proxy uses detector observations",
                    "COMPARE",
                )
            )
        except Exception as exc:
            harness.fail("ball_tracking", "detector_init_cotracker3", sample["id"], exc)


def run_strict_release(harness: Harness, config: dict[str, Any]) -> None:
    for sample in config["strict_release"]:
        try:
            indices, frames, _ = read_video_frames(Path(sample["video"]), sample["start"], sample["end"])
            pose = YoloPoseAdapter(POSE_MODEL, "largest")
            ball = YoloBallAdapter(RELEASE_MODEL, None, 0.10)
            evidence = []
            started = time.perf_counter()
            for frame_index, frame in zip(indices, frames):
                pose_row, pose_ms = pose.infer(frame)
                ball_rows, ball_ms = ball.infer(frame)
                best_ball = max(ball_rows, key=lambda item: item["confidence"], default=None)
                distance = None
                ball_center = None
                wrist = None
                if pose_row and best_ball and len(pose_row["keypoints"]) >= 11:
                    xy = np.asarray(pose_row["keypoints"], dtype=float)
                    confidence = np.asarray(pose_row["confidence"], dtype=float)
                    x1, y1, x2, y2 = best_ball["bbox"]
                    ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    ball_scale = max(x2 - x1, y2 - y1)
                    wrist_candidates = [
                        (normalized_point_distance(tuple(xy[index]), ball_center, ball_scale), index)
                        for index in (9, 10)
                        if confidence[index] >= 0.25
                    ]
                    if wrist_candidates:
                        distance, wrist_index = min(wrist_candidates)
                        wrist = tuple(float(value) for value in xy[wrist_index])
                evidence.append(
                    {
                        "frame_index": frame_index,
                        "ball_detected": best_ball is not None,
                        "ball_center": ball_center,
                        "wrist": wrist,
                        "ball_wrist_distance_diameters": distance,
                        "pose_runtime_ms": pose_ms,
                        "ball_runtime_ms": ball_ms,
                    }
                )
            predicted, risks = _decode_release(evidence, sample["pose_release"])
            elapsed_ms = (time.perf_counter() - started) * 1000
            output = {
                "sample_id": sample["id"],
                "predicted_strict_frame": predicted,
                "pose_release_frame": sample["pose_release"],
                "historical_ball_release_frame": sample["historical_ball_release"],
                "delta_from_pose": predicted - sample["pose_release"] if predicted is not None else None,
                "delta_from_historical_ball": predicted - sample["historical_ball_release"] if predicted is not None else None,
                "risk_flags": risks,
                "evidence": evidence,
                "method": "experimental_geometry_proxy_contact_to_separation",
                "runtime_ms": elapsed_ms,
            }
            detail = harness.output_root / "strict_release"
            detail.mkdir(parents=True, exist_ok=True)
            (detail / f"{sample['id']}.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            harness.results.append(
                Result(
                    "strict_release",
                    "geometry_contact_transition_v0",
                    True,
                    sample["id"],
                    json.dumps({key: output[key] for key in output if key != "evidence"}, ensure_ascii=False),
                    elapsed_ms / len(frames),
                    "contact/separation evidence and risk flags",
                    "experimental proxy; no hand-contact model or independent strict GT",
                    "ADD_EXPERIMENTAL",
                )
            )
        except Exception as exc:
            harness.fail("strict_release", "geometry_contact_transition_v0", sample["id"], exc)


def run_grounded_sam2_benchmark(harness: Harness, config: dict[str, Any]) -> None:
    sample = config["strict_release"][1]
    try:
        _, frames, _ = read_video_frames(Path(sample["video"]), 128, 142)
        result = run_grounded_sam2(
            frames, harness.output_root / "grounded_sam2" / sample["id"]
        )
        harness.results.append(
            Result(
                "video_object_segmentation",
                "grounding_dino_tiny_plus_sam2_1_hiera_tiny",
                True,
                f"{sample['id']} frames 128-142",
                json.dumps(result, ensure_ascii=False),
                result["runtime_ms"] / result["frames"],
                "text prompt to first-frame box to video mask propagation",
                "single smoke clip, CPU-only, no mask GT",
                "RESEARCH_FALLBACK",
            )
        )
    except Exception as exc:
        harness.fail(
            "video_object_segmentation",
            "grounding_dino_tiny_plus_sam2_1_hiera_tiny",
            sample["id"],
            exc,
        )


def run_phase_evidence(harness: Harness) -> None:
    report_dirs = (
        PHASE_V2_ROOT / "batch_eval_2026_08_29",
        PHASE_V2_ROOT / "batch_eval_user_clips_2026_08_30",
    )
    reports = []
    for directory in report_dirs:
        for path in directory.glob("*/phase_v2_debug.json"):
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        harness.fail("temporal_phase", "phase_v2", "historical_10_video_set", RuntimeError("phase-v2 reports not found"))
        return
    statuses: dict[str, int] = {}
    ordered = 0
    complete = 0
    for report in reports:
        status = report.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        events = report.get("events", {})
        values = [events.get(key, {}).get("frame_index") for key in ("setup", "dip", "release", "follow_through", "landing")]
        complete += int(all(value is not None for value in values))
        ordered += int(all(value is not None for value in values) and values == sorted(values))
    metrics = {
        "videos": len(reports),
        "status_counts": statuses,
        "complete_event_sequences": complete,
        "ordered_event_sequences": ordered,
        "missing_event_handling": "status_and_risk_flags",
        "runtime": "not_recorded_in_historical_runs",
        "ground_truth": "no_complete_phase_gt_diagnostic_only",
    }
    harness.results.append(
        Result(
            "temporal_phase",
            "phase_v2_dense_pose_ordered_events",
            True,
            "existing 10-video diagnostic set",
            json.dumps(metrics, ensure_ascii=False),
            None,
            "ordered events with explicit insufficient_data and risk flags",
            "no complete phase GT; historical runtime was not recorded",
            "KEEP_EXPERIMENTAL",
        )
    )


def _decode_release(evidence: list[dict[str, Any]], pose_release: int) -> tuple[int | None, list[str]]:
    risks = ["experimental_geometry_proxy", "no_learned_hand_contact_model"]
    near = [
        row for row in evidence
        if row["ball_wrist_distance_diameters"] is not None
        and row["ball_wrist_distance_diameters"] <= 2.0
        and row["frame_index"] <= pose_release + 4
    ]
    if not near:
        return None, risks + ["no_contact_candidate"]
    contact_frame = max(int(row["frame_index"]) for row in near)
    candidates = []
    for index, row in enumerate(evidence):
        if row["frame_index"] <= contact_frame:
            continue
        window = evidence[index : index + 2]
        if len(window) < 2:
            break
        distances = [item["ball_wrist_distance_diameters"] for item in window]
        centers = [item["ball_center"] for item in window]
        if all(value is not None and value >= 2.3 for value in distances) and all(centers):
            movement = math.hypot(centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])
            if movement >= 2.0:
                candidates.append(int(row["frame_index"]))
                break
    if not candidates:
        return None, risks + ["no_stable_post_release_separation"]
    predicted = candidates[0]
    if abs(predicted - pose_release) > 5:
        risks.append("large_pose_release_delta")
    return predicted, risks


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized = []
    fields: list[str] = []
    for row in rows:
        flat = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        normalized.append(flat)
        for key in flat:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def _write_contact_sheet(path: Path, frames: list[np.ndarray], indices: list[int], rows: list[dict[str, Any]]) -> None:
    selected = np.linspace(0, len(rows) - 1, min(8, len(rows)), dtype=int)
    tiles = []
    for offset in selected:
        frame = frames[offset].copy()
        row = rows[offset]
        cv2.circle(frame, (int(row["tracker_x"]), int(row["tracker_y"])), 8, (0, 255, 255), 2)
        if row["detector_x"] is not None:
            cv2.circle(frame, (int(row["detector_x"]), int(row["detector_y"])), 6, (0, 0, 255), 2)
        cv2.putText(frame, str(indices[offset]), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        scale = 320 / frame.shape[1]
        tiles.append(cv2.resize(frame, (320, int(frame.shape[0] * scale))))
    height = max(tile.shape[0] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT) for tile in tiles]
    sheet = np.concatenate(padded[:4], axis=1)
    if len(padded) > 4:
        sheet = np.concatenate([sheet, np.concatenate(padded[4:], axis=1)], axis=0)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Basketball Shot AI Reference V1 benchmark harness")
    parser.add_argument("module", choices=["all", "tracking_pose", "ball", "sahi", "point", "cotracker", "grounded_sam2", "phase", "strict"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--include-rtmlib", action="store_true")
    parser.add_argument("--ball-limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    harness = Harness(args.output_root, use_cache=not args.no_cache)
    if args.module in {"all", "tracking_pose"}:
        run_tracking_pose(harness, config, args.include_rtmlib)
    if args.module in {"all", "ball"}:
        run_ball(harness, args.ball_limit)
    if args.module in {"all", "sahi"}:
        run_sahi(harness, args.ball_limit)
    if args.module in {"all", "point"}:
        run_point_tracking(harness, config)
    if args.module in {"all", "cotracker"}:
        run_cotracker(harness, config)
    if args.module in {"all", "strict"}:
        run_strict_release(harness, config)
    if args.module in {"all", "grounded_sam2"}:
        run_grounded_sam2_benchmark(harness, config)
    if args.module in {"all", "phase"}:
        run_phase_evidence(harness)
    harness.write_summary()
    print(f"results: {args.output_root / 'benchmark_results.csv'}")
    print(f"failures: {len(harness.failures)}")


if __name__ == "__main__":
    main()
