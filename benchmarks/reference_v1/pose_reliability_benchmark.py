from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_adapters import RtmlibPoseAdapter, YoloPoseAdapter  # noqa: E402
from reference_v1.pose.metrics import evaluate_pose_rows, no_lag_metrics  # noqa: E402
from reference_v1.pose.reliability import build_analysis_pose  # noqa: E402


DEFAULT_OUTPUT = Path(r"E:\BasketballShotAI\analysis_runs\pose_reliability_pass")
SAMPLES = (
    {
        "id": "img_7215_release_drive",
        "video": Path(r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7215.MOV"),
        "start": 75,
        "end": 145,
        "release": 122,
        "timing_role": "normal_speed_regression",
    },
    {
        "id": "img_7216_release_drive",
        "video": Path(r"E:\BasketballShotAI\raw\confirmed_videos\IMG_7216.MOV"),
        "start": 72,
        "end": 160,
        "release": 135,
        "timing_role": "normal_speed_regression",
    },
    {
        "id": "bili_005_a_difficult_release",
        "video": Path(r"E:\BasketballShotAI\raw\confirmed_videos\BILI_005_A_BV1Re4y1K7Ey.mp4"),
        "start": 150,
        "end": 260,
        "release": None,
        "timing_role": "slow_motion_robustness_only",
    },
)

SKELETON = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def read_frames(video: Path, start: int, end: int) -> list[tuple[int, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for frame_index in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append((frame_index, frame))
    capture.release()
    return frames


def infer(sample: dict[str, Any], candidate: str, cache_path: Path) -> dict[str, Any]:
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    fixed_bbox = candidate.endswith("_bbox")
    base_candidate = candidate.removesuffix("_bbox")
    adapter = YoloPoseAdapter(ROOT / "yolo11n-pose.pt", "largest") if base_candidate == "yolo_raw" else RtmlibPoseAdapter(base_candidate)
    raw_by_frame: dict[int, dict[str, Any]] = {}
    analysis_by_frame: dict[int, dict[str, Any]] = {}
    if fixed_bbox:
        raw_cache = cache_path.parents[1] / "yolo_raw" / cache_path.name
        raw_data = json.loads(raw_cache.read_text(encoding="utf-8"))
        raw_rows = [{**row, "raw_pose": row.get("pose")} for row in raw_data["rows"]]
        raw_by_frame = {int(row["frame_index"]): row for row in raw_rows}
        analysis_by_frame = {int(row["frame_index"]): row for row in build_analysis_pose(raw_rows)}
    rows = []
    started = time.perf_counter()
    for frame_index, frame in read_frames(sample["video"], sample["start"], sample["end"]):
        bbox = None
        if fixed_bbox:
            raw_row = raw_by_frame.get(frame_index)
            clean_row = analysis_by_frame.get(frame_index)
            if not raw_row or not raw_row.get("raw_pose") or not clean_row or not clean_row.get("analysis_pose") or clean_row["analysis_pose"]["correction_status"] == "unavailable":
                rows.append({"frame_index": frame_index, "runtime_ms": None, "pose": None, "excluded_identity_loss": True})
                continue
            bbox = [float(value) for value in raw_row["raw_pose"]["bbox"]]
        pose, runtime_ms = adapter.infer(frame, bbox) if fixed_bbox else adapter.infer(frame)
        rows.append({"frame_index": frame_index, "runtime_ms": runtime_ms, "pose": pose})
    value = {
        "sample": {key: str(item) if isinstance(item, Path) else item for key, item in sample.items()},
        "candidate": candidate,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference V1 pose reliability micro-benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", nargs="+", choices=("yolo_raw", "yolo_analysis", "rtmpose", "rtmw", "rtmpose_bbox", "rtmw_bbox"), default=["yolo_raw"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    results = []
    for sample in SAMPLES:
        for candidate in args.candidates:
            inference_candidate = "yolo_raw" if candidate == "yolo_analysis" else candidate
            cache_candidate = inference_candidate.replace("_bbox", "_crop")
            cache_path = args.output / "cache" / cache_candidate / f"{sample['id']}.json"
            if args.no_cache and cache_path.is_file():
                cache_path.unlink()
            try:
                data = infer(sample, inference_candidate, cache_path)
                if candidate == "yolo_analysis":
                    raw_rows = [{**row, "raw_pose": row.get("pose")} for row in data["rows"]]
                    evaluated_rows = build_analysis_pose(raw_rows)
                    metrics = evaluate_pose_rows(evaluated_rows, pose_key="analysis_pose")
                    metrics["no_lag"] = no_lag_metrics(raw_rows, evaluated_rows, "right")
                else:
                    evaluated_rows = data["rows"]
                    metrics = evaluate_pose_rows(evaluated_rows)
                runtimes = [row["runtime_ms"] for row in data["rows"] if row.get("runtime_ms") is not None]
                metrics["mean_runtime_ms"] = sum(runtimes) / max(len(runtimes), 1)
                results.append({"sample": sample["id"], "candidate": candidate, "status": "ok", "metrics": metrics})
            except Exception as exc:
                results.append({"sample": sample["id"], "candidate": candidate, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "pose_reliability_benchmark.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for sample in SAMPLES:
        raw_cache = args.output / "cache" / "yolo_raw" / f"{sample['id']}.json"
        if raw_cache.is_file():
            write_raw_analysis_review(sample, json.loads(raw_cache.read_text(encoding="utf-8"))["rows"], args.output / "review")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(path)


def _draw_pose(frame: np.ndarray, pose: dict[str, Any] | None, color: tuple[int, int, int]) -> None:
    if not pose:
        return
    points = np.asarray(pose["keypoints"], dtype=float)
    confidence = np.asarray(pose.get("temporal_reliability", pose.get("confidence", [])), dtype=float)
    for first, second in SKELETON:
        if confidence[first] >= 0.25 and confidence[second] >= 0.25:
            cv2.line(frame, tuple(points[first].astype(int)), tuple(points[second].astype(int)), color, 3)
    for index in range(min(17, len(points))):
        if confidence[index] >= 0.25:
            cv2.circle(frame, tuple(points[index].astype(int)), 5, color, -1)


def write_raw_analysis_review(sample: dict[str, Any], cached_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    raw_rows = [{**row, "raw_pose": row.get("pose")} for row in cached_rows]
    analysis_rows = build_analysis_pose(raw_rows)
    by_frame = {int(row["frame_index"]): row for row in analysis_rows}
    frames = read_frames(sample["video"], sample["start"], sample["end"])
    if not frames:
        raise ValueError(f"No review frames for {sample['id']}")
    first = frames[0][1]
    height = 540
    width = round(first.shape[1] * height / first.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sample['id']}_raw_vs_analysis.mp4"
    capture = cv2.VideoCapture(str(sample["video"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
    capture.release()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    for frame_index, frame in frames:
        row = by_frame[frame_index]
        left = cv2.resize(frame, (width, height))
        right = left.copy()
        scale_x, scale_y = width / frame.shape[1], height / frame.shape[0]
        for pose_key, canvas, color in (("raw_pose", left, (0, 210, 255)), ("analysis_pose", right, (80, 255, 130))):
            pose = copy_pose = json.loads(json.dumps(row.get(pose_key))) if row.get(pose_key) else None
            if copy_pose:
                xy = np.asarray(copy_pose["keypoints"], dtype=float)
                xy[:, 0] *= scale_x
                xy[:, 1] *= scale_y
                copy_pose["keypoints"] = xy.tolist()
            _draw_pose(canvas, pose, color)
        status = row["analysis_pose"]["correction_status"] if row.get("analysis_pose") else "unavailable"
        cv2.putText(left, f"RAW | frame {frame_index}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
        cv2.putText(right, f"ANALYSIS | {status}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 130), 2)
        writer.write(np.hstack([left, right]))
    writer.release()
    return path


if __name__ == "__main__":
    main()
