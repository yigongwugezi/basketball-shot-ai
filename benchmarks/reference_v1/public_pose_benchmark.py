from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference_v1.model_adapters import RtmlibPoseAdapter, YoloPoseAdapter, bbox_iou
from benchmarks.reference_v1.pose_gt import JOINT_INDEX, body_scale, percentile
from benchmarks.reference_v1.public_pose_gt import CORE_JOINTS, read_json, sha256, write_json
from reference_v1.pose.metrics import evaluate_pose_rows
from reference_v1.pose.reliability import build_analysis_pose


PUBLIC_ROOT = Path(r"E:\BasketballShotAI\public_data")
OUTPUT_ROOT = Path(r"E:\BasketballShotAI\analysis_runs\public_pose_benchmark")
CONFIDENCE_THRESHOLD = 0.25
FILTER_NEUTRAL_PIXELS = 2.0
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
MODELS = ("yolo_raw", "yolo_filtered", "rtmpose", "rtmw")


def configure_large_data_caches(public_root: Path) -> None:
    cache = public_root / "cache"
    os.environ["TORCH_HOME"] = str(cache / "rtmlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["YOLO_CONFIG_DIR"] = str(public_root / "config" / "ultralytics")
    for path in (cache / "rtmlib" / "hub" / "checkpoints", cache / "huggingface", public_root / "config" / "ultralytics"):
        path.mkdir(parents=True, exist_ok=True)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def crop_diagnostics(sample: dict[str, Any], pose: dict[str, Any] | None) -> dict[str, Any]:
    if pose is None:
        return {"person_detection_success": False, "crop_success": False, "gt_joint_coverage": 0.0, "gt_bbox_iou": 0.0}
    box = pose["bbox"]
    points = [
        (joint["x"], joint["y"])
        for joint in sample["mapped_joints"].values()
        if joint["visibility"] != "not_labelable"
    ]
    inside = sum(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in points)
    coverage = inside / len(points) if points else 0.0
    return {
        "person_detection_success": True,
        "crop_success": coverage >= 0.75,
        "gt_joint_coverage": round(coverage, 4),
        "gt_bbox_iou": round(bbox_iou(box, xywh_to_xyxy(sample["person_bbox"])), 4),
    }


def pose_head_success(pose: dict[str, Any] | None) -> bool:
    if not pose:
        return False
    scores = pose.get("temporal_reliability", pose.get("confidence", []))
    return any(index < len(scores) and scores[index] >= CONFIDENCE_THRESHOLD for index in JOINT_INDEX.values())


def prediction(pose: dict[str, Any] | None, crop: dict[str, Any], runtime: dict[str, float], *, filter_mode: str | None = None) -> dict[str, Any]:
    head_ok = pose_head_success(pose)
    return {
        "pose": pose,
        **crop,
        "pose_head_success": head_ok,
        "final_prediction_available": bool(crop["crop_success"] and head_ok),
        "runtime_ms": runtime,
        **({"filter_mode": filter_mode} if filter_mode else {}),
    }


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded: {path}")
    return frames, fps


def infer_frame(frame: np.ndarray, yolo: YoloPoseAdapter, rtmpose: RtmlibPoseAdapter, rtmw: RtmlibPoseAdapter) -> dict[str, Any]:
    raw, yolo_ms = yolo.infer(frame)
    bbox = raw["bbox"] if raw else None
    medium, medium_ms = rtmpose.infer(frame, bbox) if bbox else (None, 0.0)
    whole, whole_ms = rtmw.infer(frame, bbox) if bbox else (None, 0.0)
    return {
        "yolo_raw": raw, "rtmpose": medium, "rtmw": whole,
        "runtime_ms": {"yolo_integrated": yolo_ms, "rtmpose_head": medium_ms, "rtmw_head": whole_ms},
    }


def run_inference(packages: dict[str, dict[str, Any]], model_path: Path, output_root: Path) -> dict[str, Any]:
    yolo = YoloPoseAdapter(model_path, "largest")
    rtmpose = RtmlibPoseAdapter("rtmpose")
    rtmw = RtmlibPoseAdapter("rtmw")
    selected: dict[str, Any] = {}
    sequences: dict[str, Any] = {}
    jhmdb_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in packages["jhmdb"]["samples"]:
        jhmdb_by_video[sample["media_path"]].append(sample)

    for offset, (video_path, samples) in enumerate(jhmdb_by_video.items(), start=1):
        frames, fps = read_video(Path(video_path))
        inferred = [infer_frame(frame, yolo, rtmpose, rtmw) for frame in frames]
        raw_rows = [{"frame_index": index, "raw_pose": row["yolo_raw"]} for index, row in enumerate(inferred)]
        started = time.perf_counter()
        filtered_rows = build_analysis_pose(raw_rows)
        filter_ms = (time.perf_counter() - started) * 1000 / len(frames)
        sequence_id = f"jhmdb/{samples[0]['sequence']}"
        sequence_models = {
            "yolo_raw": [{"pose": row["yolo_raw"]} for row in inferred],
            "yolo_filtered": [{"pose": row["analysis_pose"]} for row in filtered_rows],
            "rtmpose": [{"pose": row["rtmpose"]} for row in inferred],
            "rtmw": [{"pose": row["rtmw"]} for row in inferred],
        }
        sequences[sequence_id] = {"frames": len(frames), "fps": fps, "models": sequence_models}
        for sample in samples:
            index = sample["frame_index"]
            if index >= len(frames):
                raise IndexError(f"{sample['id']} frame {index} >= {len(frames)}")
            row = inferred[index]
            crop = crop_diagnostics(sample, row["yolo_raw"])
            selected[sample["id"]] = {
                "dataset": "jhmdb", "image_shape": list(frames[index].shape[:2]),
                "models": {
                    "yolo_raw": prediction(row["yolo_raw"], crop, {"integrated": row["runtime_ms"]["yolo_integrated"], "end_to_end": row["runtime_ms"]["yolo_integrated"]}),
                    "yolo_filtered": prediction(filtered_rows[index]["analysis_pose"], crop, {"integrated": row["runtime_ms"]["yolo_integrated"], "filter": filter_ms, "end_to_end": row["runtime_ms"]["yolo_integrated"] + filter_ms}, filter_mode="CONTIGUOUS_VIDEO_FILTER"),
                    "rtmpose": prediction(row["rtmpose"], crop, {"person_crop_provider": row["runtime_ms"]["yolo_integrated"], "pose_head": row["runtime_ms"]["rtmpose_head"], "end_to_end": row["runtime_ms"]["yolo_integrated"] + row["runtime_ms"]["rtmpose_head"]}),
                    "rtmw": prediction(row["rtmw"], crop, {"person_crop_provider": row["runtime_ms"]["yolo_integrated"], "pose_head": row["runtime_ms"]["rtmw_head"], "end_to_end": row["runtime_ms"]["yolo_integrated"] + row["runtime_ms"]["rtmw_head"]}),
                },
            }
        print(f"JHMDB {offset}/{len(jhmdb_by_video)} {samples[0]['sequence']} ({len(frames)} frames)", flush=True)

    for offset, sample in enumerate(packages["lsp"]["samples"], start=1):
        frame = cv2.imread(sample["media_path"])
        if frame is None:
            raise FileNotFoundError(sample["media_path"])
        row = infer_frame(frame, yolo, rtmpose, rtmw)
        crop = crop_diagnostics(sample, row["yolo_raw"])
        selected[sample["id"]] = {
            "dataset": "lsp", "image_shape": list(frame.shape[:2]),
            "models": {
                "yolo_raw": prediction(row["yolo_raw"], crop, {"integrated": row["runtime_ms"]["yolo_integrated"], "end_to_end": row["runtime_ms"]["yolo_integrated"]}),
                "yolo_filtered": prediction(row["yolo_raw"], crop, {"integrated": row["runtime_ms"]["yolo_integrated"], "filter": 0.0, "end_to_end": row["runtime_ms"]["yolo_integrated"]}, filter_mode="NOT_APPLICABLE_STATIC_PASS_THROUGH"),
                "rtmpose": prediction(row["rtmpose"], crop, {"person_crop_provider": row["runtime_ms"]["yolo_integrated"], "pose_head": row["runtime_ms"]["rtmpose_head"], "end_to_end": row["runtime_ms"]["yolo_integrated"] + row["runtime_ms"]["rtmpose_head"]}),
                "rtmw": prediction(row["rtmw"], crop, {"person_crop_provider": row["runtime_ms"]["yolo_integrated"], "pose_head": row["runtime_ms"]["rtmw_head"], "end_to_end": row["runtime_ms"]["yolo_integrated"] + row["runtime_ms"]["rtmw_head"]}),
            },
        }
        print(f"LSP {offset}/{len(packages['lsp']['samples'])} {sample['id']}", flush=True)
    return {"selected": selected, "sequences": sequences}


def angle(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> float:
    first = np.asarray([a["x"] - b["x"], a["y"] - b["y"]])
    second = np.asarray([c["x"] - b["x"], c["y"] - b["y"]])
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-6:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1, 1))))


def subset_names(dataset: str, sample: dict[str, Any]) -> list[str]:
    if dataset == "jhmdb":
        tags = set(sample.get("review_tags", []))
        names = ["ALL_JHMDB"]
        if sample["action"] == "shoot_ball": names.append("BASKETBALL")
        if sample["action"] in {"throw", "swing_baseball", "golf"}: names.append("THROWING_STRIKING")
        if sample["action"] == "jump": names.append("JUMPING")
        if tags & {"rapid_arm_motion", "arm_extension"}: names.append("FAST_ARM")
        if tags & {"difficult_articulation", "airborne", "hand_object_occlusion_candidate"}: names.append("DIFFICULT_POSE")
        return names
    joints = sample["mapped_joints"]
    names = ["ALL_LSP"]
    if any(joint["visibility"] == "occluded_but_inferable" for joint in joints.values()): names.append("OCCLUDED")
    else: names.append("VISIBLE")
    upper_difficult = any(
        joints[f"{side}_wrist"]["y"] < joints[f"{side}_shoulder"]["y"]
        or angle(joints[f"{side}_shoulder"], joints[f"{side}_elbow"], joints[f"{side}_wrist"]) < 100
        for side in ("left", "right")
    )
    lower_difficult = any(
        angle(joints[f"{side}_hip"], joints[f"{side}_knee"], joints[f"{side}_ankle"]) < 120
        for side in ("left", "right")
    )
    if upper_difficult: names.append("UPPER_BODY_DIFFICULT")
    if lower_difficult: names.append("LOWER_BODY_DIFFICULT")
    return names


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["pixel_error"] for row in rows if row["pixel_error"] is not None]
    normalized = [row["normalized_error"] for row in rows if row["normalized_error"] is not None]
    result: dict[str, Any] = {
        "labelable_joints": len(rows), "valid_predictions": len(errors),
        "coverage": round(len(errors) / len(rows), 4) if rows else None,
        "failure_rate": round(1 - len(errors) / len(rows), 4) if rows else None,
    }
    for name, values in (("pixel", errors), ("normalized", normalized)):
        result[f"mean_{name}_error"] = round(statistics.fmean(values), 4) if values else None
        result[f"median_{name}_error"] = round(statistics.median(values), 4) if values else None
        result[f"p90_{name}_error"] = round(percentile(values, 90), 4) if values else None
        result[f"p95_{name}_error"] = round(percentile(values, 95), 4) if values else None
    for threshold, label in ((0.05, "pck_005"), (0.10, "pck_010"), (0.20, "pck_020")):
        result[label] = round(sum(value <= threshold for value in normalized) / len(rows), 4) if rows else None
        result[f"{label}_conditional"] = round(sum(value <= threshold for value in normalized) / len(normalized), 4) if normalized else None
    return result


def build_records(packages: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset, package in packages.items():
        for sample in package["samples"]:
            scale = body_scale(sample["mapped_joints"])
            subsets = subset_names(dataset, sample)
            for model in MODELS:
                result = artifact["selected"][sample["id"]]["models"][model]
                pose = result["pose"] or {}
                xy, scores = pose.get("keypoints", []), pose.get("temporal_reliability", pose.get("confidence", []))
                for joint, truth in sample["mapped_joints"].items():
                    if truth["visibility"] == "not_labelable":
                        continue
                    index = JOINT_INDEX[joint]
                    available = bool(result["final_prediction_available"] and index < len(xy) and index < len(scores) and scores[index] >= CONFIDENCE_THRESHOLD)
                    error = math.dist(xy[index][:2], [truth["x"], truth["y"]]) if available else None
                    records[model].append({
                        "frame_id": sample["id"], "dataset": dataset, "action": sample["action"], "subsets": subsets,
                        "joint": joint, "joint_type": joint.split("_", 1)[1],
                        "body_group": "upper_body" if joint.endswith(("shoulder", "elbow", "wrist")) else "lower_body",
                        "visibility": truth["visibility"], "body_scale": scale, "detected": available,
                        "pixel_error": error, "normalized_error": error / scale if error is not None else None,
                        "gt": [truth["x"], truth["y"]], "prediction": xy[index][:2] if available else None,
                        "crop_success": result["crop_success"], "pose_head_success": result["pose_head_success"],
                    })
    return records


def grouped_accuracy(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output = {}
    for model, rows in records.items():
        groups: dict[str, list[dict[str, Any]]] = {"overall": rows}
        groups["overall_correct_crop"] = [row for row in rows if row["crop_success"]]
        for joint in ("wrist", "elbow", "shoulder", "hip", "knee", "ankle"):
            groups[joint] = [row for row in rows if row["joint_type"] == joint]
        groups["wrist_elbow"] = [row for row in rows if row["joint_type"] in {"wrist", "elbow"}]
        for body in ("upper_body", "lower_body"):
            groups[body] = [row for row in rows if row["body_group"] == body]
        for visibility in ("visible", "occluded_but_inferable"):
            groups[visibility] = [row for row in rows if row["visibility"] == visibility]
        for dataset in ("jhmdb", "lsp"):
            groups[f"dataset:{dataset}"] = [row for row in rows if row["dataset"] == dataset]
        for subset in sorted({name for row in rows for name in row["subsets"]}):
            groups[f"subset:{subset}"] = [row for row in rows if subset in row["subsets"]]
        output[model] = {name: metric_summary(group) for name, group in groups.items()}
    return output


def failure_summary(packages: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    output = {}
    total = sum(len(package["samples"]) for package in packages.values())
    for model in MODELS:
        rows = [artifact["selected"][sample["id"]]["models"][model] for package in packages.values() for sample in package["samples"]]
        output[model] = {
            "samples": total,
            "person_detection_failures": sum(not row["person_detection_success"] for row in rows),
            "crop_failures_including_wrong_person": sum(row["person_detection_success"] and not row["crop_success"] for row in rows),
            "pose_head_failures_on_correct_crop": sum(row["crop_success"] and not row["pose_head_success"] for row in rows),
            "final_predictions": sum(row["final_prediction_available"] for row in rows),
            "end_to_end_sample_coverage": round(sum(row["final_prediction_available"] for row in rows) / total, 4),
            "by_dataset": {
                dataset: {
                    "samples": len(package["samples"]),
                    "person_detection_failures": sum(not artifact["selected"][sample["id"]]["models"][model]["person_detection_success"] for sample in package["samples"]),
                    "crop_failures_including_wrong_person": sum(artifact["selected"][sample["id"]]["models"][model]["person_detection_success"] and not artifact["selected"][sample["id"]]["models"][model]["crop_success"] for sample in package["samples"]),
                    "pose_head_failures_on_correct_crop": sum(artifact["selected"][sample["id"]]["models"][model]["crop_success"] and not artifact["selected"][sample["id"]]["models"][model]["pose_head_success"] for sample in package["samples"]),
                }
                for dataset, package in packages.items()
            },
        }
    return output


def filter_damage(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw = {(row["frame_id"], row["joint"]): row for row in records["yolo_raw"]}
    filtered = {(row["frame_id"], row["joint"]): row for row in records["yolo_filtered"]}
    comparisons = []
    for key in sorted(raw.keys() & filtered.keys()):
        first, second = raw[key], filtered[key]
        a, b = first["pixel_error"], second["pixel_error"]
        delta = b - a if a is not None and b is not None else None
        if a is None and b is not None: outcome = "help"
        elif a is not None and b is None: outcome = "harm"
        elif a is None: outcome = "neutral"
        elif delta < -FILTER_NEUTRAL_PIXELS: outcome = "help"
        elif delta > FILTER_NEUTRAL_PIXELS: outcome = "harm"
        else: outcome = "neutral"
        comparisons.append({**second, "raw_error": a, "filtered_error": b, "delta_error": delta, "outcome": outcome})

    def summarize_damage(rows: list[dict[str, Any]]) -> dict[str, Any]:
        numeric = [row["delta_error"] for row in rows if row["delta_error"] is not None]
        return {
            "comparisons": len(rows), "paired_coordinates": len(numeric),
            "filter_help_rate": round(sum(row["outcome"] == "help" for row in rows) / len(rows), 4) if rows else None,
            "filter_harm_rate": round(sum(row["outcome"] == "harm" for row in rows) / len(rows), 4) if rows else None,
            "filter_neutral_rate": round(sum(row["outcome"] == "neutral" for row in rows) / len(rows), 4) if rows else None,
            "median_delta_error": round(statistics.median(numeric), 4) if numeric else None,
        }
    groups = {
        "overall": comparisons,
        "wrist_elbow": [row for row in comparisons if row["joint_type"] in {"wrist", "elbow"}],
        "difficult": [row for row in comparisons if "DIFFICULT_POSE" in row["subsets"] or "UPPER_BODY_DIFFICULT" in row["subsets"] or "LOWER_BODY_DIFFICULT" in row["subsets"]],
        "fast_arm": [row for row in comparisons if "FAST_ARM" in row["subsets"]],
    }
    for dataset in ("jhmdb", "lsp"):
        groups[f"dataset:{dataset}"] = [row for row in comparisons if row["dataset"] == dataset]
    for joint in ("wrist", "elbow", "shoulder", "hip", "knee", "ankle"):
        groups[f"joint:{joint}"] = [row for row in comparisons if row["joint_type"] == joint]
    return {
        "neutral_band_pixels": FILTER_NEUTRAL_PIXELS,
        "summary": {name: summarize_damage(rows) for name, rows in groups.items()},
        "comparisons": comparisons,
    }


def runtime_summary(packages: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for model in MODELS:
        buckets: dict[str, list[float]] = defaultdict(list)
        for package in packages.values():
            for sample in package["samples"]:
                for key, value in artifact["selected"][sample["id"]]["models"][model]["runtime_ms"].items():
                    buckets[key].append(float(value))
        output[model] = {
            key: {"mean_ms": round(statistics.fmean(values), 3), "median_ms": round(statistics.median(values), 3), "p95_ms": round(percentile(values, 95), 3)}
            for key, values in buckets.items()
        }
    return output


def temporal_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    per_sequence: dict[str, Any] = {}
    for sequence, value in artifact["sequences"].items():
        per_sequence[sequence] = {
            model: evaluate_pose_rows(rows, pose_key="pose")
            for model, rows in value["models"].items()
        }
    aggregate = {}
    for model in MODELS:
        keys = sorted({key for sequence in per_sequence.values() for key, value in sequence[model].items() if isinstance(value, (int, float)) and key != "frames"})
        aggregate[model] = {
            key: round(statistics.median([sequence[model][key] for sequence in per_sequence.values() if isinstance(sequence[model].get(key), (int, float))]), 5)
            for key in keys
        }
    return {"scope": "complete_contiguous_JHMDB_sequences_only", "aggregate_sequence_medians": aggregate, "per_sequence": per_sequence}


def sample_frame(sample: dict[str, Any]) -> np.ndarray:
    if sample["media_path"].lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(sample["media_path"])
    else:
        capture = cv2.VideoCapture(sample["media_path"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
        ok, frame = capture.read()
        capture.release()
        if not ok: frame = None
    if frame is None:
        raise FileNotFoundError(sample["media_path"])
    return frame


def draw_pose(image: np.ndarray, sample: dict[str, Any], pose: dict[str, Any] | None, title: str) -> np.ndarray:
    canvas = image.copy()
    gt = sample["mapped_joints"]
    for first, second in SKELETON:
        if gt[first]["visibility"] != "not_labelable" and gt[second]["visibility"] != "not_labelable":
            cv2.line(canvas, tuple(map(round, (gt[first]["x"], gt[first]["y"]))), tuple(map(round, (gt[second]["x"], gt[second]["y"]))), (40, 220, 80), 2, cv2.LINE_AA)
    for joint in gt.values():
        if joint["visibility"] != "not_labelable": cv2.circle(canvas, (round(joint["x"]), round(joint["y"])), 4, (40, 220, 80), -1, cv2.LINE_AA)
    if pose:
        xy, scores = pose.get("keypoints", []), pose.get("temporal_reliability", pose.get("confidence", []))
        for first, second in SKELETON:
            a, b = JOINT_INDEX[first], JOINT_INDEX[second]
            if a < len(scores) and b < len(scores) and scores[a] >= CONFIDENCE_THRESHOLD and scores[b] >= CONFIDENCE_THRESHOLD:
                cv2.line(canvas, tuple(map(round, xy[a][:2])), tuple(map(round, xy[b][:2])), (40, 80, 255), 2, cv2.LINE_AA)
        for index in JOINT_INDEX.values():
            if index < len(scores) and scores[index] >= CONFIDENCE_THRESHOLD: cv2.circle(canvas, tuple(map(round, xy[index][:2])), 3, (40, 80, 255), -1, cv2.LINE_AA)
    output = np.zeros((320, 320, 3), dtype=np.uint8)
    scale = min(320 / canvas.shape[1], 292 / canvas.shape[0])
    resized = cv2.resize(canvas, (max(1, round(canvas.shape[1] * scale)), max(1, round(canvas.shape[0] * scale))))
    x = (320 - resized.shape[1]) // 2
    y = 28 + (292 - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(output, title[:47], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def make_review(packages: dict[str, dict[str, Any]], artifact: dict[str, Any], accuracy: dict[str, Any], damage: dict[str, Any], failures: dict[str, Any], review_root: Path) -> None:
    media = review_root / "samples"
    media.mkdir(parents=True, exist_ok=True)
    samples_by_id = {sample["id"]: sample for package in packages.values() for sample in package["samples"]}
    cards = []
    for sample_id, sample in samples_by_id.items():
        frame = sample_frame(sample)
        panels = [draw_pose(frame, sample, None, "HUMAN_GT (green)")]
        for model in MODELS:
            result = artifact["selected"][sample_id]["models"][model]
            panels.append(draw_pose(frame, sample, result["pose"], f"{model} (red) | final={result['final_prediction_available']}"))
        composite = np.hstack(panels)
        name = f"{hashlib.sha1(sample_id.encode()).hexdigest()[:16]}.jpg"
        cv2.imwrite(str(media / name), composite)
        cards.append(f'<article><img loading="lazy" src="samples/{name}"><h3>{html.escape(sample_id)}</h3><p>{html.escape(sample["action"])} · {html.escape(", ".join(subset_names(artifact["selected"][sample_id]["dataset"], sample)))}</p></article>')

    damage_root = review_root / "filter_damage"
    for label in ("raw_better", "filtered_better", "neutral"):
        (damage_root / label).mkdir(parents=True, exist_ok=True)
    ordering = {
        "raw_better": sorted((row for row in damage["comparisons"] if row["outcome"] == "harm"), key=lambda row: row["delta_error"] if row["delta_error"] is not None else math.inf, reverse=True),
        "filtered_better": sorted((row for row in damage["comparisons"] if row["outcome"] == "help"), key=lambda row: row["delta_error"] if row["delta_error"] is not None else -math.inf),
        "neutral": [row for row in damage["comparisons"] if row["outcome"] == "neutral"],
    }
    damage_links = []
    for label, rows in ordering.items():
        seen = set()
        for row in rows:
            if row["frame_id"] in seen or len(seen) >= 8: continue
            seen.add(row["frame_id"])
            sample = samples_by_id[row["frame_id"]]
            frame = sample_frame(sample)
            model_rows = artifact["selected"][row["frame_id"]]["models"]
            panels = [draw_pose(frame, sample, None, "GT"), draw_pose(frame, sample, model_rows["yolo_raw"]["pose"], "RAW"), draw_pose(frame, sample, model_rows["yolo_filtered"]["pose"], "FILTERED")]
            name = f"{row['frame_id']}.jpg"
            cv2.imwrite(str(damage_root / label / name), np.hstack(panels))
        damage_links.append(f'<li><a href="filter_damage/{label}/">{label}</a> — {len(seen)} representative frames</li>')

    def table(rows: Iterable[tuple[str, dict[str, Any]]]) -> str:
        body = "".join(f"<tr><td>{html.escape(name)}</td><td>{value.get('coverage')}</td><td>{value.get('median_pixel_error')}</td><td>{value.get('p95_pixel_error')}</td><td>{value.get('pck_010')}</td></tr>" for name, value in rows)
        return f"<table><tr><th>model/group</th><th>coverage</th><th>median px</th><th>P95 px</th><th>PCK@0.10 e2e</th></tr>{body}</table>"
    overview = table((model, accuracy[model]["overall"]) for model in MODELS)
    joint_rows = table((f"{model}/{joint}", accuracy[model][joint]) for model in MODELS for joint in ("wrist", "elbow", "shoulder", "lower_body"))
    failure_rows = "".join(f"<tr><td>{model}</td><td>{row['person_detection_failures']}</td><td>{row['crop_failures_including_wrong_person']}</td><td>{row['pose_head_failures_on_correct_crop']}</td><td>{row['end_to_end_sample_coverage']}</td></tr>" for model, row in failures.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Public-GT Pose Benchmark</title><style>body{{font:14px system-ui;background:#101114;color:#eee;margin:24px}}a{{color:#65bfff}}table{{border-collapse:collapse;margin:12px 0}}th,td{{border:1px solid #555;padding:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:14px}}article{{background:#1b1d21;padding:10px}}img{{width:100%;background:#000}}p{{color:#bbb}}</style></head><body><h1>Accepted HUMAN_GT pose benchmark</h1><p>Green = accepted GT; red = prediction. Accuracy, temporal quality, runtime and failure are separate outputs. LSP filtered is a declared static pass-through.</p><h2>Overall localization</h2>{overview}<h2>Priority joints</h2>{joint_rows}<h2>Pipeline failures (66 samples)</h2><table><tr><th>model</th><th>detection fail</th><th>wrong/bad crop</th><th>pose-head fail on correct crop</th><th>final coverage</th></tr>{failure_rows}</table><h2>Filter evidence</h2><p>Neutral band ±{FILTER_NEUTRAL_PIXELS}px.</p><ul>{''.join(damage_links)}</ul><h2>All samples</h2><div class="grid">{''.join(cards)}</div></body></html>"""
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "index.html").write_text(page, encoding="utf-8")


def sanity(packages: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    checks = {"samples": 0, "labelable_joints": 0, "dimension_matches": 0, "gt_in_bounds": 0, "left_right_source_mapping": True, "rtm_outputs_in_original_coordinate_space": True}
    for dataset, package in packages.items():
        for sample in package["samples"]:
            checks["samples"] += 1
            shape = artifact["selected"][sample["id"]]["image_shape"]
            if shape == [sample["height"], sample["width"]]: checks["dimension_matches"] += 1
            for name, joint in sample["mapped_joints"].items():
                if joint["visibility"] == "not_labelable": continue
                checks["labelable_joints"] += 1
                if 0 <= joint["x"] < sample["width"] and 0 <= joint["y"] < sample["height"]: checks["gt_in_bounds"] += 1
                if joint["source_name"] != name: checks["left_right_source_mapping"] = False
            for model in ("rtmpose", "rtmw"):
                pose = artifact["selected"][sample["id"]]["models"][model]["pose"]
                if pose and any(not (-sample["width"] <= point[0] <= 2 * sample["width"] and -sample["height"] <= point[1] <= 2 * sample["height"]) for point in pose["keypoints"][:17]):
                    checks["rtm_outputs_in_original_coordinate_space"] = False
    checks["passed"] = checks["dimension_matches"] == checks["samples"] and checks["gt_in_bounds"] == checks["labelable_joints"] and checks["left_right_source_mapping"] and checks["rtm_outputs_in_original_coordinate_space"]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark four pose pipelines against accepted public HUMAN_GT")
    parser.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reuse-inference", action="store_true")
    args = parser.parse_args()
    if args.public_root.resolve().drive.upper() != "E:" or args.output_root.resolve().drive.upper() != "E:":
        raise ValueError("Public data, caches and benchmark outputs must remain on E:")
    configure_large_data_caches(args.public_root)
    packages = {dataset: read_json(args.public_root / "dataset_review" / dataset / "normalized_gt.candidate.json") for dataset in ("jhmdb", "lsp")}
    for dataset, package in packages.items():
        if package.get("ground_truth_type") != "HUMAN_GT" or package.get("user_dataset_review") != "ACCEPTED":
            raise PermissionError(f"{dataset} is not accepted HUMAN_GT")
    args.output_root.mkdir(parents=True, exist_ok=True)
    inference_path = args.output_root / "inference.v1.json"
    artifact = read_json(inference_path) if args.reuse_inference and inference_path.exists() else run_inference(packages, ROOT / "yolo11n-pose.pt", args.output_root)
    if not inference_path.exists() or not args.reuse_inference:
        write_json(inference_path, artifact)
    records = build_records(packages, artifact)
    accuracy = grouped_accuracy(records)
    damage = filter_damage(records)
    failures = failure_summary(packages, artifact)
    runtime = runtime_summary(packages, artifact)
    temporal = temporal_summary(artifact)
    sample_counts = {dataset: {"samples": len(package["samples"]), "subsets": {name: sum(name in subset_names(dataset, sample) for sample in package["samples"]) for name in sorted({value for sample in package["samples"] for value in subset_names(dataset, sample)})}} for dataset, package in packages.items()}
    checkpoint_root = args.public_root / "cache" / "rtmlib" / "hub" / "checkpoints"
    model_files = [
        ROOT / "yolo11n-pose.pt",
        checkpoint_root / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx",
        checkpoint_root / "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
    ]
    summary = {
        "schema_version": "public_pose_benchmark_v1", "created": "2026-09-02",
        "manifest_fingerprint": hashlib.sha256("".join(sample["id"] for package in packages.values() for sample in package["samples"]).encode()).hexdigest(),
        "sample_counts": sample_counts, "model_files": [{"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in model_files],
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "opencv": cv2.__version__, "compute": "CPU; torch build has no CUDA and ONNX Runtime exposes CPU/Azure providers"},
        "sanity": sanity(packages, artifact), "accuracy": accuracy, "failures": failures, "filter_damage": {key: value for key, value in damage.items() if key != "comparisons"},
        "runtime": runtime, "temporal_quality": temporal,
        "method_notes": [
            "All 66 accepted samples are shared by all four pipelines.",
            "JHMDB filtering and temporal metrics use complete contiguous source clips; only frozen manifest frames enter localization accuracy.",
            "LSP filtered output is an explicit static pass-through and is neutral in paired filter damage.",
            "A crop is correct when at least 75% of labelable HUMAN_GT core joints fall inside the selected YOLO box.",
            "RTMPose/RTMW person_crop_provider time is integrated YOLO11n-pose time, not a separately instrumented detector-only head; it conservatively overstates a future bbox-only detector.",
            "PCK without the conditional suffix uses every labelable joint as denominator, so missing/wrong-person failures remain failures.",
        ],
    }
    write_json(args.output_root / "summary.v1.json", summary)
    write_json(args.output_root / "joint_records.v1.json", records)
    write_json(args.output_root / "filter_damage.v1.json", damage)
    make_review(packages, artifact, accuracy, damage, failures, args.output_root / "review")
    print(args.output_root / "summary.v1.json")
    print(args.output_root / "review" / "index.html")


if __name__ == "__main__":
    main()
