from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.reference_v1.pose_gt import JOINT_INDEX, evaluate_predictions, validate_annotations, validate_manifest  # noqa: E402
from reference_v1.pose.metrics import evaluate_pose_rows  # noqa: E402
from reference_v1.pose.reliability import build_analysis_pose  # noqa: E402


DEFAULT_MANIFEST = Path(__file__).with_name("pose_gt_manifest.v1.json")
DEFAULT_OUTPUT = Path(r"E:\BasketballShotAI\analysis_runs\pose_accuracy_closure\pose_gt_v1")
CACHE_ROOT = Path(r"E:\BasketballShotAI\analysis_runs\pose_reliability_pass\cache")
TEMPLATE = Path(__file__).with_name("pose_gt_annotation.html")
SAMPLE_CACHE_IDS = {
    "img_7215": "img_7215_release_drive",
    "img_7216": "img_7216_release_drive",
    "bili_005_a": "bili_005_a_difficult_release",
}
MODEL_SOURCES = {
    "yolo_raw": ("yolo_raw", "raw_pose"),
    "yolo_filtered": ("yolo_raw", "analysis_pose"),
    "rtmpose": ("rtmpose_crop", "pose"),
    "rtmw": ("rtmw_crop", "pose"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(manifest_path: Path, output: Path) -> None:
    manifest = load_json(manifest_path)
    validate_manifest(manifest)
    images = output / "frames"
    predictions_dir = output / "model_predictions"
    annotations_dir = output / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for clip_id, clip in manifest["clips"].items():
        frame_map = {frame["frame_index"]: frame for frame in manifest["frames"] if frame["clip"] == clip_id}
        capture = cv2.VideoCapture(str(Path(clip["video"])))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open source video: {clip['video']}")
        for frame_index, meta in frame_map.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode {meta['id']}")
            path = images / f"{meta['id']}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"Could not write {path}")
        capture.release()
    write_overview_contact_sheet(manifest, images, output / "candidate_frames_contact_sheet.jpg")

    prediction_payloads = build_prediction_payloads(manifest)
    for payload in prediction_payloads:
        write_json(predictions_dir / f"{payload['model_id']}.json", payload)
    write_model_comparison_sheet(manifest, images, prediction_payloads, output / "model_predictions_contact_sheet.jpg")

    starter = {
        "schema_version": "pose_gt_annotations_v1",
        "benchmark_version": manifest["benchmark_version"],
        "annotator": "",
        "created_at": None,
        "updated_at": None,
        "revision": 1,
        "notes": "Human review required. Model predictions are stored separately.",
        "frames": [blank_annotation(frame) for frame in manifest["frames"]],
    }
    starter_path = annotations_dir / "pose_gt.json"
    if not starter_path.exists():
        write_json(starter_path, starter)

    package = {
        "manifest": manifest,
        "predictions": {payload["model_id"]: payload for payload in prediction_payloads},
        "image_prefix": "frames/",
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__POSE_GT_PACKAGE__*/", json.dumps(package, ensure_ascii=False))
    (output / "annotate.html").write_text(html, encoding="utf-8")
    shutil.copyfile(manifest_path, output / "manifest.json")
    (output / "README.txt").write_text(
        "POSE_GT_CLOSURE = WAITING_FOR_HUMAN_LABELS\n\n"
        "1. Open annotate.html in Edge/Chrome.\n"
        "2. Enter annotator name. For each frame press B to copy the displayed prediction, correct joints, set visibility, then mark Reviewed.\n"
        "3. Use Save JSON and replace annotations/pose_gt.json.\n"
        "4. Run:\n"
        f"   .\\.venv310\\Scripts\\python.exe benchmarks\\reference_v1\\pose_gt_benchmark.py evaluate --output \"{output}\"\n",
        encoding="utf-8",
    )
    print(f"POSE_GT_PACKAGE_READY = YES ({len(manifest['frames'])} frames)")
    print(f"annotation UI: {output / 'annotate.html'}")
    print("POSE_GT_CLOSURE = WAITING_FOR_HUMAN_LABELS")


def write_overview_contact_sheet(manifest: dict[str, Any], images: Path, path: Path) -> None:
    tiles = []
    for meta in manifest["frames"]:
        image = cv2.imread(str(images / f"{meta['id']}.jpg"))
        if image is None:
            continue
        scale = min(230 / image.shape[1], 330 / image.shape[0])
        tile = cv2.resize(image, None, fx=scale, fy=scale)
        cv2.putText(tile, meta["id"], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        return
    width, height = max(tile.shape[1] for tile in tiles), max(tile.shape[0] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, width - tile.shape[1], cv2.BORDER_CONSTANT) for tile in tiles]
    columns = 7
    while len(padded) % columns:
        padded.append(np.zeros((height, width, 3), dtype=np.uint8))
    sheet = np.vstack([np.hstack(padded[index : index + columns]) for index in range(0, len(padded), columns)])
    cv2.imwrite(str(path), sheet)


def write_model_comparison_sheet(manifest: dict[str, Any], images: Path, payloads: list[dict[str, Any]], path: Path) -> None:
    models = ["yolo_raw", "yolo_filtered", "rtmpose", "rtmw"]
    lookup = {payload["model_id"]: {frame["frame_id"]: frame for frame in payload["frames"]} for payload in payloads}
    selected = [frame for frame in manifest["frames"] if {"motion_blur", "identity_ambiguity"} & set(frame["tags"])]
    rows = []
    skeleton = (("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"))
    for meta in selected[:8]:
        image = cv2.imread(str(images / f"{meta['id']}.jpg"))
        tiles = []
        for model in models:
            scale = min(240 / image.shape[1], 340 / image.shape[0])
            tile = cv2.resize(image, None, fx=scale, fy=scale)
            joints = lookup[model][meta["id"]]["joints"]
            for first, second in skeleton:
                a, b = joints.get(first), joints.get(second)
                if a and b and a["confidence"] >= 0.25 and b["confidence"] >= 0.25:
                    cv2.line(tile, (round(a["x"] * scale), round(a["y"] * scale)), (round(b["x"] * scale), round(b["y"] * scale)), (80, 255, 150), 2)
            for joint in joints.values():
                if joint["confidence"] >= 0.25:
                    cv2.circle(tile, (round(joint["x"] * scale), round(joint["y"] * scale)), 3, (0, 210, 255), -1)
            cv2.putText(tile, f"{meta['id']} | {model}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            tiles.append(tile)
        height = max(tile.shape[0] for tile in tiles)
        tiles = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT) for tile in tiles]
        rows.append(np.hstack(tiles))
    if rows:
        width = max(row.shape[1] for row in rows)
        rows = [cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT) for row in rows]
        cv2.imwrite(str(path), np.vstack(rows))


def build_prediction_payloads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {(frame["clip"], frame["frame_index"]): frame["id"] for frame in manifest["frames"]}
    rows_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {model: {} for model in MODEL_SOURCES}
    output_frames: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_SOURCES}
    runtime: dict[str, list[float]] = {model: [] for model in MODEL_SOURCES}
    for clip_id, cache_id in SAMPLE_CACHE_IDS.items():
        raw_data = load_json(CACHE_ROOT / "yolo_raw" / f"{cache_id}.json")
        raw_rows = [{**row, "raw_pose": row.get("pose")} for row in raw_data["rows"]]
        filtered_rows = build_analysis_pose(raw_rows)
        data_by_model = {
            "yolo_raw": raw_rows,
            "yolo_filtered": filtered_rows,
            "rtmpose": load_json(CACHE_ROOT / "rtmpose_crop" / f"{cache_id}.json")["rows"],
            "rtmw": load_json(CACHE_ROOT / "rtmw_crop" / f"{cache_id}.json")["rows"],
        }
        for model_id, rows in data_by_model.items():
            pose_key = MODEL_SOURCES[model_id][1]
            rows_by_model[model_id][clip_id] = rows
            for row in rows:
                if row.get("runtime_ms") is not None:
                    runtime[model_id].append(float(row["runtime_ms"]))
                frame_id = selected.get((clip_id, int(row["frame_index"])))
                if frame_id:
                    output_frames[model_id].append(prediction_frame(frame_id, row.get(pose_key)))

    payloads = []
    for model_id, frames in output_frames.items():
        dependency = "integrated_yolo_detector_and_pose_head"
        preprocessing = "Ultralytics letterbox, imgsz=640, conf=0.20, original-pixel output"
        if model_id in {"rtmpose", "rtmw"}:
            dependency = "YOLO11 largest-person bbox; pose-head/crop comparison, not independent detector comparison"
            preprocessing = "RTMLib balanced ONNX pose head on YOLO11 bbox; output mapped to original pixels"
        elif model_id == "yolo_filtered":
            preprocessing += "; Reference V1 temporal reliability post-processing"
        payloads.append(
            {
                "schema_version": "pose_model_predictions_v1",
                "benchmark_version": manifest["benchmark_version"],
                "model_id": model_id,
                "coordinate_system": manifest["coordinate_system"],
                "bbox_dependency": dependency,
                "preprocessing": preprocessing,
                "confidence_threshold": 0.25,
                "mean_runtime_ms": round(sum(runtime[model_id]) / len(runtime[model_id]), 3) if runtime[model_id] else None,
                "frames": sorted(frames, key=lambda item: item["frame_id"]),
                "temporal_quality": {
                    clip_id: evaluate_pose_rows(rows, pose_key=MODEL_SOURCES[model_id][1])
                    for clip_id, rows in rows_by_model[model_id].items()
                },
            }
        )
    return payloads


def prediction_frame(frame_id: str, pose: dict[str, Any] | None) -> dict[str, Any]:
    if not pose:
        return {"frame_id": frame_id, "status": "not_detected", "joints": {}}
    points = pose.get("keypoints", [])
    confidence = pose.get("temporal_reliability", pose.get("confidence", []))
    joints = {}
    for name, index in JOINT_INDEX.items():
        if index < len(points) and index < len(confidence):
            joints[name] = {
                "x": round(float(points[index][0]), 4),
                "y": round(float(points[index][1]), 4),
                "confidence": round(float(confidence[index]), 6),
            }
    return {"frame_id": frame_id, "status": "ok" if joints else "not_detected", "joints": joints}


def blank_annotation(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_id": frame["id"],
        "reviewed": False,
        "reviewed_at": None,
        "notes": "",
        "joints": {name: {"x": None, "y": None, "visibility": "not_labelable"} for name in JOINT_INDEX},
    }


def evaluate(output: Path) -> None:
    manifest = load_json(output / "manifest.json")
    annotations_path = output / "annotations" / "pose_gt.json"
    annotations = load_json(annotations_path)
    try:
        validate_annotations(manifest, annotations, require_complete=True)
    except ValueError as exc:
        print(f"POSE_GT_CLOSURE = WAITING_FOR_HUMAN_LABELS ({exc})")
        raise SystemExit(2) from exc
    predictions = [load_json(path) for path in sorted((output / "model_predictions").glob("*.json"))]
    result = evaluate_predictions(manifest, annotations, predictions)
    write_json(output / "results" / "pose_accuracy_benchmark.json", result)
    write_filter_contact_sheets(output, manifest, annotations, predictions, result)
    print(output / "results" / "pose_accuracy_benchmark.json")
    print("POSE_GT_CLOSURE = COMPLETE")


def write_filter_contact_sheets(
    output: Path,
    manifest: dict[str, Any],
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    gt = {frame["frame_id"]: frame for frame in annotations["frames"]}
    pred = {payload["model_id"]: {frame["frame_id"]: frame for frame in payload["frames"]} for payload in predictions}
    for label, key in (("raw_better", "raw_better_examples"), ("filtered_better", "filtered_better_examples")):
        examples = result["filter_damage"][key][:8]
        tiles = []
        for item in examples:
            image = cv2.imread(str(output / "frames" / f"{item['frame_id']}.jpg"))
            if image is None:
                continue
            scale = min(520 / image.shape[1], 680 / image.shape[0])
            canvas = cv2.resize(image, None, fx=scale, fy=scale)
            name = item["joint"]
            colors = (("GT", gt[item["frame_id"]]["joints"].get(name), (255, 255, 255)), ("RAW", pred["yolo_raw"][item["frame_id"]]["joints"].get(name), (0, 210, 255)), ("FILTER", pred["yolo_filtered"][item["frame_id"]]["joints"].get(name), (80, 255, 130)))
            for text_label, point, color in colors:
                if point and point.get("x") is not None:
                    cv2.circle(canvas, (round(point["x"] * scale), round(point["y"] * scale)), 7, color, 2)
            delta = f"d={item['delta_error']:+.1f}px" if item["delta_error"] is not None else "d=missing transition"
            cv2.putText(canvas, f"{item['frame_id']} {name} {delta}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            tiles.append(canvas)
        if tiles:
            width = max(tile.shape[1] for tile in tiles)
            height = max(tile.shape[0] for tile in tiles)
            padded = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, width - tile.shape[1], cv2.BORDER_CONSTANT) for tile in tiles]
            while len(padded) < 8:
                padded.append(np.zeros((height, width, 3), dtype=np.uint8))
            sheet = np.vstack([np.hstack(padded[:4]), np.hstack(padded[4:8])])
            cv2.imwrite(str(output / "results" / f"{label}_contact_sheet.jpg"), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or evaluate the Reference V1 human pose GT benchmark")
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
