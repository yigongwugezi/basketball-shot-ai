"""Create conservative, explicitly non-manual pseudo labels for batch002 only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\20825\Documents\Codex\2026-05-21\1-2-ai-app-3-ai\tmp\release_ball_annotation_batch_002")
OUT_ROOT = ROOT / "datasets" / "annotations" / "release_ball_v3a_pseudo"
V2_WEIGHT = ROOT / "models" / "release_ball_v2" / "best.pt"
TEACHER_WEIGHT = ROOT / "yolo11n.pt"
V2_CONFIDENCE = 0.15  # Frozen release_ball_v2 inference threshold.
TEACHER_CONFIDENCE = 0.15
MIN_CROSS_MODEL_IOU = 0.30
MAX_NEIGHBOR_CENTER_DISTANCE_DIAGONAL = 0.20
EXCLUDED_SOURCES = {"IMG_7221.MOV", "IMG_7222.MP4"}


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def candidates(result, allowed_class: int) -> list[dict[str, object]]:
    rows = []
    for box in result.boxes:
        if int(box.cls[0].item()) != allowed_class:
            continue
        rows.append({"bbox": [round(float(value), 4) for value in box.xyxy[0].tolist()], "confidence": round(float(box.conf[0].item()), 6)})
    return rows


def agreement(v2: list[dict[str, object]], teacher: list[dict[str, object]]) -> tuple[dict[str, object] | None, float]:
    chosen, best = None, 0.0
    for left in v2:
        for right in teacher:
            score = iou(left["bbox"], right["bbox"])
            if score > best:
                chosen, best = {"v2": left, "teacher": right}, score
    return chosen, best


def temporal_support(rows: list[dict[str, object]], index: int) -> bool:
    item = rows[index]
    if not item["agreement"]:
        return False
    box = item["agreement"]["v2"]["bbox"]
    diagonal = math.hypot(item["width"], item["height"])
    for neighbor_index in (index - 1, index + 1):
        if not 0 <= neighbor_index < len(rows):
            continue
        neighbor = rows[neighbor_index]
        if not neighbor["agreement"]:
            continue
        other = neighbor["agreement"]["v2"]["bbox"]
        x, y = center(box)
        ox, oy = center(other)
        if math.hypot(x - ox, y - oy) <= diagonal * MAX_NEIGHBOR_CENTER_DISTANCE_DIAGONAL:
            return True
    return False


def adjacent_agreement(rows: list[dict[str, object]], index: int) -> bool:
    return any(0 <= neighbor_index < len(rows) and rows[neighbor_index]["agreement"] for neighbor_index in (index - 1, index + 1))


def main() -> None:
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite pseudo-label run: {OUT_ROOT}")
    templates = sorted(SOURCE_ROOT.glob("*/annotation_template.csv"))
    source_rows = []
    for template in templates:
        with template.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                frame = template.parent / "frames" / row["image_file"]
                if not frame.is_file():
                    raise FileNotFoundError(frame)
                source_rows.append({"clip_id": row["clip_id"], "frame_index": int(row["frame_index"]), "image_path": frame})
    if len(source_rows) != 217 or any(name in str(row["image_path"]) for row in source_rows for name in EXCLUDED_SOURCES):
        raise RuntimeError("unexpected batch002 inventory or excluded source")

    v2, teacher = YOLO(str(V2_WEIGHT)), YOLO(str(TEACHER_WEIGHT))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in source_rows:
        grouped.setdefault(row["clip_id"], []).append(row)
    for clip_rows in grouped.values():
        paths = [str(row["image_path"]) for row in clip_rows]
        v2_results = v2.predict(paths, imgsz=640, conf=V2_CONFIDENCE, verbose=False)
        teacher_results = teacher.predict(paths, imgsz=640, conf=TEACHER_CONFIDENCE, classes=[32], verbose=False)
        for row, v2_result, teacher_result in zip(clip_rows, v2_results, teacher_results, strict=True):
            v2_items = candidates(v2_result, 0)
            teacher_items = candidates(teacher_result, 32)
            pair, cross_iou = agreement(v2_items, teacher_items)
            height, width = v2_result.orig_shape
            row.update({"width": width, "height": height, "v2_candidates": v2_items, "teacher_candidates": teacher_items, "agreement": pair if cross_iou >= MIN_CROSS_MODEL_IOU else None, "cross_model_iou": round(cross_iou, 6)})

    output = []
    for clip_rows in grouped.values():
        for index, row in enumerate(clip_rows):
            supported = temporal_support(clip_rows, index)
            if row["agreement"] and supported:
                label, reason = "pseudo_positive", "v2_teacher_bbox_agreement_and_neighbor_trajectory"
                bbox = row["agreement"]["v2"]["bbox"]
            elif not row["v2_candidates"] and not row["teacher_candidates"] and not adjacent_agreement(clip_rows, index):
                label, reason, bbox = "pseudo_negative", "both_models_no_candidate_and_no_neighbor_trajectory", None
            else:
                label, reason, bbox = "unresolved", "insufficient_cross_model_or_temporal_evidence", None
            output.append({"clip_id": row["clip_id"], "frame_index": row["frame_index"], "image_path": str(row["image_path"]), "label_type": label, "bbox": bbox, "label_source": "pseudo_release_ball_v2_plus_coco_yolo11n", "v2_candidates": row["v2_candidates"], "teacher_sports_ball_candidates": row["teacher_candidates"], "cross_model_iou": row["cross_model_iou"], "temporal_support": supported, "reason": reason})

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "pseudo_labels.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    counts = {label: sum(row["label_type"] == label for row in output) for label in ("pseudo_positive", "pseudo_negative", "unresolved")}
    manifest = {"version": "release_ball_v3a_pseudo", "manual_gt": False, "input_frames": len(output), "models": {"release_ball_v2": {"path": str(V2_WEIGHT), "confidence": V2_CONFIDENCE}, "coco_yolo11n_teacher": {"path": str(TEACHER_WEIGHT), "class": "sports ball", "confidence": TEACHER_CONFIDENCE}}, "acceptance_rule": {"min_cross_model_iou": MIN_CROSS_MODEL_IOU, "max_neighbor_center_distance_diagonal": MAX_NEIGHBOR_CENTER_DISTANCE_DIAGONAL}, "excluded_sources": sorted(EXCLUDED_SOURCES), "counts": counts}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
