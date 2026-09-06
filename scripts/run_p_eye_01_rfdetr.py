#!/usr/bin/env python3
"""Run frozen RF-DETR baseline inference for the held-out P-EYE-01 frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2


BENCHMARK = Path(r"E:\BasketballShotAI\benchmarks\P-EYE-01\CANONICAL_EYE_TEST_01")
MODEL_CACHE = Path(r"E:\BasketballShotAI\model_zoo\P-EYE-01\RF-DETR")
DEFAULT_THRESHOLD = 0.5
# RF-DETR returns native, sparse COCO category IDs.  Sports ball is category 37
# (whereas Ultralytics' contiguous class index is 32).
SPORTS_BALL_COCO_CLASS = 37
WEIGHT_FILENAMES = {
    "medium": "rf-detr-medium.pth",
    "large": "rf-detr-large-2026.pth",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(count: int) -> str:
    return "MISSING" if count == 0 else "DETECTED" if count == 1 else "MULTIPLE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("medium", "large"), default="medium")
    args = parser.parse_args()
    # P-EYE assets must remain outside the system drive, even if the shell has
    # inherited a different Roboflow cache location.
    os.environ["RF_HOME"] = str(MODEL_CACHE)

    from rfdetr import RFDETRLarge, RFDETRMedium

    manifest = json.loads((BENCHMARK / "canonical_manifest.json").read_text(encoding="utf-8"))
    weights = MODEL_CACHE / WEIGHT_FILENAMES[args.variant]
    if not weights.is_file():
        raise FileNotFoundError(f"Frozen local checkpoint is unavailable: {weights}")
    model_type = RFDETRMedium if args.variant == "medium" else RFDETRLarge
    model = model_type(pretrain_weights=str(weights), trust_checkpoint=True)
    frame_paths = sorted((BENCHMARK / "frames").glob("frame_*.jpg"))
    if len(frame_paths) != manifest["frame_count"]:
        raise RuntimeError("Canonical frame inventory is incomplete")

    started = time.perf_counter()
    frames = []
    candidate_count = detected = multiple = 0
    for frame_index, frame_path in enumerate(frame_paths):
        prediction = model.predict(str(frame_path), threshold=DEFAULT_THRESHOLD)
        detections = []
        for xyxy, confidence, class_id in zip(prediction.xyxy, prediction.confidence, prediction.class_id):
            if int(class_id) != SPORTS_BALL_COCO_CLASS:
                continue
            x1, y1, x2, y2 = [round(float(value), 3) for value in xyxy]
            detections.append(
                {
                    "center_x": round((x1 + x2) / 2, 3),
                    "center_y": round((y1 + y2) / 2, 3),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": round(float(confidence), 6),
                    "class_id": SPORTS_BALL_COCO_CLASS,
                    "class_name": "sports ball",
                    "source_model": f"rf_detr_{args.variant}",
                    "evidence_type": "OBSERVED",
                }
            )
        frame_status = status(len(detections))
        detected += bool(detections)
        multiple += len(detections) > 1
        candidate_count += len(detections)
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp": round(frame_index / manifest["fps"], 6),
                "status": frame_status,
                "detections": detections,
            }
        )
    elapsed = time.perf_counter() - started
    payload = {
        "contract_version": "P-EYE-DETECTOR-OBSERVATION-V1",
        "benchmark_id": manifest["benchmark_id"],
        "source_video_sha256": manifest["source_video_sha256"],
        "model": {
            "id": f"rf_detr_{args.variant}",
            "source_model": f"RF-DETR {args.variant.title()} COCO (RF-DETR {args.variant.title()} COCO 目标检测器)",
            "weights_path": str(weights),
            "weights_sha256": sha256(weights),
            "inference_variant": "official default full-frame inference (官方默认整帧推理)",
            "confidence_threshold": DEFAULT_THRESHOLD,
            "threshold_origin": "official default; not tuned on canonical test",
            "device": "cpu",
            "target_class": {"id": SPORTS_BALL_COCO_CLASS, "name": "sports ball"},
        },
        "summary": {
            "frames": len(frames),
            "detected_frames": detected,
            "missing_frames": len(frames) - detected,
            "multiple_frames": multiple,
            "candidate_count": candidate_count,
            "elapsed_s": round(elapsed, 4),
            "latency_ms_per_frame": round(1000 * elapsed / len(frames), 3),
        },
        "frames": frames,
    }
    destination = BENCHMARK / "predictions" / f"rf_detr_{args.variant}_official_default.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(destination), **payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
