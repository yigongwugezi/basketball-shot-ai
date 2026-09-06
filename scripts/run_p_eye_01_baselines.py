#!/usr/bin/env python3
"""Run the two frozen P-EYE-01 detector baselines without scoring or tuning."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path(r"E:\BasketballShotAI\benchmarks\P-EYE-01\CANONICAL_EYE_TEST_01")
MODELS = (
    {
        "id": "current_general_ball",
        "path": ROOT / "yolo11n.pt",
        "confidence_threshold": 0.10,
        "classes": [0, 32],
        "output_classes": [32],
        "source_model": "YOLO11n COCO sports ball (YOLO11n COCO 运动球类)",
        "training_relation": "public COCO pretrained; canonical test not used",
    },
    {
        "id": "release_ball_v2",
        "path": ROOT / "models/release_ball_v2/best.pt",
        "confidence_threshold": 0.15,
        "classes": [0],
        "output_classes": [0],
        "source_model": "release_ball_v2 (释放球检测器第二版)",
        "training_relation": "frozen internal training set; canonical test excluded",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def status(count: int) -> str:
    return "MISSING" if count == 0 else "DETECTED" if count == 1 else "MULTIPLE"


def main() -> None:
    manifest = json.loads((BENCHMARK / "canonical_manifest.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(manifest["source_video"])
    source_frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_frames.append(frame)
    capture.release()
    if len(source_frames) != manifest["frame_count"]:
        raise ValueError(f"Expected {manifest['frame_count']} frames, found {len(source_frames)}")
    output_dir = BENCHMARK / "predictions"
    output_dir.mkdir(exist_ok=True)

    for spec in MODELS:
        model = YOLO(str(spec["path"]))
        started = time.perf_counter()
        results = [
            model.predict(
                frame,
                imgsz=640,
                conf=spec["confidence_threshold"],
                classes=spec["classes"],
                device="cpu",
                verbose=False,
            )[0]
            for frame in source_frames
        ]
        elapsed = time.perf_counter() - started
        frames = []
        detected = multiple = candidates = 0
        for frame_index, result in enumerate(results):
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = [round(float(v), 3) for v in box.xyxy[0].tolist()]
                    class_id = int(box.cls[0].item())
                    if class_id not in spec["output_classes"]:
                        continue
                    detections.append(
                        {
                            "center_x": round((x1 + x2) / 2, 3),
                            "center_y": round((y1 + y2) / 2, 3),
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "confidence": round(float(box.conf[0].item()), 6),
                            "class_id": class_id,
                            "class_name": str(model.names[class_id]),
                            "source_model": spec["id"],
                            "evidence_type": "OBSERVED",
                        }
                    )
            frame_status = status(len(detections))
            detected += bool(detections)
            multiple += len(detections) > 1
            candidates += len(detections)
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp": round(frame_index / manifest["fps"], 6),
                    "status": frame_status,
                    "detections": detections,
                }
            )
        payload = {
            "contract_version": "P-EYE-DETECTOR-OBSERVATION-V1",
            "benchmark_id": manifest["benchmark_id"],
            "source_video_sha256": manifest["source_video_sha256"],
            "model": {
                **{key: value for key, value in spec.items() if key != "path"},
                "weights_path": str(spec["path"].resolve()),
                "weights_sha256": sha256(spec["path"]),
                "inference_variant": "default 640 full-frame (默认640整帧推理)",
                "threshold_origin": "frozen project configuration; not tuned on canonical test",
                "device": "cpu",
            },
            "summary": {
                "frames": len(frames),
                "detected_frames": detected,
                "missing_frames": len(frames) - detected,
                "multiple_frames": multiple,
                "candidate_count": candidates,
                "elapsed_s": round(elapsed, 4),
                "latency_ms_per_frame": round(1000 * elapsed / len(frames), 3),
            },
            "frames": frames,
        }
        destination = output_dir / f"{spec['id']}_default.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"path": str(destination), **payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
