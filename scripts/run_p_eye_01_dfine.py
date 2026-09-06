#!/usr/bin/env python3
"""Run the official D-FINE-L COCO checkpoint on held-out P-EYE-01 frames."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image


DFINE = Path(r"E:\BasketballShotAI\model_zoo\P-EYE-01\D-FINE")
BENCHMARK = Path(r"E:\BasketballShotAI\benchmarks\P-EYE-01\CANONICAL_EYE_TEST_01")
WEIGHTS = DFINE / "weights" / "dfine_l_coco.pth"
CONFIG = DFINE / "configs" / "dfine" / "dfine_hgnetv2_l_coco.yml"
DEFAULT_THRESHOLD = 0.4  # Official tools/inference/torch_inf.py draw() default.
SPORTS_BALL_CLASS = 32  # Contiguous MS-COCO class index used with remap_mscoco_category=true.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Official checkpoint is unavailable: {WEIGHTS}")
    sys.path.insert(0, str(DFINE))
    from src.core import YAMLConfig

    manifest = json.loads((BENCHMARK / "canonical_manifest.json").read_text(encoding="utf-8"))
    config = YAMLConfig(str(CONFIG), resume=str(WEIGHTS))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(WEIGHTS, map_location="cpu")
    config.model.load_state_dict(checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"])
    model, postprocessor = config.model.deploy().eval(), config.postprocessor.deploy().eval()
    image_transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    frame_paths = sorted((BENCHMARK / "frames").glob("frame_*.jpg"))
    started, rows = time.perf_counter(), []
    with torch.inference_mode():
        for frame_index, frame_path in enumerate(frame_paths):
            image = Image.open(frame_path).convert("RGB")
            width, height = image.size
            output = postprocessor(model(image_transform(image).unsqueeze(0)), torch.tensor([[width, height]]))
            labels, boxes, scores = output
            detections = []
            for label, box, score in zip(labels[0], boxes[0], scores[0]):
                if int(label) != SPORTS_BALL_CLASS or float(score) < DEFAULT_THRESHOLD:
                    continue
                x1, y1, x2, y2 = [round(float(value), 3) for value in box.tolist()]
                detections.append({"center_x": round((x1+x2)/2, 3), "center_y": round((y1+y2)/2, 3), "bbox_xyxy": [x1, y1, x2, y2], "confidence": round(float(score), 6), "class_id": SPORTS_BALL_CLASS, "class_name": "sports ball", "source_model": "dfine_l", "evidence_type": "OBSERVED"})
            rows.append({"frame_index": frame_index, "timestamp": round(frame_index / manifest["fps"], 6), "status": "MISSING" if not detections else "DETECTED" if len(detections) == 1 else "MULTIPLE", "detections": detections})
    elapsed = time.perf_counter() - started
    payload = {"contract_version": "P-EYE-DETECTOR-OBSERVATION-V1", "benchmark_id": manifest["benchmark_id"], "source_video_sha256": manifest["source_video_sha256"], "model": {"id": "dfine_l", "source_model": "D-FINE-L COCO (D-FINE-L COCO 目标检测器)", "weights_path": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS), "inference_variant": "official 640 full-frame inference (官方640整帧推理)", "confidence_threshold": DEFAULT_THRESHOLD, "threshold_origin": "official torch inference draw default; not tuned on canonical test", "device": "cpu", "target_class": {"id": SPORTS_BALL_CLASS, "name": "sports ball"}}, "summary": {"frames": len(rows), "detected_frames": sum(bool(row["detections"]) for row in rows), "missing_frames": sum(not row["detections"] for row in rows), "multiple_frames": sum(len(row["detections"]) > 1 for row in rows), "candidate_count": sum(len(row["detections"]) for row in rows), "elapsed_s": round(elapsed, 4), "latency_ms_per_frame": round(1000 * elapsed / len(rows), 3)}, "frames": rows}
    destination = BENCHMARK / "predictions" / "dfine_l_official_default.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(destination), **payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
