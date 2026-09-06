#!/usr/bin/env python3
"""Run official WASB basketball checkpoints on the held-out P-EYE-01 frames."""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


WASB = Path(r"E:\BasketballShotAI\model_zoo\P-EYE-01\WASB-SBDT")
BENCHMARK = Path(r"E:\BasketballShotAI\benchmarks\P-EYE-01\CANONICAL_EYE_TEST_01")
sys.path.insert(0, str(WASB / "src"))

from models import build_model  # noqa: E402
from utils.image import affine_transform, get_affine_transform  # noqa: E402


MODELS = {
    "deepball": ("deepball.yaml", "deepball_basketball_best.pth.tar"),
    "deepball_large": ("deepball_large.yaml", "deepball-large_basketball_best.pth.tar"),
    "ballseg": ("ballseg.yaml", "ballseg_basketball_best.pth.tar"),
    "tracknetv2": ("tracknetv2.yaml", "tracknetv2_basketball_best.pth.tar"),
    "restracknetv2": ("restracknetv2.yaml", "restracknetv2_basketball_best.pth.tar"),
    "monotrack": ("monotrack.yaml", "monotrack_basketball_best.pth.tar"),
    "wasb": ("wasb.yaml", "wasb_basketball_best.pth.tar"),
}
THRESHOLD = 0.5  # Official WASB postprocessor default; never tuned on canonical test.


def preprocess(frame: np.ndarray, width: int, height: int) -> torch.Tensor:
    h, w = frame.shape[:2]
    transform = get_affine_transform(np.array([w / 2, h / 2], dtype=np.float32), max(h, w), 0, [width, height])
    image = cv2.warpAffine(frame, transform, (width, height), flags=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    return (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def heatmap_candidates(heatmap: np.ndarray, frame: np.ndarray) -> list[dict]:
    _, binary = cv2.threshold(heatmap, THRESHOLD, 1, cv2.THRESH_BINARY)
    count, labels = cv2.connectedComponents(binary.astype(np.uint8))
    transform = get_affine_transform(
        np.array([frame.shape[1] / 2, frame.shape[0] / 2], dtype=np.float32),
        max(frame.shape[:2]),
        0,
        [heatmap.shape[1], heatmap.shape[0]],
        inv=1,
    )
    candidates = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        weights = heatmap[ys, xs]
        point = affine_transform(np.array([np.average(xs, weights=weights), np.average(ys, weights=weights)]), transform)
        candidates.append(
            {
                "center_x": round(float(point[0]), 3),
                "center_y": round(float(point[1]), 3),
                "bbox_xyxy": None,
                "confidence": round(float(weights.max()), 6),
                "source_model": "",
                "evidence_type": "OBSERVED",
            }
        )
    return candidates


def run_model(model_id: str, frames: list[np.ndarray], manifest: dict) -> dict:
    config_name, weight_name = MODELS[model_id]
    weight_path = WASB / "pretrained_weights" / weight_name
    if not weight_path.is_file():
        return {"model_id": model_id, "status": "ASSET_UNAVAILABLE", "reason": f"Missing official weight: {weight_path}"}

    model_cfg = OmegaConf.load(WASB / "src" / "configs" / "model" / config_name)
    cfg = OmegaConf.create({"model": model_cfg})
    model = build_model(cfg)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_frames = int(model_cfg.frames_in)
    output_frame = int(model_cfg.frames_out) // 2
    prepared = [preprocess(frame, int(model_cfg.inp_width), int(model_cfg.inp_height)) for frame in frames]

    started = time.perf_counter()
    rows = []
    with torch.inference_mode():
        for index in range(len(frames)):
            window = [prepared[min(max(index + offset, 0), len(frames) - 1)] for offset in range(-(input_frames // 2), input_frames - (input_frames // 2))]
            if bool(model_cfg.rgb_diff):
                window[0] = torch.abs(window[1] - window[0])
            outputs = model(torch.cat(window).unsqueeze(0))[0]
            if model_id.startswith("deepball"):
                heatmap = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()
            else:
                heatmap = torch.sigmoid(outputs)[0, min(output_frame, outputs.shape[1] - 1)].cpu().numpy()
            candidates = heatmap_candidates(heatmap, frames[index])
            for candidate in candidates:
                candidate["source_model"] = model_id
            rows.append(
                {
                    "frame_index": index,
                    "timestamp": round(index / manifest["fps"], 6),
                    "status": "MISSING" if not candidates else "DETECTED" if len(candidates) == 1 else "MULTIPLE",
                    "detections": candidates,
                }
            )
    elapsed = time.perf_counter() - started
    candidates = sum(len(row["detections"]) for row in rows)
    return {
        "contract_version": "P-EYE-DETECTOR-OBSERVATION-V1",
        "benchmark_id": manifest["benchmark_id"],
        "source_video_sha256": manifest["source_video_sha256"],
        "model": {
            "id": model_id,
            "source_repository": "nttcom/WASB-SBDT",
            "weights_path": str(weight_path),
            "inference_variant": "official default resolution (官方默认分辨率)",
            "confidence_threshold": THRESHOLD,
            "threshold_origin": "official postprocessor default; not tuned on canonical test",
            "device": "cpu compatibility run (CPU 兼容运行)",
            "observation_geometry": "center point only; bbox unavailable",
        },
        "summary": {
            "frames": len(rows),
            "detected_frames": sum(bool(row["detections"]) for row in rows),
            "missing_frames": sum(not row["detections"] for row in rows),
            "multiple_frames": sum(len(row["detections"]) > 1 for row in rows),
            "candidate_count": candidates,
            "elapsed_s": round(elapsed, 4),
            "latency_ms_per_frame": round(elapsed * 1000 / len(rows), 3),
        },
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=tuple(MODELS))
    args = parser.parse_args()
    manifest = json.loads((BENCHMARK / "canonical_manifest.json").read_text(encoding="utf-8"))
    frame_paths = sorted((BENCHMARK / "frames").glob("frame_*.jpg"))
    frames = [cv2.imread(str(path)) for path in frame_paths]
    if len(frames) != manifest["frame_count"] or any(frame is None for frame in frames):
        raise RuntimeError("Canonical frames are unavailable")
    output_dir = BENCHMARK / "predictions"
    output_dir.mkdir(exist_ok=True)
    for model_id in args.models:
        result = run_model(model_id, frames, manifest)
        destination = output_dir / f"{model_id}_official_default.json"
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"model": model_id, "status": result.get("status", "COMPLETE"), "summary": result.get("summary")}, ensure_ascii=False))


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    main()
