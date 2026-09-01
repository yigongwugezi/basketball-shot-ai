from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2VideoModel,
    Sam2VideoProcessor,
)


def run_grounded_sam2(
    frames: list[np.ndarray], output_dir: Path, prompt: str = "basketball."
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    grounding_id = "IDEA-Research/grounding-dino-tiny"
    sam_id = "facebook/sam2.1-hiera-tiny"
    grounding_processor = AutoProcessor.from_pretrained(grounding_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_id).eval()
    sam_processor = Sam2VideoProcessor.from_pretrained(sam_id)
    sam_model = Sam2VideoModel.from_pretrained(sam_id).eval()

    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    first = Image.fromarray(rgb_frames[0])
    started = time.perf_counter()
    inputs = grounding_processor(images=first, text=prompt, return_tensors="pt")
    with torch.inference_mode():
        grounding_output = grounding_model(**inputs)
    detection = grounding_processor.post_process_grounded_object_detection(
        grounding_output,
        inputs.input_ids,
        threshold=0.20,
        text_threshold=0.20,
        target_sizes=[first.size[::-1]],
    )[0]
    if not len(detection["boxes"]):
        raise RuntimeError("Grounding DINO returned no basketball box on the first frame")
    best_index = int(torch.argmax(detection["scores"]))
    box = detection["boxes"][best_index].tolist()
    score = float(detection["scores"][best_index])

    session = sam_processor.init_video_session(
        video=rgb_frames,
        inference_device="cpu",
        video_storage_device="cpu",
        dtype=torch.float32,
    )
    sam_processor.add_inputs_to_inference_session(
        session, frame_idx=0, obj_ids=1, input_boxes=[[box]]
    )
    with torch.inference_mode():
        sam_model(inference_session=session, frame_idx=0)
        outputs = list(
            sam_model.propagate_in_video_iterator(
                session, start_frame_idx=0, max_frame_num_to_track=len(frames)
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    rows = []
    nonempty = 0
    for output in outputs:
        mask = output.pred_masks[0, 0].detach().cpu().numpy() > 0
        ys, xs = np.where(mask)
        bbox = None
        if len(xs):
            nonempty += 1
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        rows.append(
            {
                "frame_index": int(output.frame_idx),
                "mask_nonempty": bool(len(xs)),
                "mask_bbox": json.dumps(bbox),
            }
        )
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "prompt": prompt,
        "grounding_box": box,
        "grounding_score": score,
        "frames": len(frames),
        "propagated_frames": len(outputs),
        "nonempty_mask_coverage": nonempty / len(outputs) if outputs else 0,
        "runtime_ms": elapsed_ms,
        "ground_truth": "none_qualitative_smoke_only",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
