from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class YoloPoseAdapter:
    def __init__(self, model_path: Path, tracker: str = "largest") -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.tracker = tracker
        self.selected_track_id: int | None = None
        self.id_switches = 0

    def infer(self, frame: np.ndarray) -> tuple[dict[str, Any] | None, float]:
        started = time.perf_counter()
        kwargs = dict(imgsz=640, conf=0.20, verbose=False, device="cpu")
        if self.tracker == "largest":
            result = self.model.predict(frame, **kwargs)[0]
        else:
            result = self.model.track(
                frame,
                persist=True,
                tracker=f"{self.tracker}.yaml",
                **kwargs,
            )[0]
        runtime_ms = (time.perf_counter() - started) * 1000
        if result.boxes is None or result.keypoints is None or not len(result.boxes):
            return None, runtime_ms

        boxes = np.asarray(_to_list(result.boxes.xyxy), dtype=float)
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0, boxes[:, 3] - boxes[:, 1]
        )
        index = int(np.argmax(areas))
        track_ids: list[int] = []
        if result.boxes.id is not None:
            track_ids = [int(value) for value in _to_list(result.boxes.id)]
            if self.selected_track_id in track_ids:
                index = track_ids.index(self.selected_track_id)
            else:
                new_id = track_ids[index]
                if self.selected_track_id is not None and new_id != self.selected_track_id:
                    self.id_switches += 1
                self.selected_track_id = new_id

        xy = np.asarray(_to_list(result.keypoints.xy[index]), dtype=float)
        if result.keypoints.conf is None:
            confidence = np.ones(len(xy), dtype=float)
        else:
            confidence = np.asarray(_to_list(result.keypoints.conf[index]), dtype=float)
        return {
            "bbox": boxes[index].tolist(),
            "track_id": track_ids[index] if track_ids else None,
            "person_count": int(len(boxes)),
            "keypoints": xy.tolist(),
            "confidence": confidence.tolist(),
            "visible_keypoints": int(np.sum(confidence >= 0.25)),
        }, runtime_ms


class YoloBallAdapter:
    def __init__(self, model_path: Path, class_id: int | None, confidence: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.class_id = class_id
        self.confidence = confidence

    def infer(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        result = self.model.predict(
            frame,
            imgsz=640,
            conf=self.confidence,
            verbose=False,
            device="cpu",
        )[0]
        runtime_ms = (time.perf_counter() - started) * 1000
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections, runtime_ms
        for box in result.boxes:
            cls = int(box.cls[0])
            if self.class_id is not None and cls != self.class_id:
                continue
            detections.append(
                {
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    "confidence": float(box.conf[0]),
                    "class_id": cls,
                }
            )
        return detections, runtime_ms


class SahiBallAdapter:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        slice_size: int = 320,
        overlap_ratio: float = 0.20,
    ) -> None:
        from sahi import AutoDetectionModel

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=confidence,
            device="cpu",
        )
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio

    def infer(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        from sahi.predict import get_sliced_prediction

        started = time.perf_counter()
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.50,
            verbose=0,
        )
        runtime_ms = (time.perf_counter() - started) * 1000
        detections = []
        for prediction in result.object_prediction_list:
            detections.append(
                {
                    "bbox": [float(value) for value in prediction.bbox.to_xyxy()],
                    "confidence": float(prediction.score.value),
                    "class_id": int(prediction.category.id),
                }
            )
        return detections, runtime_ms


class RFDetrBallAdapter:
    def __init__(self, model_path: Path, confidence: float) -> None:
        from rfdetr import RFDETRNano

        self.model = RFDETRNano(pretrain_weights=str(model_path))
        self.confidence = confidence

    def infer(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        result = self.model.predict(frame, threshold=self.confidence)
        runtime_ms = (time.perf_counter() - started) * 1000
        detections = []
        for bbox, confidence, class_id in zip(
            result.xyxy, result.confidence, result.class_id
        ):
            if int(class_id) != 37:  # COCO category id for sports ball.
                continue
            detections.append(
                {
                    "bbox": [float(value) for value in bbox],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                }
            )
        return detections, runtime_ms


class RtmlibPoseAdapter:
    def __init__(self, candidate: str) -> None:
        from rtmlib import Body, Wholebody

        if candidate == "rtmw":
            self.model = Wholebody(
                to_openpose=False,
                mode="balanced",
                backend="onnxruntime",
                device="cpu",
            )
        elif candidate == "rtmpose":
            self.model = Body(
                to_openpose=False,
                mode="balanced",
                backend="onnxruntime",
                device="cpu",
            )
        else:
            raise ValueError(f"Unknown RTMLib candidate: {candidate}")
        self.candidate = candidate

    def infer(self, frame: np.ndarray) -> tuple[dict[str, Any] | None, float]:
        started = time.perf_counter()
        keypoints, scores = self.model(frame)
        runtime_ms = (time.perf_counter() - started) * 1000
        keypoints = np.asarray(keypoints, dtype=float)
        scores = np.asarray(scores, dtype=float)
        if keypoints.size == 0:
            return None, runtime_ms
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]
        index = int(np.argmax(np.sum(scores >= 0.25, axis=1)))
        xy = keypoints[index, :, :2]
        confidence = scores[index]
        valid = confidence >= 0.25
        if not np.any(valid):
            return None, runtime_ms
        visible_xy = xy[valid]
        x1, y1 = np.min(visible_xy, axis=0)
        x2, y2 = np.max(visible_xy, axis=0)
        return {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "track_id": None,
            "person_count": int(len(keypoints)),
            "keypoints": xy.tolist(),
            "confidence": confidence.tolist(),
            "visible_keypoints": int(np.sum(valid)),
        }, runtime_ms


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def normalized_point_distance(
    first: tuple[float, float], second: tuple[float, float], scale: float
) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1]) / max(scale, 1.0)


def optical_flow_track(
    frames: list[np.ndarray], initial_point: tuple[float, float]
) -> list[dict[str, Any]]:
    if not frames:
        return []
    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    point = np.asarray([[initial_point]], dtype=np.float32)
    rows = [{"x": float(point[0, 0, 0]), "y": float(point[0, 0, 1]), "visible": True}]
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_point, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            point,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        visible = bool(status is not None and status[0, 0])
        if visible:
            point = next_point
        rows.append(
            {
                "x": float(point[0, 0, 0]),
                "y": float(point[0, 0, 1]),
                "visible": visible,
                "error": float(error[0, 0]) if error is not None else None,
            }
        )
        previous_gray = gray
    return rows


class CoTracker3Adapter:
    def __init__(self, repo_path: Path) -> None:
        import torch

        self.torch = torch
        self.model = torch.hub.load(
            str(repo_path), "cotracker3_offline", source="local"
        ).eval()

    def track(
        self,
        frames: list[np.ndarray],
        initial_frame: int,
        initial_point: tuple[float, float],
    ) -> tuple[list[dict[str, Any]], float]:
        torch = self.torch
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float()
        query = torch.tensor(
            [[[float(initial_frame), initial_point[0], initial_point[1]]]],
            dtype=torch.float32,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            tracks, visibility = self.model(
                video, queries=query, backward_tracking=True
            )
        runtime_ms = (time.perf_counter() - started) * 1000
        xy = tracks[0, :, 0].cpu().numpy()
        visible = visibility[0, :, 0].cpu().numpy()
        return [
            {"x": float(point[0]), "y": float(point[1]), "visible": bool(is_visible)}
            for point, is_visible in zip(xy, visible)
        ], runtime_ms

    def track_reanchored(
        self,
        frames: list[np.ndarray],
        detector_rows: list[dict[str, Any]],
        initial_frame: int,
        initial_point: tuple[float, float],
    ) -> tuple[list[dict[str, Any]], float]:
        from validation_closure import reanchor_track

        tracked, runtime_ms = self.track(frames, initial_frame, initial_point)
        frame_indices = [int(row["frame_index"]) for row in detector_rows]
        tracker_rows = [
            {
                "frame_index": frame_indices[index] if index < len(frame_indices) else index,
                **row,
            }
            for index, row in enumerate(tracked)
        ]
        return reanchor_track(tracker_rows, detector_rows), runtime_ms
