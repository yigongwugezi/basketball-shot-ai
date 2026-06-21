from __future__ import annotations

import base64
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "shot-analyzer-prototype"
MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "ball_rim_player_smoke" / "weights" / "best.pt"
COCO_MODEL_PATH = ROOT / "yolo11n.pt"
POSE_MODEL_PATH = ROOT / "yolo11n-pose.pt"

FRAME_PLAN = [
    {"key": "setup", "label": "准备", "ratio": 0.14},
    {"key": "dip", "label": "下沉", "ratio": 0.32},
    {"key": "release", "label": "出手", "ratio": 0.50},
    {"key": "follow_through", "label": "随球", "ratio": 0.64},
    {"key": "landing", "label": "落地", "ratio": 0.78},
]

FRAME_KEYS = ["setup", "dip", "release", "follow_through", "landing"]

CLASS_NAMES = ["ball", "rim", "player"]
CLASS_COLORS = {
    "ball": (20, 158, 245),
    "rim": (25, 118, 210),
    "player": (15, 118, 110),
}

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

app = FastAPI(title="AI Shot Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_model: YOLO | None = None
coco_model: YOLO | None = None
pose_model: YOLO | None = None
release_ball_model: YOLO | None = None
release_ball_model_path: Path | None = None

RELEASE_BALL_WINDOW_RADIUS = 3
RELEASE_BALL_CONFIDENCE = 0.25


def get_custom_model() -> YOLO:
    global custom_model
    if custom_model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Model not found: {MODEL_PATH}")
        custom_model = YOLO(str(MODEL_PATH))
    return custom_model


def get_coco_model() -> YOLO:
    global coco_model
    if coco_model is None:
        if not COCO_MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"COCO model not found: {COCO_MODEL_PATH}")
        coco_model = YOLO(str(COCO_MODEL_PATH))
    return coco_model


def get_pose_model() -> YOLO:
    global pose_model
    if pose_model is None:
        if not POSE_MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Pose model not found: {POSE_MODEL_PATH}")
        pose_model = YOLO(str(POSE_MODEL_PATH))
    return pose_model


def env_truthy(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def release_ball_detector_enabled() -> bool:
    return env_truthy("ENABLE_RELEASE_BALL_DETECTOR")


def configured_release_ball_model_path() -> Path | None:
    raw = os.getenv("RELEASE_BALL_MODEL_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def get_release_ball_model() -> tuple[YOLO | None, Path | None, str | None]:
    global release_ball_model
    global release_ball_model_path

    model_path = configured_release_ball_model_path()
    if model_path is None:
        return None, None, "disabled_by_missing_model"
    if not model_path.exists():
        return None, model_path, "model_missing"

    if release_ball_model is None or release_ball_model_path != model_path:
        release_ball_model = YOLO(str(model_path))
        release_ball_model_path = model_path
    return release_ball_model, model_path, None


def base_release_ball_evidence(status: str, release_frame_index: int | None = None) -> dict[str, Any]:
    model_path = configured_release_ball_model_path()
    return {
        "enabled": True,
        "status": status,
        "detector_type": "release_ball_yolo",
        "model_path": str(model_path) if model_path else None,
        "window_radius": RELEASE_BALL_WINDOW_RADIUS,
        "release_frame_index": release_frame_index,
        "frames": [],
        "best_frame": None,
    }


def encode_jpeg(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration = frame_count / fps if fps > 0 else 0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def read_frame(video_path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise HTTPException(status_code=500, detail=f"Could not read frame {frame_index}")
    return frame


def angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag_a = math.hypot(bax, bay)
    mag_c = math.hypot(bcx, bcy)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    value = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(value))


def visible(kp: list[dict[str, float]], idx: int, threshold: float = 0.25) -> bool:
    return idx < len(kp) and kp[idx]["confidence"] >= threshold


def point(kp: list[dict[str, float]], idx: int) -> tuple[float, float]:
    return kp[idx]["x"], kp[idx]["y"]


def choose_shooting_side(kp: list[dict[str, float]]) -> str | None:
    candidates: list[tuple[str, float]] = []
    for side, wrist, elbow, shoulder in [
        ("left", 9, 7, 5),
        ("right", 10, 8, 6),
    ]:
        if visible(kp, wrist) and visible(kp, elbow) and visible(kp, shoulder):
            candidates.append((side, kp[wrist]["y"]))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def pose_metrics(kp: list[dict[str, float]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "visible_keypoints": sum(1 for item in kp if item["confidence"] >= 0.25),
        "shooting_side": choose_shooting_side(kp),
    }

    side = metrics["shooting_side"]
    if side == "left" and all(visible(kp, idx) for idx in (5, 7, 9)):
        metrics["shooting_elbow_angle"] = round(angle(point(kp, 5), point(kp, 7), point(kp, 9)), 1)
    elif side == "right" and all(visible(kp, idx) for idx in (6, 8, 10)):
        metrics["shooting_elbow_angle"] = round(angle(point(kp, 6), point(kp, 8), point(kp, 10)), 1)

    knee_angles = []
    if all(visible(kp, idx) for idx in (11, 13, 15)):
        knee_angles.append(angle(point(kp, 11), point(kp, 13), point(kp, 15)))
    if all(visible(kp, idx) for idx in (12, 14, 16)):
        knee_angles.append(angle(point(kp, 12), point(kp, 14), point(kp, 16)))
    if knee_angles:
        metrics["min_knee_angle"] = round(min(knee_angles), 1)

    if all(visible(kp, idx) for idx in (5, 6, 11, 12)):
        shoulder_mid = ((kp[5]["x"] + kp[6]["x"]) / 2, (kp[5]["y"] + kp[6]["y"]) / 2)
        hip_mid = ((kp[11]["x"] + kp[12]["x"]) / 2, (kp[11]["y"] + kp[12]["y"]) / 2)
        dx = shoulder_mid[0] - hip_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        metrics["torso_lean_deg"] = round(abs(math.degrees(math.atan2(dx, max(dy, 1e-6)))), 1)

    return metrics


def midpoint(kp: list[dict[str, float]], a: int, b: int) -> tuple[float, float]:
    return (kp[a]["x"] + kp[b]["x"]) / 2, (kp[a]["y"] + kp[b]["y"]) / 2


def estimate_camera_view(frames: list[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    used_frames = 0
    for frame in frames:
        pose = frame.get("pose")
        if not pose:
            continue
        kp = pose["keypoints"]
        if not all(visible(kp, idx) for idx in (5, 6, 11, 12)):
            continue

        shoulder_width = math.hypot(kp[5]["x"] - kp[6]["x"], kp[5]["y"] - kp[6]["y"])
        hip_width = math.hypot(kp[11]["x"] - kp[12]["x"], kp[11]["y"] - kp[12]["y"])
        shoulder_mid = midpoint(kp, 5, 6)
        hip_mid = midpoint(kp, 11, 12)
        torso_height = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        if torso_height <= 1:
            continue

        ratios.append(((shoulder_width + hip_width) / 2) / torso_height)
        used_frames += 1

    if not ratios:
        return {
            "view": "unknown",
            "view_label": "未知机位",
            "confidence": 0.0,
            "angle_reliability": "low",
            "angle_reliability_label": "偏低",
            "warning": "没有足够稳定的肩部和髋部关键点，暂时无法判断拍摄机位。",
        }

    avg_ratio = sum(ratios) / len(ratios)
    frame_factor = min(1.0, used_frames / 3)
    if avg_ratio < 0.42:
        view = "side"
        label = "侧面机位"
        confidence = min(0.95, (0.42 - avg_ratio) / 0.42 + 0.35) * frame_factor
        reliability = "medium"
        reliability_label = "中等"
        warning = "侧面机位适合观察手臂伸展、下沉和随球，但当前角度仍是 2D 画面投影角。"
    elif avg_ratio < 0.72:
        view = "diagonal"
        label = "45度/斜侧机位"
        confidence = (0.65 + min(0.25, abs(avg_ratio - 0.57))) * frame_factor
        reliability = "medium"
        reliability_label = "中等"
        warning = "斜侧机位比较适合 MVP 分析，但肘角、膝角仍会受透视影响。"
    else:
        view = "front_or_back"
        label = "正面/后方机位"
        confidence = min(0.95, (avg_ratio - 0.72) / 0.72 + 0.35) * frame_factor
        reliability = "low"
        reliability_label = "偏低"
        warning = "正面或后方机位不适合直接读取肘角、膝角，只适合观察左右偏移和动作对称性。"

    return {
        "view": view,
        "view_label": label,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "angle_reliability": reliability,
        "angle_reliability_label": reliability_label,
        "torso_width_ratio": round(avg_ratio, 2),
        "used_frames": used_frames,
        "warning": warning,
    }


def detect_pose(frame) -> dict[str, Any] | None:
    results = get_pose_model().predict(frame, imgsz=640, conf=0.20, verbose=False, device="cpu")
    if not results or results[0].keypoints is None or results[0].boxes is None:
        return None

    keypoints_xy = results[0].keypoints.xy
    keypoints_conf = results[0].keypoints.conf
    boxes = results[0].boxes
    if keypoints_xy is None or keypoints_conf is None or len(keypoints_xy) == 0:
        return None

    best_index = 0
    best_area = -1.0
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_index = idx
            best_area = area

    xy = keypoints_xy[best_index].tolist()
    conf = keypoints_conf[best_index].tolist()
    kp = [
        {
            "name": KEYPOINT_NAMES[i],
            "x": float(item[0]),
            "y": float(item[1]),
            "confidence": float(conf[i]),
        }
        for i, item in enumerate(xy)
    ]
    return {
        "keypoints": kp,
        "metrics": pose_metrics(kp),
    }


def shooting_wrist_y(pose: dict[str, Any]) -> float | None:
    kp = pose["keypoints"]
    side = pose["metrics"].get("shooting_side")
    wrist_idx = 9 if side == "left" else 10 if side == "right" else None
    if wrist_idx is None or not visible(kp, wrist_idx):
        return None
    return kp[wrist_idx]["y"]


def shooting_wrist_point(pose: dict[str, Any]) -> tuple[float, float] | None:
    kp = pose["keypoints"]
    side = pose["metrics"].get("shooting_side")
    wrist_idx = 9 if side == "left" else 10 if side == "right" else None
    if wrist_idx is None or not visible(kp, wrist_idx):
        return None
    return point(kp, wrist_idx)


def ball_centers(detections: list[dict[str, Any]]) -> list[tuple[float, float]]:
    centers = []
    for detection in detections:
        if detection["class_name"] != "ball":
            continue
        x1, y1, x2, y2 = detection["xyxy"]
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return centers


def closest_ball_distance(
    wrist: tuple[float, float] | None,
    detections: list[dict[str, Any]],
) -> float | None:
    if wrist is None:
        return None
    centers = ball_centers(detections)
    if not centers:
        return None
    return min(math.hypot(center[0] - wrist[0], center[1] - wrist[1]) for center in centers)


def normalize_score(value: float | None, min_value: float, max_value: float, invert: bool = False) -> float:
    if value is None or max_value <= min_value:
        return 0.0
    score = (value - min_value) / (max_value - min_value)
    score = max(0.0, min(1.0, score))
    return 1.0 - score if invert else score


def fallback_frame_indices(max_index: int, release_evidence: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in FRAME_PLAN:
        key = item["key"]
        result[key] = {
            "frame_index": min(max_index, max(0, round(max_index * float(item["ratio"])))),
            "selection_method": "fallback_ratio",
            "confidence": 0.3 if key == "release" else 0.2,
            "evidence": release_evidence if key == "release" else "固定比例关键帧",
        }
    return result


def estimate_release_from_sampled_frames(
    sampled: list[dict[str, Any]],
    fallback: dict[str, dict[str, Any]],
    max_index: int,
) -> dict[str, dict[str, Any]]:
    useful = [item for item in sampled if item["visible"] >= 8]
    if len(useful) < 3:
        return fallback

    dip_pool = [
        item
        for item in useful
        if item["knee"] is not None and item["index"] <= max_index * 0.72
    ]
    if dip_pool:
        dip = min(dip_pool, key=lambda item: item["knee"])
        dip_index = dip["index"]
    else:
        dip_index = fallback["dip"]["frame_index"]

    release_candidates = [
        item
        for item in useful
        if item["index"] > dip_index and item["wrist_y"] is not None
    ]
    if len(release_candidates) < 2:
        return fallback

    wrist_values = [item["wrist_y"] for item in release_candidates if item["wrist_y"] is not None]
    elbow_values = [item["elbow"] for item in release_candidates if item["elbow"] is not None]
    distance_values = [
        item["ball_wrist_distance"]
        for item in release_candidates
        if item["ball_wrist_distance"] is not None
    ]
    min_wrist, max_wrist = min(wrist_values), max(wrist_values)
    min_elbow = min(elbow_values) if elbow_values else 90.0
    max_elbow = max(elbow_values) if elbow_values else 180.0
    min_distance = min(distance_values) if distance_values else 0.0
    max_distance = max(distance_values) if distance_values else 1.0

    scored = []
    for position, item in enumerate(release_candidates):
        has_follow = any(later["index"] > item["index"] for later in useful)
        if not has_follow:
            continue

        next_item = release_candidates[position + 1] if position + 1 < len(release_candidates) else None
        ball_moving_away = (
            item["ball_wrist_distance"] is not None
            and next_item is not None
            and next_item["ball_wrist_distance"] is not None
            and next_item["ball_wrist_distance"] > item["ball_wrist_distance"]
        )
        wrist_score = normalize_score(item["wrist_y"], min_wrist, max_wrist, invert=True)
        elbow_score = normalize_score(item["elbow"], min_elbow, max_elbow)
        visible_score = max(0.0, min(1.0, item["visible"] / 17))
        ball_score = 0.0
        if item["ball_wrist_distance"] is not None:
            ball_score = normalize_score(item["ball_wrist_distance"], min_distance, max_distance, invert=True)
            if ball_moving_away:
                ball_score = min(1.0, ball_score + 0.2)

        score = (
            wrist_score * 0.38
            + elbow_score * 0.27
            + visible_score * 0.20
            + ball_score * 0.15
        )
        scored.append(
            {
                **item,
                "score": score,
                "wrist_score": wrist_score,
                "elbow_score": elbow_score,
                "visible_score": visible_score,
                "ball_score": ball_score,
                "ball_moving_away": ball_moving_away,
            }
        )

    if not scored:
        return fallback

    release = max(scored, key=lambda item: item["score"])
    release_index = release["index"]
    before_release = [item for item in useful if item["index"] < release_index]
    after_release = [item for item in useful if item["index"] > release_index]

    if not before_release or not after_release:
        return fallback

    dip_before_release = [
        item for item in before_release if item["knee"] is not None
    ]
    if dip_before_release:
        dip_index = min(dip_before_release, key=lambda item: item["knee"])["index"]

    setup_pool = [item for item in useful if item["index"] < dip_index]
    setup_index = setup_pool[0]["index"] if setup_pool else fallback["setup"]["frame_index"]
    follow_index = after_release[0]["index"]
    landing_index = after_release[min(len(after_release) - 1, 2)]["index"]

    indices = {
        "setup": setup_index,
        "dip": dip_index,
        "release": release_index,
        "follow_through": follow_index,
        "landing": landing_index,
    }
    ordered = sorted(indices.items(), key=lambda item: item[1])
    if [key for key, _ in ordered] != FRAME_KEYS:
        return fallback

    result = {
        key: {
            "frame_index": index,
            "selection_method": "pose_trajectory_score" if key == "release" else "pose_sequence",
            "confidence": 0.5 if key != "release" else round(max(0.35, min(0.95, release["score"])), 2),
            "evidence": "由 release 前后有效骨架帧顺序推断",
        }
        for key, index in indices.items()
    }
    result["release"]["evidence"] = {
        "wrist_y": round(release["wrist_y"], 1) if release["wrist_y"] is not None else None,
        "elbow_angle": release["elbow"],
        "visible_keypoints": release["visible"],
        "ball_detected": release["ball_wrist_distance"] is not None,
        "ball_wrist_distance": (
            round(release["ball_wrist_distance"], 1)
            if release["ball_wrist_distance"] is not None
            else None
        ),
        "ball_moving_away": release["ball_moving_away"],
        "score": round(release["score"], 2),
    }
    return result


def candidate_frame_indices(video_path: Path, meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frame_count = int(meta["frame_count"])
    max_index = max(frame_count - 1, 0)
    fallback = fallback_frame_indices(max_index, "有效骨架帧不足，使用固定比例兜底")
    if frame_count < 12:
        return fallback

    sample_count = min(25, frame_count)
    sampled: list[dict[str, Any]] = []
    for i in range(sample_count):
        idx = round(max_index * i / max(sample_count - 1, 1))
        frame = read_frame(video_path, idx)
        pose = detect_pose(frame)
        if not pose:
            continue
        detections = detect_frame(frame)
        metrics = pose["metrics"]
        wrist = shooting_wrist_point(pose)
        sampled.append(
            {
                "index": idx,
                "wrist_y": shooting_wrist_y(pose),
                "wrist": wrist,
                "elbow": metrics.get("shooting_elbow_angle"),
                "knee": metrics.get("min_knee_angle"),
                "visible": metrics.get("visible_keypoints", 0),
                "ball_count": len(ball_centers(detections)),
                "ball_wrist_distance": closest_ball_distance(wrist, detections),
            }
        )
    return estimate_release_from_sampled_frames(sampled, fallback, max_index)


def draw_detections(frame, detections: list[dict[str, Any]]):
    annotated = frame.copy()
    for detection in detections:
        label = detection["class_name"]
        x1, y1, x2, y2 = [int(v) for v in detection["xyxy"]]
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {detection['confidence']:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_pose(frame, pose: dict[str, Any] | None):
    if not pose:
        return frame
    annotated = frame.copy()
    kp = pose["keypoints"]
    line_color = (80, 220, 120)
    dot_color = (255, 230, 80)
    for a, b in SKELETON:
        if visible(kp, a) and visible(kp, b):
            cv2.line(
                annotated,
                (int(kp[a]["x"]), int(kp[a]["y"])),
                (int(kp[b]["x"]), int(kp[b]["y"])),
                line_color,
                2,
                cv2.LINE_AA,
            )
    for item in kp:
        if item["confidence"] >= 0.25:
            cv2.circle(annotated, (int(item["x"]), int(item["y"])), 4, dot_color, -1, cv2.LINE_AA)
    return annotated


def detect_frame(frame) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    custom_results = get_custom_model().predict(
        frame,
        imgsz=640,
        conf=0.12,
        verbose=False,
        device="cpu",
    )
    if custom_results and custom_results[0].boxes is not None:
        for box in custom_results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(box.conf[0].item()),
                    "xyxy": xyxy,
                    "source": "custom",
                }
            )

    coco_results = get_coco_model().predict(
        frame,
        imgsz=640,
        conf=0.10,
        classes=[0, 32],
        verbose=False,
        device="cpu",
    )
    if coco_results and coco_results[0].boxes is not None:
        for box in coco_results[0].boxes:
            coco_class_id = int(box.cls[0].item())
            coco_name = get_coco_model().names[coco_class_id]
            if coco_name == "person":
                class_id = 2
                class_name = "player"
            elif coco_name == "sports ball":
                class_id = 0
                class_name = "ball"
            else:
                continue
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(box.conf[0].item()),
                    "xyxy": xyxy,
                    "source": "coco",
                }
            )
    return detections


def best_release_ball_detection(frame) -> dict[str, Any]:
    model, model_path, missing_status = get_release_ball_model()
    if model is None:
        return {
            "has_detection": False,
            "confidence": 0.0,
            "bbox": None,
            "status": missing_status or "model_missing",
            "model_path": str(model_path) if model_path else None,
        }

    results = model.predict(
        frame,
        imgsz=640,
        conf=RELEASE_BALL_CONFIDENCE,
        verbose=False,
        device="cpu",
    )
    best_box = None
    best_conf = 0.0
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            if class_id != 0:
                continue
            confidence = float(box.conf[0].item())
            if confidence > best_conf:
                best_conf = confidence
                best_box = [float(v) for v in box.xyxy[0].tolist()]

    return {
        "has_detection": best_box is not None,
        "confidence": round(best_conf, 4) if best_box is not None else 0.0,
        "bbox": [round(v, 2) for v in best_box] if best_box is not None else None,
        "status": "ok" if best_box is not None else "no_detection",
        "model_path": str(model_path) if model_path else None,
    }


def build_release_ball_evidence(
    video_path: Path,
    meta: dict[str, Any],
    release_frame_index: int,
) -> dict[str, Any]:
    evidence = base_release_ball_evidence("ok", release_frame_index)

    try:
        model, model_path, missing_status = get_release_ball_model()
    except Exception as exc:
        evidence["status"] = "error"
        evidence["error"] = str(exc)
        return evidence

    evidence["model_path"] = str(model_path) if model_path else None
    if model is None:
        evidence["status"] = missing_status or "model_missing"
        return evidence

    max_index = max(0, int(meta["frame_count"]) - 1)
    start = max(0, release_frame_index - RELEASE_BALL_WINDOW_RADIUS)
    end = min(max_index, release_frame_index + RELEASE_BALL_WINDOW_RADIUS)
    best_frame = None

    for frame_index in range(start, end + 1):
        item = {
            "frame_index": frame_index,
            "distance_to_release": frame_index - release_frame_index,
            "confidence": 0.0,
            "bbox": None,
            "has_detection": False,
            "status": "no_detection",
        }
        try:
            frame = read_frame(video_path, frame_index)
        except Exception as exc:
            item["status"] = "read_failed"
            item["error"] = str(exc)
            evidence["frames"].append(item)
            continue

        try:
            detected = best_release_ball_detection(frame)
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
            evidence["frames"].append(item)
            continue

        item.update(
            {
                "confidence": detected["confidence"],
                "bbox": detected["bbox"],
                "has_detection": detected["has_detection"],
                "status": detected["status"],
            }
        )
        evidence["frames"].append(item)

        if item["has_detection"]:
            if best_frame is None or item["confidence"] > best_frame["confidence"]:
                best_frame = dict(item)

    evidence["best_frame"] = best_frame
    return evidence


def build_release_fusion_diagnostic(
    pose_release_frame_index: int | None,
    release_ball_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    evidence_status = (
        release_ball_evidence.get("status")
        if isinstance(release_ball_evidence, dict)
        else None
    )
    best_frame = (
        release_ball_evidence.get("best_frame")
        if isinstance(release_ball_evidence, dict)
        else None
    )
    detector_release_frame_index = (
        int(best_frame["frame_index"])
        if isinstance(best_frame, dict) and best_frame.get("frame_index") is not None
        else None
    )
    risk_flags: list[str] = []

    if pose_release_frame_index is None:
        risk_flags.append("missing_pose_release_frame")
    if evidence_status != "ok":
        risk_flags.append("detector_unavailable")
    elif detector_release_frame_index is None:
        risk_flags.append("missing_detector_frame")

    fusion = {
        "status": "ok",
        "final_source": "pose_release",
        "pose_release_frame_index": pose_release_frame_index,
        "detector_release_frame_index": detector_release_frame_index,
        "frame_delta": None,
        "agreement_level": "unknown",
        "reason": "",
        "risk_flags": risk_flags,
    }

    if pose_release_frame_index is None:
        fusion["status"] = "insufficient_data"
        fusion["reason"] = "pose release frame is missing; keeping existing release behavior"
        return fusion
    if evidence_status != "ok":
        fusion["status"] = "detector_unavailable"
        fusion["reason"] = (
            f"detector unavailable ({evidence_status or 'unknown'}); keeping pose-based release"
        )
        return fusion
    if detector_release_frame_index is None:
        fusion["status"] = "insufficient_data"
        fusion["reason"] = "detector returned no best frame; keeping pose-based release"
        return fusion

    frame_delta = detector_release_frame_index - pose_release_frame_index
    absolute_delta = abs(frame_delta)
    fusion["frame_delta"] = frame_delta
    if frame_delta == 0:
        fusion["agreement_level"] = "exact_agreement"
        fusion["reason"] = "detector agrees with pose release frame"
    elif absolute_delta <= 1:
        fusion["agreement_level"] = "near_agreement_1"
        fusion["reason"] = "detector is near pose release frame within 1 frame"
    elif absolute_delta <= 3:
        fusion["agreement_level"] = "near_agreement_3"
        fusion["reason"] = "detector is near pose release frame within 3 frames"
    else:
        fusion["agreement_level"] = "disagreement"
        fusion["reason"] = (
            f"detector differs from pose release frame by {absolute_delta} frames; "
            "keeping pose-based release for safety"
        )
        fusion["risk_flags"].append("detector_pose_disagreement")
    return fusion


def quality_checks(meta: dict[str, Any]) -> list[dict[str, str]]:
    duration = float(meta["duration"])
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])
    return [
        {
            "title": "视频长度",
            "detail": f"{duration:.1f} 秒；MVP 建议 3-8 秒。",
            "state": "ok" if 3 <= duration <= 8 else "warn",
            "label": "合格" if 3 <= duration <= 8 else "需复查",
        },
        {
            "title": "画面方向",
            "detail": f"{width} x {height}；横屏更适合篮筐/球路，竖屏也可先做人体姿态。",
            "state": "ok" if width >= height else "warn",
            "label": "横屏" if width >= height else "竖屏",
        },
        {
            "title": "帧率",
            "detail": f"{fps:.1f} fps；后续手型和出手瞬间建议 60fps 以上。",
            "state": "ok" if fps >= 55 else "warn",
            "label": "较好" if fps >= 55 else "偏低",
        },
    ]


def summarize_metrics(frames: list[dict[str, Any]]) -> list[dict[str, str]]:
    counts = {name: 0 for name in CLASS_NAMES}
    pose_frames = 0
    visible_keypoints = []
    for frame in frames:
        for detection in frame["detections"]:
            counts[detection["class_name"]] = counts.get(detection["class_name"], 0) + 1
        if frame.get("pose"):
            pose_frames += 1
            visible_keypoints.append(frame["pose"]["metrics"].get("visible_keypoints", 0))

    release = next((frame for frame in frames if frame["key"] == "release"), None)
    dip = next((frame for frame in frames if frame["key"] == "dip"), None)
    follow = next((frame for frame in frames if frame["key"] == "follow_through"), None)

    metrics = [
        {
            "title": "人体姿态",
            "detail": f"5 个关键帧中有 {pose_frames} 帧检测到人体骨架；平均可见关键点 {sum(visible_keypoints) / max(len(visible_keypoints), 1):.1f}/17。",
            "state": "ok" if pose_frames else "warn",
            "label": "已接入" if pose_frames else "未检出",
        },
        {
            "title": "篮球检测",
            "detail": f"关键帧中共检测到 {counts.get('ball', 0)} 个 ball 候选框。",
            "state": "ok" if counts.get("ball", 0) else "warn",
            "label": "已检测" if counts.get("ball", 0) else "未检出",
        },
        {
            "title": "篮筐检测",
            "detail": f"关键帧中共检测到 {counts.get('rim', 0)} 个 rim 候选框；当前篮筐仍依赖后续专项训练。",
            "state": "ok" if counts.get("rim", 0) else "warn",
            "label": "已检测" if counts.get("rim", 0) else "未检出",
        },
        {
            "title": "球员检测",
            "detail": f"关键帧中共检测到 {counts.get('player', 0)} 个 player 候选框。",
            "state": "ok" if counts.get("player", 0) else "warn",
            "label": "已检测" if counts.get("player", 0) else "未检出",
        },
    ]

    if release and release.get("pose"):
        item = release["pose"]["metrics"]
        elbow = item.get("shooting_elbow_angle")
        side = item.get("shooting_side") or "unknown"
        metrics.append(
            {
                "title": "出手帧肘角",
                "detail": f"release 关键帧估算投篮侧为 {side}，肘角约 {elbow} 度。该值用于观察，不等于最终评分。",
                "state": "ok" if elbow else "warn",
                "label": "已估算" if elbow else "不足",
            }
        )

    if dip and dip.get("pose"):
        item = dip["pose"]["metrics"]
        knee = item.get("min_knee_angle")
        metrics.append(
            {
                "title": "下沉帧膝角",
                "detail": f"dip 关键帧估算最小膝角约 {knee} 度；角度越小通常表示下肢弯曲越明显。",
                "state": "ok" if knee else "warn",
                "label": "已估算" if knee else "不足",
            }
        )

    if follow and follow.get("pose"):
        item = follow["pose"]["metrics"]
        lean = item.get("torso_lean_deg")
        metrics.append(
            {
                "title": "躯干倾斜",
                "detail": f"follow-through 关键帧估算躯干相对竖直倾斜约 {lean} 度。",
                "state": "ok" if lean is not None else "warn",
                "label": "已估算" if lean is not None else "不足",
            }
        )

    metrics.append(
        {
            "title": "模型说明",
            "detail": "当前已接入 YOLO pose 骨架检测；动作阶段仍按固定比例抽帧，下一步要自动识别 release/follow-through。",
            "state": "warn",
            "label": "实验版",
        }
    )
    return metrics


def quality_checks_v2(meta: dict[str, Any], camera: dict[str, Any] | None = None) -> list[dict[str, str]]:
    duration = float(meta["duration"])
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])
    items = [
        {
            "title": "视频长度",
            "detail": f"{duration:.1f} 秒；MVP 建议 3-8 秒。",
            "state": "ok" if 3 <= duration <= 8 else "warn",
            "label": "合格" if 3 <= duration <= 8 else "需复查",
        },
        {
            "title": "画面方向",
            "detail": f"{width} x {height}；横屏更适合篮筐、球路和全身姿态分析。",
            "state": "ok" if width >= height else "warn",
            "label": "横屏" if width >= height else "竖屏",
        },
        {
            "title": "帧率",
            "detail": f"{fps:.1f} fps；后续手型和出手瞬间建议 60fps 以上。",
            "state": "ok" if fps >= 55 else "warn",
            "label": "较好" if fps >= 55 else "偏低",
        },
    ]
    if camera:
        items.append(
            {
                "title": "拍摄机位",
                "detail": f"{camera['view_label']}；机位置信度 {camera['confidence']:.2f}。{camera['warning']}",
                "state": "ok" if camera["angle_reliability"] == "medium" else "warn",
                "label": camera["angle_reliability_label"],
            }
        )
    return items


def summarize_metrics_v2(frames: list[dict[str, Any]], camera: dict[str, Any] | None = None) -> list[dict[str, str]]:
    counts = {name: 0 for name in CLASS_NAMES}
    pose_frames = 0
    visible_keypoints = []
    for frame in frames:
        for detection in frame["detections"]:
            counts[detection["class_name"]] = counts.get(detection["class_name"], 0) + 1
        if frame.get("pose"):
            pose_frames += 1
            visible_keypoints.append(frame["pose"]["metrics"].get("visible_keypoints", 0))

    release = next((frame for frame in frames if frame["key"] == "release"), None)
    dip = next((frame for frame in frames if frame["key"] == "dip"), None)
    follow = next((frame for frame in frames if frame["key"] == "follow_through"), None)

    metrics = [
        {
            "title": "人体骨架",
            "detail": f"5 个关键帧中有 {pose_frames} 帧检测到人体骨架；平均可见关键点 {sum(visible_keypoints) / max(len(visible_keypoints), 1):.1f}/17。",
            "state": "ok" if pose_frames else "warn",
            "label": "已接入" if pose_frames else "未检出",
        },
        {
            "title": "篮球检测",
            "detail": f"关键帧中共检测到 {counts.get('ball', 0)} 个 ball 候选框。",
            "state": "ok" if counts.get("ball", 0) else "warn",
            "label": "已检出" if counts.get("ball", 0) else "未检出",
        },
        {
            "title": "篮筐检测",
            "detail": f"关键帧中共检测到 {counts.get('rim', 0)} 个 rim 候选框；当前篮筐仍依赖后续专项训练。",
            "state": "ok" if counts.get("rim", 0) else "warn",
            "label": "已检出" if counts.get("rim", 0) else "未检出",
        },
        {
            "title": "球员检测",
            "detail": f"关键帧中共检测到 {counts.get('player', 0)} 个 player 候选框。",
            "state": "ok" if counts.get("player", 0) else "warn",
            "label": "已检出" if counts.get("player", 0) else "未检出",
        },
    ]

    if camera:
        metrics.append(
            {
                "title": "2D角度可信度",
                "detail": f"当前判断为{camera['view_label']}，角度可信度为{camera['angle_reliability_label']}。当前肘角、膝角都是画面投影角，不等于真实 3D 关节角。",
                "state": "ok" if camera["angle_reliability"] == "medium" else "warn",
                "label": camera["angle_reliability_label"],
            }
        )

    if release and release.get("pose"):
        item = release["pose"]["metrics"]
        elbow = item.get("shooting_elbow_angle")
        side = item.get("shooting_side") or "unknown"
        metrics.append(
            {
                "title": "出手帧肘角",
                "detail": f"release 关键帧估算投篮侧为 {side}，2D 肘角约 {elbow} 度。该值用于观察趋势，不作为最终评分。",
                "state": "ok" if elbow else "warn",
                "label": "已估算" if elbow else "不足",
            }
        )

    if dip and dip.get("pose"):
        item = dip["pose"]["metrics"]
        knee = item.get("min_knee_angle")
        metrics.append(
            {
                "title": "下沉帧膝角",
                "detail": f"dip 关键帧估算 2D 最小膝角约 {knee} 度；角度越小通常表示下肢弯曲越明显。",
                "state": "ok" if knee else "warn",
                "label": "已估算" if knee else "不足",
            }
        )

    if follow and follow.get("pose"):
        item = follow["pose"]["metrics"]
        lean = item.get("torso_lean_deg")
        metrics.append(
            {
                "title": "躯干倾斜",
                "detail": f"follow-through 关键帧估算躯干相对竖直倾斜约 {lean} 度。",
                "state": "ok" if lean is not None else "warn",
                "label": "已估算" if lean is not None else "不足",
            }
        )

    metrics.append(
        {
            "title": "模型说明",
            "detail": "当前已接入 YOLO pose 骨架检测；动作阶段识别仍是实验版，后续要进一步优化 release/follow-through。",
            "state": "warn",
            "label": "实验版",
        }
    )
    return metrics


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_exists": MODEL_PATH.exists(),
        "coco_model_exists": COCO_MODEL_PATH.exists(),
        "pose_model_exists": POSE_MODEL_PATH.exists(),
    }


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        meta = video_metadata(temp_path)
        if meta["frame_count"] <= 0:
            raise HTTPException(status_code=400, detail="Video has no frames")

        frames: list[dict[str, Any]] = []
        frame_indices = candidate_frame_indices(temp_path, meta)
        for item in FRAME_PLAN:
            frame_selection = frame_indices[item["key"]]
            frame_index = frame_selection["frame_index"]
            frame = read_frame(temp_path, frame_index)
            detections = detect_frame(frame)
            pose = detect_pose(frame)
            annotated = draw_pose(draw_detections(frame, detections), pose)
            frames.append(
                {
                    "key": item["key"],
                    "label": item["label"],
                    "time": frame_index / meta["fps"] if meta["fps"] else 0,
                    "frame_index": frame_index,
                    "dataUrl": encode_jpeg(annotated),
                    "detections": detections,
                    "pose": pose,
                    "selection_method": frame_selection.get("selection_method"),
                    "confidence": frame_selection.get("confidence"),
                    "evidence": frame_selection.get("evidence"),
                }
            )

        camera = estimate_camera_view(frames)
        response = {
            "metadata": meta,
            "camera": camera,
            "quality": quality_checks_v2(meta, camera),
            "frames": frames,
            "metrics": summarize_metrics_v2(frames, camera),
        }
        if release_ball_detector_enabled():
            release = next((frame for frame in frames if frame["key"] == "release"), None)
            if release:
                release_ball_evidence = build_release_ball_evidence(
                    temp_path,
                    meta,
                    int(release["frame_index"]),
                )
            else:
                release_ball_evidence = base_release_ball_evidence("no_release_frame")
            response["release_ball_evidence"] = release_ball_evidence
            response["release_fusion"] = build_release_fusion_diagnostic(
                int(release["frame_index"]) if release else None,
                release_ball_evidence,
            )
        return response
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


app.mount("/", StaticFiles(directory=STATIC_ROOT), name="static")
