from __future__ import annotations

import base64
import math
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


def candidate_frame_indices(video_path: Path, meta: dict[str, Any]) -> dict[str, int]:
    frame_count = int(meta["frame_count"])
    max_index = max(frame_count - 1, 0)
    fallback = {
        item["key"]: min(max_index, max(0, round(max_index * float(item["ratio"]))))
        for item in FRAME_PLAN
    }
    if frame_count < 12:
        return fallback

    sample_count = min(17, frame_count)
    sampled: list[dict[str, Any]] = []
    for i in range(sample_count):
        idx = round(max_index * i / max(sample_count - 1, 1))
        frame = read_frame(video_path, idx)
        pose = detect_pose(frame)
        if not pose:
            continue
        metrics = pose["metrics"]
        sampled.append(
            {
                "index": idx,
                "wrist_y": shooting_wrist_y(pose),
                "elbow": metrics.get("shooting_elbow_angle"),
                "knee": metrics.get("min_knee_angle"),
                "visible": metrics.get("visible_keypoints", 0),
            }
        )

    useful = [item for item in sampled if item["visible"] >= 8]
    if len(useful) < 3:
        return fallback

    release_candidates = [item for item in useful if item["wrist_y"] is not None]
    if release_candidates:
        release = min(
            release_candidates,
            key=lambda item: (
                item["wrist_y"],
                -(item["elbow"] or 0),
            ),
        )
        release_index = release["index"]
    else:
        release_index = fallback["release"]

    before_release = [item for item in useful if item["index"] < release_index]
    after_release = [item for item in useful if item["index"] > release_index]

    if before_release:
        dip = min(before_release, key=lambda item: item["knee"] if item["knee"] is not None else 999)
        dip_index = dip["index"]
    else:
        dip_index = fallback["dip"]

    setup_pool = [item for item in useful if item["index"] < dip_index]
    setup_index = setup_pool[0]["index"] if setup_pool else fallback["setup"]

    if after_release:
        follow_index = after_release[0]["index"]
        landing_index = after_release[min(len(after_release) - 1, 2)]["index"]
    else:
        follow_index = min(max_index, release_index + round(frame_count * 0.12))
        landing_index = min(max_index, release_index + round(frame_count * 0.28))

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
    return indices


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
            frame_index = frame_indices[item["key"]]
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
                }
            )

        camera = estimate_camera_view(frames)
        return {
            "metadata": meta,
            "camera": camera,
            "quality": quality_checks_v2(meta, camera),
            "frames": frames,
            "metrics": summarize_metrics_v2(frames, camera),
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


app.mount("/", StaticFiles(directory=STATIC_ROOT), name="static")
