from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.io import loadmat


PUBLIC_ROOT = Path(r"E:\BasketballShotAI\public_data")
REGISTRY_PATH = Path(__file__).with_name("public_pose_datasets.json")
CORE_JOINTS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
TRUSTED_GT_TYPES = {"HUMAN_GT", "MOCAP_GT", "MULTIVIEW_GT"}
DECISIONS = {"PENDING", "ACCEPTED", "REJECTED"}

JHMDB_JOINTS = [
    "neck", "belly", "head", "right_shoulder", "left_shoulder",
    "right_hip", "left_hip", "right_elbow", "left_elbow",
    "right_knee", "left_knee", "right_wrist", "left_wrist",
    "right_ankle", "left_ankle",
]
LSP_JOINTS = [
    "right_ankle", "right_knee", "right_hip", "left_hip", "left_knee",
    "left_ankle", "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist", "neck", "head_top",
]
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
JHMDB_ACTION_TAGS = {
    "shoot_ball": ["basketball", "shooting", "rapid_arm_motion"],
    "throw": ["throwing", "rapid_arm_motion"],
    "swing_baseball": ["striking", "rapid_arm_motion", "difficult_articulation"],
    "golf": ["striking", "arm_extension"],
    "jump": ["jumping", "airborne", "lower_body"],
    "catch": ["ball_present", "hand_object_occlusion_candidate"],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_entry(dataset_id: str) -> dict[str, Any]:
    for item in read_json(REGISTRY_PATH)["datasets"]:
        if item["id"] == dataset_id:
            return item
    raise KeyError(dataset_id)


def ensure_large_data_root(root: Path) -> None:
    if root.resolve().drive.upper() != "E:":
        raise ValueError(f"Public pose data must use E:, not {root}")


def map_joints(names: list[str], coordinates: np.ndarray, visibilities: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = []
    for index, name in enumerate(names):
        x, y = map(float, coordinates[index])
        original.append({"index": index, "name": name, "x": x, "y": y, "visibility": visibilities[index]})
    by_name = {joint["name"]: joint for joint in original}
    mapped = {}
    for name in CORE_JOINTS:
        source = by_name.get(name)
        mapped[name] = (
            {"status": "NOT_AVAILABLE", "source_name": None, "source_index": None, "x": None, "y": None, "visibility": "not_labelable"}
            if source is None else
            {"status": "OK", "source_name": name, "source_index": source["index"], "x": source["x"], "y": source["y"], "visibility": source["visibility"]}
        )
    return original, mapped


def derived_bbox(joints: dict[str, Any], width: int, height: int) -> list[float]:
    points = [(j["x"], j["y"]) for j in joints.values() if j["status"] == "OK" and j["x"] is not None]
    xs, ys = zip(*points)
    margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.12
    x1, y1 = max(0.0, min(xs) - margin), max(0.0, min(ys) - margin)
    x2, y2 = min(float(width - 1), max(xs) + margin), min(float(height - 1), max(ys) + margin)
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]


def mark_out_of_frame(joints: dict[str, Any], width: int, height: int) -> None:
    for joint in joints.values():
        if joint["status"] != "OK" or (0 <= joint["x"] < width and 0 <= joint["y"] < height):
            continue
        joint.update({
            "status": "OUT_OF_FRAME", "source_x": joint["x"], "source_y": joint["y"],
            "x": None, "y": None, "visibility": "not_labelable",
        })


def overlay(image: np.ndarray, joints: dict[str, Any], title: str) -> np.ndarray:
    canvas = image.copy()
    for first, second in SKELETON:
        a, b = joints[first], joints[second]
        if a["status"] == b["status"] == "OK":
            cv2.line(canvas, (round(a["x"]), round(a["y"])), (round(b["x"]), round(b["y"])), (38, 220, 255), 2, cv2.LINE_AA)
    for name, joint in joints.items():
        if joint["status"] != "OK":
            continue
        color = (70, 230, 90) if name.startswith("left_") else (255, 135, 70)
        point = (round(joint["x"]), round(joint["y"]))
        thickness = -1 if joint["visibility"] == "visible" else 2
        cv2.circle(canvas, point, 4, color, thickness, cv2.LINE_AA)
        cv2.putText(canvas, name.replace("left_", "L-").replace("right_", "R-"), (point[0] + 4, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(canvas, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def base_package(dataset_id: str, samples: list[dict[str, Any]], provenance_files: Iterable[Path], notes: list[str]) -> dict[str, Any]:
    entry = registry_entry(dataset_id)
    return {
        "schema_version": "public_pose_gt_v1",
        "dataset_id": dataset_id,
        "dataset_name": entry["name"],
        "review_status": "REVIEW_READY",
        "user_dataset_review": "PENDING",
        "ground_truth_type": entry["ground_truth_type"],
        "license_class": entry["license_class"],
        "license_summary": entry["license_summary"],
        "coordinate_system": "original_image_pixels_top_left_origin",
        "joint_set": CORE_JOINTS,
        "source_urls": entry["source_urls"],
        "provenance_files": [{"path": path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in provenance_files],
        "mapping_notes": notes,
        "samples": samples,
    }


def prepare_jhmdb(root: Path, review_root: Path, sample_count: int = 36) -> dict[str, Any]:
    actions = ["shoot_ball", "throw", "swing_baseball", "golf", "jump", "catch"]
    pairs = []
    for action in actions:
        mats = sorted((root / "joint_positions" / action).glob("*/joint_positions.mat"))
        mats = [path for path in mats if ".AppleDouble" not in str(path)]
        take = 3 if action == "shoot_ball" else 2
        for mat_path in mats[:take]:
            video = root / "ReCompress_Videos" / action / f"{mat_path.parent.name}.avi"
            if video.exists():
                pairs.append((action, mat_path, video))
    frame_slots = max(1, (sample_count + len(pairs) - 1) // max(1, len(pairs)))
    output = review_root / "jhmdb"
    media = output / "overlays"
    media.mkdir(parents=True, exist_ok=True)
    samples = []
    for action, mat_path, video in pairs:
        data = loadmat(mat_path)
        positions = np.asarray(data["pos_img"], dtype=float).transpose(2, 1, 0)
        capture = cv2.VideoCapture(str(video))
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        indices = np.linspace(0, min(len(positions), video_frames) - 1, frame_slots + 2, dtype=int)[1:-1]
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, image = capture.read()
            if not ok:
                continue
            visibility = ["occluded_but_inferable"] * len(JHMDB_JOINTS)
            original, mapped = map_joints(JHMDB_JOINTS, positions[frame_index], visibility)
            height, width = image.shape[:2]
            mark_out_of_frame(mapped, width, height)
            sample_id = f"{action}_{mat_path.parent.name}_{frame_index:04d}"
            overlay_path = media / f"{hashlib.sha1(sample_id.encode()).hexdigest()[:16]}.jpg"
            cv2.imwrite(str(overlay_path), overlay(image, mapped, f"JHMDB | {action} | frame {frame_index}"))
            samples.append({
                "id": sample_id, "sequence": mat_path.parent.name, "action": action,
                "review_tags": JHMDB_ACTION_TAGS[action], "laplacian_blur_score": round(float(cv2.Laplacian(image, cv2.CV_64F).var()), 3),
                "frame_index": int(frame_index), "timestamp_seconds": round(frame_index / fps, 6),
                "fps": fps, "width": width, "height": height, "media_path": video.as_posix(),
                "overlay_path": overlay_path.as_posix(), "original_joint_names": JHMDB_JOINTS,
                "original_joints": original, "mapped_joints": mapped,
                "person_bbox": derived_bbox(mapped, width, height), "person_bbox_source": "DERIVED_FROM_HUMAN_GT_JOINTS",
                "person_crop_status": "GT_PERSON_AVAILABLE",
            })
        capture.release()
    package = base_package(
        "jhmdb", samples[:sample_count],
        [PUBLIC_ROOT / "downloads" / name for name in ("JHMDB_video.zip", "joint_positions.zip", "splits.zip", "sub_splits.zip")],
        [
            "The raw 15-joint names and coordinates are preserved for every sample.",
            "All 12 project joints map directly by name.",
            "JHMDB pos_img does not distinguish visible from occluded; mapped visibility uses the conservative labelable value occluded_but_inferable and never claims visible.",
        ],
    )
    write_review(output, package)
    return package


def prepare_lsp(root: Path, review_root: Path, sample_count: int = 30) -> dict[str, Any]:
    mat_path = root / "joints.mat"
    joints = np.asarray(loadmat(mat_path)["joints"], dtype=float).transpose(2, 1, 0)
    indices = np.linspace(0, len(joints) - 1, sample_count, dtype=int)
    output, samples = review_root / "lsp", []
    media = output / "overlays"
    media.mkdir(parents=True, exist_ok=True)
    for index in indices:
        image_path = root / "images" / f"im{index + 1:04d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        coords = joints[index, :, :2]
        visibility = ["occluded_but_inferable" if flag else "visible" for flag in joints[index, :, 2]]
        original, mapped = map_joints(LSP_JOINTS, coords, visibility)
        height, width = image.shape[:2]
        mark_out_of_frame(mapped, width, height)
        overlay_path = media / f"im{index + 1:04d}.jpg"
        cv2.imwrite(str(overlay_path), overlay(image, mapped, f"Leeds Sports Pose | im{index + 1:04d}"))
        samples.append({
            "id": f"lsp_im{index + 1:04d}", "sequence": f"im{index + 1:04d}", "action": "sports_activity_not_encoded_per_image",
            "review_tags": ["difficult_static_sports_pose"] + (["occlusion"] if any(value == "occluded_but_inferable" for value in visibility) else []),
            "laplacian_blur_score": round(float(cv2.Laplacian(image, cv2.CV_64F).var()), 3),
            "frame_index": 0, "timestamp_seconds": 0.0, "fps": 1.0, "width": width, "height": height,
            "media_path": image_path.as_posix(), "media_sha256": sha256(image_path), "overlay_path": overlay_path.as_posix(),
            "original_joint_names": LSP_JOINTS, "original_joints": original, "mapped_joints": mapped,
            "person_bbox": derived_bbox(mapped, width, height), "person_bbox_source": "DERIVED_FROM_HUMAN_GT_JOINTS",
            "person_crop_status": "GT_PERSON_AVAILABLE",
        })
    package = base_package(
        "lsp", samples, [mat_path],
        [
            "The raw 14-joint names, coordinates, and binary occlusion flags are preserved.",
            "All 12 project joints map directly by name.",
            "The source third channel is an occlusion flag: 0 maps to visible and 1 maps to occluded_but_inferable.",
        ],
    )
    write_review(output, package)
    return package


def write_review(output: Path, package: dict[str, Any]) -> None:
    write_json(output / "normalized_gt.candidate.json", package)
    entry = registry_entry(package["dataset_id"])
    downloaded = sum(item["bytes"] for item in package["provenance_files"])
    cards = []
    for sample in package["samples"]:
        rel = Path(sample["overlay_path"]).relative_to(output).as_posix()
        visible = sum(j["visibility"] == "visible" for j in sample["mapped_joints"].values())
        occluded = sum(j["visibility"] == "occluded_but_inferable" for j in sample["mapped_joints"].values())
        tags = ", ".join(sample.get("review_tags", []))
        cards.append(f'<article><img loading="lazy" src="{html.escape(rel)}"><h3>{html.escape(sample["id"])}</h3><p>{html.escape(sample["action"])} · visible {visible} · occluded/unknown-labelable {occluded}<br>{html.escape(tags)}</p></article>')
    roles = ", ".join(entry.get("recommended_roles", []))
    source_links = " · ".join(f'<a href="{html.escape(url)}">source</a>' for url in package["source_urls"])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(package['dataset_name'])} GT review</title>
<style>body{{font:14px system-ui;background:#111;color:#eee;margin:24px}}.gate{{padding:14px;background:#392f12;border:1px solid #d7aa2c}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}article{{background:#1d1d1d;padding:10px}}img{{width:100%;height:280px;object-fit:contain;background:#000}}h1,h3{{margin:.4em 0}}p{{color:#bbb}}</style></head><body>
<h1>{html.escape(package['dataset_name'])} — real GT visual review</h1><div class="gate"><b>REVIEW_READY / USER_DATASET_REVIEW=PENDING</b><br>This page is evidence for user acceptance or rejection. It does not accept the dataset automatically.</div>
<p><b>Domain:</b> {html.escape(entry['relevance'])}<br><b>GT:</b> {html.escape(package['ground_truth_type'])} · {html.escape(entry['keypoints'])}<br><b>RGB/video:</b> {entry.get('rgb_available')} / {entry.get('video_available')} · <b>review samples:</b> {len(package['samples'])}<br><b>License:</b> {html.escape(package['license_class'])} — {html.escape(package['license_summary'])}<br><b>Downloaded scope:</b> {html.escape(entry.get('downloaded_size', str(downloaded) + ' bytes'))}<br><b>Why useful:</b> {html.escape(entry['usage_recommendation'])}<br><b>Known mismatch:</b> {html.escape(entry.get('known_mismatch', 'not recorded'))}<br><b>Recommended role:</b> {html.escape(roles)}<br><b>Provenance:</b> {source_links}</p>
<p><b>Legend:</b> left joints = green; right joints = blue; filled = source-visible; hollow = occluded or visibility-unspecified. Yellow lines are GT limbs. GT is shown without model predictions.</p><div class="grid">{''.join(cards)}</div></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def decisions_path(review_root: Path) -> Path:
    return review_root / "dataset_decisions.json"


def load_decisions(review_root: Path) -> dict[str, Any]:
    path = decisions_path(review_root)
    if path.exists():
        return read_json(path)
    value = {"schema_version": "public_pose_dataset_decisions_v1", "datasets": {"jhmdb": "PENDING", "lsp": "PENDING"}}
    write_json(path, value)
    return value


def record_decision(review_root: Path, dataset_id: str, decision: str) -> None:
    decision = decision.upper()
    if decision not in DECISIONS:
        raise ValueError(f"Decision must be one of {sorted(DECISIONS)}")
    value = load_decisions(review_root)
    value["datasets"][dataset_id] = decision
    write_json(decisions_path(review_root), value)


def export_existing_schema(review_root: Path, benchmark_root: Path, dataset_id: str) -> tuple[Path, Path]:
    decision = load_decisions(review_root)["datasets"].get(dataset_id, "PENDING")
    if decision != "ACCEPTED":
        raise PermissionError(f"{dataset_id} is {decision}; USER_DATASET_REVIEW must be ACCEPTED before benchmark export")
    package = read_json(review_root / dataset_id / "normalized_gt.candidate.json")
    if package["ground_truth_type"] not in TRUSTED_GT_TYPES:
        raise ValueError("Pseudo/automatic labels cannot be exported as public GT")
    clips, frames, annotations = {}, [], []
    for sample in package["samples"]:
        clip = sample["sequence"]
        clips.setdefault(clip, {
            "video": sample["media_path"], "fps": sample["fps"], "width": sample["width"], "height": sample["height"],
            "source_role": "public_gt_candidate", "slow_motion": False, "contaminated_research_only": package["license_class"] != "COMMERCIAL_FRIENDLY",
            "shooting_side": "unknown",
        })
        frames.append({"id": sample["id"], "clip": clip, "frame_index": sample["frame_index"], "timestamp_seconds": sample["timestamp_seconds"], "tags": [sample["action"], dataset_id]})
        annotations.append({"frame_id": sample["id"], "reviewed": True, "joints": {name: {"x": joint["x"], "y": joint["y"], "visibility": joint["visibility"]} for name, joint in sample["mapped_joints"].items()}})
    manifest = {
        "schema_version": "pose_gt_manifest_v1", "benchmark_version": f"public_{dataset_id}_pose_gt_v1_2026-09-02",
        "joint_set": CORE_JOINTS, "visibility_values": ["visible", "occluded_but_inferable", "not_labelable"],
        "coordinate_system": "original_frame_pixels_top_left_origin", "sampling": "public_dataset_user_accepted_review_sample",
        "source_dataset": dataset_id, "source_ground_truth_type": package["ground_truth_type"], "clips": clips, "frames": frames,
    }
    labels = {
        "schema_version": "pose_gt_annotations_v1", "benchmark_version": manifest["benchmark_version"],
        "annotator": f"original_{dataset_id}_human_gt", "revision": 1, "frames": annotations,
    }
    output = benchmark_root / dataset_id
    manifest_path, labels_path = output / "pose_gt_manifest.v1.json", output / "pose_gt_annotations.v1.json"
    write_json(manifest_path, manifest)
    write_json(labels_path, labels)
    return manifest_path, labels_path


def evaluate_public_predictions(package: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    """Count person/crop failures separately while retaining them in end-to-end failure."""
    gt = {sample["id"]: sample for sample in package["samples"]}
    rows = {frame["frame_id"]: frame for frame in predictions.get("frames", [])}
    crop_failures = pose_head_missing = detected = labelable = 0
    errors = []
    for frame_id, sample in gt.items():
        prediction = rows.get(frame_id, {})
        crop_ok = prediction.get("person_crop_status") == "OK"
        joints = prediction.get("joints", {}) if crop_ok else {}
        for name, truth in sample["mapped_joints"].items():
            if truth["visibility"] == "not_labelable":
                continue
            labelable += 1
            if not crop_ok:
                crop_failures += 1
                continue
            item = joints.get(name)
            if not item or item.get("x") is None or item.get("y") is None:
                pose_head_missing += 1
                continue
            detected += 1
            errors.append(float(np.hypot(item["x"] - truth["x"], item["y"] - truth["y"])))
    return {
        "schema_version": "public_pose_accuracy_result_v1", "model_id": predictions.get("model_id"),
        "labelable_joints": labelable, "detected_joints": detected,
        "person_or_crop_failure_joints": crop_failures, "pose_head_missing_joints": pose_head_missing,
        "end_to_end_failure_rate": round(1 - detected / labelable, 4) if labelable else None,
        "median_pixel_error_detected_only": round(float(np.median(errors)), 3) if errors else None,
    }


def verify_package(package: dict[str, Any]) -> None:
    if package["ground_truth_type"] not in TRUSTED_GT_TYPES:
        raise ValueError("Only trusted ground truth types are valid")
    if package["review_status"] != "REVIEW_READY" or package["user_dataset_review"] != "PENDING":
        raise ValueError("Newly prepared data must stop at REVIEW_READY/PENDING")
    for sample in package["samples"]:
        if set(sample["mapped_joints"]) != set(CORE_JOINTS):
            raise ValueError(f"Incomplete mapped joint set: {sample['id']}")
        if not Path(sample["overlay_path"]).exists():
            raise FileNotFoundError(sample["overlay_path"])
        for joint in sample["mapped_joints"].values():
            if joint["status"] == "OK" and not (0 <= joint["x"] < sample["width"] and 0 <= joint["y"] < sample["height"]):
                raise ValueError(f"Out-of-frame mapped joint: {sample['id']}")


def write_review_index(review_root: Path, packages: list[dict[str, Any]]) -> None:
    links = "".join(f'<li><a href="{p["dataset_id"]}/index.html">{html.escape(p["dataset_name"])}</a> — {len(p["samples"])} samples — REVIEW_READY / PENDING</li>' for p in packages)
    page = f"<!doctype html><html><meta charset=utf-8><title>Public pose GT review</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;background:#111;color:#eee}}a{{color:#65bfff}}li{{margin:14px}}</style><body><h1>Public Pose GT Review</h1><p>These are candidate datasets. Visual review is required before benchmark acceptance.</p><ul>{links}</ul></body></html>"
    (review_root / "index.html").write_text(page, encoding="utf-8")
    load_decisions(review_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and gate public human-pose ground truth")
    parser.add_argument("command", choices=["prepare-all", "prepare-jhmdb", "prepare-lsp", "verify", "decision", "export"])
    parser.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--dataset", choices=["jhmdb", "lsp"])
    parser.add_argument("--value", choices=sorted(DECISIONS))
    args = parser.parse_args()
    ensure_large_data_root(args.public_root)
    review_root = args.public_root / "dataset_review"
    benchmark_root = args.public_root / "benchmark_ready"
    packages = []
    if args.command in {"prepare-all", "prepare-jhmdb"}:
        packages.append(prepare_jhmdb(args.public_root / "datasets" / "fast_sports" / "jhmdb", review_root))
    if args.command in {"prepare-all", "prepare-lsp"}:
        packages.append(prepare_lsp(args.public_root / "datasets" / "fast_sports" / "leeds_sports_pose", review_root))
    if packages:
        for package in packages:
            verify_package(package)
        existing = [read_json(path) for path in review_root.glob("*/normalized_gt.candidate.json")]
        write_review_index(review_root, existing)
        print(review_root / "index.html")
    elif args.command == "verify":
        for path in review_root.glob("*/normalized_gt.candidate.json"):
            verify_package(read_json(path))
            print(f"OK {path}")
    elif args.command == "decision":
        if not args.dataset or not args.value:
            parser.error("decision requires --dataset and --value")
        record_decision(review_root, args.dataset, args.value)
        print(decisions_path(review_root))
    elif args.command == "export":
        if not args.dataset:
            parser.error("export requires --dataset")
        print(*export_existing_schema(review_root, benchmark_root, args.dataset), sep="\n")


if __name__ == "__main__":
    main()
