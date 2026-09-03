from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.reference_v1.pose_gt import JOINT_INDEX, evaluate_predictions, map_crop_points, validate_annotations, validate_manifest
from reference_v1.motion import build_motion_representation, validate_motion_representation
from reference_v1.schema import event, phase


def annotations_for(manifest: dict) -> dict:
    frames = []
    for frame_offset, meta in enumerate(manifest["frames"]):
        joints = {
            name: {"x": 100.0 + joint_offset * 10 + frame_offset, "y": 200.0 + joint_offset * 12, "visibility": "visible"}
            for joint_offset, name in enumerate(JOINT_INDEX)
        }
        frames.append({"frame_id": meta["id"], "reviewed": True, "reviewed_at": "2026-09-02T00:00:00Z", "notes": "", "joints": joints})
    frames[0]["joints"]["left_ankle"] = {"x": None, "y": None, "visibility": "not_labelable"}
    return {"schema_version": "pose_gt_annotations_v1", "benchmark_version": manifest["benchmark_version"], "annotator": "test", "revision": 1, "frames": frames}


def prediction_for(manifest: dict, model_id: str, dx: float) -> dict:
    return {
        "schema_version": "pose_model_predictions_v1",
        "benchmark_version": manifest["benchmark_version"],
        "model_id": model_id,
        "frames": [
            {
                "frame_id": frame["id"],
                "status": "ok",
                "joints": {
                    name: {"x": 100.0 + joint_offset * 10 + frame_offset + dx, "y": 200.0 + joint_offset * 12, "confidence": 0.9}
                    for joint_offset, name in enumerate(JOINT_INDEX)
                },
            }
            for frame_offset, frame in enumerate(manifest["frames"])
        ],
        "temporal_quality": {"ground_truth": "none_proxy_only"},
    }


def test_pose_gt_contract() -> None:
    manifest = json.loads((ROOT / "benchmarks" / "reference_v1" / "pose_gt_manifest.v1.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    annotations = annotations_for(manifest)
    round_trip = json.loads(json.dumps(annotations))
    validate_annotations(manifest, round_trip, require_complete=True)
    before = copy.deepcopy(round_trip)
    result = evaluate_predictions(manifest, round_trip, [prediction_for(manifest, "yolo_raw", 5), prediction_for(manifest, "yolo_filtered", 1)])
    assert round_trip == before  # evaluation never overwrites or backfills GT
    assert result["accuracy"]["yolo_raw"]["all"]["labelable_joints"] == len(manifest["frames"]) * 12 - 1
    assert result["filter_damage"]["summary"]["all"]["filter_help_rate"] == 1.0
    assert result["temporal_quality"]["yolo_raw"]["ground_truth"] == "none_proxy_only"
    assert map_crop_points([[0, 0], [100, 200]], [10, 20, 210, 420], (100, 200)) == [[10.0, 20.0], [210.0, 420.0]]
    filtered = prediction_for(manifest, "yolo_filtered", 1)
    del filtered["frames"][0]["joints"]["left_shoulder"]
    result = evaluate_predictions(manifest, round_trip, [prediction_for(manifest, "yolo_raw", 5), filtered])
    damage = result["filter_damage"]["summary"]["all"]
    assert damage["raw_detected_filtered_missing"] == 1
    assert damage["filter_harm_rate"] > 0


def synthetic_report() -> tuple[dict, list[dict]]:
    fps = 30.0
    frames = {"dip_start": 5, "bottom": 10, "takeoff": 20, "pose_release": 29, "strict_ball_release": 30, "body_apex": 32, "landing": 42}
    events = {name: event(name, "ok", frame=value, fps=fps, confidence=0.8) for name, value in frames.items()}
    phases = {
        "preparation": phase("preparation", "ok", start_frame=0, end_frame=5, fps=fps),
        "dip": phase("dip", "ok", start_frame=5, end_frame=10, fps=fps),
        "upward_drive": phase("upward_drive", "ok", start_frame=10, end_frame=30, fps=fps),
        "follow_through": phase("follow_through", "ok", start_frame=30, end_frame=42, fps=fps),
        "landing_recovery": phase("landing_recovery", "ok", start_frame=42, end_frame=49, fps=fps),
    }
    observations = [{"frame": frame, "center": [250 + frame, 500 - frame * 3], "confidence": 0.8, "ball_hand_distance_diameters": 1.0 if frame < 30 else 3.0, "contact_state": "contact_supported" if frame < 30 else "released_confirmed"} for frame in range(24, 37)]
    report = {
        "input": {"fps": fps, "frame_count": 50},
        "quality": {"status": "ok"},
        "attempt": {"shooting_side": "right", "view": {"value": "side", "status": "ok"}},
        "events": events,
        "phases": phases,
        "metrics": {"elbow_extension_onset_relative_to_release": {"status": "ok", "range": [24, 30]}},
        "ball_evidence": {"status": "ok", "detector": "synthetic", "center_observations": observations, "risk_flags": []},
        "pose_reliability": {"status": "ok"},
    }
    rows = []
    for frame in range(50):
        points = [[100 + index * 5 + frame, 400 + index * 8 - frame] for index in range(17)]
        rows.append({"frame_index": frame, "analysis_pose": {"keypoints": points, "temporal_reliability": [0.9] * 17}})
    return report, rows


def test_motion_contract() -> None:
    report, rows = synthetic_report()
    original = copy.deepcopy(report)
    result = build_motion_representation(report, rows, slow_motion=True, contaminated_research_only=True)
    validate_motion_representation(result)
    assert report == original  # strict-release and all other source semantics are immutable
    assert result["schema_version"] == "shot_motion_representation_v1"
    assert result["motion_representation_version"] == 1
    assert tuple(result["events"]) == (
        "dip_start", "dip_bottom", "leg_drive_onset", "ball_rise_start",
        "elbow_extension_onset", "takeoff", "release_region_start", "release_pose",
        "strict_ball_release", "body_apex", "release_region_end", "landing",
    )
    assert tuple(result["phases"]) == ("setup", "dip", "drive", "release", "follow_through", "landing_recovery")
    assert result["kinematics"]["normalized_shot_time"]["value"]["dip_bottom"] == 0.0
    assert result["kinematics"]["normalized_shot_time"]["value"]["strict_ball_release"] == 1.0
    assert result["kinematics"]["timing_interpretation"]["status"] == "low_confidence"
    assert "no_real_time_coordination_claim" in result["uncertainty"]["restrictions"]
    relation = result["temporal_relations"]["elbow_extension_onset_to_release_pose"]
    assert relation["delta_frames"] == 5
    assert relation["status"] == "low_confidence"
    assert result["events"]["release_pose"]["frame"] != result["events"]["strict_ball_release"]["frame"]
    assert result["kinematics"]["metrics"]["elbow_extension_onset_relative_to_release"]["reliability"] in {"HIGH", "MEDIUM", "LOW"}
    report["ball_evidence"]["center_observations"] = []
    result = build_motion_representation(report, rows)
    assert result["human_ball_relations"]["ball_vertical_trajectory"]["status"] == "insufficient_data"
    report["events"]["strict_ball_release"] = event("strict_ball_release", "insufficient_data", reason="test")
    result = build_motion_representation(report, rows)
    assert result["temporal_relations"]["takeoff_to_strict_ball_release"]["status"] == "insufficient_data"
    assert result["motion_primitives"]["hand_ball_separation"]["status"] == "insufficient_data"
    report["attempt"]["view"]["status"] = "unsupported_view"
    report["metrics"]["elbow_extension_onset_relative_to_release"]["view_requirement"] = "side_or_diagonal"
    result = build_motion_representation(report, rows)
    assert result["kinematics"]["metrics"]["elbow_extension_onset_relative_to_release"]["status"] == "unsupported_view"


def test_acceptance_manifest() -> None:
    manifest = json.loads((ROOT / "reference_v1" / "acceptance_manifest.v1.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "reference_v1_acceptance_manifest_v1"
    assert manifest["default_settings_required"] is True
    assert [item["id"] for item in manifest["samples"]] == ["IMG_7215", "IMG_7216", "BILI_010_A", "BILI_002_A", "BILI_010_B"]
    assert {item["classification"] for item in manifest["samples"]} <= {"SUPPORTED_GOOD_EVIDENCE", "SUPPORTED_PARTIAL_EVIDENCE", "EXPECTED_ABSTENTION", "OUT_OF_CONTRACT"}


def main() -> None:
    test_pose_gt_contract()
    test_motion_contract()
    test_acceptance_manifest()
    print("pose GT + motion tests: PASS")


if __name__ == "__main__":
    main()
