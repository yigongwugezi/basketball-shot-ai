from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.reference_v1.public_pose_gt import (
    CORE_JOINTS,
    ensure_large_data_root,
    evaluate_public_predictions,
    export_existing_schema,
    load_decisions,
    map_joints,
    record_decision,
)
from benchmarks.reference_v1.pose_gt import validate_annotations, validate_manifest


def test_mapping_and_crop_failure_accounting() -> None:
    names = list(CORE_JOINTS)
    coords = np.asarray([[10 + i, 20 + i] for i in range(12)], dtype=float)
    original, mapped = map_joints(names, coords, ["visible"] * 11 + ["occluded_but_inferable"])
    assert len(original) == 12
    assert mapped["left_shoulder"]["source_index"] == 0
    sample = {
        "id": "frame", "mapped_joints": mapped,
    }
    result = evaluate_public_predictions(
        {"samples": [sample]},
        {"model_id": "rtmpose", "frames": [{"frame_id": "frame", "person_crop_status": "FAILURE", "joints": {}}]},
    )
    assert result["person_or_crop_failure_joints"] == 12
    assert result["pose_head_missing_joints"] == 0
    assert result["end_to_end_failure_rate"] == 1.0


def test_user_review_gate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert load_decisions(root)["datasets"]["jhmdb"] == "PENDING"
        try:
            export_existing_schema(root, root / "benchmark", "jhmdb")
        except PermissionError as error:
            assert "must be ACCEPTED" in str(error)
        else:
            raise AssertionError("Pending dataset bypassed the user review gate")
        record_decision(root, "jhmdb", "REJECTED")
        assert load_decisions(root)["datasets"]["jhmdb"] == "REJECTED"
        try:
            ensure_large_data_root(root)
        except ValueError as error:
            assert "must use E:" in str(error)
        else:
            raise AssertionError("C-drive public data root was accepted")


def test_accepted_export_matches_existing_schema_and_rejects_pseudo_gt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        samples = []
        for index in range(30):
            joints = {
                name: {"status": "OK", "source_name": name, "source_index": joint_index, "x": 10.0 + joint_index, "y": 20.0 + joint_index, "visibility": "visible"}
                for joint_index, name in enumerate(CORE_JOINTS)
            }
            samples.append({
                "id": f"frame_{index:02d}", "sequence": "clip", "action": "test", "frame_index": index,
                "timestamp_seconds": float(index), "fps": 1.0, "width": 100, "height": 100,
                "media_path": "E:/synthetic/test.mp4", "mapped_joints": joints,
            })
        package = {"ground_truth_type": "HUMAN_GT", "license_class": "RESEARCH_ONLY", "samples": samples}
        package_path = root / "jhmdb" / "normalized_gt.candidate.json"
        package_path.parent.mkdir(parents=True)
        package_path.write_text(json.dumps(package), encoding="utf-8")
        record_decision(root, "jhmdb", "ACCEPTED")
        manifest_path, labels_path = export_existing_schema(root, root / "benchmark", "jhmdb")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        validate_annotations(manifest, labels, require_complete=True)
        package["ground_truth_type"] = "PSEUDO_LABEL"
        package_path.write_text(json.dumps(package), encoding="utf-8")
        try:
            export_existing_schema(root, root / "benchmark", "jhmdb")
        except ValueError as error:
            assert "Pseudo/automatic" in str(error)
        else:
            raise AssertionError("Pseudo labels were exported as trusted GT")


if __name__ == "__main__":
    test_mapping_and_crop_failure_accounting()
    test_user_review_gate()
    test_accepted_export_matches_existing_schema_and_rejects_pseudo_gt()
