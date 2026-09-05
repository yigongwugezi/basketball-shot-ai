"""Train the once-frozen release_ball_v2 configuration and persist its identity."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed" / "yolo_release_ball_v2" / "data.yaml"
MODEL_ROOT = ROOT / "models" / "release_ball_v2"
MANIFEST = MODEL_ROOT / "training_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (MODEL_ROOT / "best.pt").exists():
        raise FileExistsError("release_ball_v2 already trained; refusing a second run")
    model = YOLO(str(ROOT / "yolo11n.pt"))
    result = model.train(data=str(DATASET), imgsz=config["image_size"], epochs=config["training_args"]["epochs"], batch=config["training_args"]["batch"], optimizer=config["training_args"]["optimizer"], seed=config["seed"], pretrained=config["training_args"]["pretrained"], project=str(ROOT / "runs" / "detect"), name="release_ball_v2", exist_ok=False)
    source = Path(result.save_dir) / "weights" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = MODEL_ROOT / "best.pt"
    shutil.copy2(source, destination)
    metadata = {"version": "release_ball_v2", "sha256": sha256(destination), "size_bytes": destination.stat().st_size, "model_architecture": config["architecture"], "dataset": config["dataset"], "seed": config["seed"], "image_size": config["image_size"], "evaluation_confidence_threshold": config["evaluation_confidence_threshold"], "training_args": config["training_args"], "source_training_run": str(result.save_dir), "trained_at_utc": datetime.now(timezone.utc).isoformat(), "training_manifest": "RELEASE_BALL_V2_TRAINING_MANIFEST.md"}
    (MODEL_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(MODEL_ROOT / "RELEASE_BALL_V2_TRAINING_MANIFEST.md", MODEL_ROOT / "training_manifest.md")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
