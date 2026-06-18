"""Download selected Roboflow Universe datasets for the shot analysis project.

Usage:
    $env:ROBOFLOW_API_KEY="your_key"
    python scripts/download_roboflow_datasets.py

The Roboflow workspace/project/version identifiers may need adjustment if a
Universe project changes. Prefer copying the official Python snippet from the
Roboflow "Download Dataset" modal and updating DATASETS below.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from roboflow import Roboflow


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = ROOT / "datasets" / "raw" / "public" / "roboflow"


DATASETS = [
    {
        "name": "basketball-and-hoop-detection",
        "workspace": "amrita-hlhw6",
        "project": "basketball-and-hoop-detection",
        "version": 1,
        "format": "yolov8",
    },
    {
        "name": "basketball-hoop-ball-and-player",
        "workspace": "personal-project-effcw",
        "project": "basketball-hoop-ball-and-player-5axdt",
        "version": 1,
        "format": "yolov8",
    },
]


def main() -> None:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing ROBOFLOW_API_KEY. Set it first, then rerun this script."
        )

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    for dataset in DATASETS:
        target_dir = DOWNLOAD_ROOT / dataset["name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {dataset['name']} to {target_dir}")

        for attempt in range(1, 6):
            try:
                workspace = rf.workspace(dataset["workspace"])
                project = workspace.project(dataset["project"])
                version = project.version(dataset["version"])
                version.download(dataset["format"], location=str(target_dir))
                break
            except Exception as exc:  # Roboflow can expose API keys in tracebacks.
                if attempt == 5:
                    raise SystemExit(
                        f"Failed to download {dataset['name']} after {attempt} attempts: "
                        f"{exc.__class__.__name__}. Check network, dataset access, or format."
                    ) from None
                print(
                    f"Attempt {attempt} failed with {exc.__class__.__name__}; retrying..."
                )
                time.sleep(5 * attempt)

    print("Done.")


if __name__ == "__main__":
    main()
