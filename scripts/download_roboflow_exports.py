"""Download Roboflow exports through the dataset export links.

This avoids relying on Roboflow SDK extraction behavior. API keys and temporary
download links are never printed.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import requests


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


def api_get_export(dataset: dict[str, object], api_key: str) -> dict[str, object]:
    url = (
        "https://api.roboflow.com/"
        f"{dataset['workspace']}/{dataset['project']}/{dataset['version']}/"
        f"{dataset['format']}"
    )
    response = requests.get(url, params={"api_key": api_key}, timeout=90)
    response.raise_for_status()
    payload = response.json()
    export = payload.get("export")
    if not isinstance(export, dict) or not export.get("link"):
        raise RuntimeError(f"No export link returned for {dataset['name']}")
    return export


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        downloaded = 0
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(f"  {percent:5.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="\r")
        print()


def extract_zip(zip_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def main() -> None:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY.")

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        name = str(dataset["name"])
        target_dir = DOWNLOAD_ROOT / name
        zip_path = DOWNLOAD_ROOT / f"{name}.zip"

        print(f"Requesting export for {name}")
        export = api_get_export(dataset, api_key)
        size_mb = export.get("size")
        if isinstance(size_mb, (int, float)):
            print(f"Downloading {name} ({size_mb:.1f} MB)")
        else:
            print(f"Downloading {name}")

        download_file(str(export["link"]), zip_path)
        print(f"Extracting {name}")
        extract_zip(zip_path, target_dir)
        zip_path.unlink(missing_ok=True)

    print("All Roboflow exports downloaded and extracted.")


if __name__ == "__main__":
    main()
