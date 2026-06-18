from __future__ import annotations

import shutil
from pathlib import Path


DATASET = "sarbagyashakya/basketball-51-dataset"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "raw" / "public" / "video" / "basketball-51"


def main() -> None:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        raise SystemExit(
            "没有找到 Kaggle API token：%USERPROFILE%\\.kaggle\\kaggle.json\n"
            "去 Kaggle -> Account -> Create New API Token 下载 kaggle.json，"
            "然后放到这个位置，再重新运行脚本。"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 kaggle 包。请先运行：.venv310\\Scripts\\python.exe -m pip install kaggle"
        ) from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} ...")
    api.dataset_download_files(DATASET, path=str(OUT_DIR), unzip=True, quiet=False)

    zip_path = OUT_DIR / "basketball-51-dataset.zip"
    if zip_path.exists():
        zip_path.unlink()

    video_count = 0
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        video_count += len(list(OUT_DIR.rglob(ext)))

    print(f"Done: {OUT_DIR}")
    print(f"Video files found: {video_count}")

    empty_dirs = [p for p in OUT_DIR.rglob("*") if p.is_dir() and not any(p.iterdir())]
    for path in empty_dirs:
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
