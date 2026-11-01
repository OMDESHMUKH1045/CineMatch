"""Download the small, official GroupLens MovieLens dataset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.config import RAW_DATA_DIR

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_dataset(destination: Path = RAW_DATA_DIR) -> Path:
    """Download and extract MovieLens latest-small unless already present."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "ml-latest-small"
    if (target / "movies.csv").is_file():
        print(f"Dataset already present: {target}")
        return target
    archive = destination / "ml-latest-small.zip"
    temporary = destination / "ml-latest-small.zip.part"
    try:
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "CineMatch/1.0 dataset downloader"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
        with zipfile.ZipFile(archive) as zipped:
            if not any(name.startswith("ml-latest-small/movies.csv") for name in zipped.namelist()):
                raise ValueError("Downloaded archive did not contain the expected MovieLens catalog")
            zipped.extractall(destination)
    finally:
        temporary.unlink(missing_ok=True)
    archive.unlink(missing_ok=True)
    print(f"Downloaded MovieLens data to {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()
    download_dataset(args.destination)
