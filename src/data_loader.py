"""Load, validate, and enrich the local MovieLens catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import RAW_DATA_DIR
from .preprocessing import enrich_with_tmdb, preprocess_movies


@dataclass
class CatalogData:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame
    metadata_sources: list[str]


def find_movielens_dir(data_dir: Path = RAW_DATA_DIR) -> Path | None:
    """Find an extracted MovieLens folder in the configured raw data location."""
    candidates = [data_dir / "ml-latest-small", data_dir / "raw" / "ml-latest-small", data_dir / "movielens", data_dir]
    for candidate in candidates:
        if (candidate / "movies.csv").is_file():
            return candidate
    return None


def _read_optional(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required MovieLens file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_catalog(data_dir: Path = RAW_DATA_DIR, enrich: bool = True) -> CatalogData:
    """Load MovieLens and optionally join TMDB 5000 metadata from raw/.

    MovieLens remains the catalog and source of interaction-derived ratings.
    TMDB enrichment is joined only where the supplied links.csv has a matching id.
    """
    folder = find_movielens_dir(data_dir)
    if folder is None:
        raise FileNotFoundError(
            f"MovieLens data is missing. Download it with: python download_dataset.py\n"
            f"Expected movies.csv under {data_dir / 'ml-latest-small'} or {data_dir / 'raw' / 'ml-latest-small'}"
        )
    raw_movies = _read_optional(folder / "movies.csv", required=True)
    ratings = _read_optional(folder / "ratings.csv")
    tags = _read_optional(folder / "tags.csv")
    links = _read_optional(folder / "links.csv")
    movies = preprocess_movies(raw_movies, ratings if not ratings.empty else None, tags if not tags.empty else None)
    sources = ["MovieLens latest-small"]

    root = data_dir.parent if data_dir.name.casefold() == "raw" else data_dir
    if enrich and not links.empty:
        metadata_path = root / "tmdb_5000_movies.csv"
        credits_path = root / "tmdb_5000_credits.csv"
        keywords_path = root / "tmdb_5000_keywords.csv"
        if metadata_path.is_file():
            metadata = pd.read_csv(metadata_path, low_memory=False)
            credits = pd.read_csv(credits_path, low_memory=False) if credits_path.is_file() else None
            keywords = pd.read_csv(keywords_path, low_memory=False) if keywords_path.is_file() else None
            movies = enrich_with_tmdb(movies, links, metadata, credits, keywords)
            sources.append("TMDB 5000 metadata (optional local enrichment)")
    return CatalogData(movies, ratings, tags, links, sources)
