"""Validation and normalization for MovieLens and optional TMDB data."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

MOVIE_COLUMNS = {"movieId", "title", "genres"}
RATING_COLUMNS = {"userId", "movieId", "rating"}


def _list_value(value: Any) -> list[dict[str, Any]]:
    """Read JSON-like list fields from the TMDB CSVs without trusting their syntax."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
    return []


def _clean_names(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if value and str(value).strip()))


def parse_genres(value: Any) -> list[str]:
    """Parse MovieLens genre separators or TMDB genre JSON into unique names."""
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return _clean_names(item.get("name", "") for item in value)
        return _clean_names(value)
    if not isinstance(value, str) or not value.strip() or value.strip() == "(no genres listed)":
        return []
    parsed = _list_value(value)
    if parsed:
        return _clean_names(item.get("name", "") for item in parsed)
    return _clean_names(part for part in value.split("|") if part)


def preprocess_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate MovieLens catalog data and aggregate real ratings and user tags."""
    missing = MOVIE_COLUMNS.difference(movies.columns)
    if missing:
        raise ValueError(f"movies.csv is missing required columns: {', '.join(sorted(missing))}")

    frame = movies.copy()
    frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    frame = frame.dropna(subset=["movieId", "title"]).copy()
    frame["movieId"] = frame["movieId"].astype("int64")
    frame["title"] = frame["title"].astype(str).str.strip()
    frame = frame[frame["title"].ne("")].drop_duplicates("movieId", keep="first")
    frame["genres"] = frame["genres"].map(parse_genres)

    year_match = frame["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    frame["year"] = pd.to_numeric(year_match, errors="coerce").astype("Int64")
    frame["display_title"] = frame["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    frame["overview"] = ""
    frame["keywords"] = [[] for _ in range(len(frame))]
    frame["cast"] = [[] for _ in range(len(frame))]
    frame["director"] = ""
    frame["runtime"] = np.nan
    frame["language"] = ""
    frame["tags"] = [[] for _ in range(len(frame))]

    if ratings is not None and len(ratings):
        absent = RATING_COLUMNS.difference(ratings.columns)
        if absent:
            raise ValueError(f"ratings.csv is missing required columns: {', '.join(sorted(absent))}")
        clean_ratings = ratings[["movieId", "rating"]].copy()
        clean_ratings["movieId"] = pd.to_numeric(clean_ratings["movieId"], errors="coerce")
        clean_ratings["rating"] = pd.to_numeric(clean_ratings["rating"], errors="coerce")
        clean_ratings = clean_ratings.dropna(subset=["movieId", "rating"])
        clean_ratings = clean_ratings[clean_ratings["rating"].between(0.5, 5.0)]
        aggregates = clean_ratings.groupby("movieId")["rating"].agg(rating_average="mean", vote_count="count")
        frame = frame.merge(aggregates, how="left", left_on="movieId", right_index=True, validate="one_to_one")
    else:
        frame["rating_average"] = np.nan
        frame["vote_count"] = 0
    frame["vote_count"] = frame["vote_count"].fillna(0).astype("int64")

    if tags is not None and len(tags) and {"movieId", "tag"}.issubset(tags.columns):
        clean_tags = tags[["movieId", "tag"]].dropna().copy()
        clean_tags["movieId"] = pd.to_numeric(clean_tags["movieId"], errors="coerce")
        clean_tags = clean_tags.dropna(subset=["movieId"])
        tag_groups = clean_tags.groupby("movieId")["tag"].apply(lambda s: _clean_names(s.tolist()))
        frame["tags"] = frame["movieId"].map(tag_groups).map(lambda v: v if isinstance(v, list) else [])

    return frame.reset_index(drop=True)


def enrich_with_tmdb(
    movies: pd.DataFrame,
    links: pd.DataFrame,
    metadata: pd.DataFrame,
    credits: pd.DataFrame | None = None,
    keywords: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Safely left-join optional TMDB 5000 CSV enrichment using linked TMDB IDs."""
    if not {"movieId", "tmdbId"}.issubset(links.columns) or "id" not in metadata.columns:
        raise ValueError("TMDB enrichment requires links(movieId, tmdbId) and metadata(id) columns")
    link = links[["movieId", "tmdbId"]].copy()
    link["movieId"] = pd.to_numeric(link["movieId"], errors="coerce")
    link["tmdbId"] = pd.to_numeric(link["tmdbId"], errors="coerce")
    link = link.dropna(subset=["movieId", "tmdbId"]).drop_duplicates("movieId")

    meta = metadata.copy()
    meta["tmdbId"] = pd.to_numeric(meta["id"], errors="coerce")
    meta = meta.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId")
    allowed = [c for c in ("tmdbId", "overview", "runtime", "original_language", "popularity", "vote_average", "vote_count", "genres", "keywords") if c in meta]
    joined = movies.merge(link, on="movieId", how="left", validate="one_to_one").merge(meta[allowed], on="tmdbId", how="left", validate="many_to_one", suffixes=("", "_tmdb"))

    if "genres_tmdb" in joined:
        joined["genres"] = joined.apply(
            lambda row: list(dict.fromkeys(list(row["genres"] or []) + parse_genres(row["genres_tmdb"]))), axis=1
        )
        joined = joined.drop(columns=["genres_tmdb"])
    if "overview_tmdb" in joined:
        existing = joined["overview"].fillna("").astype(str) if "overview" in joined else pd.Series("", index=joined.index)
        joined["overview"] = existing.where(existing.str.strip().ne(""), joined["overview_tmdb"].fillna("").astype(str))
        joined = joined.drop(columns=["overview_tmdb"])
    elif "overview" in joined:
        joined["overview"] = joined["overview"].fillna("").astype(str)
    if "runtime_tmdb" in joined:
        existing = pd.to_numeric(joined["runtime"], errors="coerce") if "runtime" in joined else pd.Series(np.nan, index=joined.index)
        joined["runtime"] = existing.fillna(pd.to_numeric(joined["runtime_tmdb"], errors="coerce"))
        joined = joined.drop(columns=["runtime_tmdb"])
    if "keywords_tmdb" in joined:
        joined["keywords_tmdb"] = joined["keywords_tmdb"].map(lambda v: _clean_names(item.get("name") for item in _list_value(v)))
        joined["keywords"] = joined.apply(
            lambda row: row["keywords"] if row["keywords"] else row["keywords_tmdb"], axis=1
        )
        joined = joined.drop(columns=["keywords_tmdb"])
    if "runtime" in joined:
        joined["runtime"] = pd.to_numeric(joined["runtime"], errors="coerce")
    if "original_language" in joined:
        joined["language"] = joined["original_language"].fillna(joined.get("language", "")).astype(str)

    if credits is not None and ("id" in credits.columns or "movie_id" in credits.columns):
        credit = credits.copy()
        credit["tmdbId"] = pd.to_numeric(credit["id"] if "id" in credit else credit["movie_id"], errors="coerce")
        credit = credit.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId")
        credit["cast"] = credit.get("cast", pd.Series(index=credit.index, dtype=object)).map(
            lambda v: _clean_names(item.get("name") for item in _list_value(v)[:8])
        )
        credit["director"] = credit.get("crew", pd.Series(index=credit.index, dtype=object)).map(
            lambda v: next((str(item.get("name", "")) for item in _list_value(v) if item.get("job") == "Director"), "")
        )
        joined = joined.merge(credit[["tmdbId", "cast", "director"]], on="tmdbId", how="left", validate="many_to_one", suffixes=("", "_tmdb"))
        joined["cast"] = joined.apply(lambda row: row["cast"] if row["cast"] else (row["cast_tmdb"] if isinstance(row.get("cast_tmdb"), list) else []), axis=1)
        joined["director"] = joined.apply(lambda row: row["director"] or row.get("director_tmdb", ""), axis=1)
        joined = joined.drop(columns=[c for c in ("cast_tmdb", "director_tmdb") if c in joined])

    if keywords is not None and {"id", "keywords"}.issubset(keywords.columns):
        keyword_frame = keywords.copy()
        keyword_frame["tmdbId"] = pd.to_numeric(keyword_frame["id"], errors="coerce")
        keyword_frame = keyword_frame.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId")
        keyword_frame["keywords"] = keyword_frame["keywords"].map(lambda v: _clean_names(item.get("name") for item in _list_value(v)))
        joined = joined.merge(keyword_frame[["tmdbId", "keywords"]], on="tmdbId", how="left", validate="many_to_one", suffixes=("", "_tmdb"))
        joined["keywords"] = joined.apply(lambda row: row["keywords"] if row["keywords"] else (row["keywords_tmdb"] if isinstance(row.get("keywords_tmdb"), list) else []), axis=1)
        joined = joined.drop(columns=[c for c in ("keywords_tmdb",) if c in joined])

    joined["overview"] = joined.get("overview", "").fillna("") if isinstance(joined.get("overview", ""), pd.Series) else ""
    for col in ("keywords", "cast"):
        if col in joined:
            joined[col] = joined[col].map(lambda v: v if isinstance(v, list) else [])
    return joined.drop(columns=[c for c in ("tmdbId", "original_language") if c in joined]).reset_index(drop=True)
