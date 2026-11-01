"""Popularity and explicit taste-profile scoring/filtering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def weighted_popularity_score(movies: pd.DataFrame, prior_vote_count: float | None = None) -> pd.Series:
    """Bayesian quality plus log vote-volume score, scaled to [0, 1].

    Bayesian quality = (v / (v + m)) * R + (m / (v + m)) * C, where R is the
    movie's mean rating, v its real rating count, C the catalog mean, and m the
    60th percentile vote-count prior (at least 5). Final score is 75% normalized
    Bayesian quality and 25% normalized log vote volume. It is not raw rating sort.
    """
    if movies.empty:
        return pd.Series(dtype=float, index=movies.index)
    ratings = pd.to_numeric(movies.get("rating_average", pd.Series(np.nan, index=movies.index)), errors="coerce")
    votes = pd.to_numeric(movies.get("vote_count", pd.Series(0, index=movies.index)), errors="coerce").fillna(0).clip(lower=0)
    valid = ratings.notna() & votes.gt(0)
    global_mean = float(ratings[valid].mean()) if valid.any() else 0.0
    prior = float(prior_vote_count) if prior_vote_count is not None else float(votes[votes.gt(0)].quantile(0.60) if votes.gt(0).any() else 5)
    prior = max(prior, 5.0)
    bayes = ((votes / (votes + prior)) * ratings.fillna(global_mean) + (prior / (votes + prior)) * global_mean).clip(0, 5)
    quality = bayes / 5.0
    max_log_votes = float(np.log1p(votes).max()) if len(votes) else 0
    volume = np.log1p(votes) / max_log_votes if max_log_votes > 0 else pd.Series(0.0, index=movies.index)
    score = (0.75 * quality + 0.25 * volume).fillna(0.0)
    return score.clip(0.0, 1.0)


def preference_match_score(row: Mapping[str, Any], preferences: Mapping[str, Any]) -> float:
    """Return a deterministic 0..1 match score from explicit user-selected tastes."""
    components: list[float] = []
    selected_genres = {str(x).casefold() for x in preferences.get("genres", [])}
    movie_genres = {str(x).casefold() for x in row.get("genres", [])} if isinstance(row.get("genres", []), list) else set()
    if selected_genres:
        components.append(len(selected_genres & movie_genres) / len(selected_genres))

    selected_decades = set(preferences.get("decades", []))
    year = row.get("year")
    if selected_decades:
        if pd.notna(year):
            decade = (int(year) // 10) * 10
            components.append(1.0 if decade in selected_decades else 0.0)
        else:
            components.append(0.0)

    languages = {str(x).casefold() for x in preferences.get("languages", [])}
    language = str(row.get("language") or "").casefold()
    if languages:
        components.append(1.0 if language in languages else 0.0)

    runtime_pref = preferences.get("runtime")
    runtime = np.nan
    if runtime_pref and runtime_pref != "Any":
        try:
            runtime = float(row.get("runtime"))
        except (TypeError, ValueError):
            pass
    if runtime_pref and runtime_pref != "Any" and math.isfinite(runtime):
        bounds = {"Under 90 min": (0, 90), "90–120 min": (90, 120), "Over 120 min": (120, math.inf)}
        low, high = bounds[runtime_pref]
        components.append(1.0 if low <= runtime < high else 0.0)

    if not components:
        return 0.5
    return float(np.mean(components))


def filter_by_preferences(movies: pd.DataFrame, preferences: Mapping[str, Any]) -> pd.DataFrame:
    """Filter the catalog using explicitly selected genres, decade, rating, etc."""
    result = movies.copy()
    selected_genres = {str(x).casefold() for x in preferences.get("genres", [])}
    if selected_genres:
        result = result[result["genres"].map(lambda xs: bool(selected_genres.intersection({str(x).casefold() for x in xs})) if isinstance(xs, list) else False)]

    selected_decades = set(preferences.get("decades", []))
    if selected_decades and "year" in result:
        years = pd.to_numeric(result["year"], errors="coerce")
        decade_matches = years.map(lambda year: pd.notna(year) and int(year) // 10 * 10 in selected_decades)
        result = result[years.notna() & decade_matches]

    min_rating = float(preferences.get("min_rating", 0) or 0)
    if min_rating:
        result = result[pd.to_numeric(result["rating_average"], errors="coerce").ge(min_rating)]

    runtime_pref = preferences.get("runtime")
    if runtime_pref and runtime_pref != "Any" and "runtime" in result:
        runtime = pd.to_numeric(result["runtime"], errors="coerce")
        bounds = {"Under 90 min": (0, 90), "90–120 min": (90, 120), "Over 120 min": (120, math.inf)}
        low, high = bounds[runtime_pref]
        result = result[runtime.notna() & runtime.ge(low) & runtime.lt(high)]

    languages = {str(x).casefold() for x in preferences.get("languages", [])}
    if languages and "language" in result:
        result = result[result["language"].fillna("").str.casefold().isin(languages)]
    return result
