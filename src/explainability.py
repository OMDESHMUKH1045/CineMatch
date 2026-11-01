"""Human-readable explanations grounded in the actual catalog fields."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .feature_engineering import ContentFeatures


def explain_recommendation(
    movies: pd.DataFrame,
    features: ContentFeatures,
    candidate_idx: int,
    seed_idx: int | None = None,
    preferences: dict[str, Any] | None = None,
    overview_similarity: float = 0.0,
    collaborative_seed: tuple[str, float] | None = None,
) -> list[str]:
    """Create only reasons supported by shared source fields or computed signals."""
    movie = movies.iloc[candidate_idx]
    reasons: list[str] = []
    if seed_idx is not None:
        base = movies.iloc[seed_idx]
        shared_genres = features.genre_sets[seed_idx] & features.genre_sets[candidate_idx]
        display_genres = [genre for genre in (base.get("genres") or []) if genre.casefold() in shared_genres]
        if display_genres:
            reasons.append("Shared genres: " + ", ".join(display_genres[:4]))
        keywords = sorted(features.keyword_sets[seed_idx] & features.keyword_sets[candidate_idx])
        tags = sorted(features.tag_sets[seed_idx] & features.tag_sets[candidate_idx])
        if keywords:
            reasons.append("Shared themes: " + ", ".join(keywords[:4]))
        if tags:
            reasons.append("Shared MovieLens community tags: " + ", ".join(tags[:3]))
        cast = sorted(features.cast_sets[seed_idx] & features.cast_sets[candidate_idx])
        if cast:
            reasons.append("Shared cast: " + ", ".join(cast[:3]))
        if features.directors[seed_idx] and features.directors[seed_idx] == features.directors[candidate_idx]:
            reasons.append("Same director: " + str(movie.get("director")))
        if overview_similarity >= 0.12 and features.overview_available[seed_idx] and features.overview_available[candidate_idx]:
            reasons.append(f"Similar plot descriptions (TF-IDF similarity {overview_similarity:.2f})")

    if preferences:
        selected = {str(value).casefold() for value in preferences.get("genres", [])}
        matched = [genre for genre in movie.get("genres", []) if genre.casefold() in selected]
        if matched:
            reasons.append("Matches your selected genres: " + ", ".join(matched[:4]))
        decades = set(preferences.get("decades", []))
        year = movie.get("year")
        if decades and pd.notna(year) and int(year) // 10 * 10 in decades:
            reasons.append(f"Released in your selected {int(year) // 10 * 10}s decade")
        languages = {str(value).casefold() for value in preferences.get("languages", [])}
        if languages and str(movie.get("language", "")).casefold() in languages:
            reasons.append(f"Matches your selected language: {movie.get('language')}")
        threshold = float(preferences.get("min_rating", 0) or 0)
        average = movie.get("rating_average")
        if threshold and pd.notna(average) and float(average) >= threshold:
            reasons.append(f"Meets your minimum MovieLens rating of {threshold:.1f}/5")
        runtime_pref = preferences.get("runtime")
        runtime = pd.to_numeric(pd.Series([movie.get("runtime")]), errors="coerce").iloc[0]
        runtime_bounds = {"Under 90 min": (0, 90), "90–120 min": (90, 120), "Over 120 min": (120, float("inf"))}
        if runtime_pref in runtime_bounds and pd.notna(runtime):
            low, high = runtime_bounds[runtime_pref]
            if low <= runtime < high:
                reasons.append(f"Matches your runtime preference ({runtime:.0f} min)")

    if collaborative_seed:
        source_title, source_rating = collaborative_seed
        reasons.append(f"Similar to {source_title}, which this MovieLens profile rated {source_rating:.1f}/5")

    average = movie.get("rating_average")
    votes = int(movie.get("vote_count") or 0)
    if pd.notna(average) and votes:
        reasons.append(f"Bayesian quality/popularity signal: {float(average):.1f}/5 from {votes:,} MovieLens ratings")
    if not reasons:
        reasons.append("Selected by the catalog-wide quality and rating-volume ranking")
    return reasons
