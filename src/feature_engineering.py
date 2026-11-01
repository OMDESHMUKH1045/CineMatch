"""Build reusable, explainable movie features for content retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass
class ContentFeatures:
    """Feature matrices and source fields used for ranking and explanations."""

    text_matrix: csr_matrix
    overview_matrix: csr_matrix
    genre_sets: list[set[str]]
    keyword_sets: list[set[str]]
    tag_sets: list[set[str]]
    cast_sets: list[set[str]]
    directors: list[str]
    overview_available: list[bool]
    vectorizer: TfidfVectorizer | None


def build_content_features(movies: pd.DataFrame) -> ContentFeatures:
    """Create one TF-IDF text matrix plus exact-match taxonomy/person features."""
    text: list[str] = []
    overviews: list[str] = []
    genres: list[set[str]] = []
    keywords: list[set[str]] = []
    tags: list[set[str]] = []
    casts: list[set[str]] = []
    directors: list[str] = []
    overview_available: list[bool] = []
    for row in movies.to_dict("records"):
        overview = str(row.get("overview") or "").strip()
        overviews.append(overview)
        movie_keywords = _as_list(row.get("keywords"))
        movie_tags = _as_list(row.get("tags"))
        movie_cast = _as_list(row.get("cast"))
        director = str(row.get("director") or "").strip()
        genre_list = _as_list(row.get("genres"))
        # Repetition gives structured terms more influence than incidental overview words.
        text.append(" ".join([" ".join(genre_list)] * 2 + movie_keywords * 2 + movie_tags + movie_cast[:8] + [director] + [overview]))
        genres.append({v.casefold() for v in genre_list})
        keywords.append({v.casefold() for v in movie_keywords})
        tags.append({v.casefold() for v in movie_tags})
        casts.append({v.casefold() for v in movie_cast})
        directors.append(director.casefold())
        overview_available.append(bool(overview))

    # Titles are a safe last-resort source when every metadata field is empty.
    if not any(value.strip() for value in text):
        text = [str(row.get("display_title") or row.get("title") or "") for row in movies.to_dict("records")]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=120_000)
    try:
        matrix = vectorizer.fit_transform(text).tocsr()
    except ValueError:
        # An empty or punctuation-only catalog still has a valid zero-feature representation.
        from scipy.sparse import csr_matrix as _csr_matrix
        matrix = _csr_matrix((len(movies), 0), dtype=float)
        vectorizer = None
    if any(overview.strip() for overview in overviews):
        overview_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=60_000)
        try:
            overview_matrix = overview_vectorizer.fit_transform(overviews).tocsr()
        except ValueError:
            overview_matrix = csr_matrix((len(movies), 0), dtype=float)
    else:
        overview_matrix = csr_matrix((len(movies), 0), dtype=float)
    return ContentFeatures(matrix, overview_matrix, genres, keywords, tags, casts, directors, overview_available, vectorizer)
