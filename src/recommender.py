"""Hybrid content, popularity, explicit-preference, and optional rating CF."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .config import DEFAULT_WEIGHTS, RecommendationWeights
from .feature_engineering import ContentFeatures, build_content_features
from .explainability import explain_recommendation
from .ranking import filter_by_preferences, preference_match_score, weighted_popularity_score


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right and left | right else 0.0


class HybridRecommender:
    """Reusable scorer with precomputed catalog features and user-item neighbors."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame | None = None,
        weights: RecommendationWeights = DEFAULT_WEIGHTS,
        neighbor_count: int = 50,
    ) -> None:
        self.movies = movies.reset_index(drop=True).copy()
        self.ratings = ratings.copy() if ratings is not None else pd.DataFrame()
        self.weights = weights
        self.features: ContentFeatures = build_content_features(self.movies)
        self.popularity = weighted_popularity_score(self.movies)
        self._indices = {int(movie_id): idx for idx, movie_id in enumerate(self.movies["movieId"])}
        self._rating_rows: dict[int, pd.DataFrame] = {}
        self._knn: NearestNeighbors | None = None
        self._item_user: csr_matrix | None = None
        self._prepare_collaborative(neighbor_count)

    def _prepare_collaborative(self, neighbor_count: int) -> None:
        required = {"userId", "movieId", "rating"}
        if not required.issubset(self.ratings.columns) or self.ratings.empty:
            return
        ratings = self.ratings[list(required)].copy()
        ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
        ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce")
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings = ratings.dropna().drop_duplicates(["userId", "movieId"], keep="last")
        ratings = ratings[ratings["movieId"].astype(int).isin(self._indices)]
        if ratings.empty:
            return
        self._rating_rows = {int(user_id): group for user_id, group in ratings.groupby("userId")}
        user_ids = sorted(ratings["userId"].astype(int).unique())
        user_index = {user_id: i for i, user_id in enumerate(user_ids)}
        row_indices = ratings["movieId"].astype(int).map(self._indices).to_numpy()
        col_indices = ratings["userId"].astype(int).map(user_index).to_numpy()
        # Explicit ratings below 3 are kept as non-preference (zero) in neighborhood
        # vectors; high ratings contribute more evidence than marginal positives.
        values = np.maximum(ratings["rating"].to_numpy(dtype=float) - 3.0, 0.0) / 2.0
        positive = values > 0
        self._item_user = csr_matrix(
            (values[positive], (row_indices[positive], col_indices[positive])),
            shape=(len(self.movies), len(user_ids)),
        )
        self._knn = NearestNeighbors(
            n_neighbors=min(max(2, neighbor_count + 1), len(self.movies)),
            metric="cosine",
            algorithm="brute",
        )
        self._knn.fit(self._item_user)

    @property
    def collaborative_available(self) -> bool:
        return self._knn is not None and bool(self._rating_rows)

    @property
    def user_ids(self) -> list[int]:
        return sorted(self._rating_rows, key=lambda uid: (-len(self._rating_rows[uid]), uid))

    def user_summary(self, user_id: int) -> dict[str, Any]:
        rows = self._rating_rows.get(int(user_id))
        if rows is None:
            return {"rating_count": 0, "mean_rating": np.nan, "liked_titles": []}
        liked = rows[rows["rating"].ge(4.0)].sort_values("rating", ascending=False)
        titles = [self.movies.iloc[self._indices[int(mid)]]["display_title"] for mid in liked["movieId"] if int(mid) in self._indices]
        return {"rating_count": len(rows), "mean_rating": float(rows["rating"].mean()), "liked_titles": titles[:5]}

    def _content_scores(self, seed_idx: int) -> dict[str, np.ndarray]:
        seed = self.features
        count = len(self.movies)
        genre = np.array([_jaccard(seed.genre_sets[seed_idx], values) for values in seed.genre_sets], dtype=float)
        cast = np.array([_jaccard(seed.cast_sets[seed_idx], values) for values in seed.cast_sets], dtype=float)
        director = np.array(
            [1.0 if seed.directors[seed_idx] and seed.directors[seed_idx] == value else 0.0 for value in seed.directors],
            dtype=float,
        )
        if seed.text_matrix.shape[1]:
            text = np.asarray((seed.text_matrix.getrow(seed_idx) @ seed.text_matrix.T).toarray()).ravel()
        else:
            text = np.zeros(count, dtype=float)
        if seed.overview_matrix.shape[1]:
            overview = np.asarray((seed.overview_matrix.getrow(seed_idx) @ seed.overview_matrix.T).toarray()).ravel()
        else:
            overview = np.zeros(count, dtype=float)

        signals: list[tuple[np.ndarray, float]] = []
        if seed.genre_sets[seed_idx]:
            signals.append((genre, 0.34))
        if seed.text_matrix.shape[1] and seed.text_matrix.getrow(seed_idx).nnz:
            signals.append((text, 0.46))
        if seed.cast_sets[seed_idx]:
            signals.append((cast, 0.15))
        if seed.directors[seed_idx]:
            signals.append((director, 0.05))
        total = sum(weight for _, weight in signals)
        content = sum((arr * weight for arr, weight in signals), np.zeros(count)) / total if total else np.zeros(count)
        return {"content_score": content, "genre_similarity": genre, "text_similarity": text, "overview_similarity": overview, "cast_similarity": cast, "director_match": director}

    def collaborative_scores(self, user_id: int) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
        """Aggregate item-KNN neighbors of the user's actual positive ratings."""
        scores = np.zeros(len(self.movies), dtype=float)
        evidence: dict[int, tuple[int, float]] = {}
        rows = self._rating_rows.get(int(user_id))
        if rows is None or self._knn is None or self._item_user is None:
            return scores, evidence
        liked = rows[rows["rating"].ge(4.0)]
        if liked.empty:
            return scores, evidence
        seed_indices = np.array([self._indices[int(mid)] for mid in liked["movieId"] if int(mid) in self._indices], dtype=int)
        if not len(seed_indices):
            return scores, evidence
        liked_values = {self._indices[int(row.movieId)]: float(row.rating) for row in liked.itertuples() if int(row.movieId) in self._indices}
        distances, neighbors = self._knn.kneighbors(self._item_user[seed_indices], return_distance=True)
        evidence_weight = np.zeros(len(scores), dtype=float)
        evidence_seed: dict[int, tuple[int, float]] = {}
        for source_idx, distance_row, neighbor_row in zip(seed_indices, distances, neighbors):
            preference_strength = max(liked_values.get(int(source_idx), 4.0) - 3.0, 0.0) / 2.0
            for distance, candidate_idx in zip(distance_row, neighbor_row):
                if candidate_idx == source_idx:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                contribution = similarity * preference_strength
                scores[candidate_idx] += contribution
                evidence_weight[candidate_idx] += similarity
                if contribution > evidence_seed.get(int(candidate_idx), (-1, 0.0))[1]:
                    evidence_seed[int(candidate_idx)] = (int(source_idx), contribution)
        nonzero = evidence_weight > 0
        scores[nonzero] /= evidence_weight[nonzero]
        maximum = scores.max(initial=0)
        if maximum:
            scores /= maximum
        return scores, evidence_seed

    def _explain(
        self,
        seed_idx: int | None,
        candidate_idx: int,
        preferences: dict[str, Any] | None = None,
        cf_evidence: dict[int, tuple[int, float]] | None = None,
        overview_similarity: float = 0.0,
    ) -> list[str]:
        collaborative_seed = None
        if cf_evidence and candidate_idx in cf_evidence:
            source_idx = cf_evidence[candidate_idx][0]
            source = self.movies.iloc[source_idx]
            user_id = preferences.get("user_id") if preferences else None
            user_rows = self._rating_rows.get(int(user_id)) if user_id is not None else None
            given = user_rows[user_rows["movieId"].astype(int).eq(int(source["movieId"]))]["rating"] if user_rows is not None else pd.Series(dtype=float)
            if not given.empty:
                collaborative_seed = (str(source["display_title"]), float(given.iloc[0]))
        return explain_recommendation(
            self.movies, self.features, candidate_idx, seed_idx, preferences,
            overview_similarity=overview_similarity, collaborative_seed=collaborative_seed,
        )

    def _build_result(
        self,
        candidate_indices: np.ndarray,
        seed_idx: int | None,
        preferences: dict[str, Any] | None,
        user_id: int | None,
        limit: int,
    ) -> pd.DataFrame:
        if len(candidate_indices) == 0:
            return self.movies.iloc[[]].copy()
        indices = candidate_indices.astype(int)
        result = self.movies.iloc[indices].copy()
        result["popularity_score"] = self.popularity.iloc[indices].to_numpy()
        result["preference_match"] = [preference_match_score(self.movies.iloc[i].to_dict(), preferences or {}) for i in indices]
        cf_scores = np.zeros(len(self.movies), dtype=float)
        evidence: dict[int, tuple[int, float]] = {}
        if user_id is not None:
            cf_scores, evidence = self.collaborative_scores(user_id)
        result["collaborative_score"] = cf_scores[indices]

        components = self._content_scores(seed_idx) if seed_idx is not None else {
            "content_score": np.zeros(len(self.movies)), "genre_similarity": np.zeros(len(self.movies)),
            "text_similarity": np.zeros(len(self.movies)), "overview_similarity": np.zeros(len(self.movies)), "cast_similarity": np.zeros(len(self.movies)),
            "director_match": np.zeros(len(self.movies)),
        }
        for key, arr in components.items():
            result[key] = arr[indices]

        signals: list[tuple[str, float]] = []
        if seed_idx is not None:
            signals.extend([("content_score", self.weights.content), ("popularity_score", self.weights.popularity)])
        else:
            signals.append(("popularity_score", self.weights.popularity))
        preference_active = preferences is not None and any(preferences.get(key) for key in ("genres", "decades", "languages", "runtime"))
        if preference_active:
            signals.append(("preference_match", self.weights.preferences))
        if user_id is not None and self.collaborative_available:
            signals.append(("collaborative_score", self.weights.collaborative))
        weight_total = sum(weight for _, weight in signals)
        if weight_total <= 0:
            signals = [("popularity_score", 1.0)]
            weight_total = 1.0
        result["hybrid_score"] = sum(result[key] * weight for key, weight in signals) / weight_total
        reason_preferences = dict(preferences or {})
        if user_id is not None:
            reason_preferences["user_id"] = user_id
        if not reason_preferences:
            reason_preferences = None
        result = result.sort_values(["hybrid_score", "popularity_score", "display_title"], ascending=[False, False, True]).head(limit).copy()
        result["explanations"] = [
            self._explain(seed_idx, int(index), reason_preferences, evidence, float(components["overview_similarity"][index]))
            for index in result.index
        ]
        return result.reset_index(drop=True)

    def recommend_similar(
        self,
        movie_id: int,
        limit: int = 10,
        preferences: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> pd.DataFrame:
        """Recommend movies similar to a selected movie, with no self-match."""
        movie_id = int(movie_id)
        if movie_id not in self._indices:
            return self.movies.iloc[[]].copy()
        seed_idx = self._indices[movie_id]
        candidates = self.movies
        if preferences:
            candidates = filter_by_preferences(candidates, preferences)
        candidate_indices = candidates.index.to_numpy(dtype=int)
        candidate_indices = candidate_indices[candidate_indices != seed_idx]
        return self._build_result(candidate_indices, seed_idx, preferences, user_id, limit)

    def recommend_personalized(
        self,
        preferences: dict[str, Any],
        limit: int = 12,
        user_id: int | None = None,
    ) -> pd.DataFrame:
        """Rank movies from explicit taste controls and/or an actual ML profile."""
        filtered = filter_by_preferences(self.movies, preferences)
        candidates = filtered.index.to_numpy(dtype=int)
        if user_id is not None:
            rows = self._rating_rows.get(int(user_id))
            if rows is not None:
                rated_indices = {self._indices[int(mid)] for mid in rows["movieId"] if int(mid) in self._indices}
                candidates = np.array([idx for idx in candidates if idx not in rated_indices], dtype=int)
        return self._build_result(candidates, None, preferences, user_id, limit)
