import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess_movies
from src.ranking import filter_by_preferences, preference_match_score, weighted_popularity_score


def test_bayesian_vote_weighted_score_is_bounded_and_rewards_vote_volume():
    frame = pd.DataFrame(
        {"rating_average": [5.0, 4.5, 3.5, np.nan], "vote_count": [1, 500, 100, 0]}
    )
    scores = weighted_popularity_score(frame)
    assert scores.between(0, 1).all()
    assert scores.iloc[1] > scores.iloc[0]
    # With no personal rating evidence, Bayesian quality falls back to the catalog mean.
    assert scores.iloc[3] == pytest.approx(0.75 * frame.rating_average.mean() / 5)


def test_profile_filters_genre_decade_rating_runtime_and_language():
    movies = pd.DataFrame(
        [
            {"movieId": 1, "genres": ["Sci-Fi"], "year": 1995, "rating_average": 4.4, "runtime": 100, "language": "en"},
            {"movieId": 2, "genres": ["Drama"], "year": 2004, "rating_average": 4.8, "runtime": 80, "language": "fr"},
            {"movieId": 3, "genres": ["Sci-Fi"], "year": 1992, "rating_average": 3.0, "runtime": 140, "language": "en"},
            {"movieId": 4, "genres": ["Sci-Fi"], "year": np.nan, "rating_average": 4.7, "runtime": 100, "language": "en"},
        ]
    )
    filtered = filter_by_preferences(movies, {"genres": ["sci-fi"], "decades": [1990], "min_rating": 4.0, "runtime": "90–120 min", "languages": ["EN"]})
    assert filtered.movieId.tolist() == [1]
    assert preference_match_score(movies.iloc[0], {"genres": ["Sci-Fi"], "decades": [1990]}) == 1.0


def test_preprocessing_preserves_unknown_runtime_as_missing(movie_rows):
    frame = preprocess_movies(movie_rows)
    assert frame.runtime.isna().all()
