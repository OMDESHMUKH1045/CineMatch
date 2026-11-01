import pandas as pd

from src.preprocessing import preprocess_movies
from src.ranking import weighted_popularity_score
from src.search import search_movies, title_suggestions


def test_search_is_case_insensitive_and_supports_partial_title(movie_rows):
    movies = preprocess_movies(movie_rows)
    result = search_movies("orbit", movies)
    assert set(result.movieId) == {1, 2}
    assert search_movies("ORBIT ONE", movies).iloc[0].movieId == 1


def test_search_understands_movielens_trailing_articles():
    movies = preprocess_movies(pd.DataFrame([{"movieId": 1, "title": "Matrix, The (1999)", "genres": "Sci-Fi"}]))
    assert search_movies("The Matrix", movies).iloc[0].movieId == 1


def test_no_match_returns_empty_and_typo_suggestions():
    movies = preprocess_movies(pd.DataFrame([{"movieId": 1, "title": "Inception (2010)", "genres": "Sci-Fi"}]))
    assert search_movies("missing", movies).empty
    assert title_suggestions("Incepton", movies) == ["Inception"]


def test_search_ranks_popular_matches_first(movie_rows):
    movies = preprocess_movies(movie_rows)
    popularity = weighted_popularity_score(pd.DataFrame({"rating_average": [3.0, 4.8, 4.0, 4.0], "vote_count": [2, 100, 10, 1]}))
    result = search_movies("it", movies, popularity)
    assert result.iloc[0].movieId == 2
