import pandas as pd

from src.preprocessing import preprocess_movies
from src.recommender import HybridRecommender


def test_recommendations_use_features_and_never_return_seed(movie_rows, rating_rows):
    tags = pd.DataFrame(
        [{"userId": 3, "movieId": 1, "tag": "space survival"}, {"userId": 4, "movieId": 2, "tag": "space survival"}]
    )
    movies = preprocess_movies(movie_rows, rating_rows, tags)
    movies.loc[movies.movieId.eq(1), "overview"] = "A crew struggles to survive a journey through space."
    movies.loc[movies.movieId.eq(2), "overview"] = "A crew travels through space and fights to survive."
    engine = HybridRecommender(movies, rating_rows)
    result = engine.recommend_similar(1, limit=3)
    assert 1 not in result.movieId.tolist()
    assert result.iloc[0].movieId == 2
    reasons = result.loc[result.movieId.eq(2), "explanations"].iloc[0]
    assert any("Shared genres" in reason for reason in reasons)
    assert any("MovieLens community tags" in reason for reason in reasons)
    assert any("Similar plot descriptions" in reason for reason in reasons)


def test_explanations_do_not_claim_unavailable_plot_data(movie_rows):
    movies = preprocess_movies(movie_rows)
    engine = HybridRecommender(movies)
    result = engine.recommend_similar(1, limit=2)
    assert all("plot descriptions" not in reason for reasons in result.explanations for reason in reasons)


def test_explanation_lists_only_preferences_the_movie_meets(movie_rows, rating_rows):
    movies = preprocess_movies(movie_rows, rating_rows)
    engine = HybridRecommender(movies, rating_rows)
    result = engine.recommend_personalized({"genres": ["Adventure"], "min_rating": 4.0}, limit=4)
    reasons = result.loc[result.movieId.eq(1), "explanations"].iloc[0]
    assert any("Matches your selected genres" in reason for reason in reasons)
    assert any("minimum MovieLens rating of 4.0" in reason for reason in reasons)


def test_profile_recommendations_are_personalized_and_exclude_rated_titles(movie_rows, rating_rows):
    movies = preprocess_movies(movie_rows, rating_rows)
    engine = HybridRecommender(movies, rating_rows)
    result = engine.recommend_personalized({"genres": ["Science Fiction"]}, limit=4, user_id=10)
    assert result.movieId.isin([1, 2]).sum() == 0
    assert engine.collaborative_available
    assert 10 in engine.user_ids
    assert any("MovieLens profile rated" in reason for reasons in result.explanations for reason in reasons)


def test_unknown_seed_and_missing_ratings_are_safe(movie_rows):
    movies = preprocess_movies(movie_rows)
    engine = HybridRecommender(movies)
    assert engine.recommend_similar(999).empty
    assert not engine.collaborative_available
    assert engine.recommend_personalized({"genres": ["Drama"]}).movieId.tolist() == [3]
