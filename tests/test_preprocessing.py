import pandas as pd
import pytest

from src.preprocessing import enrich_with_tmdb, preprocess_movies


def test_missing_required_columns_raise_helpful_error():
    with pytest.raises(ValueError, match="missing required columns"):
        preprocess_movies(pd.DataFrame({"movieId": [1], "title": ["Title"]}))


def test_duplicate_ids_malformed_rows_and_missing_values_are_handled():
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "Film (2001)", "genres": "Drama|Comedy"},
            {"movieId": 1, "title": "Duplicate (2001)", "genres": "Drama"},
            {"movieId": "bad", "title": "Malformed", "genres": "Drama"},
            {"movieId": 2, "title": None, "genres": None},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 4.0},
            {"userId": 2, "movieId": 1, "rating": 6.0},
            {"userId": 3, "movieId": 1, "rating": "bad"},
        ]
    )
    result = preprocess_movies(movies, ratings)
    assert len(result) == 1
    assert result.loc[0, "year"] == 2001
    assert result.loc[0, "genres"] == ["Drama", "Comedy"]
    assert result.loc[0, "rating_average"] == 4.0
    assert result.loc[0, "vote_count"] == 1


def test_real_tag_aggregation_and_movies_without_votes(movie_rows):
    tags = pd.DataFrame(
        [{"userId": 5, "movieId": 1, "tag": "space"}, {"userId": 6, "movieId": 1, "tag": "space"}, {"userId": 6, "movieId": 2, "tag": "survival"}]
    )
    result = preprocess_movies(movie_rows, tags=tags)
    assert result.loc[result.movieId.eq(1), "tags"].iloc[0] == ["space"]
    assert result.loc[result.movieId.eq(3), "vote_count"].iloc[0] == 0
    assert pd.isna(result.loc[result.movieId.eq(3), "rating_average"].iloc[0])


def test_tmdb_join_uses_numeric_link_ids_and_parses_real_metadata(movie_rows):
    base = preprocess_movies(movie_rows)
    links = pd.DataFrame([{"movieId": 1, "tmdbId": "900"}])
    metadata = pd.DataFrame(
        [{"id": 900, "overview": "A journey through space.", "runtime": 101, "original_language": "en", "genres": "[{\"id\": 878, \"name\": \"Science Fiction\"}]", "keywords": "[{\"id\": 1, \"name\": \"space travel\"}]"}]
    )
    credits = pd.DataFrame(
        [{"movie_id": 900, "cast": "[{\"name\": \"A. Actor\"}]", "crew": "[{\"job\": \"Director\", \"name\": \"D. Director\"}]"}]
    )
    result = enrich_with_tmdb(base, links, metadata, credits)
    enriched = result.loc[result.movieId.eq(1)].iloc[0]
    assert enriched.overview == "A journey through space."
    assert enriched.runtime == 101
    assert enriched.cast == ["A. Actor"]
    assert enriched.director == "D. Director"
    assert "Science Fiction" in enriched.genres
    assert enriched.keywords == ["space travel"]
    assert len(result) == len(base)
