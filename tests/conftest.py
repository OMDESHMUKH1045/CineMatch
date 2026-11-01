"""Small synthetic fixtures used only to test CineMatch logic."""

import pandas as pd
import pytest


@pytest.fixture
def movie_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Orbit One (2001)", "genres": "Adventure|Science Fiction"},
            {"movieId": 2, "title": "Beyond Orbit (2002)", "genres": "Adventure|Science Fiction"},
            {"movieId": 3, "title": "Quiet Summer (2003)", "genres": "Drama|Romance"},
            {"movieId": 4, "title": "Space Voyage (2004)", "genres": "Science Fiction"},
        ]
    )


@pytest.fixture
def rating_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": 10, "movieId": 1, "rating": 5.0},
            {"userId": 10, "movieId": 2, "rating": 4.5},
            {"userId": 10, "movieId": 3, "rating": 2.0},
            {"userId": 11, "movieId": 1, "rating": 4.5},
            {"userId": 11, "movieId": 4, "rating": 5.0},
            {"userId": 12, "movieId": 2, "rating": 4.0},
            {"userId": 12, "movieId": 4, "rating": 4.5},
        ]
    )
