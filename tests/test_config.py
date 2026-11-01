import pytest

from src.config import RecommendationWeights


def test_recommendation_weights_reject_negative_and_nonfinite_values():
    with pytest.raises(ValueError, match="finite and non-negative"):
        RecommendationWeights(content=-0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        RecommendationWeights(popularity=float("nan"))
