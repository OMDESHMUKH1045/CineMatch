"""Runtime configuration for CineMatch."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("CINEMATCH_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
RAW_DATA_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class RecommendationWeights:
    """Configurable hybrid scoring weights; normalized at scoring time."""

    content: float = 0.55
    popularity: float = 0.30
    preferences: float = 0.15
    collaborative: float = 0.35

    def __post_init__(self) -> None:
        values = (self.content, self.popularity, self.preferences, self.collaborative)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Recommendation weights must be finite and non-negative")


DEFAULT_WEIGHTS = RecommendationWeights(
    content=float(os.getenv("CINEMATCH_CONTENT_WEIGHT", "0.55")),
    popularity=float(os.getenv("CINEMATCH_POPULARITY_WEIGHT", "0.30")),
    preferences=float(os.getenv("CINEMATCH_PREFERENCE_WEIGHT", "0.15")),
    collaborative=float(os.getenv("CINEMATCH_COLLABORATIVE_WEIGHT", "0.35")),
)
