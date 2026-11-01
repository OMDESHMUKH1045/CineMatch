# CineMatch

**Intelligent Movie Discovery & Recommendation Platform**

CineMatch is a polished Streamlit product and a reproducible recommendation pipeline. Search a real movie catalog, explore by taste, inspect similar films, and see which observed features influenced a recommendation.

## Overview

Movie catalogs create an information-overload problem: a large collection makes it hard to find a relevant next film. CineMatch combines catalog metadata, rating evidence, and explicit session preferences so that discovery can be both useful and understandable.

## Features

- Case-insensitive partial title search with typo suggestions.
- Similar-movie discovery using genre overlap, TF-IDF text, shared tags and keywords, cast, and director when those fields exist.
- A Bayesian rating and rating-volume ranking instead of sorting by average rating alone.
- Session-only genre, decade, rating, runtime, and language preferences.
- Optional item-based collaborative filtering trained on the actual MovieLens user-rating matrix.
- A “Why you'll like this” explanation panel built from shared source fields and computed recommendation signals.
- Model-insights page with catalog statistics, feature notes, ranking formula, and limitations.
- Optional TMDB 5000 metadata enrichment for plot summaries, runtime, language, cast, director, and keywords.

## Recommendation architecture

### Content-based retrieval

Each movie has separate, reusable features: genre sets; TF-IDF text (genres, MovieLens tags, and any supplied overview, keywords, cast, and director); exact keyword and community-tag sets; cast sets; and director identity. Text uses scikit-learn's English-stopword TF-IDF vectorizer with unigrams, bigrams, and sublinear term frequency. Similarity is calculated only for a requested seed, avoiding a large all-pairs dense matrix.

The content score combines available signals: 34% genre Jaccard, 46% text cosine similarity, 15% cast Jaccard, and 5% director match. Weights are renormalized when a seed lacks a field. A reason is shown only if the two records actually share the corresponding genre, tag, keyword, cast member, director, or overview-text signal.

### Popularity and quality

The Bayesian quality estimate is:

```text
weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
```

`R` is a movie's mean MovieLens rating, `v` is its number of observed ratings, `C` is the catalog-wide mean rating, and `m` is the 60th-percentile vote-count prior (with a floor of 5). The popularity score is `0.75 * (weighted_rating / 5) + 0.25 * normalized(log1p(v))`. A low-vote title is therefore pulled toward the dataset mean, while rating volume also contributes.

### Hybrid ranking

For similar-title recommendations, the default final score is `0.55 * content + 0.30 * popularity + 0.15 * preference_match`. Absent signals are removed and remaining weights are renormalized. Taste-profile recommendations use explicit preferences and popularity; when a real MovieLens user profile is selected, a configurable collaborative signal is included as well.

### Collaborative filtering

The downloaded MovieLens release contains real, anonymized user IDs and explicit ratings. CineMatch builds a sparse movie-by-user matrix from ratings at or above 3 stars, then uses scikit-learn cosine nearest neighbors to find items related to titles rated 4 stars or higher by the selected profile. Results exclude movies that profile already rated. The feature is disabled when valid interaction data is absent. CineMatch does not invent interactions or claim to learn the current visitor's long-term behavior.

## Explainability

Explanations are assembled from observed overlaps and ranking evidence. Examples include shared genres, actual overlapping MovieLens tags, cast and director matches from optional metadata, TF-IDF overview similarity when both summaries exist, the specific movie a selected dataset profile rated highly, and the movie's mean plus rating count. Missing evidence is omitted. The interface never randomly generates a reason.

## Tech stack

- Python 3.11+
- Pandas and NumPy
- scikit-learn, SciPy sparse matrices
- Streamlit
- Plotly
- pytest

## Project architecture

```text
cinematch/
├── app.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                 # Downloaded data; excluded from Git
│   └── processed/           # Optional derived artifacts; excluded from Git
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── ranking.py
│   ├── recommender.py
│   └── search.py
└── tests/
    ├── conftest.py
    ├── test_preprocessing.py
    ├── test_ranking.py
    ├── test_recommender.py
    └── test_search.py
```

The raw data is downloaded by a script rather than committed. Unit-test fixtures are explicitly synthetic and are never used as the app's catalog.

## Dataset and attribution

The default catalog is [MovieLens latest-small from GroupLens](https://grouplens.org/datasets/movielens/), downloaded directly from [GroupLens' dataset host](https://files.grouplens.org/datasets/movielens/). The latest-small release includes movie titles, genres, user-applied tags, links to TMDB and IMDb IDs, and explicit user ratings. GroupLens describes it as approximately 100,000 ratings across about 9,000 movies from 600 users. Counts in the app are read from the downloaded files because releases can change.

MovieLens rating averages and vote counts shown in the app are calculated from its actual `ratings.csv`; they are not TMDB ratings. User IDs in the public sample are dataset identifiers, not CineMatch accounts. Review the [MovieLens dataset README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) and cite GroupLens when reusing the data. CineMatch downloads the dataset for local use and does not redistribute it.

Optional enrichment accepts the two CSVs from the [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata): `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`. The metadata is joined using the MovieLens-to-TMDB ID mapping from `links.csv`, with duplicate IDs removed and a left join that preserves the MovieLens catalog. The app labels the new coverage as optional enrichment. For any use of TMDB data, follow [TMDB's attribution and usage guidance](https://developer.themoviedb.org/docs/faq); its free API/data use is for non-commercial purposes and requires attribution. TMDB does not endorse CineMatch.

## Installation

From the project directory, Python 3.11 or newer is recommended.

### Windows PowerShell

```powershell
cd cinematch
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd cinematch
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the application

```bash
python download_dataset.py
streamlit run app.py
```

Streamlit prints the local URL, typically `http://localhost:8501`.

The data root defaults to the project's `data/` folder. Set `CINEMATCH_DATA_DIR` to use another root; the downloader writes under its `raw/` subfolder by default. For example, in PowerShell:

```powershell
$env:CINEMATCH_DATA_DIR = "D:\CineMatchData"
python download_dataset.py
streamlit run app.py
```

Hybrid weights are configurable before startup with `CINEMATCH_CONTENT_WEIGHT`, `CINEMATCH_POPULARITY_WEIGHT`, `CINEMATCH_PREFERENCE_WEIGHT`, and `CINEMATCH_COLLABORATIVE_WEIGHT`. Defaults for the similar-title score are `0.55`, `0.30`, and `0.15` for content, popularity, and preferences. Weights must be finite and non-negative; available signals are renormalized at scoring time.

To use TMDB enrichment, download the two TMDB 5000 CSVs from Kaggle and place them in `data/` (alongside the `raw/` directory):

```text
data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
```

The catalog and feature matrices are cached for the Streamlit session. Restart the app or clear Streamlit's cache after replacing the source files.

## Example usage

1. Search for a title such as “Matrix” and choose one result.
2. Open “Show similar movies” to see content and quality/popularity ranked candidates.
3. Expand “Why you'll like this” to inspect the matching evidence.
4. Use Discover for dynamic catalog filters or Taste Profile for explicit preferences.
5. Optionally select an anonymized MovieLens profile to include real interaction-based item neighbors.
6. Open Model Insights to review data coverage and the scoring formula.

## Tests

```bash
python -m pytest -q
```

Tests cover malformed and missing records, duplicate IDs, metadata joins, feature-based recommendations, no self-recommendations, collaborative profile filtering, Bayesian ranking, preference filters, and search suggestions.

## Limitations

- MovieLens is a historical research sample, not a current streaming catalog; its collection ended in 2018.
- The default MovieLens files do not contain plot summaries, cast, director, runtime, or language. Those signals appear only after optional TMDB enrichment and only where an ID matches.
- The tags are community annotations and can be sparse, inconsistent, or user-specific.
- Collaborative recommendations reflect a small public dataset and can be weak for movies with little rating overlap. A chosen profile is a public anonymized dataset profile, not the visitor's taste history.
- The explicit taste form is session-only. There are no accounts, watch history, or persistence across visits.
- Popularity is a rating-quality and vote-volume proxy, not streaming popularity or current demand.
- New or unrated movies have a cold-start problem and rely on the available content features and prior.

## Future improvements

- Optional account-backed user ratings and watch history with consent.
- Offline ranking evaluation against held-out interactions (Recall@K, NDCG@K, and catalog coverage).
- Incremental metadata refresh from a licensed API and reproducible artifact versioning.
- Calibrated diversity and novelty controls with recommendation audits.
- Add larger interaction datasets or validated embedding retrieval after measuring the baseline.

## License and data use

The CineMatch source code is provided under the MIT License. MovieLens and optional TMDB data retain their own source terms and attribution requirements; they are not bundled in the repository. See `LICENSE` and the dataset links above.
