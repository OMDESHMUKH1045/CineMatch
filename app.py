"""CineMatch Streamlit discovery app."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_DIR
from src.data_loader import CatalogData, load_catalog
from src.explainability import explain_recommendation
from src.ranking import filter_by_preferences
from src.recommender import HybridRecommender
from src.search import search_movies, title_suggestions

st.set_page_config(page_title="CineMatch · Movie Discovery", layout="wide", initial_sidebar_state="expanded")

APP_CSS = (
    "<style>\n"
    + (Path(__file__).parent / "assets" / "cinematch.css").read_text(encoding="utf-8")
    + "\n</style>"
)

FILTER_SURFACE_CSS = """
<style>
/* Hallmark · v2.0.0 · redesign: full · scope: app surfaces · genre: atmospheric · theme: Midnight (dealt 3/3) · preserve CineMatch's palette, type, and shell */
/* Hallmark · pre-emit critique: P4 H5 E4 S5 R5 V3 · weakest V, kept the shell; clarified the profile and catalog filter surfaces without changing their wiring */
div[class*="st-key-discover_advanced_filters"],
div[class*="st-key-taste_profile_controls"] {
  background:var(--cm-surface);
  border:1px solid var(--cm-border);
  border-radius:var(--radius-card);
  padding:18px 20px;
}
div[class*="st-key-discover_advanced_filters"] [data-testid="stWidgetLabel"] p,
div[class*="st-key-taste_profile_controls"] [data-testid="stWidgetLabel"] p {
  color:var(--cm-ink);
  font-size:13px;
  font-weight:600;
}
div[class*="st-key-discover_advanced_filters"] [data-testid="stCaptionContainer"] p,
div[class*="st-key-taste_profile_controls"] [data-testid="stCaptionContainer"] p {
  color:var(--cm-muted);
}
div[class*="st-key-discover_advanced_filters"] :is(button, input, [role="combobox"], [role="slider"]):focus-visible,
div[class*="st-key-taste_profile_controls"] :is(button, input, [role="combobox"], [role="slider"]):focus-visible {
  outline:2px solid var(--cm-focus);
  outline-offset:2px;
}
</style>
"""

@st.cache_resource(show_spinner="Loading the catalog and preparing recommendation features…")
def get_catalog_and_engine(data_dir: str) -> tuple[CatalogData, HybridRecommender]:
    catalog = load_catalog(Path(data_dir))
    return catalog, HybridRecommender(catalog.movies, catalog.ratings)


def display_title(row: pd.Series | dict[str, Any]) -> str:
    return str(row.get("display_title") or row.get("title") or "Untitled")


def format_title(movie_id: int, movies: pd.DataFrame) -> str:
    match = movies[movies["movieId"].eq(int(movie_id))]
    if match.empty:
        return str(movie_id)
    row = match.iloc[0]
    year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
    return f"{display_title(row)}{year}"


def render_movie_cards(frame: pd.DataFrame, columns: int = 3, show_reasons: bool = False) -> None:
    if frame.empty:
        st.info("No movies matched these selections. Try broadening the filters.")
        return
    for start in range(0, len(frame), columns):
        grid = st.columns(columns, gap="medium")
        for col, (_, movie) in zip(grid, frame.iloc[start : start + columns].iterrows()):
            title = html.escape(display_title(movie))
            year = str(int(movie["year"])) if pd.notna(movie.get("year")) else "Year unavailable"
            genres = ", ".join(movie.get("genres", [])[:3]) if isinstance(movie.get("genres"), list) else ""
            rating = movie.get("rating_average")
            rating_text = f"★ {float(rating):.1f}" if pd.notna(rating) else "No rating"
            votes = int(movie.get("vote_count") or 0)
            overview = html.escape(str(movie.get("overview") or ""))
            desc = f"<div class='movie-overview'>{overview}</div>" if overview else ""
            safe_genres = html.escape(genres or "Genre unavailable")
            with col:
                st.markdown(
                    f"<div class='movie-card'><div class='movie-card-top'><span>{year}</span><span>{rating_text} · {votes:,} ratings</span></div><h3>{title}</h3><div class='movie-meta'>{safe_genres}</div>{desc}</div>",
                    unsafe_allow_html=True,
                )
                if show_reasons and isinstance(movie.get("explanations"), list):
                    with st.expander("Why you'll like this"):
                        for reason in movie["explanations"]:
                            st.markdown(f"- {reason}")


def render_similar_movie_results(seed: pd.Series, frame: pd.DataFrame) -> None:
    seed_title = html.escape(display_title(seed))
    seed_facts: list[str] = []
    seed_year = seed.get("year")
    seed_facts.append(f"Release year: {int(seed_year)}" if pd.notna(seed_year) else "Year unavailable")
    seed_genres = seed.get("genres")
    seed_facts.append(", ".join(seed_genres) if isinstance(seed_genres, list) and seed_genres else "Genre unavailable")
    seed_rating = seed.get("rating_average")
    if pd.notna(seed_rating):
        seed_facts.append(f"MovieLens average {float(seed_rating):.1f} / 5")
    seed_vote_count = pd.to_numeric(pd.Series([seed.get("vote_count")]), errors="coerce").iloc[0]
    if pd.notna(seed_vote_count):
        seed_facts.append(f"{int(seed_vote_count):,} MovieLens ratings")

    cards: list[str] = []
    for rank, (_, movie) in enumerate(frame.iterrows(), start=1):
        title = html.escape(display_title(movie))
        year = movie.get("year")
        year_text = str(int(year)) if pd.notna(year) else "Year unavailable"
        rating = movie.get("rating_average")
        rating_text = f"MovieLens average {float(rating):.1f} / 5" if pd.notna(rating) else "No rating"
        vote_count = pd.to_numeric(pd.Series([movie.get("vote_count")]), errors="coerce").iloc[0]
        votes_text = f"{int(vote_count):,} MovieLens ratings" if pd.notna(vote_count) else "Rating count unavailable"
        genres = movie.get("genres")
        genres_text = ", ".join(genres[:3]) if isinstance(genres, list) and genres else "Genre unavailable"
        overview_value = movie.get("overview")
        overview = ""
        if pd.notna(overview_value) and str(overview_value).strip():
            overview = f"<p class='similarity-overview'>{html.escape(str(overview_value))}</p>"

        explanations = movie.get("explanations")
        reasons = [html.escape(str(reason)) for reason in explanations if str(reason).strip()] if isinstance(explanations, list) else []
        explanation = ""
        if reasons:
            extra_reasons = ""
            if len(reasons) > 1:
                extra_items = "".join(f"<li>{reason}</li>" for reason in reasons[1:])
                extra_reasons = (
                    "<details class='similarity-more-evidence'>"
                    "<summary>More evidence</summary>"
                    f"<ul class='similarity-extra-reasons'>{extra_items}</ul>"
                    "</details>"
                )
            explanation = (
                "<div class='similarity-explanation'>"
                "<p class='similarity-explanation-label'>Why you'll like this</p>"
                f"<p class='similarity-primary-reason'>{reasons[0]}</p>"
                f"{extra_reasons}"
                "</div>"
            )

        cards.append(
            "<article class='similarity-card'>"
            f"<span class='similarity-order'>Recommendation {rank}</span>"
            f"<h3>{title}</h3>"
            f"<div class='similarity-facts'><span>{year_text}</span><span>{rating_text}</span><span>{votes_text}</span></div>"
            f"<p class='similarity-genres'>{html.escape(genres_text)}</p>"
            f"{overview}{explanation}"
            "</article>"
        )

    body = "".join(cards) if cards else "<p class='similarity-empty'>No similar titles were found for this selection.</p>"
    grid = f"<div class='similarity-grid'>{body}</div>" if cards else body
    seed_fact_markup = "".join(f"<span>{html.escape(fact)}</span>" for fact in seed_facts)
    st.markdown(
        "<section class='similarity-results' aria-label='Similar movies'>"
        "<h2 class='similarity-section-title'>Similar movies</h2>"
        "<div class='similarity-seed'>"
        "<p class='similarity-label'>Selected movie</p>"
        f"<h3>{seed_title}</h3>"
        f"<div class='similarity-seed-meta'>{seed_fact_markup}</div>"
        "</div>"
        f"{grid}"
        "</section>",
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='hero'><div class='hero-title'>{html.escape(title)}</div><p class='hero-copy'>{html.escape(subtitle)}</p></div>", unsafe_allow_html=True)


def render_home_movie_rows(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No movie titles are available for this selection.")
        return

    rows: list[str] = []
    for _, movie in frame.iterrows():
        title = html.escape(display_title(movie))
        year_value = movie.get("year")
        year = str(int(year_value)) if pd.notna(year_value) else "Year unavailable"
        genre_values = movie.get("genres")
        genres = ", ".join(genre_values[:3]) if isinstance(genre_values, list) and genre_values else "Genre unavailable"
        rating = movie.get("rating_average")
        rating_text = f"MovieLens average {float(rating):.1f} / 5" if pd.notna(rating) else "No rating available"
        vote_count = pd.to_numeric(pd.Series([movie.get("vote_count")]), errors="coerce").iloc[0]
        votes_text = f"{int(vote_count):,} ratings" if pd.notna(vote_count) else "Rating count unavailable"
        overview_value = movie.get("overview")
        overview = ""
        if pd.notna(overview_value) and str(overview_value).strip():
            overview = f"<p class='home-film-overview'>{html.escape(str(overview_value))}</p>"

        rows.append(
            "<article class='home-film-row'>"
            "<div class='home-film-main'>"
            f"<h3 class='home-film-title'>{title}</h3>"
            f"<p class='home-film-meta'>{html.escape(year)} / {html.escape(genres)}</p>"
            f"{overview}"
            "</div>"
            f"<div class='home-film-rating'>{html.escape(rating_text)}<small>{html.escape(votes_text)}</small></div>"
            "</article>"
        )

    st.markdown(f"<div class='home-film-list'>{''.join(rows)}</div>", unsafe_allow_html=True)


def clear_home_recommendation() -> None:
    """Clear a selected title when the visitor changes the search query."""
    st.session_state.pop("similar_seed", None)
    st.session_state.pop("similar_seed_query", None)


def render_home_spotlight(movie: pd.Series) -> None:
    """Show a real, top-ranked MovieLens title as the Home screen's visual lead."""
    title = html.escape(display_title(movie))
    year_value = movie.get("year")
    year = str(int(year_value)) if pd.notna(year_value) else "Year unavailable"
    genre_values = movie.get("genres")
    genres = ", ".join(genre_values[:3]) if isinstance(genre_values, list) and genre_values else "Genre unavailable"
    rating = movie.get("rating_average")
    rating_text = f"MovieLens average {float(rating):.1f} / 5" if pd.notna(rating) else "No rating available"
    vote_count = pd.to_numeric(pd.Series([movie.get("vote_count")]), errors="coerce").iloc[0]
    votes_text = f"{int(vote_count):,} MovieLens ratings" if pd.notna(vote_count) else "Rating count unavailable"

    st.markdown(
        "<article class='home-spotlight'>"
        "<h2 class='home-spotlight-heading'>A popular place to begin</h2>"
        f"<h3 class='home-spotlight-title'>{title}</h3>"
        f"<p class='home-spotlight-meta'>{html.escape(year)} / {html.escape(genres)}</p>"
        f"<p class='home-spotlight-rating'>{html.escape(rating_text)}<small>{html.escape(votes_text)}</small></p>"
        "</article>",
        unsafe_allow_html=True,
    )


def render_home_search_results(
    frame: pd.DataFrame,
    query: str,
) -> None:
    """Render each title once, with its own direct similar-movie action."""
    for _, movie in frame.iterrows():
        movie_id = int(movie["movieId"])
        title_text = display_title(movie)
        title = html.escape(title_text)
        year_value = movie.get("year")
        year = str(int(year_value)) if pd.notna(year_value) else "Year unavailable"
        genre_values = movie.get("genres")
        genres = ", ".join(genre_values[:3]) if isinstance(genre_values, list) and genre_values else "Genre unavailable"
        rating = movie.get("rating_average")
        rating_text = f"MovieLens average {float(rating):.1f} / 5" if pd.notna(rating) else "No rating available"
        vote_count = pd.to_numeric(pd.Series([movie.get("vote_count")]), errors="coerce").iloc[0]
        votes_text = f"{int(vote_count):,} ratings" if pd.notna(vote_count) else "Rating count unavailable"

        details, action = st.columns([5, 1.4], gap="medium", vertical_alignment="center")
        with details:
            st.markdown(
                "<article class='home-search-result'>"
                f"<h3>{title}</h3>"
                f"<p>{html.escape(year)} / {html.escape(genres)}</p>"
                f"<p>{html.escape(rating_text)}; {html.escape(votes_text)}</p>"
                "</article>",
                unsafe_allow_html=True,
            )
        with action:
            if st.button(
                "Find similar",
                key=f"home-search-similar-{movie_id}",
                help=f"Find movies related to {title_text}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["similar_seed"] = movie_id
                st.session_state["similar_seed_query"] = query


def render_home_rating_list(frame: pd.DataFrame) -> None:
    """Show high-rated catalog entries as a compact, ranked list."""
    if frame.empty:
        st.info("No rated titles are available for this collection yet.")
        return

    rows: list[str] = []
    for _, movie in frame.iterrows():
        title = html.escape(display_title(movie))
        year_value = movie.get("year")
        year = str(int(year_value)) if pd.notna(year_value) else "Year unavailable"
        genre_values = movie.get("genres")
        genres = ", ".join(genre_values[:3]) if isinstance(genre_values, list) and genre_values else "Genre unavailable"
        rating = movie.get("rating_average")
        rating_text = f"{float(rating):.1f} / 5" if pd.notna(rating) else "No rating available"
        vote_count = pd.to_numeric(pd.Series([movie.get("vote_count")]), errors="coerce").iloc[0]
        votes_text = f"{int(vote_count):,} MovieLens ratings" if pd.notna(vote_count) else "Rating count unavailable"
        rows.append(
            "<li class='home-rating-row'>"
            f"<div><h3>{title}</h3><p>{html.escape(year)} / {html.escape(genres)}</p></div>"
            f"<div class='home-rating-value'><strong>{html.escape(rating_text)}</strong><small>{html.escape(votes_text)}</small></div>"
            "</li>"
        )
    st.markdown(f"<ol class='home-rating-list'>{''.join(rows)}</ol>", unsafe_allow_html=True)


def home_page(catalog: CatalogData, engine: HybridRecommender) -> None:
    popular = catalog.movies.copy()
    popular["_score"] = engine.popularity.to_numpy()
    popular = popular.sort_values("_score", ascending=False)
    featured = popular.iloc[0] if not popular.empty else None

    with st.container(key="home-surface"):
        with st.container(key="home-opening"):
            intro, spotlight = st.columns([1.12, 0.88], gap="large", vertical_alignment="center")
            with intro:
                st.markdown(
                    "<section class='home-intro'>"
                    "<h1 class='home-title'>Start with a film you love.</h1>"
                    "<p class='home-lede'>Pick a film you love and follow its trail: CineMatch connects movies through shared genres and community tags, ranks with real MovieLens ratings, and shows the clues behind every match.</p>"
                    "</section>",
                    unsafe_allow_html=True,
                )
                query = st.text_input(
                    "Search movie titles",
                    placeholder="Try The Matrix",
                    key="home_search",
                    on_change=clear_home_recommendation,
                )
            with spotlight:
                if featured is not None:
                    render_home_spotlight(featured)
                    featured_id = int(featured["movieId"])
                    if st.button(
                        "Find similar",
                        key="home-spotlight-action",
                        help=f"Find movies related to {display_title(featured)}",
                        type="primary",
                    ):
                        st.session_state["similar_seed"] = featured_id
                        st.session_state["similar_seed_query"] = query
                else:
                    st.info("There are no movie titles in the catalog yet.")

        if query:
            results = search_movies(query, catalog.movies, engine.popularity, limit=12)
            if results.empty:
                suggestions = title_suggestions(query, catalog.movies)
                suggestion_text = ", ".join(suggestions) if suggestions else "a shorter title"
                st.info(f"No close title matches yet. Try {suggestion_text}.")
            else:
                result_word = "title match" if len(results) == 1 else "title matches"
                st.markdown(
                    f"<h2 class='home-section-heading'>Choose a title to follow</h2><p class='home-section-note'>{len(results)} {result_word} from MovieLens.</p>",
                    unsafe_allow_html=True,
                )
                render_home_search_results(results, query)

        if (
            st.session_state.get("similar_seed") is not None
            and st.session_state.get("similar_seed_query") == query
        ):
            similar_seed = int(st.session_state["similar_seed"])
            recs = engine.recommend_similar(similar_seed, limit=6)
            seed_rows = catalog.movies[catalog.movies["movieId"].eq(similar_seed)]
            if not seed_rows.empty:
                render_similar_movie_results(seed_rows.iloc[0], recs)

        st.markdown(
            "<h2 class='home-section-heading'>More from MovieLens</h2>"
            "<p class='home-section-note'>Popular titles ranked from real rating quality and vote volume.</p>",
            unsafe_allow_html=True,
        )
        render_home_movie_rows(popular.iloc[1:6])

        positive_votes = pd.to_numeric(catalog.movies["vote_count"], errors="coerce")
        positive_votes = positive_votes[positive_votes.gt(0)]
        vote_floor = max(20, int(positive_votes.quantile(0.60))) if not positive_votes.empty else 20
        top_rated = catalog.movies[catalog.movies["vote_count"].ge(vote_floor)].sort_values("rating_average", ascending=False).head(4)
        st.markdown(
            "<h2 class='home-section-heading'>High marks, with a crowd behind them</h2>"
            f"<p class='home-section-note'>Each title has at least {vote_floor:,} MovieLens ratings.</p>",
            unsafe_allow_html=True,
        )
        render_home_rating_list(top_rated)

        available = sorted({genre for genres in catalog.movies["genres"] for genre in genres})
        st.markdown(
            "<h2 class='home-section-heading'>Wander by genre</h2>"
            "<p class='home-section-note'>Choose a MovieLens genre to see titles from that shelf.</p>",
            unsafe_allow_html=True,
        )
        if available:
            chosen_genre = st.selectbox("Choose a genre", available, key="home_genre")
            matching = catalog.movies[catalog.movies["genres"].map(lambda gs: chosen_genre in gs)].copy()
            matching["_score"] = engine.popularity.loc[matching.index].to_numpy()
            render_home_movie_rows(matching.sort_values("_score", ascending=False).head(6))
        else:
            st.info("Genre browsing is unavailable because this catalog has no genre values.")


def discover_page(catalog: CatalogData, engine: HybridRecommender) -> None:
    render_header("Explore the catalog.", "Fine-tune the catalog with real metadata and a transparent quality ranking.")
    movies = catalog.movies
    all_genres = sorted({genre for genres in movies["genres"] for genre in genres})
    year_values = pd.to_numeric(movies["year"], errors="coerce").dropna()
    min_year, max_year = (int(year_values.min()), int(year_values.max())) if not year_values.empty else (0, 0)
    decade_values = sorted({(int(year) // 10) * 10 for year in year_values})
    rating_available = pd.to_numeric(movies["rating_average"], errors="coerce").notna().any()
    runtime_available = pd.to_numeric(movies["runtime"], errors="coerce").notna().any()
    languages = sorted(catalog.movies.loc[catalog.movies["language"].fillna("").ne(""), "language"].unique().tolist())
    with st.container(key="discover_advanced_filters", border=True):
        st.markdown("#### Advanced filters")
        st.caption("Leave a field at its default to include every value. Multiple genres and decades match any selection.")
        a, b, c = st.columns(3, gap="large")
        with a:
            genres = st.multiselect(
                "Genre",
                all_genres,
                placeholder="Any genre" if all_genres else "Unavailable",
                disabled=not all_genres,
            )
        with b:
            selected_decades = st.multiselect(
                "Release decade",
                decade_values,
                format_func=lambda decade: f"{decade}s",
                placeholder="Any decade" if decade_values else "Unavailable",
                disabled=not decade_values,
            )
        with c:
            if min_year < max_year:
                year_range = st.slider("Release year range", min_year, max_year, (min_year, max_year))
            else:
                year_range = (min_year, max_year)
                if year_values.empty:
                    st.selectbox("Release year range", ["Unavailable"], disabled=True)
                else:
                    st.selectbox("Release year range", [str(min_year)], disabled=True)

        d, e, f = st.columns(3, gap="large")
        with d:
            minimum_rating = st.slider(
                "Minimum MovieLens rating",
                0.0,
                5.0,
                0.0,
                0.5,
                disabled=not rating_available,
            )
            if not rating_available:
                st.caption("Rating values are not available in this catalog.")
        with e:
            runtime_pref = st.selectbox(
                "Runtime",
                ["Any", "Under 90 min", "90–120 min", "Over 120 min"],
                disabled=not runtime_available,
            )
            if not runtime_available:
                st.caption("Runtime data is available after optional TMDB enrichment.")
        with f:
            if languages:
                language = st.selectbox("Original language", ["Any"] + languages)
            else:
                st.selectbox("Original language", ["Unavailable"], disabled=True)
                st.caption("No original language values are available in this catalog.")
                language = "Any"

        st.markdown("#### Ranking threshold")
        min_popularity = st.slider("Minimum popularity score", 0.0, 1.0, 0.0, 0.05)

    prefs = {
        "genres": genres,
        "decades": selected_decades,
        "min_rating": minimum_rating,
        "runtime": runtime_pref,
        "languages": [] if language == "Any" else [language],
    }
    filtered = filter_by_preferences(movies, prefs)
    if year_range != (min_year, max_year):
        years = pd.to_numeric(filtered["year"], errors="coerce")
        filtered = filtered[years.between(year_range[0], year_range[1])]
    filtered = filtered[engine.popularity.loc[filtered.index].ge(min_popularity)]
    filtered = filtered.copy()
    filtered["hybrid_score"] = engine.popularity.loc[filtered.index].to_numpy()
    visible = filtered.sort_values("hybrid_score", ascending=False).head(36).copy()
    visible["explanations"] = [
        explain_recommendation(movies, engine.features, int(index), preferences=prefs)
        for index in visible.index
    ]
    st.caption(f"Showing {len(filtered):,} movies, sorted by the hybrid quality/popularity score.")
    render_movie_cards(visible, show_reasons=True)


def taste_page(catalog: CatalogData, engine: HybridRecommender) -> None:
    render_header("Build your taste profile.", "Tell CineMatch what you want to watch. This profile is session-only and does not claim to learn long-term behavior.")
    genres = sorted({genre for values in catalog.movies["genres"] for genre in values})
    year_values = pd.to_numeric(catalog.movies["year"], errors="coerce").dropna().astype(int)
    decades = sorted({(year // 10) * 10 for year in year_values})
    languages = sorted(catalog.movies.loc[catalog.movies["language"].fillna("").ne(""), "language"].unique().tolist())
    rating_available = pd.to_numeric(catalog.movies["rating_average"], errors="coerce").notna().any()
    runtime_available = pd.to_numeric(catalog.movies["runtime"], errors="coerce").notna().any()
    with st.container(key="taste_profile_controls", border=True):
        st.markdown("#### Your preferences")
        st.caption("Choose genres, decades, and languages for this session. Multiple selections match any chosen value.")
        left, right = st.columns(2, gap="large")
        with left:
            chosen_genres = st.multiselect(
                "Favorite genres",
                genres,
                placeholder="Any genre" if genres else "Unavailable",
                disabled=not genres,
            )
            chosen_decades = st.multiselect(
                "Preferred decades",
                decades,
                format_func=lambda decade: f"{decade}s",
                placeholder="Any decade" if decades else "Unavailable",
                disabled=not decades,
            )
            if languages:
                chosen_languages = st.multiselect(
                    "Preferred original languages",
                    languages,
                    placeholder="Any language",
                )
            else:
                st.multiselect("Preferred original languages", [], placeholder="Unavailable", disabled=True)
                st.caption("No original language values are available in this catalog.")
                chosen_languages = []
        with right:
            min_rating = st.slider(
                "Minimum MovieLens rating",
                0.0,
                5.0,
                3.5 if rating_available else 0.0,
                0.5,
                disabled=not rating_available,
            )
            if not rating_available:
                st.caption("Rating values are not available in this catalog.")
            runtime = st.selectbox(
                "Runtime preference",
                ["Any", "Under 90 min", "90–120 min", "Over 120 min"],
                disabled=not runtime_available,
            )
            if not runtime_available:
                st.caption("Runtime data is available after optional TMDB enrichment.")
            st.markdown("##### Optional rating signal")
            use_cf = st.checkbox(
                "Use a real MovieLens rating profile",
                value=False,
                disabled=not engine.collaborative_available,
            )
            user_id = None
            if use_cf:
                user_id = st.selectbox(
                    "Anonymized MovieLens user profile",
                    engine.user_ids,
                    format_func=lambda uid: f"Profile #{uid} · {engine.user_summary(uid)['rating_count']} real ratings",
                )
                summary = engine.user_summary(int(user_id))
                if summary["liked_titles"]:
                    st.caption("High ratings in this profile: " + " · ".join(summary["liked_titles"][:4]))
    prefs = {"genres": chosen_genres, "decades": chosen_decades, "languages": chosen_languages, "min_rating": min_rating, "runtime": runtime, "user_id": user_id}
    st.caption("Genre, decade, language, minimum rating, and runtime choices shape recommendations for this session only. Nothing in this taste profile is saved. A MovieLens profile uses its real ratings and excludes titles it has already rated.")
    if use_cf and user_id is not None:
        summary = engine.user_summary(int(user_id))
        st.caption(f"Using {summary['rating_count']} real ratings from anonymized MovieLens profile #{user_id}; no account or watch history is created.")
    recommendations = engine.recommend_personalized(prefs, limit=18, user_id=int(user_id) if user_id is not None else None)
    st.markdown("### Recommendations for this session")
    st.caption(f"{len(recommendations)} shown")
    render_movie_cards(recommendations, show_reasons=True)


def insights_page(catalog: CatalogData, engine: HybridRecommender) -> None:
    render_header("Inside the recommendation engine.", "A clear view of the data, features, ranking, and limits behind each suggestion.")
    movies = catalog.movies
    user_count = catalog.ratings["userId"].nunique() if "userId" in catalog.ratings else 0
    unique_genres = sorted({genre for genres in movies["genres"] for genre in genres})
    metrics = st.columns(4)
    metrics[0].metric("Movies", f"{len(movies):,}")
    metrics[1].metric("Genres", f"{len(unique_genres)}")
    metrics[2].metric("Rating events", f"{len(catalog.ratings):,}")
    metrics[3].metric("Anonymized raters", f"{user_count:,}")
    st.markdown("### Feature engineering")
    st.markdown("- **Genres:** exact set overlap (Jaccard similarity).\n- **Text:** TF-IDF with unigrams and bigrams across genres, tags, keywords, cast, director, and any supplied overview.\n- **People:** exact cast overlap and director match.\n- **Rating interactions:** optional item-based nearest-neighbor collaborative filtering using MovieLens ratings of 4 stars or higher.")
    st.markdown("### Popularity and hybrid score")
    st.latex(r"\text{Bayesian quality} = \frac{v}{v+m}R + \frac{m}{v+m}C")
    st.write("Here, R is a movie's average rating, v is its observed rating count, C is the catalog-wide mean, and m is the 60th-percentile vote-count prior (at least 5). The popularity signal combines 75% normalized Bayesian quality with 25% normalized log rating volume. Similar-title results use configurable content, popularity, and optional preference weights; profile results can also include collaborative evidence. Available weights are renormalized when a signal is absent.")
    st.markdown("### Data coverage")
    overview_count = int(movies["overview"].fillna("").astype(str).str.strip().ne("").sum())
    cast_count = int(movies["cast"].map(bool).sum())
    language_count = int(movies["language"].fillna("").astype(str).str.strip().ne("").sum())
    st.write(f"MovieLens is the default catalog and provides titles, genres, ratings, and community tags. Optional TMDB CSV enrichment is joined by MovieLens `tmdbId` where a match exists. Current enriched coverage: {overview_count:,} plot overviews, {cast_count:,} cast lists, and {language_count:,} language values.")
    if overview_count == 0:
        st.info("This MovieLens release has no plot summaries or cast/crew. Add the optional TMDB 5000 files described in the README to enable those content signals.")
    st.markdown("### Catalog shape")
    chart_left, chart_right = st.columns(2)
    genre_rows = pd.DataFrame([(genre, len(movies[movies["genres"].map(lambda values: genre in values)])) for genre in unique_genres], columns=["Genre", "Movies"]).sort_values("Movies", ascending=False)
    with chart_left:
        st.plotly_chart(px.bar(genre_rows.head(15), x="Movies", y="Genre", orientation="h", color="Movies", color_continuous_scale="Purples", template="plotly_dark", title="Catalog coverage by genre"), use_container_width=True)
    with chart_right:
        plot_data = movies[movies["rating_average"].notna() & movies["vote_count"].gt(0)].copy()
        st.plotly_chart(px.scatter(plot_data, x="vote_count", y="rating_average", hover_name="display_title", log_x=True, opacity=.45, template="plotly_dark", title="Ratings and vote volume"), use_container_width=True)
    st.markdown("### Limitations")
    st.write("MovieLens is a research dataset and does not represent a streaming service's current catalog or audience. Its tags and ratings are community activity, not population-wide preference. No collaborative recommendation is claimed when rating records are missing. Taste-profile controls remain session-only, the public dataset's user IDs are anonymized dataset identifiers, and cold-start titles rely on available catalog metadata and the popularity prior.")


def main() -> None:
    st.markdown(APP_CSS + FILTER_SURFACE_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## CineMatch")
        st.caption("Intelligent movie discovery")
        page = st.radio("Navigate", ["Home", "Discover", "Taste Profile", "Model Insights"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Data: GroupLens MovieLens latest-small")
        st.caption("No login, tracking, or invented ratings")
    try:
        catalog, engine = get_catalog_and_engine(str(DATA_DIR))
    except FileNotFoundError as exc:
        st.title("CineMatch")
        st.markdown("Discover movies through data-driven recommendations.")
        st.warning(str(exc))
        st.code("python download_dataset.py\nstreamlit run app.py", language="powershell")
        st.stop()
    if page == "Home":
        home_page(catalog, engine)
    elif page == "Discover":
        discover_page(catalog, engine)
    elif page == "Taste Profile":
        taste_page(catalog, engine)
    else:
        insights_page(catalog, engine)


if __name__ == "__main__":
    main()
