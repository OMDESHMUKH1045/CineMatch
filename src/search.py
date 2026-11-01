"""Case-insensitive title search with useful typo suggestions."""

from __future__ import annotations

from difflib import get_close_matches
import re

import pandas as pd


def _aliases(title: str) -> list[str]:
    """Include MovieLens' trailing-article spelling as a familiar title form."""
    title = title.strip()
    values = [title]
    match = re.match(r"^(.+),\s+(The|A|An)$", title, flags=re.IGNORECASE)
    if match:
        values.append(f"{match.group(2)} {match.group(1)}")
    return [value.casefold() for value in values]


def search_movies(query: str, movies: pd.DataFrame, popularity: pd.Series | None = None, limit: int = 20) -> pd.DataFrame:
    """Find partial title matches, prioritizing exact and prefix matches."""
    query = (query or "").strip()
    if not query:
        return movies.iloc[[]].copy()
    titles = movies["display_title"].fillna(movies["title"]).astype(str)
    aliases = titles.map(_aliases)
    exact = aliases.map(lambda names: query.casefold() in names)
    prefix = aliases.map(lambda names: any(name.startswith(query.casefold()) for name in names))
    partial = aliases.map(lambda names: any(query.casefold() in name for name in names))
    result = movies[partial].copy()
    if result.empty:
        return result
    result["_exact"] = exact.loc[result.index].to_numpy()
    result["_prefix"] = prefix.loc[result.index].to_numpy()
    if popularity is not None:
        result["_popularity"] = popularity.loc[result.index].to_numpy()
    else:
        result["_popularity"] = 0.0
    result = result.sort_values(["_exact", "_prefix", "_popularity", "display_title"], ascending=[False, False, False, True])
    return result.drop(columns=["_exact", "_prefix", "_popularity"]).head(limit).reset_index(drop=True)


def title_suggestions(query: str, movies: pd.DataFrame, limit: int = 5) -> list[str]:
    """Return close title suggestions for a query that has no direct matches."""
    query = (query or "").strip()
    if not query:
        return []
    titles = movies["display_title"].fillna(movies["title"]).astype(str).drop_duplicates().tolist()
    alias_to_title = {alias: title for title in titles for alias in _aliases(title)}
    close = get_close_matches(query.casefold(), list(alias_to_title), n=limit, cutoff=0.38)
    return list(dict.fromkeys(alias_to_title[value] for value in close))[:limit]
