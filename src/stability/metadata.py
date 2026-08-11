"""TMDb metadata fetching, matching, and enrichment for movie titles."""

import difflib
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from stability.utils import canonicalize

logger = logging.getLogger(__name__)


class _HttpConfig:
    """Module-level HTTP client state with proper typing."""

    client: httpx.Client | None = None
    verify: bool = os.environ.get("TMDB_DISABLE_SSL_VERIFY") == "1"


_http = _HttpConfig()


def _get_http_client() -> httpx.Client:
    """Return a module-level HTTP client, creating one if needed."""
    client = _http.client
    if client is None:
        client = httpx.Client(verify=_http.verify, timeout=10.0)
        _http.client = client
    return client


def _request_get(
    url: str, *, params: dict[str, str | int] | None = None
) -> httpx.Response:
    """GET request using the configured module-level HTTP client."""
    client = _get_http_client()
    return client.get(url, params=params)


def parse_title_year(title_norm: str) -> tuple[str, int | None]:
    """Extract title and year from a normalized movie title string.

    Args:
        title_norm: Normalized movie title, possibly with year in parentheses.

    Returns:
        Tuple of (title, year) where year is None if not found.

    Examples:
        >>> parse_title_year("The Matrix (1999)")
        ('The Matrix', 1999)
        >>> parse_title_year("Okja")
        ('Okja', None)

    """
    match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title_norm.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title_norm.strip(), None


def build_enriched_text(
    title: str,
    genres: list[str] | None = None,
    director: str | None = None,
    cast: list[str] | None = None,
) -> str:
    """Construct metadata-enriched text for embedding.

    Builds format: "Title | Genres: G1, G2 | Director: D | Cast: A1, A2"
    Sections with None or empty values are omitted.

    Args:
        title: Movie title (should include year, e.g. "The Matrix (1999)").
        genres: List of genre names.
        director: Director name.
        cast: List of cast member names.

    Returns:
        Enriched text string with metadata sections separated by " | ".

    """
    parts = [title]

    if genres:
        parts.append(f"Genres: {', '.join(genres)}")

    if director:
        parts.append(f"Director: {director}")

    if cast:
        parts.append(f"Cast: {', '.join(cast)}")

    return " | ".join(parts)


def search_tmdb_movie(
    title: str,
    year: int | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 0.3,
    max_results: int = 5,
) -> list[dict]:
    """Search TMDb for a movie by title and optional year.

    Args:
        title: Movie title to search for.
        year: Optional release year to narrow results.
        api_key: TMDb API key. Falls back to TMDB_API_KEY env var.
        max_retries: Maximum number of retries on HTTP 429.
        retry_delay: Base delay between retries (exponential backoff).
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts (possibly empty).

    """
    if api_key is None:
        api_key = os.environ.get("TMDB_API_KEY", "")

    params: dict[str, str | int] = {"api_key": api_key, "query": title}
    if year is not None:
        params["year"] = year

    for attempt in range(max_retries):
        try:
            response = _request_get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
            )
            if response.status_code == 429:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"TMDb rate limit hit, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

        except httpx.HTTPStatusError:
            logger.exception(f"TMDb search failed for '{title}'")
            return []
        except httpx.RequestError:
            logger.exception(f"TMDb request error for '{title}'")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))
                continue
            return []
        else:
            return results[:max_results]

    logger.warning(f"TMDb search exhausted retries for '{title}'")
    return []


def fetch_tmdb_details(
    tmdb_id: int,
    api_key: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 0.3,
) -> dict | None:
    """Fetch detailed movie information from TMDb including credits.

    Args:
        tmdb_id: TMDb movie ID.
        api_key: TMDb API key. Falls back to TMDB_API_KEY env var.
        max_retries: Maximum number of retries on HTTP 429.
        retry_delay: Base delay between retries (exponential backoff).

    Returns:
        Dict with keys: genres, director, top_cast, overview. None on failure.

    """
    if api_key is None:
        api_key = os.environ.get("TMDB_API_KEY", "")

    for attempt in range(max_retries):
        try:
            response = _request_get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={"api_key": api_key, "append_to_response": "credits"},
            )
            if response.status_code == 429:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"TMDb rate limit hit, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()

        except httpx.HTTPStatusError:
            logger.exception(f"TMDb details fetch failed for ID {tmdb_id}")
            return None
        except httpx.RequestError:
            logger.exception(f"TMDb request error for ID {tmdb_id}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))
                continue
            return None
        else:
            # Extract genres
            genres = [g["name"] for g in data.get("genres", [])]

            # Extract director from credits crew
            credits_data = data.get("credits", {})
            crew = credits_data.get("crew", [])
            director = None
            for member in crew:
                if member.get("job") == "Director":
                    director = member.get("name")
                    break

            # Extract top 3 cast members
            cast_list = credits_data.get("cast", [])
            top_cast = [m["name"] for m in cast_list[:3]]

            # Extract overview
            overview = data.get("overview", "")

            return {
                "genres": genres,
                "director": director,
                "top_cast": top_cast,
                "overview": overview if overview else None,
            }

    logger.warning(f"TMDb details fetch exhausted retries for ID {tmdb_id}")
    return None


def score_tmdb_match(
    query_title: str,
    query_year: int | None,
    candidate: dict,
) -> float:
    """Score how well a TMDb candidate matches the query title and year.

    Uses canonicalized string comparison and year penalty.

    Args:
        query_title: Original query title.
        query_year: Original query year (may be None).
        candidate: TMDb search result dict with 'title' and 'release_date' keys.

    Returns:
        Match score (higher is better). Threshold of 0.6 recommended.

    """
    query_canon = canonicalize(query_title)
    candidate_title = candidate.get("title", "")
    candidate_canon = canonicalize(candidate_title)

    # Base score from fuzzy title match
    ratio = difflib.SequenceMatcher(None, query_canon, candidate_canon).ratio()
    score = ratio if ratio >= 0.85 else 0.0

    # Year penalty
    if query_year is not None:
        release_date = candidate.get("release_date", "")
        if release_date and len(release_date) >= 4:
            try:
                candidate_year = int(release_date[:4])
                year_diff = abs(query_year - candidate_year)
                penalty = min(0.1 * year_diff, 0.5)
                score -= penalty
            except ValueError:
                pass

    return score


def match_movie_to_tmdb(
    title_norm: str,
    api_key: str | None = None,
) -> dict | None:
    """Match a normalized movie title to TMDb using two-pass search.

    First pass: search with year (if available).
    Second pass: search without year (fallback).

    Args:
        title_norm: Normalized movie title (e.g. "The Matrix (1999)").
        api_key: TMDb API key. Falls back to TMDB_API_KEY env var.

    Returns:
        Dict with tmdb_id, title, year, genres, director, top_cast,
        overview, match_score. None if no match found.

    """
    title, year = parse_title_year(title_norm)

    # First pass: search with year, evaluate multiple candidates
    results = search_tmdb_movie(title, year=year, api_key=api_key)
    best_match = None
    best_score = -1.0

    for candidate in results:
        score = score_tmdb_match(title, year, candidate)
        if score >= 0.6 and score > best_score:
            best_match = candidate
            best_score = score

    # Second pass: search without year (fallback)
    if best_match is None and year is not None:
        results = search_tmdb_movie(title, year=None, api_key=api_key)
        for candidate in results:
            score = score_tmdb_match(title, year, candidate)
            if score >= 0.6 and score > best_score:
                best_match = candidate
                best_score = score

    if best_match is None:
        return None

    # Fetch full details
    tmdb_id = best_match["id"]
    details = fetch_tmdb_details(tmdb_id, api_key=api_key)

    if details is None:
        return None

    return {
        "tmdb_id": tmdb_id,
        "title": best_match.get("title", title),
        "year": year,
        "genres": details["genres"],
        "director": details["director"],
        "top_cast": details["top_cast"],
        "overview": details["overview"],
        "match_score": best_score,
    }


def enrich_movie_metadata(
    movies_df: pd.DataFrame,
    api_key: str | None = None,
    cache_path: Path | None = None,
    rate_limit_delay: float = 0.05,
    title_col: str = "title_norm",
    on_progress: Callable[[int, int, bool], None] | None = None,
) -> pd.DataFrame:
    """Enrich a DataFrame of movies with TMDb metadata and enriched text.

    Supports incremental caching: if cache_path exists, already-enriched
    titles are skipped. Checkpoints are saved every 100 movies.

    Args:
        movies_df: DataFrame with a column specified by title_col.
        api_key: TMDb API key. Falls back to TMDB_API_KEY env var.
        cache_path: Path to cache CSV for incremental runs.
        rate_limit_delay: Delay between API calls in seconds.
        title_col: Name of the column containing movie titles.
        on_progress: Optional callback ``(current, total, matched)``
            invoked after each API fetch.

    Returns:
        DataFrame with added columns: tmdb_id, genres, director,
        top_cast, overview, match_score, enriched_text.

    """
    # Load existing cache if available
    cache_records: dict[str, dict] = {}
    if cache_path is not None and cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        for _, row in cached_df.iterrows():
            cache_records[row[title_col]] = row.to_dict()
        logger.info("Loaded %d cached records from %s", len(cache_records), cache_path)

    titles = movies_df[title_col].tolist()
    to_fetch = sum(1 for t in titles if t not in cache_records)
    logger.info("%d titles to fetch (%d cached)", to_fetch, len(titles) - to_fetch)

    results: list[dict] = []
    processed_count = 0

    for title in titles:
        # Use cached result if available
        if title in cache_records:
            results.append(cache_records[title])
            continue

        # Fetch from TMDb API
        match = match_movie_to_tmdb(title, api_key=api_key)
        processed_count += 1

        if on_progress is not None:
            on_progress(processed_count, to_fetch, match is not None)

        if match is not None:
            enriched_text = build_enriched_text(
                title=title,
                genres=match["genres"],
                director=match["director"],
                cast=match["top_cast"],
            )
            record = {
                title_col: title,
                "tmdb_id": match["tmdb_id"],
                "genres": str(match["genres"]) if match["genres"] else np.nan,
                "director": match["director"],
                "top_cast": str(match["top_cast"]) if match["top_cast"] else np.nan,
                "overview": match["overview"],
                "match_score": match["match_score"],
                "enriched_text": enriched_text,
            }
        else:
            record = {
                title_col: title,
                "tmdb_id": np.nan,
                "genres": np.nan,
                "director": np.nan,
                "top_cast": np.nan,
                "overview": np.nan,
                "match_score": np.nan,
                "enriched_text": title,
            }

        results.append(record)
        cache_records[title] = record

        # Checkpoint every 100 movies
        if cache_path is not None and processed_count % 100 == 0:
            _save_cache(cache_records, cache_path)

        time.sleep(rate_limit_delay)

    # Final cache save
    if cache_path is not None and processed_count > 0:
        _save_cache(cache_records, cache_path)
        logger.info(
            f"Final cache saved: {processed_count} new movies processed, "
            f"{len(cache_records)} total cached",
        )

    result_df = pd.DataFrame(results)

    # Merge back with original DataFrame
    enrichment_cols = [
        title_col,
        "tmdb_id",
        "genres",
        "director",
        "top_cast",
        "overview",
        "match_score",
        "enriched_text",
    ]
    return movies_df.merge(
        result_df[enrichment_cols],
        on=title_col,
        how="left",
    )


def _save_cache(
    cache_records: dict[str, dict],
    cache_path: Path,
) -> None:
    """Save cache records to CSV.

    Args:
        cache_records: Dict mapping titles to record dicts.
        cache_path: Path to write the cache CSV.

    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame(list(cache_records.values()))
    cache_df.to_csv(cache_path, index=False)
