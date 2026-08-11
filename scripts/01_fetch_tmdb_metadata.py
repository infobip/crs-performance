"""Fetch TMDb metadata for all movies in the ReDial dataset.

Requires TMDB_API_KEY environment variable. Supports incremental runs via cache
file — if interrupted, re-running picks up from where it left off.

Usage:
    uv run python scripts/01_fetch_tmdb_metadata.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

import dotenv
import pandas as pd

from stability.metadata import enrich_movie_metadata

dotenv.load_dotenv()

# Global variables
DATA_PATH = Path("data")
BAR_FILL = "\u2588"
BAR_EMPTY = "\u2591"


def _progress_bar(current: int, total: int, matched: bool) -> None:
    """Overwrite a single terminal line with a progress bar."""
    _progress_bar.matched += int(matched)
    cols = shutil.get_terminal_size().columns
    pct = current / total if total else 1.0
    elapsed = time.perf_counter() - _progress_bar.t0
    eta = (elapsed / current) * (total - current) if current else 0.0

    stats = f" {current}/{total} ({pct:.0%})  {elapsed:.0f}s elapsed  ETA {eta:.0f}s"
    bar_width = max(cols - len(stats) - 2, 10)
    filled = int(bar_width * pct)
    bar = BAR_FILL * filled + BAR_EMPTY * (bar_width - filled)
    print(f"\r{bar}{stats}", end="", flush=True)


def main() -> None:
    """Fetch TMDb metadata for ReDial movies and save enriched CSV."""
    if not os.environ.get("TMDB_API_KEY"):
        print(
            "Error: TMDB_API_KEY is not set. "
            "Get a free API key at https://www.themoviedb.org/settings/api"
        )
        sys.exit(1)

    movies_path = DATA_PATH / "processed" / "movies_with_mentions_processed.csv"
    if not movies_path.exists():
        print(
            f"Error: {movies_path} not found. Run scripts/00_build_prompt_templates.py first."
        )
        sys.exit(1)

    movies_df = pd.read_csv(movies_path)
    total_movies = len(movies_df)
    print(f"Loaded {total_movies} movies from {movies_path}")

    cache_path = DATA_PATH / "processed" / "movies_metadata_tmdb.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        print(f"Resuming: {len(cached)} already cached")

    # Set up progress state and run enrichment
    _progress_bar.t0 = time.perf_counter()
    _progress_bar.matched = 0

    print("Fetching TMDb metadata...")
    enriched_df = enrich_movie_metadata(
        movies_df,
        cache_path=cache_path,
        on_progress=_progress_bar,
    )
    elapsed = time.perf_counter() - _progress_bar.t0
    print()

    # Summary
    matched = enriched_df["tmdb_id"].notna().sum()
    print(
        f"\nDone in {elapsed:.1f}s — "
        f"{matched}/{total_movies} matched ({matched / total_movies * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
