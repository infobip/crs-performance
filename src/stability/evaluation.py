"""Ranking artifact schema helpers and catalog loaders for recommendation evaluation.

This module provides functions for:
- Normalizing per-user ranked recommendation artifacts into a shared schema
- Converting ranking rows into pandas DataFrames with stable column ordering
- Loading shared movie catalog, popularity counts, tail-item sets, and genre mappings
"""

import ast
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stability.metrics import recommendation_metrics
from stability.preprocessing import normalize_movie_title

# ---------------------------------------------------------------------------
# LLM result normalization helpers
# ---------------------------------------------------------------------------


def build_title_to_item_id_mapping(
    item_path: str | Path,
) -> dict[str, int]:
    """Build a normalized title -> item_id mapping from a RecBole .item file.

    Titles are normalized via :func:`normalize_movie_title` so that LLM
    output titles can be matched consistently.

    Args:
        item_path: Path to the RecBole item file.

    Returns:
        Dictionary mapping normalized title strings to integer item IDs.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    id_to_title, _ = load_catalog(item_path)
    mapping: dict[str, int] = {}
    for item_id, title in id_to_title.items():
        norm_title = normalize_movie_title(title)
        if norm_title is not None:
            if norm_title in mapping:
                logger.warning(
                    "Duplicate normalized title '%s' for item IDs %d and %d; keeping %d",
                    norm_title,
                    mapping[norm_title],
                    item_id,
                    mapping[norm_title],
                )
                continue
            mapping[norm_title] = item_id
    return mapping


def parse_candidate_pool_label(
    n_candidates: str | int | None,
) -> tuple[str | None, str | None, str | int | None]:
    """Parse a candidate pool label into retriever metadata.

    Supports legacy count-only labels (e.g. ``c250``) and named pool
    labels (e.g. ``cf_EASE_c250`` or ``seq_SASRec_c250``).

    Args:
        n_candidates: Candidate pool label string, integer, or None.

    Returns:
        Tuple of ``(retriever_type, retriever_model, n_candidates)``.
        Legacy labels map to ``("cbf", "CBF", int)``. Named labels map
        to ``("cf", model_name, int)`` or ``("sequential", model_name, int)``.
        Unrecognised inputs return ``(None, None, original_value)``.

    Examples:
        >>> parse_candidate_pool_label("c250")
        ('cbf', 'CBF', 250)
        >>> parse_candidate_pool_label("cf_EASE_c250")
        ('cf', 'EASE', 250)
        >>> parse_candidate_pool_label("seq_SASRec_c250")
        ('sequential', 'SASRec', 250)

    """
    retriever_type: str | None = None
    retriever_model: str | None = None
    count: str | int | None = None

    if n_candidates is None:
        return retriever_type, retriever_model, count

    if isinstance(n_candidates, int):
        return "cbf", "CBF", n_candidates

    if isinstance(n_candidates, str):
        # Legacy label: c0, c250, c500, c1000, cALL
        if n_candidates.startswith("c"):
            if n_candidates == "cALL":
                return "cbf", "CBF", None
            try:
                count = int(n_candidates[1:])
            except ValueError:
                pass
            else:
                return "cbf", "CBF", count

        # Named pool label: cf_EASE_c250, seq_SASRec_c250
        parts = n_candidates.split("_")
        if len(parts) >= 3 and parts[-1].startswith("c"):
            type_prefix = parts[0]
            model_name = "_".join(parts[1:-1])
            if parts[-1] == "cALL":
                count = None
            else:
                try:
                    count = int(parts[-1][1:])
                except ValueError:
                    return None, None, n_candidates
            if type_prefix == "cf":
                retriever_type = "cf"
                retriever_model = model_name
            elif type_prefix == "seq":
                retriever_type = "sequential"
                retriever_model = model_name

    if retriever_type is None:
        count = _ensure_json_safe(n_candidates)

    return retriever_type, retriever_model, count


def normalize_llm_result_row(
    row: dict[str, Any],
    title_to_item_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Normalize an LLM evaluation result row into the shared artifact schema.

    Maps ``pred_items`` and ``ground_truth`` title lists to item IDs when a
    ``title_to_item_id`` mapping is provided.  Sets explicit retriever and
    reranker metadata based on the candidate pool label and LLM model name.

    Args:
        row: Dictionary representing a single LLM evaluation result, typically
            from a DataFrame produced by ``scripts/04_evaluation.ipynb``.
        title_to_item_id: Optional mapping from normalized title to item ID.
            When provided, titles in ``pred_items`` and ``ground_truth`` are
            looked up and converted to ``ranked_item_ids`` and
            ``ground_truth_item_ids``.  Unknown titles are silently omitted.

    Returns:
        Normalized dictionary conforming to the shared ranking artifact schema.

    """
    reranker_model = _first_present(
        row.get("model"),
        row.get("model_id"),
        row.get("reranker_model"),
    )

    n_candidates_raw = _first_present(
        row.get("candidate_pool_label"),
        row.get("n_candidates"),
    )
    parsed_retriever_type, parsed_retriever_model, n_candidates = (
        parse_candidate_pool_label(
            n_candidates_raw,
        )
    )

    # Prefer explicit retriever metadata already present in the row (e.g. named pools).
    # Pandas concatenation can materialize absent columns as NaN, which must not mask
    # metadata parsed from legacy candidate labels.
    retriever_type = _first_present(row.get("retriever_type"), parsed_retriever_type)
    retriever_model = _first_present(row.get("retriever_model"), parsed_retriever_model)

    pred_titles = _ensure_list(row.get("pred_items"))
    gt_titles = _ensure_list(row.get("ground_truth"))

    ranked_item_ids: list[int] = []
    ground_truth_item_ids: list[int] = []

    if title_to_item_id is not None:
        for title in pred_titles:
            if title is None:
                continue
            norm_title = normalize_movie_title(str(title))
            if norm_title is not None and norm_title in title_to_item_id:
                ranked_item_ids.append(title_to_item_id[norm_title])
        for title in gt_titles:
            if title is None:
                continue
            norm_title = normalize_movie_title(str(title))
            if norm_title is not None and norm_title in title_to_item_id:
                ground_truth_item_ids.append(title_to_item_id[norm_title])

    normalized: dict[str, Any] = {
        "model_type": _first_present(row.get("model_type"), "llm"),
        "model": _first_present(row.get("model"), reranker_model),
        "eval_method": "reranking",
        "retriever_type": retriever_type,
        "retriever_model": retriever_model,
        "reranker_type": "llm",
        "reranker_model": reranker_model,
        "n_candidates": n_candidates,
        "user_id": row.get("user_id"),
        "prompt_idx": _ensure_json_safe(row.get("prompt_idx")),
        "ranked_item_ids": ranked_item_ids,
        "ranked_titles": pred_titles,
        "ranked_scores": [],
        "ground_truth_item_ids": ground_truth_item_ids,
        "ground_truth_titles": gt_titles,
    }

    return normalize_ranking_row(normalized)


def normalize_llm_results(
    df: pd.DataFrame,
    item_path: str | Path | None = None,
    title_to_item_id: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Normalize a DataFrame of LLM evaluation results into the shared schema.

    Convenience wrapper around :func:`normalize_llm_result_row` that
    operates on an entire DataFrame and returns a DataFrame with stable
    column ordering.

    Args:
        df: DataFrame of LLM evaluation results.
        item_path: Optional path to a RecBole .item file.  If provided and
            ``title_to_item_id`` is not given, a title mapping is built
            automatically.
        title_to_item_id: Optional pre-built title -> item_id mapping.

    Returns:
        DataFrame conforming to the shared ranking artifact schema.

    Raises:
        ValueError: If ``df`` is empty.

    """
    if title_to_item_id is None and item_path is not None:
        title_to_item_id = build_title_to_item_id_mapping(item_path)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(normalize_llm_result_row(row.to_dict(), title_to_item_id))

    return ranking_rows_to_dataframe(rows)


logger = logging.getLogger(__name__)

RANKING_ARTIFACT_COLUMNS = [
    "model_type",
    "model",
    "eval_method",
    "retriever_type",
    "retriever_model",
    "reranker_type",
    "reranker_model",
    "n_candidates",
    "user_id",
    "prompt_idx",
    "ranked_item_ids",
    "ranked_titles",
    "ranked_scores",
    "ground_truth_item_ids",
    "ground_truth_titles",
]

RANKING_ARTIFACT_DEFAULTS: dict[str, Any] = {
    "model_type": None,
    "model": None,
    "eval_method": None,
    "retriever_type": None,
    "retriever_model": None,
    "reranker_type": None,
    "reranker_model": None,
    "n_candidates": None,
    "user_id": None,
    "prompt_idx": None,
    "ranked_item_ids": [],
    "ranked_titles": [],
    "ranked_scores": [],
    "ground_truth_item_ids": [],
    "ground_truth_titles": [],
}


def _ensure_list(value: Any) -> list[Any]:  # noqa: ANN401
    """Return a JSON/parquet-safe list from possible sequence types.

    Converts numpy arrays, tuples, and pandas Series to plain Python lists.
    Returns an empty list for None, NaN, or missing values.

    Args:
        value: Input value to convert to a list.

    Returns:
        A plain Python list with JSON-serializable elements.

    """
    if _is_missing_scalar(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _ensure_json_safe(value: Any) -> Any:  # noqa: ANN401
    """Return a JSON-serializable scalar value.

    Converts numpy scalar types to native Python types and numpy NaN to None.

    Args:
        value: Input value to make JSON-safe.

    Returns:
        A native Python type or None.

    """
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value):
            return None
        return value.item()
    return value


def _is_missing_scalar(value: Any) -> bool:  # noqa: ANN401
    """Return True for scalar missing values without treating sequences as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_present(*values: Any) -> Any:  # noqa: ANN401
    """Return the first value that is not None/NaN/empty string."""
    for value in values:
        if not _is_missing_scalar(value):
            return value
    return None


def _ensure_extra_value_safe(value: Any) -> Any:  # noqa: ANN401
    """Return a parquet-safe value for non-schema artifact columns."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return _ensure_json_safe(value)


def normalize_ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single per-user ranked recommendation row.

    Ensures the row contains all fields defined in the shared ranking artifact
    schema, filling missing optional fields with defaults. List columns are
    converted to plain Python lists. Scalar columns are made JSON-safe.

    Args:
        row: Dictionary representing a single per-user ranking result.

    Returns:
        Normalized dictionary with stable keys and safe types.

    """
    normalized: dict[str, Any] = {}

    for col in RANKING_ARTIFACT_COLUMNS:
        value = row[col] if col in row else RANKING_ARTIFACT_DEFAULTS.get(col)

        if col in {
            "ranked_item_ids",
            "ranked_titles",
            "ranked_scores",
            "ground_truth_item_ids",
            "ground_truth_titles",
        }:
            normalized[col] = _ensure_list(value)
        elif col == "n_candidates" and value is not None:
            # Allow numeric candidate count or legacy string labels like "c250"
            if isinstance(value, str) and value.startswith("c"):
                try:
                    normalized[col] = int(value[1:])
                except ValueError:
                    normalized[col] = value
            else:
                normalized[col] = _ensure_json_safe(value)
        else:
            normalized[col] = _ensure_json_safe(value)

    return normalized


def ranking_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of per-user ranking rows into a pandas DataFrame.

    Each row is normalized to the shared schema and assembled with stable
    column ordering. List columns remain as plain Python lists, making the
    resulting DataFrame safe for JSON and parquet serialization.

    Args:
        rows: List of per-user ranking dictionaries.

    Returns:
        DataFrame with stable columns and JSON/parquet-safe list types.

    Raises:
        ValueError: If rows is empty.

    """
    if not rows:
        msg = "rows must not be empty"
        raise ValueError(msg)

    normalized_rows: list[dict[str, Any]] = []
    extra_columns: list[str] = []
    for row in rows:
        normalized = normalize_ranking_row(row)
        for col, value in row.items():
            if col in RANKING_ARTIFACT_COLUMNS:
                continue
            if col not in extra_columns:
                extra_columns.append(col)
            normalized[col] = _ensure_extra_value_safe(value)
        normalized_rows.append(normalized)

    df = pd.DataFrame(normalized_rows)
    df = df[RANKING_ARTIFACT_COLUMNS + extra_columns]

    # Verify list column dtypes remain object (plain lists), not numpy arrays
    list_cols = [
        "ranked_item_ids",
        "ranked_titles",
        "ranked_scores",
        "ground_truth_item_ids",
        "ground_truth_titles",
    ]
    for col in list_cols:
        if not df[col].apply(lambda x: isinstance(x, list)).all():
            logger.warning(
                "Column %s contains non-list values after normalization", col
            )

    return df


# ---------------------------------------------------------------------------
# Catalog and popularity loaders
# ---------------------------------------------------------------------------


def load_catalog(item_path: str | Path) -> tuple[dict[int, str], list[int]]:
    """Load movie catalog from a RecBole .item file.

    Expects a tab-separated file with header columns ``item_id:token``
    and ``title:token_seq``.

    Args:
        item_path: Path to the RecBole item file.

    Returns:
        Tuple of ``(id_to_title, item_ids)`` where ``id_to_title`` maps
        integer item IDs to title strings and ``item_ids`` is the ordered
        list of item IDs as they appear in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.

    """
    path = Path(item_path)
    if not path.exists():
        msg = f"Catalog file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(
        path, sep="\t", dtype={"item_id:token": str, "title:token_seq": str}
    )

    if "item_id:token" not in df.columns or "title:token_seq" not in df.columns:
        msg = f"Missing required columns in {path}: {df.columns.tolist()}"
        raise ValueError(msg)

    id_to_title: dict[int, str] = {}
    item_ids: list[int] = []
    for raw_id, title in zip(
        df["item_id:token"],
        df["title:token_seq"],
        strict=False,
    ):
        item_id = int(raw_id)
        id_to_title[item_id] = str(title)
        item_ids.append(item_id)

    return id_to_title, item_ids


def load_popularity(
    train_inter_path: str | Path,
    catalog_item_ids: list[int],
) -> pd.Series:
    """Load item popularity counts from a RecBole train interaction file.

    Expects a tab-separated file with an ``item_id:token`` column.
    Items present in ``catalog_item_ids`` but absent from the training
    interactions receive a count of zero.

    Args:
        train_inter_path: Path to the RecBole train interaction file.
        catalog_item_ids: Ordered list of catalog item IDs.

    Returns:
        Pandas Series indexed by item ID with integer count values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the required column is missing.

    """
    path = Path(train_inter_path)
    if not path.exists():
        msg = f"Train interaction file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path, sep="\t", dtype={"item_id:token": str})

    if "item_id:token" not in df.columns:
        msg = f"Missing 'item_id:token' column in {path}: {df.columns.tolist()}"
        raise ValueError(msg)

    counts = df["item_id:token"].value_counts()
    counts.index = counts.index.astype(int)

    # Reindex to the full catalog, filling missing items with zero
    return counts.reindex(catalog_item_ids, fill_value=0).astype(int)


def compute_tail_items(
    popularity_counts: pd.Series,
    tail_fraction: float = 0.1,
) -> set[int]:
    """Return the least-popular tail item set.

    Sorts items by popularity ascending and selects the bottom
    ``tail_fraction`` of the catalog.

    Args:
        popularity_counts: Series indexed by item ID with popularity counts.
        tail_fraction: Fraction of the catalog to treat as tail (default 0.1).

    Returns:
        Set of item IDs belonging to the tail.

    """
    n_catalog = len(popularity_counts)
    if n_catalog == 0:
        return set()

    n_tail = max(1, round(tail_fraction * n_catalog))
    sorted_counts = popularity_counts.sort_values(ascending=True)
    tail_ids = sorted_counts.index[:n_tail].tolist()
    return set(tail_ids)


def load_genre_mapping(
    metadata_path: str | Path,
    catalog_id_to_title: dict[int, str],
) -> dict[int, list[str]]:
    """Load genre mapping from TMDB metadata CSV.

    The CSV is expected to have a ``title_norm`` column and a ``genres``
    column containing string representations of Python lists such as
    ``"['Thriller']"``. Titles are normalized (whitespace collapsed) before
    matching to align with the catalog titles from :func:`load_catalog`.

    Args:
        metadata_path: Path to the TMDB metadata CSV file.
        catalog_id_to_title: Mapping from item ID to title (from
            :func:`load_catalog`).

    Returns:
        Dictionary mapping item ID to a list of genre strings. Unmatched
        items or items with missing/empty genres receive an empty list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.

    """
    path = Path(metadata_path)
    if not path.exists():
        msg = f"Metadata file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(
        path,
        dtype={"title_norm": str, "genres": str},
        keep_default_na=True,
    )

    if "title_norm" not in df.columns or "genres" not in df.columns:
        msg = f"Missing required columns in {path}: {df.columns.tolist()}"
        raise ValueError(msg)

    # Build normalized title -> genres lookup
    title_to_genres: dict[str, list[str]] = {}
    for raw_title, raw_genres in zip(
        df["title_norm"],
        df["genres"],
        strict=False,
    ):
        norm_title = normalize_movie_title(str(raw_title))
        if norm_title is None:
            continue
        genres: list[str] = []
        if pd.notna(raw_genres) and str(raw_genres).strip():
            try:
                parsed = ast.literal_eval(str(raw_genres))
                if isinstance(parsed, list):
                    genres = [str(g).strip() for g in parsed if str(g).strip()]
            except (ValueError, SyntaxError):
                logger.debug(
                    "Could not parse genres for '%s': %s", norm_title, raw_genres
                )
        title_to_genres[norm_title] = genres

    # Map catalog item IDs to genres
    genre_mapping: dict[int, list[str]] = {}
    for item_id, title in catalog_id_to_title.items():
        norm_title = normalize_movie_title(title)
        if norm_title is not None and norm_title in title_to_genres:
            genre_mapping[item_id] = title_to_genres[norm_title]
        else:
            genre_mapping[item_id] = []

    return genre_mapping


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------


def _gini_index(counts: np.ndarray) -> float:
    """Compute Gini index from a count array.

    Args:
        counts: Array of non-negative integer counts.

    Returns:
        Gini index in [0, 1] or NaN if total count is zero.

    """
    n_total = counts.sum()
    if n_total == 0:
        return float(np.nan)
    probs = counts / n_total
    sorted_probs = np.sort(probs)
    n = len(probs)
    return float(
        (2 * np.sum(np.arange(1, n + 1) * sorted_probs) - n - 1) / n,
    )


def _shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy from a count array.

    Args:
        counts: Array of non-negative integer counts.

    Returns:
        Shannon entropy (natural log) or NaN if total count is zero.

    """
    n_total = counts.sum()
    if n_total == 0:
        return float(np.nan)
    probs = counts / n_total
    probs_nonzero = probs[probs > 0]
    return float(-np.sum(probs_nonzero * np.log(probs_nonzero)))


def item_coverage_at_k(
    ranked_item_ids: list[list[int]],
    catalog_item_ids: list[int],
    k: int,
) -> float:
    """Compute item coverage@K.

    Coverage is the fraction of distinct catalog items that appear in the
    top-K recommendations across all users. Unknown item IDs are excluded.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        catalog_item_ids: Ordered list of all catalog item IDs.
        k: Cutoff rank.

    Returns:
        Item coverage in [0, 1] or NaN if the catalog is empty or no
        recommendations are present.

    """
    if not catalog_item_ids:
        return float(np.nan)
    catalog_set = set(catalog_item_ids)
    recommended: set[int] = set()
    n_recs = 0
    for user_items in ranked_item_ids:
        topk = user_items[:k]
        n_recs += len(topk)
        for item_id in topk:
            if item_id in catalog_set:
                recommended.add(item_id)
    if n_recs == 0:
        return float(np.nan)
    return len(recommended) / len(catalog_item_ids)


def average_popularity_at_k(
    ranked_item_ids: list[list[int]],
    popularity_counts: pd.Series,
    k: int,
) -> float:
    """Compute average popularity@K.

    Average training popularity of each recommended item instance in the
    top-K across all users. Items not present in ``popularity_counts``
    are excluded.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        popularity_counts: Series indexed by item ID with popularity counts.
        k: Cutoff rank.

    Returns:
        Mean popularity or NaN if no valid recommendations.

    """
    scores: list[int] = []
    for user_items in ranked_item_ids:
        for item_id in user_items[:k]:
            if item_id in popularity_counts.index:
                scores.append(int(popularity_counts[item_id]))
    if not scores:
        return float(np.nan)
    return float(np.mean(scores))


def gini_index_at_k(
    ranked_item_ids: list[list[int]],
    catalog_item_ids: list[int],
    k: int,
) -> float:
    """Compute Gini index@K over item recommendation counts.

    Builds a full count vector over the entire catalog (including zeros
    for unrecommended items) and applies the Gini coefficient formula.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        catalog_item_ids: Ordered list of all catalog item IDs.
        k: Cutoff rank.

    Returns:
        Gini index or NaN if the catalog is empty.

    """
    if not catalog_item_ids:
        return float(np.nan)
    counts = np.zeros(len(catalog_item_ids), dtype=int)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(catalog_item_ids)}
    for user_items in ranked_item_ids:
        for item_id in user_items[:k]:
            if item_id in item_to_idx:
                counts[item_to_idx[item_id]] += 1
    return _gini_index(counts)


def shannon_entropy_at_k(
    ranked_item_ids: list[list[int]],
    catalog_item_ids: list[int],
    k: int,
) -> float:
    """Compute Shannon entropy@K over item recommendation counts.

    Builds a full count vector over the entire catalog and computes the
    Shannon entropy of the recommendation distribution.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        catalog_item_ids: Ordered list of all catalog item IDs.
        k: Cutoff rank.

    Returns:
        Shannon entropy (natural log) or NaN if the catalog is empty.

    """
    if not catalog_item_ids:
        return float(np.nan)
    counts = np.zeros(len(catalog_item_ids), dtype=int)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(catalog_item_ids)}
    for user_items in ranked_item_ids:
        for item_id in user_items[:k]:
            if item_id in item_to_idx:
                counts[item_to_idx[item_id]] += 1
    return _shannon_entropy(counts)


def tail_percentage_at_k(
    ranked_item_ids: list[list[int]],
    tail_item_ids: set[int],
    k: int,
) -> float:
    """Compute tail percentage@K.

    Fraction of recommended item instances in top-K that belong to the
    tail item set.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        tail_item_ids: Set of item IDs in the tail.
        k: Cutoff rank.

    Returns:
        Tail percentage in [0, 1] or NaN if no recommendations.

    """
    total = 0
    tail_count = 0
    for user_items in ranked_item_ids:
        for item_id in user_items[:k]:
            total += 1
            if item_id in tail_item_ids:
                tail_count += 1
    if total == 0:
        return float(np.nan)
    return tail_count / total


def genre_coverage_at_k(
    ranked_item_ids: list[list[int]],
    genre_mapping: dict[int, list[str]],
    k: int,
) -> float:
    """Compute genre coverage@K.

    Fraction of distinct catalog genres that appear in the top-K
    recommendations across all users.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        genre_mapping: Mapping from item ID to list of genre strings.
        k: Cutoff rank.

    Returns:
        Genre coverage in [0, 1] or NaN if the catalog has no genres.

    """
    all_genres: set[str] = set()
    for genres in genre_mapping.values():
        all_genres.update(genres)
    n_catalog_genres = len(all_genres)
    if n_catalog_genres == 0:
        return float(np.nan)

    recommended_genres: set[str] = set()
    for user_items in ranked_item_ids:
        for item_id in user_items[:k]:
            if item_id in genre_mapping:
                recommended_genres.update(genre_mapping[item_id])
    return len(recommended_genres) / n_catalog_genres


def genre_entropy_at_k(
    ranked_item_ids: list[list[int]],
    genre_mapping: dict[int, list[str]],
    k: int,
) -> float:
    """Compute genre entropy@K.

    Shannon entropy over genre occurrences in top-K recommendations.
    Each occurrence of a genre on a recommended item counts once.

    Args:
        ranked_item_ids: List of per-user ranked item ID lists.
        genre_mapping: Mapping from item ID to list of genre strings.
        k: Cutoff rank.

    Returns:
        Genre entropy (natural log) or NaN if no genres are recommended.

    """
    genre_counts: dict[str, int] = {}
    total = 0
    for user_items in ranked_item_ids:
        for item_id in user_items[:k]:
            if item_id in genre_mapping:
                for genre in genre_mapping[item_id]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    total += 1
    if total == 0:
        return float(np.nan)
    counts_arr = np.array(list(genre_counts.values()), dtype=float)
    probs = counts_arr / total
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return bootstrap confidence interval for a metric over users.

    Resamples rows (users) with replacement and recomputes ``metric_fn``
    on each bootstrap sample. The percentile method is used to construct
    the interval.

    Args:
        df: DataFrame where each row is an independent observation
            (e.g. one user).
        metric_fn: Callable that receives a resampled DataFrame and returns
            a scalar metric value. For per-user scalar metrics this can
            simply take the mean of a column. For system-level metrics it
            can extract ranked lists from the resampled rows and recompute
            the metric (e.g. coverage or entropy).
        n_bootstraps: Number of bootstrap samples. Defaults to 1000.
        ci: Confidence level, e.g. 0.95 for a 95% interval.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
        - ``metric``: the metric computed on the original (unresampled) data
        - ``lower``: lower bound of the percentile interval
        - ``upper``: upper bound of the percentile interval
        - ``n_bootstraps``: number of bootstrap samples used
        - ``ci``: confidence level
        - ``std``: standard deviation of bootstrap samples

    """
    n = len(df)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample = df.iloc[indices].reset_index(drop=True)
        estimates[i] = metric_fn(sample)

    alpha = 1 - ci
    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    original = float(metric_fn(df))

    return {
        "metric": original,
        "lower": lower,
        "upper": upper,
        "n_bootstraps": n_bootstraps,
        "ci": ci,
        "std": float(np.std(estimates, ddof=1)),
    }


def compute_per_user_accuracy(
    df: pd.DataFrame,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Compute per-user accuracy metrics from ranked titles and ground-truth titles.

    Applies :func:`stability.metrics.recommendation_metrics` row-wise and
    adds columns ``hit_rate@k``, ``mrr@k``, ``precision@k``, ``recall@k``,
    ``f1@k``, and ``ndcg@k`` for each ``k`` in ``k_values``.

    Args:
        df: DataFrame containing ``ranked_titles`` and
            ``ground_truth_titles`` list columns.
        k_values: Cutoff ranks to compute. Defaults to ``[1, 5, 10]``.

    Returns:
        A copy of ``df`` with additional per-user metric columns.

    """
    if k_values is None:
        k_values = [1, 5, 10]

    result = df.copy()
    metric_names = ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg"]
    for k in k_values:
        for name in metric_names:
            result[f"{name}@{k}"] = np.nan

    for idx, row in result.iterrows():
        predictions = row["ranked_titles"]
        ground_truth = row["ground_truth_titles"]
        metrics = recommendation_metrics(predictions, ground_truth, k_values)
        for key, value in metrics.items():
            result.loc[idx, key] = value

    return result


def aggregate_accuracy_metrics(
    df: pd.DataFrame,
    k_values: list[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate per-user accuracy metrics into long-form rows.

    Computes the mean of each per-user metric column and returns one row
    per metric-cutoff pair with optional method metadata.

    Args:
        df: DataFrame containing per-user metric columns like
            ``hit_rate@k``.
        k_values: Cutoff ranks. Defaults to ``[1, 5, 10]``.
        metadata: Optional method metadata columns to include in each row,
            such as ``model_type``, ``model``, ``retriever_type``, etc.

    Returns:
        Long-form DataFrame with columns ``metric``, ``k``, ``value``,
        and any metadata keys provided.

    """
    if k_values is None:
        k_values = [1, 5, 10]

    meta = metadata or {}
    metric_names = ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg"]
    rows: list[dict[str, Any]] = []

    for k in k_values:
        for name in metric_names:
            col = f"{name}@{k}"
            value = float(df[col].mean()) if col in df.columns else float(np.nan)
            rows.append({"metric": name, "k": k, "value": value, **meta})

    return pd.DataFrame(rows)
