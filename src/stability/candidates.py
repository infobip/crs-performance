"""Candidate selection for movie recommendations."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_candidates_block(
    user_liked: list[str],
    user_disliked: list[str],
    all_titles: list[str],
    embeds: np.ndarray,
    title_to_idx: dict[str, int],
    pool_size: int = 100,
    like_weight: float = 1.0,
    dislike_weight: float = 0.5,
    shuffle: bool = True,
    ground_truth: list[str] | None = None,
    rng: np.random.Generator | None = None,
    return_ranked: bool = False,
    scoring_strategy: str = "centroid",
    popularity_weights: np.ndarray | None = None,
    genre_bonus: dict[int, set[str]] | None = None,
) -> list[str] | tuple[list[str], list[str]]:
    """Select top-K candidates by embedding similarity.

    Strategy:
    1. Compute positive centroid from liked movies (or per-query max-pool)
    2. Optionally compute negative centroid from disliked movies
    3. Score all movies:
       like_weight * similarity_to_positive - dislike_weight * similarity_to_negative
    4. Select top pool_size movies (excluding already liked/disliked)
    5. If ground_truth provided, ensure those movies are in candidates
    6. Shuffle to prevent positional bias

    Args:
        user_liked: List of movie titles the user liked.
        user_disliked: List of movie titles the user disliked.
        all_titles: List of all movie titles in the dataset.
        embeds: Array of shape (num_movies, embedding_dim) with movie embeddings.
        title_to_idx: Mapping from movie title to its index in embeds/all_titles.
        pool_size: Number of candidate movies to select.
        like_weight: Weight for positive similarity score.
        dislike_weight: Weight for negative similarity score.
        shuffle: Whether to shuffle final candidates to prevent positional bias.
        ground_truth: Optional list of movie titles to ensure inclusion.
        rng: Optional numpy random Generator for shuffling.
        return_ranked: If True, return a tuple of (shuffled, ranked) candidates.
        scoring_strategy: Scoring method for liked movies. 'centroid' (default)
            averages liked embeddings; 'max_pool' takes max similarity per query.
        popularity_weights: Optional array of shape (num_movies,) with popularity
            scores. When provided, adds alpha * popularity_weights to scores
            (alpha=0.01) before candidate selection.
        genre_bonus: Optional dict mapping movie index to its genre set.
            When provided, adds beta * genre_overlap to scores (beta=0.1)
            where genre_overlap = |candidate_genres & liked_genres| / max(|liked_genres|, 1).

    Returns:
        List of candidate movie titles when return_ranked=False (default),
        or tuple of (shuffled, ranked) lists when return_ranked=True.

    """
    if scoring_strategy not in ("centroid", "max_pool"):
        msg = f"Unknown scoring_strategy: {scoring_strategy!r}. Use 'centroid' or 'max_pool'."
        raise ValueError(msg)

    logger.debug(
        f"Selecting {pool_size} candidates from {len(all_titles)} movies "
        f"({len(user_liked)} liked, {len(user_disliked)} disliked)",
    )

    # Filter liked movies that exist in embeddings
    liked_valid = [m for m in user_liked if m in title_to_idx]
    disliked_valid = [m for m in user_disliked if m in title_to_idx]

    if not liked_valid:
        logger.warning(
            "No valid liked movies found in embeddings. Returning random sample.",
        )
        disliked_set = set(user_disliked)
        candidates = [t for t in all_titles if t not in disliked_set]
        ranked_candidates = list(candidates[:pool_size])
        if shuffle:
            _rng = rng if rng is not None else np.random.default_rng()
            indices = _rng.permutation(len(candidates))
            candidates = [candidates[i] for i in indices]
        result = candidates[:pool_size]
        if return_ranked:
            return (result, ranked_candidates)
        return result

    # Compute positive scores from liked movies
    liked_indices = [title_to_idx[m] for m in liked_valid]

    if scoring_strategy == "max_pool":
        # Max-pool: score each movie against each liked movie, take the max
        per_query_scores = embeds @ embeds[liked_indices].T  # (N, num_liked)
        scores = like_weight * per_query_scores.max(axis=1)
    else:
        # Centroid: average liked embeddings, compute single dot product
        pos_centroid = embeds[liked_indices].mean(axis=0)
        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-9)
        scores = like_weight * (embeds @ pos_centroid)

    # Subtract negative similarity if disliked movies exist
    if disliked_valid:
        disliked_indices = [title_to_idx[m] for m in disliked_valid]
        neg_centroid = embeds[disliked_indices].mean(axis=0)
        neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-9)
        scores -= dislike_weight * (embeds @ neg_centroid)

    # Add popularity bias if provided
    if popularity_weights is not None:
        scores += 0.01 * popularity_weights

    # Add genre overlap bonus if provided
    if genre_bonus is not None:
        liked_genres: set[str] = set()
        for idx in liked_indices:
            liked_genres |= genre_bonus.get(idx, set())
        n_liked_genres = max(len(liked_genres), 1)
        for i in range(len(all_titles)):
            candidate_genres = genre_bonus.get(i, set())
            overlap = len(candidate_genres & liked_genres) / n_liked_genres
            scores[i] += 0.1 * overlap

    # Exclude already liked/disliked movies from candidates
    excluded = set(liked_valid) | set(disliked_valid)
    candidate_indices = [
        i for i, movie in enumerate(all_titles) if movie not in excluded
    ]

    if not candidate_indices:
        logger.warning("All movies excluded. Returning empty list.")
        if return_ranked:
            return ([], [])
        return []

    # Get scores for candidates only
    candidate_scores = scores[candidate_indices]

    # Select top pool_size candidates
    n_select = min(pool_size, len(candidate_indices))
    if n_select < len(candidate_indices):
        # Use argpartition for efficiency
        top_indices = np.argpartition(-candidate_scores, n_select - 1)[:n_select]
        # Sort the top indices by score (descending)
        top_indices = top_indices[np.argsort(-candidate_scores[top_indices])]
    else:
        top_indices = np.argsort(-candidate_scores)

    # Map back to movie titles
    selected = [all_titles[candidate_indices[i]] for i in top_indices[:n_select]]

    # Inject ground truth if provided (for training)
    if ground_truth:
        # Filter ground truth movies that exist in all_titles
        ground_truth_valid = [
            m for m in ground_truth if m in title_to_idx and m not in excluded
        ]

        if ground_truth_valid:
            # Remove ground truth from selected if already present
            selected_set = set(selected)
            to_inject = [m for m in ground_truth_valid if m not in selected_set]

            if to_inject:
                # Add ground truth movies
                selected.extend(to_inject)

                # If we exceed pool_size, remove lowest-scoring originals
                if len(selected) > pool_size:
                    # Keep all injected ground truth + top originals
                    n_keep_original = pool_size - len(to_inject)
                    selected = selected[:n_keep_original] + to_inject

                logger.debug(
                    f"Injected {len(to_inject)} ground truth movies into candidates",
                )

    # Save pre-shuffle ranked order if requested
    ranked_candidates = list(selected) if return_ranked else None

    # Shuffle to prevent positional bias
    if shuffle and len(selected) > 1:
        _rng = rng if rng is not None else np.random.default_rng()
        indices = _rng.permutation(len(selected))
        selected = [selected[i] for i in indices]

    logger.debug(f"Selected {len(selected)} candidates")
    if return_ranked:
        return (selected, ranked_candidates)  # type: ignore[return-value]
    return selected
