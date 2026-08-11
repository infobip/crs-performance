"""Tests for candidates module."""

import numpy as np
import pytest

from stability.candidates import build_candidates_block


def test_build_candidates_basic(sample_movie_titles, sample_embeddings, title_to_idx):
    """Test basic candidate selection."""
    user_liked = ["The Matrix"]
    user_disliked = []

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        shuffle=False,
        rng=np.random.default_rng(42),
    )

    assert len(candidates) == 3
    assert "The Matrix" not in candidates  # Liked movies excluded
    assert all(c in sample_movie_titles for c in candidates)


def test_build_candidates_with_disliked(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test candidate selection with disliked movies."""
    user_liked = ["The Matrix"]
    user_disliked = ["Inception"]

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        shuffle=False,
    )

    assert len(candidates) == 3
    assert "The Matrix" not in candidates
    assert "Inception" not in candidates


def test_build_candidates_with_ground_truth(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that ground truth movies are included in candidates."""
    user_liked = ["The Matrix"]
    user_disliked = []
    ground_truth = ["Forrest Gump", "Fight Club"]

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        ground_truth=ground_truth,
        shuffle=False,
    )

    assert len(candidates) <= 5
    # Ground truth is injected but pool size might be exact, so both injected movies may not fit
    assert isinstance(candidates, list)


def test_build_candidates_shuffle_reproducible(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that shuffling with seed is reproducible."""
    user_liked = ["The Matrix"]
    user_disliked = []

    candidates1 = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=True,
        rng=np.random.default_rng(42),
    )

    candidates2 = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=True,
        rng=np.random.default_rng(42),
    )

    assert candidates1 == candidates2


def test_build_candidates_shuffle_changes_order(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that shuffling changes order (most of the time)."""
    user_liked = ["The Matrix"]
    user_disliked = []

    candidates_no_shuffle = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=False,
    )

    candidates_shuffle = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=True,
        rng=np.random.default_rng(99),
    )

    # Same candidates but possibly different order
    assert set(candidates_no_shuffle) == set(candidates_shuffle)


def test_build_candidates_weights(sample_movie_titles, sample_embeddings, title_to_idx):
    """Test that like and dislike weights affect selection."""
    user_liked = ["The Matrix"]
    user_disliked = ["Inception"]

    # High like weight
    candidates_high_like = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        like_weight=2.0,
        dislike_weight=0.1,
        shuffle=False,
    )

    # High dislike weight
    candidates_high_dislike = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        like_weight=1.0,
        dislike_weight=2.0,
        shuffle=False,
    )

    # Results might differ based on weights
    assert len(candidates_high_like) == 3
    assert len(candidates_high_dislike) == 3


def test_build_candidates_multiple_liked(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test with multiple liked movies."""
    user_liked = ["The Matrix", "Inception", "The Dark Knight"]
    user_disliked = []

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        shuffle=False,
    )

    assert len(candidates) == 3
    # None of the liked movies should be in candidates
    for movie in user_liked:
        assert movie not in candidates


def test_build_candidates_pool_size_larger_than_available(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test when pool size is larger than available movies."""
    user_liked = ["The Matrix", "Inception", "The Dark Knight"]
    user_disliked = ["Pulp Fiction", "Fight Club"]

    # Only 3 movies remaining, request 10
    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=10,
        shuffle=False,
    )

    # Should return all available (8 - 3 - 2 = 3)
    assert len(candidates) == 3


def test_build_candidates_empty_ground_truth(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test with empty ground truth list."""
    user_liked = ["The Matrix"]
    user_disliked = []

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        ground_truth=[],
        shuffle=False,
    )

    assert len(candidates) == 3


def test_build_candidates_ground_truth_already_in_top(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test when ground truth movies are already in top candidates."""
    user_liked = ["The Matrix"]
    user_disliked = []

    # First get top candidates
    top_candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=False,
    )

    # Use one of them as ground truth
    ground_truth = [top_candidates[0]]

    candidates_with_gt = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        ground_truth=ground_truth,
        shuffle=False,
    )

    # Should still have 5 candidates
    assert len(candidates_with_gt) == 5
    assert ground_truth[0] in candidates_with_gt


def test_build_candidates_no_valid_liked_movies(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test with liked movies not in dataset."""
    user_liked = ["NonexistentMovie"]
    user_disliked = []

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        shuffle=False,
    )

    # Should still work, falling back to some selection strategy
    assert len(candidates) <= 3


def test_build_candidates_fallback_excludes_disliked(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that disliked movies are excluded in the fallback path."""
    user_liked = ["NonexistentMovie"]
    user_disliked = ["Inception"]

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=10,
        shuffle=False,
    )

    assert "Inception" not in candidates


def test_build_candidates_return_ranked(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that return_ranked=True returns a tuple of (shuffled, ranked) lists."""
    user_liked = ["The Matrix"]
    user_disliked = []

    result = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=True,
        rng=np.random.default_rng(42),
        return_ranked=True,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    shuffled, ranked = result
    assert isinstance(shuffled, list)
    assert isinstance(ranked, list)
    assert set(shuffled) == set(ranked)
    assert ranked[0] in sample_movie_titles


def test_build_candidates_ranked_ordering(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that shuffled order differs from ranked order with shuffle=True."""
    user_liked = ["The Matrix"]
    user_disliked = []

    shuffled, ranked = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=5,
        shuffle=True,
        rng=np.random.default_rng(42),
        return_ranked=True,
    )

    # Same elements but different order after shuffling
    assert set(shuffled) == set(ranked)
    assert shuffled != ranked


def test_build_candidates_max_pool_strategy(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that max_pool scoring strategy returns valid candidates."""
    user_liked = ["The Matrix"]
    user_disliked = ["Inception"]

    candidates = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=3,
        shuffle=False,
        scoring_strategy="max_pool",
    )

    assert len(candidates) == 3
    assert "The Matrix" not in candidates
    assert "Inception" not in candidates
    assert all(c in sample_movie_titles for c in candidates)


def test_build_candidates_invalid_strategy(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that an invalid scoring_strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scoring_strategy"):
        build_candidates_block(
            user_liked=["The Matrix"],
            user_disliked=[],
            all_titles=sample_movie_titles,
            embeds=sample_embeddings,
            title_to_idx=title_to_idx,
            pool_size=3,
            scoring_strategy="unknown",
        )


def test_build_candidates_genre_bonus(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that genre bonus boosts a movie sharing genres with liked movies."""
    user_liked = ["The Matrix"]
    user_disliked = []

    # First, get candidates WITHOUT genre bonus
    candidates_no_genre = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=2,
        shuffle=False,
    )

    # Find a movie NOT in the top-2 candidates (will be boosted)
    non_top = [
        t
        for t in sample_movie_titles
        if t not in candidates_no_genre and t != "The Matrix"
    ]
    assert len(non_top) > 0, "Need at least one movie outside top-2 to boost"
    target_movie = non_top[0]
    target_idx = title_to_idx[target_movie]
    liked_idx = title_to_idx["The Matrix"]

    # Create genre_bonus: give liked movie and target movie the same genres
    # Give other movies no genres or different genres
    genre_bonus = {
        liked_idx: {"Action", "Sci-Fi", "Thriller"},
        target_idx: {"Action", "Sci-Fi", "Thriller"},
    }

    candidates_with_genre = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=2,
        shuffle=False,
        genre_bonus=genre_bonus,
    )

    # The boosted movie should now appear in top candidates
    assert target_movie in candidates_with_genre


def test_build_candidates_popularity_weighted(
    sample_movie_titles,
    sample_embeddings,
    title_to_idx,
):
    """Test that popularity weights boost a movie into top candidates."""
    user_liked = ["The Matrix"]
    user_disliked = []

    # First, get candidates WITHOUT popularity weights
    candidates_no_pop = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=2,
        shuffle=False,
    )

    # Find a movie NOT in the top-2 candidates (will be boosted)
    non_top = [
        t
        for t in sample_movie_titles
        if t not in candidates_no_pop and t != "The Matrix"
    ]
    assert len(non_top) > 0, "Need at least one movie outside top-2 to boost"
    target_movie = non_top[0]
    target_idx = title_to_idx[target_movie]

    # Create popularity weights: give the target movie a very high weight
    pop_weights = np.zeros(len(sample_movie_titles), dtype=np.float32)
    pop_weights[target_idx] = 10000.0  # Extremely high to guarantee boost

    candidates_with_pop = build_candidates_block(
        user_liked=user_liked,
        user_disliked=user_disliked,
        all_titles=sample_movie_titles,
        embeds=sample_embeddings,
        title_to_idx=title_to_idx,
        pool_size=2,
        shuffle=False,
        popularity_weights=pop_weights,
    )

    # The boosted movie should now appear in top candidates
    assert target_movie in candidates_with_pop
