"""Recommendation and stability metrics for LLM evaluation.

This module provides functions for computing:
- Recommendation quality metrics (Hit@K, MRR@K, Precision@K, Recall@K, F1@K, NDCG@K)
- Diversity metrics (Gini, entropy, variation ratio, cosine diversity)
- Label distribution metrics
- Embedding distance metrics
"""

import logging

import numpy as np

from stability.utils import canonicalize

logger = logging.getLogger(__name__)


def cosine_distance_matrix(embeds: np.ndarray) -> np.ndarray:
    """Return cosine distance matrix D = 1 - cosine_similarity, with zero diagonal.

    Args:
        embeds: Input embeddings matrix (n_samples, n_features)

    Returns:
        Distance matrix (n_samples, n_samples) with zero diagonal

    """
    normalized = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def label_metrics(counts: np.ndarray) -> dict[str, float]:
    """Return label distribution metrics.

    Computes diversity metrics for categorical distributions including
    Gini coefficient, entropy, variation ratio, and unique count.

    Args:
        counts: Array of counts for each label/category

    Returns:
        Dictionary with keys: gini, entropy, variation_ratio, unique_count

    Notes:
        - Gini coefficient measures inequality
          (0=perfect equality, 1=maximal inequality)
        - Shannon entropy measures unpredictability
        - Variation ratio = 1 - (mode frequency)
        - Unique count = number of non-zero categories

    """
    n_total = counts.sum()
    if n_total <= 0:
        return {
            "gini": np.nan,
            "entropy": np.nan,
            "variation_ratio": np.nan,
            "unique_count": 0.0,
        }

    probs = counts / n_total
    sorted_probs = np.sort(probs)
    n_items = len(probs)

    # Gini coefficient
    gini = (
        2 * np.sum(np.arange(1, n_items + 1) * sorted_probs) / np.sum(probs)
        - n_items
        - 1
    ) / n_items

    # Shannon entropy with epsilon protection for numerical stability
    probs_nonzero = probs[probs > 0]
    probs_clipped = np.clip(probs_nonzero, 1e-10, 1.0)
    entropy = float(-np.sum(probs_nonzero * np.log(probs_clipped)))

    # Variation ratio and unique count
    variation_ratio = 1.0 - float(np.max(probs))
    unique_count = float(np.count_nonzero(counts))

    return {
        "gini": gini,
        "entropy": entropy,
        "variation_ratio": variation_ratio,
        "unique_count": unique_count,
    }


def expected_distance_from_counts(
    counts: np.ndarray,
    distance_matrix: np.ndarray,
) -> tuple[float, float]:
    """Return expected embedding distance E = p^T D p and its normalization.

    Args:
        counts: Array of counts for each item
        distance_matrix: Pairwise distance matrix between items

    Returns:
        Tuple of (expected_distance, normalized_distance)
        - expected_distance: E = p^T D p where p is probability distribution
        - normalized_distance: E normalized by uniform distribution distance

    """
    n_total = counts.sum()
    if n_total <= 0:
        return np.nan, np.nan

    probs = counts / n_total
    expected_dist = float(probs @ distance_matrix @ probs)

    n_items = len(probs)
    if n_items <= 1:
        return expected_dist, np.nan

    # Normalize by expected distance under uniform distribution
    uniform_probs = np.ones(n_items) / n_items
    uniform_dist = float(uniform_probs @ distance_matrix @ uniform_probs)
    normalized_dist = expected_dist / uniform_dist if uniform_dist > 0 else np.nan

    return expected_dist, normalized_dist


def mean_pairwise_cosine_distance(embeds: np.ndarray) -> float:
    """Return mean pairwise cosine distance over embeddings.

    Args:
        embeds: Input embeddings matrix (n_samples, n_features)

    Returns:
        Mean cosine distance = 1 - mean_cosine_similarity

    Notes:
        - Only computes upper triangle to avoid duplicate pairs
        - Returns NaN if fewer than 2 samples

    """
    n_samples = embeds.shape[0]
    if n_samples <= 1:
        return np.nan

    normalized = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    similarity = normalized @ normalized.T

    # Use mean of upper triangle similarities (i<j)
    upper_indices = np.triu_indices(n_samples, k=1)
    mean_sim = float(similarity[upper_indices].mean())

    return 1.0 - mean_sim


def cosine_diversity(embeds: np.ndarray) -> float:
    """Calculate cosine diversity as 1 - mean cosine similarity to centroid.

    Args:
        embeds: Input embeddings matrix (n_samples, n_features)

    Returns:
        Cosine diversity score [0, 1]
        - 0 = all embeddings identical (no diversity)
        - 1 = maximum diversity from centroid

    Notes:
        - Computes centroid of all embeddings
        - Measures average distance from centroid
        - Returns NaN if fewer than 2 samples

    """
    n_samples = embeds.shape[0]
    if n_samples <= 1:
        return np.nan

    normalized = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    centroid = np.mean(normalized, axis=0)
    centroid_normalized = centroid / np.linalg.norm(centroid)

    # Calculate cosine similarity to centroid
    similarities = np.dot(normalized, centroid_normalized)

    return 1.0 - np.mean(similarities)


def recommendation_metrics(
    predictions: list[str],
    ground_truth: list[str],
    k_values: list[int],
) -> dict[str, float]:
    """Return hit_rate@K, mrr@K, precision@K, recall@K, f1@K, and ndcg@K.

    Metrics:
        - hit_rate@K: 1 if any ground-truth item appears in top K, else 0.
        - mrr@K: Reciprocal of the rank of the first relevant item.
        - precision@K: TP / K
        - recall@K: TP / len(ground_truth)
        - f1@K: Harmonic mean of precision and recall.
        - ndcg@K: Normalized Discounted Cumulative Gain with binary relevance.

    Args:
        predictions: List of predicted recommendation strings.
        ground_truth: List of ground-truth recommendation strings.
        k_values: List of K values to compute metrics for.

    Returns:
        Dictionary with keys like 'hit_rate@10', 'mrr@10', 'precision@10', etc.

    """
    metrics: dict[str, float] = {}

    # handle empty ground truth
    if not ground_truth:
        for k in k_values:
            for name in ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg"]:
                metrics[f"{name}@{k}"] = np.nan
        return metrics

    # canonicalize for matching
    truth_canon = [canonicalize(t) for t in ground_truth if t]
    pred_canon = [canonicalize(p) for p in predictions if p]

    # run metrics for each K by limiting to top K predictions
    for k in k_values:
        # limit to top K predictions
        topk_preds = pred_canon[:k]

        # binary relevance for top K
        relevance = np.array([1 if p in truth_canon else 0 for p in topk_preds])

        # hit rate
        hit_rate_k = 1.0 if relevance.sum() > 0 else 0.0

        # MRR
        mrr_k = 0.0
        if relevance.any():
            first_hit_idx = np.argmax(relevance)
            mrr_k = 1.0 / (first_hit_idx + 1)

        # precision, recall, F1
        tp = relevance.sum()
        precision_k = tp / k
        recall_k = tp / len(truth_canon)
        f1_k = (
            2 * precision_k * recall_k / (precision_k + recall_k)
            if (precision_k + recall_k) > 0
            else 0.0
        )

        # NDCG (textbook: ideal ranking places all ground-truth items at top)
        dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
        n_relevant = min(len(truth_canon), k)
        ideal_relevance = np.zeros(k)
        ideal_relevance[:n_relevant] = 1
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
        ndcg_k = dcg / idcg if idcg > 0 else 0.0

        # collect results
        metrics[f"hit_rate@{k}"] = hit_rate_k
        metrics[f"mrr@{k}"] = mrr_k
        metrics[f"precision@{k}"] = precision_k
        metrics[f"recall@{k}"] = recall_k
        metrics[f"f1@{k}"] = f1_k
        metrics[f"ndcg@{k}"] = ndcg_k

    return metrics


def recommendation_metrics_positioned(
    predictions: list[str],
    ground_truth: list[str],
    k_values: list[int],
) -> dict[str, float]:
    """Compute metrics with position-weighted relevance for ground truth.

    This function treats ground truth as an ordered list where earlier items
    have higher importance. Relevance scores decrease by position.

    Args:
        predictions: List of predicted recommendation strings
        ground_truth: List of ground-truth recommendations in priority order
        k_values: List of K values to compute metrics for

    Returns:
        Dictionary of metrics keyed by metric name and K value

    Notes:
        - ground_truth[0] has relevance score = len(ground_truth)
        - ground_truth[i] has relevance score = len(ground_truth) - i
        - All strings are canonicalized before comparison

    """
    metrics: dict[str, float] = {}

    if not ground_truth:
        for k in k_values:
            for name in ["precision", "recall", "f1", "ndcg"]:
                metrics[f"{name}@{k}"] = np.nan
        return metrics

    truth_canon = [canonicalize(t) for t in ground_truth if t]
    pred_canon = [canonicalize(p) for p in predictions if p]

    truth_positions = {item: pos for pos, item in enumerate(truth_canon)}
    for k in k_values:
        topk_preds = pred_canon[:k]

        # Position-weighted relevance scores
        relevance = np.zeros(len(topk_preds))
        for i, pred in enumerate(topk_preds):
            pos = truth_positions.get(pred)
            relevance[i] = 0.0 if pos is None else len(truth_canon) - pos

        # Precision, recall, F1
        tp = np.sum(relevance > 0)
        precision_k = tp / k if k > 0 else 0.0
        recall_k = tp / len(truth_canon) if len(truth_canon) > 0 else 0.0
        f1_k = (
            2 * precision_k * recall_k / (precision_k + recall_k)
            if (precision_k + recall_k) > 0
            else 0.0
        )

        # DCG with position-weighted relevance
        gains = relevance / np.log2(np.arange(2, len(relevance) + 2))
        dcg = np.sum(gains)

        # IDCG with ideal ranking
        ideal_gains = np.sort(
            np.array([len(truth_canon) - i for i in range(len(truth_canon))]),
        )[::-1]
        ideal_gains = ideal_gains[:k] / np.log2(np.arange(2, len(ideal_gains[:k]) + 2))
        idcg = np.sum(ideal_gains)

        ndcg_k = dcg / idcg if idcg > 0 else 0.0

        metrics[f"precision@{k}"] = float(precision_k)
        metrics[f"recall@{k}"] = float(recall_k)
        metrics[f"f1@{k}"] = float(f1_k)
        metrics[f"ndcg@{k}"] = float(ndcg_k)

    return metrics
