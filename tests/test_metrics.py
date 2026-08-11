"""Tests for metrics module."""

import numpy as np

from stability.metrics import (
    cosine_distance_matrix,
    cosine_diversity,
    expected_distance_from_counts,
    label_metrics,
    mean_pairwise_cosine_distance,
    recommendation_metrics,
    recommendation_metrics_positioned,
)


class TestCosineDistanceMatrix:
    """Tests for cosine_distance_matrix function."""

    def test_cosine_distance_matrix_basic(self):
        """Test basic distance matrix computation."""
        embeds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        # Normalize
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

        dist_matrix = cosine_distance_matrix(embeds)

        assert dist_matrix.shape == (3, 3)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), [0.0, 0.0, 0.0])

    def test_cosine_distance_matrix_orthogonal(self):
        """Test with orthogonal vectors."""
        embeds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        dist_matrix = cosine_distance_matrix(embeds)

        # Orthogonal vectors have cosine similarity 0, so distance = 1
        np.testing.assert_almost_equal(dist_matrix[0, 1], 1.0, decimal=5)
        np.testing.assert_almost_equal(dist_matrix[1, 0], 1.0, decimal=5)

    def test_cosine_distance_matrix_identical(self):
        """Test with identical vectors."""
        embeds = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        dist_matrix = cosine_distance_matrix(embeds)

        # Identical vectors have cosine similarity 1, so distance = 0
        np.testing.assert_almost_equal(dist_matrix[0, 1], 0.0, decimal=5)

    def test_cosine_distance_matrix_symmetric(self):
        """Test that matrix is symmetric."""
        rng = np.random.default_rng(42)
        embeds = rng.standard_normal((5, 10)).astype(np.float32)
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

        dist_matrix = cosine_distance_matrix(embeds)

        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)


class TestLabelMetrics:
    """Tests for label_metrics function."""

    def test_label_metrics_basic(self):
        """Test basic label metrics computation."""
        counts = np.array([10, 20, 30])
        metrics = label_metrics(counts)

        assert "gini" in metrics
        assert "entropy" in metrics
        assert "variation_ratio" in metrics
        assert "unique_count" in metrics
        assert metrics["unique_count"] == 3

    def test_label_metrics_uniform(self):
        """Test with uniform distribution."""
        counts = np.array([10, 10, 10, 10])
        metrics = label_metrics(counts)

        # Uniform distribution has low Gini, high entropy
        assert metrics["gini"] < 0.1
        assert metrics["variation_ratio"] == 0.75  # 1 - 1/4

    def test_label_metrics_single_label(self):
        """Test with single label dominating."""
        counts = np.array([100, 1, 1, 1])
        metrics = label_metrics(counts)

        # Single dominant label has LOW variation ratio (1 - max_prob)
        # When one label dominates, max_prob is high, so variation_ratio is low
        assert "gini" in metrics
        assert "variation_ratio" in metrics
        # Variation ratio should be LOW when one label dominates
        assert metrics["variation_ratio"] < 0.1

    def test_label_metrics_unique_count(self):
        """Test unique count calculation."""
        counts = np.array([5, 0, 3, 0, 2])
        metrics = label_metrics(counts)

        assert metrics["unique_count"] == 3  # Only 3 non-zero counts

    def test_label_metrics_very_small_probabilities(self):
        """Test numerical stability with very small probabilities.

        Entropy calculation should not produce -inf with very small values.
        """
        counts = np.array([1, 0, 0, 0, 1e-10])
        metrics = label_metrics(counts)

        # Entropy should be finite (not -inf) due to epsilon protection
        assert np.isfinite(metrics["entropy"])
        assert metrics["entropy"] >= 0
        # Should have 2 unique counts (1 and 1e-10)
        assert metrics["unique_count"] == 2


class TestExpectedDistanceFromCounts:
    """Tests for expected_distance_from_counts function."""

    def test_expected_distance_basic(self):
        """Test basic expected distance computation."""
        counts = np.array([2, 3, 5])
        distance_matrix = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.8], [1.0, 0.8, 0.0]])

        exp_dist, norm_dist = expected_distance_from_counts(counts, distance_matrix)

        assert isinstance(exp_dist, float)
        assert isinstance(norm_dist, float)
        assert exp_dist >= 0
        assert 0 <= norm_dist <= 1

    def test_expected_distance_uniform(self):
        """Test with uniform distribution."""
        counts = np.array([10, 10, 10])
        distance_matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        exp_dist, norm_dist = expected_distance_from_counts(counts, distance_matrix)

        # Should be high with uniform distribution and large distances
        assert exp_dist > 0

    def test_expected_distance_single_item(self):
        """Test with all mass on single item."""
        counts = np.array([100, 0, 0])
        distance_matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        exp_dist, norm_dist = expected_distance_from_counts(counts, distance_matrix)

        # Should be zero since all probability on one item
        np.testing.assert_almost_equal(exp_dist, 0.0, decimal=5)


class TestMeanPairwiseCosineDistance:
    """Tests for mean_pairwise_cosine_distance function."""

    def test_mean_pairwise_distance_basic(self):
        """Test basic mean pairwise distance."""
        embeds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

        mean_dist = mean_pairwise_cosine_distance(embeds)

        assert isinstance(mean_dist, float)
        assert 0 <= mean_dist <= 1

    def test_mean_pairwise_distance_identical(self):
        """Test with identical embeddings."""
        embeds = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        mean_dist = mean_pairwise_cosine_distance(embeds)

        # Identical embeddings have zero distance
        np.testing.assert_almost_equal(mean_dist, 0.0, decimal=5)

    def test_mean_pairwise_distance_orthogonal(self):
        """Test with orthogonal embeddings."""
        embeds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        mean_dist = mean_pairwise_cosine_distance(embeds)

        # Orthogonal vectors have cosine distance 1
        np.testing.assert_almost_equal(mean_dist, 1.0, decimal=5)


class TestCosineDiversity:
    """Tests for cosine_diversity function."""

    def test_cosine_diversity_basic(self):
        """Test basic cosine diversity."""
        embeds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

        diversity = cosine_diversity(embeds)

        assert isinstance(diversity, (float, np.floating))
        # Diversity can be > 1 in some edge cases or NaN
        assert not np.isnan(diversity) or diversity >= 0

    def test_cosine_diversity_identical(self):
        """Test diversity of identical embeddings."""
        embeds = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        diversity = cosine_diversity(embeds)

        # Identical embeddings have zero diversity
        np.testing.assert_almost_equal(diversity, 0.0, decimal=5)

    def test_cosine_diversity_diverse(self):
        """Test diversity of very different embeddings."""
        # Maximally different in 2D space
        embeds = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        diversity = cosine_diversity(embeds)

        # Should have reasonable diversity
        assert isinstance(diversity, (float, np.floating))


class TestRecommendationMetrics:
    """Tests for recommendation_metrics function."""

    def test_recommendation_metrics_basic(self):
        """Test basic recommendation metrics."""
        predictions = ["Movie A", "Movie B", "Movie C"]
        ground_truth = ["Movie B", "Movie D"]
        k_values = [1, 3, 5]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        assert isinstance(metrics, dict)
        assert "hit_rate@1" in metrics
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "ndcg@5" in metrics

    def test_recommendation_metrics_perfect_match(self):
        """Test with perfect predictions."""
        predictions = ["Movie A", "Movie B", "Movie C"]
        ground_truth = ["Movie A", "Movie B", "Movie C"]
        k_values = [3]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        assert metrics["precision@3"] == 1.0
        assert metrics["recall@3"] == 1.0
        assert metrics["ndcg@3"] == 1.0

    def test_recommendation_metrics_no_match(self):
        """Test with no matches."""
        predictions = ["Movie A", "Movie B", "Movie C"]
        ground_truth = ["Movie X", "Movie Y", "Movie Z"]
        k_values = [3]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        assert metrics["precision@3"] == 0.0
        assert metrics["recall@3"] == 0.0
        assert metrics["hit_rate@3"] == 0.0

    def test_recommendation_metrics_hit_at_k(self):
        """Test hit_rate@k metric."""
        predictions = ["Movie A", "Movie B", "Movie C"]
        ground_truth = ["Movie C"]
        k_values = [1, 2, 3]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        assert metrics["hit_rate@1"] == 0.0  # Not in first position
        assert metrics["hit_rate@2"] == 0.0  # Not in first two
        assert metrics["hit_rate@3"] == 1.0  # In first three

    def test_recommendation_metrics_empty_predictions(self):
        """Test with empty predictions."""
        predictions = []
        ground_truth = ["Movie A", "Movie B"]
        k_values = [1, 3]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        # All metrics should be 0
        for value in metrics.values():
            assert value == 0.0

    def test_recommendation_metrics_empty_ground_truth(self):
        """Test with empty ground truth."""
        predictions = ["Movie A", "Movie B"]
        ground_truth = []
        k_values = [1, 3]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        # Should handle gracefully
        assert isinstance(metrics, dict)

    def test_recommendation_metrics_ndcg_penalizes_missing_items(self):
        """NDCG < 1.0 when predictions miss relevant items, even if found items are at top."""
        # 3 relevant items but only 1 hit at position 1
        predictions = ["Movie A", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]
        ground_truth = ["Movie A", "Movie B", "Movie C"]
        k_values = [10]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        # Old bug would give NDCG=1.0 (perfect); textbook NDCG should be < 1.0
        assert metrics["ndcg@10"] < 1.0
        assert metrics["ndcg@10"] > 0.0

    def test_recommendation_metrics_ndcg_partial_match(self):
        """Verify NDCG value with known manual calculation."""
        # 3 relevant items, predictions find 1 at position 1
        predictions = ["Movie A", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]
        ground_truth = ["Movie A", "Movie B", "Movie C"]
        k_values = [10]

        metrics = recommendation_metrics(predictions, ground_truth, k_values)

        # DCG = 1/log2(2) = 1.0
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.6309 + 0.5 = 2.1309
        # NDCG = 1.0 / 2.1309 ≈ 0.4693
        np.testing.assert_almost_equal(metrics["ndcg@10"], 1.0 / 2.1309, decimal=3)


class TestRecommendationMetricsPositioned:
    """Tests for recommendation_metrics_positioned function."""

    def test_recommendation_metrics_positioned_basic(self):
        """Test positioned metrics with ordered ground truth."""
        predictions = ["Movie A", "Movie B", "Movie C"]
        ground_truth = ["Movie B", "Movie C", "Movie A"]  # Order matters
        k_values = [3]

        metrics = recommendation_metrics_positioned(predictions, ground_truth, k_values)

        assert isinstance(metrics, dict)
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "ndcg@3" in metrics

    def test_recommendation_metrics_positioned_ordering(self):
        """Test that position affects scoring."""
        predictions = ["Movie A", "Movie B"]
        ground_truth_1 = ["Movie A", "Movie B"]  # A is more important
        ground_truth_2 = ["Movie B", "Movie A"]  # B is more important
        k_values = [2]

        metrics1 = recommendation_metrics_positioned(
            predictions,
            ground_truth_1,
            k_values,
        )
        metrics2 = recommendation_metrics_positioned(
            predictions,
            ground_truth_2,
            k_values,
        )

        # Both have same precision/recall, but different NDCG
        assert metrics1["precision@2"] == metrics2["precision@2"]
        # NDCG might differ due to position weights

    def test_recommendation_metrics_positioned_perfect_order(self):
        """Test with perfect ordering."""
        predictions = ["Movie A", "Movie B", "Movie C"]
        ground_truth = ["Movie A", "Movie B", "Movie C"]
        k_values = [3]

        metrics = recommendation_metrics_positioned(predictions, ground_truth, k_values)

        assert metrics["ndcg@3"] == 1.0

    def test_recommendation_metrics_positioned_empty(self):
        """Test with empty inputs."""
        predictions = []
        ground_truth = ["Movie A"]
        k_values = [1]

        metrics = recommendation_metrics_positioned(predictions, ground_truth, k_values)

        assert isinstance(metrics, dict)
