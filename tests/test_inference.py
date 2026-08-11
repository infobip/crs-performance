"""Tests for inference helper functions (no RecBole dependency).

Tests pure functions from scripts/12_inference.py by reimplementing them
here to avoid importing the full module (which requires torch/recbole).
"""

import numpy as np
import pytest

# --- Pure function copies (no external dependencies) ---


def _deduplicate(items: list) -> list:
    seen: set = set()
    result: list = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def scores_to_ranked_items(
    scores: np.ndarray,
    dataset,
    id_to_title: dict[int, str],
    exclude_ids: set[int],
    top_k: int,
    candidate_ids: set[int] | None = None,
) -> list[tuple[int, str, float]]:
    item_id2token = dataset.field2id_token[dataset.iid_field]
    scored_items = []
    for internal_id, score in enumerate(scores):
        if internal_id >= len(item_id2token):
            break
        token = item_id2token[internal_id]
        if token == "[PAD]":
            continue
        try:
            original_id = int(token)
        except ValueError:
            continue
        if original_id in exclude_ids:
            continue
        if candidate_ids is not None and original_id not in candidate_ids:
            continue
        title = id_to_title.get(original_id)
        if title:
            scored_items.append((original_id, title, float(score)))
    scored_items.sort(key=lambda x: x[2], reverse=True)
    return scored_items[:top_k]


# --- Test fixtures ---


class MockDataset:
    """Minimal mock of RecBole dataset for testing scores_to_ranked_items."""

    iid_field = "item_id"
    field2id_token = {
        "item_id": ["[PAD]", "100", "200", "300", "400", "500"],
    }


@pytest.fixture
def mock_dataset():
    return MockDataset()


@pytest.fixture
def id_to_title():
    return {
        100: "The Matrix (1999)",
        200: "Inception (2010)",
        300: "Interstellar (2014)",
        400: "The Dark Knight (2008)",
        500: "Pulp Fiction (1994)",
    }


@pytest.fixture
def scores():
    # Internal IDs: 0=PAD, 1=100, 2=200, 3=300, 4=400, 5=500
    return np.array([0.0, 0.5, 0.9, 0.3, 0.8, 0.1])


# --- _deduplicate tests ---


class TestDeduplicate:
    def test_removes_duplicates(self):
        assert _deduplicate(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]

    def test_preserves_order(self):
        assert _deduplicate(["C", "A", "B"]) == ["C", "A", "B"]

    def test_empty_list(self):
        assert _deduplicate([]) == []

    def test_no_duplicates(self):
        assert _deduplicate(["A", "B", "C"]) == ["A", "B", "C"]

    def test_all_duplicates(self):
        assert _deduplicate(["A", "A", "A"]) == ["A"]


# --- scores_to_ranked_items tests ---


class TestScoresToRankedItems:
    def test_basic_ranking(self, mock_dataset, id_to_title, scores):
        ranked = scores_to_ranked_items(scores, mock_dataset, id_to_title, set(), 10)
        # Sorted by score descending: 200(0.9), 400(0.8), 100(0.5), 300(0.3), 500(0.1)
        assert len(ranked) == 5
        assert ranked[0] == (200, "Inception (2010)", pytest.approx(0.9))
        assert ranked[1] == (400, "The Dark Knight (2008)", pytest.approx(0.8))
        assert ranked[2] == (100, "The Matrix (1999)", pytest.approx(0.5))

    def test_top_k_limit(self, mock_dataset, id_to_title, scores):
        ranked = scores_to_ranked_items(scores, mock_dataset, id_to_title, set(), 3)
        assert len(ranked) == 3
        assert ranked[0][0] == 200

    def test_exclude_ids(self, mock_dataset, id_to_title, scores):
        ranked = scores_to_ranked_items(
            scores, mock_dataset, id_to_title, {200, 400}, 10
        )
        item_ids = [r[0] for r in ranked]
        assert 200 not in item_ids
        assert 400 not in item_ids
        assert len(ranked) == 3

    def test_candidate_ids_filter(self, mock_dataset, id_to_title, scores):
        ranked = scores_to_ranked_items(
            scores, mock_dataset, id_to_title, set(), 10, candidate_ids={100, 300}
        )
        item_ids = [r[0] for r in ranked]
        assert set(item_ids) == {100, 300}

    def test_exclude_and_candidates_combined(self, mock_dataset, id_to_title, scores):
        ranked = scores_to_ranked_items(
            scores, mock_dataset, id_to_title, {100}, 10, candidate_ids={100, 200, 300}
        )
        item_ids = [r[0] for r in ranked]
        assert 100 not in item_ids
        assert set(item_ids) == {200, 300}

    def test_empty_scores(self, mock_dataset, id_to_title):
        ranked = scores_to_ranked_items(
            np.array([]), mock_dataset, id_to_title, set(), 10
        )
        assert ranked == []

    def test_pad_token_skipped(self, mock_dataset, id_to_title, scores):
        item_ids = [
            r[0]
            for r in scores_to_ranked_items(
                scores, mock_dataset, id_to_title, set(), 10
            )
        ]
        assert all(isinstance(iid, int) for iid in item_ids)
