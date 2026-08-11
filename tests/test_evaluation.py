"""Tests for evaluation module (ranking artifact schema helpers and catalog loaders)."""

import numpy as np
import pandas as pd
import pytest

from stability.evaluation import (
    RANKING_ARTIFACT_COLUMNS,
    RANKING_ARTIFACT_DEFAULTS,
    _ensure_json_safe,
    _ensure_list,
    aggregate_accuracy_metrics,
    average_popularity_at_k,
    bootstrap_ci,
    build_title_to_item_id_mapping,
    compute_per_user_accuracy,
    compute_tail_items,
    genre_coverage_at_k,
    genre_entropy_at_k,
    gini_index_at_k,
    item_coverage_at_k,
    load_catalog,
    load_genre_mapping,
    load_popularity,
    normalize_llm_result_row,
    normalize_llm_results,
    normalize_ranking_row,
    parse_candidate_pool_label,
    ranking_rows_to_dataframe,
    shannon_entropy_at_k,
    tail_percentage_at_k,
)
from stability.metrics import recommendation_metrics


class TestEnsureList:
    """Tests for _ensure_list helper."""

    def test_list_unchanged(self):
        """Plain list should remain unchanged."""
        assert _ensure_list([1, 2, 3]) == [1, 2, 3]

    def test_tuple_to_list(self):
        """Tuple should be converted to list."""
        assert _ensure_list((1, 2, 3)) == [1, 2, 3]

    def test_numpy_array_to_list(self):
        """NumPy array should be converted to list."""
        arr = np.array([1, 2, 3])
        result = _ensure_list(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_pandas_series_to_list(self):
        """Pandas Series should be converted to list."""
        series = pd.Series(["a", "b", "c"])
        result = _ensure_list(series)
        assert isinstance(result, list)
        assert result == ["a", "b", "c"]

    def test_none_to_empty_list(self):
        """None should become an empty list."""
        assert _ensure_list(None) == []

    def test_nan_to_empty_list(self):
        """NaN float should become an empty list."""
        assert _ensure_list(float("nan")) == []

    def test_scalar_to_singleton_list(self):
        """Scalar value should become a single-element list."""
        assert _ensure_list("hello") == ["hello"]


class TestEnsureJsonSafe:
    """Tests for _ensure_json_safe helper."""

    def test_native_types_unchanged(self):
        """Native Python types should remain unchanged."""
        assert _ensure_json_safe("text") == "text"
        assert _ensure_json_safe(42) == 42
        assert _ensure_json_safe(3.14) == 3.14

    def test_numpy_int_to_int(self):
        """NumPy integer should convert to Python int."""
        result = _ensure_json_safe(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_numpy_float_to_float(self):
        """NumPy float should convert to Python float."""
        result = _ensure_json_safe(np.float64(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-6

    def test_numpy_nan_to_none(self):
        """NumPy NaN should convert to None."""
        assert _ensure_json_safe(np.float64(np.nan)) is None

    def test_float_nan_to_none(self):
        """Float NaN should convert to None."""
        assert _ensure_json_safe(float("nan")) is None


class TestNormalizeRankingRow:
    """Tests for normalize_ranking_row function."""

    def test_full_row(self):
        """Row with all fields should normalize correctly."""
        row = {
            "model_type": "llm",
            "model": "gpt-4",
            "eval_method": "reranking",
            "retriever_type": "cbf",
            "retriever_model": "CBF",
            "reranker_type": "llm",
            "reranker_model": "gpt-4",
            "n_candidates": 250,
            "user_id": "user_1",
            "prompt_idx": 0,
            "ranked_item_ids": [101, 102, 103],
            "ranked_titles": ["A", "B", "C"],
            "ranked_scores": [0.9, 0.8, 0.7],
            "ground_truth_item_ids": [102],
            "ground_truth_titles": ["B"],
        }
        result = normalize_ranking_row(row)
        assert result["model_type"] == "llm"
        assert result["ranked_titles"] == ["A", "B", "C"]
        assert result["ranked_scores"] == [0.9, 0.8, 0.7]
        assert result["ground_truth_item_ids"] == [102]

    def test_missing_optional_fields(self):
        """Missing optional fields should be filled with defaults."""
        row = {
            "model": "test-model",
            "ranked_titles": ["Movie A"],
            "ground_truth_titles": ["Movie B"],
        }
        result = normalize_ranking_row(row)
        assert result["model_type"] is None
        assert result["retriever_type"] is None
        assert result["user_id"] is None
        assert result["ranked_item_ids"] == []
        assert result["ranked_scores"] == []
        assert result["ground_truth_item_ids"] == []

    def test_list_column_type_conversion(self):
        """List columns should be converted to plain Python lists."""
        row = {
            "ranked_item_ids": np.array([1, 2, 3]),
            "ranked_titles": ("A", "B", "C"),
            "ranked_scores": pd.Series([0.1, 0.2, 0.3]),
            "ground_truth_item_ids": np.array([2]),
            "ground_truth_titles": pd.Series(["B"]),
        }
        result = normalize_ranking_row(row)
        assert all(
            isinstance(result[col], list)
            for col in [
                "ranked_item_ids",
                "ranked_titles",
                "ranked_scores",
                "ground_truth_item_ids",
                "ground_truth_titles",
            ]
        )
        assert result["ranked_item_ids"] == [1, 2, 3]
        assert result["ranked_titles"] == ["A", "B", "C"]
        assert result["ranked_scores"] == [0.1, 0.2, 0.3]

    def test_n_candidates_legacy_string(self):
        """Legacy candidate string labels like 'c250' should be parsed."""
        row = {"n_candidates": "c250"}
        result = normalize_ranking_row(row)
        assert result["n_candidates"] == 250

    def test_n_candidates_numeric(self):
        """Numeric n_candidates should remain numeric."""
        row = {"n_candidates": 500}
        result = normalize_ranking_row(row)
        assert result["n_candidates"] == 500

    def test_n_candidates_numpy(self):
        """NumPy numeric n_candidates should convert to native type."""
        row = {"n_candidates": np.int64(100)}
        result = normalize_ranking_row(row)
        assert isinstance(result["n_candidates"], int)
        assert result["n_candidates"] == 100

    def test_empty_row(self):
        """Empty dict should produce all defaults."""
        result = normalize_ranking_row({})
        for col in RANKING_ARTIFACT_COLUMNS:
            assert col in result
            assert result[col] == RANKING_ARTIFACT_DEFAULTS.get(col)
        assert result["ranked_titles"] == []
        assert result["ground_truth_titles"] == []


class TestRankingRowsToDataFrame:
    """Tests for ranking_rows_to_dataframe function."""

    def test_stable_column_order(self):
        """DataFrame columns should follow the stable schema order."""
        rows = [
            {
                "model": "m1",
                "ranked_titles": ["A", "B"],
                "ground_truth_titles": ["B"],
            },
            {
                "model": "m2",
                "ranked_titles": ["C", "D"],
                "ground_truth_titles": ["D"],
            },
        ]
        df = ranking_rows_to_dataframe(rows)
        assert list(df.columns) == RANKING_ARTIFACT_COLUMNS

    def test_parquet_safe_types(self):
        """List columns should be plain Python lists (object dtype)."""
        rows = [
            {
                "ranked_item_ids": np.array([1, 2]),
                "ranked_titles": ("A", "B"),
                "ranked_scores": pd.Series([0.9, 0.8]),
                "ground_truth_item_ids": np.array([2]),
                "ground_truth_titles": ["B"],
            },
        ]
        df = ranking_rows_to_dataframe(rows)
        assert df["ranked_item_ids"].iloc[0] == [1, 2]
        assert df["ranked_titles"].iloc[0] == ["A", "B"]
        assert df["ranked_scores"].iloc[0] == [0.9, 0.8]
        assert isinstance(df["ranked_item_ids"].iloc[0], list)

    def test_empty_rows_raises(self):
        """Empty rows list should raise ValueError."""
        with pytest.raises(ValueError, match="rows must not be empty"):
            ranking_rows_to_dataframe([])

    def test_multiple_rows(self):
        """Multiple rows should create a DataFrame with correct shape."""
        rows = [
            {
                "model_type": "cf",
                "model": "EASE",
                "user_id": i,
                "ranked_item_ids": [i * 10 + 1, i * 10 + 2],
                "ground_truth_item_ids": [i * 10 + 1],
            }
            for i in range(5)
        ]
        df = ranking_rows_to_dataframe(rows)
        assert len(df) == 5
        assert list(df.columns) == RANKING_ARTIFACT_COLUMNS
        assert df["model"].tolist() == ["EASE"] * 5

    def test_preserves_extra_metric_columns(self):
        """Per-user metric columns should survive artifact normalization."""
        rows = [
            {
                "model": "EASE",
                "ranked_item_ids": [1, 2],
                "ground_truth_item_ids": [1],
                "ndcg@10": np.float64(1.0),
                "hit_rate@10": 1,
            },
        ]
        df = ranking_rows_to_dataframe(rows)
        assert list(df.columns) == [
            *RANKING_ARTIFACT_COLUMNS,
            "ndcg@10",
            "hit_rate@10",
        ]
        assert df["ndcg@10"].iloc[0] == 1.0
        assert df["hit_rate@10"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Catalog and popularity loader tests
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    """Tests for load_catalog function."""

    def test_load_catalog_basic(self, tmp_path):
        """Catalog file should load correctly with expected columns."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n"
            "101\tMovie A (2000)\n"
            "102\tMovie B (2001)\n"
            "103\tMovie C (2002)\n",
        )
        id_to_title, item_ids = load_catalog(item_file)
        assert item_ids == [101, 102, 103]
        assert id_to_title == {
            101: "Movie A (2000)",
            102: "Movie B (2001)",
            103: "Movie C (2002)",
        }

    def test_load_catalog_missing_file(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "nonexistent.item")

    def test_load_catalog_missing_columns(self, tmp_path):
        """File without required columns should raise ValueError."""
        bad_file = tmp_path / "bad.item"
        bad_file.write_text("foo\tbar\n1\t2\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_catalog(bad_file)


class TestLoadPopularity:
    """Tests for load_popularity function."""

    def test_load_popularity_basic(self, tmp_path):
        """Popularity counts should match interaction frequencies."""
        inter_file = tmp_path / "test.train.inter"
        inter_file.write_text(
            "user_id:token\titem_id:token\ttimestamp:float\n"
            "0\t101\t0.0\n"
            "0\t102\t1.0\n"
            "1\t101\t0.0\n"
            "1\t103\t1.0\n"
            "2\t102\t0.0\n",
        )
        catalog_ids = [101, 102, 103, 104]
        popularity = load_popularity(inter_file, catalog_ids)
        assert popularity[101] == 2
        assert popularity[102] == 2
        assert popularity[103] == 1
        assert popularity[104] == 0
        assert list(popularity.index) == catalog_ids

    def test_load_popularity_zero_for_missing(self, tmp_path):
        """Catalog items absent from training should have count zero."""
        inter_file = tmp_path / "test.train.inter"
        inter_file.write_text(
            "user_id:token\titem_id:token\ttimestamp:float\n0\t200\t0.0\n",
        )
        catalog_ids = [100, 200, 300]
        popularity = load_popularity(inter_file, catalog_ids)
        assert popularity[100] == 0
        assert popularity[200] == 1
        assert popularity[300] == 0

    def test_load_popularity_missing_file(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_popularity(tmp_path / "nonexistent.inter", [1, 2])

    def test_load_popularity_missing_column(self, tmp_path):
        """File without item_id column should raise ValueError."""
        bad_file = tmp_path / "bad.inter"
        bad_file.write_text("user_id:token\ttimestamp:float\n0\t0.0\n")
        with pytest.raises(ValueError, match="Missing 'item_id:token' column"):
            load_popularity(bad_file, [1])


class TestComputeTailItems:
    """Tests for compute_tail_items function."""

    def test_tail_basic(self):
        """Bottom 10% by popularity should be selected as tail."""
        popularity = pd.Series(
            {1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 6: 50, 7: 40, 8: 30, 9: 20, 10: 10},
        )
        tail = compute_tail_items(popularity, tail_fraction=0.1)
        # 10% of 10 = 1 item
        assert tail == {10}

    def test_tail_multiple_items(self):
        """Tail fraction should round to nearest integer, at least 1."""
        popularity = pd.Series(
            {1: 100, 2: 50, 3: 10},
        )
        tail = compute_tail_items(popularity, tail_fraction=0.34)
        # 0.34 * 3 ≈ 1 → tail size 1
        assert tail == {3}

    def test_tail_empty_catalog(self):
        """Empty catalog should return empty set."""
        popularity = pd.Series(dtype=int)
        tail = compute_tail_items(popularity, tail_fraction=0.1)
        assert tail == set()

    def test_tail_ties(self):
        """Ties should be broken by sort order (stable index ordering)."""
        popularity = pd.Series({1: 5, 2: 5, 3: 5, 4: 1, 5: 1})
        tail = compute_tail_items(popularity, tail_fraction=0.4)
        # 0.4 * 5 = 2 items → IDs 4 and 5 (lowest counts)
        assert tail == {4, 5}

    def test_tail_all_same_count(self):
        """When all counts are equal, tail is just the first N by index order."""
        popularity = pd.Series({10: 5, 20: 5, 30: 5, 40: 5})
        tail = compute_tail_items(popularity, tail_fraction=0.25)
        # 0.25 * 4 = 1 → first item in sorted order (by index)
        assert len(tail) == 1
        assert tail.issubset({10, 20, 30, 40})


class TestLoadGenreMapping:
    """Tests for load_genre_mapping function."""

    def test_load_genre_mapping_basic(self, tmp_path):
        """Genres should map from normalized titles to item IDs."""
        meta_file = tmp_path / "metadata.csv"
        meta_file.write_text(
            "title_norm,tmdb_id,genres\n"
            "Movie A (2000),1,\"['Action']\"\n"
            "Movie B (2001),2,\"['Comedy', 'Drama']\"\n"
            "Movie C (2002),3,\"['Thriller']\"\n",
        )
        catalog = {101: "Movie A (2000)", 102: "Movie B (2001)", 103: "Movie C (2002)"}
        mapping = load_genre_mapping(meta_file, catalog)
        assert mapping[101] == ["Action"]
        assert mapping[102] == ["Comedy", "Drama"]
        assert mapping[103] == ["Thriller"]

    def test_load_genre_mapping_empty_genres(self, tmp_path):
        """Items with empty or missing genres should receive empty lists."""
        meta_file = tmp_path / "metadata.csv"
        meta_file.write_text(
            "title_norm,tmdb_id,genres\n"
            "Movie A (2000),1,\"['Action']\"\n"
            "Movie B (2001),2,\n"
            'Movie C (2002),3,"[]"\n',
        )
        catalog = {101: "Movie A (2000)", 102: "Movie B (2001)", 103: "Movie C (2002)"}
        mapping = load_genre_mapping(meta_file, catalog)
        assert mapping[101] == ["Action"]
        assert mapping[102] == []
        assert mapping[103] == []

    def test_load_genre_mapping_unknown_title(self, tmp_path):
        """Catalog items absent from metadata should receive empty lists."""
        meta_file = tmp_path / "metadata.csv"
        meta_file.write_text(
            "title_norm,tmdb_id,genres\nMovie A (2000),1,\"['Action']\"\n",
        )
        catalog = {101: "Movie A (2000)", 102: "Unknown Movie (1999)"}
        mapping = load_genre_mapping(meta_file, catalog)
        assert mapping[101] == ["Action"]
        assert mapping[102] == []

    def test_load_genre_mapping_missing_file(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_genre_mapping(tmp_path / "nonexistent.csv", {})

    def test_load_genre_mapping_missing_columns(self, tmp_path):
        """File without required columns should raise ValueError."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_genre_mapping(bad_file, {1: "Foo"})


class TestBeyondAccuracyMetrics:
    """Tests for beyond-accuracy metric functions."""

    def test_item_coverage_uniform(self):
        """5 users, each gets a different item from catalog of 10."""
        ranked = [[1], [2], [3], [4], [5]]
        catalog = list(range(1, 11))
        assert item_coverage_at_k(ranked, catalog, 1) == 0.5

    def test_item_coverage_full(self):
        """All catalog items recommended."""
        ranked = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        catalog = list(range(1, 11))
        assert item_coverage_at_k(ranked, catalog, 3) == 1.0

    def test_item_coverage_empty(self):
        """Empty input returns NaN."""
        assert np.isnan(item_coverage_at_k([], [1, 2, 3], 5))

    def test_item_coverage_unknown_excluded(self):
        """Unknown item IDs should not count toward coverage."""
        ranked = [[1, 99], [2, 100]]  # 99, 100 not in catalog
        catalog = [1, 2, 3]
        assert item_coverage_at_k(ranked, catalog, 2) == 2 / 3

    def test_average_popularity_basic(self):
        """Mean popularity of recommended item instances."""
        ranked = [[1, 2], [2, 3]]
        popularity = pd.Series({1: 10, 2: 5, 3: 1})
        result = average_popularity_at_k(ranked, popularity, 2)
        assert result == (10 + 5 + 5 + 1) / 4

    def test_average_popularity_unknown_excluded(self):
        """Items missing from popularity series are excluded."""
        ranked = [[1, 99], [2]]
        popularity = pd.Series({1: 10, 2: 5})
        result = average_popularity_at_k(ranked, popularity, 2)
        assert result == (10 + 5) / 2

    def test_average_popularity_empty(self):
        """Empty input returns NaN."""
        assert np.isnan(average_popularity_at_k([], pd.Series({1: 10}), 5))

    def test_gini_index_uniform(self):
        """Uniform recommendations over catalog -> low Gini."""
        catalog = list(range(1, 11))
        # each user gets a different item
        ranked = [[i] for i in catalog]
        gini = gini_index_at_k(ranked, catalog, 1)
        assert gini < 0.1

    def test_gini_index_concentrated(self):
        """All users get the same item -> high Gini."""
        catalog = list(range(1, 11))
        ranked = [[1] for _ in range(10)]
        gini = gini_index_at_k(ranked, catalog, 1)
        assert gini > 0.8

    def test_gini_index_empty(self):
        """Empty input returns NaN."""
        assert np.isnan(gini_index_at_k([], [1, 2, 3], 5))

    def test_shannon_entropy_uniform(self):
        """Uniform distribution -> high entropy."""
        catalog = list(range(1, 5))
        ranked = [[1], [2], [3], [4]]
        entropy = shannon_entropy_at_k(ranked, catalog, 1)
        expected = -4 * (0.25 * np.log(0.25))
        np.testing.assert_almost_equal(entropy, expected, decimal=5)

    def test_shannon_entropy_concentrated(self):
        """All mass on one item -> zero entropy."""
        catalog = list(range(1, 5))
        ranked = [[1], [1], [1], [1]]
        entropy = shannon_entropy_at_k(ranked, catalog, 1)
        np.testing.assert_almost_equal(entropy, 0.0, decimal=5)

    def test_shannon_entropy_empty(self):
        """Empty input returns NaN."""
        assert np.isnan(shannon_entropy_at_k([], [1, 2, 3], 5))

    def test_tail_percentage_basic(self):
        """Half recommended items are in tail."""
        ranked = [[1, 2], [3, 4]]
        tail = {2, 4}
        result = tail_percentage_at_k(ranked, tail, 2)
        assert result == 0.5

    def test_tail_percentage_all_tail(self):
        """All recommended items are tail items."""
        ranked = [[2], [4]]
        tail = {2, 4}
        result = tail_percentage_at_k(ranked, tail, 1)
        assert result == 1.0

    def test_tail_percentage_empty(self):
        """Empty input returns NaN."""
        assert np.isnan(tail_percentage_at_k([], {1, 2}, 5))

    def test_genre_coverage_basic(self):
        """Coverage of distinct genres."""
        ranked = [[1, 2], [3]]
        genres = {1: ["Action"], 2: ["Comedy"], 3: ["Drama"]}
        result = genre_coverage_at_k(ranked, genres, 2)
        assert result == 1.0  # All 3 genres covered

    def test_genre_coverage_partial(self):
        """Partial coverage."""
        ranked = [[1], [1]]
        genres = {1: ["Action"], 2: ["Comedy"]}
        result = genre_coverage_at_k(ranked, genres, 1)
        assert result == 0.5

    def test_genre_coverage_empty_catalog_genres(self):
        """No catalog genres returns NaN."""
        ranked = [[1]]
        assert np.isnan(genre_coverage_at_k(ranked, {}, 1))

    def test_genre_entropy_basic(self):
        """Entropy over genre occurrences."""
        ranked = [[1, 2], [2, 3]]
        genres = {1: ["Action"], 2: ["Action", "Comedy"], 3: ["Comedy"]}
        # occurrences: Action=3, Comedy=3
        result = genre_entropy_at_k(ranked, genres, 2)
        expected = -2 * (0.5 * np.log(0.5))
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_genre_entropy_empty(self):
        """No genre occurrences returns NaN."""
        ranked = [[1]]
        genres = {1: []}
        assert np.isnan(genre_entropy_at_k(ranked, genres, 1))

    def test_unknown_items_excluded_all_metrics(self):
        """Unknown item IDs are excluded consistently."""
        catalog = [1, 2, 3]
        ranked = [[1, 99], [2, 99]]
        popularity = pd.Series({1: 10, 2: 5, 3: 0})
        tail = {3}
        genres = {1: ["A"], 2: ["B"], 3: ["C"]}

        assert item_coverage_at_k(ranked, catalog, 2) == 2 / 3
        assert average_popularity_at_k(ranked, popularity, 2) == (10 + 5) / 2
        gini = gini_index_at_k(ranked, catalog, 2)
        assert not np.isnan(gini)
        entropy = shannon_entropy_at_k(ranked, catalog, 2)
        assert not np.isnan(entropy)
        assert tail_percentage_at_k(ranked, tail, 2) == 0.0
        assert genre_coverage_at_k(ranked, genres, 2) == 2 / 3
        assert genre_entropy_at_k(ranked, genres, 2) == pytest.approx(
            -2 * (0.5 * np.log(0.5)),
        )


class TestComputePerUserAccuracy:
    """Tests for compute_per_user_accuracy function."""

    def test_basic_per_user_accuracy(self):
        """Per-user metrics should match manual recommendation_metrics output."""
        df = pd.DataFrame(
            {
                "ranked_titles": [
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
                ],
                "ground_truth_titles": [
                    ["B"],
                    ["A", "K"],
                ],
            }
        )
        result = compute_per_user_accuracy(df, k_values=[1, 5, 10])

        # Verify metric columns exist
        for k in [1, 5, 10]:
            for name in ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg"]:
                assert f"{name}@{k}" in result.columns

        # Verify first user manually
        expected1 = recommendation_metrics(
            df["ranked_titles"].iloc[0], ["B"], [1, 5, 10]
        )
        for key, value in expected1.items():
            assert result[key].iloc[0] == pytest.approx(value)

        # Verify second user manually
        expected2 = recommendation_metrics(
            df["ranked_titles"].iloc[1], ["A", "K"], [1, 5, 10]
        )
        for key, value in expected2.items():
            assert result[key].iloc[1] == pytest.approx(value)

    def test_empty_ground_truth(self):
        """Empty ground truth should produce NaN for all metrics."""
        df = pd.DataFrame(
            {
                "ranked_titles": [["A", "B", "C"]],
                "ground_truth_titles": [[]],
            }
        )
        result = compute_per_user_accuracy(df, k_values=[1, 5])
        for k in [1, 5]:
            for name in ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg"]:
                assert np.isnan(result[f"{name}@{k}"].iloc[0])

    def test_preserves_other_columns(self):
        """Original DataFrame columns should be preserved."""
        df = pd.DataFrame(
            {
                "model": ["m1", "m2"],
                "retriever_type": ["cbf", "cf"],
                "ranked_titles": [
                    ["B", "A"],
                    ["C", "D"],
                ],
                "ground_truth_titles": [
                    ["B"],
                    ["C"],
                ],
            }
        )
        result = compute_per_user_accuracy(df, k_values=[1])
        assert list(result["model"]) == ["m1", "m2"]
        assert list(result["retriever_type"]) == ["cbf", "cf"]
        assert result["hit_rate@1"].iloc[0] == 1.0
        assert result["hit_rate@1"].iloc[1] == 1.0


class TestAggregateAccuracyMetrics:
    """Tests for aggregate_accuracy_metrics function."""

    def test_aggregate_matches_manual_mean(self):
        """Aggregate values should match manual means from per-user rows."""
        df = pd.DataFrame(
            {
                "ranked_titles": [
                    ["B", "A"],
                    ["B", "C"],
                    ["D", "C"],
                ],
                "ground_truth_titles": [
                    ["B"],
                    ["B"],
                    ["D"],
                ],
            }
        )
        per_user = compute_per_user_accuracy(df, k_values=[1, 5])
        agg = aggregate_accuracy_metrics(per_user, k_values=[1, 5])

        # Check shape: 2 ks * 6 metrics = 12 rows
        assert len(agg) == 12

        # Verify hit_rate@1: user0=1, user1=1, user2=1 -> mean=1.0
        hit1 = agg[(agg["metric"] == "hit_rate") & (agg["k"] == 1)]["value"].iloc[0]
        assert hit1 == pytest.approx(1.0)

        # Verify precision@1: user0=1, user1=1, user2=1 -> mean=1.0
        prec1 = agg[(agg["metric"] == "precision") & (agg["k"] == 1)]["value"].iloc[0]
        assert prec1 == pytest.approx(1.0)

        # Verify ndcg@5: user0, user1, user2 all have perfect ndcg since first item matches
        ndcg5 = agg[(agg["metric"] == "ndcg") & (agg["k"] == 5)]["value"].iloc[0]
        assert ndcg5 == pytest.approx(1.0)

        # Manually verify each aggregate matches direct mean
        metric_names = ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg"]
        for k in [1, 5]:
            for name in metric_names:
                col = f"{name}@{k}"
                expected = float(per_user[col].mean())
                actual = agg[(agg["metric"] == name) & (agg["k"] == k)]["value"].iloc[0]
                assert actual == pytest.approx(expected)

    def test_with_metadata(self):
        """Metadata columns should propagate to every aggregate row."""
        df = pd.DataFrame(
            {
                "ranked_titles": [["B", "A"]],
                "ground_truth_titles": [["B"]],
            }
        )
        per_user = compute_per_user_accuracy(df, k_values=[1])
        agg = aggregate_accuracy_metrics(
            per_user,
            k_values=[1],
            metadata={"model": "EASE", "retriever_type": "cf"},
        )
        assert all(agg["model"] == "EASE")
        assert all(agg["retriever_type"] == "cf")
        hit1 = agg[(agg["metric"] == "hit_rate") & (agg["k"] == 1)]["value"].iloc[0]
        assert hit1 == pytest.approx(1.0)

    def test_missing_metric_columns(self):
        """Missing metric columns should produce NaN values gracefully."""
        df = pd.DataFrame({"ranked_titles": [["A"]], "ground_truth_titles": [["B"]]})
        # Do NOT compute per-user accuracy first
        agg = aggregate_accuracy_metrics(df, k_values=[1])
        assert all(agg["value"].isna())


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_output_shape_and_keys(self):
        """Result dict should have expected keys and types."""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]})
        result = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=100,
            seed=42,
        )
        assert set(result.keys()) == {
            "metric",
            "lower",
            "upper",
            "n_bootstraps",
            "ci",
            "std",
        }
        assert result["n_bootstraps"] == 100
        assert result["ci"] == 0.95
        assert isinstance(result["metric"], float)
        assert isinstance(result["lower"], float)
        assert isinstance(result["upper"], float)
        assert isinstance(result["std"], float)

    def test_ci_ordering(self):
        """Lower bound should be <= upper bound and metric should be between them."""
        df = pd.DataFrame({"score": np.random.RandomState(0).rand(50)})
        result = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=200,
            seed=7,
        )
        assert result["lower"] <= result["upper"]
        # Metric on original data should usually fall within the CI
        assert result["lower"] <= result["metric"] <= result["upper"]

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        df = pd.DataFrame({"score": [0.1, 0.5, 0.9, 0.3, 0.7]})
        result_a = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=500,
            seed=123,
        )
        result_b = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=500,
            seed=123,
        )
        assert result_a["metric"] == pytest.approx(result_b["metric"])
        assert result_a["lower"] == pytest.approx(result_b["lower"])
        assert result_a["upper"] == pytest.approx(result_b["upper"])
        assert result_a["std"] == pytest.approx(result_b["std"])

    def test_constant_data_zero_width_ci(self):
        """Constant data should produce a zero-width CI."""
        df = pd.DataFrame({"score": [5.0, 5.0, 5.0, 5.0]})
        result = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=200,
            seed=1,
        )
        assert result["metric"] == pytest.approx(5.0)
        assert result["lower"] == pytest.approx(5.0)
        assert result["upper"] == pytest.approx(5.0)
        assert result["std"] == pytest.approx(0.0, abs=1e-9)

    def test_per_user_scalar_metric(self):
        """Bootstrap over per-user scalar metrics (column mean)."""
        df = pd.DataFrame(
            {
                "hit_rate@10": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            }
        )
        result = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["hit_rate@10"].mean()),
            n_bootstraps=500,
            seed=99,
        )
        assert result["metric"] == pytest.approx(0.5)
        assert result["lower"] < result["upper"]
        assert 0.0 <= result["lower"] <= result["upper"] <= 1.0

    def test_system_level_metric(self):
        """Bootstrap over system-level metric recomputed from resampled rows."""
        df = pd.DataFrame(
            {
                "ranked_item_ids": [[1, 2], [2, 3], [3, 4], [4, 1]],
            }
        )
        catalog = [1, 2, 3, 4]

        def _coverage(sample: pd.DataFrame) -> float:
            return item_coverage_at_k(
                sample["ranked_item_ids"].tolist(),
                catalog,
                k=2,
            )

        result = bootstrap_ci(df, metric_fn=_coverage, n_bootstraps=300, seed=55)
        assert 0.0 <= result["metric"] <= 1.0
        assert result["lower"] <= result["upper"]
        assert result["lower"] <= result["metric"] <= result["upper"]

    def test_different_ci_level(self):
        """Changing CI level should adjust interval width."""
        df = pd.DataFrame({"score": np.random.RandomState(1).rand(30)})
        result_95 = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=500,
            ci=0.95,
            seed=10,
        )
        result_80 = bootstrap_ci(
            df,
            metric_fn=lambda s: float(s["score"].mean()),
            n_bootstraps=500,
            ci=0.80,
            seed=10,
        )
        # 80% CI should be narrower or equal to 95% CI
        width_95 = result_95["upper"] - result_95["lower"]
        width_80 = result_80["upper"] - result_80["lower"]
        assert width_80 <= width_95

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should produce NaN or raise; metric_fn handles it."""
        df = pd.DataFrame({"score": []})
        # metric_fn on empty series returns NaN -> bootstrap produces NaN
        result = bootstrap_ci(
            df,
            metric_fn=lambda s: (
                float(s["score"].mean()) if len(s) > 0 else float(np.nan)
            ),
            n_bootstraps=100,
            seed=1,
        )
        assert np.isnan(result["metric"])
        assert np.isnan(result["lower"])
        assert np.isnan(result["upper"])


# ---------------------------------------------------------------------------
# LLM result normalization tests
# ---------------------------------------------------------------------------


class TestBuildTitleToItemIdMapping:
    """Tests for build_title_to_item_id_mapping."""

    def test_basic_mapping(self, tmp_path):
        """Normalized titles should map to correct item IDs."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n"
            "101\tMovie A (2000)\n"
            "102\tMovie B (2001)\n"
            "103\tMovie C (2002)\n",
        )
        mapping = build_title_to_item_id_mapping(item_file)
        assert mapping["Movie A (2000)"] == 101
        assert mapping["Movie B (2001)"] == 102
        assert mapping["Movie C (2002)"] == 103

    def test_whitespace_normalization(self, tmp_path):
        """Extra whitespace in titles should be normalized."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n101\tMovie   A  (2000)\n",
        )
        mapping = build_title_to_item_id_mapping(item_file)
        assert mapping["Movie A (2000)"] == 101

    def test_missing_file(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_title_to_item_id_mapping(tmp_path / "nonexistent.item")

    def test_duplicate_norm_title_warns(self, tmp_path, caplog):
        """Duplicate normalized titles should keep the first item ID."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n"
            "101\tMovie A (2000)\n"
            "102\tMovie A (2000)\n",
        )
        with caplog.at_level("WARNING", logger="stability.evaluation"):
            mapping = build_title_to_item_id_mapping(item_file)
        assert mapping["Movie A (2000)"] == 101
        assert "Duplicate normalized title" in caplog.text


class TestParseCandidatePoolLabel:
    """Tests for parse_candidate_pool_label."""

    def test_legacy_c250(self):
        """Legacy c250 should map to CBF with count 250."""
        assert parse_candidate_pool_label("c250") == ("cbf", "CBF", 250)

    def test_legacy_c0(self):
        """Legacy c0 should map to CBF with count 0."""
        assert parse_candidate_pool_label("c0") == ("cbf", "CBF", 0)

    def test_legacy_c_all(self):
        """Legacy cALL should map to CBF with None count."""
        assert parse_candidate_pool_label("cALL") == ("cbf", "CBF", None)

    def test_named_cf_ease_c250(self):
        """Named cf_EASE_c250 should map to CF EASE with count 250."""
        assert parse_candidate_pool_label("cf_EASE_c250") == ("cf", "EASE", 250)

    def test_named_seq_sasrec_c250(self):
        """Named seq_SASRec_c250 should map to sequential SASRec with count 250."""
        assert parse_candidate_pool_label("seq_SASRec_c250") == (
            "sequential",
            "SASRec",
            250,
        )

    def test_named_cf_with_underscore_model(self):
        """Model names containing underscores should be handled."""
        assert parse_candidate_pool_label("cf_My_Model_c100") == (
            "cf",
            "My_Model",
            100,
        )

    def test_integer_input(self):
        """Integer input should map to CBF."""
        assert parse_candidate_pool_label(500) == ("cbf", "CBF", 500)

    def test_none_input(self):
        """None input should return all Nones."""
        assert parse_candidate_pool_label(None) == (None, None, None)

    def test_unknown_string(self):
        """Unrecognised string should return None for retriever fields."""
        assert parse_candidate_pool_label("foobar") == (None, None, "foobar")

    def test_invalid_legacy_suffix(self):
        """Legacy label with non-numeric suffix should be unrecognised."""
        assert parse_candidate_pool_label("cXYZ") == (None, None, "cXYZ")


class TestNormalizeLlmResultRow:
    """Tests for normalize_llm_result_row."""

    def test_cbf_legacy_label(self):
        """Legacy CBF row should get correct retriever metadata."""
        row = {
            "model": "Llama-3.2-3B",
            "model_type": "local_hf",
            "n_candidates": "c250",
            "prompt_idx": 0,
            "pred_items": ["Movie A (2000)", "Movie B (2001)"],
            "ground_truth": ["Movie A (2000)"],
        }
        result = normalize_llm_result_row(row)
        assert result["retriever_type"] == "cbf"
        assert result["retriever_model"] == "CBF"
        assert result["n_candidates"] == 250
        assert result["reranker_type"] == "llm"
        assert result["reranker_model"] == "Llama-3.2-3B"
        assert result["model_type"] == "local_hf"
        assert result["ranked_titles"] == ["Movie A (2000)", "Movie B (2001)"]
        assert result["ground_truth_titles"] == ["Movie A (2000)"]

    def test_cf_named_label(self):
        """Named CF pool should get correct retriever metadata."""
        row = {
            "model": "Qwen2.5-7B",
            "n_candidates": "cf_EASE_c250",
            "pred_items": ["Movie A (2000)"],
            "ground_truth": [],
        }
        result = normalize_llm_result_row(row)
        assert result["retriever_type"] == "cf"
        assert result["retriever_model"] == "EASE"
        assert result["n_candidates"] == 250
        assert result["reranker_model"] == "Qwen2.5-7B"

    def test_sequential_named_label(self):
        """Named sequential pool should get correct retriever metadata."""
        row = {
            "model": "Gemma-2-9B",
            "n_candidates": "seq_SASRec_c250",
            "pred_items": ["Movie A (2000)"],
            "ground_truth": ["Movie B (2001)"],
        }
        result = normalize_llm_result_row(row)
        assert result["retriever_type"] == "sequential"
        assert result["retriever_model"] == "SASRec"
        assert result["n_candidates"] == 250

    def test_legacy_label_ignores_nan_explicit_retriever_fields(self):
        """NaN metadata columns from mixed parquet concatenation should not mask labels."""
        row = {
            "model": "GPT-4.1",
            "model_type": "api",
            "n_candidates": "c250",
            "retriever_type": np.nan,
            "retriever_model": np.nan,
            "pred_items": ["Movie A (2000)"],
            "ground_truth": ["Movie A (2000)"],
        }
        result = normalize_llm_result_row(row)
        assert result["retriever_type"] == "cbf"
        assert result["retriever_model"] == "CBF"
        assert result["n_candidates"] == 250

    def test_candidate_pool_label_takes_precedence_over_numeric_count(self):
        """Named-pool rows should keep retriever identity after numeric normalization."""
        row = {
            "model": "GPT-4.1",
            "model_type": "api",
            "candidate_pool_label": "cf_EASE_c250",
            "n_candidates": 250,
            "retriever_type": np.nan,
            "retriever_model": np.nan,
            "pred_items": ["Movie A (2000)"],
            "ground_truth": ["Movie A (2000)"],
        }
        result = normalize_llm_result_row(row)
        assert result["retriever_type"] == "cf"
        assert result["retriever_model"] == "EASE"
        assert result["n_candidates"] == 250

    def test_title_to_item_id_mapping(self, tmp_path):
        """Title mapping should produce correct item IDs."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n"
            "101\tMovie A (2000)\n"
            "102\tMovie B (2001)\n",
        )
        mapping = build_title_to_item_id_mapping(item_file)
        row = {
            "model": "Test-LLM",
            "n_candidates": "c250",
            "pred_items": ["Movie A (2000)", "Movie B (2001)"],
            "ground_truth": ["Movie A (2000)"],
        }
        result = normalize_llm_result_row(row, title_to_item_id=mapping)
        assert result["ranked_item_ids"] == [101, 102]
        assert result["ground_truth_item_ids"] == [101]

    def test_unknown_titles_omitted(self, tmp_path):
        """Unknown titles should be omitted from item ID lists."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n101\tMovie A (2000)\n",
        )
        mapping = build_title_to_item_id_mapping(item_file)
        row = {
            "model": "Test-LLM",
            "n_candidates": "c250",
            "pred_items": ["Movie A (2000)", "Unknown Title (1999)"],
            "ground_truth": ["Also Unknown (1998)"],
        }
        result = normalize_llm_result_row(row, title_to_item_id=mapping)
        assert result["ranked_item_ids"] == [101]
        assert result["ground_truth_item_ids"] == []
        # Titles should still be preserved
        assert result["ranked_titles"] == ["Movie A (2000)", "Unknown Title (1999)"]
        assert result["ground_truth_titles"] == ["Also Unknown (1998)"]

    def test_numpy_array_inputs(self, tmp_path):
        """NumPy array pred_items and ground_truth should be handled."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n101\tMovie A (2000)\n",
        )
        mapping = build_title_to_item_id_mapping(item_file)
        row = {
            "model": "Test-LLM",
            "n_candidates": "c250",
            "pred_items": np.array(["Movie A (2000)"]),
            "ground_truth": np.array(["Movie A (2000)"]),
        }
        result = normalize_llm_result_row(row, title_to_item_id=mapping)
        assert result["ranked_item_ids"] == [101]
        assert result["ground_truth_item_ids"] == [101]
        assert isinstance(result["ranked_titles"], list)

    def test_no_mapping_provided(self):
        """When no mapping is provided, item ID lists should be empty."""
        row = {
            "model": "Test-LLM",
            "n_candidates": "c250",
            "pred_items": ["Movie A (2000)"],
            "ground_truth": ["Movie A (2000)"],
        }
        result = normalize_llm_result_row(row)
        assert result["ranked_item_ids"] == []
        assert result["ground_truth_item_ids"] == []

    def test_missing_model_falls_back(self):
        """Missing 'model' should fall back to 'model_id'."""
        row = {
            "model_id": "fallback-model",
            "n_candidates": "c250",
            "pred_items": [],
            "ground_truth": [],
        }
        result = normalize_llm_result_row(row)
        assert result["reranker_model"] == "fallback-model"
        assert result["model"] == "fallback-model"

    def test_schema_compliance(self):
        """Result should contain all schema columns exactly."""
        row = {
            "model": "Test-LLM",
            "n_candidates": "c250",
            "pred_items": [],
            "ground_truth": [],
        }
        result = normalize_llm_result_row(row)
        assert set(result.keys()) == set(RANKING_ARTIFACT_COLUMNS)


class TestNormalizeLlmResults:
    """Tests for normalize_llm_results DataFrame helper."""

    def test_basic_dataframe(self, tmp_path):
        """DataFrame should normalise to shared schema DataFrame."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n"
            "101\tMovie A (2000)\n"
            "102\tMovie B (2001)\n",
        )
        df = pd.DataFrame(
            {
                "model": ["LLM-1", "LLM-1"],
                "n_candidates": ["c250", "cf_EASE_c250"],
                "prompt_idx": [0, 1],
                "pred_items": [
                    ["Movie A (2000)"],
                    ["Movie B (2001)"],
                ],
                "ground_truth": [
                    ["Movie A (2000)"],
                    ["Movie B (2001)"],
                ],
            }
        )
        result = normalize_llm_results(df, item_path=item_file)
        assert list(result.columns) == RANKING_ARTIFACT_COLUMNS
        assert len(result) == 2
        assert result["retriever_type"].iloc[0] == "cbf"
        assert result["retriever_type"].iloc[1] == "cf"
        assert result["retriever_model"].iloc[1] == "EASE"
        assert result["ranked_item_ids"].iloc[0] == [101]
        assert result["ranked_item_ids"].iloc[1] == [102]

    def test_prebuilt_mapping(self, tmp_path):
        """Pre-built title mapping should be used when provided."""
        item_file = tmp_path / "test.item"
        item_file.write_text(
            "item_id:token\ttitle:token_seq\n101\tMovie A (2000)\n",
        )
        mapping = build_title_to_item_id_mapping(item_file)
        df = pd.DataFrame(
            {
                "model": ["LLM-1"],
                "n_candidates": ["c250"],
                "pred_items": [["Movie A (2000)"]],
                "ground_truth": [["Movie A (2000)"]],
            }
        )
        result = normalize_llm_results(df, title_to_item_id=mapping)
        assert result["ranked_item_ids"].iloc[0] == [101]

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise ValueError via ranking_rows_to_dataframe."""
        df = pd.DataFrame(
            {
                "model": pd.Series([], dtype=str),
                "n_candidates": pd.Series([], dtype=str),
                "pred_items": pd.Series([], dtype=object),
                "ground_truth": pd.Series([], dtype=object),
            }
        )
        with pytest.raises(ValueError, match="rows must not be empty"):
            normalize_llm_results(df)
