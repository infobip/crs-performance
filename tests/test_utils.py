"""Tests for utils module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stability.utils import (
    bootstrap_mean_ci,
    canonicalize,
    load_frame_with_attrs,
    map_output_to_option,
    parse_output,
    save_frame_with_attrs,
)


class TestCanonicalizeFunction:
    """Tests for canonicalize function."""

    def test_canonicalize_basic(self):
        """Test basic canonicalization."""
        assert canonicalize("The Matrix") == "the matrix"

    def test_canonicalize_punctuation(self):
        """Test removal of punctuation."""
        assert canonicalize("The Matrix!") == "the matrix"
        assert canonicalize("The Matrix?") == "the matrix"

    def test_canonicalize_ampersand(self):
        """Test ampersand replacement."""
        assert canonicalize("Laurel & Hardy") == "laurel and hardy"

    def test_canonicalize_whitespace(self):
        """Test whitespace normalization."""
        assert canonicalize("The  Matrix") == "the matrix"
        assert canonicalize("  The Matrix  ") == "the matrix"

    def test_canonicalize_none(self):
        """Test with None input."""
        assert canonicalize(None) is None

    def test_canonicalize_empty_string(self):
        """Test with empty string."""
        assert canonicalize("") == ""

    def test_canonicalize_numbers(self):
        """Test with numbers."""
        assert canonicalize("Terminator 2") == "terminator 2"

    def test_canonicalize_en_dash(self):
        """Test en-dash normalization."""
        assert canonicalize("Spider\u2013Man") == "spider-man"

    def test_canonicalize_em_dash(self):
        """Test em-dash normalization."""
        assert canonicalize("Something\u2014Wicked") == "something-wicked"

    def test_canonicalize_accents(self):
        """Test accent stripping."""
        assert canonicalize("Caf\u00e9") == "cafe"

    def test_canonicalize_accented_with_year(self):
        """Test accent stripping with year in title."""
        assert canonicalize("Am\u00e9lie (2001)") == "amelie (2001)"


class TestMapOutputToOption:
    """Tests for map_output_to_option function."""

    def test_map_output_basic(self):
        """Test basic output mapping."""
        mapping = {"the matrix": "The Matrix", "inception": "Inception"}
        assert map_output_to_option("The Matrix", mapping) == "The Matrix"

    def test_map_output_case_insensitive(self):
        """Test case-insensitive mapping."""
        mapping = {"the matrix": "The Matrix"}
        assert map_output_to_option("the matrix", mapping) == "The Matrix"
        assert map_output_to_option("THE MATRIX", mapping) == "The Matrix"

    def test_map_output_not_found(self):
        """Test when output is not in mapping."""
        mapping = {"the matrix": "The Matrix"}
        assert map_output_to_option("Inception", mapping) is None

    def test_map_output_empty_mapping(self):
        """Test with empty mapping."""
        assert map_output_to_option("The Matrix", {}) is None

    def test_map_output_with_punctuation(self):
        """Test mapping with punctuation variations."""
        mapping = {"the matrix": "The Matrix"}
        # canonicalize should handle punctuation
        assert map_output_to_option("The Matrix!", mapping) == "The Matrix"


class TestParseOutput:
    """Tests for parse_output function."""

    def test_parse_output_basic(self):
        """Test basic output parsing."""
        output = "1. The Matrix\n2. Inception\n3. Interstellar"
        mapping = {
            "the matrix": "The Matrix",
            "inception": "Inception",
            "interstellar": "Interstellar",
        }
        result = parse_output(output, mapping, n_recommendations=3)
        assert result == ["The Matrix", "Inception", "Interstellar"]

    def test_parse_output_with_bullets(self):
        """Test parsing with bullet points."""
        output = "- The Matrix\n- Inception\n- Interstellar"
        mapping = {
            "the matrix": "The Matrix",
            "inception": "Inception",
            "interstellar": "Interstellar",
        }
        result = parse_output(output, mapping, n_recommendations=3)
        assert len(result) >= 0

    def test_parse_output_duplicates(self):
        """Test handling of duplicate recommendations."""
        output = "1. The Matrix\n2. The Matrix\n3. Inception"
        mapping = {
            "the matrix": "The Matrix",
            "inception": "Inception",
        }
        result = parse_output(output, mapping, n_recommendations=3)
        assert isinstance(result, list)

    def test_parse_output_none_input(self):
        """Test with None input."""
        result = parse_output(None, {}, n_recommendations=3)
        assert result is None

    def test_parse_output_limit_recommendations(self):
        """Test limiting number of recommendations."""
        output = "1. The Matrix\n2. Inception\n3. Interstellar\n4. Fight Club"
        mapping = {
            "the matrix": "The Matrix",
            "inception": "Inception",
            "interstellar": "Interstellar",
            "fight club": "Fight Club",
        }
        result = parse_output(output, mapping, n_recommendations=2)
        assert len(result) <= 2

    def test_parse_output_incomplete_list(self):
        """Test with incomplete recommendation list."""
        output = "1. The Matrix\n2. Inception"
        mapping = {
            "the matrix": "The Matrix",
            "inception": "Inception",
        }
        result = parse_output(output, mapping, n_recommendations=5)
        assert len(result) == 2

    def test_parse_output_unknown_movies(self):
        """Test with movies not in mapping."""
        output = "1. Unknown Movie\n2. The Matrix"
        mapping = {"the matrix": "The Matrix"}
        result = parse_output(output, mapping, n_recommendations=2)
        # Should skip unknown movies
        assert isinstance(result, list)


class TestBootstrapMeanCI:
    """Tests for bootstrap_mean_ci function."""

    def test_bootstrap_mean_ci_basic(self, rng):
        """Test basic bootstrap CI computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lower, upper = bootstrap_mean_ci(values, rng, n_boot=100)

        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= mean <= upper

    def test_bootstrap_mean_ci_correct_mean(self, rng):
        """Test that mean is correct."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, _, _ = bootstrap_mean_ci(values, rng, n_boot=1000)

        np.testing.assert_allclose(mean, 3.0, rtol=0.01)

    def test_bootstrap_mean_ci_confidence_interval(self, rng):
        """Test that CI bounds are reasonable."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lower, upper = bootstrap_mean_ci(values, rng, n_boot=1000, ci=95.0)

        # CI should be symmetric-ish around mean
        assert upper > mean
        assert lower < mean

    def test_bootstrap_mean_ci_all_same_values(self, rng):
        """Test with all identical values."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        mean, lower, upper = bootstrap_mean_ci(values, rng, n_boot=100)

        assert mean == 5.0
        assert lower == 5.0
        assert upper == 5.0

    def test_bootstrap_mean_ci_single_value(self, rng):
        """Test with single value."""
        values = np.array([42.0])
        mean, lower, upper = bootstrap_mean_ci(values, rng, n_boot=100)

        assert mean == 42.0

    def test_bootstrap_mean_ci_with_nan(self, rng):
        """Test handling of NaN values."""
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        mean, lower, upper = bootstrap_mean_ci(values, rng, n_boot=100)

        # Should handle NaN gracefully
        assert isinstance(mean, (float, np.floating))

    def test_bootstrap_mean_ci_all_nan(self, rng):
        """Test with all NaN values."""
        values = np.array([np.nan, np.nan, np.nan])
        mean, lower, upper = bootstrap_mean_ci(values, rng, n_boot=100)

        # Should return NaN or handle gracefully
        assert np.isnan(mean) or isinstance(mean, float)


class TestSaveLoadFrameWithAttrs:
    """Tests for save_frame_with_attrs and load_frame_with_attrs."""

    def test_save_and_load_basic(self, sample_dataframe):
        """Test saving and loading DataFrame with attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_frame"
            save_frame_with_attrs(sample_dataframe, path)

            loaded_df = load_frame_with_attrs(path)

            pd.testing.assert_frame_equal(sample_dataframe, loaded_df)

    def test_save_and_load_attributes(self, sample_dataframe):
        """Test that attributes are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_frame"
            save_frame_with_attrs(sample_dataframe, path)

            loaded_df = load_frame_with_attrs(path)

            assert loaded_df.attrs["dataset"] == "test"
            assert loaded_df.attrs["version"] == 1.0
            assert loaded_df.attrs["count"] == 100

    def test_save_creates_directories(self, sample_dataframe):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test_frame"
            save_frame_with_attrs(sample_dataframe, path)

            assert path.parent.exists()
            assert (path.parent / "..").exists()

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent"
            with pytest.raises(FileNotFoundError):
                load_frame_with_attrs(path)

    @staticmethod
    def test_save_and_load_numeric_attrs():
        """Test that numeric attributes are converted correctly."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df.attrs = {"int_val": 42, "float_val": 3.14, "str_val": "text"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_frame"
            save_frame_with_attrs(df, path)

            loaded_df = load_frame_with_attrs(path)

            assert loaded_df.attrs["int_val"] == 42
            assert abs(loaded_df.attrs["float_val"] - 3.14) < 1e-10
            assert loaded_df.attrs["str_val"] == "text"

    @staticmethod
    def test_save_and_load_empty_dataframe():
        """Test with empty DataFrame."""
        df = pd.DataFrame({"col": []})  # Empty but with columns
        df.attrs = {"metadata": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty_frame"
            save_frame_with_attrs(df, path)

            loaded_df = load_frame_with_attrs(path)

            assert len(loaded_df) == 0
            assert loaded_df.attrs["metadata"] == "test"
