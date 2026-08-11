"""Tests for metadata module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from stability.metadata import (
    build_enriched_text,
    enrich_movie_metadata,
    fetch_tmdb_details,
    match_movie_to_tmdb,
    parse_title_year,
    score_tmdb_match,
    search_tmdb_movie,
)


class TestParseTitleYear:
    """Tests for parse_title_year function."""

    def test_parse_title_year_standard(self):
        """Standard title with year in parentheses."""
        assert parse_title_year("The Matrix (1999)") == ("The Matrix", 1999)

    def test_parse_title_year_no_year(self):
        """Title without year returns None for year."""
        assert parse_title_year("Okja") == ("Okja", None)

    def test_parse_title_year_whitespace(self):
        """Leading/trailing whitespace is handled correctly."""
        assert parse_title_year(" The Matrix (1999) ") == ("The Matrix", 1999)

    def test_parse_title_year_special_chars(self):
        """Title with special characters."""
        assert parse_title_year("Se7en (1995)") == ("Se7en", 1995)

    def test_parse_title_year_no_parentheses(self):
        """Year without parentheses is not extracted."""
        assert parse_title_year("The Matrix 1999") == ("The Matrix 1999", None)


class TestBuildEnrichedText:
    """Tests for build_enriched_text function."""

    def test_build_enriched_text_full(self):
        """All fields provided produces full enriched text."""
        result = build_enriched_text(
            "The Matrix (1999)",
            genres=["Sci-Fi", "Action"],
            director="Lana Wachowski",
            cast=["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        )
        assert result == (
            "The Matrix (1999) | Genres: Sci-Fi, Action | Director: Lana Wachowski"
            " | Cast: Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss"
        )

    def test_build_enriched_text_missing_fields(self):
        """None fields are omitted from output."""
        result = build_enriched_text(
            "The Matrix",
            genres=None,
            director=None,
            cast=["Keanu Reeves"],
        )
        assert result == "The Matrix | Cast: Keanu Reeves"
        assert "Genres" not in result
        assert "Director" not in result

    def test_build_enriched_text_title_only(self):
        """No optional args returns just the title."""
        assert build_enriched_text("The Matrix") == "The Matrix"

    def test_build_enriched_text_title_with_year(self):
        """Title with year is preserved in output."""
        result = build_enriched_text("The Matrix (1999)", genres=["Action"])
        assert result == "The Matrix (1999) | Genres: Action"

    def test_build_enriched_text_empty_strings(self):
        """Empty lists are omitted from output."""
        result = build_enriched_text("The Matrix", genres=[], cast=[])
        assert result == "The Matrix"

    def test_build_enriched_text_unicode(self):
        """Unicode characters are handled correctly."""
        result = build_enriched_text(
            "Amélie",
            genres=["Comedy"],
            director="Jean-Pierre Jeunet",
        )
        assert result == "Amélie | Genres: Comedy | Director: Jean-Pierre Jeunet"


class TestScoreTmdbMatch:
    """Tests for score_tmdb_match function."""

    def test_score_exact_match(self):
        """Same title and year gives high score."""
        candidate = {"title": "The Matrix", "release_date": "1999-03-31"}
        score = score_tmdb_match("The Matrix", 1999, candidate)
        assert score >= 0.9

    def test_score_title_only(self):
        """Matching title with no query year gives score >= 0.6."""
        candidate = {"title": "Okja", "release_date": "2017-06-28"}
        score = score_tmdb_match("Okja", None, candidate)
        assert score >= 0.6

    def test_score_year_mismatch(self):
        """Same title with 3-year difference reduces score."""
        candidate = {"title": "The Matrix", "release_date": "2002-03-31"}
        score = score_tmdb_match("The Matrix", 1999, candidate)
        # 1.0 - 0.1 * 3 = 0.7
        np.testing.assert_almost_equal(score, 0.7, decimal=5)

    def test_score_wrong_movie(self):
        """Completely different title gives score < 0.6."""
        candidate = {"title": "Inception", "release_date": "2010-07-16"}
        score = score_tmdb_match("The Matrix", 1999, candidate)
        assert score < 0.6

    def test_score_partial_title(self):
        """Partially matching title gets low fuzzy score."""
        candidate = {"title": "The Matrix Reloaded", "release_date": "2003-05-15"}
        score = score_tmdb_match("The Matrix", 2003, candidate)
        # Fuzzy ratio for "the matrix" vs "the matrix reloaded" is ~0.69, below 0.85 threshold
        assert score < 0.6

    def test_score_fuzzy_near_match(self):
        """Near-miss title 'Shawshank Redemption' vs 'The Shawshank Redemption' >= 0.6."""
        candidate = {"title": "Shawshank Redemption", "release_date": "1994-09-23"}
        score = score_tmdb_match("The Shawshank Redemption", 1994, candidate)
        # "the shawshank redemption" vs "shawshank redemption" ratio ~0.91, above 0.85
        assert score >= 0.6

    def test_score_fuzzy_too_different(self):
        """Completely different title 'Inception' vs 'The Matrix' scores < 0.6."""
        candidate = {"title": "Inception", "release_date": "2010-07-16"}
        score = score_tmdb_match("The Matrix", 1999, candidate)
        assert score < 0.6


class TestSearchTmdbMovie:
    """Tests for search_tmdb_movie with mocked HTTP."""

    def test_search_returns_results(self, monkeypatch):
        """Successful response returns list of results."""
        mock_result = {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"}
        mock_other = {"id": 999, "title": "Other"}

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [mock_result, mock_other]}

        monkeypatch.setattr(
            "stability.metadata._request_get", lambda *_a, **_kw: MockResponse()
        )
        result = search_tmdb_movie("The Matrix", year=1999, api_key="fake-key")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == mock_result

    def test_search_no_results(self, monkeypatch):
        """Empty results list returns empty list."""

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        monkeypatch.setattr(
            "stability.metadata._request_get", lambda *_a, **_kw: MockResponse()
        )
        result = search_tmdb_movie("Nonexistent Movie", api_key="fake-key")
        assert result == []

    def test_search_max_results_limits_output(self, monkeypatch):
        """max_results parameter limits the number of returned results."""
        mock_results = [
            {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"},
            {"id": 604, "title": "The Matrix Reloaded", "release_date": "2003-05-15"},
            {
                "id": 605,
                "title": "The Matrix Revolutions",
                "release_date": "2003-11-05",
            },
        ]

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": mock_results}

        monkeypatch.setattr(
            "stability.metadata._request_get", lambda *_a, **_kw: MockResponse()
        )
        result = search_tmdb_movie("The Matrix", api_key="fake-key", max_results=1)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_results[0]

    def test_search_retry_on_429(self, monkeypatch):
        """HTTP 429 triggers retry, second call succeeds."""
        call_count = 0
        mock_result = {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"}

        class MockResponse429:
            status_code = 429

            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class MockResponseOK:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [mock_result]}

        def mock_get(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockResponse429()
            return MockResponseOK()

        monkeypatch.setattr("stability.metadata._request_get", mock_get)
        monkeypatch.setattr("stability.metadata.time.sleep", lambda _: None)
        result = search_tmdb_movie("The Matrix", api_key="fake-key", retry_delay=0.0)
        assert result == [mock_result]
        assert call_count == 2


class TestFetchTmdbDetails:
    """Tests for fetch_tmdb_details with mocked HTTP."""

    def test_fetch_extracts_fields(self, monkeypatch):
        """Response with genres, credits, and overview is extracted correctly."""
        mock_data = {
            "genres": [
                {"id": 28, "name": "Action"},
                {"id": 878, "name": "Science Fiction"},
            ],
            "overview": "A computer hacker learns about the true nature of reality.",
            "credits": {
                "cast": [
                    {"name": "Keanu Reeves", "order": 0},
                    {"name": "Laurence Fishburne", "order": 1},
                    {"name": "Carrie-Anne Moss", "order": 2},
                    {"name": "Hugo Weaving", "order": 3},
                ],
                "crew": [
                    {"name": "Lana Wachowski", "job": "Director"},
                    {"name": "Joel Silver", "job": "Producer"},
                ],
            },
        }

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return mock_data

        monkeypatch.setattr(
            "stability.metadata._request_get", lambda *_a, **_kw: MockResponse()
        )
        result = fetch_tmdb_details(603, api_key="fake-key")
        assert result is not None
        assert result["genres"] == ["Action", "Science Fiction"]
        assert result["director"] == "Lana Wachowski"
        assert result["top_cast"] == [
            "Keanu Reeves",
            "Laurence Fishburne",
            "Carrie-Anne Moss",
        ]
        assert "computer hacker" in result["overview"]

    def test_fetch_no_director(self, monkeypatch):
        """Response with no Director in crew returns director=None."""
        mock_data = {
            "genres": [{"id": 18, "name": "Drama"}],
            "overview": "A story.",
            "credits": {
                "cast": [{"name": "Actor One", "order": 0}],
                "crew": [{"name": "Some Producer", "job": "Producer"}],
            },
        }

        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return mock_data

        monkeypatch.setattr(
            "stability.metadata._request_get", lambda *_a, **_kw: MockResponse()
        )
        result = fetch_tmdb_details(12345, api_key="fake-key")
        assert result is not None
        assert result["director"] is None


class TestMatchMovieToTmdb:
    """Tests for match_movie_to_tmdb with mocked search and details."""

    def test_match_first_pass_success(self, monkeypatch):
        """First pass (with year) finds a match and returns enriched dict."""
        search_result = {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"}
        details_result = {
            "genres": ["Action"],
            "director": "Lana Wachowski",
            "top_cast": ["Keanu Reeves"],
            "overview": "A hacker discovers reality.",
        }

        monkeypatch.setattr(
            "stability.metadata.search_tmdb_movie",
            lambda _title, **_kwargs: [search_result],
        )
        monkeypatch.setattr(
            "stability.metadata.fetch_tmdb_details",
            lambda _tmdb_id, api_key=None: details_result,  # noqa: ARG005
        )

        result = match_movie_to_tmdb("The Matrix (1999)", api_key="fake-key")
        assert result is not None
        assert result["tmdb_id"] == 603
        assert result["genres"] == ["Action"]
        assert result["director"] == "Lana Wachowski"
        assert result["match_score"] >= 0.6

    def test_match_second_pass_fallback(self, monkeypatch):
        """First search (with year) returns no match, second (without year) succeeds."""
        search_result = {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"}
        details_result = {
            "genres": ["Action"],
            "director": "Lana Wachowski",
            "top_cast": ["Keanu Reeves"],
            "overview": "A hacker discovers reality.",
        }
        call_count = 0

        def mock_search(_title, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            return [search_result]

        monkeypatch.setattr("stability.metadata.search_tmdb_movie", mock_search)
        monkeypatch.setattr(
            "stability.metadata.fetch_tmdb_details",
            lambda _tmdb_id, api_key=None: details_result,  # noqa: ARG005
        )

        result = match_movie_to_tmdb("The Matrix (1999)", api_key="fake-key")
        assert result is not None
        assert result["tmdb_id"] == 603
        assert call_count == 2

    def test_match_no_results(self, monkeypatch):
        """Both passes return no results, returns None."""
        monkeypatch.setattr(
            "stability.metadata.search_tmdb_movie",
            lambda _title, **_kwargs: [],
        )

        result = match_movie_to_tmdb("Nonexistent Movie (2099)", api_key="fake-key")
        assert result is None

    def test_match_picks_best_from_multiple(self, monkeypatch):
        """When TMDb returns multiple results, picks the best fuzzy match."""
        results = [
            {"id": 100, "title": "Matrix Revolutions", "release_date": "2003-11-05"},
            {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"},
            {"id": 200, "title": "Matrix Reloaded", "release_date": "2003-05-15"},
        ]
        details_result = {
            "genres": ["Action"],
            "director": "Lana Wachowski",
            "top_cast": ["Keanu Reeves"],
            "overview": "A hacker discovers reality.",
        }

        monkeypatch.setattr(
            "stability.metadata.search_tmdb_movie",
            lambda _title, **_kwargs: results,
        )
        monkeypatch.setattr(
            "stability.metadata.fetch_tmdb_details",
            lambda _tmdb_id, api_key=None: details_result,  # noqa: ARG005
        )

        result = match_movie_to_tmdb("The Matrix (1999)", api_key="fake-key")
        assert result is not None
        assert result["tmdb_id"] == 603


class TestEnrichMovieMetadata:
    """Tests for enrich_movie_metadata with mocked API."""

    def test_enrich_mocked_api(self, monkeypatch):
        """Small DataFrame gets enriched with mocked metadata."""
        movies_df = pd.DataFrame(
            {
                "title_norm": ["The Matrix (1999)", "Inception (2010)", "Okja (2017)"],
            }
        )

        def mock_match(title_norm, api_key=None):  # noqa: ARG001
            return {
                "tmdb_id": 123,
                "title": title_norm.split(" (")[0],
                "year": 1999,
                "genres": ["Action"],
                "director": "Director",
                "top_cast": ["Actor"],
                "overview": "Plot.",
                "match_score": 1.0,
            }

        monkeypatch.setattr("stability.metadata.match_movie_to_tmdb", mock_match)
        monkeypatch.setattr("stability.metadata.time.sleep", lambda _: None)

        result = enrich_movie_metadata(movies_df, api_key="fake-key")
        assert "enriched_text" in result.columns
        assert len(result) == 3

        # enriched_text should NOT contain "Plot:" (overview removed from embeddings)
        for text in result["enriched_text"]:
            assert "Plot:" not in text

    def test_enrich_uses_normalized_title(self, monkeypatch):
        """Enriched text uses the normalized title (with year), not TMDb short title."""
        movies_df = pd.DataFrame({"title_norm": ["Headhunter (2009)"]})

        def mock_match(title_norm, api_key=None):  # noqa: ARG001
            return {
                "tmdb_id": 456,
                "title": "Headhunter",  # TMDb title without year
                "year": 2009,
                "genres": ["Thriller"],
                "director": "Dir",
                "top_cast": ["Actor"],
                "overview": "A corporate headhunter story.",
                "match_score": 1.0,
            }

        monkeypatch.setattr("stability.metadata.match_movie_to_tmdb", mock_match)
        monkeypatch.setattr("stability.metadata.time.sleep", lambda _: None)

        result = enrich_movie_metadata(movies_df, api_key="fake-key")
        enriched = result["enriched_text"].iloc[0]
        # Should start with the normalized title including year
        assert enriched.startswith("Headhunter (2009)")
        # Should NOT start with just "Headhunter |"
        assert not enriched.startswith("Headhunter |")

    def test_enrich_incremental_cache(self, monkeypatch):
        """Cached rows are skipped, only new rows trigger API calls."""
        movies_df = pd.DataFrame(
            {
                "title_norm": ["Movie A", "Movie B", "Movie C"],
            }
        )

        # Create a cache with 2 already-enriched rows
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.csv"
            cache_df = pd.DataFrame(
                {
                    "title_norm": ["Movie A", "Movie B"],
                    "tmdb_id": [1, 2],
                    "genres": ["['Action']", "['Drama']"],
                    "director": ["Dir A", "Dir B"],
                    "top_cast": ["['Act A']", "['Act B']"],
                    "overview": ["Plot A", "Plot B"],
                    "match_score": [1.0, 1.0],
                    "enriched_text": [
                        "Movie A | Genres: Action",
                        "Movie B | Genres: Drama",
                    ],
                }
            )
            cache_df.to_csv(cache_path, index=False)

            call_count = 0

            def mock_match(_title_norm, api_key=None):  # noqa: ARG001
                nonlocal call_count
                call_count += 1
                return {
                    "tmdb_id": 3,
                    "title": "Movie C",
                    "year": 2020,
                    "genres": ["Comedy"],
                    "director": "Dir C",
                    "top_cast": ["Act C"],
                    "overview": "Plot C",
                    "match_score": 1.0,
                }

            monkeypatch.setattr("stability.metadata.match_movie_to_tmdb", mock_match)
            monkeypatch.setattr("stability.metadata.time.sleep", lambda _: None)

            result = enrich_movie_metadata(
                movies_df, api_key="fake-key", cache_path=cache_path
            )
            # Only Movie C should trigger API call
            assert call_count == 1
            assert len(result) == 3

    def test_enrich_api_failure(self, monkeypatch):
        """API returning None results in NaN values without raising."""
        movies_df = pd.DataFrame({"title_norm": ["Unknown Movie"]})

        monkeypatch.setattr(
            "stability.metadata.match_movie_to_tmdb",
            lambda _title_norm, api_key=None: None,  # noqa: ARG005
        )
        monkeypatch.setattr("stability.metadata.time.sleep", lambda _: None)

        result = enrich_movie_metadata(movies_df, api_key="fake-key")
        assert len(result) == 1
        assert pd.isna(result["tmdb_id"].iloc[0])
        assert result["enriched_text"].iloc[0] == "Unknown Movie"
