"""Pytest configuration and shared fixtures for stability tests."""

import numpy as np
import pandas as pd
import pytest

MOCK_EMBEDDING_DIM = 128


@pytest.fixture
def sample_movie_titles():
    """Sample movie titles for testing."""
    return [
        "The Matrix",
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Pulp Fiction",
        "The Shawshank Redemption",
        "Fight Club",
        "Forrest Gump",
    ]


@pytest.fixture
def sample_embeddings(sample_movie_titles):
    """Sample embeddings for movie titles."""
    # Create deterministic embeddings for testing
    rng = np.random.default_rng(42)
    embeds = rng.standard_normal((len(sample_movie_titles), MOCK_EMBEDDING_DIM))
    # Normalize to unit length
    embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    return embeds.astype(np.float32)


@pytest.fixture
def title_to_idx(sample_movie_titles):
    """Mapping from movie title to index."""
    return {title: idx for idx, title in enumerate(sample_movie_titles)}


@pytest.fixture
def sample_dialogue():
    """Sample ReDial dialogue for testing.

    Classification:
        m1: seen=1, liked=1 → liked
        m3: seen=1, liked=0 → disliked
        m2: seen=0, liked=1, suggested=1 → recommended_accepted
    """
    return {
        "conversationId": 123,
        "messages": [
            {
                "text": "I like action movies like @m1",
                "senderWorkerId": 1,
                "timeOffset": 0,
            },
            {
                "text": "Have you seen @m2?",
                "senderWorkerId": 2,
                "timeOffset": 10,
            },
        ],
        "movieMentions": {
            "m1": "The Matrix",
            "m2": "Inception",
            "m3": "Interstellar",
        },
        "initiatorWorkerId": 1,
        "respondentWorkerId": 2,
        "initiatorQuestions": {
            "m1": {"seen": 1, "liked": 1, "suggested": 0},
            "m3": {"seen": 1, "liked": 0, "suggested": 0},
        },
        "respondentQuestions": {
            "m2": {"suggested": 1, "seen": 0, "liked": 1},
        },
    }


@pytest.fixture
def sample_prompt_messages():
    """Sample ChatML messages for prompt building."""
    return [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation system.\n\n"
                "RULES:\n"
                " - Use the conversation tagged with <CONVERSATION> between the SEEKER (user) and RECOMMENDER (system) as context.\n"
                " - Rank movies by relevance based on the conversation and user preferences."
            ),
        },
        {
            "role": "user",
            "content": (
                "<CONVERSATION>\n"
                "SEEKER: I like action movies.\n"
                "RECOMMENDER: Have you seen The Matrix?\n"
                "</CONVERSATION>\n\n"
                "<CANDIDATES>\n"
                "1. The Matrix\n"
                "2. Inception\n"
                "3. Interstellar\n"
                "</CANDIDATES>"
            ),
        },
    ]


@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing utilities."""
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4.0, 5.0, 6.0],
            "col3": ["a", "b", "c"],
        },
    )
    df.attrs = {"dataset": "test", "version": 1.0, "count": 100}
    return df


@pytest.fixture
def sample_recommendations():
    """Sample recommendations for metric testing."""
    return {
        "predictions": ["The Matrix", "Inception", "The Dark Knight"],
        "ground_truth": ["Inception", "Interstellar", "Pulp Fiction"],
    }


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing without loading actual models."""

    class MockSentenceTransformer:
        embedding_dim = MOCK_EMBEDDING_DIM

        def encode(
            self,
            texts,
            batch_size=32,  # noqa: ARG002
            show_progress_bar=False,  # noqa: ARG002
            normalize_embeddings=True,
        ):
            """Mock encode method that returns deterministic embeddings."""
            n = len(texts) if isinstance(texts, list) else 1
            rng = np.random.default_rng(42)
            embeds = rng.standard_normal((n, MOCK_EMBEDDING_DIM)).astype(np.float32)
            if normalize_embeddings:
                embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
            return embeds

    return MockSentenceTransformer
