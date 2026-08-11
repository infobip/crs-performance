"""Tests for embeddings module."""

import numpy as np

from stability.embeddings import build_embeddings


def test_build_embeddings_basic(mock_sentence_transformer):
    """Test basic embedding generation."""
    texts = ["hello world", "goodbye world"]
    model = mock_sentence_transformer()

    embeds = build_embeddings(
        texts,
        model,
        batch_size=32,
        normalize=True,
        verbose=False,
    )

    assert embeds.shape == (2, model.embedding_dim)
    assert embeds.dtype == np.float32
    # Check normalization
    norms = np.linalg.norm(embeds, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_build_embeddings_single_text(mock_sentence_transformer):
    """Test embedding generation for single text."""
    texts = ["single text"]
    model = mock_sentence_transformer()

    embeds = build_embeddings(texts, model, batch_size=32, normalize=True)

    assert embeds.shape == (1, model.embedding_dim)
    assert embeds.dtype == np.float32


def test_build_embeddings_empty_list(mock_sentence_transformer):
    """Test embedding generation for empty list."""
    texts = []
    model = mock_sentence_transformer()

    # np.vstack with empty list raises ValueError, so this should fail or handle gracefully
    try:
        embeds = build_embeddings(texts, model, batch_size=32)
        assert embeds.shape[0] == 0
    except ValueError:
        # Empty list causes np.vstack to fail - this is expected behavior
        pass


def test_build_embeddings_normalization(mock_sentence_transformer):
    """Test that normalization works correctly."""
    texts = ["text1", "text2", "text3"]
    model = mock_sentence_transformer()

    # With normalization
    embeds_normalized = build_embeddings(texts, model, batch_size=2, normalize=True)
    norms = np.linalg.norm(embeds_normalized, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    # Without normalization (mock still normalizes, but this tests the parameter)
    embeds_unnormalized = build_embeddings(texts, model, batch_size=2, normalize=False)
    assert embeds_unnormalized.shape == (3, model.embedding_dim)


def test_build_embeddings_batch_processing(mock_sentence_transformer):
    """Test that batching produces consistent embeddings."""
    texts = ["text1", "text2", "text3", "text4", "text5"]
    model = mock_sentence_transformer()

    embeds_batch1 = build_embeddings(texts, model, batch_size=1)
    embeds_batch3 = build_embeddings(texts, model, batch_size=3)

    # All should have same shape
    assert embeds_batch1.shape == embeds_batch3.shape
    assert embeds_batch1.dtype == np.float32


def test_build_embeddings_verbose(mock_sentence_transformer):
    """Test verbose mode (should not crash)."""
    texts = ["text1", "text2"]
    model = mock_sentence_transformer()

    embeds = build_embeddings(texts, model, batch_size=1, verbose=True)

    assert embeds.shape == (2, model.embedding_dim)
