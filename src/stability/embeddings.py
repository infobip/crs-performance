"""Module for creating text embeddings in batches."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import trange

logger = logging.getLogger(__name__)


def build_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int,
    normalize: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Return embeddings for a list of texts in batches.

    Args:
        texts: List of input texts to embed.
        model: SentenceTransformer model to use for embedding.
        batch_size: Number of texts to embed in each batch.
        normalize: Whether to normalize embeddings to unit length.
        verbose: Whether to show a progress bar.

    Returns:
        Array of shape (len(texts), embedding_dim) with embeddings.

    """
    logger.debug(
        f"Building embeddings for {len(texts)} texts "
        f"using model {model.__class__.__name__} in batches of {batch_size}",
    )
    embeds: list[np.ndarray] = []
    for i in trange(0, len(texts), batch_size, desc="Batches"):
        batch = texts[i : i + batch_size]
        batch_embeds = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=verbose,
            normalize_embeddings=normalize,
        )
        embeds.append(batch_embeds)
    logger.debug("Completed building embeddings")
    return np.vstack(embeds).astype(np.float32)
