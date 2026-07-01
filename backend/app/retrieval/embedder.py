"""Sentence-transformer embedding engine — shared singleton."""

from __future__ import annotations

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(
            "Embedding model loaded (dim=%d)",
            _model.get_sentence_embedding_dimension(),
        )
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of texts → (N, dim) float32 array."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Encode a single query → (dim,) float32 array."""
    model = get_embedding_model()
    emb = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb[0].astype(np.float32)