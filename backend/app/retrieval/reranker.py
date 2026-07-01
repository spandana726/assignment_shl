"""Cross-encoder reranker for top-candidate precision boost."""

from __future__ import annotations

import logging
from typing import Any

from app.catalog.loader import CatalogProduct

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        from app.config import get_settings
        settings = get_settings()
        logger.info("Loading reranker: %s", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("Reranker loaded")
    return _reranker


def rerank(
    query: str,
    candidates: list[tuple[CatalogProduct, float]],
    top_k: int = 10,
) -> list[tuple[CatalogProduct, float]]:
    """Rerank candidates using cross-encoder. Returns top_k with new scores."""
    if not candidates:
        return []

    reranker = get_reranker()

    # Build pairs for cross-encoder
    pairs = [(query, c[0].enriched_text) for c in candidates]

    scores = reranker.predict(pairs)

    # Combine with original products
    scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
