"""BM25 sparse retrieval over catalog enriched text."""

from __future__ import annotations

import logging
import re
from rank_bm25 import BM25Okapi

from app.catalog.loader import CatalogProduct

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9+#.]+", text.lower())


class BM25Index:
    """BM25 sparse index over catalog products."""

    def __init__(self, products: list[CatalogProduct]) -> None:
        self.products = products
        corpus = [_tokenize(p.enriched_text) for p in products]
        self.index = BM25Okapi(corpus)
        logger.info("BM25 index built with %d documents", len(products))

    def search(self, query: str, top_k: int = 30) -> list[tuple[CatalogProduct, float]]:
        """Return top_k products with BM25 scores."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.products[idx], score))
        return results
