"""FAISS dense retrieval over catalog embeddings."""

from __future__ import annotations

import logging
import numpy as np

from app.catalog.loader import CatalogProduct
from app.retrieval.embedder import embed_texts, embed_query

logger = logging.getLogger(__name__)

# Try to import faiss; fall back to brute-force numpy if unavailable
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss-cpu not available, using numpy brute-force search")


class DenseIndex:
    """Dense vector index over catalog product embeddings."""

    def __init__(self, products: list[CatalogProduct]) -> None:
        self.products = products
        texts = [p.enriched_text for p in products]
        logger.info("Embedding %d catalog products...", len(texts))
        self.embeddings = embed_texts(texts)  # (N, dim)
        self.dim = self.embeddings.shape[1]

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine after normalization)
            # L2-normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, self.dim)
        else:
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms
            self.index = None
            logger.info("Numpy index built: %d vectors, dim=%d", len(products), self.dim)

    def search(self, query: str, top_k: int = 30) -> list[tuple[CatalogProduct, float]]:
        """Return top_k products with cosine similarity scores."""
        q_emb = embed_query(query).reshape(1, -1)

        if HAS_FAISS:
            faiss.normalize_L2(q_emb)
            scores, indices = self.index.search(q_emb, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0:
                    results.append((self.products[idx], float(score)))
            return results
        else:
            # Brute force cosine similarity
            q_norm = q_emb / (np.linalg.norm(q_emb) or 1)
            scores = (self.embeddings @ q_norm.T).flatten()
            top_indices = scores.argsort()[::-1][:top_k]
            return [(self.products[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
