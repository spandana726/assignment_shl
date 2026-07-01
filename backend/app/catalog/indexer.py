"""Index builder — creates and caches BM25 + FAISS indices."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from app.catalog.loader import CatalogProduct
from app.config import get_settings

logger = logging.getLogger(__name__)


def save_indices(bm25_index, dense_index, path: Path | None = None) -> None:
    """Persist indices to disk for fast cold-start."""
    settings = get_settings()
    cache_dir = path or settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save BM25
    bm25_path = cache_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
    logger.info("BM25 index saved to %s", bm25_path)

    # Save dense embeddings (numpy array)
    import numpy as np
    dense_path = cache_dir / "dense_embeddings.npy"
    np.save(str(dense_path), dense_index.embeddings)
    logger.info("Dense embeddings saved to %s", dense_path)


def load_cached_embeddings(path: Path | None = None):
    """Load cached dense embeddings if available."""
    import numpy as np
    settings = get_settings()
    cache_dir = path or settings.cache_dir
    dense_path = cache_dir / "dense_embeddings.npy"

    if dense_path.exists():
        embeddings = np.load(str(dense_path))
        logger.info("Loaded cached embeddings: shape=%s", embeddings.shape)
        return embeddings
    return None
