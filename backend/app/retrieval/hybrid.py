"""Hybrid retrieval — RRF fusion of BM25 + Dense, with metadata filtering."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.catalog.loader import CatalogProduct
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.reranker import rerank
from app.config import get_settings

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[CatalogProduct, float]]],
    k: int = 60,
) -> list[tuple[CatalogProduct, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i) for each list where the item appears.
    """
    scores: dict[str, float] = defaultdict(float)
    products: dict[str, CatalogProduct] = {}

    for result_list in result_lists:
        for rank, (product, _score) in enumerate(result_list):
            key = product.entity_id
            scores[key] += 1.0 / (k + rank + 1)
            products[key] = product

    # Sort by RRF score
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [(products[key], scores[key]) for key in sorted_keys]


def metadata_filter(
    candidates: list[tuple[CatalogProduct, float]],
    job_levels: list[str] | None = None,
    languages: list[str] | None = None,
    max_duration_minutes: int | None = None,
    excluded_names: list[str] | None = None,
) -> list[tuple[CatalogProduct, float]]:
    """Apply metadata filters. Soft filtering — move non-matching to bottom, don't remove."""
    if not any([job_levels, languages, max_duration_minutes, excluded_names]):
        return candidates

    excluded = {n.lower() for n in (excluded_names or [])}
    boosted: list[tuple[CatalogProduct, float]] = []
    rest: list[tuple[CatalogProduct, float]] = []

    for product, score in candidates:
        # Hard exclude: user explicitly rejected
        if product.name.lower() in excluded:
            continue

        boost = 0.0

        # Job level match
        if job_levels:
            if any(jl.lower() in [l.lower() for l in product.job_levels] for jl in job_levels):
                boost += 0.1

        # Language match
        if languages:
            if any(lang.lower() in [l.lower() for l in product.languages] for lang in languages):
                boost += 0.05

        # Duration constraint
        if max_duration_minutes and product.duration:
            try:
                dur_str = product.duration.lower().replace("minutes", "").replace("minute", "").strip()
                dur = int(dur_str)
                if dur <= max_duration_minutes:
                    boost += 0.05
            except (ValueError, TypeError):
                pass  # Untimed or unparseable — no penalty

        if boost > 0:
            boosted.append((product, score + boost))
        else:
            rest.append((product, score))

    # Boosted first, then rest
    result = sorted(boosted, key=lambda x: x[1], reverse=True) + rest
    return result


class HybridRetriever:
    """Hybrid retrieval engine combining BM25 + Dense + Cross-Encoder."""

    def __init__(self, products: list[CatalogProduct]) -> None:
        self.bm25_index = BM25Index(products)
        self.dense_index = DenseIndex(products)
        self.settings = get_settings()
        logger.info("HybridRetriever initialized")

    def retrieve(
        self,
        queries: list[str],
        job_levels: list[str] | None = None,
        languages: list[str] | None = None,
        max_duration_minutes: int | None = None,
        excluded_names: list[str] | None = None,
        use_reranker: bool = True,
    ) -> list[tuple[CatalogProduct, float]]:
        """Run multi-query hybrid retrieval pipeline.

        1. For each query: BM25 + Dense search
        2. RRF fusion across all queries and methods
        3. Metadata filtering
        4. Cross-encoder reranking (optional)
        """
        all_results: list[list[tuple[CatalogProduct, float]]] = []

        for query in queries:
            bm25_results = self.bm25_index.search(query, top_k=self.settings.bm25_top_k)
            dense_results = self.dense_index.search(query, top_k=self.settings.dense_top_k)
            all_results.append(bm25_results)
            all_results.append(dense_results)

        # RRF fusion
        fused = reciprocal_rank_fusion(all_results, k=self.settings.rrf_k)

        # Metadata filter
        filtered = metadata_filter(
            fused,
            job_levels=job_levels,
            languages=languages,
            max_duration_minutes=max_duration_minutes,
            excluded_names=excluded_names,
        )

        # Cross-encoder reranking on top candidates
        if use_reranker and len(filtered) > 0:
            # Use the first (primary) query for reranking
            rerank_query = " ".join(queries[:2])  # Combine primary queries
            top_for_rerank = filtered[:self.settings.rerank_top_k]
            reranked = rerank(rerank_query, top_for_rerank, top_k=self.settings.final_top_k)
            # Append remaining (non-reranked) items
            reranked_ids = {r[0].entity_id for r in reranked}
            remaining = [(p, s) for p, s in filtered[self.settings.rerank_top_k:] if p.entity_id not in reranked_ids]
            return reranked + remaining
        else:
            return filtered[:self.settings.final_top_k]
