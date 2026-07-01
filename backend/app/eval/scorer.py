"""Recall@K scorer and evaluation utilities."""

from __future__ import annotations


def recall_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """Recall@K = |relevant ∩ top_k_recommended| / |relevant|."""
    if not relevant:
        return 0.0
    top_k = set(r.lower().strip() for r in recommended[:k])
    relevant_set = set(r.lower().strip() for r in relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mean_recall_at_k(
    results: list[tuple[list[str], list[str]]],
    k: int = 10,
) -> float:
    """Mean Recall@K across multiple traces."""
    if not results:
        return 0.0
    scores = [recall_at_k(rec, rel, k) for rec, rel in results]
    return sum(scores) / len(scores)


def precision_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """Precision@K = |relevant ∩ top_k_recommended| / K."""
    if k == 0:
        return 0.0
    top_k = set(r.lower().strip() for r in recommended[:k])
    relevant_set = set(r.lower().strip() for r in relevant)
    hits = len(top_k & relevant_set)
    return hits / k


def f1_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """F1@K — harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
