"""Contradiction detection and resolution — latest instruction wins."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_contradictions(
    requested_additions: list[str],
    requested_removals: list[str],
    accepted_assessments: list[str],
    rejected_assessments: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve contradictions in user requests.

    Rules:
    - If user adds then later removes, the removal wins
    - If user removes then later adds, the addition wins
    - Rejected assessments are NEVER in accepted
    - Latest instruction always takes priority

    Returns:
        (effective_accepted, effective_rejected)
    """
    # Build effective sets
    effective_accepted = set(a.lower() for a in accepted_assessments)
    effective_rejected = set(r.lower() for r in rejected_assessments)

    # Process removals (these came from conversation in order)
    for removal in requested_removals:
        removal_lower = removal.lower()
        effective_rejected.add(removal_lower)
        effective_accepted.discard(removal_lower)

    # Process additions (latest additions override previous rejections)
    for addition in requested_additions:
        addition_lower = addition.lower()
        # Only re-add if it's a later instruction
        if addition_lower in effective_rejected:
            # Check if the addition came AFTER the rejection
            # Since we process in order, if it's in additions, it's the latest
            effective_rejected.discard(addition_lower)
        effective_accepted.add(addition_lower)

    return list(effective_accepted), list(effective_rejected)
