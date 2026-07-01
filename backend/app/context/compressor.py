"""Conversation history compression — manages token budget."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compress_history(
    messages: list[dict[str, str]],
    max_messages: int = 8,
) -> list[dict[str, str]]:
    """Compress conversation history to fit within token budget.

    Strategy:
    - Keep all messages if within limit (SHL caps at 8 turns anyway)
    - Remove filler/confirmation messages ("thanks", "ok", "perfect")
    - Summarize early turns if needed
    """
    if len(messages) <= max_messages:
        return messages

    # For SHL's 8-turn cap, we rarely exceed. But handle gracefully.
    # Keep first user message (establishes context) + last N messages
    compressed = [messages[0]]  # Keep the original query

    # Keep the most recent messages
    remaining_slots = max_messages - 1
    compressed.extend(messages[-remaining_slots:])

    logger.info("Compressed %d messages to %d", len(messages), len(compressed))
    return compressed


def remove_noise(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove low-information messages (greetings, single-word confirmations)."""
    noise_words = {
        "ok", "okay", "thanks", "thank you", "got it", "sure",
        "hi", "hello", "hey", "yes", "no", "fine",
    }

    cleaned = []
    for msg in messages:
        content = msg["content"].strip().lower()
        # Keep assistant messages always
        if msg["role"] == "assistant":
            cleaned.append(msg)
            continue
        # Keep user messages that carry information
        if content not in noise_words and len(content) > 5:
            cleaned.append(msg)
        else:
            # Still keep if it's the only/last user message
            if not any(m["role"] == "user" for m in cleaned):
                cleaned.append(msg)

    return cleaned if cleaned else messages
