"""LangGraph state definition for the SHL agent."""

from __future__ import annotations

from typing import Any, TypedDict

from app.context.state_schema import ConversationState


class AgentState(TypedDict, total=False):
    """State passed through the LangGraph nodes."""

    # ── Input ────────────────────────────────────────────────────────────
    messages: list[dict[str, str]]

    # ── Context engineering ──────────────────────────────────────────────
    conversation_state: ConversationState

    # ── Intent routing ──────────────────────────────────────────────────
    intent: str  # clarify | recommend | refine | compare | refuse | confirm

    # ── Retrieval ────────────────────────────────────────────────────────
    search_queries: list[str]
    retrieved_assessments: list[dict[str, Any]]

    # ── Recommendation ──────────────────────────────────────────────────
    draft_response: dict[str, Any]  # {reply, recommendations, end_of_conversation}

    # ── Verification ────────────────────────────────────────────────────
    verified_response: dict[str, Any]

    # ── Final ────────────────────────────────────────────────────────────
    final_response: dict[str, Any]
