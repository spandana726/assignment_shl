"""Pydantic models — the non-negotiable API contract."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


# ── Response ─────────────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # K, P, A, B, S, C, D or comma-separated combos


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] | None = None
    end_of_conversation: bool = False


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
