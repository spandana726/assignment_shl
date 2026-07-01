"""Application configuration — loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Embeddings & Reranking
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Catalog
    catalog_url: str = (
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Paths
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    cache_dir: Path = Path(__file__).resolve().parent.parent / "data" / "cache"

    # Agent
    max_turns: int = 8
    request_timeout: int = 28  # 28s to stay within 30s eval timeout
    clarification_confidence_threshold: float = 0.7
    max_recommendations: int = 10
    min_recommendations: int = 1

    # Retrieval
    bm25_top_k: int = 30
    dense_top_k: int = 30
    rerank_top_k: int = 15
    final_top_k: int = 10
    rrf_k: int = 60  # RRF constant

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
