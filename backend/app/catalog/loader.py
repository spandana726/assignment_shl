"""Fetch, parse, and cache the SHL product catalog."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Mapping from catalog key labels → single-letter test-type codes
KEY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def _derive_test_type(keys: list[str]) -> str:
    """Convert catalog key list to comma-separated type codes, e.g. 'K,S'."""
    codes: list[str] = []
    for k in keys:
        code = KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(sorted(codes)) if codes else "K"


def _build_enriched_text(product: dict[str, Any]) -> str:
    """Create a rich text representation for embedding / BM25 indexing."""
    parts = [
        product.get("name", ""),
        product.get("description", ""),
        f"Keys: {', '.join(product.get('keys', []))}",
        f"Job Levels: {', '.join(product.get('job_levels', []))}",
        f"Languages: {', '.join(product.get('languages', [])[:5])}",
    ]
    duration = product.get("duration", "")
    if duration:
        parts.append(f"Duration: {duration}")
    if product.get("adaptive") == "yes":
        parts.append("Adaptive test")
    if product.get("remote") == "yes":
        parts.append("Remote capable")
    return " | ".join(p for p in parts if p)


class CatalogProduct:
    """Normalised representation of a single SHL product."""

    __slots__ = (
        "entity_id", "name", "link", "description",
        "job_levels", "languages", "duration", "remote",
        "adaptive", "keys", "test_type", "enriched_text",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.entity_id: str = str(raw.get("entity_id", ""))
        self.name: str = raw.get("name", "").strip()
        self.link: str = raw.get("link", "").strip().rstrip("/") + "/"
        # Normalise link — ensure trailing slash
        if not self.link.startswith("https://"):
            self.link = "https://www.shl.com/products/product-catalog/view/" + self.link
        self.description: str = raw.get("description", "").strip()
        self.job_levels: list[str] = raw.get("job_levels", [])
        self.languages: list[str] = raw.get("languages", [])
        self.duration: str = raw.get("duration", "").strip()
        self.remote: bool = raw.get("remote", "no") == "yes"
        self.adaptive: bool = raw.get("adaptive", "no") == "yes"
        self.keys: list[str] = raw.get("keys", [])
        self.test_type: str = _derive_test_type(self.keys)
        self.enriched_text: str = _build_enriched_text(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "url": self.link,
            "test_type": self.test_type,
            "description": self.description,
            "job_levels": self.job_levels,
            "languages": self.languages,
            "duration": self.duration,
            "remote": self.remote,
            "adaptive": self.adaptive,
            "keys": self.keys,
        }


async def fetch_catalog() -> list[CatalogProduct]:
    """Download catalog JSON from SHL endpoint, with local-file fallback."""
    settings = get_settings()
    cache_path = settings.cache_dir / "shl_product_catalog.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw_products: list[dict[str, Any]] = []

    # Try fetching from remote
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(settings.catalog_url)
            resp.raise_for_status()
            content = resp.text
            # Handle control characters in JSON
            raw_products = json.loads(content, strict=False)
            # Cache locally
            cache_path.write_text(content, encoding="utf-8")
            logger.info("Fetched %d products from remote catalog", len(raw_products))
    except Exception as exc:
        logger.warning("Remote fetch failed (%s), trying cache", exc)
        if cache_path.exists():
            raw_products = json.loads(cache_path.read_text(encoding="utf-8"), strict=False)
            logger.info("Loaded %d products from cache", len(raw_products))
        else:
            raise RuntimeError("No catalog available: remote failed and no cache") from exc

    products = [CatalogProduct(p) for p in raw_products if p.get("name")]
    logger.info("Parsed %d valid catalog products", len(products))
    return products
