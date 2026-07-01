"""Fast catalog lookups â€” by name, URL, ID, fuzzy match."""

from __future__ import annotations

import logging
import re
from typing import Any

from rapidfuzz import fuzz, process

from app.catalog.loader import CatalogProduct

logger = logging.getLogger(__name__)
CATALOG_ALIASES: dict[str, list[str]] = {
    "Occupational Personality Questionnaire OPQ32r": ["opq", "opq32", "opq32r", "personality questionnaire"],
    "SHL Verify Interactive G+": ["verify g+", "verify interactive g+", "g+", "cognitive test", "cognitive ability", "general ability"],
    "Graduate Scenarios": ["graduate scenarios", "graduate sjt", "situational judgement", "situational judgment"],
    "Dependability and Safety Instrument (DSI)": ["dsi", "dependability safety instrument"],
    "Manufac. & Indust. - Safety & Dependability 8.0": ["safety 8.0", "safety and dependability 8.0", "industrial safety", "8.0 bundle"],
    "Contact Center Call Simulation (New)": ["contact center call simulation", "new simulation", "call simulation"],
    "Customer Service Phone Simulation": ["customer service phone simulation", "old simulation", "phone simulation"],
    "SVAR - Spoken English (US) (New)": ["svar us", "spoken english us", "english us"],
    "HIPAA (Security)": ["hipaa", "hipaa security"],
    "Medical Terminology (New)": ["medical terminology"],
    "MS Excel (New)": ["ms excel", "excel knowledge"],
    "MS Word (New)": ["ms word", "word knowledge"],
    "Microsoft Excel 365 (New)": ["excel simulation", "microsoft excel simulation"],
    "Microsoft Word 365 (New)": ["word simulation", "microsoft word simulation"],
    "Core Java (Advanced Level) (New)": ["advanced java", "core java advanced", "java advanced"],
    "Core Java (Entry Level) (New)": ["entry java", "core java entry", "java entry"],
    "Spring (New)": ["spring", "spring boot"],
    "SQL (New)": ["sql", "sql queries"],
    "Amazon Web Services (AWS) Development (New)": ["aws", "amazon web services"],
    "Docker (New)": ["docker"],
    "OPQ Leadership Report": ["leadership report"],
    "OPQ Universal Competency Report 2.0": ["universal competency report", "ucf report"],
    "OPQ MQ Sales Report": ["opq mq sales report", "mq sales report", "sales report"],
}


class CatalogStore:
    """In-memory store for O(1) lookups and fuzzy matching."""

    def __init__(self, products: list[CatalogProduct]) -> None:
        self.products = products
        self._by_name: dict[str, CatalogProduct] = {}
        self._by_url: dict[str, CatalogProduct] = {}
        self._by_id: dict[str, CatalogProduct] = {}
        self._name_list: list[str] = []

        for p in products:
            name_lower = p.name.lower().strip()
            self._by_name[name_lower] = p
            url_norm = p.link.rstrip("/").lower()
            self._by_url[url_norm] = p
            # Also index without trailing slash
            self._by_url[url_norm + "/"] = p
            self._by_id[p.entity_id] = p
            self._name_list.append(p.name)

        self.all_urls: set[str] = set()
        for p in products:
            self.all_urls.add(p.link)
            self.all_urls.add(p.link.rstrip("/"))

        logger.info("CatalogStore initialized with %d products", len(products))

    def __len__(self) -> int:
        return len(self.products)

    def find_by_name(self, name: str) -> CatalogProduct | None:
        return self._by_name.get(name.lower().strip())

    def find_by_url(self, url: str) -> CatalogProduct | None:
        norm = url.rstrip("/").lower()
        return self._by_url.get(norm) or self._by_url.get(norm + "/")

    def find_by_id(self, entity_id: str) -> CatalogProduct | None:
        return self._by_id.get(str(entity_id))

    def exists(self, name: str) -> bool:
        return name.lower().strip() in self._by_name

    def fuzzy_match(self, name: str, threshold: int = 80) -> CatalogProduct | None:
        """Find the closest catalog product by name using fuzzy matching."""
        if not name:
            return None
        result = process.extractOne(
            name,
            self._name_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            matched_name, score, _ = result
            return self.find_by_name(matched_name)
        return None


    def resolve_mention(self, text: str, threshold: int = 82) -> CatalogProduct | None:
        """Resolve a user phrase or alias to a canonical catalog product."""
        if not text:
            return None
        cleaned = re.sub(r"[<>*`_\[\](){}]", " ", text).strip().lower()
        if not cleaned:
            return None

        exact = self.find_by_name(cleaned)
        if exact:
            return exact

        for canonical, aliases in CATALOG_ALIASES.items():
            product = self.find_by_name(canonical)
            if not product:
                continue
            terms = [canonical.lower(), *[a.lower() for a in aliases]]
            if any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", cleaned) for term in terms):
                return product

        return self.fuzzy_match(cleaned, threshold=threshold)

    def products_mentioned_in(self, text: str, threshold: int = 90) -> list[CatalogProduct]:
        """Return catalog products directly named or aliased in free text."""
        if not text:
            return []
        cleaned = text.lower()
        found: list[CatalogProduct] = []
        seen: set[str] = set()

        for p in self.products:
            if p.name.lower() in cleaned and p.entity_id not in seen:
                found.append(p)
                seen.add(p.entity_id)

        for canonical, aliases in CATALOG_ALIASES.items():
            product = self.find_by_name(canonical)
            if not product or product.entity_id in seen:
                continue
            terms = [canonical.lower(), *[a.lower() for a in aliases]]
            if any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", cleaned) for term in terms):
                found.append(product)
                seen.add(product.entity_id)

        if not found:
            candidate = self.fuzzy_match(text, threshold=threshold)
            if candidate:
                found.append(candidate)
        return found
    def get_url_by_name(self, name: str) -> str | None:
        p = self.find_by_name(name) or self.fuzzy_match(name)
        return p.link if p else None

    def get_test_type(self, name: str) -> str | None:
        p = self.find_by_name(name) or self.fuzzy_match(name)
        return p.test_type if p else None

    def get_all_products(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.products]

    def search_by_keys(self, keys: list[str]) -> list[CatalogProduct]:
        """Filter products that have any of the given key categories."""
        return [p for p in self.products if any(k in p.keys for k in keys)]

    def search_by_job_level(self, level: str) -> list[CatalogProduct]:
        """Filter products matching a job level."""
        level_lower = level.lower()
        return [p for p in self.products if any(level_lower in jl.lower() for jl in p.job_levels)]



