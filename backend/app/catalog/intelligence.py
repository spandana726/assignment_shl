"""Catalog Intelligence Layer — pre-computed semantic relationships."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.catalog.loader import CatalogProduct

logger = logging.getLogger(__name__)

# ── Skill taxonomy: maps skills/topics → related search terms ────────────────

SKILL_TAXONOMY: dict[str, list[str]] = {
    # Programming languages → related tech
    "java": ["spring", "core java", "backend", "jvm", "microservice", "rest"],
    "python": ["django", "flask", "data science", "machine learning"],
    "javascript": ["angular", "react", "node", "frontend", "typescript"],
    "dotnet": [".net", "c#", "asp.net", "wcf", "wpf", "mvvm", "mvc"],
    "sql": ["database", "relational", "data", "postgresql", "mysql"],

    # Domains
    "cloud": ["aws", "azure", "docker", "kubernetes", "devops", "ci/cd"],
    "data": ["hadoop", "spark", "kafka", "hbase", "hive", "big data"],
    "security": ["cybersecurity", "hipaa", "network security"],
    "finance": ["accounting", "financial", "banking", "payable", "receivable"],
    "healthcare": ["medical", "hipaa", "health", "nursing"],
    "sales": ["selling", "customer", "commerce", "retail"],
    "manufacturing": ["industrial", "safety", "plant", "operator", "production"],

    # Assessment categories
    "leadership": ["executive", "director", "manager", "strategic", "opq"],
    "personality": ["opq32r", "behavior", "workplace", "personality"],
    "cognitive": ["verify", "reasoning", "aptitude", "numerical", "verbal", "inductive"],
    "simulation": ["call simulation", "phone simulation", "interactive"],
    "contact_center": ["customer service", "call center", "inbound", "outbound"],
}

# ── Common complement patterns (from conversation traces) ────────────────────

COMPLEMENT_PATTERNS: list[dict[str, Any]] = [
    {
        "trigger_keys": ["Knowledge & Skills"],
        "complement_names": ["Occupational Personality Questionnaire OPQ32r"],
        "reason": "Personality alongside domain knowledge (8/10 traces include OPQ32r)",
    },
    {
        "trigger_keys": ["Knowledge & Skills"],
        "complement_names": ["SHL Verify Interactive G+"],
        "reason": "Cognitive aptitude for senior roles (5/10 traces include Verify G+)",
    },
    {
        "trigger_seniority": ["graduate", "entry-level"],
        "complement_names": ["Graduate Scenarios"],
        "reason": "SJT for graduate-level candidates",
    },
    {
        "trigger_industry": ["manufacturing", "industrial", "chemical", "safety"],
        "complement_names": ["Dependability and Safety Instrument (DSI)"],
        "reason": "Safety-critical roles need personality-based safety prediction",
    },
]


class CatalogIntelligence:
    """Pre-computed relationships between SHL catalog products."""

    def __init__(self, products: list[CatalogProduct]) -> None:
        self.products = products
        self._by_category: dict[str, list[CatalogProduct]] = defaultdict(list)
        self._by_job_level: dict[str, list[CatalogProduct]] = defaultdict(list)

        for p in products:
            for key in p.keys:
                self._by_category[key].append(p)
            for level in p.job_levels:
                self._by_job_level[level.lower()].append(p)

        logger.info("CatalogIntelligence built: %d categories, %d job levels",
                     len(self._by_category), len(self._by_job_level))

    def get_complements(
        self,
        current_names: list[str],
        seniority: str | None = None,
        industry: str | None = None,
    ) -> list[dict[str, str]]:
        """Suggest complementary assessments based on what's already selected."""
        current_set = {n.lower() for n in current_names}
        suggestions: list[dict[str, str]] = []

        for pattern in COMPLEMENT_PATTERNS:
            # Check trigger conditions
            if "trigger_keys" in pattern:
                if not any(
                    any(k in p.keys for k in pattern["trigger_keys"])
                    for p in self.products
                    if p.name.lower() in current_set
                ):
                    continue

            if "trigger_seniority" in pattern and seniority:
                if not any(s in seniority.lower() for s in pattern["trigger_seniority"]):
                    continue

            if "trigger_industry" in pattern and industry:
                if not any(i in industry.lower() for i in pattern["trigger_industry"]):
                    continue

            # Don't suggest what's already included
            for comp_name in pattern["complement_names"]:
                if comp_name.lower() not in current_set:
                    suggestions.append({
                        "name": comp_name,
                        "reason": pattern["reason"],
                    })

        return suggestions

    def get_related(self, product_name: str, top_k: int = 5) -> list[CatalogProduct]:
        """Find products in the same categories."""
        target = None
        for p in self.products:
            if p.name.lower() == product_name.lower():
                target = p
                break
        if not target:
            return []

        scored: dict[str, int] = {}
        for key in target.keys:
            for p in self._by_category.get(key, []):
                if p.name != target.name:
                    scored[p.name] = scored.get(p.name, 0) + 1

        sorted_names = sorted(scored, key=scored.get, reverse=True)[:top_k]
        return [p for p in self.products if p.name in sorted_names]

    def expand_skills(self, skills: list[str]) -> list[str]:
        """Expand skill terms using taxonomy for better retrieval."""
        expanded = set(skills)
        for skill in skills:
            skill_lower = skill.lower()
            for key, related in SKILL_TAXONOMY.items():
                if skill_lower in key or key in skill_lower:
                    expanded.update(related)
                elif any(skill_lower in r or r in skill_lower for r in related):
                    expanded.add(key)
                    expanded.update(related)
        return list(expanded)

    def get_category_products(self, category: str) -> list[CatalogProduct]:
        return self._by_category.get(category, [])

    def get_job_level_products(self, level: str) -> list[CatalogProduct]:
        return self._by_job_level.get(level.lower(), [])
