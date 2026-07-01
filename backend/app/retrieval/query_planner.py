"""Multi-Query Planner — generates multiple search queries for better Recall@10."""

from __future__ import annotations

import logging
from typing import Any

from app.catalog.intelligence import SKILL_TAXONOMY

logger = logging.getLogger(__name__)

# ── Role → typical assessment categories mapping ─────────────────────────────

ROLE_ASSESSMENT_MAP: dict[str, list[str]] = {
    "developer": ["programming knowledge test", "coding assessment", "technical skills"],
    "engineer": ["technical knowledge test", "engineering assessment", "problem solving"],
    "manager": ["leadership assessment", "personality questionnaire", "management competency"],
    "sales": ["sales assessment", "personality behavior", "motivation questionnaire"],
    "customer_service": ["customer service simulation", "call center assessment", "spoken language"],
    "analyst": ["numerical reasoning", "data analysis", "financial knowledge"],
    "admin": ["office skills", "microsoft excel word", "data entry"],
    "operator": ["safety assessment", "dependability", "workplace health"],
    "graduate": ["cognitive reasoning", "situational judgement", "graduate scenarios"],
    "executive": ["leadership report", "personality questionnaire", "strategic thinking"],
}

# ── Seniority → job level mapping ────────────────────────────────────────────

SENIORITY_MAP: dict[str, list[str]] = {
    "entry": ["Entry-Level", "Graduate"],
    "junior": ["Entry-Level", "Graduate"],
    "graduate": ["Graduate", "Entry-Level"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "senior": ["Professional Individual Contributor", "Mid-Professional", "Manager"],
    "lead": ["Manager", "Professional Individual Contributor", "Director"],
    "manager": ["Manager", "Front Line Manager", "Director"],
    "director": ["Director", "Executive", "Manager"],
    "executive": ["Executive", "Director"],
    "cxo": ["Executive", "Director"],
    "vp": ["Executive", "Director"],
}

# ── Skill normalization ──────────────────────────────────────────────────────

SKILL_SYNONYMS: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "c#": "csharp",
    "db": "database sql",
    "k8s": "kubernetes",
    "aws": "amazon web services aws",
    "gcp": "google cloud platform",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "ci/cd": "continuous integration deployment devops",
    "react": "react javascript frontend",
    "angular": "angular javascript frontend",
    "node": "nodejs javascript backend",
    "rest": "restful web services api",
    "docker": "docker containerization",
}


def normalize_skill(skill: str) -> str:
    """Normalize skill abbreviations to full terms."""
    return SKILL_SYNONYMS.get(skill.lower(), skill)


def expand_skills_for_search(skills: list[str]) -> list[str]:
    """Expand skill list with normalized forms and taxonomy relations."""
    expanded = set()
    for skill in skills:
        normalized = normalize_skill(skill)
        expanded.add(normalized)
        expanded.add(skill)
        # Check taxonomy
        skill_lower = skill.lower()
        for key, related in SKILL_TAXONOMY.items():
            if skill_lower in key or key in skill_lower:
                expanded.update(related[:3])  # Top 3 related terms
    return list(expanded)


def generate_search_queries(
    role: str | None = None,
    seniority: str | None = None,
    skills: list[str] | None = None,
    industry: str | None = None,
    assessment_goals: list[str] | None = None,
    constraints: list[str] | None = None,
    raw_query: str | None = None,
) -> list[str]:
    """Generate 3-5 complementary search queries covering different facets.

    This is the core of the multi-query retrieval strategy:
    - Q1: Primary skill/role query (direct match)
    - Q2: Domain-adjacent query (expanded skills)
    - Q3: Seniority-appropriate aptitude query
    - Q4: Personality/behavioral query
    - Q5: Industry-specific query (if applicable)
    """
    queries: list[str] = []
    expanded_skills = expand_skills_for_search(skills or [])

    # Q1: Primary role + skills query
    q1_parts = []
    if role:
        q1_parts.append(role)
    if skills:
        q1_parts.extend(skills[:3])
    if assessment_goals:
        q1_parts.extend(assessment_goals[:1])
    if q1_parts:
        queries.append(" ".join(q1_parts))
    elif raw_query:
        queries.append(raw_query)

    # Q2: Expanded skills query
    if expanded_skills:
        skill_terms = [s for s in expanded_skills if s not in (skills or [])][:5]
        if skill_terms:
            q2 = " ".join(skill_terms) + " knowledge test assessment"
            queries.append(q2)

    # Q3: Seniority-appropriate cognitive/aptitude
    if seniority:
        seniority_lower = seniority.lower()
        if any(s in seniority_lower for s in ["senior", "lead", "director", "executive", "cxo", "manager"]):
            queries.append("cognitive reasoning aptitude senior leadership verify")
        elif any(s in seniority_lower for s in ["graduate", "entry", "junior"]):
            queries.append("graduate entry level cognitive reasoning situational judgement")
        else:
            queries.append("professional cognitive aptitude reasoning test")

    # Q4: Personality / behavioral component
    if not any("personality" in q.lower() for q in queries):
        if seniority and any(s in seniority.lower() for s in ["senior", "lead", "director", "executive"]):
            queries.append("personality questionnaire leadership behavior workplace OPQ")
        else:
            queries.append("occupational personality questionnaire behavior assessment")

    # Q5: Industry-specific
    if industry:
        industry_lower = industry.lower()
        industry_query_parts = [industry]
        if "safety" in industry_lower or "manufacturing" in industry_lower or "chemical" in industry_lower:
            industry_query_parts.extend(["safety", "dependability", "industrial"])
        elif "health" in industry_lower or "medical" in industry_lower:
            industry_query_parts.extend(["medical", "health", "hipaa"])
        elif "contact" in industry_lower or "call" in industry_lower or "customer" in industry_lower:
            industry_query_parts.extend(["customer service", "call simulation", "spoken"])
        elif "sales" in industry_lower:
            industry_query_parts.extend(["sales", "transformation", "selling"])
        elif "finance" in industry_lower or "banking" in industry_lower:
            industry_query_parts.extend(["financial", "accounting", "numerical"])
        queries.append(" ".join(industry_query_parts))

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm not in seen:
            seen.add(q_norm)
            unique_queries.append(q)

    logger.info("Generated %d search queries: %s", len(unique_queries), unique_queries)
    return unique_queries[:5]  # Cap at 5 queries


def get_job_level_filters(seniority: str | None) -> list[str] | None:
    """Map seniority string to catalog job level filters."""
    if not seniority:
        return None
    seniority_lower = seniority.lower()
    for key, levels in SENIORITY_MAP.items():
        if key in seniority_lower:
            return levels
    return None
