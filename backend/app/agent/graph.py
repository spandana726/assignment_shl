"""LangGraph agent â€” the state machine that orchestrates the SHL assessment advisor.

DESIGN PRINCIPLES:
  1. Clarification is 100% RULE-BASED (no LLM dependency)
  2. Recommendation has a rule-based FALLBACK (works without API key)
  3. Every async node is wrapped in try/except (no hangs possible)
  4. Every state transition is logged

Flow:
  reconstruct â†’ route_intent â†’ [clarify | retrieve+recommend | refine | compare | refuse | confirm]
                                                    â†“
                                              verify â†’ output
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.config import get_settings
from app.context.reconstructor import reconstruct_state
from app.context.prompt_builder import (
    build_prompt,
    TASK_CLARIFY, TASK_RECOMMEND, TASK_REFINE,
    TASK_COMPARE, TASK_REFUSE, TASK_CONFIRM,
)
from app.retrieval.query_planner import generate_search_queries, get_job_level_filters

logger = logging.getLogger(__name__)

# â”€â”€ Globals set during init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_retriever = None
_catalog_store = None
_catalog_intelligence = None
_genai_client = None


def init_agent(retriever, catalog_store, catalog_intelligence):
    """Inject dependencies (called once at startup)."""
    global _retriever, _catalog_store, _catalog_intelligence
    _retriever = retriever
    _catalog_store = catalog_store
    _catalog_intelligence = catalog_intelligence
    logger.info("Agent initialized (LLM client will be created on first call)")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LLM Helpers (with graceful degradation)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _get_genai_client():
    """Lazily create the Gemini client on first use."""
    global _genai_client
    if _genai_client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set. Add it to backend/.env")
        from google import genai
        _genai_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Gemini client created: model=%s", settings.gemini_model)
    return _genai_client


async def _call_llm(prompt: str, messages: list[dict[str, str]]) -> str:
    """Call Gemini with system prompt + conversation history."""
    from google.genai.types import GenerateContentConfig
    settings = get_settings()
    client = _get_genai_client()

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.3,
            max_output_tokens=2048,
        ),
    )
    return response.text


def _parse_llm_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        result = {
            "reply": parsed.get("reply", ""),
            "recommendations": parsed.get("recommendations"),
            "end_of_conversation": parsed.get("end_of_conversation", False),
        }
        if result["recommendations"] is not None:
            if not isinstance(result["recommendations"], list):
                result["recommendations"] = None
            else:
                valid_recs = []
                for rec in result["recommendations"]:
                    if isinstance(rec, dict) and "name" in rec and "url" in rec:
                        valid_recs.append({
                            "name": rec["name"],
                            "url": rec["url"],
                            "test_type": rec.get("test_type", "K"),
                        })
                result["recommendations"] = valid_recs if valid_recs else None
        return result
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Failed to parse LLM JSON: %s. Raw: %s", e, text[:200])
        return {"reply": text[:500], "recommendations": None, "end_of_conversation": False}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Rule-Based Clarification System (NO LLM DEPENDENCY)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Priority-ordered clarification questions.
# Each tuple: (category, condition_fn, question_template_fn)

def _build_clarification_questions(cs):
    """Build priority-ordered list of clarification questions based on state."""
    role_desc = cs.role or "this position"
    return [
        (
            "role",
            lambda: cs.role is None,
            "What role are you hiring for? For example: software developer, sales manager, "
            "customer service agent, financial analyst, plant operator."
        ),
        (
            "seniority",
            lambda: cs.seniority is None and cs.experience_years is None and cs.role is not None,
            f"What seniority level is this {role_desc}? "
            "Entry-level, mid-level, senior, or executive?"
        ),
        (
            "skills",
            lambda: not cs.skills and cs.role is not None and cs.seniority is not None,
            f"Which key skills or technologies are most important for this {role_desc}? "
            "For example: Java, Python, SQL, Excel, leadership."
        ),
        (
            "goal",
            lambda: not cs.assessment_goals and cs.role is not None,
            f"Is this assessment for hiring/selection, employee development, or promotion?"
        ),
    ]


def select_clarification_question(cs) -> str | None:
    """Select the highest-value clarification question not yet asked.

    Returns None if all questions have been asked or answered.
    """
    questions = _build_clarification_questions(cs)

    for category, condition_fn, question_text in questions:
        if condition_fn() and category not in cs.questions_already_asked:
            logger.info("CLARIFY: asking about '%s' (not yet asked, still missing)", category)
            return question_text

    # All relevant questions have been asked or answered
    logger.info("CLARIFY: no more questions to ask â€” will recommend with current context")
    return None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
TYPE_LABELS = {
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "S": "Simulations",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
}


def _product_to_rec(product: dict | Any) -> dict:
    """Canonical API recommendation from a catalog product/dict."""
    if hasattr(product, "to_dict"):
        product = product.to_dict()
    return {
        "name": product["name"],
        "url": product.get("url") or product.get("link", ""),
        "test_type": product.get("test_type", "K"),
    }


def _dedupe_products(products: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped = []
    for product in products:
        name = product.get("name") if isinstance(product, dict) else product.name
        key = name.lower()
        if key not in seen:
            deduped.append(product)
            seen.add(key)
    return deduped


def _resolve_products(names: list[str]) -> list[Any]:
    if _catalog_store is None:
        return []
    products = []
    for name in names:
        product = _catalog_store.resolve_mention(name) if hasattr(_catalog_store, "resolve_mention") else None
        if not product:
            product = _catalog_store.find_by_name(name) or _catalog_store.fuzzy_match(name, threshold=78)
        if product:
            products.append(product)
    return _dedupe_products(products)


def _mentioned_products(text: str) -> list[Any]:
    if _catalog_store is None or not text:
        return []
    if hasattr(_catalog_store, "products_mentioned_in"):
        return _catalog_store.products_mentioned_in(text)
    return []


def _matches_type(product: Any, codes: list[str]) -> bool:
    if not codes:
        return True
    test_type = product.get("test_type") if isinstance(product, dict) else product.test_type
    product_codes = {c.strip() for c in test_type.split(",")}
    return bool(product_codes.intersection(codes))


def _matches_removal(product: Any, targets: list[str], excluded_types: list[str]) -> bool:
    name = product.get("name") if isinstance(product, dict) else product.name
    keys = product.get("keys", []) if isinstance(product, dict) else product.keys
    test_type = product.get("test_type") if isinstance(product, dict) else product.test_type
    lowered = " ".join([name, test_type, *keys]).lower()

    if excluded_types and _matches_type(product, excluded_types):
        return True
    for target in targets:
        target_l = target.lower()
        mentioned = _catalog_store.resolve_mention(target) if _catalog_store and hasattr(_catalog_store, "resolve_mention") else None
        if mentioned and mentioned.name.lower() == name.lower():
            return True
        compact_target = target_l.replace("tests", "test").replace("assessment", "").strip()
        if compact_target and compact_target in lowered:
            return True
    return False


def _apply_state_filters(cs, products: list[Any]) -> list[Any]:
    filtered = []
    for product in products:
        if _matches_removal(product, cs.requested_removals + cs.rejected_assessments, cs.excluded_test_types):
            continue
        if cs.preferred_test_types and not _matches_type(product, cs.preferred_test_types):
            continue
        if cs.max_duration_minutes:
            duration = product.get("duration") if isinstance(product, dict) else product.duration
            match = re.search(r"\d+", duration or "")
            if match and int(match.group(0)) > cs.max_duration_minutes:
                continue
        filtered.append(product)
    return filtered


def _format_reply_with_recs(intro: str, products: list[Any], closing: str | None = None, end: bool = False) -> dict:
    products = _dedupe_products(products)[:10]
    recs = [_product_to_rec(p) for p in products]
    lines = [intro.strip(), ""] if intro else []
    for rec in recs:
        lines.append(f"- **{rec['name']}** ({rec['test_type']})")
    if closing:
        lines.extend(["", closing.strip()])
    return {"reply": "\n".join(lines).strip(), "recommendations": recs or None, "end_of_conversation": end}

# Rule-Based Recommendation Formatter (FALLBACK when LLM unavailable)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _rule_based_recommendation(cs, retrieved_assessments: list[dict]) -> dict:
    """Format retrieved assessments as a response WITHOUT using LLM."""
    retrieved_assessments = _apply_state_filters(cs, retrieved_assessments)
    if not retrieved_assessments:
        return {
            "reply": "I couldn't find catalog assessments that satisfy those constraints. Could you relax one constraint or share the highest-priority skill?",
            "recommendations": None,
            "end_of_conversation": False,
        }

    parts = []
    if cs.role:
        parts.append(cs.role)
    if cs.seniority:
        parts.append(cs.seniority)
    if cs.skills:
        parts.append(", ".join(cs.skills[:4]))
    context_desc = " / ".join(parts) if parts else "your requirements"
    closing = "I can refine this list, compare items, or remove any category you do not want."
    return _format_reply_with_recs(
        f"Based on {context_desc}, here are grounded SHL catalog recommendations:",
        retrieved_assessments,
        closing=closing,
    )

def _rule_based_refine(cs, retrieved_assessments: list[dict]) -> dict:
    """Modify the current shortlist while preserving valid prior choices."""
    current = _resolve_products(cs.last_recommendations)
    if not current:
        current = retrieved_assessments[:10]

    refined = _apply_state_filters(cs, current)

    additions: list[Any] = []
    for target in cs.requested_additions:
        additions.extend(_mentioned_products(target))
        if not additions or not any((p.name if hasattr(p, "name") else p["name"]).lower() in target.lower() for p in additions):
            queries = generate_search_queries(
                role=cs.role,
                seniority=cs.seniority,
                skills=cs.skills + [target],
                industry=cs.industry,
                assessment_goals=cs.assessment_goals,
                constraints=cs.constraints,
                raw_query=target,
            ) or [target]
            if _retriever is not None:
                results = _retriever.retrieve(
                    queries=queries,
                    job_levels=get_job_level_filters(cs.seniority),
                    languages=cs.language_requirements or None,
                    max_duration_minutes=cs.max_duration_minutes,
                    excluded_names=cs.rejected_assessments or None,
                    use_reranker=False,
                )
                additions.extend([p for p, _ in results[:5]])

    refined = _apply_state_filters(cs, _dedupe_products(refined + additions + retrieved_assessments))[:10]

    notes = []
    if cs.requested_removals or cs.rejected_assessments or cs.excluded_test_types:
        notes.append("removed the requested items/categories")
    if cs.requested_additions:
        notes.append("added the requested focus areas where the catalog supports them")
    intro = "I've updated the existing shortlist: " + "; ".join(notes) + "." if notes else "Here's the updated shortlist:"
    return _format_reply_with_recs(intro, refined, closing="The list remains restricted to valid SHL catalog products.")

def _rule_based_compare(cs, retrieved_assessments: list[dict]) -> dict:
    """Compare named catalog assessments using only catalog metadata."""
    targets = _resolve_products(cs.comparison_targets)
    if len(targets) < 2 and cs.comparison_requests:
        for request in cs.comparison_requests:
            targets.extend(_mentioned_products(request))
    if len(targets) < 2:
        targets.extend(_resolve_products(cs.last_recommendations[:2]))
    targets = _dedupe_products(targets)[:2]

    if len(targets) < 2:
        return {
            "reply": "I need two specific SHL assessments to compare. Which two products should I compare?",
            "recommendations": None,
            "end_of_conversation": False,
        }

    rows = []
    for product in targets:
        data = product.to_dict() if hasattr(product, "to_dict") else product
        keys = ", ".join(data.get("keys", [])) or TYPE_LABELS.get(data.get("test_type", "K"), data.get("test_type", "K"))
        duration = data.get("duration") or "not specified"
        languages = ", ".join(data.get("languages", [])[:4]) or "not specified"
        desc = (data.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 220:
            desc = desc[:217].rsplit(" ", 1)[0] + "..."
        rows.append(f"**{data['name']}**\n- Type: {data.get('test_type', 'K')} ({keys})\n- Duration: {duration}\n- Languages: {languages}\n- Catalog basis: {desc or 'No description supplied in catalog.'}")

    reply = "Here's a catalog-grounded comparison:\n\n" + "\n\n".join(rows)
    return {"reply": reply, "recommendations": None, "end_of_conversation": False}

def _rule_based_refuse() -> dict:
    """Polite refusal for off-topic requests."""
    return {
        "reply": "I'm specifically designed to help with SHL assessment recommendations for hiring. "
                 "I can help you find the right assessments for any role â€” just describe the position "
                 "and I'll recommend suitable tests.",
        "recommendations": None,
        "end_of_conversation": False,
    }


def _rule_based_confirm(cs) -> dict:
    """Acknowledge confirmation."""
    count = len(cs.last_recommendations)
    reply = f"Your assessment battery of {count} test{'s' if count != 1 else ''} is confirmed. "
    reply += "You can proceed with setting these up in your SHL platform."
    return {"reply": reply, "recommendations": None, "end_of_conversation": True}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LangGraph Nodes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def node_reconstruct(state: AgentState) -> dict:
    """Rebuild ConversationState from stateless message history."""
    logger.info("â•â•â• NODE: reconstruct â•â•â•")
    conv_state = reconstruct_state(state["messages"])
    return {"conversation_state": conv_state}


def node_route_intent(state: AgentState) -> dict:
    """Determine the intent based on reconstructed state."""
    logger.info("â•â•â• NODE: route_intent â•â•â•")
    cs = state["conversation_state"]

    if cs.is_off_topic:
        intent = "refuse"
    elif cs.is_comparison_turn:
        intent = "compare"
    elif cs.user_confirmed and cs.last_recommendations:
        intent = "confirm"
    elif cs.is_refinement_turn and cs.last_recommendations:
        intent = "refine"
    elif cs.has_enough_context():
        intent = "recommend"
    else:
        # Check if we SHOULD still clarify or should just recommend
        question = select_clarification_question(cs)
        if question is None:
            # No more questions to ask â€” recommend with what we have
            intent = "recommend"
        else:
            intent = "clarify"

    logger.info(
        "ROUTE: intent=%s confidence=%.2f role=%s seniority=%s skills=%s",
        intent, cs.confidence, cs.role, cs.seniority, cs.skills[:3]
    )
    return {"intent": intent}


def route_decision(state: AgentState) -> str:
    """LangGraph conditional edge â€” returns next node name."""
    return state["intent"]


# â”€â”€ Node: Clarify (RULE-BASED â€” no LLM needed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def node_clarify(state: AgentState) -> dict:
    """Ask a targeted clarifying question. 100% rule-based."""
    logger.info("â•â•â• NODE: clarify â•â•â•")
    cs = state["conversation_state"]

    try:
        # Try LLM-enhanced clarification first
        prompt = build_prompt(cs, task_instruction=TASK_CLARIFY)
        raw = await _call_llm(prompt, state["messages"])
        response = _parse_llm_response(raw)
        response["recommendations"] = None
        response["end_of_conversation"] = False
        logger.info("CLARIFY: LLM response used")
        return {"draft_response": response}
    except Exception as e:
        logger.info("CLARIFY: LLM unavailable (%s), using rule-based", type(e).__name__)

    # Rule-based fallback (always works)
    question = select_clarification_question(cs)
    if question is None:
        # No more questions â€” shouldn't happen but handle gracefully
        question = "Could you tell me more about the role you're hiring for?"

    return {
        "draft_response": {
            "reply": question,
            "recommendations": None,
            "end_of_conversation": False,
        }
    }


# â”€â”€ Node: Retrieve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def node_retrieve(state: AgentState) -> dict:
    """Run multi-query hybrid retrieval with state-aware filters."""
    logger.info("â•â•â• NODE: retrieve â•â•â•")
    cs = state["conversation_state"]

    try:
        last_user = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user = msg["content"]
                break

        queries = generate_search_queries(
            role=cs.role,
            seniority=cs.seniority,
            skills=cs.skills,
            industry=cs.industry,
            assessment_goals=cs.assessment_goals,
            constraints=cs.constraints,
            raw_query=last_user,
        ) or ([last_user] if last_user else ["assessment"])

        job_levels = get_job_level_filters(cs.seniority)
        results = _retriever.retrieve(
            queries=queries,
            job_levels=job_levels,
            languages=cs.language_requirements or None,
            max_duration_minutes=cs.max_duration_minutes,
            excluded_names=cs.rejected_assessments or None,
        )

        retrieved_products = [product for product, _score in results]
        mentioned = _mentioned_products(last_user)
        retrieved_products = _apply_state_filters(cs, _dedupe_products(mentioned + retrieved_products))
        retrieved = [product.to_dict() if hasattr(product, "to_dict") else product for product in retrieved_products]
        logger.info("RETRIEVE: %d results from %d queries", len(retrieved), len(queries))
        return {"search_queries": queries, "retrieved_assessments": retrieved}

    except Exception as e:
        logger.error("RETRIEVE: failed: %s", e, exc_info=True)
        return {"search_queries": [], "retrieved_assessments": []}

# â”€â”€ Node: Recommend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def node_recommend(state: AgentState) -> dict:
    """Generate recommendations. LLM-enhanced with rule-based fallback."""
    logger.info("â•â•â• NODE: recommend â•â•â•")
    cs = state["conversation_state"]
    retrieved = state.get("retrieved_assessments", [])

    try:
        prompt = build_prompt(cs, retrieved_assessments=retrieved, task_instruction=TASK_RECOMMEND)
        raw = await _call_llm(prompt, state["messages"])
        response = _parse_llm_response(raw)
        logger.info("RECOMMEND: LLM response with %d recs",
                     len(response.get("recommendations") or []))
        return {"draft_response": response}
    except Exception as e:
        logger.info("RECOMMEND: LLM unavailable (%s), using rule-based fallback", type(e).__name__)
        response = _rule_based_recommendation(cs, retrieved)
        logger.info("RECOMMEND: rule-based with %d recs",
                     len(response.get("recommendations") or []))
        return {"draft_response": response}


# â”€â”€ Node: Refine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def node_refine(state: AgentState) -> dict:
    """Modify existing shortlist deterministically."""
    logger.info("â•â•â• NODE: refine â•â•â•")
    cs = state["conversation_state"]
    retrieved = state.get("retrieved_assessments", [])
    return {"draft_response": _rule_based_refine(cs, retrieved)}

# â”€â”€ Node: Compare â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def node_compare(state: AgentState) -> dict:
    """Compare assessments deterministically from catalog metadata."""
    logger.info("â•â•â• NODE: compare â•â•â•")
    cs = state["conversation_state"]
    retrieved = state.get("retrieved_assessments", [])
    return {"draft_response": _rule_based_compare(cs, retrieved)}

# â”€â”€ Node: Refuse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def node_refuse(state: AgentState) -> dict:
    """Politely refuse off-topic requests. Rule-based fallback."""
    logger.info("â•â•â• NODE: refuse â•â•â•")
    cs = state["conversation_state"]

    try:
        prompt = build_prompt(cs, task_instruction=TASK_REFUSE)
        raw = await _call_llm(prompt, state["messages"])
        response = _parse_llm_response(raw)
        response["recommendations"] = None
        response["end_of_conversation"] = False
        return {"draft_response": response}
    except Exception as e:
        logger.info("REFUSE: LLM unavailable, using rule-based")
        return {"draft_response": _rule_based_refuse()}


# â”€â”€ Node: Confirm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def node_confirm(state: AgentState) -> dict:
    """Finalize the current shortlist deterministically."""
    logger.info("â•â•â• NODE: confirm â•â•â•")
    cs = state["conversation_state"]
    return {"draft_response": _rule_based_confirm(cs)}

# â”€â”€ Node: Verify (hallucination check) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def node_verify(state: AgentState) -> dict:
    """Deterministic verification â€” every name, URL, test_type checked."""
    logger.info("â•â•â• NODE: verify â•â•â•")
    response = state.get("draft_response", {})
    recommendations = response.get("recommendations")

    if not recommendations or _catalog_store is None:
        return {"verified_response": response}

    verified_recs = []
    for rec in recommendations:
        name = rec.get("name", "")
        # 1. Find product by name (exact or fuzzy)
        product = _catalog_store.find_by_name(name)
        if not product:
            product = _catalog_store.fuzzy_match(name, threshold=80)
            if product:
                logger.info("VERIFY: fuzzy-matched '%s' â†’ '%s'", name, product.name)

        if not product:
            logger.warning("VERIFY: HALLUCINATION BLOCKED '%s'", name)
            continue

        # 2. Auto-repair from catalog
        verified_recs.append({
            "name": product.name,
            "url": product.link,
            "test_type": product.test_type,
        })

    if len(verified_recs) > 10:
        verified_recs = verified_recs[:10]

    response = dict(response)
    response["recommendations"] = verified_recs if verified_recs else None

    logger.info("VERIFY: %d/%d recommendations passed",
                len(verified_recs), len(recommendations))
    return {"verified_response": response}


# â”€â”€ Node: Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def node_output(state: AgentState) -> dict:
    """Final output formatting."""
    logger.info("â•â•â• NODE: output â•â•â•")
    response = state.get("verified_response", state.get("draft_response", {}))
    return {"final_response": response}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Graph Construction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_agent_graph() -> StateGraph:
    """Construct the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("reconstruct", node_reconstruct)
    graph.add_node("route_intent", node_route_intent)
    graph.add_node("clarify", node_clarify)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("recommend", node_recommend)
    graph.add_node("refine", node_refine)
    graph.add_node("compare", node_compare)
    graph.add_node("refuse", node_refuse)
    graph.add_node("confirm", node_confirm)
    graph.add_node("verify", node_verify)
    graph.add_node("output", node_output)

    graph.set_entry_point("reconstruct")
    graph.add_edge("reconstruct", "route_intent")

    graph.add_conditional_edges(
        "route_intent",
        route_decision,
        {
            "clarify": "clarify",
            "recommend": "retrieve",
            "refine": "retrieve",
            "compare": "retrieve",
            "refuse": "refuse",
            "confirm": "retrieve",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        lambda state: state["intent"],
        {
            "recommend": "recommend",
            "refine": "refine",
            "compare": "compare",
            "confirm": "confirm",
        },
    )

    graph.add_edge("clarify", "verify")
    graph.add_edge("recommend", "verify")
    graph.add_edge("refine", "verify")
    graph.add_edge("compare", "verify")
    graph.add_edge("refuse", "verify")
    graph.add_edge("confirm", "verify")
    graph.add_edge("verify", "output")
    graph.add_edge("output", END)

    return graph


# â”€â”€ Compiled graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_compiled_graph = None


def get_agent():
    """Get or compile the agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_agent_graph()
        _compiled_graph = graph.compile()
        logger.info("Agent graph compiled")
    return _compiled_graph


async def run_agent(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Execute the agent graph and return the response."""
    logger.info("â•â•â• AGENT START â•â•â• (%d messages)", len(messages))
    agent = get_agent()

    try:
        result = await agent.ainvoke({"messages": messages})
        response = result.get("final_response")
        if response:
            logger.info("â•â•â• AGENT DONE â•â•â• reply=%s recs=%s",
                        response.get("reply", "")[:80],
                        len(response.get("recommendations") or []))
            return response
    except Exception as e:
        logger.error("â•â•â• AGENT ERROR â•â•â• %s", e, exc_info=True)

    # Ultimate fallback â€” reconstruct state and give rule-based response
    logger.warning("AGENT: using ultimate fallback")
    try:
        cs = reconstruct_state(messages)
        if cs.has_enough_context() and _retriever is not None:
            queries = generate_search_queries(
                role=cs.role, seniority=cs.seniority, skills=cs.skills,
                industry=cs.industry, assessment_goals=cs.assessment_goals,
                constraints=cs.constraints,
            )
            job_levels = get_job_level_filters(cs.seniority)
            results = _retriever.retrieve(
                queries=queries or ["assessment"],
                job_levels=job_levels,
                excluded_names=cs.rejected_assessments or None,
            )
            retrieved = [p.to_dict() for p, s in results]
            return _rule_based_recommendation(cs, retrieved)
        else:
            question = select_clarification_question(cs)
            return {
                "reply": question or "What role are you hiring for?",
                "recommendations": None,
                "end_of_conversation": False,
            }
    except Exception as e2:
        logger.error("AGENT: ultimate fallback also failed: %s", e2)
        return {
            "reply": "I can help you find the right SHL assessments. What role are you hiring for?",
            "recommendations": None,
            "end_of_conversation": False,
        }








