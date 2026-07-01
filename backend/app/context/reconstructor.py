"""Stateless Memory Reconstruction â€” rebuild ConversationState from message history.

This is the CORE of context engineering.  Every /chat request calls
reconstruct_state() to rebuild the full conversation context from the
raw message list.  No server-side memory â€” pure extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.context.state_schema import ConversationState

logger = logging.getLogger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Extraction Patterns
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

SENIORITY_PATTERNS = {
    r"\b(cxo|c-suite|executive|chief)\b": "executive",
    r"\b(director|vp|vice president)\b": "director",
    r"\b(manager|management|lead)\b": "manager",
    r"\b(senior|sr\.?|principal|staff)\b": "senior",
    r"\b(mid[- ]?level|mid[- ]?professional|intermediate)\b": "mid-professional",
    r"\b(junior|jr\.?|associate)\b": "junior",
    r"\b(entry[- ]?level|intern|trainee|fresher)\b": "entry-level",
    r"\b(graduate|grad|recent graduate|final[- ]?year)\b": "graduate",
    r"\b(15\+?\s*years|20\+?\s*years)\b": "executive",
    r"\b(10\+?\s*years|12\+?\s*years)\b": "senior",
    r"\b(5\+?\s*years|7\+?\s*years|8\+?\s*years)\b": "senior",
    r"\b(3\+?\s*years|4\+?\s*years)\b": "mid-professional",
    r"\b(1\+?\s*years?|2\+?\s*years?)\b": "junior",
    r"\b(no experience|no work experience|fresh)\b": "entry-level",
}

GOAL_PATTERNS = {
    r"\b(selection|select|hiring|hire|recruit|screen|screening)\b": "selection",
    r"\b(development|develop|re-?skill|upskill|talent audit|coaching)\b": "development",
    r"\b(benchmark|compare|comparing)\b": "benchmarking",
    r"\b(promotion|succession|internal)\b": "promotion",
}

CONFIRMATION_PATTERNS = re.compile(
    r"\b(perfect|confirmed?|that'?s?\s*(good|great|what we need|it|fine)|"
    r"lock\s*(it|ing)\s*in|keep|final|done|approved|agreed|works|looks?\s*good)\b",
    re.IGNORECASE,
)

REJECTION_PATTERNS = re.compile(
    r"\b(drop|remove|exclude|skip|don'?t\s*(include|want|need)|no\s*(need|thanks)|"
    r"without|take\s*out|ditch|cut)\b",
    re.IGNORECASE,
)

ADDITION_PATTERNS = re.compile(
    r"\b(add|include|also\s*(add|include)|throw\s*in|plus|supplement|append|"
    r"we\s*(also|want|need)\s+(a |an )?)\b",
    re.IGNORECASE,
)

COMPARISON_PATTERNS = re.compile(
    r"\b(difference|differ|compare|comparison|vs\.?|versus|between|which\s+(?:is|one)\s+better)\b",
    re.IGNORECASE,
)

OFF_TOPIC_PATTERNS = re.compile(
    r"\b(legal(ly)?|law\b|comply|compliance|regulation|lawsuit|attorney|lawyer|"
    r"weather|recipe|joke|song|poem|story|game|movie|politics|religion)\b",
    re.IGNORECASE,
)

CORRECTION_PATTERNS = re.compile(
    r"\b(actually|instead|change\s+(?:it|that)\s+to|no\s*,?\s*(?:make|switch|change)|wait|correction|scratch\s+that|never\s*mind)\b",
    re.IGNORECASE,
)

REPLACEMENT_PATTERNS = re.compile(
    r"(?:replace|swap|switch|substitute)\s+(.+?)\s+(?:with|for|to)\s+(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)

SKILL_PATTERNS = re.compile(
    r"\b(java|python|javascript|typescript|angular|react|vue|node\.?js|"
    r"c\+\+|c#|\.net|ruby|php|go|golang|rust|swift|kotlin|scala|"
    r"sql|mysql|postgres|mongodb|redis|oracle|"
    r"aws|azure|gcp|docker|kubernetes|k8s|"
    r"spring|spring\s*boot|django|flask|express|laravel|"
    r"html|css|sass|rest\s*api|"
    r"linux|unix|networking|security|hipaa|"
    r"excel|word|powerpoint|outlook|"
    r"accounting|finance|medical|nursing|"
    r"data\s*science|machine\s*learning|ai|"
    r"ci/cd|devops|git|agile|scrum|"
    r"leadership|management|communication|"
    r"customer\s*service|sales|marketing)\b",
    re.IGNORECASE,
)

EXPERIENCE_PATTERN = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)

VOLUME_PATTERN = re.compile(
    r"(\d+)\s*(candidates|people|agents|applicants|hires)|"
    r"(high[- ]?volume|mass|bulk|large[- ]?scale)",
    re.IGNORECASE,
)


DURATION_LIMIT_PATTERN = re.compile(
    r"\b(?:under|less than|within|shorter than|below|max(?:imum)?|no more than)\s*(\d+)\s*(?:min|mins|minutes?)\b",
    re.IGNORECASE,
)

PROMPT_INJECTION_PATTERNS = re.compile(
    r"\b(ignore|forget|bypass|override)\b.*\b(instruction|prompt|system|developer|policy|schema)\b|"
    r"\b(reveal|show|print|dump)\b.*\b(prompt|system message|instructions)\b|"
    r"\bact as\b|\bjailbreak\b|\bdo anything now\b",
    re.IGNORECASE,
)
LANGUAGE_PATTERNS = re.compile(
    r"\b(english|spanish|french|german|chinese|japanese|korean|"
    r"portuguese|dutch|italian|swedish|norwegian|danish|finnish|"
    r"russian|arabic|hindi|thai|indonesian|turkish|polish|czech|"
    r"hungarian|romanian|serbian|slovak|latin\s*american\s*spanish|"
    r"bilingual|multilingual)\b",
    re.IGNORECASE,
)

# â”€â”€ Role keywords for broad detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


TYPE_KEYWORDS = {
    "personality": "P",
    "behavior": "P",
    "behaviour": "P",
    "cognitive": "A",
    "aptitude": "A",
    "reasoning": "A",
    "technical": "K",
    "knowledge": "K",
    "skills": "K",
    "simulation": "S",
    "simulations": "S",
    "situational judgement": "B",
    "situational judgment": "B",
    "sjt": "B",
    "competency": "C",
    "competencies": "C",
    "development": "D",
}
ROLE_KEYWORDS = {
    "developer", "engineer", "programmer", "architect", "coder",
    "analyst", "scientist", "researcher",
    "designer", "admin", "assistant", "secretary", "coordinator",
    "operator", "technician", "worker", "mechanic",
    "nurse", "doctor", "physician", "pharmacist",
    "agent", "representative", "specialist", "advisor", "consultant",
    "manager", "leader", "director", "executive", "supervisor",
    "sales", "recruiter", "trainer", "teacher", "instructor",
    "accountant", "auditor", "clerk",
    "trainee", "graduate", "intern",
}

# â”€â”€ Patterns to detect what the assistant already asked â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

QUESTION_CATEGORIES = {
    "role": [
        r"what\s+role", r"what.*position", r"what.*hiring\s+for",
        r"who\s+are\s+you\s+hiring", r"what.*job", r"what\s+kind\s+of",
        r"tell\s+me.*role", r"describe\s+the\s+role",
    ],
    "seniority": [
        r"what\s+seniority", r"what\s+level", r"entry.level.*senior",
        r"how\s+senior", r"experience\s+level", r"what.*level.*position",
        r"junior.*mid.*senior", r"seniority",
    ],
    "skills": [
        r"what.*skills", r"what.*technologies", r"which.*technologies",
        r"key\s+skills", r"technical\s+requirements", r"tech\s+stack",
        r"important.*skills",
    ],
    "goal": [
        r"selection.*development", r"hiring.*development", r"what.*purpose",
        r"what.*goal", r"is\s+this\s+for", r"screening.*development",
    ],
    "industry": [
        r"what\s+industry", r"which\s+industry", r"what\s+sector",
        r"what.*domain", r"what\s+field",
    ],
    "language": [
        r"what\s+language", r"which\s+language", r"language.*preference",
        r"need.*specific\s+language",
    ],
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Main reconstruction function
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def reconstruct_state(messages: list[dict[str, str]]) -> ConversationState:
    """Reconstruct the full conversation state from stateless message history.

    This is the CORE of context engineering.  Every request rebuilds this.
    No server-side memory.  Pure extraction from conversation.
    """
    state = ConversationState()
    state.turn_count = len(messages)
    state.turns_remaining = max(0, 8 - state.turn_count)

    all_user_text = ""
    all_assistant_text = ""
    last_user_message = ""
    last_assistant_message = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            all_user_text += " " + content
            last_user_message = content
        elif role == "assistant":
            all_assistant_text += " " + content
            last_assistant_message = content

    # â”€â”€ Track which questions the assistant already asked â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    state.questions_already_asked = _detect_asked_questions(messages)
    logger.debug("Questions already asked: %s", state.questions_already_asked)

    # â”€â”€ Extract skills (accumulate from all user messages) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    skill_matches = SKILL_PATTERNS.findall(all_user_text)
    state.skills = list(dict.fromkeys(s.strip() for s in skill_matches))

    # â”€â”€ Extract seniority (LATEST wins for contradictions) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Process per-message in order so later corrections override
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg["content"]
        for pattern, level in SENIORITY_PATTERNS.items():
            if re.search(pattern, content, re.IGNORECASE):
                state.seniority = level  # Latest message wins
                break

    # â”€â”€ Extract experience years (latest wins) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for msg in messages:
        if msg.get("role") != "user":
            continue
        exp_match = EXPERIENCE_PATTERN.search(msg["content"])
        if exp_match:
            state.experience_years = int(exp_match.group(1))

    # â”€â”€ Extract assessment goals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for pattern, goal in GOAL_PATTERNS.items():
        if re.search(pattern, all_user_text, re.IGNORECASE):
            if goal not in state.assessment_goals:
                state.assessment_goals.append(goal)

    # â”€â”€ Extract languages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lang_matches = LANGUAGE_PATTERNS.findall(all_user_text)
    state.language_requirements = list(dict.fromkeys(l.strip() for l in lang_matches))

    # â”€â”€ Extract volume â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vol_match = VOLUME_PATTERN.search(all_user_text)
    if vol_match:
        state.volume = vol_match.group(0)

    # â”€â”€ Detect role (latest correction wins) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # First try latest user message for corrections
    if CORRECTION_PATTERNS.search(last_user_message):
        corrected_role = _extract_role(last_user_message)
        if corrected_role:
            state.role = corrected_role
            logger.info("CORRECTION: role overridden to '%s'", corrected_role)
        else:
            state.role = _extract_role(all_user_text)
    else:
        state.role = _extract_role(all_user_text)

    # â”€â”€ Detect replacements ("replace X with Y") â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for msg in messages:
        if msg.get("role") != "user":
            continue
        rep_match = REPLACEMENT_PATTERNS.search(msg["content"])
        if rep_match:
            old_item = rep_match.group(1).strip().rstrip(".")
            new_item = rep_match.group(2).strip().rstrip(".")
            if old_item and new_item:
                if old_item not in state.rejected_assessments:
                    state.rejected_assessments.append(old_item)
                if new_item not in state.requested_additions:
                    state.requested_additions.append(new_item)
                state.skills = [s for s in state.skills if s.lower() not in old_item.lower()]
                if SKILL_PATTERNS.search(new_item):
                    for skill in SKILL_PATTERNS.findall(new_item):
                        if skill not in state.skills:
                            state.skills.append(skill)
                state.is_refinement_turn = True
                logger.info("REPLACEMENT: '%s' -> '%s'", old_item, new_item)

    # â”€â”€ Detect industry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    industry_patterns = [
        (r"\b(chemical|petrochemical|oil|gas)\b", "chemical/industrial"),
        (r"\b(manufactur|industrial|plant|factory|production)\b", "manufacturing"),
        (r"\b(health|medical|hospital|clinic|pharma|nursing)\b", "healthcare"),
        (r"\b(bank|financ|insurance|invest)\b", "finance/banking"),
        (r"\b(retail|store|shop|commerce)\b", "retail"),
        (r"\b(contact\s*cent|call\s*cent)\b", "contact center"),
        (r"\b(tech|software|startup|saas)\b", "technology"),
        (r"\b(consult|profession|service)\b", "professional services"),
    ]
    for pattern, industry in industry_patterns:
        if re.search(pattern, all_user_text, re.IGNORECASE):
            state.industry = industry
            break

    # â”€â”€ Detect constraints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    constraint_patterns = [
        (r"\b(quick|fast|short|brief|rapid)\b", "short duration"),
        (r"\b(under|less than|within)\s*(\d+)\s*min", "time-constrained"),
        (r"\b(remote|online|virtual)\b", "remote"),
        (r"\b(adaptive)\b", "adaptive"),
        (r"\b(no personality|skip personality|without personality)\b", "no personality"),
    ]
    for pattern, constraint in constraint_patterns:
        if re.search(pattern, all_user_text, re.IGNORECASE):
            if constraint not in state.constraints:
                state.constraints.append(constraint)

    duration_match = DURATION_LIMIT_PATTERN.search(all_user_text)
    if duration_match:
        state.max_duration_minutes = int(duration_match.group(1))
        if "time-constrained" not in state.constraints:
            state.constraints.append("time-constrained")

    lowered_all = all_user_text.lower()
    for label, code in TYPE_KEYWORDS.items():
        if re.search(rf"\b(?:no|without|exclude|remove|drop|skip)\s+(?:all\s+|any\s+)?{re.escape(label)}", lowered_all):
            if code not in state.excluded_test_types:
                state.excluded_test_types.append(code)
        if re.search(rf"\b(?:only|just|keep only|include|add|need|want|prefer)\s+(?:all\s+|any\s+)?{re.escape(label)}", lowered_all):
            if code not in state.preferred_test_types:
                state.preferred_test_types.append(code)

    # â”€â”€ Detect additions and removals (latest turn priority) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg["content"]

        # Removals
        if REJECTION_PATTERNS.search(content):
            remove_targets = re.findall(
                r"(?:drop|remove|exclude|skip|cut|ditch|take\s*out)\s+(?:the\s+)?(.+?)(?:\.|,|$)",
                content, re.IGNORECASE,
            )
            for target in remove_targets:
                target = target.strip().rstrip(".")
                if target and len(target) < 100:
                    state.requested_removals.append(target)
                    if target not in state.rejected_assessments:
                        state.rejected_assessments.append(target)

        # Additions
        if ADDITION_PATTERNS.search(content):
            add_targets = re.findall(
                r"(?:add|include|also\s+add|throw\s+in|plus)\s+(?:a\s+)?(.+?)(?:\.|,|$)",
                content, re.IGNORECASE,
            )
            for target in add_targets:
                target = target.strip().rstrip(".")
                if target and len(target) < 100:
                    state.requested_additions.append(target)

    # â”€â”€ Detect comparison requests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if COMPARISON_PATTERNS.search(last_user_message):
        state.is_comparison_turn = True
        state.comparison_requests.append(last_user_message)
        pieces = re.split(r"\b(?:and|vs\.?|versus|between|from|with)\b", last_user_message, flags=re.IGNORECASE)
        for piece in pieces:
            cleaned = re.sub(r"\b(what'?s|what is|difference|compare|comparison|is|the|a|an)\b", " ", piece, flags=re.IGNORECASE).strip(" ?.,:-")
            if 2 < len(cleaned) < 80:
                state.comparison_targets.append(cleaned)

    # â”€â”€ Detect confirmation (check BEFORE refinement) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    has_change_request = bool(
        ADDITION_PATTERNS.search(last_user_message)
        or REJECTION_PATTERNS.search(last_user_message)
        or CORRECTION_PATTERNS.search(last_user_message)
        or REPLACEMENT_PATTERNS.search(last_user_message)
    )
    is_confirmation = bool(CONFIRMATION_PATTERNS.search(last_user_message)) and not has_change_request

    # â”€â”€ Detect refinement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not is_confirmation and (ADDITION_PATTERNS.search(last_user_message) or
            REJECTION_PATTERNS.search(last_user_message)):
        state.is_refinement_turn = True

    # â”€â”€ Detect off-topic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if OFF_TOPIC_PATTERNS.search(last_user_message) or PROMPT_INJECTION_PATTERNS.search(last_user_message):
        state.is_off_topic = True

    if is_confirmation:
        state.user_confirmed = True

    # â”€â”€ Extract last recommendations from assistant messages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg["content"]
        # Look for assessment names â€” bullet points or table rows
        if "â€¢" in content or "|" in content or "**" in content:
            # Extract from bullet points: â€¢ **Name** or - **Name**
            bullet_names = re.findall(r"[â€¢\-]\s*\*\*(.+?)\*\*", content)
            if bullet_names:
                state.last_recommendations = [n.strip() for n in bullet_names]
                break
            # Extract from markdown table rows
            table_rows = re.findall(r"\|\s*\d+\s*\|\s*(.+?)\s*\|", content)
            if table_rows:
                state.last_recommendations = [n.strip() for n in table_rows if n.strip()]
                break
            # Extract from linked names: [Name](url)
            linked_names = re.findall(r"\[([^\]]+)\]\(https?://", content)
            if linked_names:
                state.last_recommendations = [n.strip() for n in linked_names]
                break

    # â”€â”€ Compute confidence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    confidence = 0.0
    if state.role:
        confidence += 0.35
    if state.seniority or state.experience_years:
        confidence += 0.20
    if state.skills:
        confidence += 0.15
    if state.assessment_goals:
        confidence += 0.10
    if state.industry:
        confidence += 0.10
    if state.language_requirements:
        confidence += 0.05
    if state.constraints:
        confidence += 0.05
    state.confidence = min(confidence, 1.0)

    # â”€â”€ Determine missing info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not state.role:
        state.missing_information.append("role/function")
    if not state.seniority and not state.experience_years:
        state.missing_information.append("seniority/experience level")

    logger.info(
        "STATE: role=%s seniority=%s skills=%s conf=%.2f turn=%d asked=%s",
        state.role, state.seniority, state.skills[:3],
        state.confidence, state.turn_count, state.questions_already_asked,
    )
    return state


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helper: Role extraction (FIXED)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _extract_role(text: str) -> str | None:
    """Extract role from user text using targeted patterns.

    Strategy:
    1. Try phrase-level extraction ("hiring a senior Java developer")
    2. If phrase contains a role keyword, use the phrase
    3. Fall back to keyword-only extraction ("developer")
    4. Enhance keyword with skill context ("Java developer")
    """
    text = text.strip()
    if not text:
        return None

    # Stage 1: Phrase-level extraction with targeted patterns
    # These are ordered most-specific first
    phrase_patterns = [
        # "hiring/hire/recruit a [role]"
        r"\b(?:hiring|hire|recruit(?:ing)?)\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|\s+who\b|\s+with\b|\s+for\s+(?:a|our|the)\b|$)",
        # "need to hire/recruit/fill a [role]"
        r"\bneed\s+to\s+(?:hire|recruit|fill)\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|\s+who\b|\s+with\b|$)",
        # "looking for a [role]"
        r"\blooking\s+for\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|\s+who\b|\s+with\b|$)",
        # "assessments/tests for [role]"
        r"\b(?:assessments?|tests?|solutions?)\s+for\s+(.+?)(?:\.|,|$)",
        # "screen/assess [role]" 
        r"\b(?:screen(?:ing)?|assess(?:ing)?)\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|$)",
        # "position/role of/for [role]"
        r"\b(?:position|role|job)\s+(?:of|for|is)\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|$)",
    ]

    for pattern in phrase_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            role_text = match.group(1).strip().rstrip(".")
            # Validate: must be reasonable length and contain a role-like word
            if 2 < len(role_text) < 80 and _contains_role_keyword(role_text):
                logger.debug("Role extracted via phrase: '%s'", role_text)
                return role_text

    # Stage 2: Keyword-level extraction with context enrichment
    # Find role keywords in the text
    words = text.lower().split()
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^a-z]", "", word)
        if clean_word in ROLE_KEYWORDS:
            # Try to extract context: look for qualifiers around the keyword
            # Build context window: up to 3 words before + the keyword
            start = max(0, i - 3)
            context_words = words[start:i + 1]
            # Filter out noise words
            noise = {"i", "we", "a", "an", "the", "am", "is", "are", "need",
                     "want", "to", "for", "of", "and", "or", "our", "some",
                     "have", "my", "at", "be", "been", "being", "get"}
            meaningful = [w.strip(".,;:!?") for w in context_words
                         if w.strip(".,;:!?").lower() not in noise and len(w) > 1]
            if meaningful:
                role = " ".join(meaningful)
                logger.debug("Role extracted via keyword: '%s'", role)
                return role

    return None


def _contains_role_keyword(text: str) -> bool:
    """Check if text contains at least one role-indicating keyword."""
    text_lower = text.lower()
    for keyword in ROLE_KEYWORDS:
        if keyword in text_lower:
            return True
    # Also check common role-adjacent words
    extras = ["leadership", "agent", "staff", "professional", "team",
              "personnel", "officer", "candidate", "hire", "trainee"]
    for word in extras:
        if word in text_lower:
            return True
    return False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helper: Detect what the assistant already asked
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _detect_asked_questions(messages: list[dict[str, str]]) -> set[str]:
    """Scan assistant messages to determine which question categories were asked."""
    asked = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg["content"].lower()
        for category, patterns in QUESTION_CATEGORIES.items():
            if any(re.search(p, content, re.IGNORECASE) for p in patterns):
                asked.add(category)
    return asked



