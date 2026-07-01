"""ConversationState schema â€” the reconstructed memory from stateless history."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """Extracted from full message history on every request. This IS the memory."""

    # â”€â”€ What the user needs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    role: str | None = None                     # "Java developer", "plant operator"
    seniority: str | None = None                # "senior IC", "entry-level", "CXO"
    industry: str | None = None                 # "chemical facility", "healthcare"
    skills: list[str] = field(default_factory=list)        # ["Java", "Spring", "SQL"]
    experience_years: int | None = None         # 5, 15
    language_requirements: list[str] = field(default_factory=list)  # ["English", "Spanish"]
    assessment_goals: list[str] = field(default_factory=list)       # ["selection", "development"]
    volume: str | None = None                   # "500 candidates", "high-volume"
    constraints: list[str] = field(default_factory=list)            # ["quick", "under 30 min"]
    max_duration_minutes: int | None = None       # explicit duration cap, e.g. 15
    preferred_test_types: list[str] = field(default_factory=list)    # ["K", "A", "S"]
    excluded_test_types: list[str] = field(default_factory=list)     # ["P"]

    # â”€â”€ Conversation dynamics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    accepted_assessments: list[str] = field(default_factory=list)   # user confirmed
    rejected_assessments: list[str] = field(default_factory=list)   # user explicitly dropped
    requested_additions: list[str] = field(default_factory=list)    # "add personality"
    requested_removals: list[str] = field(default_factory=list)     # "drop OPQ"
    comparison_requests: list[str] = field(default_factory=list)    # "difference between X and Y"
    comparison_targets: list[str] = field(default_factory=list)     # names/aliases mentioned this turn
    last_recommendations: list[str] = field(default_factory=list)   # names from last shortlist

    # â”€â”€ Meta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    clarifications_given: dict[str, str] = field(default_factory=dict)  # what user answered
    missing_information: list[str] = field(default_factory=list)        # what we still don't know
    questions_already_asked: set[str] = field(default_factory=set)      # which categories already asked
    confidence: float = 0.0                     # 0.0 â€“ 1.0
    turn_count: int = 0
    turns_remaining: int = 8
    user_confirmed: bool = False                # user said "confirmed", "perfect", etc.
    is_comparison_turn: bool = False
    is_refinement_turn: bool = False
    is_off_topic: bool = False

    def has_enough_context(self) -> bool:
        """Do we have enough to recommend?"""
        return self.confidence >= 0.7 or (self.role is not None and self.turns_remaining <= 2)

    def to_prompt_section(self) -> str:
        """Format state as a prompt section for the LLM."""
        lines = ["## Reconstructed Conversation State"]
        if self.role:
            lines.append(f"- Role: {self.role}")
        if self.seniority:
            lines.append(f"- Seniority: {self.seniority}")
        if self.industry:
            lines.append(f"- Industry: {self.industry}")
        if self.skills:
            lines.append(f"- Skills: {', '.join(self.skills)}")
        if self.experience_years:
            lines.append(f"- Experience: {self.experience_years} years")
        if self.language_requirements:
            lines.append(f"- Language requirements: {', '.join(self.language_requirements)}")
        if self.assessment_goals:
            lines.append(f"- Assessment goals: {', '.join(self.assessment_goals)}")
        if self.volume:
            lines.append(f"- Volume: {self.volume}")
        if self.constraints:
            lines.append(f"- Constraints: {', '.join(self.constraints)}")
        if self.accepted_assessments:
            lines.append(f"- Accepted assessments: {', '.join(self.accepted_assessments)}")
        if self.rejected_assessments:
            lines.append(f"- Rejected assessments (DO NOT recommend): {', '.join(self.rejected_assessments)}")
        if self.requested_additions:
            lines.append(f"- Requested additions: {', '.join(self.requested_additions)}")
        if self.requested_removals:
            lines.append(f"- Requested removals: {', '.join(self.requested_removals)}")
        if self.last_recommendations:
            lines.append(f"- Current shortlist: {', '.join(self.last_recommendations)}")
        if self.missing_information:
            lines.append(f"- Still unknown: {', '.join(self.missing_information)}")
        lines.append(f"- Confidence: {self.confidence:.0%}")
        lines.append(f"- Turns remaining: {self.turns_remaining}")
        return "\n".join(lines)


