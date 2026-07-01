"""Tests for context reconstruction — the core of context engineering."""

import pytest

from app.context.reconstructor import reconstruct_state


class TestStateReconstruction:
    """Verify that ConversationState is correctly extracted from messages."""

    def test_empty_conversation(self):
        state = reconstruct_state([])
        assert state.turn_count == 0
        assert state.turns_remaining == 8
        assert state.confidence == 0.0

    def test_single_user_message_with_role(self):
        messages = [
            {"role": "user", "content": "I'm hiring a senior Java developer"}
        ]
        state = reconstruct_state(messages)
        assert state.seniority == "senior"
        assert "Java" in state.skills
        assert state.confidence > 0.5

    def test_extracts_multiple_skills(self):
        messages = [
            {"role": "user", "content": "Need tests for Java, Spring, SQL, and AWS"}
        ]
        state = reconstruct_state(messages)
        assert "Java" in state.skills
        assert "Spring" in state.skills
        assert "SQL" in state.skills
        assert "AWS" in state.skills

    def test_extracts_seniority_from_experience(self):
        messages = [
            {"role": "user", "content": "Candidates with 15+ years experience"}
        ]
        state = reconstruct_state(messages)
        assert state.experience_years == 15
        # 15 years should map to executive or senior
        assert state.seniority in ["executive", "senior"]

    def test_detects_entry_level(self):
        messages = [
            {"role": "user", "content": "Hiring entry-level graduates with no experience"}
        ]
        state = reconstruct_state(messages)
        assert state.seniority in ["entry-level", "graduate"]

    def test_detects_assessment_goals(self):
        messages = [
            {"role": "user", "content": "We need selection and screening assessments for hiring"}
        ]
        state = reconstruct_state(messages)
        assert "selection" in state.assessment_goals

    def test_detects_language_requirements(self):
        messages = [
            {"role": "user", "content": "Candidates need to speak Spanish and English"}
        ]
        state = reconstruct_state(messages)
        assert "Spanish" in state.language_requirements
        assert "English" in state.language_requirements

    def test_detects_industry(self):
        messages = [
            {"role": "user", "content": "Hiring plant operators for a chemical facility"}
        ]
        state = reconstruct_state(messages)
        assert "chemical" in state.industry.lower() or "industrial" in state.industry.lower()

    def test_detects_comparison_intent(self):
        messages = [
            {"role": "user", "content": "What's the difference between DSI and Safety 8.0?"}
        ]
        state = reconstruct_state(messages)
        assert state.is_comparison_turn is True

    def test_detects_removal_request(self):
        messages = [
            {"role": "user", "content": "Drop the OPQ from the list"}
        ]
        state = reconstruct_state(messages)
        assert state.is_refinement_turn is True
        assert len(state.rejected_assessments) > 0

    def test_detects_addition_request(self):
        messages = [
            {"role": "user", "content": "Add a personality assessment to the list"}
        ]
        state = reconstruct_state(messages)
        assert state.is_refinement_turn is True
        assert len(state.requested_additions) > 0

    def test_detects_off_topic(self):
        messages = [
            {"role": "user", "content": "What are the legal requirements for hiring?"}
        ]
        state = reconstruct_state(messages)
        assert state.is_off_topic is True

    def test_multi_turn_accumulation(self):
        messages = [
            {"role": "user", "content": "I'm hiring a Java developer"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Senior, 10+ years with Spring and AWS"},
        ]
        state = reconstruct_state(messages)
        assert "Java" in state.skills
        assert "Spring" in state.skills
        assert "AWS" in state.skills
        assert state.seniority == "senior"
        assert state.experience_years == 10
        assert state.turn_count == 3

    def test_confidence_increases_with_info(self):
        # Minimal info
        s1 = reconstruct_state([{"role": "user", "content": "Hi"}])
        # More info
        s2 = reconstruct_state([
            {"role": "user", "content": "I'm hiring a senior Java developer for a tech company"}
        ])
        assert s2.confidence > s1.confidence

    def test_has_enough_context(self):
        messages = [
            {"role": "user", "content": "I'm hiring a senior Java developer with Spring, SQL, AWS for selection in tech"}
        ]
        state = reconstruct_state(messages)
        assert state.has_enough_context() is True

    def test_confirmation_detection(self):
        messages = [
            {"role": "user", "content": "Perfect, that's what we need. Lock it in."}
        ]
        state = reconstruct_state(messages)
        assert state.user_confirmed is True


class TestConversationTraceReconstruction:
    """Test with patterns from actual SHL conversation traces."""

    def test_c1_senior_leadership(self):
        """C1: Senior leadership selection."""
        messages = [
            {"role": "user", "content": "We need a solution for senior leadership."},
            {"role": "assistant", "content": "What level of leadership?"},
            {"role": "user", "content": "CXOs, director-level positions; people with more than 15 years of experience."},
            {"role": "assistant", "content": "Is this for selection or development?"},
            {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."},
        ]
        state = reconstruct_state(messages)
        assert state.seniority in ["executive", "director"]
        assert "selection" in state.assessment_goals
        assert state.experience_years == 15

    def test_c6_plant_operators_safety(self):
        """C6: Plant operators in a chemical facility."""
        messages = [
            {"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners."},
        ]
        state = reconstruct_state(messages)
        assert "chemical" in (state.industry or "").lower() or "manufacturing" in (state.industry or "").lower()
        assert state.role is not None

    def test_c9_full_stack_engineer(self):
        """C9: Senior full-stack engineer with JD."""
        messages = [
            {"role": "user", "content": "Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker."},
            {"role": "assistant", "content": "Here are some recommendations..."},
            {"role": "user", "content": "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional."},
        ]
        state = reconstruct_state(messages)
        assert "Java" in state.skills
        assert "Spring" in state.skills
        assert "SQL" in state.skills
        assert "Angular" in state.skills
        assert "AWS" in state.skills
        assert "Docker" in state.skills
        assert state.seniority == "senior"
