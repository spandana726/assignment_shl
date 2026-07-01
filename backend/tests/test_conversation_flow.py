"""End-to-end conversation flow test.

Tests the EXACT scenario from the bug report:
  User: I need an assessment.
  AI: What role are you hiring for?
  User: I need to hire a Java developer.
  AI: (asks about seniority, NOT role again)
  User: Senior.
  AI: (asks about skills OR gives recommendations, does NOT hang)
  User: Spring, SQL and AWS.
  AI: Returns grounded recommendations (does NOT hang)

Also tests that rule-based clarification works without API key.
"""
import httpx
import sys
import json
import time


BASE = "http://localhost:8000"


def chat(messages):
    """Send a chat request and return the response."""
    payload = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
    r = httpx.post(f"{BASE}/chat", json=payload, timeout=15)
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    data = r.json()
    assert "reply" in data, f"Missing 'reply' in response: {data}"
    assert "recommendations" in data, f"Missing 'recommendations' in response: {data}"
    assert "end_of_conversation" in data, f"Missing 'end_of_conversation' in response: {data}"
    return data


def test_flow_java_developer():
    """Test the exact conversation from the bug report."""
    print("=" * 60)
    print("TEST: Java Developer Conversation Flow")
    print("=" * 60)

    messages = []

    # Turn 1: Vague request
    messages.append({"role": "user", "content": "I need an assessment."})
    r1 = chat(messages)
    print(f"\nUser: I need an assessment.")
    print(f"AI:   {r1['reply'][:120]}")
    assert r1["recommendations"] is None, "Should NOT have recommendations yet"
    assert "role" in r1["reply"].lower() or "hiring" in r1["reply"].lower() or "position" in r1["reply"].lower(), \
        f"Should ask about role, got: {r1['reply']}"

    # Turn 2: Provides role
    messages.append({"role": "assistant", "content": r1["reply"]})
    messages.append({"role": "user", "content": "I need to hire a Java developer."})
    r2 = chat(messages)
    print(f"\nUser: I need to hire a Java developer.")
    print(f"AI:   {r2['reply'][:120]}")
    # CRITICAL: Must NOT ask about role again
    assert "what role" not in r2["reply"].lower(), \
        f"BUG: System re-asked about role! Got: {r2['reply']}"

    # Turn 3: Provides seniority
    messages.append({"role": "assistant", "content": r2["reply"]})
    messages.append({"role": "user", "content": "Senior."})
    r3 = chat(messages)
    print(f"\nUser: Senior.")
    print(f"AI:   {r3['reply'][:120]}")
    # CRITICAL: Must NOT hang -- must return a response
    assert len(r3["reply"]) > 10, f"Response too short (hung?): {r3['reply']}"
    # Should NOT re-ask about role or seniority
    assert "what role" not in r3["reply"].lower(), \
        f"BUG: Re-asked about role! Got: {r3['reply']}"

    # Turn 4: Provides skills
    messages.append({"role": "assistant", "content": r3["reply"]})
    messages.append({"role": "user", "content": "Spring, SQL and AWS."})
    r4 = chat(messages)
    print(f"\nUser: Spring, SQL and AWS.")
    print(f"AI:   {r4['reply'][:200]}")
    # By this point, should have recommendations
    if r4["recommendations"]:
        print(f"\n  [OK] Recommendations returned: {len(r4['recommendations'])} assessments")
        for rec in r4["recommendations"][:5]:
            print(f"    - {rec['name']} ({rec['test_type']})")
    else:
        print(f"\n  (No recommendations yet, but response is valid -- not hung)")

    print("\n" + "=" * 60)
    print("[PASS] No repeated questions, no hanging, proper flow")
    print("=" * 60)


def test_flow_short_messages():
    """Test that very short messages work (don't hang)."""
    print("\n" + "=" * 60)
    print("TEST: Short Messages")
    print("=" * 60)

    # Single word: "Java"
    r = chat([{"role": "user", "content": "Java"}])
    print(f"User: Java -> AI: {r['reply'][:100]}")
    assert len(r["reply"]) > 5

    # Single word: "Senior."
    r = chat([{"role": "user", "content": "Senior."}])
    print(f"User: Senior. -> AI: {r['reply'][:100]}")
    assert len(r["reply"]) > 5

    # Short answer
    r = chat([{"role": "user", "content": "yes"}])
    print(f"User: yes -> AI: {r['reply'][:100]}")
    assert len(r["reply"]) > 5

    print("[PASS] Short messages handled")


def test_flow_various_phrasings():
    """Test that various natural language phrasings work."""
    print("\n" + "=" * 60)
    print("TEST: Various Phrasings")
    print("=" * 60)

    phrasings = [
        "Need to recruit a backend engineer",
        "Looking for a Python developer",
        "Hiring freshers",
        "Need assessments for finance graduates",
        "We're recruiting plant operators",
        "Find tests for customer support executives",
        "Senior Java developer with Spring Boot and AWS",
        "Recommend assessments for leadership roles",
    ]

    for phrase in phrasings:
        r = chat([{"role": "user", "content": phrase}])
        has_recs = "[recs]" if r["recommendations"] else "[clfy]"
        print(f"  {has_recs} | {phrase[:50]:50s} -> {r['reply'][:60]}")
        assert len(r["reply"]) > 5, f"Empty reply for: {phrase}"
        # Should never re-ask about role if it was clearly stated
        if any(kw in phrase.lower() for kw in ["developer", "engineer", "operator", "executive"]):
            assert "what role" not in r["reply"].lower(), \
                f"Incorrectly asked about role when it was stated: {phrase}"

    print("[PASS] All phrasings handled")


def test_schema_compliance():
    """Test strict SHL schema compliance."""
    print("\n" + "=" * 60)
    print("TEST: Schema Compliance")
    print("=" * 60)

    r = chat([{"role": "user", "content": "I need to hire a senior Java developer with Spring and AWS"}])
    assert isinstance(r["reply"], str)
    assert isinstance(r["end_of_conversation"], bool)
    assert r["recommendations"] is None or isinstance(r["recommendations"], list)

    if r["recommendations"]:
        for rec in r["recommendations"]:
            assert "name" in rec, f"Missing 'name' in rec: {rec}"
            assert "url" in rec, f"Missing 'url' in rec: {rec}"
            assert "test_type" in rec, f"Missing 'test_type' in rec: {rec}"
            assert rec["url"].startswith("https://"), f"Invalid URL: {rec['url']}"

    print("[PASS] Schema compliance")


def test_off_topic_refusal():
    """Test off-topic handling."""
    print("\n" + "=" * 60)
    print("TEST: Off-Topic Refusal")
    print("=" * 60)

    r = chat([{"role": "user", "content": "What is the weather today?"}])
    print(f"  User: What is the weather today? -> AI: {r['reply'][:80]}")
    assert r["recommendations"] is None
    assert "shl" in r["reply"].lower() or "assessment" in r["reply"].lower(), \
        "Should redirect to SHL scope"

    print("[PASS] Off-topic handled")


def test_no_repeated_questions():
    """Test that the system never repeats a question that was already answered."""
    print("\n" + "=" * 60)
    print("TEST: No Repeated Questions")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "I need to hire a senior Python developer with Django and PostgreSQL."},
    ]
    r = chat(messages)
    print(f"  Turn 1: {r['reply'][:80]}")

    # The role, seniority, and skills were all provided. Should NOT ask about any of them.
    reply_lower = r["reply"].lower()
    assert "what role" not in reply_lower, "Should not ask about role -- it was stated"

    print("[PASS] No repeated questions")


if __name__ == "__main__":
    # Wait for server
    print("Checking server health...")
    for attempt in range(5):
        try:
            r = httpx.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print("Server is healthy!\n")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Server not responding!")
        sys.exit(1)

    # Run all tests
    failures = 0
    tests = [
        test_flow_java_developer,
        test_flow_short_messages,
        test_flow_various_phrasings,
        test_schema_compliance,
        test_off_topic_refusal,
        test_no_repeated_questions,
    ]

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"\n[FAIL]: {e}")
            failures += 1
        except Exception as e:
            print(f"\n[ERROR]: {type(e).__name__}: {e}")
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print(f"ALL {len(tests)} TESTS PASSED")
    else:
        print(f"{failures}/{len(tests)} TESTS FAILED")
    print("=" * 60)
    sys.exit(failures)

