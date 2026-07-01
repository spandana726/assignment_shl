# SHL AI Assessment Intelligence

> **An AI-native conversational recommendation platform for SHL assessments powered by Context Engineering, LangGraph, Hybrid Retrieval, and Deterministic Verification.**

---

## Overview

SHL AI Assessment Intelligence helps recruiters discover the most appropriate SHL assessments through natural conversation instead of keyword search.

Unlike a traditional chatbot, the system reconstructs conversation context from stateless history, understands changing recruiter requirements, retrieves relevant assessments using a hybrid retrieval pipeline, validates every recommendation against the official SHL catalog, and returns grounded recommendations with exact catalog URLs.

The project was designed around the official SHL evaluation criteria including:

- Stateless conversation handling
- Clarification of vague recruiter requests
- Recommendation generation
- Recommendation refinement
- Assessment comparison
- Hallucination prevention
- Recall@10 optimization
- Schema compliance

---

# Key Features

- Context Engineering with stateless memory reconstruction
- LangGraph-based conversational workflow
- Hybrid Retrieval (BM25 + FAISS + RRF + Cross-Encoder)
- Dynamic prompt construction
- Deterministic hallucination verification
- Recommendation refinement without restarting conversations
- Catalog-grounded assessment comparison
- Automated evaluation using Recall@K, Precision@K and F1@K
- Premium interactive web interface

---

# Architecture

```text
Recruiter
    │
POST /chat
    │
Conversation Reconstruction
    │
Intent Routing
    │
Multi-Query Planner
    │
Hybrid Retrieval
(BM25 + Dense Search)
    │
Reciprocal Rank Fusion
    │
Cross-Encoder Reranking
    │
Recommendation Generation
    │
Catalog Verification
    │
Structured JSON Response
```

---

# Backend Components

| Module | Responsibility |
|---------|----------------|
| Context Engineering | Reconstruct conversation state, contradiction resolution, prompt building |
| Retrieval | BM25, FAISS, Hybrid Search, Query Planning, RRF |
| Catalog Intelligence | Catalog indexing, fuzzy matching, metadata filtering |
| LangGraph Agent | Clarify, Recommend, Refine, Compare, Refuse |
| Evaluation | Recall@K, Precision@K, F1@K |
| Verification | Deterministic validation of assessment names, URLs and test types |

---

# Retrieval Pipeline

1. Reconstruct conversation state
2. Generate multiple complementary search queries
3. Execute BM25 retrieval
4. Execute FAISS semantic retrieval
5. Merge using Reciprocal Rank Fusion
6. Apply metadata filtering
7. Cross-Encoder reranking
8. Generate recommendations
9. Verify against SHL catalog

---

# API

## GET /health

Returns service readiness.

```json
{"status":"ok"}
```

## POST /chat

Accepts the complete conversation history and returns:

- Assistant reply
- 1–10 SHL recommendations (when appropriate)
- Conversation completion flag

---

# Frontend

The application includes:

- Conversational Chat
- Assessment Explorer
- Evaluation Dashboard
- Developer Console

---

# Testing

Current validation includes:

- Context reconstruction
- Skill extraction
- Seniority detection
- Language extraction
- Industry detection
- Recommendation refinement
- Assessment comparison
- Off-topic detection

**Current Result:** 19/19 Context Reconstruction Tests Passed.

---

# Tech Stack

**Backend**

- FastAPI
- Python
- LangGraph
- FAISS
- BM25
- Google Gemini
- Pydantic

**Frontend**

- HTML
- CSS
- JavaScript

---

# Local Setup

```bash
git clone <repository>
cd assignment_shl
pip install -r requirements.txt
```

Create:

```text
backend/.env
```

```
GEMINI_API_KEY=YOUR_API_KEY
```

Run:

```bash
python -m uvicorn app.main:app --reload
```

---

# Design Principles

- Grounded recommendations only
- Catalog-first architecture
- Stateless API
- Context-aware conversations
- High retrieval recall
- Robust multi-turn dialogue
- Deterministic verification before every response

---

# AI-Assisted Development

AI tools were used to accelerate architecture exploration, implementation, testing, debugging, and documentation. All generated code was reviewed, modified, validated, and integrated before inclusion.

---

# License

Developed as part of the SHL AI Assessment Recommendation assignment.
