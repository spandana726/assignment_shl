"""FastAPI application — the SHL AI Assessment Recommendation API.

Endpoints:
  GET  /health → {"status": "ok"}
  POST /chat   → {reply, recommendations, end_of_conversation}
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.catalog.loader import fetch_catalog
from app.catalog.store import CatalogStore
from app.catalog.intelligence import CatalogIntelligence
from app.retrieval.hybrid import HybridRetriever
from app.agent.graph import init_agent, run_agent

logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────────────────
_catalog_store: CatalogStore | None = None
_retriever: HybridRetriever | None = None
_intelligence: CatalogIntelligence | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load catalog, build indices, init agent."""
    global _catalog_store, _retriever, _intelligence

    t0 = time.time()
    logger.info("=== Starting SHL AI Platform ===")

    # 1. Load catalog
    products = await fetch_catalog()
    _catalog_store = CatalogStore(products)
    _intelligence = CatalogIntelligence(products)
    logger.info("Catalog loaded: %d products (%.1fs)", len(products), time.time() - t0)

    # 2. Build retrieval indices
    t1 = time.time()
    _retriever = HybridRetriever(products)
    logger.info("Retrieval indices built (%.1fs)", time.time() - t1)

    # 3. Initialize agent
    init_agent(_retriever, _catalog_store, _intelligence)
    logger.info("=== Platform ready (total %.1fs) ===", time.time() - t0)

    yield

    logger.info("=== Shutting down ===")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL AI Assessment Recommendation Platform",
    description="Conversational AI agent for SHL assessment discovery",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness probe. Returns {"status": "ok"} with HTTP 200."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Stateless chat endpoint.

    Accepts full conversation history, returns next agent reply
    with optional recommendations.
    """
    t0 = time.time()

    # Convert to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        # Run agent with timeout
        result = await asyncio.wait_for(
            run_agent(messages),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("Agent timed out after %ds", settings.request_timeout)
        # Context-aware fallback instead of 504
        from app.context.reconstructor import reconstruct_state
        from app.agent.graph import select_clarification_question
        cs = reconstruct_state(messages)
        question = select_clarification_question(cs)
        result = {
            "reply": question or "I'm having trouble processing your request. What role are you hiring for?",
            "recommendations": None,
            "end_of_conversation": False,
        }
    except Exception as e:
        logger.error("Agent error: %s", e, exc_info=True)
        from app.context.reconstructor import reconstruct_state
        from app.agent.graph import select_clarification_question
        cs = reconstruct_state(messages)
        question = select_clarification_question(cs)
        result = {
            "reply": question or "I can help you find the right SHL assessments. What role are you hiring for?",
            "recommendations": None,
            "end_of_conversation": False,
        }

    # Build response with strict validation
    recommendations = None
    if result.get("recommendations"):
        recs = []
        for r in result["recommendations"][:10]:
            try:
                recs.append(Recommendation(
                    name=r["name"],
                    url=r["url"],
                    test_type=r.get("test_type", "K"),
                ))
            except Exception as e:
                logger.warning("Skipping invalid recommendation: %s", e)
        recommendations = recs if recs else None

    response = ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recommendations,
        end_of_conversation=result.get("end_of_conversation", False),
    )

    elapsed = time.time() - t0
    logger.info(
        "Chat response: %d recs, eoc=%s, %.1fs",
        len(recommendations) if recommendations else 0,
        response.end_of_conversation,
        elapsed,
    )
    return response


# ── API metadata endpoints ──────────────────────────────────────────────────

@app.get("/api/catalog")
async def get_catalog():
    """Return the full catalog (for frontend explorer)."""
    if _catalog_store is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    return _catalog_store.get_all_products()


@app.get("/api/catalog/{entity_id}")
async def get_product(entity_id: str):
    """Return a single product by entity_id."""
    if _catalog_store is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    product = _catalog_store.find_by_id(entity_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product.to_dict()


# ── Frontend static files ────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

if FRONTEND_DIR.exists():
    @app.get("/app")
    @app.get("/app/{path:path}")
    async def serve_frontend(path: str = ""):
        """Serve the frontend SPA."""
        file_path = FRONTEND_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend-static")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level,
    )
