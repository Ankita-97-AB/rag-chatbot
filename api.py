"""
FastAPI server for the RAG Chatbot.

Run locally:
    uvicorn api:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from data_loader import load_all_files
from vector_store import VectorStore
from rag_chain import answer

load_dotenv()

# ── Global state ──────────────────────────────────────────────────────────────
store: VectorStore | None = None
client: OpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build vector index on startup."""
    global store, client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    store = VectorStore(client)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Dataset directory not found: {data_dir}")

    print("Loading documents...")
    chunks = load_all_files(data_dir)
    print(f"Building FAISS index for {len(chunks)} chunks...")
    store.build(chunks)
    print("Ready!")

    yield  # app is running

    # Cleanup (if needed)


app = FastAPI(
    title="RAG Chatbot API",
    description="Car buying assistant powered by RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Serve frontend ───────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── Request/Response models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    chat_history: list[dict] = []
    top_k: int = 6


class Source(BaseModel):
    source: str
    page: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if store is None or client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    reply, retrieved = answer(
        query=request.query,
        store=store,
        client=client,
        chat_history=request.chat_history,
        top_k=request.top_k,
    )

    sources = [
        Source(
            source=c.get("source", ""),
            page=str(c.get("page", "")),
            score=round(c.get("score", 0), 4),
        )
        for c in retrieved
    ]

    return ChatResponse(answer=reply, sources=sources)
