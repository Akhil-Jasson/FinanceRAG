"""
FinRAG FastAPI application entry point.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.settings import API_HOST, API_PORT, API_RELOAD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinRAG – Agentic Financial Research Assistant",
    description=(
        "Multi-agent system using LangGraph and CrewAI that retrieves SEC filings "
        "via RAG (FAISS/Pinecone), synthesises cited financial summaries, and tracks "
        "evaluation metrics (citation completeness, citation validity, numeric "
        "citation coverage, and retrieval Recall@5) in MLflow."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
def root():
    return {
        "service": "FinRAG",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )
