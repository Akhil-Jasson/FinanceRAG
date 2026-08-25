"""
Pydantic request/response models for the FinRAG FastAPI backend.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    """A prior user or assistant turn used to resolve follow-up questions."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

class QueryRequest(BaseModel):
    """Request body for the /research endpoint."""

    query: str = Field(..., min_length=3, description="Natural-language research query")
    ticker: Optional[str] = Field(
        None,
        description="Optional ticker filter for retrieval (e.g. AAPL)",
    )
    reranker_enabled: Optional[bool] = Field(
        None,
        description="Enable the local cross-encoder reranker for this query",
    )
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="Recent chat turns used to understand follow-up questions",
    )
    backend: str = Field(
        "langgraph",
        pattern="^(langgraph|crewai)$",
        description="Agent backend to use: 'langgraph' or 'crewai'",
    )
    evaluate: bool = Field(
        True,
        description="Whether to run the evaluation pipeline and log metrics to MLflow",
    )


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL, MSFT)")
    form_types: List[str] = Field(
        default=["10-K"],
        description="SEC form types to ingest (10-K, 10-Q, 8-K)",
    )
    max_filings: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of filings to ingest",
    )


# ── Response models ────────────────────────────────────────────────────────────

class CitationModel(BaseModel):
    id: int
    ticker: str
    form: str
    filingDate: str
    accessionNumber: str
    chunk_index: int


class QueryResponse(BaseModel):
    query: str
    final_report: str
    analysis: Optional[str] = None
    citations: List[CitationModel] = []
    evaluation: Optional[Dict] = None


class IngestResponse(BaseModel):
    ticker: str
    filings_ingested: int
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store_backend: str
    llm_model: str


class EvaluationSummaryResponse(BaseModel):
    total_runs: int
    avg_citation_validity: Optional[float] = None
    avg_citation_completeness: Optional[float] = None
    avg_numeric_citation_coverage: Optional[float] = None
    avg_citation_accuracy: Optional[float] = None
    avg_citation_coverage: Optional[float] = None
    avg_grounded_claim_rate: Optional[float] = None
    avg_hallucination_rate: Optional[float] = None
    avg_faithfulness: Optional[float] = None
    avg_numeric_claim_support: Optional[float] = None
    avg_ticker_consistency: Optional[float] = None
    avg_claims_evaluated: Optional[float] = None
    avg_retrieved_chunks: Optional[float] = None
    avg_retrieved_filings: Optional[float] = None
    latest_recall_at_5: Optional[float] = None
    benchmark_case_count: Optional[int] = None


class RetrievalBenchmarkResponse(BaseModel):
    case_count: int
    recall_at_5: float
    cases: List[Dict]


class ResetMetricsResponse(BaseModel):
    deleted_runs: int
    message: str
