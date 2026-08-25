"""
FastAPI route handlers for FinRAG.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import (
    EvaluationSummaryResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResetMetricsResponse,
    RetrievalBenchmarkResponse,
)
from config.settings import LLM_MODEL, VECTOR_STORE_BACKEND

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return system health and configuration summary."""
    return HealthResponse(
        status="ok",
        vector_store_backend=VECTOR_STORE_BACKEND,
        llm_model=LLM_MODEL,
    )


# ── Research ──────────────────────────────────────────────────────────────────

@router.post("/research", response_model=QueryResponse, tags=["Research"])
def research(request: QueryRequest) -> QueryResponse:
    """
    Run the multi-agent research pipeline for a natural-language query.

    Choose *backend* = "langgraph" (default) or "crewai".
    Set *evaluate* = true to log quality metrics to MLflow.
    """
    try:
        if request.backend == "crewai":
            from agents.crewai_agents import run_crew_research

            result = run_crew_research(
                request.query,
                request.ticker,
                request.reranker_enabled,
                [message.model_dump() for message in request.conversation_history],
            )
            citations: List[dict] = []
        else:
            from agents.langgraph_agents import run_research_pipeline

            result = run_research_pipeline(
                request.query,
                request.ticker,
                request.reranker_enabled,
                [message.model_dump() for message in request.conversation_history],
            )
            citations = result.get("citations", [])

    except FileNotFoundError as exc:
        logger.warning("Research requested before FAISS index was created: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=(
                "No FAISS index was found yet. Use the Ingest Filings panel "
                "or POST /api/v1/ingest first so the app has data to search."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    evaluation_result = None
    if request.evaluate:
        try:
            from rag.retriever import get_retriever
            from evaluation.mlflow_tracker import log_evaluation

            retrieval_query = result.get("retrieval_query", request.query)
            retriever = get_retriever(request.ticker, request.reranker_enabled)
            docs = retriever.invoke(retrieval_query)
            docs_text = [d.page_content for d in docs]
            evaluation_result = log_evaluation(
                query=request.query,
                report=result["final_report"],
                citations=citations,
                retrieved_docs_text=docs_text,
                extra_params={
                    "backend": request.backend,
                    "multi_turn": bool(request.conversation_history),
                },
                run_name=f"{request.backend}-query",
                requested_ticker=request.ticker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation logging failed: %s", exc)

    return QueryResponse(
        query=request.query,
        final_report=result["final_report"],
        analysis=result.get("analysis"),
        citations=citations,
        evaluation=evaluation_result,
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Data"])
def ingest_filings(request: IngestRequest) -> IngestResponse:
    """
    Fetch SEC filings for *ticker* and index them into the vector store.

    For FAISS: the index is persisted to disk.
    For Pinecone: documents are upserted to the cloud index.
    """
    try:
        from rag.sec_fetcher import fetch_filings_for_ticker

        filings = fetch_filings_for_ticker(
            ticker=request.ticker,
            form_types=request.form_types,
            max_filings=request.max_filings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Filing fetch error for %s", request.ticker)
        raise HTTPException(status_code=502, detail=f"SEC EDGAR error: {exc}") from exc

    if not filings:
        raise HTTPException(
            status_code=404,
            detail=f"No {request.form_types} filings found for {request.ticker}",
        )

    try:
        if VECTOR_STORE_BACKEND == "pinecone":
            from rag.pinecone_store import upsert_filings_to_pinecone

            upsert_filings_to_pinecone(filings)
            # Pinecone's batch API does not return the inserted chunk count.
            # Keep this conservative estimate for that optional backend only.
            from config.settings import CHUNK_SIZE

            total_text_len = sum(len(f.get("text", "")) for f in filings)
            filings_ingested = len(filings)
            chunks_indexed = max(1, total_text_len // CHUNK_SIZE)
        else:
            from rag.faiss_store import add_filings_to_faiss_index

            filings_ingested, chunks_indexed = add_filings_to_faiss_index(filing_dicts=filings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store indexing error")
        raise HTTPException(status_code=500, detail=f"Indexing error: {exc}") from exc

    if filings_ingested == 0:
        message = (
            f"No new filings were indexed for {request.ticker.upper()}; "
            "the selected filing accession number(s) are already present."
        )
    else:
        message = f"Successfully indexed {filings_ingested} new filing(s) for {request.ticker.upper()}"

    return IngestResponse(
        ticker=request.ticker.upper(),
        filings_ingested=filings_ingested,
        chunks_indexed=chunks_indexed,
        message=message,
    )


# ── Evaluation summary ────────────────────────────────────────────────────────

@router.post(
    "/evaluation/retrieval-benchmark",
    response_model=RetrievalBenchmarkResponse,
    tags=["Evaluation"],
)
def run_retrieval_benchmark() -> RetrievalBenchmarkResponse:
    """Run and store the manual Recall@5 retrieval benchmark."""
    try:
        from evaluation.mlflow_tracker import log_retrieval_benchmark
        from evaluation.retrieval_benchmark import run_retrieval_benchmark as run_benchmark
        from rag.retriever import get_retriever

        result = run_benchmark(lambda ticker: get_retriever(ticker))
        log_retrieval_benchmark(result)
        return RetrievalBenchmarkResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval benchmark failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/evaluation/reset",
    response_model=ResetMetricsResponse,
    tags=["Evaluation"],
)
def reset_evaluation_metrics() -> ResetMetricsResponse:
    """Clear MLflow research-evaluation runs without touching filing data."""
    try:
        from evaluation.mlflow_tracker import reset_evaluation_runs

        deleted_runs = reset_evaluation_runs()
        return ResetMetricsResponse(
            deleted_runs=deleted_runs,
            message="Evaluation metrics reset. Filing data and retrieval benchmarks were preserved.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation reset failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.get(
    "/evaluation/summary",
    response_model=EvaluationSummaryResponse,
    tags=["Evaluation"],
)
def evaluation_summary() -> EvaluationSummaryResponse:
    """Return aggregate evaluation metrics from the MLflow experiment."""
    try:
        from evaluation.mlflow_tracker import get_experiment_summary

        summary = get_experiment_summary()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to retrieve MLflow summary")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EvaluationSummaryResponse(
        total_runs=summary.get("total_runs", 0),
        avg_citation_validity=summary.get("avg_citation_validity"),
        avg_citation_completeness=summary.get("avg_citation_completeness"),
        avg_numeric_citation_coverage=summary.get("avg_numeric_citation_coverage"),
        avg_citation_accuracy=summary.get("avg_citation_accuracy"),
        avg_citation_coverage=summary.get("avg_citation_coverage"),
        avg_grounded_claim_rate=summary.get("avg_grounded_claim_rate"),
        avg_hallucination_rate=summary.get("avg_hallucination_rate"),
        avg_faithfulness=summary.get("avg_faithfulness"),
        avg_numeric_claim_support=summary.get("avg_numeric_claim_support"),
        avg_ticker_consistency=summary.get("avg_ticker_consistency"),
        avg_claims_evaluated=summary.get("avg_claims_evaluated"),
        avg_retrieved_chunks=summary.get("avg_retrieved_chunks"),
        avg_retrieved_filings=summary.get("avg_retrieved_filings"),
        latest_recall_at_5=summary.get("latest_recall_at_5"),
        benchmark_case_count=summary.get("benchmark_case_count"),
    )
