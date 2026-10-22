"""
MLflow tracking for FinRAG agent evaluation.

Logs citation, grounding, numeric-support, retrieval, and ticker-consistency
metrics under the configured experiment.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Generator

import mlflow

from config.settings import (
    MLFLOW_BENCHMARK_EXPERIMENT_NAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from evaluation.metrics import evaluate_response

logger = logging.getLogger(__name__)


def _setup_mlflow(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> None:
    """Configure MLflow tracking URI and ensure the experiment exists."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str | None = None) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that wraps an MLflow run with automatic setup."""
    _setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_evaluation(
    query: str,
    report: str,
    citations: list,
    retrieved_docs_text: list,
    extra_params: Dict | None = None,
    run_name: str | None = None,
    requested_ticker: str | None = None,
) -> Dict:
    """
    Evaluate a research response and log all metrics to MLflow.

    Parameters
    ----------
    query:               Original user query.
    report:              Generated research report text.
    citations:           Citation list from the pipeline.
    retrieved_docs_text: Raw text of retrieved documents (for grounding checks).
    extra_params:        Additional key-value pairs to log as MLflow params.
    run_name:            Optional name for the MLflow run.

    Returns
    -------
    The evaluation results dict (same as evaluate_response output).
    """
    results = evaluate_response(
        query,
        report,
        citations,
        retrieved_docs_text,
        requested_ticker=requested_ticker,
    )

    with mlflow_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("query", query[:250])  # MLflow param limit
        mlflow.log_param("num_retrieved_docs", len(retrieved_docs_text))
        if extra_params:
            for key, value in extra_params.items():
                mlflow.log_param(key, value)

        # Log metrics
        mlflow.log_metric("citation_validity", results["citation_validity"])
        mlflow.log_metric("citation_completeness", results["citation_completeness"])
        mlflow.log_metric(
            "numeric_citation_coverage", results["numeric_citation_coverage"]
        )
        mlflow.log_metric("citation_accuracy", results["citation_accuracy"])
        mlflow.log_metric("citation_coverage", results["citation_coverage"])
        mlflow.log_metric("grounded_claim_rate", results["grounded_claim_rate"])
        mlflow.log_metric("hallucination_rate", results["hallucination_rate"])
        mlflow.log_metric("faithfulness", results["faithfulness"])
        mlflow.log_metric("numeric_claim_support", results["numeric_claim_support"])
        mlflow.log_metric("ticker_consistency", results["ticker_consistency"])
        mlflow.log_metric("claims_evaluated", results["claims_evaluated"])
        mlflow.log_metric("retrieved_chunks", results["retrieved_chunks"])
        mlflow.log_metric("retrieved_filings", results["retrieved_filings"])
        mlflow.log_metric("num_citations", results["num_citations"])
        mlflow.log_metric("report_length_chars", results["report_length_chars"])

        logger.info(
            "[MLflow] run_id=%s  citation_accuracy=%.4f  "
            "hallucination_rate=%.4f  faithfulness=%.4f",
            run.info.run_id,
            results["citation_accuracy"],
            results["hallucination_rate"],
            results["faithfulness"],
        )

    return results


def get_experiment_summary() -> Dict:
    """
    Return a summary of all runs in the configured experiment.

    Includes averages for the primary citation, grounding, numeric-support,
    retrieval, and ticker-consistency metrics.
    """
    _setup_mlflow()
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return {"error": "Experiment not found"}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    score_column = "metrics.citation_completeness"
    if runs.empty or score_column not in runs.columns:
        summary: Dict = {"total_runs": 0}
        summary.update(get_latest_retrieval_benchmark())
        return summary

    runs = runs[runs[score_column].notna()]
    if runs.empty:
        summary = {"total_runs": 0}
        summary.update(get_latest_retrieval_benchmark())
        return summary

    metrics = [
        "citation_validity",
        "citation_completeness",
        "numeric_citation_coverage",
        "citation_accuracy",
        "citation_coverage",
        "grounded_claim_rate",
        "hallucination_rate",
        "faithfulness",
        "numeric_claim_support",
        "ticker_consistency",
        "claims_evaluated",
        "retrieved_chunks",
        "retrieved_filings",
    ]
    summary: Dict = {"total_runs": len(runs)}
    for metric in metrics:
        col = f"metrics.{metric}"
        if col in runs.columns:
            summary[f"avg_{metric}"] = round(float(runs[col].mean()), 4)

    benchmark = get_latest_retrieval_benchmark()
    summary.update(benchmark)
    return summary


def reset_evaluation_runs() -> int:
    """Soft-delete all research-evaluation runs while preserving the experiment."""
    _setup_mlflow()
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return 0

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs([experiment.experiment_id], max_results=50000)
    for run in runs:
        client.delete_run(run.info.run_id)
    logger.info("[MLflow] reset %d evaluation runs", len(runs))
    return len(runs)


def log_retrieval_benchmark(result: Dict) -> Dict:
    """Persist a manually run Recall@5 benchmark in a separate experiment."""
    _setup_mlflow(MLFLOW_BENCHMARK_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="retrieval-benchmark"):
        mlflow.log_metric("recall_at_5", float(result["recall_at_5"]))
        mlflow.log_metric("case_count", int(result["case_count"]))
    logger.info(
        "[MLflow] retrieval benchmark: recall_at_5=%.4f across %d cases",
        result["recall_at_5"],
        result["case_count"],
    )
    return result


def get_latest_retrieval_benchmark() -> Dict:
    """Return the latest stored Recall@5 result without running the benchmark."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_BENCHMARK_EXPERIMENT_NAME)
    if experiment is None:
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return {}

    latest = runs.iloc[0]
    result: Dict = {}
    if "metrics.recall_at_5" in runs.columns:
        result["latest_recall_at_5"] = round(float(latest["metrics.recall_at_5"]), 4)
    if "metrics.case_count" in runs.columns:
        result["benchmark_case_count"] = int(latest["metrics.case_count"])
    return result
