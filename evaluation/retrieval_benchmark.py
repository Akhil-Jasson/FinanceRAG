"""Offline Recall@5 benchmark for FinRAG's retrieval layer.

The starter cases target the six latest 10-Q filings already indexed in the
local demo corpus. Run this only when deliberately evaluating retrieval: each
case performs one hosted embedding query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from langchain_core.documents import Document


class Retriever(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    name: str
    query: str
    ticker: str
    expected_accession_number: str


STARTER_BENCHMARK_CASES = (
    RetrievalBenchmarkCase(
        "aapl_latest_10q",
        "What revenue trends were reported in Apple's latest 10-Q?",
        "AAPL",
        "0000320193-26-000020",
    ),
    RetrievalBenchmarkCase(
        "msft_latest_10q",
        "What revenue trends were reported in Microsoft's latest 10-Q?",
        "MSFT",
        "0001193125-26-191507",
    ),
    RetrievalBenchmarkCase(
        "nvda_latest_10q",
        "What revenue trends were reported in NVIDIA's latest 10-Q?",
        "NVDA",
        "0001045810-26-000052",
    ),
    RetrievalBenchmarkCase(
        "amzn_latest_10q",
        "What revenue trends were reported in Amazon's latest 10-Q?",
        "AMZN",
        "0001018724-26-000026",
    ),
    RetrievalBenchmarkCase(
        "googl_latest_10q",
        "What revenue trends were reported in Alphabet's latest 10-Q?",
        "GOOGL",
        "0001652044-26-000071",
    ),
    RetrievalBenchmarkCase(
        "meta_latest_10q",
        "What revenue trends were reported in Meta's latest 10-Q?",
        "META",
        "0001628280-26-050705",
    ),
)


def recall_at_k(
    documents: Iterable[Document],
    expected_accession_number: str,
    k: int = 5,
) -> float:
    """Return 1.0 when the expected filing appears within the first *k* hits."""
    return float(
        any(
            document.metadata.get("accessionNumber") == expected_accession_number
            for document in list(documents)[:k]
        )
    )


def run_retrieval_benchmark(
    retriever_factory: Callable[[str], Retriever],
    cases: Iterable[RetrievalBenchmarkCase] = STARTER_BENCHMARK_CASES,
    k: int = 5,
) -> dict:
    """Evaluate a ticker-scoped retriever and return aggregate Recall@k."""
    results = []
    for case in cases:
        documents = retriever_factory(case.ticker).invoke(case.query)
        score = recall_at_k(documents, case.expected_accession_number, k)
        results.append({"name": case.name, f"recall_at_{k}": score})

    return {
        "case_count": len(results),
        f"recall_at_{k}": round(sum(row[f"recall_at_{k}"] for row in results) / len(results), 4)
        if results
        else 0.0,
        "cases": results,
    }


if __name__ == "__main__":
    from rag.retriever import get_retriever

    result = run_retrieval_benchmark(lambda ticker: get_retriever(ticker))
    print(result)
