"""
LangGraph multi-agent orchestration for FinRAG.

Graph topology:
  START ──► retriever_node ──► analyst_node ──► writer_node ──► END

State is a typed dict that flows through the graph.
"""

from __future__ import annotations

import logging
from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from agents.conversation import rewrite_follow_up_query
from config.settings import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_TEMPERATURE,
)
from rag.retriever import enrich_comparison_query, get_retriever, infer_requested_form_types

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    query: str
    retrieval_query: str
    tickers: List[str]
    reranker_enabled: Optional[bool]
    conversation_history: List[dict]
    retrieved_docs: List[Document]
    analysis: str
    final_report: str
    citations: List[dict]
    messages: Annotated[List[BaseMessage], lambda a, b: a + b]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm() -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        max_tokens=LLM_MAX_TOKENS,
        reasoning_effort=(LLM_REASONING_EFFORT if "gpt-oss" in LLM_MODEL else None),
    )


def _invoke_nonempty(chain, payload: dict, stage: str) -> str:
    """Invoke an LLM chain once, retrying a successful-but-empty completion."""
    for attempt in range(2):
        response = chain.invoke(payload)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        text = str(content or "").strip()
        if text:
            return text
        logger.warning("[LangGraph] %s returned empty content (attempt %d)", stage, attempt + 1)
    raise RuntimeError(f"The {stage} model returned an empty response twice. Please retry.")


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citation = (
            f"[{i}] {meta.get('ticker', 'N/A')} "
            f"{meta.get('form', '')} "
            f"({meta.get('filingDate', 'unknown date')})"
        )
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(docs: List[Document]) -> List[dict]:
    """Build a citation list from document metadata."""
    citations = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citations.append(
            {
                "id": i,
                "ticker": meta.get("ticker", "N/A"),
                "form": meta.get("form", ""),
                "filingDate": meta.get("filingDate", ""),
                "accessionNumber": meta.get("accessionNumber", ""),
                "chunk_index": meta.get("chunk_index", 0),
            }
        )
    return citations


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retriever_node(state: ResearchState) -> ResearchState:
    """Retrieve relevant SEC filing chunks for the user query."""
    search_query = state.get("retrieval_query") or state["query"]
    search_query = enrich_comparison_query(search_query, state.get("tickers"))
    logger.info("[LangGraph] retriever_node: query=%s", search_query)
    retriever = get_retriever(
        state.get("tickers"),
        state.get("reranker_enabled"),
        infer_requested_form_types(search_query),
    )
    docs = retriever.invoke(search_query)
    citations = _extract_citations(docs)
    return {
        **state,
        "retrieval_query": search_query,
        "retrieved_docs": docs,
        "citations": citations,
        "messages": [HumanMessage(content=state["query"])],
    }


def analyst_node(state: ResearchState) -> ResearchState:
    """Analyse the retrieved documents and produce structured insights."""
    logger.info("[LangGraph] analyst_node")
    context = _format_docs(state["retrieved_docs"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert financial analyst specialising in SEC filings. "
                "Analyse the provided filing excerpts and produce structured insights "
                "covering: revenue trends, profitability, risk factors, and key metrics. "
                "Every paragraph or bullet containing a factual or numeric claim must end "
                "with one or more supporting citation numbers [N]. Repeat citations on "
                "each bullet; never use one citation line to cover a preceding list. "
                "Omit claims that are not supported by the excerpts. For a multi-company "
                "comparison, compare every supported revenue, profitability, liquidity, "
                "and risk item; if one dimension is unavailable, label only that dimension "
                "as unavailable rather than declining the entire analysis.",
            ),
            (
                "human",
                "Query: {query}\n\nFiling excerpts:\n{context}\n\n"
                "Provide a detailed financial analysis with citations.",
            ),
        ]
    )
    chain = prompt | _llm()
    analysis = _invoke_nonempty(
        chain,
        {"query": state["query"], "context": context},
        "analyst",
    )
    return {
        **state,
        "analysis": analysis,
        "messages": [AIMessage(content=analysis)],
    }


def writer_node(state: ResearchState) -> ResearchState:
    """Synthesise the analysis into a well-structured research report."""
    logger.info("[LangGraph] writer_node")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial report writer. Convert the analysis into a concise, "
                "professional research report suitable for investors. "
                "Include an executive summary, key findings, risk assessment, and conclusion. "
                "Preserve all citation references [N]. "
                "Return valid, readable Markdown using clear ## section headings, short "
                "bullets, and Markdown tables where they improve readability. Financial "
                "tables must use exactly '| Metric | Value | Change | Source |' and risk "
                "tables must use exactly '| Risk | Potential Impact | Source |'. Every "
                "data row must have the same number of cells and a valid [N] citation in "
                "its Source cell. Do not place section labels or blank rows inside tables. "
                "Keep citations outside emphasis, with a space before each citation, for "
                "example: 'Net sales were $109.4 billion [3].' Never place [N] inside **, "
                "never leave unmatched Markdown markers, and never merge words or values. "
                "Every factual or numeric paragraph and every bullet MUST end with its own "
                "supporting [N] citation; each table row MUST contain one in Source. This "
                "includes executive-summary and conclusion text. "
                "Never put a citation on a separate line to cover earlier bullets. Repeat "
                "the citation for every applicable bullet. Omit any unsupported claim.",
            ),
            (
                "human",
                "Analysis:\n{analysis}\n\nOriginal query: {query}\n\n"
                "Write the final investment research report. Before returning it, verify "
                "that every factual paragraph and bullet ends with a valid citation already "
                "present in the analysis.",
            ),
        ]
    )
    chain = prompt | _llm()
    report = _invoke_nonempty(
        chain,
        {"analysis": state["analysis"], "query": state["query"]},
        "writer",
    )
    return {
        **state,
        "final_report": report,
        "messages": [AIMessage(content=report)],
    }


# ── Graph construction ────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """Build and compile the FinRAG LangGraph research pipeline."""
    graph = StateGraph(ResearchState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_research_pipeline(
    query: str,
    tickers: str | Sequence[str] | None = None,
    reranker_enabled: bool | None = None,
    conversation_history: List[dict] | None = None,
) -> dict:
    """
    Execute the full LangGraph research pipeline for *query*.

    Returns a dict with keys: query, final_report, analysis, citations.
    """
    app = build_research_graph()
    history = conversation_history or []
    retrieval_query = rewrite_follow_up_query(query, history)
    initial_state: ResearchState = {
        "query": query,
        "retrieval_query": retrieval_query,
        "tickers": _normalize_tickers(tickers),
        "reranker_enabled": reranker_enabled,
        "conversation_history": history,
        "retrieved_docs": [],
        "analysis": "",
        "final_report": "",
        "citations": [],
        "messages": [],
    }
    final_state = app.invoke(initial_state)
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "final_report": final_state["final_report"],
        "analysis": final_state["analysis"],
        "citations": final_state["citations"],
    }


def _normalize_tickers(tickers: str | Sequence[str] | None) -> List[str]:
    """Normalise optional single or comparison ticker input for retrieval."""
    values = [tickers] if isinstance(tickers, str) else list(tickers or [])
    return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))
