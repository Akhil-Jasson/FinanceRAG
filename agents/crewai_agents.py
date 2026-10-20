"""
CrewAI-based multi-agent crew for FinRAG.

Roles:
- SEC Research Specialist  – retrieves and summarises SEC filings
- Financial Analyst        – interprets financial data and trends
- Investment Report Writer – produces the final cited research report
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_groq import ChatGroq

from agents.conversation import rewrite_follow_up_query
from config.settings import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_TEMPERATURE,
    MAX_AGENT_ITERATIONS,
)
from rag.retriever import enrich_comparison_query, get_retriever, infer_requested_form_types

logger = logging.getLogger(__name__)


# ── RAG tool ──────────────────────────────────────────────────────────────────

class SECRetrieverTool(BaseTool):
    """LangChain-backed RAG tool that searches the SEC filings vector store."""

    name: str = "SEC Filings Retriever"
    description: str = (
        "Search and retrieve relevant excerpts from SEC filings (10-K, 10-Q, 8-K). "
        "Input should be a concise search query about a company or financial topic."
    )
    tickers: List[str] = []
    reranker_enabled: Optional[bool] = None

    def _run(self, query: str) -> str:  # noqa: D401
        search_query = enrich_comparison_query(query, self.tickers)
        retriever = get_retriever(
            self.tickers,
            self.reranker_enabled,
            infer_requested_form_types(search_query),
        )
        docs = retriever.invoke(search_query)
        if not docs:
            return "No relevant SEC filing excerpts found."
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            citation = (
                f"[{i}] {meta.get('ticker', 'N/A')} "
                f"{meta.get('form', '')} "
                f"({meta.get('filingDate', 'unknown')})"
            )
            parts.append(f"{citation}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


# ── LLM factory ───────────────────────────────────────────────────────────────

def _llm() -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        max_tokens=LLM_MAX_TOKENS,
        reasoning_effort=(LLM_REASONING_EFFORT if "gpt-oss" in LLM_MODEL else None),
    )


# ── Agents ────────────────────────────────────────────────────────────────────

def create_researcher_agent(
    tickers: str | Sequence[str] | None = None,
    reranker_enabled: bool | None = None,
) -> Agent:
    """SEC Research Specialist – retrieves filing excerpts."""
    return Agent(
        role="SEC Research Specialist",
        goal=(
            "Retrieve the most relevant excerpts from SEC filings for the given query. "
            "Always include citation metadata (ticker, form type, filing date)."
        ),
        backstory=(
            "You are a former SEC examiner with 15 years of experience reading 10-K, "
            "10-Q, and 8-K filings. You excel at extracting the most financially "
            "significant information quickly and accurately."
        ),
        tools=[SECRetrieverTool(tickers=_normalize_tickers(tickers), reranker_enabled=reranker_enabled)],
        llm=_llm(),
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )


def create_analyst_agent(
    tickers: str | Sequence[str] | None = None,
    reranker_enabled: bool | None = None,
) -> Agent:
    """Financial Analyst – interprets the retrieved data."""
    return Agent(
        role="Financial Analyst",
        goal=(
            "Analyse the retrieved SEC filing excerpts to identify key financial metrics, "
            "trends, and risks. Provide quantitative and qualitative insights with citations."
        ),
        backstory=(
            "You are a CFA charterholder with expertise in equity research. "
            "You can interpret complex financial statements and regulatory disclosures, "
            "turning raw data into actionable investment intelligence."
        ),
        tools=[SECRetrieverTool(tickers=_normalize_tickers(tickers), reranker_enabled=reranker_enabled)],
        llm=_llm(),
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )


def create_writer_agent() -> Agent:
    """Investment Report Writer – produces the final cited research report."""
    return Agent(
        role="Investment Report Writer",
        goal=(
            "Synthesise the financial analysis into a concise, professional research "
            "report with an executive summary, key findings, risk assessment, and "
            "conclusion. All factual claims must reference citation numbers. Return valid, "
            "readable Markdown with clear headings, short bullets, and well-formed tables. "
            "Financial tables must use '| Metric | Value | Change | Source |' and risk "
            "tables must use '| Risk | Potential Impact | Source |'. Every data row must "
            "have equal cells and a valid [N] citation in its Source cell. "
            "Keep citations outside emphasis "
            "with a space before each citation, for example: 'Net sales were $109.4 billion "
            "[3].' Never use unmatched Markdown markers or merge words and values. Every "
            "factual paragraph and bullet must end with its own supporting citation. Repeat "
            "citations for each bullet and omit unsupported claims."
        ),
        backstory=(
            "You are a senior investment research writer for a top-tier asset management "
            "firm. Your reports are known for being clear, accurate, and well-structured, "
            "always backed by primary-source citations."
        ),
        tools=[],
        llm=_llm(),
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────

def create_retrieval_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Search for the most relevant SEC filing excerpts related to: '{query}'. "
            "Return the top excerpts with citation metadata (ticker, form, date)."
        ),
        expected_output=(
            "A list of relevant SEC filing excerpts, each labelled with a citation "
            "reference [N] including ticker, form type, and filing date."
        ),
        agent=agent,
    )


def create_analysis_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Using the retrieved SEC filing excerpts, analyse the financial situation "
            f"related to: '{query}'. "
            "Cover: revenue/earnings trends, key financial ratios, risk factors, "
            "and management guidance. Every factual paragraph or bullet must end with its "
            "own citation; never use one citation line to cover a preceding list."
        ),
        expected_output=(
            "A structured financial analysis (3-5 paragraphs) with inline citations [N] "
            "covering: revenue, profitability, risks, and forward-looking statements."
        ),
        agent=agent,
    )


def create_report_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Based on the financial analysis for '{query}', write a professional "
            "investment research report. Include: executive summary, key findings, "
            "risk assessment, and investment conclusion. Preserve all citation references. "
            "Use properly aligned Markdown tables for financial metrics and risks, with a "
            "Source column containing [N] on every data row. Every factual paragraph and "
            "bullet also needs its own citation, including the executive summary and "
            "conclusion; omit unsupported claims."
        ),
        expected_output=(
            "A polished 400-600 word investment research report in Markdown format, "
            "with clear sections and inline citation references [N]. Citations must appear "
            "after the full supported statement, never inside bold formatting or alone on "
            "a line intended to cover preceding bullets."
        ),
        agent=agent,
    )


# ── Crew ─────────────────────────────────────────────────────────────────────

def build_research_crew(
    query: str,
    tickers: str | Sequence[str] | None = None,
    reranker_enabled: bool | None = None,
) -> Crew:
    """Assemble the FinRAG CrewAI research crew for a given *query*."""
    researcher = create_researcher_agent(tickers, reranker_enabled)
    analyst = create_analyst_agent(tickers, reranker_enabled)
    writer = create_writer_agent()

    tasks = [
        create_retrieval_task(researcher, query),
        create_analysis_task(analyst, query),
        create_report_task(writer, query),
    ]

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_crew_research(
    query: str,
    tickers: str | Sequence[str] | None = None,
    reranker_enabled: bool | None = None,
    conversation_history: List[dict] | None = None,
) -> dict:
    """
    Execute the full CrewAI research pipeline for *query*.

    Returns a dict with: query, final_report (the crew's output string).
    """
    retrieval_query = rewrite_follow_up_query(query, conversation_history or [])
    crew = build_research_crew(retrieval_query, tickers, reranker_enabled)
    result = crew.kickoff()
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "final_report": str(result),
    }


def _normalize_tickers(tickers: str | Sequence[str] | None) -> List[str]:
    """Normalise optional single or comparison ticker input for retrieval."""
    values = [tickers] if isinstance(tickers, str) else list(tickers or [])
    return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))
