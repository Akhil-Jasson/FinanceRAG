"""Conversation helpers shared by the LangGraph and CrewAI pipelines."""

from __future__ import annotations

from typing import Iterable, Mapping

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config.settings import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    MAX_CONVERSATION_CHARS,
    MAX_CONVERSATION_MESSAGES,
)


def format_conversation_history(
    history: Iterable[Mapping[str, object]] | None,
    max_messages: int = MAX_CONVERSATION_MESSAGES,
    max_chars: int = MAX_CONVERSATION_CHARS,
) -> str:
    """Format a bounded, recent conversation for prompts."""
    if not history:
        return ""

    selected = list(history)[-max_messages:]
    parts: list[str] = []
    remaining = max_chars
    for message in selected:
        role = str(message.get("role", "user")).lower()
        role = "Assistant" if role == "assistant" else "User"
        content = str(message.get("content", "")).strip()
        if not content or remaining <= 0:
            continue
        content = content[:remaining]
        parts.append(f"{role}: {content}")
        remaining -= len(content)
    return "\n\n".join(parts)


def rewrite_follow_up_query(
    query: str,
    history: Iterable[Mapping[str, object]] | None = None,
) -> str:
    """Resolve references such as 'its risks' using recent chat history."""
    history_text = format_conversation_history(history)
    if not history_text:
        return query

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.0,
        groq_api_key=GROQ_API_KEY,
        max_tokens=LLM_MAX_TOKENS,
        reasoning_effort=(LLM_REASONING_EFFORT if "gpt-oss" in LLM_MODEL else None),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the latest user question as a standalone question for a search "
                "engine over SEC filings. Resolve pronouns and references from the chat "
                "history. Preserve the user's intent and financial terms. Return only "
                "the rewritten question, without explanation. If it is already standalone, "
                "return it unchanged.",
            ),
            (
                "human",
                "Chat history:\n{history}\n\nLatest user question:\n{query}",
            ),
        ]
    )
    response = (prompt | llm).invoke({"history": history_text, "query": query})
    rewritten = str(response.content).strip()
    return rewritten or query
