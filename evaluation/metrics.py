"""Grounding and retrieval metrics for FinRAG responses.

The original evaluator treated a sentence as grounded when any two words
appeared anywhere in the combined context. The primary metrics below are more
specific: claims must cite a retrieved document, lexical evidence is checked
against the cited document(s), and numeric claims are checked separately.
"""

from __future__ import annotations

import re
from typing import List


STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "its", "it", "this", "that", "by", "as", "from", "we", "our",
    "their", "they", "he", "she", "which", "what", "how", "about", "into",
    "than", "also", "can", "may", "could", "would", "should", "according",
}


def _extract_citation_ids(text: str) -> List[int]:
    """Extract all [N] citation references from *text*."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    return {word for word in words if word not in STOP_WORDS}


def _numeric_tokens(text: str) -> set[str]:
    text = re.sub(r"\[\d+\]", "", text)
    values = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    return {value.replace(",", "") for value in values}


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _extract_evidence_units(text: str) -> List[str]:
    """Extract prose paragraphs, bullets, and data rows needing attribution.

    Citations placed at the end of a paragraph conventionally support that whole
    paragraph. Markdown headings, horizontal rules, and table headers are not
    factual answer units and therefore should not reduce citation completeness.
    """
    lines = text.splitlines()
    units: List[str] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or re.fullmatch(r"-{3,}", line):
            continue
        if re.match(r"^#{1,6}\s+", line) or _is_table_separator(line):
            continue
        if index + 1 < len(lines) and _is_table_separator(lines[index + 1].strip()):
            continue

        cleaned = re.sub(r"[`*_#|]", " ", line).strip(" -•")
        if len(_content_words(cleaned)) >= 3 or _numeric_tokens(cleaned):
            units.append(cleaned)
    return units


def _extract_claims(text: str) -> List[str]:
    """Extract substantive report claims while ignoring short headings."""
    claims: List[str] = []
    for line in re.split(r"\r?\n|<br\s*/?>", text):
        line = re.sub(r"[`*_#|]", " ", line).strip()
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip(" -•")
            if len(_content_words(part)) >= 6 or _numeric_tokens(part):
                claims.append(part)
    return claims


def citation_accuracy(report: str, citations: List[dict]) -> float:
    """Return the fraction of referenced citation IDs that are valid."""
    referenced_ids = set(_extract_citation_ids(report))
    if not referenced_ids:
        return 1.0
    valid_ids = {c["id"] for c in citations}
    return len(referenced_ids & valid_ids) / len(referenced_ids)


def citation_coverage(report: str, citations: List[dict]) -> float:
    """Return the fraction of substantive claims carrying a valid citation."""
    claims = _extract_claims(report)
    if not claims:
        return 1.0
    valid_ids = {c["id"] for c in citations}
    covered = sum(bool(set(_extract_citation_ids(claim)) & valid_ids) for claim in claims)
    return covered / len(claims)


def citation_validity(report: str, citations: List[dict]) -> float:
    """Return the share of inline references that resolve to retrieved sources."""
    referenced_ids = set(_extract_citation_ids(report))
    if not referenced_ids:
        return 0.0
    valid_ids = {c["id"] for c in citations}
    return len(referenced_ids & valid_ids) / len(referenced_ids)


def citation_completeness(report: str, citations: List[dict]) -> float:
    """Return the share of factual answer blocks with a valid inline citation."""
    units = _extract_evidence_units(report)
    if not units:
        return 0.0
    valid_ids = {c["id"] for c in citations}
    covered = sum(bool(set(_extract_citation_ids(unit)) & valid_ids) for unit in units)
    return covered / len(units)


def numeric_citation_coverage(report: str, citations: List[dict]) -> float:
    """Return the share of numeric answer blocks carrying a valid citation.

    This measures source attribution rather than brittle string equality, so
    equivalent presentations such as 109,417 million and 109.4 billion are not
    incorrectly labelled unsupported.
    """
    valid_ids = {c["id"] for c in citations}
    numeric_units = [unit for unit in _extract_evidence_units(report) if _numeric_tokens(unit)]
    if not numeric_units:
        return 0.0 if not report.strip() else 1.0
    covered = sum(
        bool(set(_extract_citation_ids(unit)) & valid_ids) for unit in numeric_units
    )
    return covered / len(numeric_units)


def _claim_support(claim: str, cited_context: str) -> bool:
    """Apply lexical and numeric support checks to one cited claim."""
    claim_words = _content_words(claim)
    context_words = _content_words(cited_context)
    if not claim_words:
        return True

    overlap = claim_words & context_words
    lexical_support = len(overlap) >= 2 and len(overlap) / len(claim_words) >= 0.18

    claim_numbers = _numeric_tokens(claim)
    context_numbers = _numeric_tokens(cited_context)
    numeric_support = not claim_numbers or claim_numbers <= context_numbers
    return lexical_support and numeric_support


def _documents_by_id(citations: List[dict], context_docs: List[str]) -> dict[int, str]:
    return {
        citation["id"]: context_docs[index]
        for index, citation in enumerate(citations)
        if index < len(context_docs)
    }


def grounded_claim_rate(
    report: str,
    citations: List[dict],
    context_docs: List[str],
) -> float:
    """Return the fraction of claims supported by their cited document(s)."""
    claims = _extract_claims(report)
    if not claims:
        return 1.0

    documents_by_id = _documents_by_id(citations, context_docs)
    grounded = 0
    for claim in claims:
        cited_ids = [cid for cid in _extract_citation_ids(claim) if cid in documents_by_id]
        cited_context = " ".join(documents_by_id[cid] for cid in cited_ids)
        if cited_context and _claim_support(claim, cited_context):
            grounded += 1
    return grounded / len(claims)


def numeric_claim_support(
    report: str,
    citations: List[dict],
    context_docs: List[str],
) -> float:
    """Return the fraction of numeric claims whose numbers occur in cited text."""
    documents_by_id = _documents_by_id(citations, context_docs)
    numeric_claims = [claim for claim in _extract_claims(report) if _numeric_tokens(claim)]
    if not numeric_claims:
        return 1.0

    supported = 0
    for claim in numeric_claims:
        cited_ids = [cid for cid in _extract_citation_ids(claim) if cid in documents_by_id]
        cited_context = " ".join(documents_by_id[cid] for cid in cited_ids)
        if cited_context and _numeric_tokens(claim) <= _numeric_tokens(cited_context):
            supported += 1
    return supported / len(numeric_claims)


def ticker_consistency(requested_ticker: str | None, citations: List[dict]) -> float:
    """Return the fraction of citations belonging to the requested ticker."""
    if not requested_ticker:
        return 1.0
    if not citations:
        return 0.0
    ticker = requested_ticker.strip().upper()
    return sum(c.get("ticker", "").upper() == ticker for c in citations) / len(citations)


def _split_sentences(text: str) -> List[str]:
    """Backward-compatible simple sentence splitter."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def hallucination_rate(
    report: str,
    context_docs: List[str],
    citations: List[dict] | None = None,
) -> float:
    """Return unsupported-claim rate, using cited claims when available.

    The optional legacy form is retained for callers that do not have citation
    metadata yet.
    """
    if citations is not None:
        return 1.0 - grounded_claim_rate(report, citations, context_docs)

    sentences = _split_sentences(report)
    if not sentences:
        return 0.0
    combined_context = " ".join(context_docs).lower()
    unsupported = 0
    for sentence in sentences:
        words = _content_words(sentence)
        if sum(word in combined_context for word in words) < 2:
            unsupported += 1
    return unsupported / len(sentences)


def faithfulness(
    report: str,
    context_docs: List[str],
    citations: List[dict] | None = None,
) -> float:
    """Return the complement of the unsupported-claim rate."""
    return 1.0 - hallucination_rate(report, context_docs, citations)


def evaluate_response(
    query: str,
    report: str,
    citations: list,
    retrieved_docs_text: list,
    requested_ticker: str | None = None,
) -> dict:
    """Run citation-aware claim, numeric, entity, and retrieval metrics."""
    grounded = grounded_claim_rate(report, citations, retrieved_docs_text)
    return {
        "query": query,
        "citation_validity": round(citation_validity(report, citations), 4),
        "citation_completeness": round(citation_completeness(report, citations), 4),
        "numeric_citation_coverage": round(
            numeric_citation_coverage(report, citations), 4
        ),
        "citation_accuracy": round(citation_accuracy(report, citations), 4),
        "citation_coverage": round(citation_coverage(report, citations), 4),
        "grounded_claim_rate": round(grounded, 4),
        "hallucination_rate": round(1.0 - grounded, 4),
        "faithfulness": round(grounded, 4),
        "numeric_claim_support": round(
            numeric_claim_support(report, citations, retrieved_docs_text), 4
        ),
        "ticker_consistency": round(ticker_consistency(requested_ticker, citations), 4),
        "claims_evaluated": len(_extract_claims(report)),
        "retrieved_chunks": len(retrieved_docs_text),
        "retrieved_filings": len({c.get("accessionNumber") for c in citations}),
        "num_citations": len(citations),
        "report_length_chars": len(report),
    }
