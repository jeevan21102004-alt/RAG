"""Deterministic query classification layer (Phase 6A).

Classifies enterprise benchmark questions into transparent query types
using ONLY local features derived from the question text plus existing
benchmark metadata (category, relevant_documents).

ZERO external calls: no LLM, no Gemini, no network. Pure stdlib.

Classification rules (explicit, applied in priority order):

1. MULTI_DOCUMENT -- len(relevant_documents) > 1 OR the benchmark
   ``category`` ends with "_multi".
2. CROSS_DOMAIN -- the question text contains vocabulary from two or
   more distinct enterprise domains. Checked only if rule 1 missed.
3. TECHNICAL -- the question text contains a term from TECHNICAL_TERMS.
   Checked only if rules 1-2 missed.
4. SIMPLE -- everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping


class QueryType(str, Enum):
    """Transparent query types for the Phase 6A baseline."""

    SIMPLE = "SIMPLE"
    MULTI_DOCUMENT = "MULTI_DOCUMENT"
    CROSS_DOMAIN = "CROSS_DOMAIN"
    TECHNICAL = "TECHNICAL"


CONJUNCTION_MARKERS: tuple[str, ...] = (
    " and ",
    " & ",
    " plus ",
    " as well as ",
    ";",
)

TECHNICAL_TERMS: tuple[str, ...] = (
    "api",
    "p95",
    "latency",
    "deployment",
    "production",
    "pull request",
    "coverage",
    "sev1",
    "incident",
    "endpoint",
    "gateway",
    "millisecond",
    "tech lead",
    "sre",
)

DOMAIN_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "hr": ("leave", "remote work", "wellness", "parental", "attendance",
           "benefit", "stipend", "arrival"),
    "engineering": ("deployment", "api", "pull request", "incident",
                    "coverage", "latency", "production", "sev1"),
    "finance": ("reimbursement", "expense", "procurement", "hotel",
                "purchase", "cfo", "director approval"),
    "product": ("release", "pricing", "plan", "contract", "discount"),
    "security": ("password", "device", "mdm", "access", "security"),
}

_MULTI_CATEGORY_SUFFIX = "_multi"


@dataclass(frozen=True)
class QueryFeatures:
    """Deterministic numerical/textual features of one question."""

    case_id: str = ""
    question: str = ""
    category: str = ""
    word_count: int = 0
    char_count: int = 0
    num_relevant_documents: int = 0
    has_multi_suffix: bool = False
    conjunction_count: int = 0
    technical_term_hits: int = 0
    technical_terms_found: List[str] = field(default_factory=list)
    domains_mentioned: List[str] = field(default_factory=list)
    num_domains_mentioned: int = 0

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value


def extract_query_features(
    question: Any,
    category: Any = "",
    relevant_documents: Any = None,
    case_id: Any = "",
) -> QueryFeatures:
    """Extract deterministic features; coerces odd inputs to safe defaults."""
    q = _safe_text(question)
    cat = _safe_text(category)
    cid = _safe_text(case_id)
    lowered = q.lower()
    words = lowered.split()
    if relevant_documents is None:
        docs: List[Any] = []
    elif isinstance(relevant_documents, (list, tuple)):
        docs = list(relevant_documents)
    else:
        docs = [relevant_documents]
    conjunction_count = sum(lowered.count(m) for m in CONJUNCTION_MARKERS)
    terms_found = [t for t in TECHNICAL_TERMS if t in lowered]
    domains = sorted(
        d for d, kws in DOMAIN_KEYWORDS.items()
        if any(k in lowered for k in kws)
    )
    return QueryFeatures(
        case_id=cid,
        question=q,
        category=cat,
        word_count=len(words),
        char_count=len(q),
        num_relevant_documents=len(docs),
        has_multi_suffix=cat.lower().endswith(_MULTI_CATEGORY_SUFFIX),
        conjunction_count=conjunction_count,
        technical_term_hits=len(terms_found),
        technical_terms_found=terms_found,
        domains_mentioned=domains,
        num_domains_mentioned=len(domains),
    )


def classify_features(features: QueryFeatures) -> QueryType:
    """Apply the documented priority rules to pre-extracted features."""
    if features.num_relevant_documents > 1 or features.has_multi_suffix:
        return QueryType.MULTI_DOCUMENT
    if features.num_domains_mentioned >= 2:
        return QueryType.CROSS_DOMAIN
    if features.technical_term_hits >= 1:
        return QueryType.TECHNICAL
    return QueryType.SIMPLE


def classify_query(
    question: Any,
    category: Any = "",
    relevant_documents: Any = None,
    case_id: Any = "",
) -> QueryType:
    """Classify one question deterministically (features + rules)."""
    return classify_features(
        extract_query_features(question, category, relevant_documents, case_id)
    )


def classify_benchmark_question(item: Mapping[str, Any]) -> QueryType:
    """Classify a single enterprise-benchmark row dict."""
    if not isinstance(item, Mapping):
        raise TypeError(f"Benchmark row must be a mapping, got {type(item)!r}")
    return classify_query(
        item.get("question", ""),
        item.get("category", ""),
        item.get("relevant_documents", []),
        item.get("case_id", ""),
    )


def classify_all(items: List[Mapping[str, Any]]) -> Dict[str, QueryType]:
    """Classify every benchmark row; returns {case_id: QueryType}."""
    result: Dict[str, QueryType] = {}
    for item in items:
        result[_safe_text(item.get("case_id", ""))] = classify_benchmark_question(
            item
        )
    return result

