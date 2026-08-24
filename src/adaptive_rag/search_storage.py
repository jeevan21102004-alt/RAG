"""JSON-based storage for automated retrieval search results.

This module persists :class:`SearchResult` objects as JSON files under
``experiments/search_results/``.  No database, no secrets, and no large
temporary caches are stored.
"""

from __future__ import annotations

import json
from pathlib import Path

from .search_result import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_RESULTS_DIR = PROJECT_ROOT / "experiments" / "search_results"


def save_search_result(
    result: SearchResult,
    path: Path | str | None = None,
) -> Path:
    """Save a :class:`SearchResult` as a JSON file.

    Parameters
    ----------
    result:
        The search result to persist.
    path:
        Optional file path.  If omitted, the result is saved to
        ``experiments/search_results/{search_id}.json``.

    Returns
    -------
    Path
        The path where the result was written.
    """
    if path is not None:
        out = Path(path)
    else:
        out = SEARCH_RESULTS_DIR / f"{result.search_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json(), encoding="utf-8")
    return out


def load_search_result(path: Path | str) -> SearchResult:
    """Load a :class:`SearchResult` from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    SearchResult
        The deserialized search result.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _from_dict(data)


def _from_dict(data: dict) -> SearchResult:
    """Rebuild a :class:`SearchResult` from its JSON dictionary."""
    from .search_result import SearchRankingEntry

    ranking = [
        SearchRankingEntry(
            experiment_id=entry["experiment_id"],
            parameters=dict(entry.get("parameters", {})),
            objective_score=entry.get("objective_score"),
            retrieval_f1=entry.get("retrieval_f1"),
            context_relevance=entry.get("context_relevance"),
            latency_ms=entry.get("latency_ms"),
            status=entry.get("status", "SUCCESS"),
        )
        for entry in data.get("ranking", [])
    ]

    best_data = data.get("best_configuration")
    best = None
    if best_data is not None:
        best = SearchRankingEntry(
            experiment_id=best_data["experiment_id"],
            parameters=dict(best_data.get("parameters", {})),
            objective_score=best_data.get("objective_score"),
            retrieval_f1=best_data.get("retrieval_f1"),
            context_relevance=best_data.get("context_relevance"),
            latency_ms=best_data.get("latency_ms"),
            status=best_data.get("status", "SUCCESS"),
        )

    return SearchResult(
        search_id=data.get("search_id", ""),
        total_configurations=data.get("total_configurations", 0),
        completed_configurations=data.get("completed_configurations", 0),
        failed_configurations=data.get("failed_configurations", 0),
        ranking=ranking,
        best_configuration=best,
        best_objective_score=data.get("best_objective_score"),
        objective_definition=dict(data.get("objective_definition", {})),
        metadata=dict(data.get("metadata", {})),
    )