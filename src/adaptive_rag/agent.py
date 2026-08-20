from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import generate_text


SEARCH = "SEARCH"
ANSWER = "ANSWER"
MAX_RETRIEVAL_ATTEMPTS = 2


@dataclass(frozen=True)
class AgentDecision:
    action: str
    reason: str


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            body = "\n".join(lines[1:])
            if body.endswith("```"):
                body = body[:-3]
            return body.strip()
    return stripped


def _parse_decision(text: str, default_action: str, default_reason: str) -> AgentDecision:
    try:
        data = json.loads(_extract_json_block(text))
    except json.JSONDecodeError:
        return AgentDecision(action=default_action, reason=default_reason)

    action = str(data.get("action", default_action)).upper()
    if action not in {SEARCH, ANSWER}:
        action = default_action

    reason = str(data.get("reason", default_reason)).strip() or default_reason
    return AgentDecision(action=action, reason=reason)


def _build_search_decision_prompt(question: str) -> str:
    return (
        "You are an agent deciding whether retrieval is needed before answering.\n"
        "Return only valid JSON in this exact shape:\n"
        '{"action":"SEARCH","reason":"..."}\n'
        "or:\n"
        '{"action":"ANSWER","reason":"..."}\n\n'
        "Choose SEARCH when the question likely needs project documents.\n"
        "Choose ANSWER when the question can be answered without retrieval.\n"
        "Do not include markdown fences or extra text.\n\n"
        f"Question: {question}"
    )


def _build_context_evaluation_prompt(question: str, context: str) -> str:
    return (
        "You are an agent judging whether retrieved context is sufficient.\n"
        "Return only valid JSON in this exact shape:\n"
        '{"action":"ANSWER","reason":"..."}\n'
        "or:\n"
        '{"action":"SEARCH","reason":"..."}\n\n'
        "Choose ANSWER only if the retrieved context is sufficient to answer the question.\n"
        "Choose SEARCH if the context is insufficient or incomplete.\n"
        "Do not include markdown fences or extra text.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}"
    )


def decide_initial_action(question: str) -> AgentDecision:
    prompt = _build_search_decision_prompt(question)
    response_text = generate_text(prompt)
    return _parse_decision(
        response_text,
        default_action=SEARCH,
        default_reason="The LLM returned invalid JSON, so retrieval is used as a safe fallback.",
    )


def evaluate_context(question: str, context: str) -> AgentDecision:
    prompt = _build_context_evaluation_prompt(question, context)
    response_text = generate_text(prompt)
    return _parse_decision(
        response_text,
        default_action=SEARCH,
        default_reason="The LLM returned invalid JSON, so another retrieval attempt is used as a safe fallback.",
    )