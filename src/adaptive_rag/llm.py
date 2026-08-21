from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError
from openai import APITimeoutError
from openai import InternalServerError
from openai import OpenAI
from openai import RateLimitError


DEFAULT_MODEL_NAME = "gemini-3.7-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def build_prompt(question: str, context: str) -> str:
    if not context.strip():
        return (
            "You are a careful assistant for a retrieval-augmented generation system.\n"
            "Answer the question directly and concisely.\n"
            "Do not invent unsupported facts.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return (
        "You are a careful assistant for a retrieval-augmented generation system.\n"
        "Answer using only the supplied retrieved context.\n"
        "Do not invent facts that are not supported by the context.\n"
        "If the context is insufficient, explicitly say that the available context is insufficient.\n"
        "Give a concise answer.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_text(prompt: str, max_retries: int = 1) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

    base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_BASE_URL)
    model_name = os.getenv("GEMINI_MODEL_NAME", DEFAULT_MODEL_NAME)

    client = OpenAI(api_key=api_key, base_url=base_url)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("Gemini returned an empty response.")
            message = choices[0].message
            answer = getattr(message, "content", None)
            if not answer:
                raise RuntimeError("Gemini returned an empty message.")
            return answer.strip()
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as error:
            last_error = error
            if attempt >= max_retries:
                break
            time.sleep(5.0)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed.")


def generate_answer(question: str, context: str) -> str:
    prompt = build_prompt(question, context)
    return generate_text(prompt)