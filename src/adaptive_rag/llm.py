from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL_NAME = "meta/llama-3.1-70b-instruct"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


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


def generate_text(prompt: str) -> str:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing. Add it to the .env file.")

    model_name = os.getenv("NVIDIA_MODEL_NAME", DEFAULT_MODEL_NAME)
    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL)

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("NVIDIA returned an empty response.")

    message = choices[0].message
    answer = getattr(message, "content", None)
    if not answer:
        raise RuntimeError("NVIDIA returned an empty message.")

    return answer.strip()


def generate_answer(question: str, context: str) -> str:
    prompt = build_prompt(question, context)
    return generate_text(prompt)