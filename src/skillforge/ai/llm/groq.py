"""
SkillForge AI — Groq LLM Provider.

Calls Groq's high-speed LLM API (e.g. llama-3.1-70b-versatile) using an OpenAI-compatible
endpoint via httpx with token usage metrics and structured output support.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from src.skillforge.ai.llm.base import LLMProvider, LLMResponse
from src.skillforge.utils.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from src.skillforge.utils.logging import logger

T = TypeVar("T", bound=BaseModel)


class GroqProvider(LLMProvider):
    """Groq API implementation of LLMProvider interface."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.1-70b-versatile",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Groq Provider.

        Args:
            api_key: Groq API key.
            model_name: Model identifier (e.g. 'llama-3.1-70b-versatile').
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate text using Groq API."""
        if not self.api_key:
            raise LLMConnectionError(
                "Groq API key is not configured. Set GROQ_API_KEY in your .env file."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.API_URL, headers=headers, json=payload)

            if response.status_code == 429:
                raise LLMRateLimitError("Groq API rate limit exceeded. Please try again shortly.")

            if response.status_code != 200:
                raise LLMResponseError(
                    f"Groq API returned status {response.status_code}: {response.text}"
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMResponseError("Groq returned empty choices list in response.")

            first_choice = choices[0]
            content = first_choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                finish_reason=first_choice.get("finish_reason", "stop"),
            )

        except httpx.RequestError as e:
            logger.error("Groq network request failed", error=str(e))
            raise LLMConnectionError(f"Failed to connect to Groq API: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str = "",
    ) -> T:
        """Generate structured output adhering to a Pydantic model."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        structured_system = (
            f"{system_prompt}\n\n"
            f"You must respond ONLY with valid JSON conforming to this JSON Schema:\n{schema_json}"
        )

        response = self.generate(
            prompt=prompt,
            system_prompt=structured_system,
            temperature=0.2,
        )

        try:
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            parsed = json.loads(raw_content)
            return response_model.model_validate(parsed)

        except Exception as e:
            logger.error("Failed to parse Groq structured response", error=str(e), content=response.content)
            raise LLMResponseError(f"Failed to parse structured response into {response_model.__name__}: {e}") from e
