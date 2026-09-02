"""
SkillForge AI — Google Gemini LLM Provider.

Calls Google Gemini API (e.g. gemini-1.5-flash) using httpx with robust error handling,
system instructions, token usage tracking, and structured Pydantic output generation.
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


class GeminiProvider(LLMProvider):
    """Google Gemini implementation of LLMProvider interface."""

    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Gemini LLM Provider.

        Args:
            api_key: Google Gemini API key.
            model_name: Model identifier (default: 'gemini-1.5-flash').
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
        """Generate text using Google Gemini API."""
        if not self.api_key:
            raise LLMConnectionError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in your .env file."
            )

        url = f"{self.API_BASE_URL}/{self.model_name}:generateContent?key={self.api_key}"

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)

            if response.status_code == 429:
                raise LLMRateLimitError("Gemini API rate limit exceeded. Please try again shortly.")

            if response.status_code != 200:
                error_detail = response.text
                raise LLMResponseError(
                    f"Gemini API returned status {response.status_code}: {error_detail}"
                )

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMResponseError("Gemini returned empty candidates list in response.")

            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            if not parts:
                raise LLMResponseError("Gemini returned candidate with empty parts.")

            content_text = parts[0].get("text", "")
            usage = data.get("usageMetadata", {})

            return LLMResponse(
                content=content_text,
                model=self.model_name,
                usage=usage,
                finish_reason=first_candidate.get("finishReason", "stop"),
            )

        except httpx.RequestError as e:
            logger.error("Gemini network request failed", error=str(e))
            raise LLMConnectionError(f"Failed to connect to Gemini API: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str = "",
    ) -> T:
        """Generate structured JSON adhering to the specified Pydantic schema."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        structured_system = (
            f"{system_prompt}\n\n"
            f"You must respond ONLY with valid JSON conforming to this JSON Schema:\n{schema_json}"
        )

        response = self.generate(
            prompt=prompt,
            system_prompt=structured_system,
            temperature=0.2,  # Lower temperature for schema adherence
        )

        try:
            raw_content = response.content.strip()
            # Clean markdown code fences if model returned ```json ... ```
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
            logger.error("Failed to parse Gemini structured response", error=str(e), content=response.content)
            raise LLMResponseError(f"Failed to parse structured response into {response_model.__name__}: {e}") from e
