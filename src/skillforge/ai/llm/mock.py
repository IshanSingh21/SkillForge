"""
SkillForge AI — Mock LLM Provider.

Provides a deterministic, configurable Mock LLM for offline testing of the RAG
pipeline and generation services without making network calls or requiring API keys.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from src.skillforge.ai.llm.base import LLMProvider, LLMResponse

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """
    Mock LLM backend for deterministic testing and development.

    Allows configuring canned responses, recording calls, and inspecting
    prompts and system instructions passed into the model.
    """

    def __init__(
        self,
        default_response: str = "This is a mock LLM response grounded in retrieved context.",
        model_name: str = "mock-model-v1",
    ) -> None:
        self.default_response = default_response
        self.model_name = model_name
        self.call_history: list[dict[str, Any]] = []
        self.custom_responses: dict[str, str] = {}
        self.should_raise: Exception | None = None

    def set_response_for_query(self, query_substr: str, response: str) -> None:
        """Map a query substring to a specific mock response."""
        self.custom_responses[query_substr.lower()] = response

    def set_error(self, error: Exception | None) -> None:
        """Configure the mock provider to raise an exception on next call."""
        self.should_raise = error

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a mock response, recording call parameters."""
        if self.should_raise:
            raise self.should_raise

        self.call_history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        response_text = self.default_response
        for query_substr, custom_resp in self.custom_responses.items():
            if query_substr in prompt.lower():
                response_text = custom_resp
                break

        return LLMResponse(
            content=response_text,
            model=self.model_name,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_text.split()), "total_tokens": len(prompt.split()) + len(response_text.split())},
            finish_reason="stop",
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str = "",
    ) -> T:
        """Generate a mock structured object conforming to response_model."""
        if self.should_raise:
            raise self.should_raise

        self.call_history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_model": response_model.__name__,
            }
        )

        # Build a minimal valid instance of response_model with default fields
        fields_data = {}
        for field_name, field_info in response_model.model_fields.items():
            if field_info.default is not None:
                fields_data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                fields_data[field_name] = field_info.default_factory()
            else:
                # Provide reasonable dummy value for required fields
                annotation = str(field_info.annotation)
                if "str" in annotation:
                    fields_data[field_name] = f"mock_{field_name}"
                elif "int" in annotation:
                    fields_data[field_name] = 1
                elif "float" in annotation:
                    fields_data[field_name] = 1.0
                elif "list" in annotation:
                    fields_data[field_name] = []
                elif "dict" in annotation:
                    fields_data[field_name] = {}
                else:
                    fields_data[field_name] = None

        return response_model(**fields_data)

    def get_last_prompt(self) -> str:
        """Return the most recently received prompt."""
        return self.call_history[-1]["prompt"] if self.call_history else ""

    def get_last_system_prompt(self) -> str:
        """Return the most recently received system prompt."""
        return self.call_history[-1]["system_prompt"] if self.call_history else ""
