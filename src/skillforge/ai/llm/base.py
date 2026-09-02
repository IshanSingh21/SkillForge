"""
SkillForge AI — Abstract LLM Provider Interface.

Defines the contract that all LLM backends must implement.
Provides uniform text and structured generation across Gemini, Groq,
Ollama, and Mock providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    content: str = Field(..., description="Generated text response")
    model: str = Field(default="", description="Model identifier that generated the response")
    usage: dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")
    finish_reason: str = Field(default="stop", description="Reason generation finished")


class LLMProvider(ABC):
    """
    Abstract base class defining the contract for all LLM backends.

    Any provider (Gemini, Groq, Ollama, Mock) implements this interface,
    ensuring the core RAG and service layer is completely decoupled from
    any specific API or vendor.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt or formatted query.
            system_prompt: System instructions guiding the model behavior.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse containing the generated text and metadata.
        """
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str = "",
    ) -> T:
        """
        Generate a structured response conforming to a Pydantic model.

        Args:
            prompt: User prompt.
            response_model: Pydantic model class to validate and instantiate.
            system_prompt: System instructions.

        Returns:
            Instantiated and validated Pydantic model object.
        """
        ...
