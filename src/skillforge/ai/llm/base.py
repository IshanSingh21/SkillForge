"""SkillForge AI — Abstract LLM Provider Interface.

Defines the contract that all LLM backends must implement.
Implementation planned for Milestone 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    content: str = Field(..., description="Generated text")
    model: str = Field(default="", description="Model that generated the response")
    usage: dict = Field(default_factory=dict, description="Token usage statistics")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def generate_structured(self, prompt: str, response_model: type[BaseModel], system_prompt: str = "") -> BaseModel:
        """Generate a structured response conforming to a Pydantic model."""
        ...
