"""SkillForge AI — Groq LLM Provider.

Implementation planned for Milestone 2.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.skillforge.ai.llm.base import LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    """Groq API implementation of LLMProvider."""

    def __init__(self, api_key: str, model_name: str = "llama-3.1-70b-versatile") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        """Generate text using Groq."""
        raise NotImplementedError("GroqProvider will be implemented in Milestone 2")

    def generate_structured(self, prompt: str, response_model: type[BaseModel], system_prompt: str = "") -> BaseModel:
        """Generate structured output using Groq."""
        raise NotImplementedError("GroqProvider will be implemented in Milestone 2")
