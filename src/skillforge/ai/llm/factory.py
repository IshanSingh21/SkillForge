"""SkillForge AI — LLM Provider Factory.

Creates the appropriate LLM provider based on configuration.
Implementation planned for Milestone 2.
"""

from __future__ import annotations

from src.skillforge.ai.llm.base import LLMProvider


def create_llm_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider from config."""
    raise NotImplementedError("LLM factory will be implemented in Milestone 2")
