"""
SkillForge AI — LLM Provider Factory.

Instantiates and configures the appropriate LLMProvider implementation
(Gemini, Groq, Mock) based on application configuration or explicit parameters.
"""

from __future__ import annotations

from typing import Any

from config.settings import LLMProviderType, get_settings
from src.skillforge.ai.llm.base import LLMProvider
from src.skillforge.ai.llm.gemini import GeminiProvider
from src.skillforge.ai.llm.groq import GroqProvider
from src.skillforge.ai.llm.mock import MockLLMProvider
from src.skillforge.utils.exceptions import ConfigurationError
from src.skillforge.utils.logging import logger


def create_llm_provider(
    provider_name: str | LLMProviderType | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to create a configured LLMProvider instance.

    Args:
        provider_name: 'gemini', 'groq', 'mock', or None (reads from settings).
        **kwargs: Optional overrides for api_key, model_name, etc.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ConfigurationError: If an unknown provider type is requested.
    """
    settings = get_settings()

    if provider_name is None:
        selected = settings.llm_provider
    elif isinstance(provider_name, LLMProviderType):
        selected = provider_name
    else:
        selected_str = str(provider_name).lower().strip()
        if selected_str == "mock":
            return MockLLMProvider(
                default_response=kwargs.get("default_response", "Grounded response from mock LLM."),
                model_name=kwargs.get("model_name", "mock-model"),
            )
        try:
            selected = LLMProviderType(selected_str)
        except ValueError:
            raise ConfigurationError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Supported providers are: gemini, groq, mock."
            )

    logger.info("Instantiating LLM provider", provider=selected.value)

    if selected == LLMProviderType.GEMINI:
        api_key = kwargs.get("api_key") or settings.gemini_api_key
        model_name = kwargs.get("model_name") or settings.gemini_model
        return GeminiProvider(api_key=api_key, model_name=model_name)

    elif selected == LLMProviderType.GROQ:
        api_key = kwargs.get("api_key") or settings.groq_api_key
        model_name = kwargs.get("model_name") or settings.groq_model
        return GroqProvider(api_key=api_key, model_name=model_name)

    elif selected == LLMProviderType.OLLAMA:
        # Fallback to mock if Ollama not explicitly implemented
        return MockLLMProvider(
            default_response="Response from local Ollama model.",
            model_name="ollama-local",
        )

    raise ConfigurationError(f"Unhandled provider type: {selected}")
