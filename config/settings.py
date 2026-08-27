"""
SkillForge AI — Application Configuration.

Loads all configuration from environment variables using Pydantic Settings.
Supports .env files for local development. Provides typed, validated access
to every configurable parameter in the application.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.llm_provider)
"""

from __future__ import annotations

from functools import lru_cache
from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderType(str, Enum):
    """Supported LLM provider backends."""

    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Central configuration for SkillForge AI.

    All values are loaded from environment variables (or a .env file).
    Defaults are provided for non-sensitive settings so the app can
    start with minimal configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = Field(default="SkillForge AI", description="Application display name")
    app_version: str = Field(default="0.1.0", description="Application version")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging verbosity")

    # --- LLM Provider ---
    llm_provider: LLMProviderType = Field(
        default=LLMProviderType.GEMINI,
        description="Which LLM backend to use",
    )
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key")

    # --- Model Names ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model identifier",
    )
    groq_model: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq model identifier",
    )

    # --- RAG Configuration ---
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4096,
        description="Max characters per text chunk",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=512,
        description="Overlap between consecutive chunks",
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for RAG",
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for skill matching",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v: int, info) -> int:
        """Ensure overlap is strictly less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @property
    def active_api_key(self) -> str:
        """Return the API key for the currently selected LLM provider."""
        key_map = {
            LLMProviderType.GEMINI: self.gemini_api_key,
            LLMProviderType.GROQ: self.groq_api_key,
            LLMProviderType.OLLAMA: "",  # Ollama runs locally, no key needed
        }
        return key_map.get(self.llm_provider, "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Uses lru_cache so the .env file is only read once per process.
    Call `get_settings.cache_clear()` in tests to reset.
    """
    return Settings()
