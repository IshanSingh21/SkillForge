"""SkillForge AI — Skill Extraction Service.

Hybrid NLP + embedding-based skill extraction.
Implementation planned for Milestone 3.
"""

from __future__ import annotations

from src.skillforge.models.resume import Skill


class SkillExtractor:
    """Extracts skills from text using NLP, embeddings, and optional LLM refinement."""

    def extract_skills(self, text: str, source: str = "resume") -> list[Skill]:
        """Extract skills from text."""
        raise NotImplementedError("SkillExtractor will be implemented in Milestone 3")
