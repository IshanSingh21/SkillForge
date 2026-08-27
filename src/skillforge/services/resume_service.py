"""SkillForge AI — Resume Processing Service.

Orchestrates the full resume processing pipeline.
Implementation planned for Milestones 2-3.
"""

from __future__ import annotations

from src.skillforge.models.resume import ResumeAnalysis


class ResumeService:
    """Orchestrates resume upload, parsing, cleaning, chunking, and skill extraction."""

    def process_resume(self, pdf_bytes: bytes, filename: str = "") -> ResumeAnalysis:
        """Process a resume PDF through the full pipeline."""
        raise NotImplementedError("ResumeService will be implemented in Milestone 2")
