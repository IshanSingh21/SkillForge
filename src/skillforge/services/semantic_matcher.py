"""SkillForge AI — Semantic Matching Service.

Embedding-based resume-to-job-description matching.
Implementation planned for Milestone 3.
"""

from __future__ import annotations

from src.skillforge.models.matching import MatchResult
from src.skillforge.models.resume import ResumeAnalysis


class SemanticMatcher:
    """Computes semantic similarity between resume and job description."""

    def match(self, resume: ResumeAnalysis, job_description: str) -> MatchResult:
        """Analyze how well a resume matches a job description."""
        raise NotImplementedError("SemanticMatcher will be implemented in Milestone 3")
