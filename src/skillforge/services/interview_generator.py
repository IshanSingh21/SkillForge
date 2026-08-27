"""SkillForge AI — Interview Question Generator.

LLM-powered interview question generation.
Implementation planned for Milestone 5.
"""

from __future__ import annotations

from src.skillforge.models.matching import MatchResult
from src.skillforge.models.roadmap import InterviewQuestionSet


class InterviewQuestionGenerator:
    """Generates tailored interview questions based on role and skill gaps."""

    def generate_questions(self, match_result: MatchResult, target_role: str) -> InterviewQuestionSet:
        """Generate interview questions for the target role."""
        raise NotImplementedError("InterviewQuestionGenerator will be implemented in Milestone 5")
