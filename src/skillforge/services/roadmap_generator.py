"""SkillForge AI — Learning Roadmap Generator.

LLM-powered personalized learning plan generation.
Implementation planned for Milestone 5.
"""

from __future__ import annotations

from src.skillforge.models.matching import MatchResult
from src.skillforge.models.roadmap import LearningRoadmap


class RoadmapGenerator:
    """Generates personalized learning roadmaps from skill-gap analysis."""

    def generate_roadmap(self, match_result: MatchResult, target_role: str) -> LearningRoadmap:
        """Generate a learning roadmap based on skill gaps."""
        raise NotImplementedError("RoadmapGenerator will be implemented in Milestone 5")
