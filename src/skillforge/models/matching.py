"""
SkillForge AI — Matching Data Models.

Pydantic models for semantic matching results: individual skill matches,
skill gaps, and the overall match result with score breakdown.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.skillforge.models.resume import Skill


class SkillMatch(BaseModel):
    """A pair of matched skills with their semantic similarity score."""

    resume_skill: Skill = Field(..., description="Skill from the resume")
    job_skill: Skill = Field(..., description="Corresponding skill from the job description")
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the two skill embeddings",
    )


class SkillGap(BaseModel):
    """A skill required by the job but missing or weak in the resume."""

    skill: Skill = Field(..., description="The missing or weak skill")
    importance: str = Field(
        default="required",
        description="How important this skill is: 'required', 'preferred', 'nice_to_have'",
    )
    closest_resume_skill: str = Field(
        default="",
        description="Most similar skill found in the resume (if any)",
    )
    closest_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity of the closest resume skill",
    )
    recommendation: str = Field(
        default="",
        description="Brief recommendation for closing this gap",
    )


class MatchResult(BaseModel):
    """
    Complete output of the semantic matching pipeline.

    Contains the overall match score, matched and missing skills,
    partial matches, and a breakdown of how the score was computed.
    """

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Weighted match score (0–100)",
    )
    matched_skills: list[SkillMatch] = Field(
        default_factory=list,
        description="Skills that matched above the similarity threshold",
    )
    missing_skills: list[SkillGap] = Field(
        default_factory=list,
        description="Skills required by the job but absent in the resume",
    )
    partial_matches: list[SkillMatch] = Field(
        default_factory=list,
        description="Skills with moderate similarity (between thresholds)",
    )
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Component scores (e.g., {'skill_match': 65, 'partial_bonus': 12, ...})",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of the match analysis",
    )

    @property
    def match_percentage(self) -> str:
        """Return the score as a formatted percentage string."""
        return f"{self.overall_score:.1f}%"

    @property
    def total_job_skills(self) -> int:
        """Return the total number of job skills analyzed."""
        return len(self.matched_skills) + len(self.missing_skills) + len(self.partial_matches)
