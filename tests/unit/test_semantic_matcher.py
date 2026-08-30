"""
Tests for src.skillforge.services.semantic_matcher — Semantic Matching.

Tests cover:
- Full match pipeline (skills → score + classifications)
- Strong/partial/gap classification correctness
- Score computation and range
- Edge cases (no resume skills, no job skills)
- Text-to-text convenience matching
- Match result structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.ai.embeddings import EmbeddingEngine
from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.resume import Skill, SkillCategory
from src.skillforge.services.semantic_matcher import SemanticMatcher


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine for all tests."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def matcher(engine: EmbeddingEngine) -> SemanticMatcher:
    """Shared matcher for all tests."""
    return SemanticMatcher(embedding_engine=engine)


def _skill(name: str, category: SkillCategory = SkillCategory.TECHNICAL) -> Skill:
    """Helper to create a Skill quickly."""
    return Skill(name=name, category=category, confidence=1.0)


# ── Sample Skill Sets ──────────────────────────────────────────────────

RESUME_SKILLS = [
    _skill("Python", SkillCategory.LANGUAGE),
    _skill("JavaScript", SkillCategory.LANGUAGE),
    _skill("React", SkillCategory.FRAMEWORK),
    _skill("Docker", SkillCategory.TOOL),
    _skill("PostgreSQL", SkillCategory.TOOL),
    _skill("AWS", SkillCategory.TOOL),
    _skill("Leadership", SkillCategory.SOFT),
]

JOB_SKILLS = [
    _skill("Python", SkillCategory.LANGUAGE),
    _skill("Go", SkillCategory.LANGUAGE),
    _skill("Kubernetes", SkillCategory.TOOL),
    _skill("Docker", SkillCategory.TOOL),
    _skill("PostgreSQL", SkillCategory.TOOL),
    _skill("AWS", SkillCategory.TOOL),
    _skill("Communication", SkillCategory.SOFT),
]


# ── Full Match Pipeline Tests ──────────────────────────────────────────


class TestMatchPipeline:
    """Tests for the full semantic matching pipeline."""

    def test_match_produces_result(self, matcher: SemanticMatcher):
        """Should return a MatchResult with all fields populated."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        assert isinstance(result, MatchResult)
        assert result.overall_score >= 0.0
        assert result.overall_score <= 100.0
        assert result.summary is not None

    def test_exact_skills_are_strong_matches(self, matcher: SemanticMatcher):
        """Skills present in both lists should be strong matches."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        strong_job_names = {m.job_skill.name for m in result.matched_skills}
        # Python, Docker, PostgreSQL, AWS are in both lists
        assert "Python" in strong_job_names
        assert "Docker" in strong_job_names

    def test_missing_skills_identified(self, matcher: SemanticMatcher):
        """Skills in JD but not in resume should be gaps or partial matches."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        # "Go" is in JD but not in resume — should be gap or partial
        all_gap_names = {g.skill.name for g in result.missing_skills}
        all_partial_names = {m.job_skill.name for m in result.partial_matches}
        all_strong_names = {m.job_skill.name for m in result.matched_skills}

        # Go is not in resume at all — should not be a strong match
        # It could be a partial (since "Go" as a language might partially
        # match "Python" or other languages) or a gap
        assert "Go" in all_gap_names or "Go" in all_partial_names

    def test_all_job_skills_classified(self, matcher: SemanticMatcher):
        """Every job skill should appear in exactly one category."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        total = (
            len(result.matched_skills)
            + len(result.partial_matches)
            + len(result.missing_skills)
        )
        assert total == len(JOB_SKILLS)

    def test_score_breakdown_present(self, matcher: SemanticMatcher):
        """Score breakdown dict should be populated."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        assert "overall" in result.score_breakdown
        assert "strong_component" in result.score_breakdown
        assert "coverage" in result.score_breakdown
        assert "matched_count" in result.score_breakdown


# ── Score Tests ────────────────────────────────────────────────────────


class TestScoring:
    """Tests for match score computation."""

    def test_perfect_match_has_high_score(self, matcher: SemanticMatcher):
        """When all skills match, score should be high."""
        skills = [
            _skill("Python"),
            _skill("Docker"),
            _skill("PostgreSQL"),
        ]
        result = matcher.match(skills, skills)

        assert result.overall_score > 60.0

    def test_no_overlap_has_low_score(self, matcher: SemanticMatcher):
        """Completely different skill sets should have a low score."""
        resume = [_skill("Italian Cooking"), _skill("Painting")]
        job = [_skill("Python"), _skill("Kubernetes")]

        result = matcher.match(resume, job)
        assert result.overall_score < 30.0

    def test_score_in_valid_range(self, matcher: SemanticMatcher):
        """Score should always be between 0 and 100."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)
        assert 0.0 <= result.overall_score <= 100.0


# ── Edge Cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_job_skills(self, matcher: SemanticMatcher):
        """Empty job skills should produce a zero-score result."""
        result = matcher.match(RESUME_SKILLS, [])

        assert result.overall_score == 0.0
        assert "No job skills" in result.summary

    def test_no_resume_skills(self, matcher: SemanticMatcher):
        """Empty resume skills should produce a zero-score result."""
        result = matcher.match([], JOB_SKILLS)

        assert result.overall_score == 0.0
        assert len(result.missing_skills) == len(JOB_SKILLS)

    def test_single_skill_match(self, matcher: SemanticMatcher):
        """Should work with single-skill lists."""
        result = matcher.match(
            [_skill("Python")],
            [_skill("Python")],
        )
        assert result.overall_score > 50.0
        assert len(result.matched_skills) == 1


# ── Text Matching Tests ────────────────────────────────────────────────


class TestTextMatching:
    """Tests for the text-to-text convenience method."""

    def test_similar_texts_high_score(self, matcher: SemanticMatcher):
        """Similar texts should produce high similarity."""
        score = matcher.match_texts(
            "Senior Python backend developer",
            "Experienced Python backend engineer",
        )
        assert score > 0.7

    def test_unrelated_texts_low_score(self, matcher: SemanticMatcher):
        """Unrelated texts should produce low similarity."""
        score = matcher.match_texts(
            "Python machine learning engineer",
            "Italian pasta cooking recipe",
        )
        assert score < 0.3

    def test_score_is_non_negative(self, matcher: SemanticMatcher):
        """Text matching score should always be >= 0."""
        score = matcher.match_texts("Python", "Italian cooking")
        assert score >= 0.0


# ── Match Result Structure Tests ───────────────────────────────────────


class TestMatchResultStructure:
    """Tests for the structure of SkillMatch and SkillGap objects."""

    def test_skill_match_has_similarity(self, matcher: SemanticMatcher):
        """SkillMatch objects should have a similarity score."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        for m in result.matched_skills:
            assert isinstance(m, SkillMatch)
            assert m.similarity >= 0.0
            assert m.similarity <= 1.0
            assert m.resume_skill is not None
            assert m.job_skill is not None

    def test_skill_gap_has_skill(self, matcher: SemanticMatcher):
        """SkillGap objects should reference the missing job skill."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        for g in result.missing_skills:
            assert isinstance(g, SkillGap)
            assert g.skill is not None
            assert len(g.skill.name) > 0

    def test_summary_is_readable(self, matcher: SemanticMatcher):
        """Summary should contain score and match info."""
        result = matcher.match(RESUME_SKILLS, JOB_SKILLS)

        assert "Match Score" in result.summary
        assert "Strong matches" in result.summary
