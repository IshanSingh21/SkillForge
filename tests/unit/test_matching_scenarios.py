"""
Tests for Milestone 5 — Resume-Job Matching & Explainable Scoring Scenarios.

Tests cover:
- Multi-factor explainable scoring calculation & mathematical bounds
- High match scenario (Senior Backend Engineer matching Senior Backend JD)
- Experience gap scenario (Junior candidate applying to Senior role)
- Stack mismatch / partial overlap scenario (Frontend dev applying to Backend role)
- Required vs Preferred skill classification and coverage weighting
- Experience years and seniority extraction utilities
- End-to-end match_resume_to_job API with text and ResumeAnalysis
- Summary generation and explainability breakdown
- Edge cases and boundary conditions
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
from src.skillforge.models.resume import ResumeAnalysis, ResumeSection, Skill, SkillCategory
from src.skillforge.services.semantic_matcher import SemanticMatcher
from src.skillforge.services.skill_extractor import SkillExtractor


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Shared embedding engine for scenario tests."""
    return EmbeddingEngine(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def extractor() -> SkillExtractor:
    """Shared skill extractor."""
    return SkillExtractor(use_nlp=True)


@pytest.fixture(scope="module")
def matcher(engine: EmbeddingEngine, extractor: SkillExtractor) -> SemanticMatcher:
    """Shared matcher configured with embedding engine and extractor."""
    return SemanticMatcher(embedding_engine=engine, skill_extractor=extractor)


# ── Sample Data Fixtures ───────────────────────────────────────────────

SENIOR_BACKEND_RESUME = """
Jane Smith — Senior Backend Engineer
jane.smith@example.com | San Francisco, CA

SUMMARY
Senior Software Engineer with 6+ years of experience designing and deploying scalable
cloud-native microservices using Python, FastAPI, PostgreSQL, Docker, and AWS.

EXPERIENCE
Senior Backend Engineer — CloudScale Inc (2021–Present)
- Architected RESTful microservices in Python (FastAPI/Django) handling 50M requests/day.
- Deployed distributed services on AWS (ECS, EKS, RDS, S3, Lambda) with Terraform.
- Optimized PostgreSQL database queries and managed Redis caching layers.
- Led technical design reviews and mentored 4 junior developers.

Backend Developer — DataTech Corp (2018–2021)
- Built Python data pipelines and REST APIs using PostgreSQL and Redis.
- Implemented CI/CD pipelines using GitHub Actions and Docker.
- Collaborated in an Agile/Scrum team.

SKILLS
Languages: Python, SQL, Go, Bash
Frameworks: FastAPI, Django, Flask
Databases & Tools: PostgreSQL, Redis, MongoDB, Docker, Kubernetes, Terraform, Git, CI/CD
Cloud: AWS (EC2, S3, RDS, ECS, Lambda)
"""

JUNIOR_FRONTEND_RESUME = """
Alex Lee — Junior Frontend Developer
alex.lee@example.com

SUMMARY
Passionate Junior Frontend Developer with 1 year of experience building responsive
web interfaces with JavaScript, React, HTML5, and CSS3.

EXPERIENCE
Junior Web Developer — WebStudio (2023–Present)
- Developed user interfaces using React and Tailwind CSS.
- Integrated REST APIs with React frontend components.
- Maintained code repositories using Git and GitHub.

SKILLS
Languages: JavaScript, TypeScript, HTML, CSS
Frameworks: React, Next.js, Tailwind CSS
Tools: Git, Webpack, npm
"""

SENIOR_BACKEND_JD = """
Senior Backend Engineer — CloudTech Solutions

About the Role:
We are seeking a Senior Backend Engineer to build our scalable cloud infrastructure.

Requirements:
- 5+ years of software engineering experience in backend development
- Strong proficiency in Python and Go
- Proven experience with microservices architecture and REST APIs
- Deep knowledge of PostgreSQL and Redis
- Hands-on experience with Docker, Kubernetes, and AWS cloud services
- Strong communication and mentoring skills

Nice to have:
- Experience with GraphQL
- Familiarity with Kafka or event-driven systems
- Machine learning pipeline experience
"""


# ── Scenario Tests ─────────────────────────────────────────────────────


class TestMatchingScenarios:
    """Test realistic candidate-to-job matching scenarios."""

    def test_senior_backend_strong_match(self, matcher: SemanticMatcher):
        """A senior backend engineer should score highly against a senior backend role."""
        result = matcher.match_resume_to_job(
            resume=SENIOR_BACKEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
            target_role="Senior Backend Engineer",
        )

        assert isinstance(result, MatchResult)
        assert result.overall_score >= 70.0, f"Expected high score, got {result.overall_score}"

        # Matched skills should include key technologies
        matched_names = {m.job_skill.name for m in result.matched_skills}
        assert "Python" in matched_names
        assert "Docker" in matched_names
        assert "PostgreSQL" in matched_names

        # Required coverage should be high
        assert result.required_skills_coverage >= 70.0

        # Experience should be satisfied
        assert result.experience_analysis.get("years_detected") >= 5
        assert result.experience_analysis.get("experience_score") >= 85.0

    def test_junior_frontend_lower_score_on_senior_backend(self, matcher: SemanticMatcher):
        """A junior frontend developer should receive a significantly lower score on a senior backend role."""
        result = matcher.match_resume_to_job(
            resume=JUNIOR_FRONTEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
            target_role="Senior Backend Engineer",
        )

        assert result.overall_score < 55.0, f"Expected lower score for mismatch, got {result.overall_score}"

        # Gaps should include core backend requirements
        gap_names = {g.skill.name for g in result.missing_skills}
        assert any(s in gap_names for s in ["PostgreSQL", "Redis", "AWS", "Kubernetes", "Go"])

        # Experience analysis should flag seniority gap
        assert result.experience_analysis.get("candidate_seniority") in ("junior", "entry", "Not specified")

    def test_score_ranking_consistency(self, matcher: SemanticMatcher):
        """Senior backend candidate must score strictly higher than junior frontend candidate for senior backend JD."""
        senior_result = matcher.match_resume_to_job(
            resume=SENIOR_BACKEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
            target_role="Senior Backend Engineer",
        )
        junior_result = matcher.match_resume_to_job(
            resume=JUNIOR_FRONTEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
            target_role="Senior Backend Engineer",
        )

        assert senior_result.overall_score > junior_result.overall_score + 25.0


# ── Requirement Importance Classification Tests ────────────────────────


class TestRequirementClassification:
    """Test classification of Required vs. Nice-to-Have skills."""

    def test_classifies_required_and_nice_to_have_skills(self, matcher: SemanticMatcher):
        """Skills in 'Requirements' should be required; skills in 'Nice to have' should be nice_to_have."""
        skills = matcher.extractor.extract_skills(SENIOR_BACKEND_JD)
        classified = matcher.classify_job_skill_importance(SENIOR_BACKEND_JD, skills)

        importance_dict = {s.name: imp for s, imp in classified}

        # Required skills
        assert importance_dict.get("Python") == "required"
        assert importance_dict.get("PostgreSQL") == "required"

        # Nice to have skills
        if "GraphQL" in importance_dict:
            assert importance_dict.get("GraphQL") == "nice_to_have"
        if "Apache Kafka" in importance_dict:
            assert importance_dict.get("Apache Kafka") == "nice_to_have"

    def test_missing_required_skills_flagged_in_gaps(self, matcher: SemanticMatcher):
        """Gaps should preserve required vs preferred classification."""
        result = matcher.match_resume_to_job(
            resume=JUNIOR_FRONTEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
        )

        req_gaps = result.required_gaps
        assert len(req_gaps) > 0
        assert all(g.importance == "required" for g in req_gaps)


# ── Experience & Seniority Extraction Tests ────────────────────────────


class TestExperienceExtraction:
    """Test extraction of years of experience and seniority levels."""

    def test_extracts_required_years(self, matcher: SemanticMatcher):
        """Should detect '5+ years' in JD."""
        assert matcher.extract_required_years(SENIOR_BACKEND_JD) == 5.0

    def test_extracts_years_range(self, matcher: SemanticMatcher):
        """Should detect range '3-5 years' and take the lower bound."""
        assert matcher.extract_required_years("Requires 3-5 years of experience in Python.") == 3.0

    def test_extracts_candidate_years_from_text(self, matcher: SemanticMatcher):
        """Should detect '6+ years' in candidate summary."""
        assert matcher.extract_candidate_years(SENIOR_BACKEND_RESUME) == 6.0

    def test_extracts_seniority_levels(self, matcher: SemanticMatcher):
        """Should correctly detect Senior, Lead, Junior levels."""
        assert matcher.extract_seniority_level("Senior Backend Engineer") == "senior"
        assert matcher.extract_seniority_level("Lead Architect & Staff Developer") in ("lead", "principal")
        assert matcher.extract_seniority_level("Junior Frontend Developer") == "junior"


# ── Explainability & Score Breakdown Tests ─────────────────────────────


class TestExplainability:
    """Test score breakdown structure and explainability."""

    def test_score_breakdown_contains_all_components(self, matcher: SemanticMatcher):
        """Breakdown dictionary should contain all 4 sub-scores and weights."""
        result = matcher.match_resume_to_job(
            resume=SENIOR_BACKEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
        )

        breakdown = result.score_breakdown
        assert "overall" in breakdown
        assert "skill_match_score" in breakdown
        assert "required_skills_coverage" in breakdown
        assert "semantic_similarity_score" in breakdown
        assert "experience_alignment_score" in breakdown
        assert "weights" in breakdown

    def test_formula_weights_sum_to_one(self, matcher: SemanticMatcher):
        """The 4 weights in the formula must sum to 1.0."""
        assert abs(
            matcher.WEIGHT_SKILL_MATCH
            + matcher.WEIGHT_REQUIRED_COVERAGE
            + matcher.WEIGHT_SEMANTIC_SIMILARITY
            + matcher.WEIGHT_EXPERIENCE_ALIGNMENT
            - 1.0
        ) < 1e-6

    def test_summary_includes_score_breakdown_text(self, matcher: SemanticMatcher):
        """The human-readable summary should explain the sub-scores."""
        result = matcher.match_resume_to_job(
            resume=SENIOR_BACKEND_RESUME,
            job_description=SENIOR_BACKEND_JD,
        )

        assert "Score Breakdown:" in result.summary
        assert "Skill Match Quality:" in result.summary
        assert "Required Skills Coverage:" in result.summary


# ── ResumeAnalysis Object Integration Tests ────────────────────────────


class TestResumeAnalysisIntegration:
    """Test matching with ResumeAnalysis input object."""

    def test_matches_resume_analysis_instance(self, matcher: SemanticMatcher):
        """match_resume_to_job should accept a ResumeAnalysis object."""
        analysis = ResumeAnalysis(
            raw_text=SENIOR_BACKEND_RESUME,
            cleaned_text=SENIOR_BACKEND_RESUME,
            sections=[
                ResumeSection(title="Summary", content="Senior engineer with 6+ years..."),
                ResumeSection(title="Experience", content="Senior Backend Engineer at CloudScale..."),
            ],
            skills=matcher.extractor.extract_skills(SENIOR_BACKEND_RESUME),
            chunks=[],
        )

        result = matcher.match_resume_to_job(
            resume=analysis,
            job_description=SENIOR_BACKEND_JD,
        )

        assert isinstance(result, MatchResult)
        assert result.overall_score > 65.0
        assert len(result.matched_skills) > 0


# ── Edge Cases ─────────────────────────────────────────────────────────


class TestScenarioEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_job_description(self, matcher: SemanticMatcher):
        """Empty job description returns 0 score gracefully."""
        result = matcher.match_resume_to_job(resume=SENIOR_BACKEND_RESUME, job_description="")
        assert result.overall_score == 0.0

    def test_empty_resume(self, matcher: SemanticMatcher):
        """Empty resume returns 0 score gracefully."""
        result = matcher.match_resume_to_job(resume="", job_description=SENIOR_BACKEND_JD)
        assert result.overall_score == 0.0
        assert len(result.missing_skills) > 0

    def test_score_bounded_between_0_and_100(self, matcher: SemanticMatcher):
        """Score should strictly never exceed 100 or fall below 0."""
        result = matcher.match_resume_to_job(
            resume=SENIOR_BACKEND_RESUME,
            job_description=SENIOR_BACKEND_RESUME,  # Perfect self-match
        )
        assert 0.0 <= result.overall_score <= 100.0
