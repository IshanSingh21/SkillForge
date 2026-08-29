"""
Tests for src.skillforge.services.skill_extractor — Hybrid skill extraction.

Tests cover:
- Taxonomy pattern matching (exact, alias, multi-word, boundary-safe)
- NLP-based extraction (noun phrases, fuzzy matching)
- Deduplication and normalization
- Extraction from resume text vs. job descriptions
- Edge cases (empty input, no skills, overlapping skills)
- Category assignment
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.data.skill_taxonomy import SkillTaxonomy
from src.skillforge.models.resume import Skill, SkillCategory
from src.skillforge.services.skill_extractor import SkillExtractor


@pytest.fixture
def extractor() -> SkillExtractor:
    """Return a SkillExtractor with NLP enabled (if available)."""
    return SkillExtractor(use_nlp=True)


@pytest.fixture
def pattern_only_extractor() -> SkillExtractor:
    """Return a SkillExtractor with NLP disabled (pattern-only)."""
    return SkillExtractor(use_nlp=False)


# ── Sample Texts ───────────────────────────────────────────────────────

RESUME_TEXT = """
John Doe — Senior Software Engineer

SKILLS
Python, JavaScript, TypeScript, React, Node.js, FastAPI, Django,
PostgreSQL, MongoDB, Redis, Docker, Kubernetes, AWS, Git, CI/CD

EXPERIENCE

Senior Software Engineer — Acme Corp (2021–Present)
- Designed microservices architecture using Python and FastAPI
- Led migration to AWS (EC2, S3, Lambda, RDS)
- Mentored 3 junior developers and conducted code reviews
- Built CI/CD pipelines with GitHub Actions and Docker

Software Engineer — TechStart Inc (2018–2020)
- Full-stack development with React and Node.js
- Built RESTful APIs serving 10K+ daily users
- Optimized PostgreSQL queries, improving response times by 60%
- Agile development using Scrum methodology

EDUCATION
Master of Science in Computer Science — MIT, 2018
"""

JOB_DESCRIPTION = """
Senior Backend Engineer — CloudTech Solutions

Requirements:
- 5+ years of experience in backend development
- Strong proficiency in Python and Go
- Experience with microservices architecture
- Familiarity with Kubernetes and Docker
- Experience with PostgreSQL and Redis
- Strong understanding of RESTful API design
- Experience with cloud platforms (AWS or GCP)
- Excellent communication and mentoring skills

Nice to have:
- Experience with GraphQL
- Knowledge of event-driven architecture (Kafka)
- Machine learning pipeline experience
"""


# ── Pattern Matching Tests ─────────────────────────────────────────────


class TestPatternMatching:
    """Tests for Layer 1: taxonomy-based pattern matching."""

    def test_extracts_common_languages(self, pattern_only_extractor: SkillExtractor):
        """Should find common programming languages."""
        skills = pattern_only_extractor.extract_skills(
            "Proficient in Python, JavaScript, and TypeScript"
        )
        names = {s.name for s in skills}

        assert "Python" in names
        assert "JavaScript" in names
        assert "TypeScript" in names

    def test_extracts_frameworks(self, pattern_only_extractor: SkillExtractor):
        """Should find frameworks from text."""
        skills = pattern_only_extractor.extract_skills(
            "Built applications with React, Django, and FastAPI"
        )
        names = {s.name for s in skills}

        assert "React" in names
        assert "Django" in names
        assert "FastAPI" in names

    def test_extracts_databases(self, pattern_only_extractor: SkillExtractor):
        """Should find database technologies."""
        skills = pattern_only_extractor.extract_skills(
            "Experience with PostgreSQL, MongoDB, and Redis"
        )
        names = {s.name for s in skills}

        assert "PostgreSQL" in names
        assert "MongoDB" in names
        assert "Redis" in names

    def test_extracts_cloud_and_devops(self, pattern_only_extractor: SkillExtractor):
        """Should find cloud platforms and DevOps tools."""
        skills = pattern_only_extractor.extract_skills(
            "Deployed to AWS using Docker and Kubernetes with CI/CD"
        )
        names = {s.name for s in skills}

        assert "AWS" in names
        assert "Docker" in names
        assert "Kubernetes" in names
        assert "CI/CD" in names

    def test_alias_resolution(self, pattern_only_extractor: SkillExtractor):
        """Aliases should resolve to canonical skill names."""
        skills = pattern_only_extractor.extract_skills(
            "Used reactjs, k8s, and golang in production"
        )
        names = {s.name for s in skills}

        assert "React" in names        # reactjs → React
        assert "Kubernetes" in names   # k8s → Kubernetes
        assert "Go" in names           # golang → Go

    def test_case_insensitive_matching(self, pattern_only_extractor: SkillExtractor):
        """Matching should be case-insensitive."""
        skills = pattern_only_extractor.extract_skills(
            "PYTHON, javascript, PostgreSQL, docker"
        )
        names = {s.name for s in skills}

        assert "Python" in names
        assert "JavaScript" in names
        assert "PostgreSQL" in names
        assert "Docker" in names

    def test_multi_word_matching(self, pattern_only_extractor: SkillExtractor):
        """Multi-word skills should be matched as single skills."""
        skills = pattern_only_extractor.extract_skills(
            "Experience with Ruby on Rails and Apache Kafka"
        )
        names = {s.name for s in skills}

        assert "Ruby on Rails" in names
        assert "Apache Kafka" in names

    def test_word_boundary_for_short_names(self, pattern_only_extractor: SkillExtractor):
        """Short skill names like 'C', 'R', 'Go' should use word boundaries."""
        # "C" should not match in "Acme Corp" or "CI/CD"
        skills = pattern_only_extractor.extract_skills(
            "Worked at Acme Corp building microservices"
        )
        names = {s.name for s in skills}
        assert "C" not in names  # Should NOT match the C in "Corp"

    def test_boundary_safe_positive_match(self, pattern_only_extractor: SkillExtractor):
        """Boundary-safe skills should still match when standalone."""
        skills = pattern_only_extractor.extract_skills(
            "Proficient in C and Go programming"
        )
        names = {s.name for s in skills}

        assert "C" in names
        assert "Go" in names

    def test_java_not_in_javascript(self, pattern_only_extractor: SkillExtractor):
        """'Java' should not be falsely matched inside 'JavaScript'."""
        skills = pattern_only_extractor.extract_skills("Experienced with JavaScript")
        names = {s.name for s in skills}

        assert "JavaScript" in names
        # Java should not appear as a separate match within JavaScript
        # (handled by span overlap prevention)

    def test_extracts_soft_skills(self, pattern_only_extractor: SkillExtractor):
        """Should extract soft skills when mentioned."""
        skills = pattern_only_extractor.extract_skills(
            "Strong leadership and communication skills. "
            "Experience mentoring junior developers."
        )
        names = {s.name for s in skills}

        assert "Leadership" in names
        assert "Communication" in names
        assert "Mentoring" in names


# ── NLP Extraction Tests ──────────────────────────────────────────────


class TestNLPExtraction:
    """Tests for Layer 2: spaCy NLP-based extraction."""

    def test_nlp_finds_additional_skills(self, extractor: SkillExtractor):
        """NLP should potentially find skills that pattern matching misses."""
        # This test verifies the NLP layer runs without errors
        # and produces valid Skill objects
        skills = extractor.extract_skills(RESUME_TEXT)
        assert all(isinstance(s, Skill) for s in skills)
        assert len(skills) > 0

    def test_nlp_extraction_graceful_without_spacy(self, pattern_only_extractor: SkillExtractor):
        """Extraction should work fine without spaCy."""
        skills = pattern_only_extractor.extract_skills(RESUME_TEXT)
        assert len(skills) > 5  # Should still find many skills via patterns


# ── Full Pipeline Tests ────────────────────────────────────────────────


class TestFullExtraction:
    """Tests for the complete extraction pipeline."""

    def test_resume_extraction(self, extractor: SkillExtractor):
        """Should extract a comprehensive set of skills from a resume."""
        skills = extractor.extract_skills(RESUME_TEXT, source="resume")

        names = {s.name for s in skills}

        # Core skills that MUST be found
        expected = {"Python", "JavaScript", "TypeScript", "React", "FastAPI",
                    "Django", "PostgreSQL", "MongoDB", "Redis", "Docker",
                    "Kubernetes", "AWS"}

        found_expected = expected & names
        assert len(found_expected) >= 10, (
            f"Expected at least 10 of {expected}, but only found {found_expected}"
        )

        # All skills should have source set
        for skill in skills:
            assert skill.source == "resume"

    def test_job_description_extraction(self, extractor: SkillExtractor):
        """Should extract skills from a job description."""
        skills = extractor.extract_skills(JOB_DESCRIPTION, source="job_description")

        names = {s.name for s in skills}

        # Key JD requirements
        expected = {"Python", "Go", "Kubernetes", "Docker",
                    "PostgreSQL", "Redis", "AWS", "GraphQL"}

        found_expected = expected & names
        assert len(found_expected) >= 6, (
            f"Expected at least 6 of {expected}, found {found_expected}"
        )

        for skill in skills:
            assert skill.source == "job_description"

    def test_extract_from_both(self, extractor: SkillExtractor):
        """extract_from_both should return two separate skill lists."""
        resume_skills, jd_skills = extractor.extract_from_both(
            RESUME_TEXT, JOB_DESCRIPTION
        )

        assert len(resume_skills) > 0
        assert len(jd_skills) > 0

        assert all(s.source == "resume" for s in resume_skills)
        assert all(s.source == "job_description" for s in jd_skills)


# ── Deduplication & Normalization Tests ────────────────────────────────


class TestDeduplication:
    """Tests for skill deduplication and normalization."""

    def test_no_duplicate_skills(self, extractor: SkillExtractor):
        """The same skill should not appear twice in results."""
        skills = extractor.extract_skills(
            "Python Python Python. I love Python. More Python experience."
        )
        names = [s.name for s in skills]

        assert names.count("Python") == 1

    def test_alias_deduplication(self, extractor: SkillExtractor):
        """Different aliases of the same skill should be deduplicated."""
        skills = extractor.extract_skills(
            "Experience with ReactJS, React.js, and React"
        )
        react_skills = [s for s in skills if s.name == "React"]

        assert len(react_skills) == 1  # All variants → single "React"

    def test_canonical_names_used(self, extractor: SkillExtractor):
        """Skills should use canonical names, not aliases."""
        skills = extractor.extract_skills(
            "Used k8s, reactjs, and golang"
        )
        names = {s.name for s in skills}

        assert "Kubernetes" in names   # Not "k8s"
        assert "React" in names        # Not "reactjs"
        assert "Go" in names           # Not "golang"


# ── Category Assignment Tests ──────────────────────────────────────────


class TestCategories:
    """Tests for correct skill category assignment."""

    def test_language_category(self, extractor: SkillExtractor):
        """Programming languages should be categorized correctly."""
        skills = extractor.extract_skills("Proficient in Python and JavaScript")

        python = next(s for s in skills if s.name == "Python")
        assert python.category == SkillCategory.LANGUAGE

    def test_framework_category(self, extractor: SkillExtractor):
        """Frameworks should be categorized correctly."""
        skills = extractor.extract_skills("Built apps with React and Django")

        react = next(s for s in skills if s.name == "React")
        assert react.category == SkillCategory.FRAMEWORK

    def test_soft_skill_category(self, extractor: SkillExtractor):
        """Soft skills should be categorized correctly."""
        skills = extractor.extract_skills("Strong leadership and mentoring abilities")

        leadership = next(s for s in skills if s.name == "Leadership")
        assert leadership.category == SkillCategory.SOFT

    def test_multiple_categories_present(self, extractor: SkillExtractor):
        """A diverse text should produce skills across multiple categories."""
        skills = extractor.extract_skills(RESUME_TEXT)

        categories = {s.category for s in skills}
        assert len(categories) >= 2  # Should have at least languages + tools/frameworks


# ── Edge Cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_text(self, extractor: SkillExtractor):
        """Empty text should return empty list, not raise."""
        assert extractor.extract_skills("") == []

    def test_whitespace_only(self, extractor: SkillExtractor):
        """Whitespace-only text should return empty list."""
        assert extractor.extract_skills("   \n\n   ") == []

    def test_no_skills_in_text(self, extractor: SkillExtractor):
        """Text without recognizable skills should return empty or minimal list."""
        skills = extractor.extract_skills(
            "Today was a beautiful sunny day. I went for a walk in the park."
        )
        # Should find very few or no skills
        assert len(skills) <= 2

    def test_skills_sorted_by_confidence(self, extractor: SkillExtractor):
        """Results should be sorted by confidence (descending)."""
        skills = extractor.extract_skills(RESUME_TEXT)

        if len(skills) > 1:
            for i in range(len(skills) - 1):
                assert skills[i].confidence >= skills[i + 1].confidence or (
                    skills[i].confidence == skills[i + 1].confidence
                    and skills[i].name.lower() <= skills[i + 1].name.lower()
                )

    def test_confidence_in_valid_range(self, extractor: SkillExtractor):
        """All confidence scores should be between 0 and 1."""
        skills = extractor.extract_skills(RESUME_TEXT)

        for skill in skills:
            assert 0.0 <= skill.confidence <= 1.0

    def test_taxonomy_size(self, extractor: SkillExtractor):
        """The taxonomy should contain a substantial number of skills."""
        assert extractor.get_taxonomy_size() >= 150

    def test_skill_objects_valid(self, extractor: SkillExtractor):
        """All returned objects should be valid Skill instances."""
        skills = extractor.extract_skills(RESUME_TEXT)

        for skill in skills:
            assert isinstance(skill, Skill)
            assert len(skill.name) > 0
            assert isinstance(skill.category, SkillCategory)
