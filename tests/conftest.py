"""
SkillForge AI — Shared Test Fixtures.

Provides reusable fixtures for pytest: sample texts, mock PDF bytes,
configuration overrides, and component instances.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.data.chunker import TextChunker
from src.skillforge.data.pdf_parser import PDFParser
from src.skillforge.data.preprocessor import TextPreprocessor


# ── Sample Data ────────────────────────────────────────────────────────


SAMPLE_RESUME_TEXT = """
John Doe
Software Engineer | john.doe@email.com | (555) 123-4567 | linkedin.com/in/johndoe

SUMMARY
Experienced software engineer with 5+ years of experience in Python, JavaScript,
and cloud technologies. Passionate about building scalable applications and
leading cross-functional teams.

EXPERIENCE

Senior Software Engineer — Acme Corp
January 2021 - Present
- Designed and implemented microservices architecture using Python and FastAPI
- Led migration of monolithic application to AWS, reducing costs by 40%
- Mentored 3 junior developers and conducted code reviews
- Implemented CI/CD pipelines using GitHub Actions and Docker

Software Engineer — TechStart Inc
June 2018 - December 2020
- Developed full-stack web applications using React and Node.js
- Built RESTful APIs serving 10K+ daily active users
- Optimized database queries, improving response times by 60%

EDUCATION

Master of Science in Computer Science
University of Technology — 2018

Bachelor of Science in Mathematics
State University — 2016

SKILLS
Python, JavaScript, TypeScript, React, Node.js, FastAPI, Django, AWS, Docker,
Kubernetes, PostgreSQL, MongoDB, Redis, Git, CI/CD, Agile, Scrum

CERTIFICATIONS
AWS Solutions Architect Associate — 2022
Google Cloud Professional Data Engineer — 2023

PROJECTS

Open Source CLI Tool — github.com/johndoe/cli-tool
- Built a command-line tool for automated code analysis with 500+ GitHub stars
- Written in Python using Click and Rich libraries
"""

SAMPLE_JOB_DESCRIPTION = """
Senior Backend Engineer — CloudTech Solutions

We are looking for a Senior Backend Engineer to join our platform team.

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
- Contributions to open source projects
- Experience with machine learning pipelines
"""


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_resume_text() -> str:
    """Return sample resume text for testing."""
    return SAMPLE_RESUME_TEXT


@pytest.fixture
def sample_job_description() -> str:
    """Return sample job description for testing."""
    return SAMPLE_JOB_DESCRIPTION


@pytest.fixture
def pdf_parser() -> PDFParser:
    """Return a fresh PDFParser instance."""
    return PDFParser()


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Return a fresh TextPreprocessor instance."""
    return TextPreprocessor()


@pytest.fixture
def chunker() -> TextChunker:
    """Return a TextChunker with default settings."""
    return TextChunker(chunk_size=512, chunk_overlap=50)


@pytest.fixture
def small_chunker() -> TextChunker:
    """Return a TextChunker with small chunk size for testing."""
    return TextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """
    Return minimal valid PDF bytes for testing.

    This is a bare-minimum valid PDF with a single page containing
    the text 'Hello SkillForge' — useful for testing the PDF parser
    without needing an actual resume file.
    """
    # Minimal valid PDF structure
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello SkillForge) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n441\n%%EOF\n"
    )
    return pdf_content
