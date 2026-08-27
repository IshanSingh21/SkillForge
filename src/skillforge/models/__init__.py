"""
SkillForge AI — Data Models Package.

Re-exports all model classes for convenient importing:
    from src.skillforge.models import ResumeAnalysis, MatchResult, RAGResponse
"""

from src.skillforge.models.matching import MatchResult, SkillGap, SkillMatch
from src.skillforge.models.rag import (
    CitationSource,
    ConversationHistory,
    ConversationMessage,
    ConversationRole,
    RAGResponse,
    SourceCitation,
)
from src.skillforge.models.resume import (
    ResumeAnalysis,
    ResumeSection,
    Skill,
    SkillCategory,
    TextChunk,
)
from src.skillforge.models.roadmap import (
    InterviewQuestion,
    InterviewQuestionSet,
    LearningResource,
    LearningRoadmap,
    QuestionCategory,
    QuestionDifficulty,
    ResourceType,
    RoadmapMilestone,
)

__all__ = [
    # Resume
    "ResumeAnalysis",
    "ResumeSection",
    "Skill",
    "SkillCategory",
    "TextChunk",
    # Matching
    "MatchResult",
    "SkillMatch",
    "SkillGap",
    # Roadmap & Interview
    "LearningRoadmap",
    "RoadmapMilestone",
    "LearningResource",
    "ResourceType",
    "InterviewQuestion",
    "InterviewQuestionSet",
    "QuestionCategory",
    "QuestionDifficulty",
    # RAG
    "RAGResponse",
    "SourceCitation",
    "CitationSource",
    "ConversationMessage",
    "ConversationHistory",
    "ConversationRole",
]
