"""
SkillForge AI — Interview Preparation Page.

Renders personalized, role-specific, and gap-aware interview questions
grounded in the candidate's resume and the career knowledge base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.skillforge.ai.llm.factory import create_llm_provider
from src.skillforge.models.matching import MatchResult
from src.skillforge.models.resume import ResumeAnalysis
from src.skillforge.models.roadmap import (
    InterviewQuestion,
    InterviewQuestionSet,
    QuestionCategory,
    QuestionDifficulty,
)
from src.skillforge.services.interview_generator import InterviewQuestionGenerator
from src.skillforge.services.retrieval_service import RetrievalService

st.set_page_config(page_title="Interview Preparation | SkillForge AI", page_icon="🎤", layout="wide")


@st.cache_resource
def get_interview_generator() -> InterviewQuestionGenerator:
    """Create a cached InterviewQuestionGenerator instance."""
    retrieval_service = RetrievalService()
    kb_path = PROJECT_ROOT / "knowledge_base"
    if kb_path.exists():
        retrieval_service.index_knowledge_base_directory(kb_path)

    settings = get_settings()
    try:
        if settings.active_api_key:
            provider = create_llm_provider()
        else:
            provider = None
    except Exception:
        provider = None

    return InterviewQuestionGenerator(retrieval_service=retrieval_service, llm_provider=provider)


st.title("🎤 Personalized Interview Preparation")
st.markdown("Customized technical, conceptual, project-based, and behavioral interview questions tailored to your profile and target role.")

# Check session state for match result and resume analysis
match_result: MatchResult | None = st.session_state.get("match_result")
resume_analysis: ResumeAnalysis | None = st.session_state.get("resume_analysis")
target_role: str = st.session_state.get("target_role", "Target Role")
job_description: str = st.session_state.get("job_description", "")

if not match_result and not resume_analysis:
    st.info("📋 Please upload your resume and complete the **Match Analysis** step on the Upload page first.")
    if st.button("⬅️ Go to Upload Page", type="primary"):
        st.switch_page("pages/01_📄_Upload.py")
    st.stop()

generator = get_interview_generator()

if "interview_question_set" not in st.session_state or st.session_state.get("interview_role") != target_role:
    with st.spinner("🧠 Generating personalized interview questions & retrieving domain rubrics..."):
        question_set = generator.generate_questions(
            match_result=match_result,
            resume_analysis=resume_analysis,
            target_role=target_role,
            job_description=job_description,
            total_questions=8,
        )
        st.session_state.interview_question_set = question_set
        st.session_state.interview_role = target_role
else:
    question_set: InterviewQuestionSet = st.session_state.interview_question_set

# ── Header Metrics & Summary ───────────────────────────────────────────

st.divider()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🎯 Target Role", target_role)
with c2:
    st.metric("📊 Calibrated Difficulty", question_set.difficulty_level)
with c3:
    st.metric("❓ Total Questions", question_set.total_questions)
with c4:
    st.metric("🗂️ Categories", "4 Types")

st.info(f"💡 **Interview Strategy:** {question_set.summary}")

if question_set.focus_areas:
    st.markdown(f"**🔥 Key Focus Themes:** " + " • ".join([f"`{a}`" for a in question_set.focus_areas]))

st.divider()

# ── Filter Tabs ────────────────────────────────────────────────────────

tab_all, tab_tech, tab_concept, tab_proj, tab_behav = st.tabs([
    "📋 All Questions",
    "💻 Technical",
    "🧠 Conceptual",
    "🛠️ Project-Based",
    "🤝 Behavioral",
])


def render_question_card(idx: int, q: InterviewQuestion) -> None:
    """Render an individual interactive interview question card."""
    diff_badge = "🔴 HARD" if q.difficulty == QuestionDifficulty.HARD else ("🟡 MEDIUM" if q.difficulty == QuestionDifficulty.MEDIUM else "🟢 EASY")
    cat_icons = {
        QuestionCategory.TECHNICAL: "💻 Technical",
        QuestionCategory.CONCEPTUAL: "🧠 Conceptual",
        QuestionCategory.PROJECT_BASED: "🛠️ Project-Based",
        QuestionCategory.BEHAVIORAL: "🤝 Behavioral",
    }
    cat_label = cat_icons.get(q.category, q.category.value.capitalize())

    with st.expander(f"**Q{idx}: {q.question}**", expanded=True):
        col_meta1, col_meta2, col_meta3 = st.columns([1, 1, 2])
        with col_meta1:
            st.markdown(f"**Type:** `{cat_label}`")
        with col_meta2:
            st.markdown(f"**Difficulty:** `{diff_badge}`")
        with col_meta3:
            if q.related_skill:
                st.markdown(f"**Focus Area / Skill:** `{q.related_skill}`")

        st.markdown(f"🎯 **Why this is asked:** *{q.why_this_question}*")

        st.divider()

        col_eval, col_sample = st.columns(2)
        with col_eval:
            st.markdown("#### 🧐 What the Interviewer Looks For:")
            for pt in q.evaluation_points:
                st.markdown(f"- {pt}")

        with col_sample:
            st.markdown("#### 💡 Key Talking Points for Strong Answer:")
            for pt in q.sample_answer_points:
                st.markdown(f"- {pt}")

        if q.guidance:
            st.info(f"🧭 **Coaching Tip:** {q.guidance}")

        if q.supporting_citations:
            st.markdown(f"##### 📌 Knowledge Base Grounding ({len(q.supporting_citations)} references)")
            for cit in q.supporting_citations:
                st.caption(
                    f"• **[{cit.source_name}]** *(Section: {cit.section})* — Relevance: `{cit.relevance_score:.2f}`\n\n"
                    f"> {cit.content_preview[:200]}..."
                )


with tab_all:
    for i, q in enumerate(question_set.questions, 1):
        render_question_card(i, q)

with tab_tech:
    tech_qs = [q for q in question_set.questions if q.category == QuestionCategory.TECHNICAL]
    for i, q in enumerate(tech_qs, 1):
        render_question_card(i, q)

with tab_concept:
    concept_qs = [q for q in question_set.questions if q.category == QuestionCategory.CONCEPTUAL]
    for i, q in enumerate(concept_qs, 1):
        render_question_card(i, q)

with tab_proj:
    proj_qs = [q for q in question_set.questions if q.category == QuestionCategory.PROJECT_BASED]
    for i, q in enumerate(proj_qs, 1):
        render_question_card(i, q)

with tab_behav:
    behav_qs = [q for q in question_set.questions if q.category == QuestionCategory.BEHAVIORAL]
    for i, q in enumerate(behav_qs, 1):
        render_question_card(i, q)

st.divider()

# ── General Preparation Tips Expander ──────────────────────────────────

with st.expander("📝 General Interview Strategy & Execution Tips", expanded=False):
    for tip in question_set.preparation_tips:
        st.markdown(f"- {tip}")
