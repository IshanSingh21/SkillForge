"""
SkillForge AI — Learning Roadmap Page.

Renders a personalized, multi-stage learning roadmap based on the candidate's
skill-gap analysis and grounded in the career knowledge base.
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
from src.skillforge.models.roadmap import LearningRoadmap, RoadmapPriority, SkillGapStatus
from src.skillforge.services.retrieval_service import RetrievalService
from src.skillforge.services.roadmap_generator import RoadmapGenerator

st.set_page_config(page_title="Learning Roadmap | SkillForge AI", page_icon="🗺️", layout="wide")


@st.cache_resource
def get_roadmap_generator() -> RoadmapGenerator:
    """Create a cached RoadmapGenerator instance."""
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

    return RoadmapGenerator(retrieval_service=retrieval_service, llm_provider=provider)


st.title("🗺️ Personalized Learning Roadmap")
st.markdown("Your custom, step-by-step career development plan to close skill gaps and achieve target role readiness.")

# Check if match analysis exists in session state
match_result: MatchResult | None = st.session_state.get("match_result")
target_role: str = st.session_state.get("target_role", "Target Role")

if not match_result:
    st.info("📋 Please upload your resume and complete the **Match Analysis** step on the Upload / Analysis pages first.")
    if st.button("⬅️ Go to Upload Page", type="primary"):
        st.switch_page("pages/01_📄_Upload.py")
    st.stop()

# Generate or retrieve roadmap from session state
generator = get_roadmap_generator()

if "learning_roadmap" not in st.session_state or st.session_state.get("roadmap_for_role") != target_role:
    with st.spinner("🧠 Generating personalized learning roadmap & retrieving career knowledge..."):
        roadmap = generator.generate_roadmap(match_result=match_result, target_role=target_role)
        st.session_state.learning_roadmap = roadmap
        st.session_state.roadmap_for_role = target_role
else:
    roadmap: LearningRoadmap = st.session_state.learning_roadmap

# ── Header Metrics & Summary ───────────────────────────────────────────

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Target Role", target_role)
with col2:
    st.metric("📊 Current Match", f"{roadmap.overall_match_score:.1f}%")
with col3:
    st.metric("⏳ Total Duration", f"{roadmap.total_estimated_weeks:.1f} Weeks")
with col4:
    st.metric("🪜 Sequential Stages", f"{len(roadmap.stages)} Stages")

st.info(f"💡 **Executive Summary:** {roadmap.summary}")

if roadmap.key_focus_areas:
    st.markdown(f"**🔥 Top Focus Areas:** " + " • ".join([f"`{a}`" for a in roadmap.key_focus_areas]))

st.divider()

# ── 3 Sequential Learning Stages ───────────────────────────────────────

st.subheader("📚 Sequential Learning Plan")

stage_tabs = st.tabs([f"🪜 {s.stage_name.split(':')[0]} ({len(s.skills)} Skills)" for s in roadmap.stages])

for i, stage in enumerate(roadmap.stages):
    with stage_tabs[i]:
        st.markdown(f"### {stage.stage_name}")
        st.markdown(f"**🎯 Stage Focus:** {stage.focus_area}")
        st.markdown(f"**⏱️ Estimated Duration:** `{stage.estimated_duration_weeks:.1f} Weeks`")
        st.divider()

        if not stage.skills:
            st.success("🎉 No remaining gaps for this stage! You are fully on track.")
        else:
            for item in stage.skills:
                # Skill Card
                priority_color = "🔴" if item.priority == RoadmapPriority.HIGH else ("🟡" if item.priority == RoadmapPriority.MEDIUM else "🟢")
                status_badge = "⚠️ Missing Requirement" if item.status == SkillGapStatus.MISSING else "🔄 Partial Match (Enhance)"

                with st.expander(
                    f"{priority_color} **{item.skill}** — Priority: `{item.priority.value.upper()}` ({item.priority_score:.0f}/100) | Est: `{item.estimated_weeks:.1f} wks`",
                    expanded=True,
                ):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"**Status:** `{status_badge}`")
                        st.markdown(f"**Why it matters:** {item.priority_reason}")
                        st.markdown(f"**Role Context:** {item.relationship_to_role}")
                    with c2:
                        if item.prerequisites:
                            st.markdown(f"**Prerequisites:** " + ", ".join([f"`{p}`" for p in item.prerequisites]))
                        else:
                            st.markdown("**Prerequisites:** `None (Foundational)`")
                        st.markdown(f"**Category:** `{item.category}`")

                    st.markdown("#### 🎯 Core Learning Objectives")
                    for obj in item.learning_objectives:
                        st.markdown(f"- {obj}")

                    st.markdown(f"#### 🛠️ Recommended Practical Project")
                    st.success(f"**Portfolio Project:** {item.recommended_project}")

                    # Grounding Citations
                    if item.supporting_citations:
                        st.markdown(f"##### 📌 Grounded in Career Knowledge Base ({len(item.supporting_citations)} references)")
                        for cit in item.supporting_citations:
                            st.caption(
                                f"• **[{cit.source_name}]** *(Section: {cit.section})* — Relevance: `{cit.relevance_score:.2f}`\n\n"
                                f"> {cit.content_preview[:200]}..."
                            )

st.divider()

# ── Mastered & Demonstrated Skills Section ─────────────────────────────

with st.expander(f"⭐ Mastered Skills Already Demonstrated ({len(roadmap.demonstrated_skills)})", expanded=False):
    st.markdown(
        "These skills are already strongly demonstrated in your resume ($\ge 70\%$ match) "
        "and are **excluded** from your learning stages so you can focus exclusively on critical gaps:"
    )
    if roadmap.demonstrated_skills:
        skill_cols = st.columns(4)
        for idx, s in enumerate(roadmap.demonstrated_skills):
            skill_cols[idx % 4].markdown(f"✅ `{s}`")
    else:
        st.markdown("_No skills strongly matched yet._")
