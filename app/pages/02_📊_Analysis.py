"""
SkillForge AI — Match Analysis Dashboard.

Interactive dashboard that analyzes how well an uploaded resume matches
a target job description using explainable multi-factor scoring:
    1. Overall match score with tier indicator
    2. Detailed breakdown across the 4 scoring factors
    3. Strong matching skills with similarity and category badges
    4. Partially matching skills with closest resume alternatives
    5. Missing skill gaps with importance classification and recommendations
    6. Experience and seniority alignment assessment
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.services.semantic_matcher import SemanticMatcher
from src.skillforge.utils.exceptions import MatchingError
from src.skillforge.utils.logging import logger

st.set_page_config(page_title="Match Analysis | SkillForge AI", page_icon="📊", layout="wide")


@st.cache_resource
def get_matcher() -> SemanticMatcher:
    """Create a cached SemanticMatcher instance (shared across reruns)."""
    return SemanticMatcher()


# ── Session State Initialization ──────────────────────────────────────

if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "target_role" not in st.session_state:
    st.session_state.target_role = ""
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None

# ── Page Header ────────────────────────────────────────────────────────

st.title("📊 Resume-Job Match Analysis")
st.markdown(
    "Explainable AI matching between your resume and target job description "
    "using semantic embeddings, required skill coverage, and experience alignment."
)

st.divider()

# ── Data Availability Check ────────────────────────────────────────────

resume = st.session_state.resume_analysis
job_desc = st.session_state.job_description
target_role = st.session_state.target_role

if not resume or not job_desc.strip():
    st.warning("⚠️ **No resume or job description found.**")
    st.info(
        "Please go to **📄 Upload** in the sidebar to upload your resume and enter a target job description."
    )

    # Quick demo / fallback expander
    with st.expander("💡 Or enter resume & job description directly here for quick analysis:"):
        quick_resume = st.text_area("Resume text:", height=150, placeholder="Paste resume text...")
        quick_jd = st.text_area("Job Description:", height=150, placeholder="Paste job description...")
        quick_role = st.text_input("Target Role (optional):", placeholder="e.g. Senior Backend Engineer")

        if st.button("🚀 Analyze Now", type="primary", disabled=not (quick_resume and quick_jd)):
            matcher = get_matcher()
            with st.spinner("Analyzing match..."):
                st.session_state.match_result = matcher.match_resume_to_job(
                    resume=quick_resume,
                    job_description=quick_jd,
                    target_role=quick_role,
                )
                st.rerun()

    if not st.session_state.match_result:
        st.stop()

# ── Compute Match if Needed ────────────────────────────────────────────

matcher = get_matcher()

col_title, col_rerun = st.columns([4, 1])
with col_title:
    role_display = f" for **{target_role}**" if target_role else ""
    st.subheader(f"Match Results{role_display}")
with col_rerun:
    if st.button("🔄 Re-run Analysis", use_container_width=True):
        st.session_state.match_result = None

if st.session_state.match_result is None and resume and job_desc:
    with st.spinner("🔄 Computing semantic matching and explainable scoring..."):
        try:
            st.session_state.match_result = matcher.match_resume_to_job(
                resume=resume,
                job_description=job_desc,
                target_role=target_role,
            )
            logger.info("Match computed via Analysis page", score=st.session_state.match_result.overall_score)
        except Exception as e:
            st.error(f"❌ Error computing match: {e}")
            logger.exception("Match analysis failed")
            st.stop()

result = st.session_state.match_result

if result is None:
    st.stop()

# ── Overall Score & KPI Metrics ────────────────────────────────────────

score = result.overall_score
breakdown = result.score_breakdown

if score >= 80:
    tier_label = "🌟 Strong Match"
    tier_color = "green"
elif score >= 65:
    tier_label = "✅ Good Match"
    tier_color = "blue"
elif score >= 50:
    tier_label = "⚡ Moderate Match"
    tier_color = "orange"
else:
    tier_label = "⚠️ Significant Gap"
    tier_color = "red"

# Hero Score Card
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 24px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: #94a3b8; margin: 0; font-size: 16px;">OVERALL MATCH SCORE</h3>
                <h1 style="color: #f8fafc; margin: 4px 0 0 0; font-size: 48px; font-weight: 800;">
                    {score:.1f}%
                </h1>
                <span style="display: inline-block; margin-top: 6px; padding: 4px 12px; border-radius: 20px;
                             font-size: 14px; font-weight: 600; background: rgba(59, 130, 246, 0.2); color: #60a5fa;">
                    {tier_label}
                </span>
            </div>
            <div style="text-align: right; color: #cbd5e1;">
                <p style="margin: 0; font-size: 14px;">🎯 <b>{result.strong_matches_count}</b> Matched Skills</p>
                <p style="margin: 4px 0; font-size: 14px;">⚡ <b>{result.partial_matches_count}</b> Partial Matches</p>
                <p style="margin: 0; font-size: 14px;">❌ <b>{result.missing_skills_count}</b> Skill Gaps</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 Sub-Score Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="🎯 Skill Match (40%)",
        value=f"{breakdown.get('skill_match_score', 0):.1f}%",
        help="Weighted similarity across all extracted technical & soft skills",
    )
with kpi2:
    st.metric(
        label="📋 Required Coverage (25%)",
        value=f"{result.required_skills_coverage:.1f}%",
        help="Percentage of mandatory job requirements satisfied",
    )
with kpi3:
    st.metric(
        label="🔍 Semantic Fit (20%)",
        value=f"{result.semantic_similarity_score:.1f}%",
        help="Dense embedding cosine similarity of full resume vs job description",
    )
with kpi4:
    exp_score = result.experience_analysis.get("experience_score", 0)
    st.metric(
        label="📈 Experience Fit (15%)",
        value=f"{exp_score:.1f}%",
        help="Years of experience and seniority level alignment",
    )

st.divider()

# ── Detailed Analysis Tabs ─────────────────────────────────────────────

tab_matched, tab_partial, tab_missing, tab_exp, tab_methodology = st.tabs(
    [
        f"🎯 Matched Skills ({result.strong_matches_count})",
        f"⚡ Partial Matches ({result.partial_matches_count})",
        f"❌ Skill Gaps ({result.missing_skills_count})",
        "📈 Experience Analysis",
        "📐 Scoring Breakdown",
    ]
)

# Tab 1: Matched Skills
with tab_matched:
    if result.matched_skills:
        st.markdown("**Skills with strong semantic or exact matches:**")
        for m in sorted(result.matched_skills, key=lambda x: -x.similarity):
            match_icon = "⭐" if m.match_type == "exact" else "✅"
            col_m1, col_m2, col_m3 = st.columns([3, 2, 2])
            with col_m1:
                st.markdown(f"{match_icon} **{m.job_skill.name}**")
                st.caption(f"Matched from resume: `{m.resume_skill.name}`")
            with col_m2:
                st.markdown(f"Category: `{m.job_skill.category.value.title()}`")
            with col_m3:
                st.progress(m.similarity, text=f"{m.similarity * 100:.0f}% similarity")
            st.divider()
    else:
        st.info("No strong skill matches found.")

# Tab 2: Partial Matches
with tab_partial:
    if result.partial_matches:
        st.markdown(
            "**Skills with moderate similarity (50%–70%):** Transferable skills that provide partial credit."
        )
        for p in sorted(result.partial_matches, key=lambda x: -x.similarity):
            col_p1, col_p2, col_p3 = st.columns([3, 2, 2])
            with col_p1:
                st.markdown(f"⚡ **{p.job_skill.name}** (Target)")
                st.caption(f"Related resume skill: `{p.resume_skill.name}`")
            with col_p2:
                st.markdown(f"Category: `{p.job_skill.category.value.title()}`")
            with col_p3:
                st.progress(p.similarity, text=f"{p.similarity * 100:.0f}% similarity")
            st.divider()
    else:
        st.info("No partial matches. All skills are either strongly matched or clear gaps.")

# Tab 3: Missing Skills
with tab_missing:
    if result.missing_skills:
        req_gaps = result.required_gaps
        pref_gaps = result.preferred_gaps

        if req_gaps:
            st.markdown(f"### 🚨 Mandatory Requirements Missing ({len(req_gaps)}):")
            for g in req_gaps:
                with st.container():
                    st.markdown(f"❌ **{g.skill.name}** (`{g.skill.category.value.title()}`)")
                    st.caption(f"💡 **Recommendation:** {g.recommendation}")
                    st.divider()

        if pref_gaps:
            st.markdown(f"### 💡 Preferred / Nice-to-Have Gaps ({len(pref_gaps)}):")
            for g in pref_gaps:
                with st.container():
                    st.markdown(f"🔹 **{g.skill.name}** (`{g.skill.category.value.title()}`)")
                    st.caption(f"💡 **Recommendation:** {g.recommendation}")
                    st.divider()
    else:
        st.success("🎉 Outstanding! No skill gaps detected for this role.")

# Tab 4: Experience Analysis
with tab_exp:
    exp = result.experience_analysis
    st.subheader("Experience & Seniority Evaluation")

    c1, c2 = st.columns(2)
    with c1:
        req_yrs = exp.get("years_required")
        st.markdown(f"**Required Experience:** `{f'{req_yrs:g}+ years' if req_yrs else 'Not specified'}`")
        cand_yrs = exp.get("years_detected")
        st.markdown(f"**Detected Candidate Experience:** `{f'~{cand_yrs:g} years' if cand_yrs else 'Not detected'}`")
    with c2:
        st.markdown(f"**Target Role Seniority:** `{exp.get('job_seniority', 'Not specified').title()}`")
        st.markdown(f"**Candidate Seniority:** `{exp.get('candidate_seniority', 'Not specified').title()}`")

    if exp.get("notes"):
        st.markdown("#### Observations:")
        for note in exp["notes"]:
            st.markdown(f"- 📌 {note}")

# Tab 5: Scoring Methodology
with tab_methodology:
    st.subheader("📐 Transparent Scoring Formula")
    st.markdown(
        r"""
        SkillForge AI calculates the overall match score using a weighted multi-factor formula:

        $$\text{Overall Score} = 0.40 \cdot S_{\text{skills}} + 0.25 \cdot S_{\text{req}} + 0.20 \cdot S_{\text{semantic}} + 0.15 \cdot S_{\text{exp}}$$

        | Factor | Weight | Sub-Score | Contribution | Description |
        |---|---|---|---|---|
        """
        + f"""| **Skill Match Quality ($S_{{\\text{{skills}}}}$)** | 40% | {breakdown.get('skill_match_score', 0):.1f}% | {0.40 * breakdown.get('skill_match_score', 0):.1f} pts | Weighted average of semantic & exact skill matches |
        | **Required Skills Coverage ($S_{{\\text{{req}}}}$)** | 25% | {result.required_skills_coverage:.1f}% | {0.25 * result.required_skills_coverage:.1f} pts | Proportion of mandatory requirements satisfied |
        | **Content Semantic Fit ($S_{{\\text{{semantic}}}}$)** | 20% | {result.semantic_similarity_score:.1f}% | {0.20 * result.semantic_similarity_score:.1f} pts | Dense sentence embedding similarity of full documents |
        | **Experience & Seniority ($S_{{\\text{{exp}}}}$)** | 15% | {exp.get('experience_score', 0):.1f}% | {0.15 * exp.get('experience_score', 0):.1f} pts | Alignment of years of experience and seniority |
        | **TOTAL OVERALL SCORE** | **100%** | — | **{score:.1f}%** | **{tier_label}** |
        """
    )

    with st.expander("📄 View Explainable Text Report"):
        st.text(result.summary)

# ── Next Steps ─────────────────────────────────────────────────────────
st.divider()
st.info(
    "🗺️ **Ready to close your skill gaps?** Navigate to 🗺️ **Roadmap** in the sidebar to generate your personalized learning plan.",
    icon="🚀",
)
