"""
SkillForge AI — Resume Upload & Processing Page.

This page handles:
    1. PDF file upload with validation
    2. Optional raw text paste (fallback)
    3. Job description input
    4. Target role input
    5. Resume processing through the ResumeService pipeline
    6. Display of extraction results (text, sections, chunks)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path Setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.services.resume_service import ResumeService
from src.skillforge.utils.exceptions import PDFParsingError, PreprocessingError, SkillForgeError
from src.skillforge.utils.logging import logger

# ── Page Config ────────────────────────────────────────────────────────

st.set_page_config(page_title="Upload Resume | SkillForge AI", page_icon="📄")

# ── Initialize Session State ──────────────────────────────────────────

if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "target_role" not in st.session_state:
    st.session_state.target_role = ""
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


@st.cache_resource
def get_resume_service() -> ResumeService:
    """Create a cached ResumeService instance (shared across reruns)."""
    return ResumeService()


# ── Page Header ────────────────────────────────────────────────────────

st.title("📄 Upload Resume")
st.markdown(
    "Upload your resume and enter the target job description. "
    "SkillForge will extract, clean, and structure your resume text."
)

st.divider()

# ── Resume Input ───────────────────────────────────────────────────────

st.subheader("1. Your Resume")

input_method = st.radio(
    "How would you like to provide your resume?",
    ["📁 Upload PDF", "📋 Paste Text"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded_file = None
pasted_text = ""

if input_method == "📁 Upload PDF":
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your resume as a PDF. Max size: 20 MB. Must contain selectable text (not scanned images).",
    )
    if uploaded_file:
        file_size_kb = uploaded_file.size / 1024
        file_size_str = (
            f"{file_size_kb:.1f} KB"
            if file_size_kb < 1024
            else f"{file_size_kb / 1024:.1f} MB"
        )
        st.success(f"✅ **{uploaded_file.name}** uploaded ({file_size_str})")

else:
    pasted_text = st.text_area(
        "Paste your resume text",
        height=300,
        placeholder="Paste the full text of your resume here...",
        help="Use this if your PDF contains scanned images or if you prefer to paste directly.",
    )
    if pasted_text.strip():
        word_count = len(pasted_text.split())
        st.success(f"✅ Resume text received ({word_count} words)")

st.divider()

# ── Job Description Input ──────────────────────────────────────────────

st.subheader("2. Target Job Description")

job_description = st.text_area(
    "Paste the job description",
    value=st.session_state.job_description,
    height=200,
    placeholder="Paste the full job description for the role you're targeting...",
    help="Include the full job posting — responsibilities, requirements, and nice-to-haves.",
)

target_role = st.text_input(
    "Target Role Title",
    value=st.session_state.target_role,
    placeholder="e.g., Senior Backend Engineer, ML Engineer, Product Manager",
    help="The specific role title you're applying for.",
)

# Save to session state
st.session_state.job_description = job_description
st.session_state.target_role = target_role

st.divider()

# ── Process Button ─────────────────────────────────────────────────────

has_resume = uploaded_file is not None or bool(pasted_text.strip())
has_job = bool(job_description.strip())

if not has_resume:
    st.info("👆 Upload a resume PDF or paste your resume text to begin.")
elif not has_job:
    st.info("👆 Paste a job description to enable full analysis (optional for resume processing).")

col_btn, col_spacer = st.columns([1, 3])

with col_btn:
    process_clicked = st.button(
        "🚀 Process Resume",
        type="primary",
        disabled=not has_resume,
        use_container_width=True,
    )

# ── Processing Pipeline ───────────────────────────────────────────────

if process_clicked and has_resume:
    service = get_resume_service()

    with st.spinner("🔄 Processing your resume..."):
        try:
            if uploaded_file is not None:
                pdf_bytes = uploaded_file.getvalue()
                analysis = service.process_resume(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                )
            else:
                analysis = service.process_text(
                    raw_text=pasted_text,
                    source_name="pasted_resume",
                )

            st.session_state.resume_analysis = analysis
            st.session_state.processing_complete = True
            st.success("✅ Resume processed successfully!")

            logger.info(
                "Resume processed via UI",
                filename=analysis.filename,
                pages=analysis.page_count,
                sections=len(analysis.sections),
                chunks=len(analysis.chunks),
                words=analysis.word_count,
            )

        except PDFParsingError as e:
            st.error(f"❌ **PDF Error:** {e}")
            if e.detail:
                st.caption(f"💡 {e.detail}")
            logger.error("PDF processing failed in UI", error=str(e))

        except PreprocessingError as e:
            st.error(f"❌ **Processing Error:** {e}")
            if e.detail:
                st.caption(f"💡 {e.detail}")
            logger.error("Text processing failed in UI", error=str(e))

        except SkillForgeError as e:
            st.error(f"❌ **Error:** {e}")
            logger.error("Processing failed in UI", error=str(e))

        except Exception as e:
            st.error(f"❌ **Unexpected Error:** {e}")
            logger.exception("Unexpected error in UI processing")

# ── Display Results ────────────────────────────────────────────────────

analysis = st.session_state.resume_analysis

if analysis is not None:
    st.divider()
    st.subheader("📋 Processing Results")

    # ── Summary Metrics ────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📄 Pages", analysis.page_count or "N/A")
    with col2:
        st.metric("📝 Words", f"{analysis.word_count:,}")
    with col3:
        st.metric("📂 Sections", len(analysis.sections))
    with col4:
        st.metric("🎯 Skills", len(analysis.skills))
    with col5:
        st.metric("🧩 Chunks", len(analysis.chunks))

    # ── Warnings ───────────────────────────────────────────────
    if analysis.processing_errors:
        with st.expander(f"⚠️ Warnings ({len(analysis.processing_errors)})", expanded=False):
            for warning in analysis.processing_errors:
                st.warning(warning)

    # ── Tabs for detailed output ───────────────────────────────
    tab_clean, tab_skills, tab_sections, tab_chunks, tab_raw = st.tabs(
        ["🧹 Cleaned Text", "🎯 Extracted Skills", "📂 Sections", "🧩 Chunks", "📄 Raw Text"]
    )

    with tab_clean:
        st.markdown("**Cleaned and normalized text** (encoding artifacts removed, whitespace normalized):")
        st.text_area(
            "Cleaned text",
            value=analysis.cleaned_text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    with tab_skills:
        if analysis.skills:
            st.markdown(f"**{len(analysis.skills)} skills extracted from resume:**")
            # Group by category
            categories = {}
            for s in analysis.skills:
                cat_name = s.category.value.title()
                categories.setdefault(cat_name, []).append(s)

            for cat_name, skill_list in sorted(categories.items()):
                st.markdown(f"**{cat_name}** ({len(skill_list)}):")
                skill_badges = " ".join([f"`{s.name}`" for s in skill_list])
                st.markdown(skill_badges)
        else:
            st.info("No skills detected in the resume text.")

    with tab_sections:
        if analysis.sections:
            st.markdown(f"**{len(analysis.sections)} sections detected:**")
            for i, section in enumerate(analysis.sections):
                with st.expander(
                    f"📌 {section.title} ({len(section.content)} chars)",
                    expanded=(i == 0),
                ):
                    st.text(section.content[:2000])
                    if len(section.content) > 2000:
                        st.caption(f"... ({len(section.content) - 2000} more characters)")
        else:
            st.info(
                "No standard sections detected. The resume may use non-standard headings. "
                "Full-text chunking was used instead."
            )

    with tab_chunks:
        st.markdown(f"**{len(analysis.chunks)} text chunks** generated for embedding:")
        for i, chunk in enumerate(analysis.chunks[:20]):  # Show first 20
            section_label = f" • Section: {chunk.section}" if chunk.section else ""
            with st.expander(
                f"Chunk {i + 1} ({len(chunk.content)} chars{section_label})",
                expanded=(i == 0),
            ):
                st.code(chunk.content, language=None)
                st.caption(f"ID: `{chunk.chunk_id}` | Source: `{chunk.source}`")

        if len(analysis.chunks) > 20:
            st.caption(f"... and {len(analysis.chunks) - 20} more chunks")

    with tab_raw:
        st.markdown("**Raw extracted text** (before cleaning):")
        st.text_area(
            "Raw text",
            value=analysis.raw_text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    # ── Next Steps ─────────────────────────────────────────────
    st.divider()
    st.info(
        "✅ **Resume processed!** Navigate to 📊 **Analysis** in the sidebar to view your detailed match score and skill breakdown.",
        icon="🚀",
    )
