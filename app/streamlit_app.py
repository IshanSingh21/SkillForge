"""
SkillForge AI — Streamlit Application Entry Point.

This is the main entry point for the SkillForge AI web application.
Run with: streamlit run app/streamlit_app.py

The app uses Streamlit's multi-page architecture:
- Pages are defined in app/pages/ and appear in the sidebar navigation.
- Shared components live in app/components/.
- Application state is managed via st.session_state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path Setup ─────────────────────────────────────────────────────────
# Add project root to sys.path so that 'src', 'config' are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from src.skillforge.utils.logging import setup_logging, logger  # noqa: E402

# ── App Configuration ──────────────────────────────────────────────────

settings = get_settings()
setup_logging(log_level=settings.log_level.value)

st.set_page_config(
    page_title="SkillForge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/YOUR_USERNAME/skillforge-ai",
        "Report a Bug": "https://github.com/YOUR_USERNAME/skillforge-ai/issues",
        "About": f"# {settings.app_name}\nVersion {settings.app_version}\n\n"
        "Intelligent Resume Analysis & Career Development Platform",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e88e5 0%, #7c4dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1e88e5;
    }

    /* Status badges */
    .status-ready {
        color: #2e7d32;
        font-weight: 600;
    }
    .status-coming {
        color: #f57c00;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("SkillForge AI")
    st.caption(f"v{settings.app_version}")

    st.divider()

    st.markdown("### 🧭 Navigation")
    st.markdown(
        """
        Use the sidebar pages to:
        - **📄 Upload** your resume
        - **📊 Analyze** your match score
        - **🗺️ Roadmap** your learning path
        - **🎤 Interview** preparation
        - **💬 Assistant** for career questions
        """
    )

    st.divider()

    # Configuration info
    with st.expander("⚙️ Configuration", expanded=False):
        st.write(f"**LLM Provider:** {settings.llm_provider.value}")
        st.write(f"**Embedding Model:** {settings.embedding_model}")
        st.write(f"**Chunk Size:** {settings.chunk_size}")
        st.write(f"**Top-K Retrieval:** {settings.retrieval_top_k}")

        api_configured = bool(settings.active_api_key)
        if api_configured:
            st.success("✅ API Key Configured")
        else:
            st.warning("⚠️ No API Key Set — Add to .env file")

# ── Main Page Content ──────────────────────────────────────────────────

st.markdown('<p class="main-header">🧠 SkillForge AI</p>', unsafe_allow_html=True)
st.markdown(
    "#### Intelligent Resume Analysis & Career Development Platform",
)
st.markdown(
    "Upload your resume, paste a job description, and get AI-powered insights "
    "to bridge your skill gaps and ace your next interview."
)

st.divider()

# ── Feature Overview ───────────────────────────────────────────────────

st.markdown("### ✨ What SkillForge Can Do")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📄 Resume Analysis")
    st.markdown(
        "Upload your resume PDF and get structured text extraction, "
        "section detection, and intelligent skill identification."
    )
    st.caption("✅ Ready")

    st.markdown("#### 🔗 Semantic Matching")
    st.markdown(
        "Go beyond keyword matching — understand how your skills "
        "semantically relate to job requirements."
    )
    st.caption("🔜 Milestone 3")

with col2:
    st.markdown("#### 📊 Match Score")
    st.markdown(
        "Get a weighted, explainable match score showing exactly "
        "how well your resume fits the target role."
    )
    st.caption("🔜 Milestone 3")

    st.markdown("#### 🗺️ Learning Roadmap")
    st.markdown(
        "Receive a personalized learning plan with resources, "
        "timelines, and milestones to close your skill gaps."
    )
    st.caption("🔜 Milestone 5")

with col3:
    st.markdown("#### 💬 Career Assistant")
    st.markdown(
        "Ask questions grounded in your resume, the job description, "
        "and a curated career knowledge base — with source citations."
    )
    st.caption("🔜 Milestone 4")

    st.markdown("#### 🎤 Interview Prep")
    st.markdown(
        "Generate tailored interview questions based on the role "
        "and your specific skill gaps."
    )
    st.caption("🔜 Milestone 5")

st.divider()

# ── Quick Start Guide ─────────────────────────────────────────────────

st.markdown("### 🚀 Quick Start")

st.info(
    "👈 **Get started** by navigating to the **📄 Upload** page in the sidebar "
    "to upload your resume and enter a job description.",
    icon="💡",
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        **Step 1: Upload**
        1. Go to 📄 Upload page
        2. Upload your resume PDF
        3. Paste the target job description
        4. Enter the target role title
        """
    )

with col_b:
    st.markdown(
        """
        **Step 2: Analyze**
        1. View your match score
        2. See matched & missing skills
        3. Review the skill-gap analysis
        """
    )

with col_c:
    st.markdown(
        """
        **Step 3: Develop**
        1. Follow your learning roadmap
        2. Practice interview questions
        3. Ask the career assistant
        """
    )

# ── Footer ─────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Built with ❤️ using Streamlit, SentenceTransformers, FAISS, and Gemini/Groq | "
    f"{settings.app_name} v{settings.app_version}"
)

logger.info("SkillForge AI main page loaded")
