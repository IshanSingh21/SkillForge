"""SkillForge AI — Resume Upload Page."""

import streamlit as st

st.set_page_config(page_title="Upload Resume | SkillForge AI", page_icon="📄")

st.title("📄 Upload Resume")
st.markdown("Upload your resume and enter the target job description.")

st.divider()

# Resume upload
st.subheader("1. Upload Your Resume (PDF)")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload your resume in PDF format. Max size: 20 MB.",
)

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
    st.info("📋 Resume processing will be available in Milestone 2.")

st.divider()

# Job description input
st.subheader("2. Target Job Description")
job_description = st.text_area(
    "Paste the job description here",
    height=200,
    placeholder="Paste the full job description for the role you're targeting...",
)

# Target role
target_role = st.text_input(
    "Target Role Title",
    placeholder="e.g., Senior Backend Engineer",
)

st.divider()

if st.button("🚀 Analyze Resume", type="primary", disabled=not (uploaded_file and job_description)):
    st.info("🔧 Full analysis pipeline will be available in Milestone 3.")
