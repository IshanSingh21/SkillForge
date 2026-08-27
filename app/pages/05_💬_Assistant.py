"""SkillForge AI — RAG Career Assistant Page."""

import streamlit as st

st.set_page_config(page_title="Career Assistant | SkillForge AI", page_icon="💬")

st.title("💬 Career Assistant")
st.markdown("Ask questions grounded in your resume, job description, and career knowledge base.")

st.divider()
st.info("📋 The RAG-powered career assistant will be available in Milestone 4.")

# Chat interface placeholder
st.text_input("Ask a career question...", disabled=True, placeholder="e.g., How should I prepare for system design interviews?")
