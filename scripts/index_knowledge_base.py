"""
Script to embed and index knowledge base documents into the FAISS vector store.

Usage:
    python scripts/index_knowledge_base.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skillforge.services.retrieval_service import RetrievalService


def main() -> None:
    """Index all knowledge base documents and test retrieval."""
    print("=" * 60)
    print("🧠 SkillForge AI — Knowledge Base Vector Indexing")
    print("=" * 60)

    kb_dir = PROJECT_ROOT / "knowledge_base"
    print(f"Reading markdown documents from: {kb_dir}")

    service = RetrievalService()
    num_chunks = service.index_knowledge_base_directory(kb_dir)

    print(f"✅ Successfully indexed {num_chunks} chunks into FAISS vector store ('knowledge_base' namespace)!")
    print("\nRunning test queries:")

    test_queries = [
        "How to prepare for technical coding interviews and STAR method?",
        "What are common strategies for transitioning from backend to ML engineer?",
        "What are the most in-demand cloud and AI skills in 2025?",
    ]

    for q in test_queries:
        print(f"\n🔍 Query: '{q}'")
        citations = service.retrieve_from_knowledge_base(q, top_k=2)
        for i, c in enumerate(citations, 1):
            print(f"   [{i}] Source: {c.source_name} ({c.source_type.value}) | Score: {c.relevance_score:.2f}")
            print(f"       Preview: {c.content_preview}")

    print("\n" + "=" * 60)
    print("✨ Knowledge Base indexing and retrieval verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
