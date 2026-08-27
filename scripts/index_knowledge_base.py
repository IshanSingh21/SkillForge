"""Script to embed and index knowledge base documents into the vector store.

Usage:
    python scripts/index_knowledge_base.py

Implementation planned for Milestone 4.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    """Index all knowledge base documents."""
    print("Knowledge base indexing will be implemented in Milestone 4.")
    print("This script will:")
    print("  1. Read all markdown files from knowledge_base/")
    print("  2. Chunk them using TextChunker")
    print("  3. Generate embeddings using EmbeddingEngine")
    print("  4. Store them in VectorStore with 'knowledge_base' namespace")


if __name__ == "__main__":
    main()
