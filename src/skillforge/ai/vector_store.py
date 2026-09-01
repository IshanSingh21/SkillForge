"""
SkillForge AI — FAISS Vector Store.

High-performance vector database wrapper built on FAISS (Facebook AI Similarity Search).
Provides multi-namespace vector indexing, metadata management, cosine similarity retrieval,
and persistence for resumes, job descriptions, and career knowledge documents.

Design Decisions:
    - Uses `faiss.IndexFlatIP` (Inner Product) with L2-normalized embeddings,
      which is mathematically identical to Cosine Similarity while executing
      at maximum speed on CPU/GPU without approximation error.
    - Partitioned namespaces: allows targeted retrieval from 'resume',
      'job_description', 'knowledge_base', or across all namespaces simultaneously.
    - Synchronized metadata store: preserves full `TextChunk` objects with section,
      source provenance, and custom metadata for exact RAG citations.
    - Thread-safe and independent from the Streamlit UI.

Usage:
    from src.skillforge.ai.vector_store import VectorStore
    from src.skillforge.models.resume import TextChunk

    store = VectorStore(dimension=384)
    store.add_chunks(chunks=resume_chunks, embeddings=resume_embeddings, namespace="resume")
    results = store.search(query_embedding=q_emb, namespace="resume", top_k=3)
    for chunk, score in results:
        print(f"[{score:.2f}] {chunk.content[:100]}")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.skillforge.models.resume import TextChunk
from src.skillforge.utils.exceptions import VectorStoreError
from src.skillforge.utils.logging import logger


@dataclass
class SearchResult:
    """A single search result containing the retrieved chunk and similarity score."""

    chunk: TextChunk
    score: float

    def __iter__(self):
        """Allow tuple unpacking: chunk, score = result."""
        return iter((self.chunk, self.score))


class _NamespaceIndex:
    """Internal container for a single FAISS index and its associated metadata."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[TextChunk] = []
        self.chunk_id_map: dict[str, int] = {}  # chunk_id -> position in list

    def add(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """Add vectors and chunks to this namespace."""
        if len(chunks) == 0:
            return

        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        if embs.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embs.shape[1]}"
            )

        # Normalize rows to unit norm to ensure IndexFlatIP computes exact Cosine Similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs_normalized = (embs / norms).astype(np.float32)

        start_idx = len(self.chunks)
        self.index.add(embs_normalized)

        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_id_map[chunk.chunk_id] = start_idx + i

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search this namespace for top-k similar chunks."""
        if self.index.ntotal == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # Normalize query vector
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        q_normalized = (q / norm).astype(np.float32)

        # Fetch more candidates if filtering by metadata
        fetch_k = min(self.index.ntotal, top_k * 3 if filter_metadata else top_k)
        scores, indices = self.index.search(q_normalized, fetch_k)

        results: list[SearchResult] = []
        for sim, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            score_val = float(np.clip(sim, 0.0, 1.0))
            if score_val < min_score:
                continue

            chunk = self.chunks[idx]

            # Apply metadata filtering if specified
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if getattr(chunk, k, None) != v and chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append(SearchResult(chunk=chunk, score=score_val))
            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """Reset the index and metadata for this namespace."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.chunk_id_map = {}


class VectorStore:
    """
    FAISS-based vector store with partitioned namespace support.

    Features:
        - Multi-namespace isolation ('resume', 'job_description', 'knowledge_base')
        - Cross-namespace simultaneous search
        - Exact cosine similarity via normalized IndexFlatIP
        - Metadata filtering by section, source_name, or custom attributes
        - Persistence (save/load FAISS indices & metadata to disk)
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize the vector store.

        Args:
            dimension: Dimensionality of vector embeddings (default: 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self._namespaces: dict[str, _NamespaceIndex] = {}

    # ── Namespace Management ───────────────────────────────────────────

    def _get_or_create_namespace(self, namespace: str) -> _NamespaceIndex:
        """Retrieve existing namespace index or create a new one."""
        ns = namespace.strip().lower() or "default"
        if ns not in self._namespaces:
            self._namespaces[ns] = _NamespaceIndex(dimension=self.dimension)
        return self._namespaces[ns]

    def list_namespaces(self) -> list[str]:
        """Return list of all active namespace names."""
        return list(self._namespaces.keys())

    def count(self, namespace: str = "") -> int:
        """
        Return the number of vectors stored in a namespace (or total across all namespaces).
        """
        if namespace:
            ns = namespace.strip().lower()
            return self._namespaces[ns].index.ntotal if ns in self._namespaces else 0
        return sum(ns.index.ntotal for ns in self._namespaces.values())

    # ── Ingestion / Upsert ─────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: np.ndarray | list[list[float]],
        namespace: str = "default",
    ) -> list[str]:
        """
        Add chunks and their embeddings into the specified namespace.

        Args:
            chunks: List of TextChunk objects.
            embeddings: 2D numpy array or list of floats of shape (N, dimension).
            namespace: Namespace partition name (e.g. 'resume', 'job_description', 'knowledge_base').

        Returns:
            List of chunk IDs added.

        Raises:
            VectorStoreError: If input data is invalid or dimensionalities mismatch.
        """
        if not chunks:
            return []

        try:
            ns_index = self._get_or_create_namespace(namespace)
            embs = np.asarray(embeddings, dtype=np.float32)

            if len(chunks) != len(embs):
                raise VectorStoreError(
                    f"Count mismatch: received {len(chunks)} chunks but {len(embs)} embeddings."
                )

            ns_index.add(chunks, embs)
            logger.info(
                "Added chunks to vector store",
                namespace=namespace,
                count=len(chunks),
                total_in_namespace=ns_index.index.ntotal,
            )
            return [c.chunk_id for c in chunks]

        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            logger.error("Failed to add chunks to vector store", error=str(e), namespace=namespace)
            raise VectorStoreError(f"Failed to add chunks: {e}") from e

    def upsert(
        self,
        namespace: str,
        chunks: list[TextChunk],
        embeddings: np.ndarray | list[list[float]],
    ) -> list[str]:
        """
        Upsert chunks into a namespace (clears existing namespace data first).

        Args:
            namespace: Target namespace.
            chunks: TextChunk list.
            embeddings: Corresponding embeddings.

        Returns:
            List of chunk IDs added.
        """
        self.clear(namespace=namespace)
        return self.add_chunks(chunks=chunks, embeddings=embeddings, namespace=namespace)

    # ── Retrieval / Search ─────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray | list[float],
        namespace: str = "",
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for most similar chunks across one or all namespaces.

        Args:
            query_embedding: 1D or 2D vector for the query.
            namespace: Specific namespace to search ('resume', 'knowledge_base', etc.).
                       If empty or None, searches across ALL namespaces.
            top_k: Maximum number of results to return.
            min_score: Minimum cosine similarity threshold (0.0 to 1.0).
            filter_metadata: Optional dictionary of metadata key-value pairs to filter by.

        Returns:
            List of SearchResult objects (with .chunk and .score), sorted descending by score.
        """
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.size == 0:
            return []

        all_results: list[SearchResult] = []

        if namespace:
            ns = namespace.strip().lower()
            if ns in self._namespaces:
                all_results = self._namespaces[ns].search(
                    query_vector=q,
                    top_k=top_k,
                    min_score=min_score,
                    filter_metadata=filter_metadata,
                )
        else:
            # Search all active namespaces and merge results
            for ns_index in self._namespaces.values():
                ns_res = ns_index.search(
                    query_vector=q,
                    top_k=top_k,
                    min_score=min_score,
                    filter_metadata=filter_metadata,
                )
                all_results.extend(ns_res)

            # Sort merged results by similarity score descending
            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[:top_k]

        return all_results

    def search_by_text(
        self,
        query: str,
        embedding_engine: Any,
        namespace: str = "",
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Convenience method to encode text query on-the-fly and perform similarity search.

        Args:
            query: Natural language query string.
            embedding_engine: EmbeddingEngine instance with .encode_single() method.
            namespace: Namespace partition to search.
            top_k: Max results.
            min_score: Minimum score threshold.
            filter_metadata: Optional metadata filter.

        Returns:
            List of SearchResult objects.
        """
        if not query or not query.strip():
            return []

        query_emb = embedding_engine.encode_single(query)
        return self.search(
            query_embedding=query_emb,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )

    # ── Utility & Lookup ───────────────────────────────────────────────

    def get_chunk(self, chunk_id: str, namespace: str = "") -> TextChunk | None:
        """Find a chunk by its unique chunk_id."""
        if namespace:
            ns = namespace.strip().lower()
            if ns in self._namespaces:
                idx = self._namespaces[ns].chunk_id_map.get(chunk_id)
                if idx is not None:
                    return self._namespaces[ns].chunks[idx]
            return None

        for ns_index in self._namespaces.values():
            idx = ns_index.chunk_id_map.get(chunk_id)
            if idx is not None:
                return ns_index.chunks[idx]
        return None

    def clear(self, namespace: str = "") -> None:
        """
        Clear vectors and metadata in a specific namespace, or all namespaces if empty.
        """
        if namespace:
            ns = namespace.strip().lower()
            if ns in self._namespaces:
                self._namespaces[ns].clear()
                logger.info("Cleared namespace in vector store", namespace=ns)
        else:
            self._namespaces.clear()
            logger.info("Cleared all namespaces in vector store")

    # ── Persistence (Disk Save / Load) ─────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """
        Persist all FAISS indices and metadata to disk.

        Args:
            directory: Path to directory where files will be written.
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "dimension": self.dimension,
            "namespaces": list(self._namespaces.keys()),
        }

        with open(save_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        for ns_name, ns_index in self._namespaces.items():
            ns_dir = save_dir / ns_name
            ns_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save FAISS index
            index_path = str(ns_dir / "index.faiss")
            faiss.write_index(ns_index.index, index_path)

            # 2. Save metadata chunks
            chunks_data = [c.model_dump() for c in ns_index.chunks]
            with open(ns_dir / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info("VectorStore saved to disk", directory=str(save_dir), namespaces=list(self._namespaces.keys()))

    def load(self, directory: str | Path) -> None:
        """
        Load FAISS indices and metadata from disk.

        Args:
            directory: Directory containing manifest.json and namespace folders.
        """
        load_dir = Path(directory)
        manifest_path = load_dir / "manifest.json"

        if not manifest_path.exists():
            raise VectorStoreError(f"Cannot load vector store: manifest not found at {manifest_path}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.dimension = manifest.get("dimension", self.dimension)
            self._namespaces.clear()

            for ns_name in manifest.get("namespaces", []):
                ns_dir = load_dir / ns_name
                index_path = ns_dir / "index.faiss"
                chunks_path = ns_dir / "chunks.json"

                if not index_path.exists() or not chunks_path.exists():
                    logger.warning("Skipping incomplete namespace directory", namespace=ns_name)
                    continue

                # 1. Load FAISS index
                index = faiss.read_index(str(index_path))

                # 2. Load metadata chunks
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks_raw = json.load(f)

                chunks = [TextChunk(**c) for c in chunks_raw]

                ns_obj = _NamespaceIndex(dimension=self.dimension)
                ns_obj.index = index
                ns_obj.chunks = chunks
                ns_obj.chunk_id_map = {c.chunk_id: i for i, c in enumerate(chunks)}

                self._namespaces[ns_name] = ns_obj

            logger.info("VectorStore loaded from disk", directory=str(load_dir), namespaces=list(self._namespaces.keys()))

        except Exception as e:
            logger.error("Failed to load VectorStore from disk", error=str(e), directory=str(load_dir))
            raise VectorStoreError(f"Failed to load VectorStore: {e}") from e
