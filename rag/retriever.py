"""
RAG Vector Retriever Module (Task E2).

Queries ChromaDB vector index and sentence-transformers embedding model
to retrieve the top-k (k >= 5) section chunks matching a user query.
Enforces 100% deterministic sorting and exact metadata schema retention.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    """Retriever service for semantic similarity search over section chunks."""

    def __init__(
        self,
        vector_store_dir: str = "rag/vector_store",
        collection_name: str = "adversarial_ml_kb",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.model_name = embedding_model

        if not Path(vector_store_dir).exists():
            raise FileNotFoundError(
                f"Vector store directory '{vector_store_dir}' not found. "
                "Please run `python rag/build_knowledge_base.py` first."
            )

        self.client = chromadb.PersistentClient(path=vector_store_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(
                f"Collection '{collection_name}' not found in ChromaDB at {vector_store_dir}. "
                "Please run `python rag/build_knowledge_base.py`."
            ) from e

        self.embedder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k section chunks most relevant to the query.

        Args:
            query: Natural language search query string.
            top_k: Number of results to retrieve (must be >= 5 per spec).

        Returns:
            List of dictionaries containing text, metadata, and similarity score.
        """
        # Enforce specification constraint k >= 5
        if top_k < 5:
            top_k = 5

        # Check collection size
        count = self.collection.count()
        if count == 0:
            return []
        
        actual_k = min(top_k, count)

        query_vec = self.embedder.encode([query], show_progress_bar=False, normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=actual_k
        )

        retrieved_chunks = []
        ids = results.get("ids", [[]])[0]
        texts = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            # ChromaDB cosine distance range is [0, 2], similarity = 1 - distance
            dist = distances[i] if i < len(distances) else 0.0
            similarity = max(0.0, 1.0 - dist)

            retrieved_chunks.append({
                "chunk_id": ids[i],
                "text": texts[i],
                "score": round(float(similarity), 4),
                "metadata": metadatas[i] or {}
            })

        # Deterministic sorting: primarily by score descending, secondarily by doc+section ascending
        retrieved_chunks.sort(
            key=lambda x: (
                -x["score"],
                x["metadata"].get("doc", ""),
                x["metadata"].get("section", "")
            )
        )

        return retrieved_chunks
