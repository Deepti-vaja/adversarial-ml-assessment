"""
Knowledge Base Construction Script (Task E1).

Orchestrates MLflow export, reads reports and documentation, splits documents
using the section chunker, generates embeddings via sentence-transformers,
and persists the vector index to disk using ChromaDB.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from rag.mlflow_exporter import export_mlflow_runs
from rag.chunker import chunk_document, Chunk


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_knowledge_base(config_path: str = "configs/rag.yaml"):
    print("[1/5] Loading RAG configuration...")
    cfg = load_config(config_path)

    paths_cfg = cfg.get("paths", {})
    chunk_cfg = cfg.get("chunking", {})
    ret_cfg = cfg.get("retrieval", {})

    mlflow_export_dir = paths_cfg.get("mlflow_export_dir", "mlflow_export")
    reports_dir = paths_cfg.get("reports_dir", "reports")
    vector_store_dir = paths_cfg.get("vector_store_dir", "rag/vector_store")

    max_chars = chunk_cfg.get("max_chars", 1200)
    overlap_chars = chunk_cfg.get("overlap_chars", 150)
    model_name = ret_cfg.get("embedding_model", "all-MiniLM-L6-v2")

    print("[2/5] Exporting MLflow tracking runs...")
    exported_files = export_mlflow_runs(export_dir=mlflow_export_dir)
    print(f"      Exported {len(exported_files)} run summary files.")

    print("[3/5] Gathering documents and section chunking...")
    target_files = []

    # Gather MLflow exports
    export_path = Path(mlflow_export_dir)
    if export_path.exists():
        for p in export_path.rglob("*.md"):
            target_files.append(str(p))

    # Gather reports
    reports_path = Path(reports_dir)
    if reports_path.exists():
        for p in reports_path.iterdir():
            if p.suffix.lower() in [".md", ".csv"] and p.is_file():
                target_files.append(str(p))

    # Gather README.md
    if Path("README.md").exists():
        target_files.append("README.md")

    all_chunks: List[Chunk] = []
    for fpath in target_files:
        try:
            doc_chunks = chunk_document(fpath, max_chars=max_chars, overlap_chars=overlap_chars)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"      [WARNING] Failed to chunk {fpath}: {e}")

    print(f"      Total documents gathered: {len(target_files)}")
    print(f"      Total section chunks generated: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("No chunks were generated. Aborting knowledge base build.")

    print(f"[4/5] Loading embedding model ({model_name})...")
    embedder = SentenceTransformer(model_name)

    print(f"[5/5] Indexing into ChromaDB at {vector_store_dir}...")
    Path(vector_store_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=vector_store_dir)

    collection_name = "adversarial_ml_kb"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        ids = [f"chk_{i+j}_{c.chunk_id}" for j, c in enumerate(batch)]
        texts = [c.text for c in batch]
        metadatas = [c.metadata for c in batch]

        # Generate embeddings
        embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"\n[SUCCESS] Knowledge base built successfully! Indexed {len(all_chunks)} section chunks.")


if __name__ == "__main__":
    build_knowledge_base()
