"""
Unit tests for the Section-Level Document Chunker (Task E1).
"""

import os
import tempfile
from pathlib import Path
import pytest
from rag.chunker import chunk_document, Chunk


def test_chunk_document_section_splitting():
    content = """# Experiment Run: test_run

**Run ID**: fa943ecb252d44a1a21d57ffb11323e4

## Run Parameters

- param1: 0.1
- param2: 64

## Run Metrics

- acc: 0.85
- loss: 0.32
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    try:
        chunks = chunk_document(temp_path)
        assert len(chunks) == 3
        
        # Verify metadata schema
        for c in chunks:
            assert isinstance(c, Chunk)
            assert "doc" in c.metadata
            assert "section" in c.metadata
            assert "run_id" in c.metadata
            assert c.metadata["run_id"] == "fa943ecb252d44a1a21d57ffb11323e4"

        sections = [c.metadata["section"] for c in chunks]
        assert "Experiment Run: test_run" in sections
        assert "Run Parameters" in sections
        assert "Run Metrics" in sections
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_chunk_document_csv_handling():
    content = "model,acc,overhead\nbaseline,0.75,0.01\nadv_trained,0.72,0.05"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    try:
        chunks = chunk_document(temp_path)
        assert len(chunks) >= 1
        expected_title = Path(temp_path).stem.replace("_", " ").title()
        assert expected_title in chunks[0].metadata["section"]
        assert "```csv" in chunks[0].text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
