"""
Section-Level Document Chunker Module (Task E1).

Splits structured text, markdown, and tabular report files strictly along section
boundaries (Markdown headers `#`, `##`, `###`) to avoid sentence-level fragmentation.
Enforces the mandatory metadata schema: `{doc, section, run_id}`.
"""

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any


@dataclass
class Chunk:
    """Represents a section chunk with required source metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Optional[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_document(file_path: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[Chunk]:
    """
    Chunk a document into section-level blocks with exact source metadata.

    Args:
        file_path: Absolute or relative path to the file.
        max_chars: Maximum character ceiling per chunk if a single section exceeds this limit.
        overlap_chars: Character overlap when splitting oversized sections.

    Returns:
        List of Chunk objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    content = path.read_text(encoding="utf-8")
    doc_name = path.name

    # Check if this file is CSV or tabular data
    if path.suffix.lower() == ".csv":
        title = doc_name.replace("_", " ").replace(".csv", "").title()
        content = f"# {title}\n\n```csv\n{content.strip()}\n```"

    # Extract run_id from filename or text if available
    run_id_match = re.search(r"run_([a-fA-F0-9]{32})", doc_name)
    if not run_id_match:
        run_id_match = re.search(r"\*\*Run ID\*\*:\s*([a-fA-F0-9]{32})", content)
    run_id = run_id_match.group(1) if run_id_match else None

    # Split document strictly by markdown headers
    header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    chunks = []
    sections = []

    if not matches:
        sections.append(("Overview", content.strip()))
    else:
        # Check preamble before first header
        if matches[0].start() > 0:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sections.append(("Overview", preamble))

        for i, match in enumerate(matches):
            section_title = match.group(2).strip()
            start_idx = match.start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[start_idx:end_idx].strip()
            if section_text:
                sections.append((section_title, section_text))

    chunk_counter = 0
    for section_title, section_text in sections:
        # If section fits within ceiling, keep as single chunk
        if len(section_text) <= max_chars:
            chunks.append(Chunk(
                chunk_id=f"{doc_name}_{chunk_counter}",
                text=section_text,
                metadata={
                    "doc": doc_name,
                    "section": section_title,
                    "run_id": run_id or ""
                }
            ))
            chunk_counter += 1
        else:
            # Sub-split large section along paragraph breaks or newlines
            paragraphs = section_text.split("\n\n")
            current_chunk = []
            current_len = 0

            for p in paragraphs:
                p_len = len(p)
                if current_len + p_len + 2 > max_chars and current_chunk:
                    chunk_str = "\n\n".join(current_chunk).strip()
                    chunks.append(Chunk(
                        chunk_id=f"{doc_name}_{chunk_counter}",
                        text=chunk_str,
                        metadata={
                            "doc": doc_name,
                            "section": section_title,
                            "run_id": run_id or ""
                        }
                    ))
                    chunk_counter += 1

                    # Keep overlap from last paragraph if reasonable
                    overlap_p = current_chunk[-1] if len(current_chunk[-1]) <= overlap_chars * 2 else current_chunk[-1][-overlap_chars:]
                    current_chunk = [overlap_p, p]
                    current_len = len(overlap_p) + 2 + p_len
                else:
                    current_chunk.append(p)
                    current_len += p_len + (2 if current_chunk else 0)

            if current_chunk:
                chunk_str = "\n\n".join(current_chunk).strip()
                chunks.append(Chunk(
                    chunk_id=f"{doc_name}_{chunk_counter}",
                    text=chunk_str,
                    metadata={
                        "doc": doc_name,
                        "section": section_title,
                        "run_id": run_id or ""
                    }
                ))
                chunk_counter += 1

    return chunks
