"""
Explicit Prompt Builder Module (Task E2).

Constructs structured prompts combining system instructions, retrieved context blocks
with mandatory section-level citations [Source: {doc} | Section: {section}], and
the user query without using external frameworks (No LangChain/LlamaIndex).
"""

from typing import List, Dict, Any


MANDATORY_HALLUCINATION_GUARD = (
    "Insufficient evidence in experiment logs. Please consult raw MLflow runs."
)


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved section chunks and user query into a clean prompt string.

    Args:
        query: Natural language question from user.
        retrieved_chunks: List of retrieved chunk dictionaries from VectorRetriever.

    Returns:
        Fully formatted prompt string.
    """
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        meta = c.get("metadata", {})
        doc = meta.get("doc", "unknown")
        sec = meta.get("section", "Overview")
        text = c.get("text", "").strip()

        block = (
            f"--- Context Block [{i}] ---\n"
            f"[Source: {doc} | Section: {sec}]\n"
            f"{text}\n"
        )
        context_blocks.append(block)

    joined_context = "\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

    prompt = (
        "You are the Adversarial ML Assessment RAG Chatbot. Your task is to answer "
        "stakeholder questions regarding model robustness, adversarial attacks, decision boundaries, "
        "and defenses strictly grounded in the provided experiment logs and reports.\n\n"
        "INSTRUCTIONS:\n"
        "1. First Sentence Directness (BLUF): The very first sentence MUST answer the user's question directly and concisely, stating the exact conclusion, winner/loser, or parameter value.\n"
        "2. Numerical Comparison: Include numerical values whenever available. For comparison questions (strongest, weakest, largest, smallest, best, worst, highest, lowest, compare, more robust, less robust, improved, degraded), inspect retrieved tables and compare exact numerical figures.\n"
        "3. Strict Grounding: Explain conclusions ONLY when the explanation is explicitly supported by the retrieved evidence. Do NOT invent causal explanations or infer mechanisms not present in the knowledge base.\n"
        "4. Factual Lookup & Missing Info: For factual lookups (optimizer, learning rate, batch size, epochs, window size, bit depth, epsilon, architecture), return the exact value if present. If absent, clearly state that the information is not present in the retrieved knowledge base instead of hallucinating.\n"
        "5. Citations: Include supporting inline citation tags matching the context header exactly, e.g., `[Source: attack_evaluation.md | Section: 4. Per-Class Vulnerability Analysis]`.\n"
        f"6. Hallucination Guard: If the provided context lacks sufficient evidence to answer out-of-domain queries accurately, respond EXACTLY:\n"
        f"\"{MANDATORY_HALLUCINATION_GUARD}\"\n\n"
        "================ CONTEXT BLOCKS ================\n"
        f"{joined_context}\n"
        "================================================\n\n"
        f"USER QUERY: {query}\n\n"
        "GROUNDED EXECUTIVE ANSWER:"
    )

    return prompt
