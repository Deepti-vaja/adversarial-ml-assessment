"""
RAG Chatbot CLI & Evaluation Runner (Tasks E3 & E4).

Provides interactive terminal interface and automated benchmark demo runner answering
the 5 required question types (performance, attack, defense, boundary, comparison)
with explicit source citations. Enforces mandatory hallucination guard policy and
generates `chatbot_eval.json` registered as an MLflow artifact.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
from typing import Dict, Any, List

import yaml
import mlflow

from rag.retriever import VectorRetriever
from rag.prompt_builder import build_prompt, MANDATORY_HALLUCINATION_GUARD
from rag.llm_client import LLMClient


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def answer_query(query: str, config_path: str = "configs/rag.yaml", retriever: VectorRetriever = None) -> Dict[str, Any]:
    """
    Execute end-to-end RAG retrieval, prompt construction, and LLM synthesis.

    Args:
        query: User natural language query.
        config_path: Path to RAG YAML configuration file.
        retriever: Optional pre-initialized VectorRetriever instance.

    Returns:
        Dictionary containing query, retrieved_chunks (with metadata), and answer.
    """
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    ret_cfg = cfg.get("retrieval", {})
    llm_cfg = cfg.get("llm", {})

    vector_store_dir = paths_cfg.get("vector_store_dir", "rag/vector_store")
    model_name = ret_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    top_k = ret_cfg.get("top_k", 5)
    sim_threshold = ret_cfg.get("similarity_threshold", 0.15)

    if retriever is None:
        retriever = VectorRetriever(
            vector_store_dir=vector_store_dir,
            embedding_model=model_name
        )

    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    # Pre-retrieval verification check for empty or low-confidence evidence
    max_score = max((c["score"] for c in retrieved_chunks), default=0.0)
    if not retrieved_chunks or max_score < sim_threshold:
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": MANDATORY_HALLUCINATION_GUARD
        }

    prompt = build_prompt(query, retrieved_chunks)

    llm = LLMClient(
        provider=llm_cfg.get("provider", "mock"),
        model_name=llm_cfg.get("model_name", "Mistral-7B-Instruct-v0.2"),
        max_new_tokens=llm_cfg.get("max_new_tokens", 512),
        temperature=llm_cfg.get("temperature", 0.1)
    )

    answer = llm.generate(prompt, retrieved_chunks=retrieved_chunks)

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer
    }


def run_demo(config_path: str = "configs/rag.yaml") -> List[Dict[str, Any]]:
    """Run the 5 specification benchmark questions + out-of-domain guard check."""
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    ret_cfg = cfg.get("retrieval", {})
    retriever = VectorRetriever(
        vector_store_dir=paths_cfg.get("vector_store_dir", "rag/vector_store"),
        embedding_model=ret_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    )

    benchmark_queries = [
        # 1. Performance
        "What clean test accuracy and validation loss did the baseline FraudCNN model achieve?",
        # 2. Attack
        "How did PGD attack perturbation affect accuracy compared to the clean baseline evaluation?",
        # 3. Defense
        "What accuracy trade-offs did adversarial training and feature squeezing demonstrate?",
        # 4. Boundary
        "What did PCA decision region analysis reveal regarding adversarial feature space perturbations?",
        # 5. Comparison
        "Compare the training overhead and robustness differences between baseline and defense models."
    ]

    print("\n============================================================")
    print("        EXECUTING RAG CHATBOT BENCHMARK DEMO (TASK E3)")
    print("============================================================\n")

    eval_results = []
    for i, q in enumerate(benchmark_queries, 1):
        print(f"[Question {i}] {q}")
        res = answer_query(q, config_path=config_path, retriever=retriever)
        print(f"\n[Answer] {res['answer']}\n")
        print("-" * 60)
        eval_results.append(res)

    # Out-of-domain test for Hallucination Guard (Task E4)
    ood_query = "What is the current stock price of Apple?"
    print(f"[Guard Verification] {ood_query}")
    ood_res = answer_query(ood_query, config_path=config_path, retriever=retriever)
    print(f"\n[Guard Response] {ood_res['answer']}\n")
    print("============================================================\n")

    # Save to chatbot_eval.json and log to MLflow
    cfg = load_config(config_path)
    eval_json_path = cfg.get("paths", {}).get("eval_json_path", "rag/chatbot_eval.json")
    Path(eval_json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Benchmark evaluation logged to {eval_json_path}")

    # Register as MLflow artifact per spec
    try:
        mlflow.set_experiment("AML-RAG-EVAL")
        with mlflow.start_run(run_name="rag_chatbot_eval"):
            mlflow.log_artifact(eval_json_path, artifact_path="evaluation")
        print("[MLflow] Registered chatbot_eval.json as MLflow artifact.")
    except Exception as e:
        print(f"[WARNING] Could not log artifact to MLflow: {e}")

    return eval_results


def interactive_loop(config_path: str = "configs/rag.yaml"):
    print("\nAntigravity RAG Chatbot CLI (Type 'quit' or 'exit' to end)")
    print("-" * 60)
    while True:
        try:
            query = input("\nQuery> ").strip()
            if query.lower() in ["quit", "exit"]:
                break
            if not query:
                continue
            res = answer_query(query, config_path=config_path)
            print(f"\nAnswer:\n{res['answer']}\n")
            print("Citations / Retrieved Sections:")
            for c in res["retrieved_chunks"][:3]:
                meta = c["metadata"]
                print(f"  • [{meta.get('doc', '')}] Section: {meta.get('section', '')} (Score: {c['score']})")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nExiting chatbot CLI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial ML Assessment RAG Chatbot CLI")
    parser.add_argument("--demo", action="store_true", help="Run automated 5-question demo script and save eval JSON")
    parser.add_argument("--query", type=str, help="Single query to answer non-interactively")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="Path to config file")

    args = parser.parse_args()

    if args.demo:
        run_demo(args.config)
    elif args.query:
        result = answer_query(args.query, config_path=args.config)
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer']}\n")
    else:
        interactive_loop(args.config)
