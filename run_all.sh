#!/usr/bin/env bash
#
# run_all.sh — Single-command end-to-end execution of the Adversarial ML Assessment pipeline.
#
# Enforces strict error handling: exits immediately with non-zero status on any command failure.
set -e

echo "========================================================================"
echo "🛡️ Starting End-to-End Adversarial ML & RAG Evaluation Pipeline"
echo "========================================================================"

echo ""
echo "[Stage 1/6] Dataset Preparation & Baseline Model Training (Task A)..."
python training/train_baseline.py

echo ""
echo "[Stage 2/6] Evasion Attack Evaluation Campaign (Task B)..."
python attacks/run_attack_campaign.py

echo ""
echo "[Stage 3/6] Decision Boundary Analysis & Visualization (Task C)..."
python visualization/umap_plot.py
python visualization/pca_regions.py
python visualization/boundary_probe.py
python visualization/boundary_distance.py

echo ""
echo "[Stage 4/6] Defense Benchmarking & Evaluation (Task D)..."
python defenses/adversarial_training.py
python defenses/feature_squeezing.py
python defenses/evaluate_defenses.py

echo ""
echo "[Stage 5/6] RAG Knowledge Base Construction & Vector Indexing (Task E)..."
python rag/build_knowledge_base.py

echo ""
echo "[Stage 6/6] RAG Chatbot Evaluation & Automated Demo Transcript..."
python rag/chatbot_cli.py --demo

echo ""
echo "========================================================================"
echo "✅ SUCCESS: All stages of the Adversarial ML & RAG pipeline completed!"
echo "========================================================================"
