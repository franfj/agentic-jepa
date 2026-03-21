#!/bin/bash
# ============================================================
# Agentic-JEPA — Full training + benchmark pipeline for Vast.ai
# ============================================================
# Launch on Vast.ai with:
#   - Image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
#   - GPU: A100 40GB (or A10G 24GB)
#   - Disk: 30GB
#
# Then SSH in and run:
#   bash scripts/run_vastai.sh
# ============================================================

set -e

echo "========================================"
echo "Agentic-JEPA Training Pipeline"
echo "========================================"

# --- Setup ---
echo "[1/7] Installing dependencies..."
pip install -q transformers torch pyyaml scikit-learn numpy matplotlib bitsandbytes accelerate anthropic

# Clone repo if not already present
if [ ! -f "src/model.py" ]; then
    git clone https://github.com/franfj/agentic-jepa.git /workspace/agentic-jepa
    cd /workspace/agentic-jepa
fi

# --- Generate trajectories ---
echo "[2/7] Generating trajectories..."
python -m src.data.generate_trajectories \
    --output data/trajectories.jsonl \
    --oracle-episodes 100 \
    --random-episodes 200 \
    --seed 42

# --- Train GPT-2 (baseline) ---
echo "[3/7] Training GPT-2 model..."
python -m src.train \
    --config configs/default.yaml \
    --data data/trajectories.jsonl

# --- Train Llama 3.2 ---
echo "[4/7] Training Llama 3.2 model..."
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
python -m src.train \
    --config configs/llama.yaml \
    --data data/trajectories.jsonl

# --- Benchmark GPT-2 ---
echo "[5/7] Benchmarking GPT-2..."
python -m src.benchmarks.run_all \
    --checkpoint outputs/default/best.pt \
    --config configs/default.yaml \
    --episodes 20 \
    --max-steps 50 \
    --output results/benchmark_gpt2.json

# --- Benchmark Llama 3.2 ---
echo "[6/7] Benchmarking Llama 3.2..."
python -m src.benchmarks.run_all \
    --checkpoint outputs/llama/best.pt \
    --config configs/llama.yaml \
    --episodes 20 \
    --max-steps 50 \
    --output results/benchmark_llama.json

# --- Visualize embeddings ---
echo "[7/7] Generating embedding visualizations..."
mkdir -p figures

python -m src.visualize_embeddings \
    --checkpoint outputs/default/best.pt \
    --config configs/default.yaml \
    --output-dir figures/gpt2

python -m src.visualize_embeddings \
    --checkpoint outputs/llama/best.pt \
    --config configs/llama.yaml \
    --output-dir figures/llama

echo "========================================"
echo "DONE! Results in:"
echo "  results/benchmark_gpt2.json"
echo "  results/benchmark_llama.json"
echo "  figures/embedding_gpt2.png"
echo "  figures/embedding_llama.png"
echo "========================================"
