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

# --- Train Qwen2.5-1.5B ---
echo "[4/7] Training Qwen2.5-1.5B model..."
python -m src.train \
    --config configs/qwen.yaml \
    --data data/trajectories.jsonl

# --- Benchmark GPT-2 ---
echo "[5/7] Benchmarking GPT-2..."
python -m src.benchmarks.run_all \
    --checkpoint outputs/default/best.pt \
    --config configs/default.yaml \
    --episodes 20 \
    --max-steps 50 \
    --output results/benchmark_gpt2.json

# --- Benchmark Qwen2.5-1.5B ---
echo "[6/7] Benchmarking Qwen2.5-1.5B..."
python -m src.benchmarks.run_all \
    --checkpoint outputs/qwen/best.pt \
    --config configs/qwen.yaml \
    --episodes 20 \
    --max-steps 50 \
    --output results/benchmark_qwen.json

# --- Visualize embeddings ---
echo "[7/7] Generating embedding visualizations..."
mkdir -p figures

python -m src.visualize_embeddings \
    --checkpoint outputs/default/best.pt \
    --config configs/default.yaml \
    --output-dir figures/gpt2

python -m src.visualize_embeddings \
    --checkpoint outputs/qwen/best.pt \
    --config configs/qwen.yaml \
    --output-dir figures/qwen

echo "========================================"
echo "DONE! Results in:"
echo "  results/benchmark_gpt2.json"
echo "  results/benchmark_qwen.json"
echo "  figures/gpt2/embedding_space.png"
echo "  figures/qwen/embedding_space.png"
echo "========================================"
