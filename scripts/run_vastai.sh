#!/bin/bash
# ============================================================
# Agentic-JEPA v2 — Full experiment pipeline for Vast.ai
# ============================================================
#
# GPU: RTX 3090 24GB (~$0.20/hr) o RTX 4090 24GB (~$0.35/hr)
#   - GPT-2 training: ~3GB VRAM
#   - Qwen 2.5-1.5B: ~10GB VRAM
#   - Llama 3.2-1B: ~8GB VRAM
#
# Vast.ai launch settings:
#   - Image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
#   - Disk: 40GB
#   - GPU: 1x RTX 3090 (24GB) minimum
#
# Then SSH in and run:
#   bash scripts/run_vastai.sh 2>&1 | tee experiment_log.txt
#
# Estimated total time: ~2-3 hours on RTX 3090
# ============================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "========================================"
echo "Agentic-JEPA v2 — Experiment Pipeline"
echo "Started: $(date)"
echo "========================================"

# --- Setup ---
echo "[SETUP] Installing dependencies..."
pip install -q transformers torch pyyaml scikit-learn numpy matplotlib \
    bitsandbytes accelerate openai umap-learn

# Clone repo if not already present
if [ ! -f "src/model.py" ]; then
    echo "[SETUP] Cloning repository..."
    git clone https://github.com/franfj/agentic-jepa.git /workspace/agentic-jepa
    cd /workspace/agentic-jepa
else
    echo "[SETUP] Repository already present"
    git pull origin main || true
fi

# Check GPU
echo "[SETUP] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p data results figures outputs

# ============================================================
# PHASE 1: Generate training data (CPU, ~2 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 1: Generate trajectories (8 train envs)"
echo "========================================"

python -m src.data.generate_trajectories \
    --output data/trajectories_v2.jsonl \
    --oracle-episodes 200 \
    --random-episodes 400 \
    --seed 42

# ============================================================
# PHASE 2: Train default model — GPT-2 (GPU, ~20 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 2: Train GPT-2 default model (2000 steps)"
echo "========================================"

python -m src.train \
    --config configs/default.yaml \
    --data data/trajectories_v2.jsonl

# ============================================================
# PHASE 3: Benchmark with multi-step planning (GPU, ~15 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 3: Benchmark GPT-2 (rollout k=1,2,3,5)"
echo "========================================"

python -m src.benchmarks.run_all \
    --checkpoint outputs/default/best.pt \
    --config configs/default.yaml \
    --episodes 20 \
    --max-steps 50 \
    --rollout-depths 1 2 3 5 \
    --output results/v2_gpt2.json

# ============================================================
# PHASE 4: World model analysis (GPU, ~5 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 4: World model accuracy analysis"
echo "========================================"

python scripts/analyze_world_model.py \
    --checkpoint outputs/default/best.pt \
    --config configs/default.yaml \
    --output results/world_model_gpt2.json \
    --seeds 10

# ============================================================
# PHASE 5: EMA ablation (GPU, ~40 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 5: EMA momentum ablation (5 variants)"
echo "========================================"

python scripts/run_ablations.py \
    --data data/trajectories_v2.jsonl \
    --ablation ema

# ============================================================
# PHASE 6: Predictor ablation (GPU, ~50 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 6: Predictor architecture ablation (7 variants)"
echo "========================================"

python scripts/run_ablations.py \
    --data data/trajectories_v2.jsonl \
    --ablation predictor_depth

python scripts/run_ablations.py \
    --data data/trajectories_v2.jsonl \
    --ablation predictor_width

# ============================================================
# PHASE 7: Alternative backbones (GPU, ~30 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 7: Qwen 2.5-1.5B backbone"
echo "========================================"

python -m src.train \
    --config configs/qwen.yaml \
    --data data/trajectories_v2.jsonl

python -m src.benchmarks.run_all \
    --checkpoint outputs/qwen/best.pt \
    --config configs/qwen.yaml \
    --episodes 20 \
    --max-steps 50 \
    --rollout-depths 1 3 \
    --output results/v2_qwen.json

python scripts/analyze_world_model.py \
    --checkpoint outputs/qwen/best.pt \
    --config configs/qwen.yaml \
    --output results/world_model_qwen.json \
    --seeds 10

# ============================================================
# PHASE 8: Generate figures (CPU, ~1 min)
# ============================================================
echo ""
echo "========================================"
echo "PHASE 8: Generate paper figures"
echo "========================================"

python scripts/generate_figures.py \
    --results results/v2_gpt2.json \
    --train-log outputs/default/train_log.json \
    --ablation-dir outputs \
    --output figures/

# ============================================================
# PHASE 9: Package results for download
# ============================================================
echo ""
echo "========================================"
echo "PHASE 9: Packaging results"
echo "========================================"

tar czf "agentic_jepa_results_${TIMESTAMP}.tar.gz" \
    results/ \
    figures/ \
    outputs/default/train_log.json \
    outputs/default/best.pt \
    outputs/qwen/train_log.json \
    outputs/qwen/best.pt \
    outputs/ablation_summary.json 2>/dev/null || true

echo ""
echo "========================================"
echo "ALL DONE! $(date)"
echo "========================================"
echo ""
echo "Results:"
echo "  results/v2_gpt2.json           — GPT-2 benchmark (k=1,2,3,5)"
echo "  results/v2_qwen.json           — Qwen 2.5-1.5B benchmark"
echo "  results/world_model_gpt2.json  — World model accuracy (GPT-2)"
echo "  results/world_model_qwen.json  — World model accuracy (Qwen)"
echo "  outputs/ablation_summary.json  — All ablation results"
echo "  figures/                       — Paper figures (PDF)"
echo ""
echo "Download: agentic_jepa_results_${TIMESTAMP}.tar.gz"
echo ""
echo "To copy results to local machine:"
echo "  scp -P PORT root@IP:~/agentic_jepa_results_${TIMESTAMP}.tar.gz ."
