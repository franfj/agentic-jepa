# %% [markdown]
# # Agentic-JEPA Training Notebook
#
# This notebook trains the Agentic-JEPA world model on text-based environment trajectories.
# Designed for Google Colab with GPU runtime.

# %% Cell 1: Install dependencies
# !pip install torch transformers pyyaml wandb scikit-learn
# !git clone https://github.com/franfj/AI-Factory.git  # adjust as needed
# %cd AI-Factory/products/projects/agentic-jepa

# %% Cell 2: Generate trajectory data
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "src.data.generate_trajectories",
        "--output", "data/trajectories.jsonl",
        "--oracle-episodes", "100",
        "--random-episodes", "200",
        "--seed", "42",
    ],
    check=True,
)

print("Trajectory data generated.")

# %% Cell 3: Verify GPU availability
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif device.type == "mps":
    print("Using Apple Silicon GPU (MPS)")
else:
    print("WARNING: No GPU detected. Training will be slow.")

# %% Cell 4: Smoke test with small_test config
subprocess.run(
    [
        sys.executable, "-m", "src.train",
        "--config", "configs/small_test.yaml",
        "--data", "data/trajectories.jsonl",
    ],
    check=True,
)

print("Small test completed successfully.")

# %% Cell 5: Train with default config
subprocess.run(
    [
        sys.executable, "-m", "src.train",
        "--config", "configs/default.yaml",
        "--data", "data/trajectories.jsonl",
    ],
    check=True,
)

print("Default training completed.")

# %% Cell 6: Run benchmarks
subprocess.run(
    [
        sys.executable, "-m", "src.benchmarks.run_all",
        "--checkpoint", "outputs/default/best.pt",
        "--config", "configs/default.yaml",
        "--episodes", "20",
    ],
    check=True,
)

# %% Cell 7: Results summary
import json
from pathlib import Path

# Load training log if available
log_path = Path("outputs/default/train_log.json")
if log_path.exists():
    with open(log_path) as f:
        train_log = json.load(f)
    print("Training Summary:")
    print(f"  Final loss: {train_log.get('final_loss', 'N/A')}")
    print(f"  Best val loss: {train_log.get('best_val_loss', 'N/A')}")
    print(f"  Total steps: {train_log.get('total_steps', 'N/A')}")
else:
    print("No training log found. Check outputs/default/ for checkpoints.")

# Check if checkpoint exists
ckpt_path = Path("outputs/default/best.pt")
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")
    if "step" in ckpt:
        print(f"Saved at step: {ckpt['step']}")
    if "val_loss" in ckpt:
        print(f"Val loss at save: {ckpt['val_loss']:.4f}")
else:
    print("No checkpoint found at outputs/default/best.pt")

print("\nDone. Check benchmark output above for agent comparison results.")
