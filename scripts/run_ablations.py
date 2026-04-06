"""Run all ablation experiments for Agentic-JEPA v2.

This script generates configs, trains models, and runs benchmarks for:
1. EMA momentum ablation: {0.0, 0.9, 0.99, 0.996, 0.999}
2. Predictor depth ablation: {1, 2, 4, 6} layers
3. Predictor width ablation: {256, 512, 1024} hidden dim
4. Training data ablation: oracle-only, random-only, mixed
5. Backbone ablation: GPT-2, Qwen 2.5-1.5B, Llama 3.2-1B

Usage:
    python scripts/run_ablations.py --data data/trajectories_v2.jsonl --ablation ema
    python scripts/run_ablations.py --data data/trajectories_v2.jsonl --ablation predictor_depth
    python scripts/run_ablations.py --data data/trajectories_v2.jsonl --ablation all
    python scripts/run_ablations.py --list  # Show all ablation variants
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
BASE_CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"


def load_base_config() -> dict:
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


ABLATIONS: dict[str, list[dict]] = {}


def _ema_variants() -> list[dict]:
    """EMA momentum ablation variants."""
    base = load_base_config()
    variants = []
    for m in [0.0, 0.9, 0.99, 0.996, 0.999]:
        cfg = copy.deepcopy(base)
        tag = f"ema_{str(m).replace('.', '')}"
        cfg["ema"]["momentum"] = m
        # For m=0.0, EMA is disabled (target = online at all times)
        if m == 0.0:
            cfg["ema"]["momentum_end"] = 0.0
        cfg["training"]["output_dir"] = f"outputs/ablation_ema/{tag}"
        variants.append({"name": tag, "config": cfg})
    return variants


def _predictor_depth_variants() -> list[dict]:
    """Predictor layer depth ablation."""
    base = load_base_config()
    variants = []
    for layers in [1, 2, 4, 6]:
        cfg = copy.deepcopy(base)
        tag = f"pred_L{layers}"
        cfg["model"]["predictor_layers"] = layers
        cfg["training"]["output_dir"] = f"outputs/ablation_predictor/{tag}"
        variants.append({"name": tag, "config": cfg})
    return variants


def _predictor_width_variants() -> list[dict]:
    """Predictor hidden dimension ablation."""
    base = load_base_config()
    variants = []
    for hidden in [256, 512, 1024]:
        cfg = copy.deepcopy(base)
        tag = f"pred_H{hidden}"
        cfg["model"]["predictor_hidden"] = hidden
        cfg["training"]["output_dir"] = f"outputs/ablation_predictor/{tag}"
        variants.append({"name": tag, "config": cfg})
    return variants


def _backbone_variants() -> list[dict]:
    """Backbone model ablation."""
    variants = []
    for config_name in ["default", "qwen", "llama"]:
        config_path = BASE_DIR / "configs" / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            backbone = cfg["model"]["backbone"]
            tag = backbone.split("/")[-1].lower().replace("-", "_").replace(".", "")
            cfg["training"]["output_dir"] = f"outputs/ablation_backbone/{tag}"
            variants.append({"name": f"backbone_{tag}", "config": cfg})
    return variants


ABLATIONS["ema"] = _ema_variants
ABLATIONS["predictor_depth"] = _predictor_depth_variants
ABLATIONS["predictor_width"] = _predictor_width_variants
ABLATIONS["backbone"] = _backbone_variants


def run_variant(name: str, config: dict, data_path: str, benchmark_only: bool = False) -> None:
    """Train a model variant and run benchmarks."""
    output_dir = Path(config["training"]["output_dir"])
    config_path = output_dir / "config.yaml"
    save_config(config, config_path)

    checkpoint_path = output_dir / "best.pt"

    if not benchmark_only:
        logger.info(f"=== Training: {name} ===")
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", str(config_path),
            "--data", data_path,
        ]
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        if result.returncode != 0:
            logger.error(f"Training failed for {name}")
            return

    if not checkpoint_path.exists():
        logger.warning(f"No checkpoint found at {checkpoint_path}, skipping benchmark")
        return

    logger.info(f"=== Benchmarking: {name} ===")
    results_path = output_dir / "benchmark_results.json"
    cmd = [
        sys.executable, "-m", "src.benchmarks.run_all",
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--episodes", "20",
        "--max-steps", "50",
        "--rollout-depths", "1", "3",
        "--output", str(results_path),
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic-JEPA ablation experiments")
    parser.add_argument("--data", type=str, default="data/trajectories_v2.jsonl",
                        help="Path to trajectory data file")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=list(ABLATIONS.keys()) + ["all"],
                        help="Which ablation to run")
    parser.add_argument("--list", action="store_true",
                        help="List all ablation variants without running them")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip training, only run benchmarks on existing checkpoints")
    args = parser.parse_args()

    ablation_names = list(ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]

    for abl_name in ablation_names:
        variants_fn = ABLATIONS[abl_name]
        variants = variants_fn()

        if args.list:
            print(f"\n=== {abl_name} ({len(variants)} variants) ===")
            for v in variants:
                cfg = v["config"]
                print(f"  {v['name']}: output_dir={cfg['training']['output_dir']}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {abl_name} ({len(variants)} variants)")
        logger.info(f"{'='*60}")

        for variant in variants:
            run_variant(variant["name"], variant["config"], args.data,
                        benchmark_only=args.benchmark_only)

    if args.list:
        return

    # Collect all results into a summary
    logger.info("\n=== Collecting results ===")
    summary = {}
    for abl_name in ablation_names:
        variants = ABLATIONS[abl_name]()
        for v in variants:
            results_path = Path(v["config"]["training"]["output_dir"]) / "benchmark_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    summary[v["name"]] = json.load(f)

    if summary:
        summary_path = BASE_DIR / "outputs" / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
