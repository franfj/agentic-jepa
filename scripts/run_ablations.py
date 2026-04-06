"""Run all ablation experiments for Agentic-JEPA v2.

Supports multiple training seeds for robust results (mean ± std across seeds).

This script generates configs, trains models, and runs benchmarks for:
1. EMA momentum ablation: {0.0, 0.9, 0.99, 0.996, 0.999}
2. Predictor depth ablation: {1, 2, 4, 6} layers
3. Predictor width ablation: {256, 512, 1024} hidden dim
4. Backbone ablation: GPT-2, Qwen 2.5-1.5B, Llama 3.2-1B

Usage:
    python scripts/run_ablations.py --data data/trajectories_v2.jsonl --ablation ema
    python scripts/run_ablations.py --data data/trajectories_v2.jsonl --ablation all --seeds 3
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

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
BASE_CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"

TRAINING_SEEDS = [42, 123, 456]  # Default seeds for multi-seed training


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


def run_single_seed(
    name: str,
    config: dict,
    data_path: str,
    seed: int,
    episodes: int,
    benchmark_only: bool = False,
) -> dict | None:
    """Train and benchmark a single seed. Returns benchmark results dict or None."""
    cfg = copy.deepcopy(config)
    base_output = cfg["training"]["output_dir"]
    seed_dir = Path(base_output) / f"seed_{seed}"
    cfg["training"]["output_dir"] = str(seed_dir)
    cfg["seed"] = seed

    config_path = seed_dir / "config.yaml"
    save_config(cfg, config_path)
    checkpoint_path = seed_dir / "best.pt"

    if not benchmark_only:
        logger.info(f"  Training {name} (seed={seed})...")
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", str(config_path),
            "--data", data_path,
        ]
        result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"  Training failed for {name} seed={seed}: {result.stderr[-500:]}")
            return None

    if not checkpoint_path.exists():
        logger.warning(f"  No checkpoint at {checkpoint_path}")
        return None

    logger.info(f"  Benchmarking {name} (seed={seed})...")
    results_path = seed_dir / "benchmark_results.json"
    cmd = [
        sys.executable, "-m", "src.benchmarks.run_all",
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--episodes", str(episodes),
        "--max-steps", "50",
        "--rollout-depths", "1", "3",
        "--output", str(results_path),
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)

    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def aggregate_seed_results(seed_results: list[dict]) -> dict:
    """Aggregate benchmark results across multiple training seeds.

    Returns dict with mean and std for each metric per agent per environment.
    """
    if not seed_results:
        return {}

    # Collect all agent/env/metric values across seeds
    all_agents = set()
    all_envs = set()
    for sr in seed_results:
        for agent in sr:
            all_agents.add(agent)
            for env in sr[agent]:
                all_envs.add(env)

    aggregated = {}
    for agent in sorted(all_agents):
        aggregated[agent] = {}
        for env in sorted(all_envs):
            metrics_across_seeds: dict[str, list[float]] = {}
            for sr in seed_results:
                if agent in sr and env in sr[agent]:
                    for metric, value in sr[agent][env].items():
                        if isinstance(value, (int, float)) and not metric.endswith("_std"):
                            metrics_across_seeds.setdefault(metric, []).append(value)

            env_agg = {}
            for metric, values in metrics_across_seeds.items():
                arr = np.array(values)
                env_agg[metric] = float(arr.mean())
                env_agg[f"{metric}_seed_std"] = float(arr.std())
                env_agg[f"{metric}_n_seeds"] = len(values)
            aggregated[agent][env] = env_agg

    return aggregated


def run_variant(
    name: str,
    config: dict,
    data_path: str,
    seeds: list[int],
    episodes: int,
    benchmark_only: bool = False,
) -> dict | None:
    """Train a model variant across multiple seeds and aggregate results."""
    logger.info(f"=== {name} ({len(seeds)} seeds) ===")

    seed_results = []
    for seed in seeds:
        result = run_single_seed(name, config, data_path, seed, episodes, benchmark_only)
        if result is not None:
            seed_results.append(result)

    if not seed_results:
        logger.warning(f"  No successful runs for {name}")
        return None

    aggregated = aggregate_seed_results(seed_results)

    # Save aggregated results
    output_dir = Path(config["training"]["output_dir"])
    agg_path = output_dir / "aggregated_results.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump({
            "variant": name,
            "n_seeds": len(seed_results),
            "seeds_used": seeds[:len(seed_results)],
            "results": aggregated,
        }, f, indent=2)
    logger.info(f"  Aggregated {len(seed_results)} seeds → {agg_path}")

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic-JEPA ablation experiments")
    parser.add_argument("--data", type=str, default="data/trajectories_v2.jsonl",
                        help="Path to trajectory data file")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=list(ABLATIONS.keys()) + ["all"],
                        help="Which ablation to run")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of training seeds (default: 3)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Evaluation episodes per benchmark run (default: 50)")
    parser.add_argument("--list", action="store_true",
                        help="List all ablation variants without running them")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip training, only run benchmarks on existing checkpoints")
    args = parser.parse_args()

    seeds = TRAINING_SEEDS[:args.seeds]
    ablation_names = list(ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]

    for abl_name in ablation_names:
        variants_fn = ABLATIONS[abl_name]
        variants = variants_fn()

        if args.list:
            print(f"\n=== {abl_name} ({len(variants)} variants × {args.seeds} seeds) ===")
            for v in variants:
                cfg = v["config"]
                print(f"  {v['name']}: output_dir={cfg['training']['output_dir']}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {abl_name} ({len(variants)} variants × {len(seeds)} seeds)")
        logger.info(f"{'='*60}")

        for variant in variants:
            run_variant(
                variant["name"], variant["config"], args.data,
                seeds=seeds, episodes=args.episodes,
                benchmark_only=args.benchmark_only,
            )

    if args.list:
        return

    # Collect all aggregated results into a summary
    logger.info("\n=== Collecting aggregated results ===")
    summary = {}
    for abl_name in ablation_names:
        variants = ABLATIONS[abl_name]()
        for v in variants:
            agg_path = Path(v["config"]["training"]["output_dir"]) / "aggregated_results.json"
            if agg_path.exists():
                with open(agg_path) as f:
                    summary[v["name"]] = json.load(f)

    if summary:
        summary_path = BASE_DIR / "outputs" / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
