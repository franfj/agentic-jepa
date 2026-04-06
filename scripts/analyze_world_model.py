"""Analyze the JEPA world model quality per environment.

Computes:
1. World model accuracy: fraction of steps where the predictor ranks the oracle action #1
2. Oracle action rank distribution: how often is oracle action top-1, top-2, top-3?
3. Per-step action scores: cosine similarity scores for all actions at each step

Usage:
    python scripts/analyze_world_model.py \
        --checkpoint outputs/default/best.pt \
        --config configs/default.yaml \
        --output results/world_model_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from src.benchmarks.environments import make_all, TRAIN_ENVIRONMENTS, TEST_ENVIRONMENTS
from src.model import AgenticJEPAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_environment(
    model: AgenticJEPAModel,
    tokenizer: AutoTokenizer,
    env_cls: type,
    max_length: int,
    device: torch.device,
    n_seeds: int = 10,
) -> dict:
    """Analyze world model accuracy on a single environment."""
    train_env_names = {cls(seed=0).name for cls in TRAIN_ENVIRONMENTS}

    total_steps = 0
    oracle_rank_1 = 0
    oracle_rank_top3 = 0
    rank_distribution = []
    per_step_data = []

    for seed in range(n_seeds):
        env = env_cls(seed=seed)
        state = env.reset()
        goal = env.goal

        while not env.done:
            valid_actions = env.get_valid_actions()
            oracle_action = env.get_oracle_action()
            if oracle_action is None:
                break

            # Tokenize
            state_tok = tokenizer([state], max_length=max_length, padding="max_length",
                                  truncation=True, return_tensors="pt")
            goal_tok = tokenizer([goal], max_length=max_length, padding="max_length",
                                 truncation=True, return_tensors="pt")
            actions_tok = tokenizer(valid_actions, max_length=max_length, padding="max_length",
                                    truncation=True, return_tensors="pt")

            state_tok = {k: v.to(device) for k, v in state_tok.items()}
            goal_tok = {k: v.to(device) for k, v in goal_tok.items()}
            actions_tok = {k: v.to(device) for k, v in actions_tok.items()}

            scores = model.score_actions(
                state_input_ids=state_tok["input_ids"],
                state_attention_mask=state_tok["attention_mask"],
                action_input_ids=actions_tok["input_ids"],
                action_attention_mask=actions_tok["attention_mask"],
                goal_input_ids=goal_tok["input_ids"],
                goal_attention_mask=goal_tok["attention_mask"],
            )

            scores_list = scores.cpu().tolist()
            ranked_indices = scores.argsort(descending=True).cpu().tolist()

            oracle_idx = valid_actions.index(oracle_action)
            oracle_rank = ranked_indices.index(oracle_idx) + 1  # 1-indexed

            total_steps += 1
            if oracle_rank == 1:
                oracle_rank_1 += 1
            if oracle_rank <= 3:
                oracle_rank_top3 += 1
            rank_distribution.append(oracle_rank)

            per_step_data.append({
                "seed": seed,
                "step": env.step_count,
                "oracle_rank": oracle_rank,
                "oracle_score": scores_list[oracle_idx],
                "best_score": max(scores_list),
                "score_gap": max(scores_list) - scores_list[oracle_idx],
            })

            # Take oracle action to continue
            result = env.step(oracle_action)
            state = result.state

    env_name = env_cls(seed=0).name
    return {
        "environment": env_name,
        "in_distribution": env_name in train_env_names,
        "total_steps": total_steps,
        "world_model_accuracy": oracle_rank_1 / max(total_steps, 1),
        "top3_accuracy": oracle_rank_top3 / max(total_steps, 1),
        "mean_oracle_rank": sum(rank_distribution) / max(len(rank_distribution), 1),
        "rank_distribution": rank_distribution,
        "per_step_data": per_step_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JEPA world model quality")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/world_model_analysis.json")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model_cfg = config["model"]
    model = AgenticJEPAModel(
        backbone=model_cfg["backbone"],
        predictor_hidden=model_cfg.get("predictor_hidden", 512),
        predictor_heads=model_cfg.get("predictor_heads", 8),
        predictor_layers=model_cfg.get("predictor_layers", 2),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_length = model_cfg.get("max_length", 128)

    all_envs = list(TRAIN_ENVIRONMENTS) + list(TEST_ENVIRONMENTS)
    results = []

    for env_cls in all_envs:
        env_name = env_cls(seed=0).name
        logger.info(f"Analyzing: {env_name}")
        analysis = analyze_environment(model, tokenizer, env_cls, max_length, device, args.seeds)
        results.append(analysis)
        logger.info(
            f"  World model accuracy: {analysis['world_model_accuracy']:.3f} "
            f"(top-3: {analysis['top3_accuracy']:.3f}), "
            f"mean oracle rank: {analysis['mean_oracle_rank']:.2f}"
        )

    # Summary
    id_results = [r for r in results if r["in_distribution"]]
    ood_results = [r for r in results if not r["in_distribution"]]

    summary = {
        "in_distribution": {
            "mean_accuracy": sum(r["world_model_accuracy"] for r in id_results) / max(len(id_results), 1),
            "mean_top3": sum(r["top3_accuracy"] for r in id_results) / max(len(id_results), 1),
        },
        "out_of_distribution": {
            "mean_accuracy": sum(r["world_model_accuracy"] for r in ood_results) / max(len(ood_results), 1),
            "mean_top3": sum(r["top3_accuracy"] for r in ood_results) / max(len(ood_results), 1),
        },
    }

    logger.info(f"\nSummary:")
    logger.info(f"  ID accuracy:  {summary['in_distribution']['mean_accuracy']:.3f}")
    logger.info(f"  OOD accuracy: {summary['out_of_distribution']['mean_accuracy']:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip per_step_data for compact output (keep only summary)
    compact = {
        "summary": summary,
        "environments": [{k: v for k, v in r.items() if k != "per_step_data"} for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(compact, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
