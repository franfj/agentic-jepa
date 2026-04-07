"""Run the full Agentic-JEPA benchmark suite.

Evaluates multiple agents across all text environments, collects metrics, and
prints a comparison table. Optionally logs results to Weights & Biases.

Usage:
    python -m src.benchmarks.run_all --episodes 20 --max-steps 50
    python -m src.benchmarks.run_all --checkpoint checkpoints/best.pt --config configs/default.yaml
    python -m src.benchmarks.run_all --episodes 50 --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from ..model import AgenticJEPAModel
from .baselines import BaseAgent, GPT2VanillaAgent, GreedyTextAgent, JEPAPlanningAgent, LLMBaselineAgent, OracleAgent, RandomAgent
from .environments import TRAIN_ENVIRONMENTS, TEST_ENVIRONMENTS, TextEnvironment, make_all, make_train, make_test
from .metrics import EpisodeLog, MetricResult, compute_episode_metrics, compute_significance_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_episode(
    agent: BaseAgent,
    env: TextEnvironment,
    max_steps: int = 50,
) -> EpisodeLog:
    """Run a single episode: agent interacts with environment until goal or max_steps.

    Args:
        agent: The agent to evaluate.
        env: The environment instance (will be reset).
        max_steps: Maximum number of steps before termination.

    Returns:
        A complete ``EpisodeLog`` for metric computation.
    """
    log = EpisodeLog(goal=env.goal, max_steps=max_steps)

    state = env.reset()
    log.states.append(state)

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        oracle_action = env.get_oracle_action()

        log.valid_actions_per_step.append(valid_actions)
        log.oracle_actions.append(oracle_action or "")

        action = agent.select_action(state, valid_actions, env.goal)
        log.actions_taken.append(action)

        result = env.step(action)
        state = result.state
        log.states.append(state)

        if result.done:
            log.reached_goal = True
            break

    return log


def build_agents(
    env: TextEnvironment,
    model: AgenticJEPAModel | None = None,
    tokenizer: AutoTokenizer | None = None,
    max_length: int = 128,
    device: torch.device | None = None,
    seed: int = 0,
    gpt2_vanilla_agent: GPT2VanillaAgent | None = None,
    llm_agent: LLMBaselineAgent | None = None,
    rollout_depths: list[int] | None = None,
) -> list[BaseAgent]:
    """Build the standard set of agents for evaluation.

    Args:
        env: Environment instance (needed for Oracle agent).
        model: Trained JEPA model (optional; JEPA agent skipped if None).
        tokenizer: Tokenizer for the JEPA agent.
        max_length: Token sequence max length for the JEPA agent.
        device: Torch device for the JEPA agent.
        seed: Random seed for the random agent.
        gpt2_vanilla_agent: Pre-created GPT2VanillaAgent to reuse across calls.
        llm_agent: Pre-created LLMBaselineAgent to reuse across calls (optional).
        rollout_depths: List of rollout depths for JEPA agents (default: [1]).

    Returns:
        List of agent instances.
    """
    if rollout_depths is None:
        rollout_depths = [1]

    agents: list[BaseAgent] = [
        RandomAgent(seed=seed),
        GreedyTextAgent(),
    ]
    if gpt2_vanilla_agent is not None:
        agents.append(gpt2_vanilla_agent)
    else:
        agents.append(GPT2VanillaAgent(device=device))
    if llm_agent is not None:
        agents.append(llm_agent)
    if model is not None and tokenizer is not None:
        for depth in rollout_depths:
            agents.append(
                JEPAPlanningAgent(
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    device=device,
                    rollout_depth=depth,
                )
            )
    agents.append(OracleAgent(env))
    return agents


def run_benchmark(
    episodes: int = 50,
    max_steps: int = 50,
    checkpoint: str | None = None,
    config_path: str | None = None,
    wandb_log: bool = False,
    rollout_depths: list[int] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the full benchmark: all agents x all environments x N episodes.

    Args:
        episodes: Number of episodes per (agent, environment) pair.
        max_steps: Max steps per episode.
        checkpoint: Path to a trained JEPA model checkpoint.
        config_path: Path to a YAML config file for model hyperparameters.
        wandb_log: Whether to log results to Weights & Biases.

    Returns:
        Nested dict: ``results[agent_name][env_name][metric_name] = mean_value``.
    """
    # Load config
    config: dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Load model if checkpoint provided
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model: AgenticJEPAModel | None = None
    tokenizer: AutoTokenizer | None = None
    max_length = 128

    if checkpoint and Path(checkpoint).exists():
        model_cfg = config.get("model", {})
        backbone = model_cfg.get("backbone", "gpt2")
        max_length = model_cfg.get("max_length", 128)

        quantize = model_cfg.get("quantize", False)
        model = AgenticJEPAModel(
            backbone=backbone,
            predictor_hidden=model_cfg.get("predictor_hidden", 512),
            predictor_heads=model_cfg.get("predictor_heads", 8),
            predictor_layers=model_cfg.get("predictor_layers", 2),
            quantize=quantize,
        )
        if not quantize:
            model = model.to(device)
        else:
            model.predictor = model.predictor.to(device)
            model.target_encoder = model.target_encoder.to(device)
            model.loss_fn = model.loss_fn.to(device)

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=not quantize)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(backbone)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info(f"Loaded JEPA model from {checkpoint}")
    else:
        logger.info("No checkpoint provided; skipping JEPA agent.")

    # Wandb setup
    if wandb_log:
        try:
            import wandb
            wandb.init(project="agentic-jepa-benchmark", config={
                "episodes": episodes,
                "max_steps": max_steps,
                "checkpoint": checkpoint,
            })
        except ImportError:
            logger.warning("wandb not installed; skipping wandb logging.")
            wandb_log = False

    # Run evaluations
    # Structure: results[agent_name][env_name][metric_name] = mean_value
    results: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Create GPT2VanillaAgent once (avoids reloading the model per episode)
    gpt2_vanilla = GPT2VanillaAgent(device=device)

    # Create LLM baseline agent if API key is available
    llm_agent: LLMBaselineAgent | None = None
    if os.environ.get("OPENAI_API_KEY"):
        llm_agent = LLMBaselineAgent()
        logger.info("OPENAI_API_KEY found; including LLM (GPT-4o-mini) baseline.")
    else:
        logger.info("OPENAI_API_KEY not set; skipping LLM baseline.")

    for seed in range(episodes):
        envs = make_all(seed=seed)

        for env in envs:
            env_name = env.name
            agents = build_agents(
                env=env,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                device=device,
                seed=seed,
                gpt2_vanilla_agent=gpt2_vanilla,
                llm_agent=llm_agent,
                rollout_depths=rollout_depths,
            )

            for agent in agents:
                # Oracle agent needs the specific env instance; it already has it
                log = run_episode(agent, env, max_steps=max_steps)
                metrics = compute_episode_metrics(log, env.oracle_steps)

                agent_name = agent.name
                # Accumulate for averaging
                for metric_name in [
                    "steps_to_goal",
                    "planning_efficiency",
                    "cumulative_similarity_auc",
                    "action_quality_mean",
                ]:
                    key = f"_episodes_{metric_name}"
                    results[agent_name][env_name].setdefault(key, [])
                    results[agent_name][env_name][key].append(  # type: ignore[union-attr]
                        getattr(metrics, metric_name)
                    )

                results[agent_name][env_name].setdefault("_successes", [])
                results[agent_name][env_name]["_successes"].append(  # type: ignore[union-attr]
                    1.0 if metrics.success else 0.0
                )

    # Aggregate means and standard deviations
    final: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for agent_name, env_dict in results.items():
        for env_name, metric_dict in env_dict.items():
            for key, values in metric_dict.items():
                if key.startswith("_episodes_"):
                    metric_name = key[len("_episodes_"):]
                    assert isinstance(values, list)
                    arr = np.array(values)
                    final[agent_name][env_name][metric_name] = float(arr.mean())
                    final[agent_name][env_name][f"{metric_name}_std"] = float(arr.std())
                elif key == "_successes":
                    assert isinstance(values, list)
                    arr = np.array(values)
                    final[agent_name][env_name]["success_rate"] = float(arr.mean())
                    final[agent_name][env_name]["success_rate_std"] = float(arr.std())

    # --- Statistical significance tests ---
    # Collect per-episode success scores across all environments for each agent
    agent_success_all: dict[str, list[float]] = defaultdict(list)
    for agent_name, env_dict in results.items():
        for env_name, metric_dict in env_dict.items():
            successes = metric_dict.get("_successes", [])
            assert isinstance(successes, list)
            agent_success_all[agent_name].extend(successes)

    if len(agent_success_all) >= 2:
        try:
            sig_results = compute_significance_matrix(dict(agent_success_all), test_type="bootstrap")
            # Store significance in final results under a special key
            final["__significance__"] = {}  # type: ignore[assignment]
            for agent_a, comparisons in sig_results.items():
                for agent_b, test_result in comparisons.items():
                    key = f"{agent_a}_vs_{agent_b}"
                    final["__significance__"][key] = test_result  # type: ignore[index]

            # Print significance summary
            logger.info("\n--- Significance Tests (bootstrap, success rate) ---")
            # Find JEPA agents to compare against baselines
            jepa_agents = [a for a in agent_success_all if "JEPA" in a and "Vanilla" not in a]
            baseline_agents = [a for a in agent_success_all if a not in jepa_agents and a != "Oracle"]
            for ja in jepa_agents:
                for ba in baseline_agents:
                    if ja in sig_results and ba in sig_results[ja]:
                        r = sig_results[ja][ba]
                        sig_marker = "***" if r.get("significant_at_001") else ("**" if r.get("significant_at_005") else "n.s.")
                        logger.info(
                            f"  {ja} > {ba}: delta={r['delta']:.3f}, "
                            f"p={r['p_value']:.4f} {sig_marker}, "
                            f"95% CI=[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
                        )
        except Exception as e:
            logger.warning(f"Significance tests failed: {e}")

    # Wandb logging
    if wandb_log:
        import wandb
        for agent_name, env_dict in final.items():
            if agent_name.startswith("__"):
                continue
            for env_name, metrics_dict in env_dict.items():
                for metric_name, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        wandb.log({
                            f"{agent_name}/{env_name}/{metric_name}": value,
                        })
        wandb.finish()

    return dict(final)


def print_results_table(results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Print a formatted comparison table: Agent x Environment x Metric.

    Separates in-distribution (train) and out-of-distribution (test) environments.
    """
    if not results:
        logger.warning("No results to display.")
        return

    train_env_names = {cls(seed=0).name for cls in TRAIN_ENVIRONMENTS}
    test_env_names = {cls(seed=0).name for cls in TEST_ENVIRONMENTS}

    env_names = sorted({env for agent_envs in results.values() for env in agent_envs})
    agent_names = sorted(results.keys())
    metric_names = ["steps_to_goal", "success_rate", "planning_efficiency", "cumulative_similarity_auc", "action_quality_mean"]

    # Header
    header = f"{'Agent':<15} {'Environment':<25}"
    for m in metric_names:
        header += f" {m:<20}"
    print("\n" + "=" * len(header))
    print("AGENTIC-JEPA BENCHMARK RESULTS")
    print("=" * len(header))

    for section, section_envs in [("IN-DISTRIBUTION (train envs)", train_env_names), ("OUT-OF-DISTRIBUTION (test envs)", test_env_names)]:
        section_env_list = sorted(e for e in env_names if e in section_envs)
        if not section_env_list:
            continue
        print(f"\n--- {section} ---")
        print(header)
        print("-" * len(header))

        for agent_name in agent_names:
            for env_name in section_env_list:
                metrics = results.get(agent_name, {}).get(env_name, {})
                row = f"{agent_name:<15} {env_name:<25}"
                for m in metric_names:
                    val = metrics.get(m, float("nan"))
                    std = metrics.get(f"{m}_std", float("nan"))
                    cell = f"{val:.3f}\u00b1{std:.3f}"
                    row += f" {cell:<20}"
                print(row)
            print("-" * len(header))

    # Other envs (DataPipeline, ResearchTask — not in train or test)
    other_envs = sorted(e for e in env_names if e not in train_env_names and e not in test_env_names)
    if other_envs:
        print(f"\n--- OTHER ---")
        print(header)
        print("-" * len(header))
        for agent_name in agent_names:
            for env_name in other_envs:
                metrics = results.get(agent_name, {}).get(env_name, {})
                row = f"{agent_name:<15} {env_name:<25}"
                for m in metric_names:
                    val = metrics.get(m, float("nan"))
                    std = metrics.get(f"{m}_std", float("nan"))
                    cell = f"{val:.3f}\u00b1{std:.3f}"
                    row += f" {cell:<20}"
                print(row)
            print("-" * len(header))

    print()

    # Summary per section
    for section, section_envs in [("IN-DISTRIBUTION", train_env_names), ("OUT-OF-DISTRIBUTION", test_env_names), ("ALL", set(env_names))]:
        section_env_list = sorted(e for e in env_names if e in section_envs)
        if not section_env_list:
            continue
        print(f"SUMMARY — {section} (mean across environments)")
        print("-" * 80)
        summary_header = f"{'Agent':<15}"
        for m in metric_names:
            summary_header += f" {m:<20}"
        print(summary_header)
        print("-" * 100)

        for agent_name in agent_names:
            row = f"{agent_name:<15}"
            for m in metric_names:
                vals = [
                    results[agent_name][env_name].get(m, float("nan"))
                    for env_name in section_env_list
                    if env_name in results.get(agent_name, {})
                ]
                std_vals = [
                    results[agent_name][env_name].get(f"{m}_std", float("nan"))
                    for env_name in section_env_list
                    if env_name in results.get(agent_name, {})
                ]
                arr = np.array(vals)
                arr_std = np.array(std_vals)
                mean_val = float(arr.mean()) if len(vals) else float("nan")
                mean_std = float(arr_std.mean()) if len(std_vals) else float("nan")
                cell = f"{mean_val:.3f}\u00b1{mean_std:.3f}"
                row += f" {cell:<20}"
            print(row)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Agentic-JEPA benchmark suite."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for model hyperparameters.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained JEPA model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes per (agent, environment) pair (default: 50).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode before termination.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON.",
    )
    parser.add_argument(
        "--rollout-depths",
        type=int,
        nargs="+",
        default=[1],
        help="Rollout depths for JEPA multi-step planning (e.g., --rollout-depths 1 2 3 5).",
    )
    args = parser.parse_args()

    results = run_benchmark(
        episodes=args.episodes,
        max_steps=args.max_steps,
        checkpoint=args.checkpoint,
        config_path=args.config,
        wandb_log=args.wandb,
        rollout_depths=args.rollout_depths,
    )

    print_results_table(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
