"""Generate trajectory data from text environments using Oracle and Random agents.

Produces a JSONL file where each line is a JSON-encoded trajectory (list of transitions).
Each transition is a dict with keys: state, action, next_state, done.

Usage:
    python -m src.data.generate_trajectories --output data/trajectories.jsonl
    python -m src.data.generate_trajectories --oracle-episodes 200 --random-episodes 400 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from ..benchmarks.environments import TRAIN_ENVIRONMENTS, TextEnvironment
from ..benchmarks.baselines import OracleAgent, RandomAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_episode(
    agent_type: str,
    env: TextEnvironment,
    seed: int,
    max_steps: int = 50,
) -> list[dict[str, str | bool]]:
    """Run one episode and collect transition tuples.

    Args:
        agent_type: "oracle" or "random".
        env: Environment instance (will be reset with the given seed).
        seed: Seed for this episode.
        max_steps: Maximum steps before terminating the episode.

    Returns:
        List of transition dicts: {state, action, next_state, done}.
    """
    # Re-seed the environment for this episode
    env._seed = seed
    state = env.reset()

    if agent_type == "oracle":
        agent = OracleAgent(env)
    elif agent_type == "random":
        agent = RandomAgent(seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    transitions: list[dict[str, str | bool]] = []

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, env.goal)

        result = env.step(action)
        transitions.append({
            "state": state,
            "action": action,
            "next_state": result.state,
            "done": result.done,
        })

        if result.done:
            break

        state = result.state

    return transitions


def generate_trajectories(
    output_path: str,
    oracle_episodes: int = 100,
    random_episodes: int = 200,
    seed: int = 42,
    max_steps: int = 50,
) -> None:
    """Generate trajectory data and save as JSONL.

    Args:
        output_path: Path for the output JSONL file.
        oracle_episodes: Number of oracle episodes per environment.
        random_episodes: Number of random episodes per environment.
        seed: Base random seed.
        max_steps: Maximum steps per episode.
    """
    rng = random.Random(seed)
    all_trajectories: list[dict] = []

    env_counts: dict[str, int] = {}
    env_transitions: dict[str, int] = {}
    total_transitions = 0

    for env_cls in TRAIN_ENVIRONMENTS:
        env_name = env_cls(seed=0).name
        env_counts[env_name] = 0
        env_transitions[env_name] = 0

        # Oracle episodes
        for i in range(oracle_episodes):
            ep_seed = rng.randint(0, 2**31)
            env = env_cls(seed=ep_seed)
            transitions = collect_episode("oracle", env, seed=ep_seed, max_steps=max_steps)

            if transitions:
                trajectory = {
                    "environment": env_name,
                    "agent": "oracle",
                    "seed": ep_seed,
                    "transitions": transitions,
                }
                all_trajectories.append(trajectory)
                env_counts[env_name] += 1
                env_transitions[env_name] += len(transitions)
                total_transitions += len(transitions)

        # Random episodes
        for i in range(random_episodes):
            ep_seed = rng.randint(0, 2**31)
            env = env_cls(seed=ep_seed)
            transitions = collect_episode("random", env, seed=ep_seed, max_steps=max_steps)

            if transitions:
                trajectory = {
                    "environment": env_name,
                    "agent": "random",
                    "seed": ep_seed,
                    "transitions": transitions,
                }
                all_trajectories.append(trajectory)
                env_counts[env_name] += 1
                env_transitions[env_name] += len(transitions)
                total_transitions += len(transitions)

    # Shuffle trajectories for training
    rng.shuffle(all_trajectories)

    # Write JSONL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for traj in all_trajectories:
            f.write(json.dumps(traj) + "\n")

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAJECTORY GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output file: {output}")
    logger.info(f"Total trajectories: {len(all_trajectories)}")
    logger.info(f"Total transitions: {total_transitions}")
    logger.info(f"Oracle episodes per env: {oracle_episodes}")
    logger.info(f"Random episodes per env: {random_episodes}")
    logger.info("-" * 60)
    logger.info(f"{'Environment':<25} {'Episodes':>10} {'Transitions':>12} {'Avg Length':>12}")
    logger.info("-" * 60)
    for env_name in env_counts:
        count = env_counts[env_name]
        trans = env_transitions[env_name]
        avg_len = trans / count if count > 0 else 0
        logger.info(f"{env_name:<25} {count:>10} {trans:>12} {avg_len:>12.1f}")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trajectory data from text environments."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--oracle-episodes",
        type=int,
        default=100,
        help="Number of oracle episodes per environment.",
    )
    parser.add_argument(
        "--random-episodes",
        type=int,
        default=200,
        help="Number of random episodes per environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode.",
    )
    args = parser.parse_args()

    generate_trajectories(
        output_path=args.output,
        oracle_episodes=args.oracle_episodes,
        random_episodes=args.random_episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
