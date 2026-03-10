"""Verify all text environments work correctly.

Instantiates each environment, runs one episode with OracleAgent to confirm
it reaches the goal in exactly oracle_steps, and runs one episode with
RandomAgent to verify no crashes.

Usage:
    python -m src.benchmarks.verify_envs
"""

from __future__ import annotations

import logging
import sys

from .baselines import OracleAgent, RandomAgent
from .environments import ALL_ENVIRONMENTS, TextEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_oracle(env: TextEnvironment, max_steps: int = 100) -> tuple[bool, int, str]:
    """Run one oracle episode and verify goal is reached in oracle_steps.

    Returns:
        (passed, steps_taken, message)
    """
    state = env.reset()
    agent = OracleAgent(env)
    steps = 0

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, env.goal)
        result = env.step(action)
        steps += 1
        state = result.state

        if result.done:
            if steps == env.oracle_steps:
                return True, steps, "OK"
            else:
                return False, steps, f"Goal reached in {steps} steps, expected {env.oracle_steps}"

    return False, steps, f"Goal NOT reached within {max_steps} steps"


def verify_random(env: TextEnvironment, max_steps: int = 50) -> tuple[bool, str]:
    """Run one random episode and verify it does not crash.

    Returns:
        (passed, message)
    """
    try:
        state = env.reset()
        agent = RandomAgent(seed=12345)

        for _ in range(max_steps):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, env.goal)
            result = env.step(action)
            state = result.state

            if result.done:
                break

        return True, "OK (no crash)"
    except Exception as e:
        return False, f"CRASH: {e}"


def main() -> None:
    print()
    print("=" * 80)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 80)
    print()
    print(f"{'Environment':<25} {'Oracle Steps':<15} {'Oracle Test':<25} {'Random Test':<25}")
    print("-" * 90)

    all_passed = True

    for env_cls in ALL_ENVIRONMENTS:
        env = env_cls(seed=0)
        env_name = env.name

        # Oracle verification
        oracle_passed, oracle_steps, oracle_msg = verify_oracle(env)

        # Random verification (fresh env instance)
        env_random = env_cls(seed=99)
        random_passed, random_msg = verify_random(env_random)

        oracle_status = f"PASS ({oracle_steps} steps)" if oracle_passed else f"FAIL: {oracle_msg}"
        random_status = f"PASS" if random_passed else f"FAIL: {random_msg}"

        print(f"{env_name:<25} {env.oracle_steps:<15} {oracle_status:<25} {random_status:<25}")

        if not oracle_passed or not random_passed:
            all_passed = False

    print("-" * 90)

    if all_passed:
        print("\nAll environments verified successfully.")
    else:
        print("\nSome environments FAILED verification.")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
