"""Evaluation metrics for latent planning benchmarks.

All metrics operate on episode-level data (lists of states, actions, etc.)
and return scalar values suitable for aggregation and tabulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


@dataclass
class EpisodeLog:
    """Record of a single episode for metric computation.

    Attributes:
        states: Sequence of state descriptions (including initial state).
        actions_taken: Sequence of actions the agent selected.
        oracle_actions: Sequence of oracle actions at each step.
        valid_actions_per_step: Valid action lists at each step.
        goal: Goal description string.
        reached_goal: Whether the episode ended successfully.
        max_steps: Maximum allowed steps.
    """

    states: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    oracle_actions: list[str] = field(default_factory=list)
    valid_actions_per_step: list[list[str]] = field(default_factory=list)
    goal: str = ""
    reached_goal: bool = False
    max_steps: int = 50


@dataclass
class MetricResult:
    """Container for all computed metrics from one episode."""

    steps_to_goal: int
    success: bool
    planning_efficiency: float
    cumulative_similarity_auc: float
    action_quality_mean: float


def steps_to_goal(log: EpisodeLog) -> int:
    """Number of steps the agent took to reach the goal.

    Returns ``max_steps`` if the goal was not reached.
    """
    if log.reached_goal:
        return len(log.actions_taken)
    return log.max_steps


def success_rate(logs: list[EpisodeLog]) -> float:
    """Fraction of episodes where the goal was reached within ``max_steps``."""
    if not logs:
        return 0.0
    return sum(1 for log in logs if log.reached_goal) / len(logs)


def planning_efficiency(log: EpisodeLog, oracle_steps: int) -> float:
    """Ratio of oracle steps to agent steps. 1.0 = optimal, lower = worse.

    Returns 0.0 if the agent did not reach the goal.
    """
    if not log.reached_goal:
        return 0.0
    agent_steps = len(log.actions_taken)
    if agent_steps == 0:
        return 0.0
    return oracle_steps / agent_steps


def cumulative_similarity_auc(log: EpisodeLog) -> float:
    """Area under the similarity-to-goal curve over the episode.

    Uses TF-IDF cosine similarity between each visited state and the goal.
    The AUC is normalized to [0, 1] by dividing by the number of states.
    Higher is better: an agent that quickly reaches goal-like states scores
    higher even if it takes extra steps.
    """
    if len(log.states) < 2:
        return 0.0

    corpus = [log.goal] + log.states
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    goal_vec = tfidf[0:1]
    state_vecs = tfidf[1:]
    sims = sklearn_cosine(goal_vec, state_vecs).flatten()

    # Trapezoidal AUC normalized by number of intervals
    auc = float(np.trapz(sims, dx=1.0) / max(len(sims) - 1, 1))
    return auc


def action_quality(log: EpisodeLog) -> float:
    """Mean normalized rank of the chosen action compared to oracle ranking.

    At each step, the oracle action gets rank 0 (best). The chosen action's
    rank among valid actions is computed. We return the mean fraction
    ``(n_actions - rank - 1) / (n_actions - 1)`` so that 1.0 = always picking
    oracle action, and 0.0 = always picking worst.
    """
    if not log.actions_taken:
        return 0.0

    scores: list[float] = []
    for action, oracle, valid_actions in zip(
        log.actions_taken, log.oracle_actions, log.valid_actions_per_step
    ):
        n = len(valid_actions)
        if n <= 1:
            scores.append(1.0)
            continue

        if action == oracle:
            scores.append(1.0)
        else:
            # Action is not optimal; assign a score based on position
            # Oracle is best (score 1.0), all others share equal non-optimal score
            scores.append(0.0)

    return float(np.mean(scores))


def compute_episode_metrics(log: EpisodeLog, oracle_steps: int) -> MetricResult:
    """Compute all metrics for a single episode."""
    return MetricResult(
        steps_to_goal=steps_to_goal(log),
        success=log.reached_goal,
        planning_efficiency=planning_efficiency(log, oracle_steps),
        cumulative_similarity_auc=cumulative_similarity_auc(log),
        action_quality_mean=action_quality(log),
    )
