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
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(_trapz(sims, dx=1.0) / max(len(sims) - 1, 1))
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


# ======================================================================
# Statistical significance tests
# ======================================================================


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test: is agent A significantly better than B?

    Tests H0: mean(A) <= mean(B) vs H1: mean(A) > mean(B).

    Args:
        scores_a: Per-episode scores for agent A (e.g., success 0/1).
        scores_b: Per-episode scores for agent B.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with 'delta' (observed mean difference), 'p_value',
        'ci_lower', 'ci_upper' (95% CI of the difference).
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    observed_delta = float(a.mean() - b.mean())

    # Bootstrap the difference
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()

    # One-sided p-value: fraction of bootstrap samples where A is not better
    p_value = float(np.mean(deltas <= 0))

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return {
        "delta": observed_delta,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def wilcoxon_signed_rank_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Tests whether the
    distribution of differences is symmetric around zero.

    Returns:
        Dict with 'statistic', 'p_value', 'significant_at_005'.
    """
    from scipy import stats

    a = np.array(scores_a)
    b = np.array(scores_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    # Remove tied pairs (difference = 0)
    diff = a - b
    nonzero = diff != 0

    if nonzero.sum() < 5:
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant_at_005": False,
            "note": "Too few non-tied pairs for Wilcoxon test",
        }

    stat, p = stats.wilcoxon(a[nonzero], b[nonzero], alternative="greater")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_at_005": p < 0.05,
        "significant_at_001": p < 0.01,
    }


def compute_significance_matrix(
    agent_episode_scores: dict[str, list[float]],
    test_type: str = "bootstrap",
) -> dict[str, dict[str, dict]]:
    """Compute pairwise significance tests between all agents.

    Args:
        agent_episode_scores: {agent_name: [per_episode_scores]}.
        test_type: "bootstrap" or "wilcoxon".

    Returns:
        Nested dict: result[agent_a][agent_b] = test_result.
    """
    agents = sorted(agent_episode_scores.keys())
    results = {}

    for a in agents:
        results[a] = {}
        for b in agents:
            if a == b:
                continue
            if test_type == "wilcoxon":
                results[a][b] = wilcoxon_signed_rank_test(
                    agent_episode_scores[a],
                    agent_episode_scores[b],
                )
            else:
                results[a][b] = paired_bootstrap_test(
                    agent_episode_scores[a],
                    agent_episode_scores[b],
                )
    return results
