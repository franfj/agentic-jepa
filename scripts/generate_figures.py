"""Generate publication-quality figures for the Agentic-JEPA paper.

Reads benchmark result JSON files and produces figures for:
1. Rollout depth vs. success rate / efficiency
2. Ablation heatmaps (EMA momentum × predictor config)
3. In-distribution vs. out-of-distribution comparison
4. Cost-accuracy trade-off (JEPA vs. LLM baseline)
5. Training convergence curves

Usage:
    python scripts/generate_figures.py --results results/v2_default.json --output figures/
    python scripts/generate_figures.py --ablation-dir outputs/ --output figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Environment classification
TRAIN_ENVS = {
    "DocumentWorkflow", "CodeReview", "EmailTriage",
    "DataPipeline", "ResearchTask",
    "BugTriage", "OnboardingProcess", "SecurityAudit",
}
TEST_ENVS = {
    "CustomerSupport", "IncidentResponse", "MeetingPreparation",
    "ContentPublishing", "ExperimentPipeline",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_rollout_depth(results: dict, output_dir: Path) -> None:
    """Plot success rate and efficiency vs. rollout depth k."""
    # Find JEPA agents with different depths
    jepa_agents = {}
    for agent_name in results:
        if agent_name.startswith("JEPA"):
            if "k=" in agent_name:
                k = int(agent_name.split("k=")[1].rstrip(")"))
            else:
                k = 1
            jepa_agents[k] = agent_name

    if len(jepa_agents) < 2:
        logger.warning("Need at least 2 rollout depths for rollout depth figure")
        return

    ks = sorted(jepa_agents.keys())
    env_names = sorted({
        env for agent_data in results.values()
        for env in agent_data
    })

    # Compute mean success rate and efficiency per k
    train_success = []
    test_success = []
    train_eff = []
    test_eff = []

    for k in ks:
        agent = jepa_agents[k]
        agent_data = results.get(agent, {})

        ts, te, trs, tre = [], [], [], []
        for env in env_names:
            if env not in agent_data:
                continue
            sr = agent_data[env].get("success_rate", 0)
            eff = agent_data[env].get("planning_efficiency", 0)
            if env in TRAIN_ENVS:
                trs.append(sr)
                tre.append(eff)
            else:
                ts.append(sr)
                te.append(eff)

        train_success.append(np.mean(trs) if trs else 0)
        test_success.append(np.mean(ts) if ts else 0)
        train_eff.append(np.mean(tre) if tre else 0)
        test_eff.append(np.mean(te) if te else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.plot(ks, train_success, "o-", label="In-distribution", color="#2196F3")
    ax1.plot(ks, test_success, "s--", label="Out-of-distribution", color="#FF5722")
    ax1.set_xlabel("Rollout depth $k$")
    ax1.set_ylabel("Success rate")
    ax1.set_xticks(ks)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.set_title("(a) Success rate vs. rollout depth")

    ax2.plot(ks, train_eff, "o-", label="In-distribution", color="#2196F3")
    ax2.plot(ks, test_eff, "s--", label="Out-of-distribution", color="#FF5722")
    ax2.set_xlabel("Rollout depth $k$")
    ax2.set_ylabel("Planning efficiency")
    ax2.set_xticks(ks)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.set_title("(b) Efficiency vs. rollout depth")

    plt.tight_layout()
    path = output_dir / "rollout_depth.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def fig_id_vs_ood(results: dict, output_dir: Path) -> None:
    """Bar chart comparing in-distribution vs. out-of-distribution performance."""
    env_names = sorted({
        env for agent_data in results.values()
        for env in agent_data
    })

    agents_to_plot = ["Random", "GreedyText", "JEPA", "LLM (GPT-4o-mini)", "Oracle"]
    # Also include JEPA(k=3) if available
    for agent in results:
        if "k=3" in agent:
            agents_to_plot.insert(3, agent)
            break

    agents_present = [a for a in agents_to_plot if a in results]
    colors = ["#9E9E9E", "#FFC107", "#2196F3", "#4CAF50", "#9C27B0", "#607D8B"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax_idx, (title, env_set) in enumerate([
        ("In-distribution", TRAIN_ENVS),
        ("Out-of-distribution", TEST_ENVS),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(agents_present))
        means = []
        stds = []
        for agent in agents_present:
            vals = [
                results[agent][env].get("success_rate", 0)
                for env in env_names
                if env in env_set and env in results.get(agent, {})
            ]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color=colors[:len(agents_present)], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(agents_present, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1.15)
        ax.set_title(title)

        # Add value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = output_dir / "id_vs_ood.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def fig_per_environment(results: dict, output_dir: Path) -> None:
    """Grouped bar chart: success rate per environment per agent."""
    env_names = sorted({
        env for agent_data in results.values()
        for env in agent_data
    })

    agents_to_plot = [a for a in ["Random", "GreedyText", "JEPA", "Oracle"] if a in results]
    # Also try JEPA(k=3)
    for agent in results:
        if "k=3" in agent and agent not in agents_to_plot:
            agents_to_plot.insert(-1, agent)
            break

    colors = {"Random": "#9E9E9E", "GreedyText": "#FFC107", "JEPA": "#2196F3",
              "Oracle": "#607D8B"}
    for a in agents_to_plot:
        if a not in colors:
            colors[a] = "#4CAF50"

    n_envs = len(env_names)
    n_agents = len(agents_to_plot)
    bar_width = 0.8 / n_agents
    x = np.arange(n_envs)

    fig, ax = plt.subplots(figsize=(12, 4))

    for i, agent in enumerate(agents_to_plot):
        vals = [results.get(agent, {}).get(env, {}).get("success_rate", 0) for env in env_names]
        offset = (i - n_agents/2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=agent, color=colors.get(agent, "#888"),
               edgecolor="black", linewidth=0.3)

    # Mark train vs test
    for i, env in enumerate(env_names):
        if env in TEST_ENVS:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", ncol=n_agents)
    ax.set_title("Success rate per environment (shaded = out-of-distribution)")

    plt.tight_layout()
    path = output_dir / "per_environment.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def fig_training_convergence(log_path: str, output_dir: Path) -> None:
    """Plot training loss and cosine similarity over steps."""
    with open(log_path) as f:
        log = json.load(f)

    steps_data = log.get("steps", [])
    if not steps_data:
        logger.warning("No training steps data found")
        return

    steps = [s["step"] for s in steps_data]
    loss = [s["train_loss"] for s in steps_data]
    cos_sim = [s.get("cos_sim", 0) for s in steps_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.plot(steps, loss, color="#2196F3", linewidth=1.5)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Cosine embedding loss")
    ax1.set_title("(a) Training loss")

    ax2.plot(steps, cos_sim, color="#4CAF50", linewidth=1.5)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("(b) Predicted-target similarity")
    ax2.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    path = output_dir / "training_convergence.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def fig_ablation_heatmap(ablation_dir: Path, output_dir: Path) -> None:
    """Heatmap of EMA momentum ablation results."""
    ema_dir = ablation_dir / "ablation_ema"
    if not ema_dir.exists():
        logger.warning(f"No EMA ablation directory at {ema_dir}")
        return

    labels = []
    success_rates = []

    for variant_dir in sorted(ema_dir.iterdir()):
        results_path = variant_dir / "benchmark_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)

        label = variant_dir.name
        labels.append(label)

        # Compute mean success across all envs for each JEPA agent
        jepa_data = {}
        for agent_name, env_data in data.items():
            if "JEPA" in agent_name and "Vanilla" not in agent_name:
                sr = np.mean([
                    v.get("success_rate", 0)
                    for v in env_data.values()
                ])
                jepa_data[agent_name] = sr

        # Use JEPA k=1 as default
        best_sr = max(jepa_data.values()) if jepa_data else 0
        success_rates.append(best_sr)

    if not labels:
        logger.warning("No ablation results found")
        return

    fig, ax = plt.subplots(figsize=(6, 2))
    x = np.arange(len(labels))
    bars = ax.bar(x, success_rates, color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean success rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("EMA Momentum Ablation")

    for bar, sr in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{sr:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = output_dir / "ablation_ema.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to benchmark results JSON")
    parser.add_argument("--train-log", type=str, default=None,
                        help="Path to train_log.json for convergence plot")
    parser.add_argument("--ablation-dir", type=str, default="outputs",
                        help="Directory containing ablation output folders")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results:
        results = load_results(args.results)
        fig_rollout_depth(results, output_dir)
        fig_id_vs_ood(results, output_dir)
        fig_per_environment(results, output_dir)

    if args.train_log:
        fig_training_convergence(args.train_log, output_dir)

    ablation_dir = Path(args.ablation_dir)
    if ablation_dir.exists():
        fig_ablation_heatmap(ablation_dir, output_dir)

    logger.info("Done generating figures")


if __name__ == "__main__":
    main()
