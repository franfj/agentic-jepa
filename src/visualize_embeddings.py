"""Visualize JEPA state and goal embeddings across environments.

Loads a trained checkpoint, runs oracle trajectories through all 8 environments,
encodes states and goals into the learned latent space, reduces to 2D with
UMAP (or t-SNE fallback), and produces a publication-quality scatter plot.

Usage:
    python -m src.visualize_embeddings --checkpoint checkpoints/best.pt --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from .model import AgenticJEPAModel
from .benchmarks.baselines import OracleAgent
from .benchmarks.environments import TextEnvironment, make_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colorblind-friendly palette (8 colors)
ENV_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tokenize(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize with left-padding, matching training convention."""
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


@torch.no_grad()
def _encode_texts(
    texts: list[str],
    model: AgenticJEPAModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
    use_target_encoder: bool = False,
) -> np.ndarray:
    """Encode a list of texts into latent embeddings.

    Args:
        texts: Strings to encode.
        model: Trained JEPA model.
        tokenizer: Tokenizer (left-padding configured).
        max_length: Max token length.
        device: Torch device.
        use_target_encoder: If True, use the EMA target encoder (for goals).

    Returns:
        (N, D) numpy array of embeddings.
    """
    tok = _tokenize(texts, tokenizer, max_length, device)
    encoder = model.target_encoder if use_target_encoder else model.state_encoder
    embeddings = encoder(tok["input_ids"], tok["attention_mask"])
    return embeddings.cpu().numpy()


def collect_trajectories(
    envs: list[TextEnvironment],
    max_steps: int = 50,
) -> list[dict]:
    """Run oracle agent on each environment and collect state trajectories.

    Returns:
        List of dicts with keys: 'env_name', 'states', 'goal'.
    """
    trajectories = []
    for env in envs:
        oracle = OracleAgent(env)
        state = env.reset()
        states = [state]

        for _ in range(max_steps):
            if env.done:
                break
            valid_actions = env.get_valid_actions()
            action = oracle.select_action(state, valid_actions, env.goal)
            result = env.step(action)
            state = result.state
            states.append(state)
            if result.done:
                break

        trajectories.append({
            "env_name": env.name,
            "states": states,
            "goal": env.goal,
        })

    return trajectories


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D with UMAP or t-SNE fallback."""
    try:
        from umap import UMAP
        logger.info("Using UMAP for dimensionality reduction.")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.manifold import TSNE
        logger.info("UMAP not available; falling back to t-SNE.")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        return reducer.fit_transform(embeddings)


def build_plot(
    trajectories: list[dict],
    state_coords: np.ndarray,
    goal_coords: np.ndarray,
    output_dir: Path,
) -> None:
    """Create and save the embedding visualization.

    Args:
        trajectories: List of trajectory dicts from collect_trajectories.
        state_coords: (N_states, 2) array of 2D state coordinates.
        goal_coords: (N_envs, 2) array of 2D goal coordinates.
        output_dir: Directory to save figures.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    offset = 0
    for env_idx, traj in enumerate(trajectories):
        n_states = len(traj["states"])
        color = ENV_COLORS[env_idx % len(ENV_COLORS)]
        coords = state_coords[offset : offset + n_states]

        # Plot state points
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color,
            s=40,
            alpha=0.7,
            label=traj["env_name"],
            zorder=3,
        )

        # Draw arrows for transitions
        for i in range(n_states - 1):
            dx = coords[i + 1, 0] - coords[i, 0]
            dy = coords[i + 1, 1] - coords[i, 1]
            ax.annotate(
                "",
                xy=(coords[i + 1, 0], coords[i + 1, 1]),
                xytext=(coords[i, 0], coords[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=0.5,
                    lw=1.2,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=2,
            )

        # Mark start state
        ax.scatter(
            coords[0, 0],
            coords[0, 1],
            c=color,
            s=100,
            marker="o",
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )

        offset += n_states

    # Plot goal states as stars
    for env_idx, traj in enumerate(trajectories):
        color = ENV_COLORS[env_idx % len(ENV_COLORS)]
        ax.scatter(
            goal_coords[env_idx, 0],
            goal_coords[env_idx, 1],
            c=color,
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

    # Add a single star entry to the legend for goals
    ax.scatter([], [], c="gray", s=250, marker="*", edgecolors="black", linewidths=1.0, label="Goal state")
    ax.scatter([], [], c="gray", s=100, marker="o", edgecolors="black", linewidths=1.5, label="Start state")

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title("Agentic-JEPA Embedding Space: Oracle Trajectories Across Environments", fontsize=14)
    ax.legend(
        loc="best",
        fontsize=9,
        framealpha=0.9,
        ncol=2,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "embedding_space.png"
    pdf_path = output_dir / "embedding_space.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved figures to {png_path} and {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize JEPA embedding space across environments."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained JEPA model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for model hyperparameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures (default: figures/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for environments.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per oracle trajectory.",
    )
    args = parser.parse_args()

    # Load config
    config: dict = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Setup device
    device = _get_device()
    logger.info(f"Using device: {device}")

    # Load model
    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "gpt2")
    max_length = model_cfg.get("max_length", 128)

    model = AgenticJEPAModel(
        backbone=backbone,
        predictor_hidden=model_cfg.get("predictor_hidden", 512),
        predictor_heads=model_cfg.get("predictor_heads", 8),
        predictor_layers=model_cfg.get("predictor_layers", 2),
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Collect oracle trajectories from all 8 environments
    envs = make_all(seed=args.seed)
    logger.info(f"Collecting oracle trajectories from {len(envs)} environments...")
    trajectories = collect_trajectories(envs, max_steps=args.max_steps)

    for traj in trajectories:
        logger.info(f"  {traj['env_name']}: {len(traj['states'])} states")

    # Encode all states
    all_state_texts: list[str] = []
    for traj in trajectories:
        all_state_texts.extend(traj["states"])

    logger.info(f"Encoding {len(all_state_texts)} state texts...")
    state_embeddings = _encode_texts(
        all_state_texts, model, tokenizer, max_length, device, use_target_encoder=False
    )

    # Encode all goals (using target encoder, matching score_actions convention)
    goal_texts = [traj["goal"] for traj in trajectories]
    logger.info(f"Encoding {len(goal_texts)} goal texts...")
    goal_embeddings = _encode_texts(
        goal_texts, model, tokenizer, max_length, device, use_target_encoder=True
    )

    # Combine for joint dimensionality reduction
    all_embeddings = np.concatenate([state_embeddings, goal_embeddings], axis=0)
    logger.info(f"Reducing {all_embeddings.shape[0]} embeddings from {all_embeddings.shape[1]}D to 2D...")
    coords_2d = reduce_to_2d(all_embeddings)

    n_states = state_embeddings.shape[0]
    state_coords = coords_2d[:n_states]
    goal_coords = coords_2d[n_states:]

    # Build and save the plot
    output_dir = Path(args.output_dir)
    build_plot(trajectories, state_coords, goal_coords, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
