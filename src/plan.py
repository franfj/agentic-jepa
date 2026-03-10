"""Planning loop: score and rank candidate actions using JEPA predictor."""

import argparse
import logging

import torch
import yaml
from transformers import AutoTokenizer

from .model import AgenticJEPAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def tokenize(tokenizer: AutoTokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Tokenize a list of texts into a batch."""
    encoded = tokenizer(
        texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in encoded.items()}


def plan(config: dict, checkpoint_path: str | None = None) -> None:
    """Run the planning loop on the toy environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    model_cfg = config["model"]
    model = AgenticJEPAModel(
        backbone=model_cfg["backbone"],
        predictor_hidden=model_cfg["predictor_hidden"],
        predictor_heads=model_cfg["predictor_heads"],
        predictor_layers=model_cfg["predictor_layers"],
    ).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = model_cfg["max_length"]

    env = config["toy_env"]
    goal = env["goal"]
    current_state = env["initial_state"]
    actions = env["actions"]

    logger.info(f"Goal: {goal}")
    logger.info(f"Initial state: {current_state}")
    logger.info(f"Candidate actions: {len(actions)}")
    logger.info("---")

    # Tokenize
    state_tok = tokenize(tokenizer, [current_state], max_len, device)
    goal_tok = tokenize(tokenizer, [goal], max_len, device)
    actions_tok = tokenize(tokenizer, actions, max_len, device)

    # Score actions
    scores = model.score_actions(
        state_input_ids=state_tok["input_ids"],
        state_attention_mask=state_tok["attention_mask"],
        action_input_ids=actions_tok["input_ids"],
        action_attention_mask=actions_tok["attention_mask"],
        goal_input_ids=goal_tok["input_ids"],
        goal_attention_mask=goal_tok["attention_mask"],
    )

    # Rank
    top_k = config["planning"].get("top_k", 3)
    ranked_indices = scores.argsort(descending=True)

    logger.info("Action ranking (by predicted similarity to goal):")
    for rank, idx in enumerate(ranked_indices[:top_k]):
        logger.info(f"  #{rank + 1}: [{scores[idx]:.4f}] {actions[idx]}")

    logger.info(f"\nBest action: {actions[ranked_indices[0]]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic-JEPA Planning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    plan(config, args.checkpoint)


if __name__ == "__main__":
    main()
