"""Training loop for Agentic-JEPA (self-supervised state prediction).

Trains the JEPA world model on trajectory data: the predictor learns to map
(state_t, action_t) -> predicted_state_t+1, with the target provided by an
EMA-updated copy of the state encoder.

Usage:
    python -m src.train --config configs/default.yaml --data data/trajectories.jsonl
    python -m src.train --config configs/small_test.yaml --data data/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import yaml

from .data.dataset import build_dataloaders
from .ema import EMAUpdater
from .model import AgenticJEPAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # Cosine decay from base_lr to min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def validate(
    model: AgenticJEPAModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Run validation and return mean loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                state_input_ids=batch["state_input_ids"],
                state_attention_mask=batch["state_attention_mask"],
                action_input_ids=batch["action_input_ids"],
                action_attention_mask=batch["action_attention_mask"],
                next_state_input_ids=batch["next_state_input_ids"],
                next_state_attention_mask=batch["next_state_attention_mask"],
            )
            total_loss += outputs["loss"].item()
            count += 1

    model.train()
    return total_loss / max(count, 1)


def train(config: dict, data_path: str) -> None:
    """Full training loop for Agentic-JEPA.

    Args:
        config: Parsed YAML configuration dict.
        data_path: Path to JSONL trajectory file.
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    torch.manual_seed(config["seed"])

    # --- Model ---
    model_cfg = config["model"]
    quantize = model_cfg.get("quantize", False)
    model = AgenticJEPAModel(
        backbone=model_cfg["backbone"],
        predictor_hidden=model_cfg["predictor_hidden"],
        predictor_heads=model_cfg["predictor_heads"],
        predictor_layers=model_cfg["predictor_layers"],
        quantize=quantize,
    )
    if not quantize:
        model = model.to(device)
    else:
        # With quantization, backbones are already on GPU via device_map.
        # Move predictor, target encoder, and loss to the same device.
        model.predictor = model.predictor.to(device)
        model.target_encoder = model.target_encoder.to(device)
        model.loss_fn = model.loss_fn.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --- Data ---
    train_cfg = config["training"]
    train_loader, val_loader = build_dataloaders(
        jsonl_path=data_path,
        tokenizer_name=model_cfg["backbone"],
        max_length=model_cfg["max_length"],
        batch_size=train_cfg["batch_size"],
        val_fraction=0.1,
        num_workers=0,
        seed=config["seed"],
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # --- EMA ---
    ema_cfg = config["ema"]
    ema = EMAUpdater(
        momentum=ema_cfg["momentum"],
        momentum_end=ema_cfg["momentum_end"],
        warmup_steps=ema_cfg["warmup_steps"],
    )

    # --- Wandb ---
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "agentic-jepa"),
                entity=wandb_cfg.get("entity"),
                config=config,
            )
        except ImportError:
            logger.warning("wandb not installed; disabling wandb logging.")
            use_wandb = False

    # --- Output directory ---
    output_dir = Path(train_cfg.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    max_steps = train_cfg["max_steps"]
    log_every = train_cfg.get("log_every", 50)
    checkpoint_every = train_cfg.get("checkpoint_every", 1000)
    warmup_steps = train_cfg["warmup_steps"]
    base_lr = train_cfg["learning_rate"]

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    running_loss = 0.0
    log_steps = 0
    start_time = time.time()

    train_log: dict = {
        "config": config,
        "steps": [],
    }

    logger.info(f"Starting training for {max_steps} steps")
    logger.info(f"Batch size: {train_cfg['batch_size']}, LR: {base_lr}")

    data_iter = iter(train_loader)

    while global_step < max_steps:
        # Get next batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # Linear warmup + cosine decay LR schedule
        lr = get_lr(global_step, warmup_steps, max_steps, base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        outputs = model(
            state_input_ids=batch["state_input_ids"],
            state_attention_mask=batch["state_attention_mask"],
            action_input_ids=batch["action_input_ids"],
            action_attention_mask=batch["action_attention_mask"],
            next_state_input_ids=batch["next_state_input_ids"],
            next_state_attention_mask=batch["next_state_attention_mask"],
        )
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        # EMA update of target encoder
        momentum = ema.momentum_at(global_step)
        EMAUpdater.update(model.state_encoder, model.target_encoder, momentum)

        # Tracking
        running_loss += loss.item()
        log_steps += 1
        global_step += 1

        # Logging
        if global_step % log_every == 0:
            avg_loss = running_loss / log_steps
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed

            # Cosine similarity between predicted and target (diagnostic)
            with torch.no_grad():
                cos_sim = torch.nn.functional.cosine_similarity(
                    outputs["predicted"], outputs["target"], dim=-1
                ).mean().item()

            log_entry = {
                "step": global_step,
                "train_loss": avg_loss,
                "lr": lr,
                "ema_momentum": momentum,
                "cos_sim": cos_sim,
                "steps_per_sec": steps_per_sec,
            }

            logger.info(
                f"Step {global_step}/{max_steps} | "
                f"loss={avg_loss:.4f} | cos_sim={cos_sim:.4f} | "
                f"lr={lr:.2e} | ema={momentum:.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )

            if use_wandb:
                import wandb
                wandb.log(log_entry, step=global_step)

            train_log["steps"].append(log_entry)
            running_loss = 0.0
            log_steps = 0

        # Checkpoint
        if global_step % checkpoint_every == 0 or global_step == max_steps:
            # Validate
            val_loss = validate(model, val_loader, device)
            logger.info(f"Step {global_step} | val_loss={val_loss:.4f}")

            if use_wandb:
                import wandb
                wandb.log({"val_loss": val_loss}, step=global_step)

            # Save checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "val_loss": val_loss,
                "config": config,
            }

            ckpt_path = output_dir / f"checkpoint_{global_step}.pt"
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best model (val_loss={val_loss:.4f}): {best_path}")

            model.train()

    # --- Finalize ---
    elapsed = time.time() - start_time
    logger.info(f"Training complete. {global_step} steps in {elapsed:.1f}s")
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    train_log["final_loss"] = running_loss / max(log_steps, 1)
    train_log["best_val_loss"] = best_val_loss
    train_log["total_steps"] = global_step
    train_log["elapsed_seconds"] = elapsed

    # Save training log
    log_path = output_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2, default=str)
    logger.info(f"Training log saved: {log_path}")

    if use_wandb:
        import wandb
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Agentic-JEPA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSONL trajectory file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args.data)


if __name__ == "__main__":
    main()
