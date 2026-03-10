"""PyTorch dataset for JEPA trajectory data.

Loads JSONL trajectory files and yields tokenized (state, action, next_state) triples
for training the JEPA predictor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """Dataset of (state, action, next_state) transitions from JSONL trajectory files.

    Each JSONL line is a trajectory dict containing a list of transitions.
    This dataset flattens all transitions across all trajectories into individual
    training samples.

    The GPT-2 tokenizer uses left-padding (pad_token = eos_token) so that the
    last token is always meaningful for the decoder-only architecture.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str = "gpt2",
        max_length: int = 128,
    ) -> None:
        """
        Args:
            jsonl_path: Path to JSONL trajectory file.
            tokenizer_name: HuggingFace tokenizer name (must match model backbone).
            max_length: Maximum token sequence length.
        """
        self.max_length = max_length

        # Set up tokenizer with left-padding for decoder-only models
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load and flatten all transitions
        self.transitions: list[dict[str, str | bool]] = []
        self._load_transitions(jsonl_path)

        logger.info(
            f"Loaded {len(self.transitions)} transitions from {jsonl_path}"
        )

    def _load_transitions(self, jsonl_path: str) -> None:
        """Load JSONL file and flatten trajectories into individual transitions."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {jsonl_path}")

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trajectory = json.loads(line)
                for transition in trajectory["transitions"]:
                    self.transitions.append(transition)

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize a single text string with left-padding."""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Squeeze batch dimension: (1, L) -> (L,)
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return tokenized (state, action, next_state) triple.

        Returns:
            Dict with keys:
                state_input_ids, state_attention_mask,
                action_input_ids, action_attention_mask,
                next_state_input_ids, next_state_attention_mask
        """
        transition = self.transitions[idx]

        state_tok = self._tokenize(transition["state"])
        action_tok = self._tokenize(transition["action"])
        next_state_tok = self._tokenize(transition["next_state"])

        return {
            "state_input_ids": state_tok["input_ids"],
            "state_attention_mask": state_tok["attention_mask"],
            "action_input_ids": action_tok["input_ids"],
            "action_attention_mask": action_tok["attention_mask"],
            "next_state_input_ids": next_state_tok["input_ids"],
            "next_state_attention_mask": next_state_tok["attention_mask"],
        }


def build_dataloaders(
    jsonl_path: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 128,
    batch_size: int = 16,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a JSONL trajectory file.

    Args:
        jsonl_path: Path to JSONL trajectory file.
        tokenizer_name: HuggingFace tokenizer name.
        max_length: Maximum token sequence length.
        batch_size: Batch size for both loaders.
        val_fraction: Fraction of data to use for validation.
        num_workers: Number of dataloader worker processes.
        seed: Random seed for the train/val split.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = TrajectoryDataset(
        jsonl_path=jsonl_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )

    total = len(dataset)
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    logger.info(f"Train: {train_size} transitions, Val: {val_size} transitions")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
