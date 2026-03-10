"""Agentic-JEPA model: JEPA world model for software agents.

Uses a decoder-only LLM (GPT-2) as backbone, extracting last-token embeddings
as state/action representations — aligned with how agentic AI systems work.
"""

import copy

import torch
import torch.nn as nn
from transformers import GPT2Model


def _last_token_embedding(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last non-padding token's hidden state for each sequence.

    Works correctly with both left-padding and right-padding by finding the
    position of the last real token (last 1 in attention_mask).

    Args:
        hidden_states: (B, L, D) from the model.
        attention_mask: (B, L) with 1s for real tokens.

    Returns:
        (B, D) embeddings.
    """
    seq_len = attention_mask.size(1)
    # Create position indices and mask out padding positions with -1
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
    masked_positions = positions * attention_mask - (1 - attention_mask)
    # argmax finds the last (rightmost) real token position
    last_idx = masked_positions.argmax(dim=-1)  # (B,)
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, last_idx]


class StateEncoder(nn.Module):
    """Encodes task context text into a latent state vector using GPT-2."""

    def __init__(self, backbone: str = "gpt2") -> None:
        super().__init__()
        self.encoder = GPT2Model.from_pretrained(backbone)

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.n_embd

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to last-token embedding. Shape: (B, D)."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return _last_token_embedding(output.last_hidden_state, attention_mask)


class ActionEncoder(nn.Module):
    """Encodes action description text into a latent action vector.

    Shares the same backbone architecture but is independently parameterized.
    """

    def __init__(self, backbone: str = "gpt2") -> None:
        super().__init__()
        self.encoder = GPT2Model.from_pretrained(backbone)

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.n_embd

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode action text to last-token embedding. Shape: (B, D)."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return _last_token_embedding(output.last_hidden_state, attention_mask)


class JEPAPredictor(nn.Module):
    """Takes concatenated [s_t, a_t] and predicts next state s_hat_t+1."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(hidden_dim, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state from current state and action.

        Args:
            state: Current state embedding (B, D_s).
            action: Action embedding (B, D_a).

        Returns:
            Predicted next state embedding (B, D_s).
        """
        combined = torch.cat([state, action], dim=-1)  # (B, D_s + D_a)
        h = self.proj_in(combined).unsqueeze(1)  # (B, 1, hidden)
        h = self.transformer(h)  # (B, 1, hidden)
        return self.proj_out(h.squeeze(1))  # (B, D_s)


class AgenticJEPAModel(nn.Module):
    """Full Agentic-JEPA model.

    Components:
    - state_encoder: GPT-2 encodes task context → s_t (last-token pooling)
    - action_encoder: GPT-2 encodes candidate action → a_t (last-token pooling)
    - predictor: [s_t, a_t] → s_hat_t+1
    - target_encoder: EMA of state_encoder (for training loss)
    """

    def __init__(
        self,
        backbone: str = "gpt2",
        predictor_hidden: int = 512,
        predictor_heads: int = 8,
        predictor_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(backbone)
        self.action_encoder = ActionEncoder(backbone)
        self.target_encoder = copy.deepcopy(self.state_encoder)

        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        d = self.state_encoder.hidden_size
        self.predictor = JEPAPredictor(
            state_dim=d,
            action_dim=self.action_encoder.hidden_size,
            hidden_dim=predictor_hidden,
            num_heads=predictor_heads,
            num_layers=predictor_layers,
        )

        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(
        self,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        action_input_ids: torch.Tensor,
        action_attention_mask: torch.Tensor,
        next_state_input_ids: torch.Tensor,
        next_state_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            state_*: Current state text tokens.
            action_*: Action text tokens.
            next_state_*: Ground truth next state text tokens.

        Returns:
            Dict with 'loss', 'predicted', 'target'.
        """
        s_t = self.state_encoder(state_input_ids, state_attention_mask)
        a_t = self.action_encoder(action_input_ids, action_attention_mask)
        predicted = self.predictor(s_t, a_t)

        with torch.no_grad():
            target = self.target_encoder(next_state_input_ids, next_state_attention_mask)

        labels = torch.ones(predicted.size(0), device=predicted.device)
        loss = self.loss_fn(predicted, target, labels)

        return {"loss": loss, "predicted": predicted, "target": target}

    @torch.no_grad()
    def score_actions(
        self,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        action_input_ids: torch.Tensor,
        action_attention_mask: torch.Tensor,
        goal_input_ids: torch.Tensor,
        goal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate actions by cosine similarity to goal state.

        Args:
            state_*: Current state text (B=1).
            action_*: Candidate actions (N, L).
            goal_*: Goal state text (B=1).

        Returns:
            Scores tensor of shape (N,).
        """
        s_t = self.state_encoder(state_input_ids, state_attention_mask)  # (1, D)
        # Goal encoded with target_encoder: the predictor was trained to predict
        # target_encoder representations, so comparison must be in that space.
        s_goal = self.target_encoder(goal_input_ids, goal_attention_mask)  # (1, D)

        a_t = self.action_encoder(action_input_ids, action_attention_mask)  # (N, D)

        # Expand state to match N actions
        s_t_expanded = s_t.expand(a_t.size(0), -1)  # (N, D)
        predicted = self.predictor(s_t_expanded, a_t)  # (N, D)

        cos = nn.CosineSimilarity(dim=-1)
        scores = cos(predicted, s_goal.expand(predicted.size(0), -1))  # (N,)
        return scores
