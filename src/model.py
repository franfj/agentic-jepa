"""Agentic-JEPA model: JEPA world model for software agents.

Uses a decoder-only LLM as backbone (GPT-2, Llama, etc.), extracting last-token
embeddings as state/action representations — aligned with how agentic AI systems work.
"""

import copy

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig


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


def _get_hidden_size(config: AutoConfig) -> int:
    """Extract hidden size from model config (varies by architecture)."""
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot determine hidden size from config: {type(config)}")


def _load_backbone(backbone: str, quantize: bool = False) -> AutoModel:
    """Load a backbone model, optionally with 4-bit quantization."""
    kwargs = {}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    return AutoModel.from_pretrained(backbone, **kwargs)


class StateEncoder(nn.Module):
    """Encodes task context text into a latent state vector."""

    def __init__(self, backbone: str = "gpt2", quantize: bool = False) -> None:
        super().__init__()
        self.encoder = _load_backbone(backbone, quantize=quantize)
        self._hidden_size = _get_hidden_size(self.encoder.config)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to last-token embedding. Shape: (B, D)."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return _last_token_embedding(output.last_hidden_state, attention_mask)


class ActionEncoder(nn.Module):
    """Encodes action description text into a latent action vector.

    Shares the same backbone architecture but is independently parameterized.
    """

    def __init__(self, backbone: str = "gpt2", quantize: bool = False) -> None:
        super().__init__()
        self.encoder = _load_backbone(backbone, quantize=quantize)
        self._hidden_size = _get_hidden_size(self.encoder.config)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

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
        combined = torch.cat([state, action], dim=-1).float()  # (B, D_s + D_a)
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
        quantize: bool = False,
    ) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(backbone, quantize=quantize)
        self.action_encoder = ActionEncoder(backbone, quantize=quantize)
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

    @torch.no_grad()
    def score_actions_multistep(
        self,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        action_input_ids: torch.Tensor,
        action_attention_mask: torch.Tensor,
        goal_input_ids: torch.Tensor,
        goal_attention_mask: torch.Tensor,
        rollout_depth: int = 1,
    ) -> torch.Tensor:
        """Score candidate actions with k-step lookahead rollouts.

        For each candidate action, predicts the next state embedding, then
        greedily rolls out (rollout_depth - 1) more steps by reusing the
        predicted embedding as the new state. The final predicted state is
        scored against the goal.

        Args:
            state_*: Current state text (B=1).
            action_*: Candidate actions (N, L).
            goal_*: Goal state text (B=1).
            rollout_depth: Number of steps to look ahead (1 = original behavior).

        Returns:
            Scores tensor of shape (N,).
        """
        if rollout_depth <= 1:
            return self.score_actions(
                state_input_ids, state_attention_mask,
                action_input_ids, action_attention_mask,
                goal_input_ids, goal_attention_mask,
            )

        s_t = self.state_encoder(state_input_ids, state_attention_mask)  # (1, D)
        s_goal = self.target_encoder(goal_input_ids, goal_attention_mask)  # (1, D)
        a_all = self.action_encoder(action_input_ids, action_attention_mask)  # (N, D)

        n_actions = a_all.size(0)
        cos = nn.CosineSimilarity(dim=-1)

        scores = torch.zeros(n_actions, device=s_t.device)

        for i in range(n_actions):
            # Step 1: predict next state for action i
            s_curr = s_t  # (1, D)
            a_i = a_all[i:i+1]  # (1, D)
            s_pred = self.predictor(s_curr, a_i)  # (1, D)

            # Steps 2..k: greedily pick best action at each simulated step
            for _ in range(rollout_depth - 1):
                # Score all actions from the predicted state
                s_pred_expanded = s_pred.expand(n_actions, -1)  # (N, D)
                next_preds = self.predictor(s_pred_expanded, a_all)  # (N, D)
                goal_expanded = s_goal.expand(n_actions, -1)
                step_scores = cos(next_preds, goal_expanded)  # (N,)
                best_action_idx = step_scores.argmax()
                # Advance to the best predicted state
                s_pred = next_preds[best_action_idx:best_action_idx+1]  # (1, D)

            # Final score: similarity of rolled-out state to goal
            scores[i] = cos(s_pred, s_goal).item()

        return scores
