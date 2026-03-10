"""Verify forward pass and planning loop for Agentic-JEPA."""

import torch
import pytest

from src.model import AgenticJEPAModel, StateEncoder, ActionEncoder, JEPAPredictor


@pytest.fixture(scope="module")
def model() -> AgenticJEPAModel:
    return AgenticJEPAModel(backbone="gpt2")


def _dummy_tokens(batch_size: int = 2, seq_len: int = 16) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(0, 50257, (batch_size, seq_len)),  # GPT-2 vocab size
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


class TestStateEncoder:
    def test_output_shape(self) -> None:
        enc = StateEncoder("gpt2")
        tok = _dummy_tokens(batch_size=3)
        out = enc(tok["input_ids"], tok["attention_mask"])
        assert out.shape == (3, 768)


class TestActionEncoder:
    def test_output_shape(self) -> None:
        enc = ActionEncoder("gpt2")
        tok = _dummy_tokens(batch_size=5)
        out = enc(tok["input_ids"], tok["attention_mask"])
        assert out.shape == (5, 768)


class TestJEPAPredictor:
    def test_output_shape(self) -> None:
        pred = JEPAPredictor(state_dim=768, action_dim=768, hidden_dim=512)
        s = torch.randn(4, 768)
        a = torch.randn(4, 768)
        out = pred(s, a)
        assert out.shape == (4, 768)


class TestAgenticJEPAModel:
    def test_forward_keys(self, model: AgenticJEPAModel) -> None:
        state = _dummy_tokens()
        action = _dummy_tokens()
        next_state = _dummy_tokens()
        out = model(
            state_input_ids=state["input_ids"],
            state_attention_mask=state["attention_mask"],
            action_input_ids=action["input_ids"],
            action_attention_mask=action["attention_mask"],
            next_state_input_ids=next_state["input_ids"],
            next_state_attention_mask=next_state["attention_mask"],
        )
        assert "loss" in out
        assert "predicted" in out
        assert "target" in out

    def test_loss_is_finite(self, model: AgenticJEPAModel) -> None:
        state = _dummy_tokens()
        action = _dummy_tokens()
        next_state = _dummy_tokens()
        out = model(
            state_input_ids=state["input_ids"],
            state_attention_mask=state["attention_mask"],
            action_input_ids=action["input_ids"],
            action_attention_mask=action["attention_mask"],
            next_state_input_ids=next_state["input_ids"],
            next_state_attention_mask=next_state["attention_mask"],
        )
        assert torch.isfinite(out["loss"])

    def test_score_actions(self, model: AgenticJEPAModel) -> None:
        state = _dummy_tokens(batch_size=1)
        goal = _dummy_tokens(batch_size=1)
        actions = _dummy_tokens(batch_size=5)
        scores = model.score_actions(
            state_input_ids=state["input_ids"],
            state_attention_mask=state["attention_mask"],
            action_input_ids=actions["input_ids"],
            action_attention_mask=actions["attention_mask"],
            goal_input_ids=goal["input_ids"],
            goal_attention_mask=goal["attention_mask"],
        )
        assert scores.shape == (5,)
        assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)

    def test_target_encoder_frozen(self, model: AgenticJEPAModel) -> None:
        for p in model.target_encoder.parameters():
            assert not p.requires_grad
