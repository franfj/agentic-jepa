"""Baseline agents for benchmark comparison.

All agents implement the same interface: ``select_action(state, valid_actions, goal) -> action``.
This allows drop-in comparison across environments.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from transformers import AutoTokenizer

from ..model import AgenticJEPAModel
from .environments import TextEnvironment


class BaseAgent(ABC):
    """Interface for all benchmark agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier for reporting."""

    @abstractmethod
    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        """Choose an action given the current state and goal.

        Args:
            state: Natural language description of the current state.
            valid_actions: List of valid action strings.
            goal: Natural language description of the goal.

        Returns:
            The selected action string (must be in ``valid_actions``).
        """


class RandomAgent(BaseAgent):
    """Picks a uniformly random valid action each step."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        return self._rng.choice(valid_actions)


class GreedyTextAgent(BaseAgent):
    """Picks the action whose text is most similar to the goal using TF-IDF cosine.

    This is a strong text-only baseline that uses no learned world model.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()

    @property
    def name(self) -> str:
        return "GreedyText"

    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        corpus = [goal] + valid_actions
        tfidf = self._vectorizer.fit_transform(corpus)
        goal_vec = tfidf[0:1]
        action_vecs = tfidf[1:]
        sims = sklearn_cosine(goal_vec, action_vecs).flatten()
        return valid_actions[int(sims.argmax())]


class JEPAPlanningAgent(BaseAgent):
    """Uses the AgenticJEPAModel's latent predictor to score actions.

    For each candidate action, the JEPA predictor simulates the next latent
    state and selects the action whose predicted state is closest to the
    goal embedding.
    """

    def __init__(
        self,
        model: AgenticJEPAModel,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        device: torch.device | None = None,
        rollout_depth: int = 1,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        # Match training padding side (left-padding for decoder-only models)
        self._tokenizer.padding_side = "left"
        self._max_length = max_length
        self._device = device or next(model.parameters()).device
        self._rollout_depth = rollout_depth
        self._model.eval()

    @property
    def name(self) -> str:
        if self._rollout_depth > 1:
            return f"JEPA(k={self._rollout_depth})"
        return "JEPA"

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer(
            texts,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in encoded.items()}

    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        state_tok = self._tokenize([state])
        goal_tok = self._tokenize([goal])
        actions_tok = self._tokenize(valid_actions)

        scores = self._model.score_actions_multistep(
            state_input_ids=state_tok["input_ids"],
            state_attention_mask=state_tok["attention_mask"],
            action_input_ids=actions_tok["input_ids"],
            action_attention_mask=actions_tok["attention_mask"],
            goal_input_ids=goal_tok["input_ids"],
            goal_attention_mask=goal_tok["attention_mask"],
            rollout_depth=self._rollout_depth,
        )
        best_idx = scores.argmax().item()
        return valid_actions[best_idx]


class GPT2VanillaAgent(BaseAgent):
    """Uses GPT-2 pretrained (no JEPA training) with the same architecture.

    Same as JEPAPlanningAgent but with untrained JEPA components.
    This isolates the contribution of JEPA training: if JEPA >> GPT2Vanilla,
    the improvement comes from the world model, not from GPT-2 representations.
    """

    def __init__(
        self,
        backbone: str = "gpt2",
        max_length: int = 128,
        device: torch.device | None = None,
    ) -> None:
        # Fresh model with random predictor — no JEPA training
        self._model = AgenticJEPAModel(backbone=backbone)
        self._tokenizer = AutoTokenizer.from_pretrained(backbone)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Match training padding side (left-padding for decoder-only models)
        self._tokenizer.padding_side = "left"
        self._max_length = max_length
        self._device = device or torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    @property
    def name(self) -> str:
        return "GPT2Vanilla"

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer(
            texts,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in encoded.items()}

    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        state_tok = self._tokenize([state])
        goal_tok = self._tokenize([goal])
        actions_tok = self._tokenize(valid_actions)

        scores = self._model.score_actions(
            state_input_ids=state_tok["input_ids"],
            state_attention_mask=state_tok["attention_mask"],
            action_input_ids=actions_tok["input_ids"],
            action_attention_mask=actions_tok["attention_mask"],
            goal_input_ids=goal_tok["input_ids"],
            goal_attention_mask=goal_tok["attention_mask"],
        )
        best_idx = scores.argmax().item()
        return valid_actions[best_idx]


class LLMBaselineAgent(BaseAgent):
    """Zero-shot LLM baseline using OpenAI's GPT-4o-mini.

    Sends the current state, goal, and valid actions to the model and asks it
    to pick the best action. Tracks cumulative cost and latency.
    Requires the ``openai`` package and a valid ``OPENAI_API_KEY`` environment variable.
    """

    # Pricing per 1M tokens (GPT-4o-mini, as of 2026)
    INPUT_COST_PER_M = 0.15
    OUTPUT_COST_PER_M = 0.60

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        import time
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency_ms = 0.0
        self._total_calls = 0

    @property
    def name(self) -> str:
        return "LLM (GPT-4o-mini)"

    @property
    def total_cost_usd(self) -> float:
        return (
            self._total_input_tokens * self.INPUT_COST_PER_M / 1_000_000
            + self._total_output_tokens * self.OUTPUT_COST_PER_M / 1_000_000
        )

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / max(self._total_calls, 1)

    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        import time

        numbered = "\n".join(f"{i}: {a}" for i, a in enumerate(valid_actions))
        prompt = (
            f"You are an agent navigating an environment.\n\n"
            f"Current state: {state}\n"
            f"Goal: {goal}\n\n"
            f"Valid actions:\n{numbered}\n\n"
            f"Pick the single best action to make progress toward the goal. "
            f"Respond with ONLY the action number (a single integer)."
        )

        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=16,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        # Track usage
        self._total_calls += 1
        self._total_latency_ms += elapsed_ms
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens

        text = response.choices[0].message.content.strip()

        # Parse the number from the response
        # Handle cases like "0", "Action 0", "The best action is 0"
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            idx = int(numbers[0])
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx]

        # Fallback: check if the response matches an action directly
        for action in valid_actions:
            if action in text:
                return action

        # Last resort: return first action
        return valid_actions[0]


class OracleAgent(BaseAgent):
    """Always picks the optimal action. Serves as the upper bound.

    Requires access to the environment to query the oracle action.
    """

    def __init__(self, env: TextEnvironment) -> None:
        self._env = env

    @property
    def name(self) -> str:
        return "Oracle"

    def select_action(self, state: str, valid_actions: list[str], goal: str) -> str:
        oracle = self._env.get_oracle_action()
        if oracle is None:
            # Fallback: should not happen if env is not done
            return valid_actions[0]
        assert oracle in valid_actions, f"Oracle action {oracle!r} not in valid actions."
        return oracle
