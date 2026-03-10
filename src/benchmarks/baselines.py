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
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        # Match training padding side (left-padding for decoder-only models)
        self._tokenizer.padding_side = "left"
        self._max_length = max_length
        self._device = device or next(model.parameters()).device
        self._model.eval()

    @property
    def name(self) -> str:
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
