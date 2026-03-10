"""EMA updater — shared logic with XLM-JEPA."""

import torch.nn as nn


class EMAUpdater:
    """Updates target encoder parameters as EMA of source encoder."""

    def __init__(self, momentum: float = 0.996, momentum_end: float = 1.0, warmup_steps: int = 500) -> None:
        self.base = momentum
        self.end = momentum_end
        self.warmup_steps = warmup_steps

    def momentum_at(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.end
        ratio = min(step / self.warmup_steps, 1.0)
        return self.base + (self.end - self.base) * ratio

    @staticmethod
    def update(source: nn.Module, target: nn.Module, momentum: float) -> None:
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.mul_(momentum).add_(src_p.data, alpha=1.0 - momentum)
