"""Validation-only plateau policy for rhythm and source-attribute training."""
from dataclasses import dataclass
import math


@dataclass
class Convergence:
    lr: float = 5e-5
    min_lr: float = 3.125e-6
    factor: float = .5
    reduce_patience: int = 3
    stop_patience: int = 8
    min_steps: int = 10000
    f1_delta: float = .002
    attribute_relative_delta: float = .005
    best_f1: float = -math.inf
    best_attribute_nll: float = math.inf
    stale: int = 0
    floor_stale: int = 0

    def observe(self, f1: float, attribute_nll: float, step: int) -> str:
        if not math.isfinite(f1) or not math.isfinite(attribute_nll):
            raise FloatingPointError("Nonfinite convergence metric")
        improved_f1 = f1 > self.best_f1 + self.f1_delta
        improved_attributes = attribute_nll < self.best_attribute_nll * (1 - self.attribute_relative_delta)
        if improved_f1:
            self.best_f1 = f1
        if improved_attributes:
            self.best_attribute_nll = attribute_nll
        if improved_f1 or improved_attributes:
            self.stale = self.floor_stale = 0
            return "improved"
        self.stale += 1
        if self.lr <= self.min_lr:
            self.floor_stale += 1
            if step >= self.min_steps and self.floor_stale >= self.stop_patience:
                return "converged"
        elif self.stale >= self.reduce_patience:
            self.lr = max(self.min_lr, self.lr * self.factor)
            self.stale = 0
            self.floor_stale = 0
            return "reduce_lr"
        return "continue"


def attribute_nll(metrics):
    heads = metrics["attributes"].values()
    values = [h["reference_nll"] / h["reference_count"] for h in heads if h["reference_count"]]
    if len(values) != len(metrics["attributes"]) or not values:
        raise ValueError("Convergence requires supervised validation for every attribute head")
    return sum(values) / len(values)
