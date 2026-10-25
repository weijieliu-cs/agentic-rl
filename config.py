"""Strict TOML configuration with all effective values saved in every run."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "grpo"
    seed: int = 0
    iterations: int = 200
    batch_size: int = 32
    group_size: int = 4
    learning_rate: float = 1.0
    update_epochs: int = 4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 1.0
    baseline_decay: float = 0.9
    max_grad_norm: float = 1.0
    train_tasks: int = 256
    eval_tasks: int = 256
    eval_interval: int = 20
    max_steps: int = 4
    tool_cost: float = 0.05

    def __post_init__(self) -> None:
        if self.algorithm not in ("reinforce", "grpo"):
            raise ValueError("algorithm must be 'reinforce' or 'grpo'")
        integer_fields = (
            "seed",
            "iterations",
            "batch_size",
            "group_size",
            "update_epochs",
            "train_tasks",
            "eval_tasks",
            "eval_interval",
            "max_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "learning_rate",
            "clip_ratio",
            "entropy_coef",
            "gamma",
            "baseline_decay",
            "max_grad_norm",
            "tool_cost",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1)")
        if self.entropy_coef < 0 or not 0 <= self.gamma <= 1:
            raise ValueError("entropy_coef must be nonnegative; gamma must be in [0, 1]")
        if not 0 <= self.baseline_decay < 1 or not 0 <= self.tool_cost < 1:
            raise ValueError("baseline_decay and tool_cost must be in [0, 1)")
        if self.max_steps < 2:
            raise ValueError("max_steps must be >= 2 to allow a tool call and an answer")
        if self.algorithm == "grpo" and (self.group_size < 2 or self.gamma != 1):
            raise ValueError("grpo requires group_size >= 2 and gamma = 1 (episode returns)")
        if self.algorithm == "reinforce" and self.update_epochs != 1:
            raise ValueError("reinforce requires update_epochs = 1 for on-policy updates")


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    unknown = values.keys() - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    return TrainConfig(**values)
