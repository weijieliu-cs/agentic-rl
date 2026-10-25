"""A tabular softmax policy and an explicit, non-learning upper bound."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from environment import Action, Observation, TaskKind


@dataclass(frozen=True)
class ActionChoice:
    action: Action
    log_prob: float


def log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class TabularPolicy:
    """One categorical distribution per observable state; starts uniformly random."""

    def __init__(self, n_states: int, n_actions: int):
        if n_states < 1 or n_actions != len(Action):
            raise ValueError("require positive n_states and the tool-routing action space")
        self.logits = np.zeros((n_states, n_actions), dtype=np.float64)

    def select(
        self, observation: Observation, rng: np.random.Generator, *, greedy: bool = False
    ) -> ActionChoice:
        log_probs = log_softmax(self.logits[observation.state_id])
        action = int(
            np.argmax(log_probs) if greedy else rng.choice(len(log_probs), p=np.exp(log_probs))
        )
        return ActionChoice(Action(action), float(log_probs[action]))


class OraclePolicy:
    """Hand-written tool routing. This is a sanity check, not a learned baseline."""

    def select(
        self, observation: Observation, rng: np.random.Generator, *, greedy: bool = False
    ) -> ActionChoice:
        if observation.evidence is not None:
            action = Action.ANSWER
        elif observation.kind == TaskKind.ARITHMETIC:
            action = Action.CALCULATOR
        else:
            action = Action.LOOKUP
        return ActionChoice(action, 0.0)
