"""A deliberately small tool-selection task with a bounded interaction budget."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np


class TaskKind(IntEnum):
    ARITHMETIC = 0
    LOOKUP = 1


class Action(IntEnum):
    CALCULATOR = 0
    LOOKUP = 1
    ANSWER = 2


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: TaskKind
    prompt: str
    expected_answer: str
    operands: tuple[int, int] | None = None
    key: str | None = None


@dataclass(frozen=True)
class Observation:
    state_id: int
    kind: TaskKind
    prompt: str
    evidence: str | None
    tool_message: str | None
    steps_remaining: int


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    success: bool


# Synthetic data authored for this project. The tool knows both partitions;
# training prompts use only train keys, and evaluation prompts use only eval keys.
KNOWLEDGE_BASE = {
    "amber": "17",
    "birch": "23",
    "cedar": "31",
    "dahlia": "43",
    "elm": "59",
    "fern": "61",
    "garnet": "71",
    "hazel": "83",
    "iris": "97",
    "jade": "101",
    "kelp": "109",
    "larch": "127",
}


def make_tasks(
    count: int, rng: np.random.Generator, *, split: Literal["train", "eval"]
) -> list[Task]:
    """Generate balanced task kinds with disjoint operands and lookup keys by split."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    low, high = (1, 21) if split == "train" else (21, 41)
    keys = list(KNOWLEDGE_BASE)[:6] if split == "train" else list(KNOWLEDGE_BASE)[6:]
    tasks = []
    for index in range(count):
        task_id = f"{split}-{index:05d}"
        if index % 2 == 0:
            left, right = (int(value) for value in rng.integers(low, high, size=2))
            tasks.append(
                Task(
                    task_id=task_id,
                    kind=TaskKind.ARITHMETIC,
                    prompt=f"Use the calculator to find {left} + {right}, then submit the result.",
                    expected_answer=str(left + right),
                    operands=(left, right),
                )
            )
        else:
            key = str(rng.choice(keys))
            tasks.append(
                Task(
                    task_id=task_id,
                    kind=TaskKind.LOOKUP,
                    prompt=f"Look up the code for '{key}', then submit the result.",
                    expected_answer=KNOWLEDGE_BASE[key],
                    key=key,
                )
            )
    return tasks


class ToolRoutingEnv:
    """Learn which tool to call and when to answer, rather than learning tool contents.

    CALCULATOR and LOOKUP use arguments already supplied by the task. ANSWER
    submits the current evidence verbatim. All three actions are always legal.
    The observation includes a task-kind feature, but never the expected answer.
    """

    n_actions = len(Action)

    def __init__(self, max_steps: int = 4, tool_cost: float = 0.05):
        if type(max_steps) is not int or max_steps < 2:
            raise ValueError("max_steps must be an integer >= 2")
        if not np.isfinite(tool_cost) or not 0 <= tool_cost < 1:
            raise ValueError("tool_cost must be finite and in [0, 1)")
        self.max_steps = max_steps
        self.tool_cost = tool_cost
        self.n_states = len(TaskKind) * 2 * (max_steps + 1)
        self._task: Task | None = None
        self._done = True

    def reset(self, task: Task) -> Observation:
        if not isinstance(task.kind, TaskKind):
            raise ValueError("unknown task kind")
        if task.kind == TaskKind.ARITHMETIC and task.operands is None:
            raise ValueError("arithmetic tasks must provide operands")
        if task.kind == TaskKind.LOOKUP and task.key not in KNOWLEDGE_BASE:
            raise ValueError("lookup tasks must provide a known key")
        self._task = task
        self._evidence: str | None = None
        self._tool_message: str | None = None
        self._remaining = self.max_steps
        self._done = False
        return self._observe()

    def _observe(self) -> Observation:
        assert self._task is not None
        state_id = (int(self._task.kind) * 2 + int(self._evidence is not None)) * (
            self.max_steps + 1
        ) + self._remaining
        return Observation(
            state_id=state_id,
            kind=self._task.kind,
            prompt=self._task.prompt,
            evidence=self._evidence,
            tool_message=self._tool_message,
            steps_remaining=self._remaining,
        )

    def step(self, action: Action) -> StepResult:
        if self._done or self._task is None:
            raise RuntimeError(
                "reset an environment before stepping; completed episodes need reset"
            )
        try:
            action = Action(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid action: {action!r}") from exc
        self._remaining -= 1
        terminated = action == Action.ANSWER
        success = terminated and self._evidence == self._task.expected_answer
        reward = float(success) if terminated else -self.tool_cost
        if action == Action.CALCULATOR:
            if self._task.operands is not None:
                left, right = self._task.operands
                self._evidence = str(left + right)
                self._tool_message = f"calculator: {self._evidence}"
            else:
                self._tool_message = "calculator: this task has no arithmetic operands"
        elif action == Action.LOOKUP:
            if self._task.key is not None:
                self._evidence = KNOWLEDGE_BASE[self._task.key]
                self._tool_message = f"lookup: {self._evidence}"
            else:
                self._tool_message = "lookup: this task has no lookup key"
        else:
            self._tool_message = f"submitted: {self._evidence}"
        truncated = not terminated and self._remaining == 0
        self._done = terminated or truncated
        return StepResult(self._observe(), reward, terminated, truncated, success)
