"""Shared execution for training, evaluation, and readable trajectory export."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from environment import Action, Observation, Task, ToolRoutingEnv
from policy import OraclePolicy, TabularPolicy


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: Action
    log_prob: float
    reward: float
    next_observation: Observation


@dataclass(frozen=True)
class Episode:
    task_id: str
    transitions: tuple[Transition, ...]
    success: bool
    terminated: bool
    truncated: bool

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.transitions)

    def to_dict(self) -> dict:
        """Keep readable tool traces, without exposing reward labels to the policy."""
        return {
            "task_id": self.task_id,
            "prompt": self.transitions[0].observation.prompt,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "total_reward": self.total_reward,
            "steps": [
                {
                    "state_id": step.observation.state_id,
                    "action": step.action.name.lower(),
                    "log_prob": step.log_prob,
                    "reward": step.reward,
                    "evidence": step.next_observation.evidence,
                    "tool_message": step.next_observation.tool_message,
                }
                for step in self.transitions
            ],
        }


def collect_episode(
    env: ToolRoutingEnv,
    policy: TabularPolicy | OraclePolicy,
    task: Task,
    rng: np.random.Generator,
    *,
    greedy: bool = False,
) -> Episode:
    observation = env.reset(task)
    transitions = []
    while True:
        choice = policy.select(observation, rng, greedy=greedy)
        result = env.step(choice.action)
        transitions.append(
            Transition(
                observation, choice.action, choice.log_prob, result.reward, result.observation
            )
        )
        observation = result.observation
        if result.terminated or result.truncated:
            return Episode(
                task.task_id,
                tuple(transitions),
                result.success,
                result.terminated,
                result.truncated,
            )


def episode_metrics(episodes: Sequence[Episode]) -> dict[str, float | int]:
    if not episodes:
        raise ValueError("evaluation needs at least one episode")
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([episode.success for episode in episodes])),
        "mean_return": float(np.mean([episode.total_reward for episode in episodes])),
        "mean_steps": float(np.mean([len(episode.transitions) for episode in episodes])),
        "mean_tool_calls": float(
            np.mean(
                [
                    sum(step.action != Action.ANSWER for step in episode.transitions)
                    for episode in episodes
                ]
            )
        ),
        "truncation_rate": float(np.mean([episode.truncated for episode in episodes])),
    }


def evaluate(
    env: ToolRoutingEnv,
    policy: TabularPolicy | OraclePolicy,
    tasks: Sequence[Task],
    *,
    seed: int,
    greedy: bool = False,
) -> tuple[dict[str, float | int], list[Episode]]:
    # Evaluation owns its RNG: changing eval_interval cannot change training samples.
    rng = np.random.default_rng(seed)
    episodes = [collect_episode(env, policy, task, rng, greedy=greedy) for task in tasks]
    return episode_metrics(episodes), episodes
