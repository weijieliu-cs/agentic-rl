"""Policy-gradient building blocks with explicit NumPy gradients.

The objective is maximized. GRPO uses per-action old-policy ratios, clipping,
and group-normalized episode returns. No critic or reference-model KL is used.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from config import TrainConfig
from policy import TabularPolicy, log_softmax
from rollout import Episode


def discounted_returns(rewards: Sequence[float], gamma: float) -> NDArray[np.float64]:
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    result = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def group_relative_advantages(rewards: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows are tasks, columns are independent trajectories for the same task."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not np.isfinite(rewards).all():
        raise ValueError("group rewards must be finite with shape (tasks, group_size >= 2)")
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    # Population std (ddof=0). An all-equal group has exactly zero advantages.
    return centered / (rewards.std(axis=1, keepdims=True) + 1e-8)


def objective_and_gradient(
    logits: NDArray[np.float64],
    states: NDArray[np.int64],
    actions: NDArray[np.int64],
    advantages: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    entropy_coef: float = 0.0,
    old_log_probs: NDArray[np.float64] | None = None,
    clip_ratio: float = 0.2,
) -> tuple[float, NDArray[np.float64], dict[str, float]]:
    """Return a scalar objective and its exact gradient with respect to the logits.

    REINFORCE weights each step by gamma**t / batch_episodes. GRPO weights each step
    by 1 / (batch_episodes * episode_length), averaging episodes equally.
    Old log probabilities and advantages stay fixed over GRPO update epochs.
    """
    log_probs = log_softmax(logits[states])
    probabilities = np.exp(log_probs)
    selected = log_probs[np.arange(len(states)), actions]
    entropy = -(probabilities * log_probs).sum(axis=1)
    clip_fraction = 0.0
    approx_kl = 0.0
    if old_log_probs is None:
        surrogate = selected * advantages
        coefficients = advantages
    else:
        log_ratio = selected - old_log_probs
        ratio = np.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        surrogate = np.minimum(unclipped, clipped)
        coefficients = np.where(unclipped <= clipped, ratio * advantages, 0.0)
        clip_fraction = float(np.average(np.abs(ratio - 1) > clip_ratio, weights=weights))
        approx_kl = float(np.average((ratio - 1) - log_ratio, weights=weights))
    score_gradient = -probabilities.copy()
    score_gradient[np.arange(len(states)), actions] += 1
    entropy_gradient = -probabilities * (log_probs + entropy[:, None])
    step_gradient = weights[:, None] * (
        coefficients[:, None] * score_gradient + entropy_coef * entropy_gradient
    )
    gradient = np.zeros_like(logits)
    # Repeated states must accumulate, rather than overwrite, their contributions.
    np.add.at(gradient, states, step_gradient)
    objective = float(np.sum(weights * (surrogate + entropy_coef * entropy)))
    return (
        objective,
        gradient,
        {
            "objective": objective,
            "entropy": float(np.average(entropy, weights=weights)),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
        },
    )


def update_policy(
    policy: TabularPolicy,
    episodes: Sequence[Episode],
    config: TrainConfig,
    baseline: np.ndarray,
) -> dict[str, float]:
    states = np.array(
        [step.observation.state_id for episode in episodes for step in episode.transitions],
        dtype=np.int64,
    )
    actions = np.array(
        [step.action for episode in episodes for step in episode.transitions], dtype=np.int64
    )
    old_log_probs = np.array(
        [step.log_prob for episode in episodes for step in episode.transitions], dtype=np.float64
    )
    lengths = np.array([len(episode.transitions) for episode in episodes])
    if config.algorithm == "grpo":
        rewards = np.array([episode.total_reward for episode in episodes]).reshape(
            config.batch_size, config.group_size
        )
        episode_advantages = group_relative_advantages(rewards).ravel()
        advantages = np.repeat(episode_advantages, lengths)
        weights = np.repeat(1.0 / (len(episodes) * lengths), lengths)
    else:
        returns = np.concatenate(
            [
                discounted_returns([step.reward for step in episode.transitions], config.gamma)
                for episode in episodes
            ]
        )
        # Use only the preceding batches' baseline when computing this batch's advantages.
        advantages = returns - baseline[states]
        # The outer gamma**t makes this the gradient of the discounted
        # episode objective, not just a discounted reward-to-go heuristic.
        weights = np.concatenate([config.gamma ** np.arange(length) for length in lengths])
        weights /= len(episodes)
        counts = np.bincount(states, minlength=len(baseline))
        totals = np.bincount(states, weights=returns, minlength=len(baseline))
        visited = counts > 0
        baseline[visited] = (
            config.baseline_decay * baseline[visited]
            + (1 - config.baseline_decay) * totals[visited] / counts[visited]
        )
    stats = {}
    for _ in range(config.update_epochs):
        _, gradient, stats = objective_and_gradient(
            policy.logits,
            states,
            actions,
            advantages,
            weights,
            entropy_coef=config.entropy_coef,
            old_log_probs=old_log_probs if config.algorithm == "grpo" else None,
            clip_ratio=config.clip_ratio,
        )
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm):
            raise ValueError(
                "non-finite policy gradient; check the learning rate and configuration"
            )
        gradient *= min(1.0, config.max_grad_norm / max(norm, 1e-12))
        policy.logits += config.learning_rate * gradient
        stats["grad_norm"] = norm
    return stats
