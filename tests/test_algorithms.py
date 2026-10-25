import numpy as np
import pytest

from algorithms import (
    discounted_returns,
    group_relative_advantages,
    objective_and_gradient,
    update_policy,
)
from config import TrainConfig
from environment import Action, Observation, TaskKind
from policy import TabularPolicy, log_softmax
from rollout import Episode, Transition


def test_discounted_returns_include_future_rewards_and_tool_costs():
    np.testing.assert_allclose(discounted_returns([-0.05, -0.05, 1.0], 0.9), [0.715, 0.85, 1.0])
    np.testing.assert_array_equal(discounted_returns([-0.05, 1.0], 0), [-0.05, 1.0])


@pytest.mark.parametrize("gamma", [-1, 1.01, float("nan")])
def test_invalid_discount(gamma):
    with pytest.raises(ValueError):
        discounted_returns([1.0], gamma)


def test_advantages_are_normalized_within_each_task_not_across_tasks():
    rewards = np.array([[1.0, 3.0, 2.0], [9.0, 9.0, 9.0], [-5.0, -3.0, -4.0]])
    advantages = group_relative_advantages(rewards)
    np.testing.assert_allclose(advantages.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(advantages[0], advantages[2])
    np.testing.assert_array_equal(advantages[1], 0)
    np.testing.assert_allclose(advantages[[0, 2]].std(axis=1), 1, atol=1e-7)


@pytest.mark.parametrize("rewards", [np.ones(3), np.ones((2, 1)), np.array([[0, np.nan]])])
def test_invalid_groups(rewards):
    with pytest.raises(ValueError):
        group_relative_advantages(rewards)


@pytest.mark.parametrize("clipped", [False, True])
def test_analytic_gradient_matches_finite_differences_including_entropy_and_repeated_states(
    clipped,
):
    logits = np.array([[0.3, -0.2, 0.1], [-0.5, 0.2, 0.7]], dtype=np.float64)
    states = np.array([0, 0, 1, 1], dtype=np.int64)
    actions = np.array([0, 2, 1, 2], dtype=np.int64)
    advantages = np.array([1.5, -0.8, 0.6, -1.1])
    weights = np.array([0.25, 0.25, 0.5, 0.5])
    old = log_softmax(logits[states])[np.arange(4), actions] if clipped else None
    if old is not None:
        # Exercise both flat clipped regions and non-clipped regions, away from boundaries.
        old = old - np.log(np.array([1.5, 0.5, 1.1, 1.1]))

    def compute(values):
        return objective_and_gradient(
            values,
            states,
            actions,
            advantages,
            weights,
            entropy_coef=0.07,
            old_log_probs=old,
            clip_ratio=0.2,
        )

    _, gradient, _ = compute(logits)
    numerical = np.zeros_like(logits)
    epsilon = 1e-6
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (compute(plus)[0] - compute(minus)[0]) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("advantage,ratio", [(1.0, 1.5), (-1.0, 0.5)])
def test_clipping_stops_gradient_only_in_improving_direction(advantage, ratio):
    logits = np.zeros((1, 3))
    old = np.array([-np.log(3) - np.log(ratio)])
    _, gradient, _ = objective_and_gradient(
        logits,
        np.array([0]),
        np.array([0]),
        np.array([advantage]),
        np.array([1.0]),
        old_log_probs=old,
    )
    np.testing.assert_array_equal(gradient, 0)
    _, gradient, _ = objective_and_gradient(
        logits,
        np.array([0]),
        np.array([0]),
        np.array([-advantage]),
        np.array([1.0]),
        old_log_probs=old,
    )
    assert np.linalg.norm(gradient) > 0


def test_softmax_remains_finite_for_large_logits():
    log_probs = log_softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isfinite(log_probs).all()
    np.testing.assert_allclose(np.exp(log_probs).sum(), 1.0)


def test_discounted_reinforce_update_matches_exact_expected_return_gradient():
    # Enumerate all equiprobable trajectories of a two-step process. Compare
    # the actual update with the gradient of its independently computed return.
    policy = TabularPolicy(2, 3)
    gamma = 0.5
    observations = [
        Observation(index, TaskKind.ARITHMETIC, "synthetic", None, None, 2 - index)
        for index in range(2)
    ]
    episodes = []
    for first in range(3):
        for second in range(3):
            transitions = (
                Transition(
                    observations[0], Action(first), -np.log(3), float(first == 0), observations[1]
                ),
                Transition(
                    observations[1],
                    Action(second),
                    -np.log(3),
                    2.0 * (second == 1),
                    observations[1],
                ),
            )
            episodes.append(Episode("enumerated", transitions, False, True, False))

    def exact_return(logits):
        probabilities = np.exp(log_softmax(logits))
        return probabilities[0, 0] + gamma * 2 * probabilities[1, 1]

    numerical = np.zeros_like(policy.logits)
    for index in np.ndindex(policy.logits.shape):
        plus, minus = policy.logits.copy(), policy.logits.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (exact_return(plus) - exact_return(minus)) / 2e-6
    config = TrainConfig(
        algorithm="reinforce",
        gamma=gamma,
        group_size=1,
        update_epochs=1,
        entropy_coef=0.0,
        learning_rate=0.1,
        max_grad_norm=100,
    )
    update_policy(policy, episodes, config, np.zeros(2))
    np.testing.assert_allclose(policy.logits / config.learning_rate, numerical, atol=1e-8)
