from dataclasses import replace

import numpy as np
import pytest

from environment import Action, TaskKind, ToolRoutingEnv, make_tasks


@pytest.mark.parametrize("kind", [TaskKind.ARITHMETIC, TaskKind.LOOKUP])
def test_tool_then_answer_is_successful(kind):
    tasks = make_tasks(2, np.random.default_rng(0), split="train")
    task = next(task for task in tasks if task.kind == kind)
    env = ToolRoutingEnv()
    first = env.reset(task)
    assert first.evidence is None
    tool = Action.CALCULATOR if kind == TaskKind.ARITHMETIC else Action.LOOKUP
    result = env.step(tool)
    assert result.reward == -0.05
    assert result.observation.evidence == task.expected_answer
    assert result.observation.state_id != first.state_id
    assert not result.terminated and not result.truncated
    result = env.step(Action.ANSWER)
    assert result.success and result.terminated and not result.truncated
    assert result.reward == 1.0


def test_premature_answer_and_wrong_tool_do_not_succeed():
    task = make_tasks(1, np.random.default_rng(0), split="train")[0]
    env = ToolRoutingEnv()
    env.reset(task)
    result = env.step(Action.LOOKUP)
    assert result.observation.evidence is None
    assert not result.success
    result = env.step(Action.ANSWER)
    assert result.terminated and not result.success and result.reward == 0


def test_verifier_checks_answer_not_just_tool_use():
    task = make_tasks(1, np.random.default_rng(0), split="train")[0]
    env = ToolRoutingEnv()
    env.reset(replace(task, expected_answer="incorrect-label"))
    env.step(Action.CALCULATOR)
    assert not env.step(Action.ANSWER).success


def test_truncation_and_last_step_answer_are_distinct():
    task = make_tasks(1, np.random.default_rng(0), split="train")[0]
    env = ToolRoutingEnv(max_steps=2)
    env.reset(task)
    env.step(Action.CALCULATOR)
    result = env.step(Action.CALCULATOR)
    assert result.truncated and not result.terminated and not result.success
    assert result.observation.steps_remaining == 0
    assert 0 <= result.observation.state_id < env.n_states
    env.reset(task)
    env.step(Action.CALCULATOR)
    result = env.step(Action.ANSWER)
    assert result.terminated and result.success and not result.truncated


def test_lifecycle_and_invalid_action():
    env = ToolRoutingEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.ANSWER)
    env.reset(make_tasks(1, np.random.default_rng(0), split="train")[0])
    with pytest.raises(ValueError, match="invalid action"):
        env.step(999)
    env.step(Action.ANSWER)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.ANSWER)


def test_task_generation_is_reproducible_and_split_contents_are_disjoint():
    train = make_tasks(100, np.random.default_rng(11), split="train")
    assert train == make_tasks(100, np.random.default_rng(11), split="train")
    held_out = make_tasks(100, np.random.default_rng(11), split="eval")
    assert {task.key for task in train if task.key}.isdisjoint(
        {task.key for task in held_out if task.key}
    )
    assert all(1 <= value <= 20 for task in train for value in task.operands or ())
    assert all(21 <= value <= 40 for task in held_out for value in task.operands or ())


@pytest.mark.parametrize(
    "kwargs", [{"max_steps": 1}, {"tool_cost": -1}, {"tool_cost": float("nan")}]
)
def test_environment_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ToolRoutingEnv(**kwargs)


@pytest.mark.parametrize("count,split", [(0, "train"), (True, "train"), (2, "unknown")])
def test_task_generation_rejects_invalid_input(count, split):
    with pytest.raises(ValueError):
        make_tasks(count, np.random.default_rng(0), split=split)


def test_reset_rejects_malformed_tasks():
    arithmetic, lookup = make_tasks(2, np.random.default_rng(0), split="train")
    env = ToolRoutingEnv()
    for task in (
        replace(arithmetic, operands=None),
        replace(lookup, key="missing"),
        replace(arithmetic, kind="invalid"),
    ):
        with pytest.raises(ValueError):
            env.reset(task)
