import json
from dataclasses import replace

import numpy as np
import pytest

from config import TrainConfig, load_config
from environment import ToolRoutingEnv, make_tasks
from evaluate import main as evaluate_main
from policy import OraclePolicy, TabularPolicy
from rollout import episode_metrics, evaluate
from train import main as train_main
from train import run_experiment
from utils import load_checkpoint, save_checkpoint


@pytest.mark.parametrize(
    "kwargs",
    [
        {"algorithm": "unknown"},
        {"seed": -1},
        {"batch_size": True},
        {"iterations": 0},
        {"learning_rate": float("inf")},
        {"learning_rate": -0.1},
        {"gamma": 2},
        {"entropy_coef": -0.1},
        {"baseline_decay": 1},
        {"max_steps": 1},
        {"clip_ratio": 1},
        {"tool_cost": 1},
        {"group_size": 1},
        {"gamma": 0.9},
        {"algorithm": "reinforce", "update_epochs": 2},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("lerning_rate = 0.1\n")
    with pytest.raises(ValueError, match="unknown configuration fields: lerning_rate"):
        load_config(path)


@pytest.mark.parametrize("algorithm", ["grpo", "reinforce"])
def test_committed_recipes_are_valid(algorithm):
    assert load_config(f"configs/{algorithm}.toml").algorithm == algorithm


def test_oracle_is_perfect_and_metrics_reject_empty_batches():
    env = ToolRoutingEnv()
    tasks = make_tasks(10, np.random.default_rng(0), split="eval")
    metrics, _ = evaluate(env, OraclePolicy(), tasks, seed=0)
    assert metrics["success_rate"] == 1
    assert metrics["mean_steps"] == 2
    assert metrics["mean_tool_calls"] == 1
    assert metrics["mean_return"] == pytest.approx(0.95)
    with pytest.raises(ValueError):
        episode_metrics([])


def test_checkpoint_roundtrip_and_validation(tmp_path):
    config = TrainConfig()
    env = ToolRoutingEnv(config.max_steps, config.tool_cost)
    policy = TabularPolicy(env.n_states, env.n_actions)
    policy.logits[:] = np.random.default_rng(3).normal(size=policy.logits.shape)
    path = tmp_path / "policy.npz"
    save_checkpoint(path, policy, config)
    loaded, loaded_config = load_checkpoint(path)
    assert config == loaded_config
    np.testing.assert_array_equal(policy.logits, loaded.logits)
    tasks = make_tasks(10, np.random.default_rng(11), split="eval")
    assert evaluate(env, policy, tasks, seed=5)[0] == evaluate(env, loaded, tasks, seed=5)[0]
    with np.load(path, allow_pickle=False) as saved:
        fields = {name: saved[name] for name in saved.files}
    np.savez(path, **{**fields, "logits": np.zeros((1, 1))})
    with pytest.raises(ValueError, match="invalid shape"):
        load_checkpoint(path)
    np.savez(path, **{**fields, "format_version": 999})
    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(path)
    np.savez(path, logits=policy.logits)
    with pytest.raises(ValueError, match="invalid policy checkpoint"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "algorithm,iterations,epochs,group_size", [("grpo", 90, 4, 4), ("reinforce", 150, 1, 1)]
)
def test_training_improves_held_out_tool_routing(
    tmp_path, algorithm, iterations, epochs, group_size
):
    config = TrainConfig(
        algorithm=algorithm,
        iterations=iterations,
        update_epochs=epochs,
        group_size=group_size,
        batch_size=32,
        eval_tasks=128,
        eval_interval=iterations,
    )
    output = tmp_path / algorithm
    summary = run_experiment(config, output)
    assert summary["final_eval"]["success_rate"] > 0.75
    assert summary["final_eval"]["success_rate"] > summary["initial_eval"]["success_rate"] + 0.4
    assert summary["oracle_eval"]["success_rate"] == 1.0
    assert summary["greedy_eval"]["success_rate"] == 1.0
    assert json.loads((output / "summary.json").read_text()) == summary
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == iterations + 1
    traces = [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]
    assert len(traces) == config.eval_tasks
    assert np.mean([trace["success"] for trace in traces]) == summary["final_eval"]["success_rate"]
    with pytest.raises(FileExistsError):
        run_experiment(config, output)


def test_seed_reproducibility_and_evaluation_cadence_do_not_change_training(tmp_path):
    config = TrainConfig(iterations=7, batch_size=4, eval_tasks=10, eval_interval=7)
    first = run_experiment(config, tmp_path / "first")
    repeated = run_experiment(config, tmp_path / "repeated")
    frequent = run_experiment(replace(config, eval_interval=1), tmp_path / "frequent")
    assert first == repeated == frequent
    expected, _ = load_checkpoint(tmp_path / "first" / "policy.npz")
    actual, _ = load_checkpoint(tmp_path / "frequent" / "policy.npz")
    np.testing.assert_array_equal(expected.logits, actual.logits)


def test_train_and_evaluate_scripts_end_to_end(tmp_path, capsys):
    output = tmp_path / "scripts"
    train_main(
        [
            "--config",
            "configs/grpo.toml",
            "--output",
            str(output),
            "--iterations",
            "2",
            "--seed",
            "3",
        ]
    )
    assert json.loads((output / "config.json").read_text())["seed"] == 3
    assert "eval_success=" in capsys.readouterr().out
    trajectory_path = tmp_path / "eval" / "traces.jsonl"
    evaluate_main(
        [
            "--checkpoint",
            str(output / "policy.npz"),
            "--episodes",
            "6",
            "--seed",
            "123",
            "--greedy",
            "--trajectories",
            str(trajectory_path),
        ]
    )
    assert json.loads(capsys.readouterr().out)["episodes"] == 6
    assert len(trajectory_path.read_text().splitlines()) == 6
    with pytest.raises(SystemExit) as exc:
        evaluate_main(["--checkpoint", str(output / "missing.npz")])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_policy_rejects_incompatible_action_space():
    with pytest.raises(ValueError):
        TabularPolicy(10, 2)
