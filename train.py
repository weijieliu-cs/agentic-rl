"""Train a tabular tool-routing policy from an experiment configuration."""

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from algorithms import update_policy
from config import TrainConfig, load_config
from environment import ToolRoutingEnv, make_tasks
from policy import OraclePolicy, TabularPolicy
from rollout import collect_episode, episode_metrics, evaluate
from utils import git_revision, save_checkpoint, source_fingerprint, write_json, write_trajectories


def run_experiment(
    config: TrainConfig,
    output: str | Path,
    *,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Train from scratch. The output directory must not exist, preventing silent overwrites."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(
        output / "metadata.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_revision": git_revision(),
            "source_sha256": source_fingerprint(),
        },
    )
    data_seed, rollout_seed = np.random.SeedSequence(config.seed).spawn(2)
    train_rng = np.random.default_rng(rollout_seed)
    train_tasks = make_tasks(config.train_tasks, np.random.default_rng(data_seed), split="train")
    evaluation_seed = config.seed + 100_000
    eval_tasks = make_tasks(config.eval_tasks, np.random.default_rng(evaluation_seed), split="eval")
    env = ToolRoutingEnv(config.max_steps, config.tool_cost)
    eval_env = ToolRoutingEnv(config.max_steps, config.tool_cost)
    policy = TabularPolicy(env.n_states, env.n_actions)
    baseline = np.zeros(env.n_states)
    initial_eval, _ = evaluate(eval_env, policy, eval_tasks, seed=evaluation_seed)
    oracle_eval, _ = evaluate(eval_env, OraclePolicy(), eval_tasks, seed=evaluation_seed)
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps({"iteration": 0, "eval": initial_eval}, allow_nan=False) + "\n")
        for iteration in range(1, config.iterations + 1):
            episodes = []
            for index in train_rng.integers(0, len(train_tasks), size=config.batch_size):
                # Freeze the behavior policy until every group has been collected.
                for _ in range(config.group_size):
                    episodes.append(collect_episode(env, policy, train_tasks[index], train_rng))
            update_stats = update_policy(policy, episodes, config, baseline)
            record = {
                "iteration": iteration,
                "train": episode_metrics(episodes),
                "update": update_stats,
            }
            if iteration % config.eval_interval == 0 or iteration == config.iterations:
                final_eval, final_episodes = evaluate(
                    eval_env, policy, eval_tasks, seed=evaluation_seed
                )
                record["eval"] = final_eval
                if on_progress is not None:
                    on_progress(record)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
    greedy_eval, _ = evaluate(eval_env, policy, eval_tasks, seed=evaluation_seed, greedy=True)
    summary = {
        "algorithm": config.algorithm,
        "seed": config.seed,
        "iterations": config.iterations,
        "training_episodes": config.iterations * config.batch_size * config.group_size,
        "evaluation_seed": evaluation_seed,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "greedy_eval": greedy_eval,
        "oracle_eval": oracle_eval,
    }
    save_checkpoint(output / "policy.npz", policy, config)
    write_trajectories(output / "trajectories.jsonl", final_episodes)
    write_json(output / "summary.json", summary)
    return summary


def _progress(record: dict) -> None:
    print(
        f"iteration={record['iteration']:4d} "
        f"train_success={record['train']['success_rate']:.3f} "
        f"eval_success={record['eval']['success_rate']:.3f} "
        f"eval_return={record['eval']['mean_return']:.3f}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TOML experiment settings")
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--seed", type=int, help="override the configuration seed")
    parser.add_argument("--iterations", type=int, help="override the training iteration count")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        overrides = {
            name: getattr(args, name)
            for name in ("seed", "iterations")
            if getattr(args, name) is not None
        }
        summary = run_experiment(replace(config, **overrides), args.output, on_progress=_progress)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
