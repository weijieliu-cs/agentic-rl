"""Evaluate a saved policy on held-out tool-routing tasks."""

import argparse
import json
from pathlib import Path

import numpy as np

from environment import ToolRoutingEnv, make_tasks
from rollout import evaluate
from utils import load_checkpoint, write_trajectories


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--greedy", action="store_true", help="choose the most likely action")
    parser.add_argument("--trajectories", type=Path, help="export trajectories to a new file")
    args = parser.parse_args(argv)
    try:
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        if args.trajectories is not None and args.trajectories.exists():
            raise ValueError(f"trajectory output already exists: {args.trajectories}")
        policy, config = load_checkpoint(args.checkpoint)
        tasks = make_tasks(args.episodes, np.random.default_rng(args.seed), split="eval")
        metrics, episodes = evaluate(
            ToolRoutingEnv(config.max_steps, config.tool_cost),
            policy,
            tasks,
            seed=args.seed,
            greedy=args.greedy,
        )
        if args.trajectories is not None:
            args.trajectories.parent.mkdir(parents=True, exist_ok=True)
            write_trajectories(args.trajectories, episodes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"seed": args.seed, "greedy": args.greedy, **metrics}, indent=2))


if __name__ == "__main__":
    main()
