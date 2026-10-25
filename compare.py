"""Run the committed recipes across multiple seeds and export aggregate metrics.

Usage: python compare.py --seeds 0 1 2 --output runs/comparison
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import load_config
from train import run_experiment
from utils import write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.output.exists():
        parser.error("output directory already exists; choose a new one")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parent
    results = {}
    for algorithm in ("reinforce", "grpo"):
        config = load_config(root / "configs" / f"{algorithm}.toml")
        summaries = []
        for seed in args.seeds:
            summary = run_experiment(
                replace(config, seed=seed), args.output / algorithm / f"seed-{seed}"
            )
            summaries.append(summary)
            print(
                f"{algorithm} seed={seed}: {summary['final_eval']['success_rate']:.4f}", flush=True
            )
        results[algorithm] = {"seeds": args.seeds, "runs": summaries}
        for metric in ("success_rate", "mean_return", "mean_steps", "mean_tool_calls"):
            values = [summary["final_eval"][metric] for summary in summaries]
            results[algorithm][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    write_json(args.output / "comparison.json", results)
    print(json.dumps({name: data["success_rate"] for name, data in results.items()}, indent=2))


if __name__ == "__main__":
    main()
