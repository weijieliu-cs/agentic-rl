"""Run files, source provenance, and NumPy policy checkpoints."""

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import TrainConfig
from environment import ToolRoutingEnv
from policy import TabularPolicy
from rollout import Episode


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_trajectories(path: Path, episodes: Sequence[Episode]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), allow_nan=False) + "\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_fingerprint() -> str:
    """Identify the experiment source files, including uncommitted edits."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


FORMAT_VERSION = 1


def save_checkpoint(path: Path, policy: TabularPolicy, config: TrainConfig) -> None:
    np.savez_compressed(
        path,
        format_version=np.array(FORMAT_VERSION),
        logits=policy.logits,
        config_json=np.array(json.dumps(asdict(config), allow_nan=False)),
    )


def load_checkpoint(path: str | Path) -> tuple[TabularPolicy, TrainConfig]:
    try:
        with np.load(path, allow_pickle=False) as saved:
            if saved["format_version"].item() != FORMAT_VERSION:
                raise ValueError("unsupported checkpoint format version")
            config = TrainConfig(**json.loads(str(saved["config_json"].item())))
            env = ToolRoutingEnv(config.max_steps, config.tool_cost)
            logits = np.array(saved["logits"], dtype=np.float64)
            if logits.shape != (env.n_states, env.n_actions) or not np.isfinite(logits).all():
                raise ValueError("checkpoint logits have an invalid shape or non-finite values")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid policy checkpoint: {exc}") from exc
    policy = TabularPolicy(env.n_states, env.n_actions)
    policy.logits[:] = logits
    return policy, config
