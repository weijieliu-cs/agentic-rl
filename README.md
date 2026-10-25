# Agentic Reinforcement Learning

Research code for studying reinforcement learning in tool-using agents, organized
around readable algorithms and reproducible experiments.

[中文](README.zh-CN.md) · [Task](docs/task.md) ·
[Algorithms](docs/algorithms.md) · [Experiments](docs/experiments.md) ·
[References](docs/references.md)

## Research setting

The current experiment asks a small question: **can a policy learn which tool to
call and when to submit an answer, using only interaction rewards?**

A synthetic environment provides arithmetic and lookup tasks. A tabular softmax
policy chooses among calculator, lookup, and answer actions. Each tool call costs
0.05; a correct submission earns 1. The optimal trajectory uses the right tool,
then submits its result, for a return of 0.95.

Two baselines are implemented with explicit NumPy gradients:

- **REINFORCE:** reward-to-go and a state baseline estimated from previous batches.
- **Simplified GRPO:** same-task trajectory groups, normalized returns, and clipped
  action probability ratios.

The task supplies the task kind and tool arguments. This experiment measures tool
routing and stopping behavior; it does not evaluate language understanding or
train a language model. The present code is an initial research baseline, with no
associated paper claimed. See the [task definition](docs/task.md) and
[algorithm assumptions](docs/algorithms.md).

## Setup

Python 3.11+ and a CPU are sufficient. Run these commands from the repository root:

```bash
git clone https://github.com/weijieliu-cs/agentic-rl.git
cd agentic-rl
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py --config configs/grpo.toml --output runs/grpo
python evaluate.py --checkpoint runs/grpo/policy.npz --episodes 256

# Deterministic evaluation and readable interaction traces.
python evaluate.py --checkpoint runs/grpo/policy.npz --greedy \
  --trajectories runs/grpo/greedy.jsonl

# REINFORCE baseline.
python train.py --config configs/reinforce.toml --output runs/reinforce

# Short run with configuration overrides.
python train.py --config configs/grpo.toml --output runs/smoke --seed 7 --iterations 5
```

Use a new output directory for each run. Each directory contains the effective
configuration, runtime/source metadata, per-iteration metrics, final policy,
evaluation summary, and action/tool traces. Checkpoints support offline evaluation;
training resume is not implemented.

## Reproduce the baseline results

```bash
python compare.py --seeds 0 1 2 --output runs/comparison
```

The script runs both recipes and saves each seed's artifacts plus `comparison.json`.
The recorded CPU experiment uses 256 held-out prompts per seed. Mean and sample
standard deviation over three seeds:

| Policy | Sampled success | Mean return |
| --- | --- | --- |
| Uniform initial policy | 33.07% ± 1.63 pp | 0.2540 ± 0.0189 |
| REINFORCE | 99.74% ± 0.45 pp | 0.9424 ± 0.0050 |
| Simplified GRPO | 99.74% ± 0.45 pp | 0.9470 ± 0.0047 |
| Hand-written oracle | 100.00% ± 0.00 pp | 0.9500 ± 0.0000 |

`pp` denotes percentage points. Both trained policies reach 100% greedy success in
these runs. The recipes use different sampling budgets, so this table supports a
learning sanity check, not an efficiency ranking. Full settings, metric definitions,
and limitations are in [experiments](docs/experiments.md); raw summaries and source
fingerprints are in [results/tool-routing-cpu.json](results/tool-routing-cpu.json).

## Code guide

```text
train.py                # Training loop and experiment entry point
evaluate.py             # Evaluate a saved policy and export traces
compare.py              # Run both baselines across seeds
algorithms.py           # REINFORCE / GRPO objectives, gradients, batch updates
environment.py          # Synthetic tasks, tools, observations, rewards
policy.py               # Tabular softmax policy and hand-written oracle
rollout.py              # Episode collection and evaluation metrics
config.py               # Experiment settings and TOML loading
utils.py                # Checkpoints, run files, source fingerprints
configs/                # Baseline hyperparameters
results/                # Recorded experiment summaries
docs/                   # Task, algorithms, evaluation protocol, references
tests/                  # Numerical and experiment correctness checks
```

For a first reading, follow `environment.py` → `policy.py` → `rollout.py` →
`algorithms.py` → `train.py`, then inspect a saved trajectory alongside the code.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
ruff check .
ruff format --check .
```

The checks cover numerical gradients, rewards and episode endings, reproducible
sampling, checkpoint evaluation, and learning on held-out prompts.
`pyproject.toml` contains only settings for these checking tools.

## License

[MIT](LICENSE). External papers and repositories retain their own attribution and licenses.
