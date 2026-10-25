# Experiments and reproducibility

## Reproduce the committed recipes

From the repository root, after activating a Python 3.11+ environment:

```bash
python -m pip install -r requirements.txt
python compare.py --seeds 0 1 2 --output runs/comparison
```

The output directory must be new. The comparison script saves each complete run
under `<output>/<algorithm>/seed-<seed>/`, then writes aggregate metrics and all
run summaries to `<output>/comparison.json`. Standard deviations in that file use
`ddof=1` across seeds, with zero reported for a single-seed run.

For an individual run and standalone evaluation:

```bash
python train.py --config configs/grpo.toml --output runs/grpo-seed-7 --seed 7
python evaluate.py --checkpoint runs/grpo-seed-7/policy.npz \
  --episodes 256 --seed 100007
python evaluate.py --checkpoint runs/grpo-seed-7/policy.npz \
  --episodes 256 --seed 100007 --greedy \
  --trajectories runs/grpo-seed-7/greedy.jsonl
```

To reproduce a run's final sampled evaluation, set the standalone evaluation seed
to the training seed plus 100000, and match `eval_tasks`. The evaluation script's
standalone default is always seed 100000 and 256 episodes. Changing the evaluation seed
changes both prompt generation and action sampling. Greedy evaluation chooses
the maximum-probability action; ties use the first action index.

## Protocol

| Setting | REINFORCE | Simplified GRPO |
| --- | --- | --- |
| Recipe | `configs/reinforce.toml` | `configs/grpo.toml` |
| Training seeds | 0, 1, 2 | 0, 1, 2 |
| Iterations | 300 | 200 |
| Sampled tasks per iteration | 64 | 32 |
| Episodes per sampled task | 1 | 4 |
| Total training episodes per seed | 19,200 | 25,600 |
| Gradient epochs per batch | 1 | 4 |
| Learning rate | 1.0 | 1.0 |
| Entropy coefficient | 0.01 | 0.01 |
| Discount | 1.0 | 1.0 |
| Maximum actions per episode | 4 | 4 |
| Cost per tool call | 0.05 | 0.05 |
| Training prompt pool | 256 | 256 |
| Evaluation prompts per seed | 256 | 256 |

Evaluation excludes training-pool contents by using different arithmetic operand
ranges and lookup keys. Task types, tools, and the state abstraction are shared.
The synthetic prompt pool can contain repeated contents. These experiments test
whether each implementation learns the intended control flow.

Each evaluation seed samples one episode for each of its 256 generated prompts.
All reported final sampled metrics use the **last training iteration**, with no
selection of a checkpoint based on evaluation scores. The table below reports
mean and sample standard deviation across training seeds. This is descriptive
variation over three small runs, not a confidence interval or a statistical
significance test.

The recipe budgets differ in episode count and optimizer steps. Use equal episode
or interaction budgets, additional seeds, and an explicit tuning protocol before
claiming sample or compute efficiency. The comparison script currently runs the
committed recipes; it does not automatically match their budgets.

## Recorded CPU results

Measured on **2026-10-25** with Python **3.11.16** and NumPy **2.4.6** on Linux
x86-64, with NumPy pinned in `requirements.txt`. The reorganized scripts
were rerun for all six algorithm/seed combinations. Machine-readable measurements,
recipe hashes, the requirements hash, and the source fingerprint are preserved in
[tool-routing-cpu.json](../results/tool-routing-cpu.json).

| Policy | Sampled success, mean ± SD | Mean return ± SD | Mean actions ± SD |
| --- | --- | --- | --- |
| Uniform initial policy | 33.07% ± 1.63 pp | 0.2540 ± 0.0189 | 2.3555 ± 0.1182 |
| REINFORCE | 99.74% ± 0.45 pp | 0.9424 ± 0.0050 | 2.0990 ± 0.0098 |
| Simplified GRPO | 99.74% ± 0.45 pp | 0.9470 ± 0.0047 | 2.0078 ± 0.0068 |
| Hand-written oracle | 100.00% ± 0.00 pp | 0.9500 ± 0.0000 | 2.0000 ± 0.0000 |

`pp` means percentage points. Both learned policies achieve 100% success with
greedy actions in all three runs, with two actions and one tool call per episode.
The uniform policy's sampled result differs from its greedy result because tied
logits always choose the first action in greedy mode.

The near-perfect scores reflect the small state and action spaces and the supplied
task-kind feature. They do not measure arithmetic learning, language understanding,
or general agent performance. The oracle directly implements the intended routing
rule and provides a reward/lifecycle sanity check.

## Artifacts and metric definitions

| Artifact | Contents |
| --- | --- |
| `config.json` | Complete effective configuration, including default and overridden values |
| `metadata.json` | UTC start time, software versions, platform, Git revision if available, source SHA-256 |
| `metrics.jsonl` | Initial evaluation at iteration zero, training/update metrics each iteration, periodic evaluation |
| `policy.npz` | Final logits, checkpoint version, and serialized configuration |
| `summary.json` | Initial, final sampled, final greedy, and oracle evaluations; training episode budget |
| `trajectories.jsonl` | The exact final sampled evaluation episodes used in `final_eval` |

Environment metrics are:

- `episodes`: number of evaluated or collected trajectories.
- `success_rate`: fraction with a verified correct submission.
- `mean_return`: mean sum of rewards, including every tool cost.
- `mean_steps`: mean number of all actions, including submission.
- `mean_tool_calls`: mean number of calculator and lookup calls, including failed calls.
- `truncation_rate`: fraction exhausting the action budget before submission.

See [algorithm diagnostics](algorithms.md#logged-update-statistics) for the update
statistics. Training and sampled evaluation preserve stochastic action selection;
greedy evaluation is separately labeled in `summary.json`.

## Reproducibility checks

The test suite verifies that repeated runs with the same configuration and seed
produce identical summaries and that changing evaluation cadence leaves learned
logits unchanged. It also checks checkpoint round trips, numerical gradients,
discounted policy-gradient semantics, environment termination, and both algorithms'
improvement over their initial policy.

Use the pinned NumPy dependency and corresponding repository sources for the
recorded results. New Python/NumPy versions may change numeric details or random
sampling. Timestamps and platform metadata naturally differ between runs. The
lightweight CI runs the correctness checks and a short train/evaluate
experiment on Python 3.11. Exact benchmark scores across Python versions are not
required.
