# Algorithms

Both recipes optimize a tabular categorical policy. For observable state `s`,
the trainable parameters are logits `z[s, :]`, with `pi(a | s) = softmax(z[s])[a]`.
They start at zero, producing a uniform random policy. Stable log-softmax avoids
overflow when computing action log probabilities.

The implementation uses explicit gradients in NumPy and gradient ascent. It clips
the global gradient norm before each update. Tests check the analytic gradients
against central finite differences, including repeated states, entropy, and both
signs of the clipped objective.

## REINFORCE

For rewards `r_t` and discount `gamma`, reward-to-go is:

```text
G_t = r_t + gamma * G_(t+1)
A_t = G_t - b_previous(s_t)
J   = (1 / N) * sum_i sum_t gamma^t * [log pi(a_it | s_it) * A_it + alpha * H(pi(. | s_it))]
```

`N` is the number of episodes, including all task repetitions in the batch. Step
contributions are summed within an episode; there is no inverse episode-length
weight on the return objective. The outer `gamma^t` weights discounted state
visitation; it is one at every step in the committed recipe. `alpha` is
`entropy_coef`, and `H` is categorical
entropy. As in many implementations, the entropy bonus here is a surrogate on
visited states, rather than a differentiated future-entropy return.

The baseline is an exponential moving average of observed returns for each state:

```text
b_new(s) = decay * b_previous(s) + (1 - decay) * mean_batch_return_at(s)
```

The advantages use the preceding batches' baseline. The current batch updates the
baseline only after its advantages have been computed, avoiding a baseline fitted
to the same sampled actions. Unvisited states retain their previous value.

There is exactly **one update per collected batch**. Configuration rejects
`update_epochs != 1` for REINFORCE, because additional vanilla policy-gradient
updates would reuse samples without an importance correction.

Truncated trajectories are complete finite-budget failures in this environment.
Their rewards include tool costs; returns have no value bootstrap. This treatment
would need revision for a continuing task with an artificial time limit.

For the policy-logit derivative, `grad_z log pi(a | s) = one_hot(a) - pi(. | s)`.
State contributions are accumulated with `np.add.at`, which handles repeated state
indices correctly.

## Simplified GRPO

The recipe follows the grouped relative advantage and clipped-surrogate structure
introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), with simplifications
listed below. The [verl GRPO documentation](https://verl.readthedocs.io/en/latest/algo/grpo.html)
describes the corresponding grouped sampling and training concepts.

For each sampled task, collect `G >= 2` independent trajectories from the same
unchanged behavior policy. For their undiscounted episode returns `R_i`, compute:

```text
A_i = (R_i - mean_group(R)) / (std_group(R, ddof=0) + 1e-8)
rho_it = pi_current(a_it | s_it) / pi_behavior(a_it | s_it)
S_it = min(rho_it * A_i, clip(rho_it, 1-epsilon, 1+epsilon) * A_i)
J = (1 / N) * sum_i (1 / T_i) * sum_t [S_it + alpha * H(pi_current(. | s_it))]
```

`T_i` is an episode's action count, and `N = batch_size * group_size`. Normalization
is per task group, never across unrelated prompts. The episode advantage is shared
by every action in that episode. An all-equal group has zero advantages; its reward
term contributes no gradient, though the entropy term may still contribute.

Old action log probabilities and group advantages stay fixed for all update epochs.
The ratio is per action at its observed state, not a product over a trajectory.
Clipping limits updates in the direction that improves an already extreme ratio:
positive advantages stop increasing a ratio above `1 + epsilon`, and negative
advantages stop decreasing one below `1 - epsilon`. Updates in the opposite
direction retain their gradient.

The clipping mechanism comes from [PPO](https://arxiv.org/abs/1707.06347). This
project does not implement PPO's learned value function or generalized advantage
estimation and therefore does not expose a PPO recipe.

### Differences from LLM GRPO

| Aspect | This project |
| --- | --- |
| Policy outputs | Three discrete control actions, rather than text tokens |
| Training | Tabular softmax with explicit SGD, rather than a neural model optimizer |
| Group score | Entire environment return, including tool costs |
| Discount | `gamma = 1`, enforced for GRPO |
| Objective weighting | Equal weight per episode, mean over that episode's actions |
| Reference model / KL | Absent; equivalent to choosing `beta = 0` for the KL term |
| Exploration | Configurable categorical entropy bonus |
| Batching | Synchronous collection followed by full-batch updates |

Inverse episode-length weighting and group standard-deviation normalization affect
the objective. They are explicit design choices, not a claim that this variant
matches every modern GRPO recipe. In particular, it is not DrGRPO, token-mean GRPO,
or a reproduction of DeepSeekMath's language-model training.

## Logged update statistics

| Field | Definition |
| --- | --- |
| `objective` | Maximized surrogate plus weighted entropy |
| `entropy` | Weighted mean categorical entropy at sampled states |
| `grad_norm` | Global gradient norm before norm clipping |
| `clip_fraction` | Weighted fraction of action ratios outside the clipping range |
| `approx_kl` | Weighted mean of `rho - 1 - log(rho)` against the behavior policy |

For multiple epochs these report the final update epoch, before its parameter
change. For REINFORCE the ratio statistics are zero, because its objective has no
behavior-policy ratio. `approx_kl` is a sampled diagnostic against the behavior
policy, not a penalty against a frozen reference model.
