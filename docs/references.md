# References

## Foundations and algorithms

| Reading | Why it matters here |
| --- | --- |
| Williams, 1992. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696) | REINFORCE and the score-function gradient |
| Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Clipped importance-ratio surrogate used in the simplified GRPO update |
| Shao et al., 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | Same-prompt groups and relative rewards for GRPO |
| Luo et al., 2025. [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680) | Decoupling agent execution from reinforcement-learning training |

## Suggested learning sequence

1. Trace `ToolRoutingEnv` through a correct tool call, a premature answer, and a
   truncated episode. Read a saved evaluation trajectory beside the source.
2. Work through reward-to-go and the REINFORCE gradient, then run its recipe.
3. Inspect per-task reward groups, clipping, and the numerical gradient tests
   before running the GRPO recipe.
4. Compare multiple seeds using the [experiment protocol](experiments.md), keeping
   sample budgets and policy evaluation modes explicit.
5. Study the larger reference frameworks before adding a language-model backend;
   generation, model training, masking, and tool orchestration introduce additional
   correctness and infrastructure requirements.

## Further reading

- [rLLM](https://github.com/rllm-org/rllm)
- [Agent Lightning](https://github.com/microsoft/agent-lightning)
- [verl](https://github.com/verl-project/verl)
- [verl GRPO documentation](https://verl.readthedocs.io/en/latest/algo/grpo.html)

These are related projects for further study. The local experiments depend only on
NumPy; none of these frameworks is integrated here.
