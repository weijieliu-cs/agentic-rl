# Agentic Reinforcement Learning

围绕工具型智能体强化学习的研究代码，按方法实现、实验运行和结果复现组织。

[English](README.md) · [任务定义](docs/task.md) · [算法](docs/algorithms.md) ·
[实验与复现](docs/experiments.md) · [参考文献](docs/references.md)

## 研究问题与方法

当前实验研究一个基础问题：**策略能否通过交互奖励，学会选择正确工具并在合适时机提交答案？**

合成环境包含算术与本地查询任务，表格 softmax 策略在“计算器、查询、提交答案”三个动作中选择。
每次工具调用扣除 0.05，正确提交获得 1；最优轨迹为“正确工具 → 提交”，总回报为 0.95。

用 NumPy 显式实现两个基线：

- **REINFORCE**：逐步折扣回报，使用历史批次估计的状态基线。
- **简化 GRPO**：同任务多轨迹采样、组内回报归一化、动作概率比裁剪和多轮更新。

任务类型和工具参数由环境直接提供，因此结果反映工具路由与停止行为的学习。
当前代码是初步研究基线，尚未对应一篇已发表论文，也未训练语言模型。
具体假设见[任务定义](docs/task.md)与[算法说明](docs/algorithms.md)。

## 环境准备

需要 Python 3.11+ 和 CPU。以下命令均在仓库根目录执行：

```bash
git clone https://github.com/weijieliu-cs/agentic-rl.git
cd agentic-rl
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 训练与评估

```bash
python train.py --config configs/grpo.toml --output runs/grpo
python evaluate.py --checkpoint runs/grpo/policy.npz --episodes 256

# 贪心评估，并导出可读的工具调用轨迹。
python evaluate.py --checkpoint runs/grpo/policy.npz --greedy \
  --trajectories runs/grpo/greedy.jsonl

# REINFORCE 基线。
python train.py --config configs/reinforce.toml --output runs/reinforce

# 覆盖随机种子和训练轮数，快速检查实验。
python train.py --config configs/grpo.toml --output runs/smoke --seed 7 --iterations 5
```

每次训练使用一个新输出目录，其中保存完整生效配置、运行环境与源码指纹、逐轮指标、最终模型、
评估摘要和交互轨迹。检查点用于离线评估，尚未实现断点续训。

## 复现实验

```bash
python compare.py --seeds 0 1 2 --output runs/comparison
```

脚本依次运行两个基线的各个种子，保存完整实验产物与 `comparison.json` 汇总。
每个种子在 256 个内容独立的评估任务上测试，以下为三个种子的均值与样本标准差：

| 策略 | 采样成功率 | 平均回报 |
| --- | --- | --- |
| 初始均匀随机策略 | 33.07% ± 1.63 个百分点 | 0.2540 ± 0.0189 |
| REINFORCE | 99.74% ± 0.45 个百分点 | 0.9424 ± 0.0050 |
| 简化 GRPO | 99.74% ± 0.45 个百分点 | 0.9470 ± 0.0047 |
| 手写规则策略 | 100.00% ± 0.00 个百分点 | 0.9500 ± 0.0000 |

两个训练后策略在这些实验中的贪心成功率均为 100%。两套配方的采样预算不同，
这些结果用于验证是否学会预期行为，不能据此比较算法效率。
完整设置与局限见[实验文档](docs/experiments.md)，原始汇总见
[results/tool-routing-cpu.json](results/tool-routing-cpu.json)。

## 代码组织

```text
train.py                # 训练入口与实验循环
evaluate.py             # 加载策略、评估与导出轨迹
compare.py              # 多随机种子基线实验
algorithms.py           # REINFORCE / GRPO 目标、梯度与批次更新
environment.py          # 合成任务、工具、观测和奖励
policy.py               # 表格策略与手写规则策略
rollout.py              # 轨迹采集与评估指标
config.py               # 实验配置与 TOML 读取
utils.py                # 模型存取、实验文件和源码指纹
configs/                # 基线超参数
results/                # 已记录的实验汇总
docs/                   # 任务、算法、评估协议与参考文献
tests/                  # 数值与实验正确性检查
```

建议依次阅读 `environment.py` → `policy.py` → `rollout.py` → `algorithms.py` → `train.py`，
再对照一条保存的轨迹理解交互过程。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
ruff check .
ruff format --check .
```

检查覆盖数值梯度、奖励与终止语义、随机数复现、模型加载评估和两个基线的学习效果。
`pyproject.toml` 仅保存检查工具设置。许可证为 [MIT](LICENSE)。
