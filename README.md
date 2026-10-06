# Control Clock

Control Clock compares fresh-process CPU time to qualify policies on two Gymnasium control tasks.

**Question:** how fast does PPO solve these tasks on a laptop, and does it beat simple linear policy search?

The candidate in [`control_clock/ppo.py`](control_clock/ppo.py) uses checked NumPy batch dynamics and CPU PyTorch. [`control_clock/protocol.py`](control_clock/protocol.py) owns the common evaluation and clock; [`control_clock/search.py`](control_clock/search.py) implements affine-policy ARS and CEM. The library comparators include attributed CleanRL code and Stable-Baselines3 with Zoo settings; Gymnasium/RLPy and CleanRL license notices are retained.

**Result:** linear policy search wins CartPole here. Zoo PPO has the lowest observed Acrobot median; the batched PPO candidate is not the fastest method.

## Results

Seconds from process launch to the first passing evaluation, including Python imports, construction, first-use work and **all evaluation time**. Median and interquartile interval are among successful seeds. Success fractions include every planned seed. All numbers and per-seed plotting records come from [`results/summary.json`](results/summary.json), which links every raw record.

| Task | Method | Median seconds [Q1, Q3] | Solved |
|---|---|---:|---:|
| CartPole-v1 | ARS, affine policies | **1.08 [0.78, 1.28]** | 20/20 |
| CartPole-v1 | CEM, affine policies | 1.15 [0.93, 1.42] | 20/20 |
| CartPole-v1 | Batched PPO candidate | 4.15 [2.71, 5.46] | 20/20 |
| CartPole-v1 | SB3 PPO, Zoo settings | 4.72 [4.64, 5.23] | 5/5 |
| CartPole-v1 | CleanRL PPO defaults, adapted | 5.55 [5.33, 5.81] | 5/5 |
| Acrobot-v1 | SB3 PPO, Zoo settings | **10.05 [9.94, 12.52]** | 5/5 |
| Acrobot-v1 | CleanRL PPO defaults, adapted | 15.15 [7.28, 28.49] | 5/5 |
| Acrobot-v1 | CEM, affine policies | 15.34 [13.18, 17.11] | 20/20 |
| Acrobot-v1 | Batched PPO candidate | 25.01 [20.64, 31.52] | 20/20 |
| Acrobot-v1 | ARS, affine policies | 28.15 [25.48, 29.90] | 20/20 |

Passing means a raw mean return of at least 475 on CartPole or -100 on Acrobot over 100 complete, **greedy deterministic** evaluation episodes, with seeds disjoint from training. Wall limits are 120 and 300 seconds, respectively. Initial random policies are evaluated too: [passing at initialization](results/final/Acrobot-v1__ppo-cpu__default__019.json) counts, so these are qualification times, not necessarily time spent learning.

ARS/CEM are reward-driven black-box search, without a value function or policy-gradient updates. CartPole's result is not a reason to use a neural network where a simple affine controller works.

## Reproduce

Measured on an AMD Ryzen AI 5 PRO 340, Fedora Linux 44, one Torch/BLAS thread, CPU only; **$0 paid compute** ([hardware and compiler](results/machine.json)). Installation needs a C++ compiler for Box2D; the lockfile supplies its SWIG build tool. On an otherwise idle CPU, from the repository root:

```sh
uv sync --frozen --python 3.12
nice -n 19 .venv/bin/python measure.py --plan configs/final.json --output results/replication
uv run python summarize.py --input results/replication --output results/replication-summary.json
```

The original measurements are never overwritten. Protocol, effective settings, environment checks, failed experiments and independent review are linked from [run notes](docs/RUNNING.md). CI runs the fast correctness and evidence-consistency tests.

## Limits

- The fixed evaluation set is reused for stopping and development selection; it is not an untouched final test.
- One CPU, uncleared OS package caches, blocked method order, and smaller library cohorts: this is a small comparison, not a general tool ranking.
- Checkpoint grids and training-step horizons differ. The candidate's objective/normalization also differs from SB3; differences cannot be attributed solely to vectorized physics.
- State equivalence uses stated numerical tolerances, not bitwise identity; learning trajectories can diverge.
- LunarLander has fallback tests and single-seed development attempts, but its isolated final cohort was **not run**.

## Prior work

[Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [CleanRL](https://github.com/vwxyzjn/cleanrl), [Mania, Guy and Recht's ARS](https://arxiv.org/abs/1803.07055), and [Rubinstein's cross-entropy method](https://doi.org/10.1023/A:1010091220143).

Written with AI coding assistance.
