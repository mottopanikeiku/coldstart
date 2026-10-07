# Control Clock

I measure fresh-process time to qualify policies on two small Gymnasium control tasks.

**Question:** does PPO—or a GPU—beat simple policy search once startup and evaluation count?

I built CPU PPO with checked NumPy dynamics, affine-policy ARS/CEM in [`search.py`](control_clock/search.py), and a shared [clock and evaluator](control_clock/protocol.py). The comparators include attributed CleanRL code and Stable-Baselines3 with RL Zoo settings. My extra [JAX PPO comparison](control_clock/jax_ppo.py) follows [Chris Lu's PureJaxRL](https://github.com/luchris429/purejaxrl); its Apache-2.0 license is retained.

**Result:** simple policy search wins CartPole on the laptop; Zoo PPO wins Acrobot. All **140/140 laptop runs** passed. The separate L4 comparison passed **6/6**, but did not improve fresh-process latency with these settings.

## Original laptop results

Seconds from Python process launch through the first passing evaluation, including imports, construction, first-use work and **every evaluation**. Medians and interquartile intervals are among successes; fractions include all planned seeds. [`results/summary.json`](results/summary.json) links all raw records.

| Task | Method | Median seconds [Q1, Q3] | Solved |
|---|---|---:|---:|
| CartPole-v1 | ARS, affine policies | **1.08 [0.78, 1.28]** | 20/20 |
| CartPole-v1 | CEM, affine policies | 1.15 [0.93, 1.42] | 20/20 |
| CartPole-v1 | Batched CPU PPO | 4.15 [2.71, 5.46] | 20/20 |
| CartPole-v1 | SB3 PPO, Zoo settings | 4.72 [4.64, 5.23] | 5/5 |
| CartPole-v1 | CleanRL PPO, adapted | 5.55 [5.33, 5.81] | 5/5 |
| Acrobot-v1 | SB3 PPO, Zoo settings | **10.05 [9.94, 12.52]** | 5/5 |
| Acrobot-v1 | CleanRL PPO, adapted | 15.15 [7.28, 28.49] | 5/5 |
| Acrobot-v1 | CEM, affine policies | 15.34 [13.18, 17.11] | 20/20 |
| Acrobot-v1 | Batched CPU PPO | 25.01 [20.64, 31.52] | 20/20 |
| Acrobot-v1 | ARS, affine policies | 28.15 [25.48, 29.90] | 20/20 |

Passing means raw mean return **475 or higher on CartPole, -100 or higher on Acrobot**, over **100 complete greedy Gymnasium episodes**. Evaluation seeds are disjoint from training. Limits are 120/300 seconds. Initial random policies count too, so this is qualification time, not necessarily time spent learning ([protocol](docs/PROTOCOL.md)).

## Extra comparison: different hardware, not a laptop-table entry

I ran JAX-vectorized PPO with Gymnax training on a **Modal NVIDIA L4**, requesting two host CPU cores and 4 GiB RAM. The evaluator, thresholds and time limits are unchanged. **Python imports, CUDA initialization, JIT compilation and all evaluations stay on the process clock.** At checkpoints I copy weights to the host and evaluate the greedy policy in actual Gymnasium.

| Task | JAX PPO median seconds [Q1, Q3] | Solved |
|---|---:|---:|
| CartPole-v1 | 26.38 [25.99, 33.77] | 3/3 |
| Acrobot-v1 | 97.43 [80.20, 126.36] | 3/3 |

Source: [`results/gpu/summary.json`](results/gpu/summary.json), [six raw runs](results/gpu/final). This changes hardware, training backend and PPO settings together: it does **not** isolate a GPU speedup or measure already-compiled throughput. Small tasks do not automatically benefit from buying a GPU when startup and evaluation count.

Cloud client preparation, image work, provisioning and shutdown are **outside** each policy's process clock. Their combined overhead was about **18 seconds for the final batch** and **218 seconds for the pilot**, not an extra charge added to every seed. My total conservative cloud-cost estimate is **$0.3745**, including two failed setup attempts; it is not an invoice ([cost and overhead](results/gpu/cost.json)). Gymnax's default dynamics match within stated tolerances, with real precision, PRNG, timeout and wrapping differences documented in [environment notes](docs/ENVIRONMENTS.md#gymnax-training-in-the-extra-gpu-comparison).

## Reproduce

The original CPU measurements used an AMD Ryzen AI 5 PRO 340, Fedora Linux 44, and one Torch/BLAS thread ([machine](results/machine.json)). CPU installation needs a C++ compiler for Box2D. The GPU command needs a Modal account already authenticated with `modal setup`, but no local GPU. Choose fresh output directories; use an otherwise idle CPU for laptop measurements.

```sh
uv sync --frozen --python 3.12
nice -n 19 .venv/bin/python measure.py --plan configs/final.json --output results/replication
uvx --from modal==1.5.3 modal run modal_app.py --phase final --output results/gpu-replication
```

[Run notes](docs/RUNNING.md) cover summaries, settings, attribution, failed development trials and correctness checks. No new laptop timings were used for the GPU comparison.

## Limits

- The fixed evaluation set is reused for stopping and development selection, not held out.
- Two small tasks, one laptop and three GPU seeds per task do not establish a general ranking.
- Objectives, normalization, checkpoint grids and training horizons differ between methods.
- Dynamics are tolerance-checked, not bitwise identical; OS/driver caches and cloud CPU placement are not controlled.
- LunarLander's isolated final cohort remains **not run**; only development attempts exist.

## Prior work

[Gymnasium](https://gymnasium.farama.org/), [Gymnax](https://github.com/RobertTLange/gymnax), [PureJaxRL](https://github.com/luchris429/purejaxrl), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [CleanRL](https://github.com/vwxyzjn/cleanrl), [Mania, Guy and Recht's ARS](https://arxiv.org/abs/1803.07055), and [Rubinstein's cross-entropy method](https://doi.org/10.1023/A:1010091220143).

Written with AI coding assistance.
