# Where the seconds go

Evaluation is the biggest cost in five of the six configurations I profiled, and while faster training environments help, they account for only 22 to 27% of an ARS or CEM run.

All numbers are from one Zen 5 core of the laptop (AMD Ryzen AI 5 PRO 340, 4.9 GHz max, `performance` governor), with one thread each for NumPy, BLAS, XLA and EnvPool. The raw data is in [`results/bench/`](../results/bench/), tabulated by [`bench_summary.py`](../bench_summary.py), and the machine details are in [`machine.json`](../results/bench/machine.json).

## Training phases

Seeds 0 to 4 of ARS, CEM and batched PPO ran twice, once normally and once with [`phases.py`](../control_clock/phases.py) timing each phase, and all 60 runs produced identical evaluations (step counts and all 100 returns). The phases add up to the solve time within 0.05%, and the table shows medians over the five seeds:

| Configuration | Solve s | Evaluation | Rollout (env step) | Update | Imports | Setup | Startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CartPole ARS | 1.18 | 65% | 30% (22%) | 0% | 5% | 0% | 1% |
| CartPole CEM | 1.25 | 64% | 29% (21%) | 0% | 5% | 0% | 1% |
| CartPole PPO | 3.66 | 14% | 13% (3%) | 47% | 16% | 10% | 0% |
| Acrobot ARS | 24.96 | 70% | 29% (26%) | 0% | 0% | 0% | 0% |
| Acrobot CEM | 18.13 | 74% | 25% (23%) | 0% | 0% | 0% | 0% |
| Acrobot PPO | 27.77 | 81% | 5% (2%) | 9% | 2% | 2% | 0% |

Shares are per-phase medians, so rows needn't sum to 100%. By Amdahl's law, even free environment simulation would cut a search run by about a quarter. Evaluating a saved policy for 100 episodes costs 0.24 to 0.33 s in Gymnasium and 0.02 to 0.04 s in the NumPy batch, with identical returns, and on Acrobot Gymnasium costs about 32 µs per transition.

Import times, as the median of ten fresh processes, are `numpy` 0.031 s, `gymnasium` 0.063, `torch` 0.683, `stable_baselines3` 1.063, `jax` 0.245, `gymnax` 1.183 and `envpool` 0.637, and interpreter startup adds another 10 to 11 ms.

## Environment throughput

The tables give million transitions per second as the median of five processes, measured through the host NumPy interface except in the scan rows. Each process timed `max(128, 8192/B)` calls after 32 warm-up calls at batch size B.

| CartPole-v1 | B=1 | B=8 | B=64 | B=512 | B=4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| My NumPy batch | 0.039 | 0.298 | 1.37 | 2.64 | 3.07 |
| Gymnasium `SyncVectorEnv` | 0.098 | 0.156 | 0.168 | 0.149 | 0.120 |
| Gymnasium `CartPoleVectorEnv` | 0.036 | 0.285 | 2.13 | 11.0 | 28.2 |
| EnvPool 1.2.7, one thread | 0.051 | 0.342 | 1.42 | 2.40 | 2.53 |
| Gymnax, `jit(vmap(step))` | 0.072 | 0.367 | 1.93 | 9.52 | 20.2 |
| Gymnax, 64 steps in `lax.scan` | 0.948 | 2.23 | 9.34 | 19.5 | 23.4 |

| Acrobot-v1 | B=1 | B=8 | B=64 | B=512 | B=4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| My NumPy batch | 0.011 | 0.088 | 0.619 | 2.23 | 3.84 |
| Gymnasium `SyncVectorEnv` | 0.026 | 0.029 | 0.030 | 0.028 | 0.028 |
| EnvPool 1.2.7, one thread | 0.052 | 0.307 | 1.07 | 1.58 | 1.53 |
| Gymnax, `jit(vmap(step))` | 0.046 | 0.321 | 1.43 | 4.49 | 6.06 |
| Gymnax, 64 steps in `lax.scan` | 0.547 | 1.43 | 3.47 | 4.70 | 4.97 |

Because Gymnasium has no vectorized Acrobot, the comparison there is with its sequential wrapper, which my NumPy Acrobot beats by 140× at B=4096. CartPole is dominated by resets: my NumPy batch builds a generator per restarted slot to match `reset(seed=s)`, which takes 90% of its time at B=4096. Gymnax's first call at each batch size takes 0.2 to 0.8 s of compilation, which is expensive when a whole ARS CartPole run uses about 0.14 million training transitions.

The first three processes ran at a load average of 2.2 to 2.6 and the last two at about 1.1. Gymnasium's large sequential batches were 10 to 25% faster in the quiet ones, but the tables use all five.

### C batch against NumPy

I rewrote both batches in C ([`classic.c`](../control_clock/classic.c), [details](environments.md#fused-c-batch)) and timed it against NumPy in a second window, same core and workload ([machine](../results/bench/native-window/machine.json), [window](../results/bench/native-window/window.json)):

| CartPole-v1 | B=1 | B=8 | B=64 | B=512 | B=4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy | 0.040 | 0.310 | 1.42 | 2.75 | 3.16 |
| C | 0.430 | 2.54 | 10.9 | 30.5 | 41.3 |

| Acrobot-v1 | B=1 | B=8 | B=64 | B=512 | B=4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy | 0.012 | 0.092 | 0.650 | 2.46 | 4.09 |
| C | 0.420 | 2.23 | 4.97 | 5.95 | 6.10 |

C is 7.7 to 13× faster on CartPole and 1.5 to 35× on Acrobot, and at B=4096 a CartPole reset costs 0.12 µs per slot instead of 6.35 µs. Every timed process first checked a 64-step trace against NumPy bit for bit, but the qualification results all still use the NumPy batch, so none of this speedup is in the README's times.

## Saved policies on new seeds

ARS, CEM and batched PPO save their passing policy after the clock stops ([`policies.py`](../control_clock/policies.py)). [`replay.py`](../replay.py) reloads each one in a new process, checks it reproduces the original 100-episode returns, then runs 1,000 episodes on seeds 2,000,000 to 2,000,999, which no earlier step used ([results](../results/bench/replay.json)). The first passing evaluation often picks a policy near the threshold, so some fail on fresh seeds:

| Configuration | Median selection mean | Median new-seed mean | Passing on 1,000 new episodes | Passing 100-episode blocks |
| --- | ---: | ---: | ---: | ---: |
| CartPole ARS | 486.5 | 486.2 | 5/5 | 50/50 |
| CartPole CEM | 492.7 | 495.6 | 4/5 | 40/50 |
| CartPole PPO | 500.0 | 500.0 | 5/5 | 50/50 |
| Acrobot ARS | −96.2 | −99.3 | 4/5 | 31/50 |
| Acrobot CEM | −97.7 | −97.4 | 4/5 | 36/50 |
| Acrobot PPO | −93.9 | −96.1 | 4/5 | 37/50 |

Four of the 30 policies fell below the threshold on the 1,000 new episodes. Only the seed 0 to 4 runs of my three methods were replayed, since SB3 and CleanRL don't save policies.

## Reproduce

```sh
uv sync --frozen --python 3.12 --extra jax --extra bench
.venv/bin/python -m control_clock.native build
.venv/bin/python bench.py all --output results/bench-replication --cpu 4 --reps 5
.venv/bin/python bench.py env --backends numpy,native --output results/bench-replication/native-window --cpu 4 --reps 5
.venv/bin/python bench_summary.py --input results/bench-replication --output results/bench-replication/summary.json
.venv/bin/python replay.py --input results/bench-replication/decompose --output results/bench-replication/replay.json
```

Pick a full-speed core and keep the machine idle while these run. EnvPool 1.2.7 needs `libEGL.so.1`, `libOpenGL.so.0` and `libGLdispatch.so.0` (Ubuntu: `libegl1`, `libopengl0`, `libglvnd0`).
