# Timing protocol

The benchmark measures wall time from Python launch to a policy's first passing evaluation. PPO, ARS and CEM share the clock and evaluator in [`control_clock/protocol.py`](../control_clock/protocol.py).

## Passing thresholds

Each evaluation runs 100 complete Gymnasium episodes and takes the arithmetic mean of their raw returns.

| Task | Passing mean | Episode cap | Per-seed time limit |
|---|---:|---:|---:|
| CartPole-v1 | ≥ 475 | 500 steps | 120 seconds |
| Acrobot-v1 | ≥ −100 | 500 steps | 300 seconds |
| LunarLander-v3 | ≥ 200 | 1,000 steps | 300 seconds |

LunarLander only has development trials, with a shorter 120-second limit.

Actions are greedy: argmax of policy logits, with the lowest index winning a tie. Evaluation seeds are 1,000,000 through 1,000,099. CPU training RNG and reset seeds stay below 1,000,000.

The same evaluation seeds are reused at every checkpoint and across methods. Configuration selection also used them. The separate [policy replay](SYSTEMS.md#saved-policies-and-a-fresh-evaluation-cohort) measures performance on new seeds.

## Clock and checkpoints

The parent starts a monotonic timer immediately before spawning a single-seed Python process. It passes that start timestamp to the worker. The clock includes:

- Python startup and imports;
- environment and model construction, compilation and first-use work;
- training and every evaluation through the first pass.

The worker timestamps the end of the passing evaluation. Dependency installation, policy saving, result-file writing and process teardown are outside that interval. The result also includes the parent's total elapsed time.

Evaluation runs before training, then at the first complete update after each additional 10,000 training transitions. A search generation counts as an update. Checkpoints never use a partially updated policy. Evaluation does not update model parameters or training statistics, and always steps Gymnasium environments.

An initial random policy can pass. This happened for batched PPO on CartPole seed 4 and Acrobot seed 19. No method starts from pretrained weights.

An evaluation that finishes after the deadline cannot count as a pass. The parent kills a worker that exceeds the limit plus five seconds. Timeouts, worker errors and exhausted training horizons all produce unsuccessful records.

## Methods and seeds

[configs/final.json](../configs/final.json) contains the laptop run list:

| Method | Seeds per task | Configuration |
|---|---:|---|
| Batched CPU PPO | 0–19 | Zoo-shaped CartPole; original Acrobot batch/network |
| ARS | 0–19 | Affine policies; one-million/four-million transition horizons |
| CEM | 0–19 | Affine policies; one-million/four-million transition horizons |
| SB3 PPO | 0–4 | RL Baselines3 Zoo task settings and training horizons |
| CleanRL PPO | 0–4 | Four environments and 500,000 requested transitions |

The search horizons are for CartPole and Acrobot respectively. [Development experiments](DEVELOPMENT.md) explain the choices; [baseline settings](BASELINES.md) give the PPO details.

Laptop measurements used otherwise idle CPU windows, one seed at a time, at nice 19. Torch, OMP, OpenBLAS and MKL each used one thread. Development trials ran separately on the shared machine. Result files include settings, versions, worker commands, seeds and every evaluation return.

## Summary statistics

[`summarize.py`](../summarize.py) reports the success fraction and the median, 25th percentile and 75th percentile of solve times among successes. A method with no passes has no solve-time median.

The JSON also includes a Kaplan–Meier median, with unsuccessful runs censored at their actual stop time, capped at the task's deadline. Early training-horizon stops may be related to how long a policy would need to pass, so that estimate is descriptive. The plotted per-seed records keep actual stop times, including deadline overshoots. All 140 final laptop runs passed.

## Environment checks

[`tests/test_equivalence.py`](../tests/test_equivalence.py) compares resets, observations, internal states, rewards, termination and time-limit truncation against Gymnasium. Observations and states use floating-point tolerances; rewards and flags must match exactly. Boundary tests cover termination thresholds, clipping, wrapping and simultaneous termination/truncation. [Environment notes](ENVIRONMENTS.md) specify the tolerances and optional C implementation.

## Separate L4 comparison

JAX PPO ran seeds 0–2 for each task on one Modal L4. It uses the same thresholds, 100 Gymnasium episodes and 120/300-second limits. Training uses JAX-split PRNG keys. A separate pilot used seed 100.

The initialized policy and each full update are evaluated. With 128 environments and 128 steps, updates are 16,384 transitions apart. GPU work is synchronized before reading the clock. Imports, CUDA initialization, JIT compilation, weight transfer and CPU evaluation all count; cloud provisioning and image construction do not.

The hardware, training backend and PPO settings all differ from the laptop run. [JAX settings](BASELINES.md#separate-l4-comparison-jax-ppo) and [Gymnax differences](ENVIRONMENTS.md#gymnax-training-in-the-separate-l4-comparison) describe that comparison.
