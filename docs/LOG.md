# Experiment log

The protocol was committed on `main` before tuning (`9a8486a`). Development records live under `results/development/`; they are not used in the README timing table. Final cohorts use the fixed seeds and budgets in [PROTOCOL.md](PROTOCOL.md). Entries below retain failed ideas as well as successful ones.

Commands below show repository-relative executables and omit host-specific orchestration prefixes. Those publication changes do not alter numerical observations or algorithm settings.

## Setup

- Pinned Python 3.12, NumPy 2.2.6, Gymnasium 1.2.1, Torch 2.8.0 CPU and Stable-Baselines3 2.7.0.
- The first isolated Box2D build failed because SWIG was not in its build environment. Adding `box2d-py = ["swig==4.3.1"]` to uv's extra build dependencies made `uv sync --python 3.12` succeed. No OS package installation or paid compute was required. This is an installation result, not a training-speed measurement.

## Candidate definitions before development measurements

- CPU PPO uses NumPy batch dynamics, one Torch thread, a 32-unit single-layer actor and critic on CartPole, and 64-by-64 separate networks on Acrobot. Rollout batch sizes are 512 and 8,192, with optimizer minibatches 512 and 1,024. Unlike the CleanRL baseline it bootstraps time-limit truncations but ends GAE propagation at every episode boundary. It uses fixed physical observation scaling on Acrobot, not evaluation-fitted statistics.
- Planned comparisons of the same PPO candidate: Gymnasium dynamics instead of vectorized dynamics; a smaller environment/optimizer batch; and wider networks. These are changes to try, not claimed improvements. Each measured attempt will be recorded below.
- ARS uses symmetric affine-policy perturbations and a standard-deviation-scaled update. CEM fits an elite Gaussian population. Both use four common-seed training episodes per candidate. Fixed velocity scales are disclosed rather than called the original ARS observation normalization.

## Initial one-seed pilot: defaults can fail

Command: `nice -n 19 .venv/bin/python sweep.py --plan configs/development.json --chunk 0`. Every row is seed 0, a 60-second development limit, and includes process startup and all evaluations. These timings are not final comparisons: concurrent development work can affect them.

| Task | Method | First pass (seconds) | Outcome |
|---|---|---:|---|
| CartPole | CPU PPO | — | Reached 100,352 training steps without passing; best recorded mean 161.8 |
| CartPole | ARS | 1.93 | Passed at 142,461 training steps |
| CartPole | CEM | 1.57 | Passed at 115,840 training steps |
| CartPole | SB3 Zoo | 6.16 | Passed at 20,480 training steps |
| CartPole | CleanRL | 4.98 | Passed at 30,720 training steps |
| Acrobot | CPU PPO | 13.57 | Passed at 131,072 training steps |
| Acrobot | ARS | — | Reached 1,001,159 steps; last mean -166.3 |
| Acrobot | CEM | — | Reached 1,008,140 steps; last mean -123.0 |
| Acrobot | SB3 Zoo | 13.39 | Passed at 36,864 training steps |
| Acrobot | CleanRL | 20.21 | Passed at 61,440 training steps |

Source: [raw initial development records](../results/development/). The algorithm switch to affine policy search looked promising on CartPole, but neither search method solved Acrobot under its initial one-million-transition horizon. The small single-layer CartPole PPO candidate was worse than both library comparators, not an improvement. All failures remain in the data.

## Environment, batch and network comparisons

Command: `nice -n 19 .venv/bin/python sweep.py --plan configs/development.json --chunk 1`. Seeds 0 and 1, 60-second limits. Median and quartiles below are among successful seeds; these two-seed development comparisons are exploratory, not stable rankings. Raw records are linked through [the development directory](../results/development/).

| Task | Change from the initial CPU PPO | Passed | Median seconds [Q1, Q3] |
|---|---|---:|---:|
| CartPole | Gymnasium dynamics, same initial network/batch | 1/2 | 5.69 [5.69, 5.69] |
| CartPole | Optimizer minibatch 512 to 256 | 2/2 | 8.16 [7.80, 8.53] |
| CartPole | Network 32-by-1 to 128-by-2 | 2/2 | 7.65 [7.06, 8.25] |
| Acrobot | Gymnasium dynamics, same initial network/batch | 2/2 | 39.57 [34.89, 44.25] |
| Acrobot | Environments 64 to 16; minibatch 1024 to 256 | 2/2 | 14.38 [13.52, 15.24] |
| Acrobot | Network 64-by-2 to 128-by-2 | 2/2 | 26.85 [24.24, 29.46] |

CartPole seed 0 failed both initial NumPy and Gymnasium dynamics configurations, but passed with the smaller minibatch and wider network. On Acrobot seed 0, the NumPy candidate passed at 131,072 transitions and the Gymnasium version at 212,992: numerical trajectory differences affect learning, so their time ratio is **not** an isolated physics-throughput speedup. The wider network did not improve Acrobot's end-to-end pilot. I retained the original Acrobot batch/network for the final candidate and tested a Zoo-shaped CartPole schedule next.

## Additional attempted settings

`configs/tuning.json` records a CartPole candidate with 8 environments, a 64-by-64 actor/critic, 20 epochs, minibatches of 256, and linear learning-rate/clip schedules matching Zoo's task settings. Its PPO implementation still uses the separately disclosed truncation and advantage handling. The initial affine searches also receive a separately labeled four-million-transition horizon, to check whether the earlier Acrobot failures were just premature step-horizon stops. Neither change alters the wall budget, evaluation seeds or passing threshold.

The training-step counter was also corrected to record partially completed rollouts rather than only the last full update/episode. This changes bookkeeping, not policies or checkpoint timing. Feature scales and the dynamics class are now recorded explicitly in CPU PPO result configurations.

Command: `nice -n 19 .venv/bin/python sweep.py --plan configs/tuning.json --chunk 0`. The Zoo-shaped CartPole candidate passed all three pilot seeds (0–2): median 7.23 seconds, quartiles [7.03, 8.74]. The extended Acrobot ARS search passed both seeds (0–1): median 36.87 seconds [36.17, 37.58]; extended CEM also passed both: 25.20 seconds [23.07, 27.34]. The earlier one-million-transition failures were not proof that affine policies cannot solve Acrobot. These development results support using the longer search horizon, not a claim that search is faster than PPO there.

## Selected final configurations

Selected before the final cohorts start, in [configs/final.json](../configs/final.json):

- CartPole CPU PPO: `zoo-shape`, to match the Zoo task's network/batch/schedules rather than selecting the fastest two-seed candidate. This is not claimed to outperform SB3.
- Acrobot CPU PPO: `default`, original 64-environment, 64-by-64, 1,024-minibatch configuration.
- CartPole ARS/CEM: `default`, one-million-transition horizon.
- Acrobot ARS/CEM: `longer`, four-million-transition horizon. Large generations can overshoot the requested horizon; their full transition counts are recorded.
- Library PPO comparators: unchanged pinned settings, five seeds each.

All mandatory methods receive seeds 0–19 except the explicitly smaller library cohorts. Final cohorts are scheduled no earlier than 05:00 local and use separate exclusive CPU windows for each bounded chunk. Waiting for a window is not part of a seed's process-start clock. No final seed is used to change these configurations.

## LunarLander stretch: one development seed, not a final comparison

Command: `nice -n 19 .venv/bin/python sweep.py --plan configs/tuning.json --chunk 1`. Seed 0 for each method, a 120-second development limit. This is not the 300-second final protocol budget, and these are shared-machine development timings.

| Method | Outcome | Best recorded 100-episode mean | Training transitions at stop |
|---|---|---:|---:|
| CPU PPO | Time limit | 137.78 | 466,944 |
| ARS | One-million-step horizon | 196.67 | 1,016,526 |
| CEM | One-million-step horizon | 157.97 | 1,032,366 |
| SB3 Zoo | Passed in 118.64 seconds | 202.16 | 475,136 |
| CleanRL | Time limit | -137.06 | 307,200 |

Sources: `results/development/LunarLander-v3__*__default__000.json`. The pinned Zoo configuration reached the threshold in this pilot; neither the CPU-oriented candidate nor affine searches did. The best intermediate ARS mean was close but below 200 and is not counted as a success. No isolated final LunarLander cohort was run; it is not included in the main timing table. The final comparison remains CartPole and Acrobot, while installation, dynamics-fallback tests and the unsuccessful stretch attempts are retained.

## Cold review before measurement

The independent reviewer found no correctness blocker but identified interpretation limits and three useful test gaps; the historical assessment is in [COLD_REVIEW.md](COLD_REVIEW.md). I added raw internal-state comparisons to the dynamics tests, a scalar reference for the existing CEM update, a real parent/child startup-delay check, and an SB3 test that executes every optimizer epoch of the first rollout before allowing its checkpoint. The CEM equations were extracted without changing their elite selection, smoothing or standard-deviation floor.

The strengthened environment/plan selection passed 35 tests, and the targeted algorithm/baseline/clock selection passed 17 tests. [results/checks.json](../results/checks.json) records the exact commands and earlier failures. These are correctness checks, not timing comparisons; the complete data-dependent suite is run after the final cohort and summary exist.

## Replication output without deleting original measurements

The measurement and sweep drivers now accept `--output` so a fresh clone containing committed measurements can rerun the full plan into a new directory. Previously the default would correctly refuse to overwrite the original files but offered no whole-plan output override. Two routing/preservation tests passed. The default selected plan, worker parameters, clocks and scheduled final output are unchanged.

## Portable public commands

Before final timing, executable paths were made repository-relative and the public driver was changed to plain `nice -n 19` invocations. CPU exclusivity remains a measurement condition, handled outside the public package. The old marker-only final-run guard was removed because it never established exclusivity itself. Earlier development command metadata was normalized for publication; no rewards, clocks, settings or seed outcomes were changed. Three launcher/routing checks passed after the cutover.

## Per-seed plotting data

The derived summary now includes seed-sorted `records` for every cohort, carrying success flags, solve/actual-stop seconds, and stop reasons. Existing aggregate fields are unchanged. Failure points are retained rather than omitted or assigned a made-up solve time. The protocol/aggregation selection passed 10 tests, including the new ordering and actual-stop checks; this is a reporting change, not a new experiment or a policy change.

## Completed final comparison

All 28 final chunks completed successfully. The fixed plan produced 140 records: 20 seeds each for CPU PPO, ARS and CEM on each mandatory task, and five seeds each for SB3 Zoo and CleanRL. All 140 reached the threshold before their protocol limits; no final failure was dropped. [The summary](../results/summary.json) links each raw record, and [batch logs](../results/batches/) retain the bounded invocations.

The table is generated with `nice -n 19 uv run python summarize.py --input results/final --output results/summary.json`. Conditional median seconds are:

| Method | CartPole | Acrobot |
|---|---:|---:|
| CPU PPO | 4.15 | 25.01 |
| ARS | 1.08 | 28.15 |
| CEM | 1.15 | 15.34 |
| SB3 Zoo | 4.72 | 10.05 |
| CleanRL | 5.55 | 15.15 |

The candidate is not the fastest method. Linear search has the lowest observed CartPole times, and SB3 Zoo the lowest Acrobot median. The library cohorts are smaller, training objectives/settings differ, and this is not a general implementation ranking.

Two CPU PPO seeds qualified at initialization: [CartPole seed 4](../results/final/CartPole-v1__ppo-cpu__zoo-shape__004.json) and [Acrobot seed 19](../results/final/Acrobot-v1__ppo-cpu__default__019.json). They remain in the planned denominator; the fixed protocol measures qualification time, not necessarily learning time. Final LunarLander measurement was not run.

The final local correctness/evidence suite passed 57 tests with three upstream SWIG deprecation warnings. Ruff passed and its format check found 18 files already formatted. Commands and outcomes are recorded in [results/checks.json](../results/checks.json); CI runs the same checks.
