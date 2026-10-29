# Systems benchmark

The main table answers "which complete configuration qualifies first". This page splits that clock into parts and measures the environments on their own. Every number below comes from [`results/bench/summary.json`](../results/bench/summary.json), which [`bench_summary.py`](../bench_summary.py) derives from the raw files beside it. A test recomputes it.

## How I measured

- **Window.** I ran everything in one exclusive window on the shared laptop. One lock was held throughout, and I started only after the 1-minute load average fell below 1.5. The run went from 04:35:46 to 04:57:06 on 2026-10-29. The load average was 1.44 at the start and 1.12 at the end, and it counts the benchmark's own process ([window](../results/bench/window.json)).
- **Machine.** Every child process was pinned to CPU 4, a Zen 5 core of the AMD Ryzen AI 5 PRO 340 with a 4.9 GHz maximum; its SMT sibling is CPU 10. Governor, energy preference and platform profile were all `performance`. Each process used one NumPy/BLAS/XLA thread and EnvPool used one worker thread. This chip mixes core types: CPUs 2, 3 and 5 (and their siblings) top out at 3.4 GHz. The published unpinned runs could have landed on either kind. CPU, kernel, package versions, NumPy SIMD support and the measured commit are in [machine.json](../results/bench/machine.json). Load and frequency before every process are in each raw record.
- **Equivalence gate.** Before timing, every backend steps 64 copies of the task from its own reset states. Gymnasium steps the same states with the same actions. Rewards and termination must match exactly and observations within 1e-4; a backend that fails is not timed. All 11 backend/task pairs passed in all five processes. The largest observation differences were 0 for this repository's NumPy batch and Gymnasium, about 2e-7 for EnvPool and 5e-6 for float32 Gymnax. The fused scan returns no observations. In the committed run, its rows were gated only through the unfused Gymnax step, so they shared that gate. After the window, I added `scan_trace`, and the gate now runs it. It runs 64 fused steps and 64 unfused gated calls from the same PRNG key, and requires equal episode-end counts and final states within 1e-5. I ran it after the window: final states matched exactly on both tasks, with 171 = 171 episode ends on CartPole. Acrobot had no episode ends in 64 steps, so its auto-reset path was not exercised. A mutation test shows that corrupted fused states fail the gate.
- **Fixed workloads.** Random actions come from a fixed seed. At batch size B, a process makes max(128, 8192/B) timed calls after 32 warm-up calls. Gymnax-scan instead makes max(16, 1024/B) calls of 64 fused steps. Five fresh processes per backend and task each sweep the batch sizes in shuffled order. I add up only time spent inside backend calls, including this repository's explicit `reset_done`. Slots that a next-step backend spends restarting are not counted as transitions.
- **Interference.** Processes 0–2 ran at a 1-minute load of 2.2–2.6, and processes 3 and 4 at about 1.1. For Gymnasium's per-environment loop at B=512 and B=4096, the later processes were 10–25% faster, which gives that cell a 25% interquartile spread. I report the median of all five and did not re-run, because the window had ended. No ranking below depends on a difference that small.

## Environment throughput

Million simulated transitions per second, median of five processes. Host NumPy actions go in and host NumPy results come out, as a training loop with a host policy needs them.

| CartPole-v1 | B=1 | B=8 | B=64 | B=512 | B=4096 |
|---|---:|---:|---:|---:|---:|
| this repository's NumPy batch | 0.039 | 0.298 | 1.37 | 2.64 | 3.07 |
| Gymnasium `SyncVectorEnv` | 0.098 | 0.156 | 0.168 | 0.149 | 0.120 |
| Gymnasium's NumPy `CartPoleVectorEnv` | 0.036 | 0.285 | 2.13 | 11.0 | **28.2** |
| EnvPool 1.2.7, one thread | 0.051 | 0.342 | 1.42 | 2.40 | 2.53 |
| Gymnax, `jit(vmap(step))` per call | 0.072 | 0.367 | 1.93 | 9.52 | 20.2 |
| Gymnax, 64 steps fused in `lax.scan` | 0.948 | 2.23 | 9.34 | 19.5 | 23.4 |

| Acrobot-v1 | B=1 | B=8 | B=64 | B=512 | B=4096 |
|---|---:|---:|---:|---:|---:|
| this repository's NumPy batch | 0.011 | 0.088 | 0.619 | 2.23 | 3.84 |
| Gymnasium `SyncVectorEnv` | 0.026 | 0.029 | 0.030 | 0.028 | 0.028 |
| EnvPool 1.2.7, one thread | 0.052 | 0.307 | 1.07 | 1.58 | 1.53 |
| Gymnax, `jit(vmap(step))` per call | 0.046 | 0.321 | 1.43 | 4.49 | **6.06** |
| Gymnax, 64 steps fused in `lax.scan` | 0.547 | 1.43 | 3.47 | 4.70 | 4.97 |

Gymnasium has no vectorized Acrobot. The fused Gymnax rows never return observations to the host, so they bound on-device simulation; they are not a host training loop.

What I take from this:

- **My Acrobot batch is fast at large batches.** At B=4096 it runs 3.84 M/s: 140× Gymnasium's loop and 2.5× single-thread EnvPool. Only Gymnax beats it, at 6.06 M/s.
- **My CartPole batch is mostly spent resetting.** Gymnasium's own NumPy CartPole runs 28.2 M/s at B=4096; mine runs 3.07 M/s. A median step call at that size takes 137 µs in both. The difference is `reset_done`. To match Gymnasium's `reset(seed=s)` exactly, I build one NumPy generator per restarted slot, which costs 6.6–10 µs per slot. That is 46% of my batch's time at B=64, 79% at B=512 and 90% at B=4096. Gymnasium's vector env draws all resets from one generator, so it gives up per-seed equivalence. Batching the seeded resets is the clear target for a faster backend.
- **Small batches are dominated by per-call overhead.** At B=1 the median call takes 25 µs on my CartPole batch and 87 µs on my Acrobot batch, against 10/39 µs for Gymnasium and 17/18 µs for EnvPool. My methods train at B=8 to B=64.
- **JAX pays its compile up front.** The first Gymnax call at each new batch size took 233–515 ms on CartPole and 374–821 ms on Acrobot, in line with fresh-process JIT cost. At 20 M/s, Gymnax would need millions of CartPole transitions to repay one compile against my batch. A whole ARS CartPole qualification uses about 0.14 M.

## Imports

Seconds to import one module in a fresh pinned process, median of 10 (quartiles in the summary). Interpreter startup to the first statement took 10–11 ms in every case.

| numpy | gymnasium | `control_clock.search` | torch | `control_clock.ppo` | stable_baselines3 | jax | gymnax | envpool |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.031 | 0.063 | 0.031 | 0.683 | 0.682 | 1.063 | 0.245 | 1.183 | 0.637 |

The measuring script imports `argparse`, `json` and `pathlib` first, so those are excluded. The page cache was warm, as it was for the published runs.

## Where a qualification run's seconds go

For each of my three methods' final configurations, I reran seeds 0–4 twice under the published protocol. One run used the normal worker; the other used [`control_clock/phases.py`](../control_clock/phases.py), which wraps the calls that delimit phases without editing the training code. The order alternated by seed.

- **Equivalence gate.** All 30 plain and 30 profiled runs reproduced every evaluation of their published records exactly: the same step counts and all 100 returns of each evaluation.
- **Reconciliation.** The phases add up to the protocol's own solve clock. Time outside every phase was within 0.05% of a run; the largest was 0.045%.
- **Profiling overhead.** The median solve time grew by 0.005–0.018 s for four configurations. It grew by 0.15 s (0.8%) for Acrobot CEM and by 0.87 s (3.5% of 25 s) for Acrobot ARS. I think that is because those runs make the most wrapped environment calls, but I did not test it.

Median share of the solve clock over five seeds; the profiled run's phases ([summary](../results/bench/summary.json)):

| Configuration | Plain solve s, median | Evaluation | Rollout (env step) | Update | Imports | Setup | Startup |
|---|---:|---:|---:|---:|---:|---:|---:|
| CartPole ARS | 1.18 | 65% | 30% (22%) | 0% | 5% | 0% | 1% |
| CartPole CEM | 1.25 | 64% | 29% (21%) | 0% | 5% | 0% | 1% |
| CartPole PPO | 3.66 | 14% | 13% (3%) | 47% | 16% | 10% | 0% |
| Acrobot ARS | 24.96 | 70% | 29% (26%) | 0% | 0% | 0% | 0% |
| Acrobot CEM | 18.13 | 74% | 25% (23%) | 0% | 0% | 0% | 0% |
| Acrobot PPO | 27.77 | 81% | 5% (2%) | 9% | 2% | 2% | 0% |

PPO also spends 1% or less in GAE. Shares are medians per phase, so a row need not add to exactly 100%.

So the evaluator, not the environment, is the largest single cost in five of the six configurations. For these methods the "fast NumPy dynamics" are a minority of the clock. The exception is CartPole PPO: there the optimizer epochs take 47%, and Torch's import plus first-use setup take another 26%.

## The evaluator, measured alone

[`systems.py evaluation`](../control_clock/systems.py) times the protocol's evaluator on the seed-0 policies saved by the plain runs above, plus the all-zero initial policy. Each gets five evaluations of 100 episodes. I compared it with the same episodes on my NumPy batch, a counterfactual evaluator; the protocol does not use it.

| Policy | Transitions | Protocol s | NumPy batch s | µs per transition, protocol vs NumPy | Returns identical |
|---|---:|---:|---:|---:|---|
| CartPole, ARS seed 0 | 49,947 | 0.237 | 0.019 | 4.7 vs 0.37 | yes |
| CartPole, PPO seed 0 | 50,000 | 0.257 | 0.040 | 5.1 vs 0.81 | yes |
| Acrobot, zero weights | 50,000 | 1.568 | 0.058 | 31.4 vs 1.16 | yes |
| Acrobot, ARS seed 0 | 9,631 | 0.309 | 0.036 | 32.1 vs 3.72 | yes |
| Acrobot, PPO seed 0 | 9,939 | 0.327 | 0.031 | 32.9 vs 3.11 | yes |

All eight policies, including CEM's (in the summary), gave identical returns on both evaluators. Gymnasium's Acrobot costs about 32 µs per step. It integrates RK4 one environment at a time in Python, which I expect explains most of that cost; I did not profile inside it. That cost is why Acrobot qualification is mostly evaluation. An evaluator that passed this equality gate could save most of the time. The protocol's evaluator is unchanged, though, so the published numbers still include it.

## Saved policies and a fresh evaluation cohort

The worker now saves the passing policy of every method that exposes `policy_parameters`: my ARS, CEM and PPO. Stable-Baselines3 and CleanRL runs save nothing. Saving happens after the clock stops ([`policies.py`](../control_clock/policies.py)). If a run stops within 10 s of its time limit, the worker skips the save, because the parent's watchdog fires at the limit plus 5 s; it records that reason in place of the policy. A saved policy holds its parameters and a contract: task, threshold, evaluator, and the Gymnasium, NumPy and, for MLPs, Torch versions. Its record holds the source commit and whether tracked files had uncommitted edits, from `run.py`, or from `bench.py` for benchmark runs. In the same process, the worker rebuilds the policy from the saved JSON and checks two things. The rebuilt and live policies must choose identical actions on every observation batch of a protocol evaluation. The rebuilt policy must also reproduce the passing evaluation's 100 returns. It then records a SHA-256 digest of every action.

The 30 policies below predate `bench.py`'s source recording, so their records have `"source": null`. What I know: [machine.json](../results/bench/machine.json) records HEAD as 18cd145 when the window started. The worker, policy and training code did not change between that commit and 8dd2203, the commit with the `bench.py` fix used when the decomposition resumed. The dirty state of tracked files was not recorded for these runs.

[`replay.py`](../replay.py) runs in a new process. It replays each saved policy on the 100 protocol seeds, requires the recorded returns and action digest, then plays 1,000 fresh episodes. The fresh seeds run from 2,000,000 to 2,000,999. Training draws seeds below 1,000,000 and the protocol uses 1,000,000–1,000,099, so these seeds took no part in training, stopping or development.

I replayed the 30 policies from the plain decomposition runs (seeds 0–4 of each configuration, [replay.json](../results/bench/replay.json)). All 30 matched their recorded returns and action digests.

| Configuration | Median selection mean | Median fresh mean | Policies passing on 1,000 fresh episodes | Fresh 100-episode blocks passing |
|---|---:|---:|---:|---:|
| CartPole ARS | 486.5 | 486.2 | 5/5 | 50/50 |
| CartPole CEM | 492.7 | 495.6 | 4/5 | 40/50 |
| CartPole PPO | 500.0 | 500.0 | 5/5 | 50/50 |
| Acrobot ARS | −96.2 | −99.3 | 4/5 | 31/50 |
| Acrobot CEM | −97.7 | −97.4 | 4/5 | 36/50 |
| Acrobot PPO | −93.9 | −96.1 | 4/5 | 37/50 |

The protocol stops at the first evaluation that passes, so it picks policies near the bar. On Acrobot, 31, 36 and 37 of 50 fresh 100-episode blocks passed for ARS, CEM and PPO: 62–74%. Those are block counts pooled over five policies each, not a pass probability for a typical policy. On CartPole, one CEM policy fell to a fresh mean of 473.1. "Qualified" in the main table means "passed one fixed 100-episode test". It does not mean "passes reliably". That is five seeds per configuration, not the full 20.

## Reproduce

```sh
uv sync --frozen --python 3.12 --extra jax --extra bench
.venv/bin/python bench.py all --output results/bench-replication --cpu 4 --reps 5
.venv/bin/python bench_summary.py --input results/bench-replication --output results/bench-replication/summary.json
.venv/bin/python replay.py --input results/bench-replication/decompose --output results/bench-replication/replay.json
```

Pick a CPU that is a full-speed core on your machine and keep the machine otherwise idle. EnvPool 1.2.7 loads `libEGL.so.1`, `libOpenGL.so.0` and `libGLdispatch.so.0` on import; on Ubuntu, they come from `libegl1`, `libopengl0` and `libglvnd0`. `bench.py` passes `--jax-platform` through to JAX, so the Gymnax rows can run on a GPU later without code changes. The other backends are CPU-only.

## Limits

- One laptop, one pinned core and one thread. EnvPool and XLA scale with threads; I did not measure that.
- Five processes per cell, and the first three ran at higher background load than the last two (see Interference above).
- The gate checks short trajectories from reset states, not every reachable state. Gymnax runs in float32.
- The decomposition and fresh cohort use 5 of the 20 published seeds per configuration, and only my three methods. Stable-Baselines3 and CleanRL are not decomposed, and their cohorts are still 5 seeds.
- The NumPy evaluator is a measured counterfactual. The protocol and every published number still use Gymnasium's evaluator.
- The published timings are unchanged. The solve times in this benchmark come from pinned reruns and are not merged into the main table.
