# Process-start control benchmark

Recorded before implementation and tuning on 2026-10-06. The initial Git commit timestamps this document; no minute-level creation time was measured.

## Question and endpoints

How much CPU wall time does a fresh process need to learn a policy that passes a standard control threshold? Compare PPO implementations with black-box policy search; these policy-search methods use episode rewards but no value function or temporal-difference update. Calling them entirely non-RL is a convention, not a claim that reward-based search is unrelated to RL.

Tasks and passing arithmetic means over 100 complete episodes:

| Gymnasium task | Passing return | Episode cap |
|---|---:|---:|
| CartPole-v1 | >= 475 | 500 steps |
| Acrobot-v1 | >= -100 | 500 steps |
| LunarLander-v3 (stretch) | >= 200 | 1,000 steps |

Evaluation always uses deterministic greedy action selection (argmax logits, lowest-index tie break). There are 100 fixed evaluation seeds, 1,000,000 through 1,000,099, never used in training. Training RNGs and reset seeds are confined below 1,000,000. The same evaluation set is reused across checkpoints and all methods: this is a time-to-first-pass benchmark, not an unbiased final generalization estimate. Test-set adaptation through repeated evaluation is a limitation.

The parent starts a monotonic timer immediately before spawning the single-seed Python process and supplies that absolute start time to it. Time to solve ends after the first passing evaluation finishes. Python startup, imports, environment construction, compilation (if any), learning and all earlier evaluations are included. Dependency installation is excluded. A worker emits the evaluation-finish timestamp; process teardown and writing the final JSON are not part of time to solve. Parent process elapsed time is also recorded. No method receives a pretrained policy.

A full evaluation runs once before training and thereafter at the first completed algorithm update after each additional 10,000 environment transitions. A policy-search generation is an update. Evaluation costs remain on the clock; evaluations do not update parameters or training statistics. Gymnasium is the evaluation reference for every method, including optimized environments.

## Comparators and budgets

- Stable-Baselines3 PPO: pinned RL Baselines3 Zoo task hyperparameters, including observation/reward normalization for Acrobot and linear schedules where specified. Zoo training-step horizons are preserved.
- CleanRL PPO: the pinned discrete-action `ppo.py` defaults, CPU, four environments and 500,000 training transitions. Instrumentation omits TensorBoard and video, not algorithm updates.
- CPU-oriented PPO: a separately labeled candidate; changes are logged, not retrospectively presented as defaults.
- Augmented random search (ARS) and cross-entropy method (CEM): deterministic affine linear policies (bias included), or a separately labeled nonlinear candidate if linear search fails. Every algorithm and hyperparameter is explicit in the result record.

Final mandatory tasks: 20 independent seeds (0 through 19) for CPU-oriented PPO, ARS and CEM. Slow library comparators may use five seeds (0 through 4), stated beside each result. CartPole per-seed wall budget: 120 seconds. Acrobot: 300 seconds. LunarLander stretch: 300 seconds and five seeds per method initially, explicitly a small sample. All checkpoints use the same 100-episode evaluation. If an evaluation finishes after the wall budget it cannot count as success. A hard parent watchdog ensures a worker cannot hang indefinitely; interrupted runs remain failures, not missing records. Step-horizon exhaustion is also a failure within the stated wall budget.

Report success fraction and median plus 25th–75th percentile of solve times among successes. This conditional median is not the median for the entire seed cohort. Also report an all-seed Kaplan–Meier median when identifiable; unsolved runs are right-censored at their actual stop time, and a median not reached is shown as such. Five-seed comparisons and censoring do not support broad rankings.

## Measurement and development

Reported wall measurements use exclusive CPU windows, with no other CPU-heavy jobs, one seed or a batch under 30 minutes. Development trials share CPU resources and are labeled development, never mixed with final measurements. All processes run at nice 19. One Torch thread; OMP, OpenBLAS and MKL thread counts are one. CPU only, no paid compute. Results include dependency versions, task configuration, hardware information, repository-relative worker commands, evaluation trajectory and the seed. Host-specific orchestration prefixes are omitted from published commands; numerical observations are unchanged. Raw JSON is committed; a script generates the summary JSON and README table.

## Correctness

Any reimplemented dynamics must be tested against Gymnasium, including reset, every state observation, reward, termination and time-limit truncation over many seeded random action sequences and boundary cases. Stated float tolerances apply only to observations, never flags or rewards. Fast CI includes these tests and clock/aggregation tests. Optimized environments never replace the Gymnasium evaluation reference.

## Changes

The endpoints, seed split, clock and budget above are fixed. Implementation defects may be repaired and documented; hyperparameter exploration belongs in `docs/LOG.md`, with measured development effects including failures. Final methods and configurations are selected before their final cohort starts. No cherry-picking the best seed or removing failures.
