# Independent cold review

Reviewed on 2026-10-06, before final timing/results and the final README. Scope: the directives, protocol, all algorithm/environment/harness/summary source, configuration plans, tests, attribution, development records, and the pinned upstream sources linked below. This was a static review: I ran no tests, builds, linters, formatters, training or timing experiments and changed no source. The only review artifact is this document.

## Verdict and blocking findings

**No demonstrated correctness blocker found. Final measurement can proceed with the selected configurations.** The current code supports a defensible small, CPU-only comparison of fresh-process time to the first passing *fixed evaluation set*. It does not support an unbiased held-out generalization claim, an isolated environment-throughput claim, or a broad ranking of RL implementations. Missing final numbers and README are intentionally outside this intermediate verdict.

There is no P0/P1/P2 implementation finding requiring a pre-measurement repair. The prioritized limitations below are real interpretation/evidence boundaries, not bugs disguised as change requests. The later results review still needs to establish cohort completeness and consistency with the selected plan.

## Prioritized non-blocking limitations and concrete handling

### 1. Reused evaluation seeds are a stopping/selection set, not untouched test data

**Evidence:** `control_clock/protocol.py:14-16,49-87` resets the same 100 seeds at every eligible checkpoint and stops on the first threshold crossing. Development selected configurations using that evaluation procedure (`docs/LOG.md`, “Additional attempted settings” and “Selected final configurations”). Training reset seeds remain disjoint; that does not eliminate repeated-checkpoint or development-selection adaptation to the evaluation set. `docs/PROTOCOL.md` already acknowledges this.

**Handling:** In the final README call this time to first pass on a fixed 100-seed evaluation set, not time to generalize or an independent final test. Keep the existing limitation visible beside the comparison. An unbiased generalization question would require a separate, previously unused seed set evaluated only after selection; that would be a separate endpoint, not a silent replacement of the current protocol. No change to the scheduled measurements is needed.

### 2. Early horizon failures make the KM median descriptive

**Evidence:** `summarize.py:13-33` treats unsolved records as censored at actual stop time, capped by the wall limit. `control_clock/worker.py:39-47` labels ordinary algorithm return as step-horizon exhaustion. Real early failures exist: `results/development/CartPole-v1__ppo-cpu__default__000.json:1-5` stops at about 9.88 seconds without success after 100,352 transitions; the initial Acrobot search records also stop at their horizons. This is not equivalent to observing every seed through its full wall budget. `docs/RUNNING.md`, “Tables,” correctly warns that early censoring need not be independent of eventual solve time.

**Handling:** Always show success count/cohort size alongside success-conditional median and quartiles. Carry the informative-censoring caveat into final presentation whenever KM is shown; do not describe a KM estimate as the unconditional median of the finite-horizon procedure or turn it into an unsupported ranking. In particular, a KM median can exist with fewer than half the cohort successful after early failures leave the risk set. The implemented event-before-censor ordering is correct; this concern is interpretation, not an arithmetic defect.

### 3. This compares complete algorithm packages, not just faster physics

**Evidence:** CPU PPO and libraries differ in batches, normalization, initialization and truncation/advantage handling (`control_clock/ppo.py:12-50,80-87,107-119,179-214`; `docs/BASELINES.md`). CPU PPO uses a half-MSE value loss multiplied by 0.5 (`ppo.py:210-213`), as CleanRL does; SB3 multiplies a full MSE by its value coefficient. “Zoo-shaped” therefore means the disclosed network/batch/schedules, not identical SB3 mathematics. Search uses affine deterministic policies, fixed physical velocity scales, four common-seed episodes per candidate, and full generations (`control_clock/search.py:17-43,47-117`). The optimized-versus-Gymnasium Acrobot pilot reaches different training steps before passing; `docs/LOG.md` explicitly avoids calling that timing ratio a physics-only speedup.

**Handling:** Preserve the candidate/variant labels, effective settings and raw training-step counts. Describe ARS as the disclosed discrete-action adaptation, not a reproduction of the original continuous-control ARS results. Do not attribute an end-to-end difference solely to NumPy dynamics. The five-seed library cohorts versus twenty custom seeds must remain explicit; they support a small descriptive comparison, not precise population-wide superiority claims. These differences were selected and documented before final cohorts, so no cutover is requested.

### 4. Acceptance coverage is stronger for dynamics than for end-to-end wiring

**Evidence:** `tests/test_equivalence.py:42-74` compares observations and all four internal state coordinates against independent Gymnasium instances over 600-step sequences for three action-RNG seeds, with exact rewards/flags. Additional tests cover slot-order resets, inactive masks, full 500-step classic time limits, terminal observations, simultaneous termination/truncation, boundary thresholds, wrapping/clipping and the Lunar time-limit boundary (`:78-295`). Observation/internal-state tolerance is `atol=2e-6, rtol=2e-7`, not bitwise equivalence. Lunar physics is delegated to Gymnasium rather than independently reimplemented.

`tests/test_algorithms.py:8-60` independently checks GAE, ARS updates and affine actions. The suite does not independently compare the CEM elite-distribution update or a full optimizer update to upstream. `tests/test_baselines.py:10-65` checks selected settings, frozen observation statistics and SAME_STEP reset behavior, but not an entire instrumented training iteration. Clock tests cover an inherited offset and a late final evaluation, not an actual parent/child launch/import-delay test (`tests/test_protocol.py:17-70`). Static source tracing finds the intended wiring; this is not a claim that those missing integration scenarios were exercised.

**Handling:** State the checked behaviors accurately rather than claiming every algorithm is exhaustively equivalence-tested. For later changes to the harness/adapters, the most targeted additions are a parent/child timestamp test, a checkpoint-after-full-update test, and a scalar CEM-update reference. They are not prerequisites established by this review for the unchanged code. `results/checks.json` records an earlier 48-pass run and formatting history; it is not evidence that the subsequently extended suite passed. Main owns verification of the final integrated tree.

## Explicit acceptance assessment

### Clock origin, propagation and first pass

`run.py:63-70` takes `time.perf_counter()` immediately before `Popen`, passes its full float representation as `--started`, and does not substitute a worker-local origin. `worker.py:25-40` forwards the value to `RunContext` and dispatches every advertised method to its training entry point. On this Linux host the monotonic clock is shared across processes. Worker module imports, Gymnasium/Torch/library imports, environment/model construction, rollouts, updates and previous evaluations all occur after that origin.

`protocol.py:72-87` uses the timestamp after the last evaluation transition as solve time, and accepts only a completed 100-episode evaluation at or before the deadline. JSON writing, version metadata collection and teardown are outside that endpoint; parent elapsed time is separately retained. The per-checkpoint `evaluation_seconds` starts after initial evaluation-environment construction (`protocol.py:53-55`), so it is not a complete additive breakdown of evaluation setup/import cost. Those costs still remain in `elapsed_seconds` and `solve_seconds`; there is no solve-clock subtraction bug.

Every method performs an initial evaluation. PPO checkpoints follow all optimizer epochs/minibatches; SB3 explicitly subclasses `train()` rather than evaluating at rollout end (`library_baselines.py:79-83,132-133`). Search checkpoints follow a completed generation (`search.py:95-117`). The context then schedules another checkpoint after 10,000 additional training transitions, rounded to the next completed update (`protocol.py:49-52,83`). Different update sizes therefore produce different checkpoint grids, as the protocol states.

### Seeds, deterministic actions and normalization

Training seed draws are in `[0,1000000)` (`protocol.py:38-42`); evaluation seeds are exactly 1,000,000 through 1,000,099. CPU PPO and search use only context-generated reset seeds. SB3 bounds the reset base so every `base + lane` stays below the split and overrides the queued model-seed reset before `learn` (`library_baselines.py:85-95,122-123`). CleanRL uses recorded per-lane seeds and SAME_STEP autoresets (`cleanrl_baseline.py:93-94,160-164,182-184`). Later library autoresets continue each seeded RNG; seed disjointness is not a claim that all training trajectories are identical across methods.

All custom/CleanRL policies use argmax; SB3 uses deterministic categorical prediction. There is no evaluation action sampling, parameter update, training reset or fitted custom normalization on evaluation observations. SB3 evaluation calls `VecNormalize.normalize_obs`, which does not update statistics (`library_baselines.py:125-130`; pinned upstream source below); raw Gymnasium rewards are accumulated by the context. Acrobot training does use both observation and reward normalization in the Zoo comparator. Gymnasium seeding uses PCG64, matching the custom classic resets, including Acrobot float32 initial-state rounding. Frozen observation normalization is explicitly tested; reward-statistic immutability is established here by the evaluation call path, not a separate test assertion.

### Timeouts, horizons and environment endings

Time checks occur in evaluation and training loops/minibatches. Late final batches cannot pass (`protocol.py:79-87`; `run.py:116-117`). The parent kills a nonreturning worker after limit plus five seconds and retains a failure record (`run.py:69-85`); watchdog time is not a successful solve time. Cooperative budget interruption can discard an incomplete evaluation/update. Horizon failures are retained rather than omitted.

CPU PPO bootstraps the *final* observation on truncation but not on true termination, and stops GAE propagation at either boundary (`ppo.py:80-87,163-180`). Explicit-reset batches return stable final-observation snapshots and do not step completed active slots (`vector_env.py:42-58,77-89,101-120`). SB3 retains upstream terminal-observation timeout bootstrapping; CleanRL intentionally masks both endings, matching its pinned source. Simultaneous termination/truncation is not treated as a bootstrap opportunity.

Complete rollouts/generations can overshoot requested horizons; CleanRL floors complete iterations to 499,712 transitions. SB3 and Zoo-shaped CPU PPO preserve schedules over the requested CartPole horizon: the rounded-up final rollout can make progress, learning rate and clip negative. The upstream SB3 behavior and preservation choice are already disclosed in `docs/BASELINES.md`; it is not a newly invented safety clamp or a baseline-default mismatch. Do not silently claim exact horizon caps.

### Cohorts, statistical arithmetic and evidence provenance

`configs/final.json` covers both mandatory tasks with custom PPO/ARS/CEM seeds 0–19 and library seeds 0–4. Acrobot search uses the selected longer horizon, and CartPole custom PPO uses the selected Zoo-shaped variant; these values route through `sweep.py`, `run.py`, `worker.py` and the consuming algorithm branches. Each five-seed Acrobot window allows 1,525 watchdog seconds, below 30 minutes. The measurement protocol requires exclusive CPU windows and nice 19; public reproduction commands assume an otherwise idle CPU. `run.py` sets one-thread numeric-library environments; the worker sets both Torch thread limits.

`summarize.py:36-51` retains failures in success fractions and computes NumPy quartiles only among successes, using separate conditional labels. Its KM arithmetic handles tied events before censor removals. Grouping preserves variants, rejects duplicate seeds and mixed phases, but the summarizer alone does not enforce a complete final plan. `tests/test_results.py:29-65` supplies that exact cohort/endpoints check and `:69-74` recomputes saved summaries. Run those data-dependent acceptance checks only after final records and summary exist; their absence before timing is expected, not an intermediate defect.

Git history shows the protocol/dependencies committed in `9a8486a` before implementation commit `1c1888d`. Development raw JSON labels phase, seed, budget, evaluations, commands and versions, including unsuccessful attempts; `docs/LOG.md` keeps development timing separate and records configuration selection before final timing. Source revisions and metadata need to remain tied to the final integrated tree; no final results or README assertions have been reviewed here.

## Attribution and primary-source checks

The vector environment header credits Gymnasium and Acrobot/RLPy, with complete MIT/BSD notices and pinned source links in `docs/ENVIRONMENTS.md`. The CleanRL adaptation names its source commit and retains its MIT notice in `third_party/CleanRL-LICENSE`. `docs/PRIOR_WORK.md` distinguishes the discrete ARS/CEM adaptations and notes EnvPool; it does not claim batched environment execution is novel.

I read the relevant pinned primary sources rather than accepting the adapters own descriptions: [Zoo task settings](https://github.com/DLR-RM/rl-baselines3-zoo/blob/eec15dc22e1ec7167064aae0c1218eefccaf0ff4/hyperparams/ppo.yml), [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py), [SB3 PPO](https://github.com/DLR-RM/stable-baselines3/blob/bf51a6233a8f934a68430f8f78e44360410d23ca/stable_baselines3/ppo/ppo.py), [SB3 normalization](https://github.com/DLR-RM/stable-baselines3/blob/bf51a6233a8f934a68430f8f78e44360410d23ca/stable_baselines3/common/vec_env/vec_normalize.py), [SB3 rollout/timeout handling](https://github.com/DLR-RM/stable-baselines3/blob/bf51a6233a8f934a68430f8f78e44360410d23ca/stable_baselines3/common/on_policy_algorithm.py), and Gymnasium 1.2.1 [seeding](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/utils/seeding.py), [CartPole](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/classic_control/cartpole.py), and [Acrobot](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/classic_control/acrobot.py). The inspected defaults and intentional compatibility changes match the stated comparator definitions.

## Main-owned verification and later evidence review

No verification was executed by this reviewer. Main should retain outcomes for the exact CI commands in `.github/workflows/ci.yml`: `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q` on the final integrated tree after the data-dependent files land. The later README/results pass should check raw cohort completeness, first-pass/deadline arithmetic, conditional spread, censoring language, benchmark batch provenance, and each published number against its linked record. This intermediate review approves proceeding with measurement, not publishing unobserved results.


## Final-data assessment — 2026-10-06

This section supersedes the earlier pending-results/test-coverage statements without rewriting the historical review. I re-read the README, selected plan, complete summary, final-record inventory, initialization-pass records, updated drivers/tests, check evidence and reachable external timing records. No tests, builds, linters, formatters or experiments were run by this reviewer. Only this review document was edited.

**Verdict: no substantive correctness or evidence-consistency blocker found. The published result is defensible as the stated small, fixed-set qualification-time comparison.** It is a useful negative result for the CPU PPO candidate, not evidence that batching physics generally improves time to learn.

### Results and cohorts

The ten groups in `results/summary.json` match `configs/final.json`: both mandatory tasks, custom PPO/ARS/CEM seeds 0–19 and library seeds 0–4, totaling 140 records. The final directory inventory contains those cohorts; summary file references and seed-sorted plotting records retain every planned seed. All ten groups report complete success, and a source search of final records found no unsuccessful outcome, worker error, nonzero return code, watchdog kill or non-final phase. The exercised data-consistency tests recorded by main additionally check exact plan equality, duplicates, evaluation return counts/means, threshold/deadline flags, first-pass placement and aggregate recomputation (`tests/test_results.py`). This is not an independently rerun acceptance suite.

All ten README medians, quartiles and success fractions agree with the summary after rounding to two decimal places. CartPole medians are ARS 1.08, CEM 1.15, candidate PPO 4.15, Zoo PPO 4.72 and CleanRL 5.55 seconds. Acrobot medians are Zoo PPO 10.05, CleanRL 15.15, CEM 15.34, candidate PPO 25.01 and ARS 28.15 seconds. The README correctly says Zoo has the *lowest observed* Acrobot median, states unequal cohort sizes, and does not claim the candidate wins. No final LunarLander group exists; its single-seed development attempts remain explicitly separate.

Two candidate policies pass before training: CartPole seed 4 has `steps=0`, all 100 returns equal to 500, and qualification time 1.3868 seconds; Acrobot seed 19 has `steps=0`, mean return -95.57 and qualification time 1.4257 seconds. Both remain in their cohorts. The README explicitly discloses initialization qualification and links the Acrobot record. Its opening now says qualify policies, so these observations are not misrepresented as learning progress. The unchanged protocol explicitly required initial evaluation.

### Clocks, provenance, privacy and replication

The public launcher still supplies the parent timestamp to the worker and uses the common deterministic 100-episode Gymnasium evaluator, unchanged thresholds, seed split and deadline checks. Portable executable paths and removal of a marker-only guard do not change the timing origin. CEM extraction preserves its equations, and its returned mean/std are actually consumed by the training branch.

Reachable external driver source invokes the exclusive timing wrapper separately for all 28 selected chunks. Its append-only batch record contains indices 0–27 exactly once, all successful, with sequential non-overlapping invocations starting no earlier than the scheduled time; every interval is under 30 minutes. Public batch timestamps fall within their corresponding external invocation intervals. For example, the first public batch spans 05:01:25–05:01:49, and the last spans 06:25:26–06:27:08. Queue waits are outside worker clocks. This supports actual timing provenance beyond the portable published command strings; I did not independently monitor scheduler behavior while runs occurred. External artifacts are intentionally not copied into the tracked review.

`measure.py` now launches nice-19 batches rather than owning machine exclusivity. The README accurately requires an otherwise idle CPU for replication. `--output` is forwarded through `measure.py` and `sweep.py` to `run.py`; seed-result overwrite refusal remains, and replication batch-log names are separate from original logs. The two driver tests cover routing and preservation. The documented three-command reproduction writes to a new result directory and does not delete originals.

A repository content search found none of the prohibited home-path, private-launcher or workspace identifiers. Public commands use relative paths; `docs/RUNNING.md` explicitly explains publication normalization without claiming the observations changed. This search assesses the current publishable content, not historical Git objects or future additions.

### Coverage and remaining limits

The three previously identified coverage gaps now have concrete tests: scalar CEM elite/smoothing/floor behavior including stable ties, a real child-startup delay before worker imports, and an SB3 first update whose twenty Adam steps all precede the updated checkpoint. Main records Ruff success, 18 already-formatted files, and a full 57-pass suite with three upstream SWIG warnings in `results/checks.json`. I inspected that evidence and test source, not a new execution.

The README carries the important remaining limits: reused evaluation seeds, one machine with warm OS caches and blocked method order, five-seed library cohorts, differing checkpoint grids/horizons/objectives, numerical rather than bitwise dynamics equivalence, and no final stretch cohort. Since all mandatory final seeds succeeded, informative early censoring does not affect this README table; the historical KM caveat still applies to development failures and later replications. No broad ranking, untouched-test generalization or physics-only speedup claim is warranted. No source repair or additional timing run is required by this final assessment.
