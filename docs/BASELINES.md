# Library PPO comparators

These are attributed comparators, not proposed PPO algorithms. Both expose
`train(ctx)` and use ordinary Gymnasium environments, CPU only, with one Torch
thread set by the worker. No optimized dynamics, pretrained weights, TensorBoard,
video, external tracking, progress bars or per-episode printing are used.
Both record their effective settings and source revisions in `ctx.configuration`.
The shared [protocol](PROTOCOL.md) owns evaluation environments and their cleanup;
each baseline closes its training environments in `finally`, including a partial
construction or an interrupted rollout/update. `Finished` propagates to the harness.

## Stable-Baselines3 and RL Baselines3 Zoo

[`control_clock/library_baselines.py`](../control_clock/library_baselines.py) uses
Stable-Baselines3 **2.7.0**, source commit
[`bf51a6233a8f934a68430f8f78e44360410d23ca`](https://github.com/DLR-RM/stable-baselines3/tree/bf51a6233a8f934a68430f8f78e44360410d23ca).
[`configs/zoo_ppo.json`](../configs/zoo_ppo.json) transcribes the task entries from
RL Baselines3 Zoo's [`hyperparams/ppo.yml`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/eec15dc22e1ec7167064aae0c1218eefccaf0ff4/hyperparams/ppo.yml)
at commit **`eec15dc22e1ec7167064aae0c1218eefccaf0ff4`**. The Zoo authors supply the
tuned settings; this adapter does not tune them. SB3 and Zoo are
[MIT-licensed](https://github.com/DLR-RM/rl-baselines3-zoo/blob/eec15dc22e1ec7167064aae0c1218eefccaf0ff4/LICENSE).
The installed SB3 library supplies the PPO implementation.

| Setting | CartPole-v1 | Acrobot-v1 | LunarLander-v3 |
|---|---:|---:|---:|
| Environments | 8 | 16 | 16 |
| Steps per environment per rollout | 32 | 256 | 1,024 |
| Minibatch size | 256 | 64 | 64 |
| PPO epochs per rollout | 20 | 4 | 4 |
| Gamma | 0.98 | 0.99 | 0.999 |
| GAE lambda | 0.8 | 0.94 | 0.98 |
| Learning rate | linear 0.001 | 0.0003 | 0.0003 |
| Policy clip range | linear 0.2 | 0.2 | 0.2 |
| Entropy coefficient | 0 | 0 | 0.01 |
| Requested transition horizon | 100,000 | 1,000,000 | 1,000,000 |
| Observation/reward VecNormalize | no | yes/yes | no |

Unspecified settings retain SB3 2.7.0 defaults. The JSON records them explicitly:
`MlpPolicy`, separate 64x64 Tanh policy/value networks, orthogonal initialization,
Adam with epsilon `1e-5`, advantage normalization within minibatches, no value-loss
clipping, value coefficient `0.5`, gradient norm cap `0.5`, no target-KL early stop,
no gSDE. Default Adam betas are `(0.9, 0.999)` and weight decay is zero. The remaining
constructor defaults are disclosed in the JSON; `policy_kwargs=None` deliberately
leaves policy construction to the pinned library rather than rebuilding it.

A Zoo `lin_` value is `initial * progress_remaining`, using SB3's progress over the
original requested horizon. There is no restart of `learn` at evaluation points
and no shortened schedule for the wall budget. SB3 finishes complete rollouts,
so horizon exhaustion occurs at 100,096, 1,003,520 and 1,015,808 transitions,
respectively, if not interrupted earlier. The adapter does not cap or otherwise
change the upstream schedule at the final rounded-up rollout.

Training uses `Monitor` plus `DummyVecEnv`, SB3's ordinary sequential vector
wrapper with same-step reset. Acrobot additionally uses `VecNormalize` with
training enabled, observation and reward normalization enabled, both clipping
limits `10`, gamma `0.99`, epsilon `1e-8`, and the library's running-statistics
initialization. SB3 retains its time-limit bootstrapping behavior.

The model RNG seed is `ctx.seed % (1_000_000 - n_envs + 1)`. The reset base is one
`ctx.training_seeds(1)` draw modulo that same bound. SB3 resets lane `i` with
`base + i`; the adapter replaces SB3's queued model-seed reset before `learn`.
All actually supplied model/action-space/reset seeds are below 1,000,000 and the
full lane reset list is recorded. Later autoresets use each lane's advancing RNG,
not the evaluation seed set.

### Exact checkpoint placement

There is one forced checkpoint after model construction and before `learn`.
A small PPO subclass calls `super().train()` unchanged and then immediately calls
`ctx.checkpoint(policy, num_timesteps)`. Thus a checkpoint never evaluates the
pre-update policy at SB3's `on_rollout_end`, never misses the final completed
update, and never evaluates a partially completed update. The context enforces
the 10,000-transition interval, rounded to completed PPO updates.

A callback checks time after each vector step and records transitions. A
`RolloutBuffer` subclass only checks time as the original minibatch iterator
produces each batch; it changes neither batching nor PPO mathematics. Time is
also checked before an update and by the checkpoint after it. A time interruption
between minibatches discards the incomplete update as an evaluation opportunity.

Evaluation calls `model.predict(..., deterministic=True)`. For Acrobot it first
calls the training wrapper's `normalize_obs`: SB3 documents that this method does
**not** update statistics. It does not reset/step the training wrapper during
evaluation. The shared context steps raw Gymnasium evaluation environments and
sums raw rewards, so neither rewards nor evaluation statistics are normalized.

## CleanRL discrete PPO

[`control_clock/cleanrl_baseline.py`](../control_clock/cleanrl_baseline.py) is an
adaptation of CleanRL's [`cleanrl/ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py)
at commit **`e421c2e50b81febf639fced51a69e2602593d50d`**. Copyright belongs to the
CleanRL developers. Its applicable MIT notice is retained verbatim in
[`third_party/CleanRL-LICENSE`](../third_party/CleanRL-LICENSE); unrelated notices
for other upstream algorithms are not copied. The network initialization,
rollout storage, GAE, minibatch shuffling and PPO losses are adapted from this
source, not presented as original work.

The same pinned defaults are used for every task:

- Four environments, 128 steps each, 512-transition rollouts; four epochs and
  four minibatches of 128 transitions per epoch.
- Separate 64x64 Tanh actor and critic. The critic is initialized first, as
  upstream. Orthogonal hidden weights have gain `sqrt(2)`, actor output gain
  `0.01`, critic output gain `1`, and every bias starts at zero.
- Adam learning rate `2.5e-4`, epsilon `1e-5`, betas `(0.9, 0.999)`, zero weight
  decay. Learning rate is annealed by
  `1 - (iteration - 1) / num_iterations`, without evaluation-driven restarts.
- Gamma `0.99`, GAE lambda `0.95`, minibatch advantage normalization with sample
  standard deviation plus `1e-8`, policy clipping `0.2`, entropy coefficient
  `0.01`, value coefficient `0.5`, and gradient norm cap `0.5`.
- Clipped value loss is half the mean of the elementwise maximum of unclipped
  squared error and squared error after clipping the change from the old value
  to `[-0.2, 0.2]`. Target KL is `None`; no early stopping is introduced.
- Requested horizon 500,000 transitions. Upstream floors the number of complete
  iterations: `500000 // 512 = 976`, or **499,712** actual transitions. This floor
  and the resulting annealing denominator are preserved.
- No observation or reward normalization. As in this particular upstream file,
  both termination and time-limit truncation mask GAE bootstrapping. This differs
  from SB3's time-limit handling and is intentional, not silently repaired.

The changes are instrumentation and runtime compatibility: a callable entry
point, lazy Torch import, CPU instead of automatic CUDA selection, bounded
protocol seeds, common deterministic evaluation, time checks, cleanup, and
removal of CLI/logging/video/reporting code. Episode-statistics wrappers remain.
The Python/NumPy/Torch model RNG seed is `ctx.seed % 1_000_000`; initial lane resets
use the explicitly recorded list from `ctx.training_seeds(4)`, rather than
upstream's consecutive `seed + lane` expansion. Autoresets continue the seeded
lane RNGs. The worker sets the Torch thread count to one.

`SyncVectorEnv` explicitly uses Gymnasium **`AutoresetMode.SAME_STEP`**. A terminal
step supplies the final reward and done flags, plus the next episode's reset
observation, matching older Gymnasium behavior assumed by the pinned script.
The new default `NEXT_STEP` would instead insert a reset-only transition with an
ignored action and zero reward, which this adapter does not train on.

There is a forced deterministic argmax evaluation before the training reset,
then `ctx.checkpoint` after all epochs/minibatches of every complete iteration.
The context rounds its evaluation interval to these updates. Time is checked
before each rollout step and optimizer minibatch, at each iteration start and in
the shared checkpoint. The evaluation actor uses `torch.no_grad()` and no
stochastic action sampling, so it neither trains the network nor advances the
training action RNG. Returns are always raw Gymnasium episode returns.

## Test coverage

The current baseline tests cover the following properties:

| Property | Test and scope |
|---|---|
| Pinned settings and schedules | [`test_baselines.py::test_zoo_task_settings_and_schedules`](../tests/test_baselines.py#L12) checks selected task-specific Zoo fields and the linear schedule. [`test_baselines.py::test_cleanrl_defaults_and_same_step_reset`](../tests/test_baselines.py#L45) checks selected CleanRL defaults and horizon rounding. |
| Greedy action batches | [`test_baselines.py::test_sb3_checkpoint_follows_all_optimizer_epochs`](../tests/test_baselines.py#L70) asserts the SB3 evaluation action shape. [`test_baselines.py::test_cleanrl_defaults_and_same_step_reset`](../tests/test_baselines.py#L45) exercises CleanRL's argmax action path through the vector environment. |
| Frozen SB3 evaluation normalization | [`test_baselines.py::test_sb3_evaluation_normalization_is_frozen`](../tests/test_baselines.py#L28) checks that `normalize_obs` preserves the observation-statistics count, mean and variance. |
| SAME_STEP truncation reset | [`test_baselines.py::test_cleanrl_defaults_and_same_step_reset`](../tests/test_baselines.py#L45) checks final-observation delivery and that the next action is executed rather than skipped after a truncation. |
| SB3 checkpoint-after-update ordering | [`test_baselines.py::test_sb3_checkpoint_follows_all_optimizer_epochs`](../tests/test_baselines.py#L70) checks that the first updated checkpoint follows every optimizer epoch, with the initialization checkpoint preceding training. |

Not covered by dedicated tests:

- Every individual pinned configuration field beyond the selected assertions above.
- Exact-tie action choices in SB3 and CleanRL policies.
- Termination-triggered SAME_STEP reset, rather than truncation-triggered reset.
- CleanRL checkpoint-after-update ordering.
- Environment cleanup when `Finished` interrupts initialization, rollout or optimization. The SB3 ordering test interrupts at a completed-update checkpoint but does not assert environment closure.

Training measurements use the shared process-start clock and evaluation protocol, not direct timing of a PPO update.

