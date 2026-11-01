# PPO implementations

The CPU library adapters expose `train(ctx)` and use Gymnasium training environments with one Torch thread. The common [protocol](PROTOCOL.md) handles evaluation and time limits. Each adapter saves its effective configuration in the result and closes its training environments when it finishes or is interrupted.

## Stable-Baselines3 and RL Baselines3 Zoo

[`library_baselines.py`](../control_clock/library_baselines.py) uses installed Stable-Baselines3 2.7.0 ([source](https://github.com/DLR-RM/stable-baselines3/tree/bf51a6233a8f934a68430f8f78e44360410d23ca)). [configs/zoo_ppo.json](../configs/zoo_ppo.json) copies the task settings from RL Baselines3 Zoo's [PPO configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/eec15dc22e1ec7167064aae0c1218eefccaf0ff4/hyperparams/ppo.yml). Both projects use the MIT license.

| Setting | CartPole-v1 | Acrobot-v1 | LunarLander-v3 |
|---|---:|---:|---:|
| Environments | 8 | 16 | 16 |
| Steps per environment | 32 | 256 | 1,024 |
| Minibatch size | 256 | 64 | 64 |
| PPO epochs | 20 | 4 | 4 |
| Gamma | 0.98 | 0.99 | 0.999 |
| GAE lambda | 0.8 | 0.94 | 0.98 |
| Learning rate | linear 0.001 | 0.0003 | 0.0003 |
| Policy clip range | linear 0.2 | 0.2 | 0.2 |
| Entropy coefficient | 0 | 0 | 0.01 |
| Requested transition horizon | 100,000 | 1,000,000 | 1,000,000 |
| Observation/reward normalization | no | yes/yes | no |

Other settings are SB3 2.7.0 defaults: `MlpPolicy`, separate 64-by-64 tanh actor and critic, orthogonal initialization, Adam with epsilon `1e-5` and betas `(0.9, 0.999)`, minibatch advantage normalization, value coefficient `0.5` and gradient-norm cap `0.5`. Value-loss clipping, target-KL early stopping, weight decay and gSDE are disabled. The JSON lists all constructor settings.

A Zoo `lin_` schedule is `initial * progress_remaining` over the original requested horizon. Evaluation does not restart `learn` or the schedule. SB3 completes full rollouts, giving horizons of 100,096, 1,003,520 and 1,015,808 transitions if no earlier stop occurs.

Training uses `Monitor` and the sequential `DummyVecEnv`, with same-step resets. Acrobot adds `VecNormalize`: observation and reward normalization, clipping at 10, gamma `0.99`, epsilon `1e-8` and the library's initial running statistics. SB3 bootstraps time-limit truncations.

The model seed is `ctx.seed % (1_000_000 - n_envs + 1)`. A training-seed draw sets the reset base modulo the same bound; lane `i` gets `base + i`. The adapter replaces SB3's queued model-seed reset before `learn`. Later resets advance each lane's RNG.

### Checkpoints and deadlines

The first checkpoint follows model construction. A small PPO subclass calls `super().train()` and then `ctx.checkpoint(policy, num_timesteps)`, so evaluations follow all optimizer epochs. The context applies the 10,000-transition interval.

A callback checks time after each vector step. A `RolloutBuffer` subclass checks time as its original minibatch iterator yields batches. Time is also checked before and after updates; interrupted updates are never evaluated.

Evaluation uses `model.predict(..., deterministic=True)`. Acrobot observations pass through the training wrapper's `normalize_obs`, which leaves its running statistics unchanged. The evaluator collects raw Gymnasium rewards and never steps the training wrapper.

## CleanRL discrete PPO

[`cleanrl_baseline.py`](../control_clock/cleanrl_baseline.py) adapts [CleanRL's discrete-action PPO](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py). Its network initialization, rollout storage, GAE, minibatch shuffling and losses follow that source. Copyright belongs to the CleanRL developers; the MIT notice is in [third_party/CleanRL-LICENSE](../third_party/CleanRL-LICENSE).

All tasks use the same settings:

| Setting | Value |
|---|---|
| Rollout | 4 environments × 128 steps = 512 transitions |
| Optimization | 4 epochs × 4 minibatches of 128 |
| Actor and critic | Separate 64-by-64 tanh networks |
| Adam | Learning rate `2.5e-4`, epsilon `1e-5`, betas `(0.9, 0.999)` |
| Gamma / GAE lambda | `0.99` / `0.95` |
| Policy / value clip | `0.2` / `0.2` |
| Entropy / value coefficients | `0.01` / `0.5` |
| Gradient-norm cap | `0.5` |
| Requested horizon | 500,000 transitions |

The critic is initialized first. Orthogonal hidden weights use gain `sqrt(2)`; actor and critic outputs use `0.01` and `1`. Biases start at zero. Adam has zero weight decay. Learning rate is multiplied by `1 - (iteration - 1) / num_iterations`.

Advantages are normalized within each minibatch using sample standard deviation plus `1e-8`. The clipped value loss is half the mean of the larger of the unclipped squared error and the error after clipping the value change to `[-0.2, 0.2]`. There is no target-KL early stop.

The horizon rounds down to `500000 // 512 = 976` iterations, or 499,712 transitions. Observations and rewards are unnormalized. Both termination and time-limit truncation mask GAE bootstrapping, matching the source and differing from SB3.

The adapter adds a callable entry point, lazy Torch import, CPU selection, protocol seeds, deterministic evaluation, time checks and cleanup. Logging, video and tracking code are removed; episode-statistics wrappers remain. Model RNGs use `ctx.seed % 1_000_000`. The four initial lane resets come from `ctx.training_seeds(4)`; later resets advance those RNGs.

`SyncVectorEnv` uses `AutoresetMode.SAME_STEP`: a terminal step returns the final reward and flags together with the next episode's starting observation. This matches the Gymnasium behavior expected by the original script.

Evaluation runs before the training reset and after every complete iteration's epochs and minibatches. It uses argmax under `torch.no_grad()`, leaving the training action RNG unchanged. Time checks occur before rollout steps, optimizer minibatches and checkpoints.

## Tests

[`tests/test_baselines.py`](../tests/test_baselines.py) checks selected Zoo settings and schedules, CleanRL defaults, frozen SB3 evaluation normalization, same-step truncation resets and checkpoints after complete optimizer updates. The CleanRL ordering test also checks environment closure at that checkpoint. [`tests/test_algorithms.py`](../tests/test_algorithms.py) checks the same update ordering for batched PPO, ARS and CEM.

## Separate L4 comparison: JAX PPO

[`jax_ppo.py`](../control_clock/jax_ppo.py) follows [Chris Lu's PureJaxRL](https://github.com/luchris429/purejaxrl). `vmap` batches environments; nested `lax.scan` loops collect rollouts, calculate GAE and update minibatches. I split the host loop at full updates so it can run the common evaluator. The Apache-2.0 license is in [third_party/PureJaxRL-LICENSE](../third_party/PureJaxRL-LICENSE).

| Setting | Value |
|---|---|
| Actor and critic | Separate 64-by-64 tanh networks, orthogonal initialization |
| Rollout | 128 environments × 128 steps = 16,384 transitions |
| Optimization | 4 epochs, 4 minibatches |
| Adam learning rate | Constant `0.0003` |
| Gamma / GAE lambda | `0.99` / `0.95` |
| Policy / value clip | `0.2` / `0.2` |
| Entropy / value coefficients | `0.01` / `0.5` |
| Gradient-norm cap | `0.5` |

There is no observation/reward normalization or learning-rate annealing. Gymnax combines termination and time limits into `done`; both mask GAE bootstrapping, as in PureJaxRL.

Each seed starts a new Python process on the L4. Imports, CUDA initialization, JIT compilation, training and evaluation stay on the clock. Each update is synchronized before checking time. The initialized policy and every complete update are evaluated.

At a checkpoint, actor weights move to the host for a NumPy implementation of the same greedy tanh policy. Weight transfer and the 100 Gymnasium episodes count toward elapsed time. The thresholds and 120/300-second limits are the same as the CPU comparison.

[gpu-requirements.txt](../gpu-requirements.txt) pins the packages, matching the `jax` extra; Modal adds CUDA 12 wheels. [`tests/test_jax.py`](../tests/test_jax.py) checks scalar GAE, an optimizer update, matching host/JAX greedy actions and Gymnax dynamics against Gymnasium. CI runs it on CPU; the Modal job runs it on L4 before timing seeds.
