# PPO baselines

The library adapters expose `train(ctx)`, train on Gymnasium environments with one Torch thread, and save their effective configuration in each result. Evaluation and time limits come from the shared [protocol](protocol.md).

## Stable-Baselines3 with Zoo settings

[`library_baselines.py`](../control_clock/library_baselines.py) runs Stable-Baselines3 2.7.0 with the task settings from [RL Baselines3 Zoo's PPO config](https://github.com/DLR-RM/rl-baselines3-zoo/blob/eec15dc22e1ec7167064aae0c1218eefccaf0ff4/hyperparams/ppo.yml), copied into [`configs/zoo_ppo.json`](../configs/zoo_ppo.json). Both projects are MIT licensed.

| Setting | CartPole-v1 | Acrobot-v1 |
| --- | ---: | ---: |
| Environments | 8 | 16 |
| Steps per environment | 32 | 256 |
| Minibatch size | 256 | 64 |
| Epochs | 20 | 4 |
| Gamma / GAE lambda | 0.98 / 0.8 | 0.99 / 0.94 |
| Learning rate | linear 0.001 | 0.0003 |
| Clip range | linear 0.2 | 0.2 |
| Entropy coefficient | 0 | 0 |
| Requested horizon | 100,000 | 1,000,000 |
| Observation and reward normalization | no | yes |

Everything else is an SB3 2.7.0 default: `MlpPolicy` with separate 64×64 tanh actor and critic, Adam (epsilon 1e-5), advantage normalization per minibatch, value coefficient 0.5, gradient-norm cap 0.5. Zoo `lin_` schedules run over the originally requested horizon. SB3 finishes whole rollouts, so the horizons are really 100,096 and 1,003,520 transitions.

The first evaluation follows model construction. A small `PPO` subclass calls `train()` and then the checkpoint, so every evaluation comes after a full set of optimizer epochs. Time is checked after each vector step and between minibatches; an interrupted update is never evaluated. Evaluation uses `predict(deterministic=True)` and, on Acrobot, the training wrapper's frozen observation normalizer. Rewards come from raw Gymnasium environments.

## CleanRL discrete PPO

[`cleanrl_baseline.py`](../control_clock/cleanrl_baseline.py) adapts [CleanRL's `ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py): same initialization, rollout storage, GAE, minibatch shuffling and losses. MIT notice: [third_party/CleanRL-LICENSE](../third_party/CleanRL-LICENSE). Logging, video and tracking are stripped; the protocol's seeds, evaluation, time checks and cleanup are added.

| Setting | Value |
| --- | --- |
| Rollout | 4 environments × 128 steps = 512 transitions |
| Optimization | 4 epochs × 4 minibatches of 128 |
| Networks | Separate 64×64 tanh actor and critic |
| Adam | learning rate 2.5e-4 (linearly annealed), epsilon 1e-5 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Policy / value clip | 0.2 / 0.2 |
| Entropy / value coefficients | 0.01 / 0.5 |
| Requested horizon | 500,000 transitions (976 iterations, 499,712 actual) |

No observation or reward normalization. Termination and time-limit truncation both mask GAE bootstrapping, as in the source (SB3 bootstraps truncations). Evaluation uses argmax under `torch.no_grad()`, so the training action RNG is untouched.

## JAX PPO on the L4

[`jax_ppo.py`](../control_clock/jax_ppo.py) follows [PureJaxRL](https://github.com/luchris429/purejaxrl) (Apache-2.0, [notice](../third_party/PureJaxRL-LICENSE)): `vmap` over environments, nested `lax.scan` for rollouts, GAE and minibatch updates. The host loop is split at full updates so the shared evaluator can run.

128 environments × 128 steps = 16,384 transitions per update, 4 epochs, 4 minibatches, constant learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01, value 0.5, gradient-norm cap 0.5. No normalization or annealing. Actor weights go to the host after each update for the NumPy greedy policy; that transfer and the 100 evaluation episodes are on the clock. Training uses [Gymnax](https://github.com/RobertTLange/gymnax) 0.0.9, whose float32 dynamics and PRNG differ from Gymnasium's (see [environments.md](environments.md)). Pins: [`gpu-requirements.txt`](../gpu-requirements.txt), plus CUDA 12 wheels on Modal.

## Tests

[`tests/test_baselines.py`](../tests/test_baselines.py) checks the Zoo settings and schedules, CleanRL defaults, frozen SB3 evaluation normalization, same-step truncation resets and checkpoint ordering. [`tests/test_algorithms.py`](../tests/test_algorithms.py) checks the same ordering for batched PPO, ARS and CEM. [`tests/test_jax.py`](../tests/test_jax.py) checks GAE, an optimizer step, host/JAX action agreement and Gymnax dynamics against Gymnasium.

## References

- [Mania, Guy and Recht 2018](https://arxiv.org/abs/1803.07055), *Simple random search provides a competitive approach to reinforcement learning* (ARS). This version searches discrete-action affine policies with fixed feature scales, not running observation normalization.
- [Rubinstein 1999](https://doi.org/10.1023/A:1010091220143), the cross-entropy method (CEM).
- [Gymnasium](https://gymnasium.farama.org/), [EnvPool](https://arxiv.org/abs/2206.10558) and Gymnax appear in the [systems measurements](systems.md).
