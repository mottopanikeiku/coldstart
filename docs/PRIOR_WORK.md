# Prior work

## Algorithms

- [Mania, Guy and Recht (2018)](https://arxiv.org/abs/1803.07055), *Simple random search provides a competitive approach to reinforcement learning*. My discrete-action ARS searches affine policies, ranks symmetric perturbations by episodic return and scales the update by return standard deviation. It uses fixed physical feature scales rather than the paper's running observation normalization.
- [Rubinstein (1999)](https://doi.org/10.1023/A:1010091220143), *The Cross-Entropy Method for Combinatorial and Continuous Optimization*. CEM fits a Gaussian to the best affine-policy candidates. Both search methods use episode rewards without a value function.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) provide the library implementation and tuned task settings.
- [CleanRL discrete-action PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) provides the second CPU baseline. I adapt its training loop to the shared clock, seeds and evaluator.
- [Chris Lu's PureJaxRL](https://github.com/luchris429/purejaxrl) supplies the design for the JAX PPO comparison: vectorized environments and nested scans for rollouts and updates.

The [baseline notes](BASELINES.md) identify the source versions, settings and adaptations. License notices for adapted code are in [third_party](../third_party/).

## Environments

[Gymnasium](https://gymnasium.farama.org/) defines the reference dynamics, episode limits and evaluator. The NumPy and C batches implement its default CartPole and Acrobot tasks. [Environment notes](ENVIRONMENTS.md) describe their reset and numerical behavior.

[EnvPool](https://arxiv.org/abs/2206.10558), Weng et al. (2022), implements parallel environments in C++. [Gymnax](https://github.com/RobertTLange/gymnax) implements environments in JAX. Both appear in the [systems comparison](SYSTEMS.md), alongside Gymnasium's sequential wrapper and vectorized CartPole.

The comparison here starts the clock before Python imports and includes every 100-episode evaluation. That makes startup and evaluation costs visible alongside training throughput.
