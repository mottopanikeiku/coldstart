# Prior work

Reviewed before implementation on 2026-10-06.

- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) is the library comparator. Its examples warn that demonstration hyperparameters need not solve environments. [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) supplies task-specific tuned configurations and normalizers. This repository measures those configurations rather than claiming a new PPO algorithm.
- [CleanRL discrete-action PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), [documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/), implements the algorithm in one file and makes implementation choices visible. We retain its default learning settings in the baseline and instrument evaluation/time limits. Any reused source is attributed and its MIT license retained.
- [Mania, Guy and Recht (2018)](https://arxiv.org/abs/1803.07055), *Simple random search provides a competitive approach to reinforcement learning*, establishes linear policy search as a necessary comparator, not an incidental toy. Our discrete-action ARS variant ranks symmetric perturbations by episodic returns, uses their standard-deviation-scaled update and a bias term. This is not a reproduction of its continuous-control results.
- [Cross-entropy method](https://doi.org/10.1023/A:1010091220143), Rubinstein (1999), fits a search distribution to elite candidates. Here it searches affine policies with episodic rewards. Neither ARS nor CEM uses gradients through the environment or a value function.
- [Gymnasium classic-control](https://gymnasium.farama.org/environments/classic_control/) provides reference task dynamics and thresholds. [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) depends on Box2D and is a stretch measurement, not silently substituted for Acrobot.

What this adds: a small laptop-only comparison with a clock that begins before Python imports, a common 100-episode deterministic evaluation against Gymnasium, raw per-seed trajectories, explicit unsuccessful seeds, and checked vectorized dynamics. It does not establish a general-purpose RL speed record; CartPole can be solved by very simple policy search, and shared evaluation seeds can be adapted to by repeated checkpoint selection.

## Related execution work checked during integration

[EnvPool](https://arxiv.org/abs/2206.10558), Weng et al. (2022), already demonstrates highly parallel environment execution and its integration with RL libraries, including laptop runs. Batching environments is not a novel idea here. This repository instead keeps a small inspectable NumPy implementation for two tasks and tests it against Gymnasium, then asks about startup-to-threshold time with a large evaluation set. EnvPool itself is not measured in this small comparison; no general speed claim against it is made.
