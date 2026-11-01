# Development experiments

These trials chose the configurations in [configs/final.json](../configs/final.json). They ran on a shared CPU, so their times are separate from the main laptop comparison. Raw runs are in [results/development](../results/development/); the original timing protocol is at revision `9a8486a`.

## Starting configurations

My first CPU PPO used a single 32-unit hidden layer on CartPole and separate 64-by-64 actor and critic networks on Acrobot. Rollout sizes were 512 and 8,192 transitions, with optimizer minibatches of 512 and 1,024. It bootstraps time-limit truncations but ends GAE at each episode boundary. Acrobot uses fixed physical observation scales.

ARS perturbs affine policies in symmetric pairs and scales its update by the return standard deviation. CEM fits a Gaussian to the best candidates. Both use four common-seed training episodes per candidate.

The first trial used seed 0 and a 60-second limit:

| Task | Method | First pass, seconds | Training outcome |
|---|---|---:|---|
| CartPole | CPU PPO | — | Stopped at 100,352 steps; best mean 161.8 |
| CartPole | ARS | 1.93 | Passed at 142,461 steps |
| CartPole | CEM | 1.57 | Passed at 115,840 steps |
| CartPole | SB3 Zoo | 6.16 | Passed at 20,480 steps |
| CartPole | CleanRL | 4.98 | Passed at 30,720 steps |
| Acrobot | CPU PPO | 13.57 | Passed at 131,072 steps |
| Acrobot | ARS | — | Stopped at 1,001,159 steps; last mean −166.3 |
| Acrobot | CEM | — | Stopped at 1,008,140 steps; last mean −123.0 |
| Acrobot | SB3 Zoo | 13.39 | Passed at 36,864 steps |
| Acrobot | CleanRL | 20.21 | Passed at 61,440 steps |

The small CartPole PPO failed while both search methods passed in under two seconds. Acrobot's searches exhausted their one-million-transition horizon.

## Dynamics, batch size and network width

I then tried these changes to CPU PPO on seeds 0 and 1, with the same 60-second limit. Times are medians among passing runs.

| Task | Change | Passed | Median seconds |
|---|---|---:|---:|
| CartPole | Gymnasium dynamics | 1/2 | 5.69 |
| CartPole | Optimizer minibatch 512 → 256 | 2/2 | 8.16 |
| CartPole | Network 32-by-1 → 128-by-2 | 2/2 | 7.65 |
| Acrobot | Gymnasium dynamics | 2/2 | 39.57 |
| Acrobot | Environments 64 → 16; minibatch 1,024 → 256 | 2/2 | 14.38 |
| Acrobot | Network 64-by-2 → 128-by-2 | 2/2 | 26.85 |

On Acrobot seed 0, NumPy PPO passed at 131,072 transitions and the Gymnasium version at 212,992. Small numerical differences changed the training trajectory as well as the runtime. Wider networks did not improve the Acrobot trial.

## Longer searches and Zoo-shaped PPO

The next CartPole PPO used eight environments, 64-by-64 networks, 20 optimizer epochs, minibatches of 256 and Zoo's linear learning-rate and clip schedules. It passed all three trial seeds: median 7.23 seconds, with quartiles 7.03 and 8.74.

Extending Acrobot search to four million transitions let both ARS and CEM pass seeds 0 and 1. Their medians were 36.87 and 25.20 seconds. The earlier failures came from stopping the search too soon.

The final choices were:

| Task | Batched PPO | ARS and CEM |
|---|---|---|
| CartPole | `zoo-shape`: eight environments, Zoo-sized networks and schedules | `default`: one million transitions |
| Acrobot | `default`: 64 environments, 64-by-64 networks, minibatches of 1,024 | `longer`: four million transitions |

The library PPO settings stayed unchanged. Full generations and rollouts can overshoot a requested transition horizon.

## LunarLander

Only seed 0 was tried for each method, with a 120-second development limit. No final LunarLander comparison was run.

| Method | Outcome | Best 100-episode mean | Transitions at stop |
|---|---|---:|---:|
| CPU PPO | Time limit | 137.78 | 466,944 |
| ARS | One-million-step horizon | 196.67 | 1,016,526 |
| CEM | One-million-step horizon | 157.97 | 1,032,366 |
| SB3 Zoo | Passed in 118.64 seconds | 202.16 | 475,136 |
| CleanRL | Time limit | −137.06 | 307,200 |

The settings and commands for these trials are in [configs/development.json](../configs/development.json) and [configs/tuning.json](../configs/tuning.json). [Run instructions](RUNNING.md) show how to send new trials to a separate output directory.

