# coldstart

A stopwatch for small RL runs. It starts before the Python interpreter does and stops at the first policy that passes a fixed-seed evaluation on CartPole or Acrobot, so imports, setup, training and every evaluation count.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figure-dark.svg"><img alt="Seconds from Python launch to a passing policy, one dot per seed. Random search wins CartPole at about 1.1 seconds; SB3 PPO wins Acrobot at 10 seconds." src="docs/figure-light.svg" width="560"></picture>

On my Ryzen AI 5 laptop, random search over linear policies passes the CartPole evaluation about a second after launch. The three PPO implementations take four to six. On Acrobot random search comes last and Stable-Baselines3's tuned PPO is fastest.

Median seconds to the first evaluation that averages at least 475 over 100 CartPole episodes (−100 on Acrobot). Those 100 episodes use fixed seeds, which I also used while choosing the settings. The last column repeats CartPole, with fewer seeds, on a shared 16-core Arm VM under heavy load.

| Method | CartPole | Acrobot | CartPole, Arm VM |
| --- | ---: | ---: | ---: |
| ARS (random search over linear policies) | **1.08** | 28.1 | 1.7 |
| CEM (cross-entropy method) | 1.15 | 15.3 | 2.1 |
| Batched CPU PPO (mine) | 4.15 | 25.0 | 8.6 |
| Stable-Baselines3 PPO, Zoo settings | 4.72 | **10.0** | 10.6 |
| CleanRL PPO | 5.55 | 15.2 | 17.0 |

All 140 laptop runs passed: 20 seeds each for ARS, CEM and my PPO, 5 for the library PPOs. On 1,000 fresh episodes, 26 of 30 saved policies (my three methods, five seeds each, both tasks) still passed.

## Run one

```sh
git clone https://github.com/mottopanikeiku/coldstart && cd coldstart
uv sync --frozen --python 3.12
.venv/bin/python run.py --phase final --task CartPole-v1 --method ars --seeds 0:1 --output results/first-run
```

It prints `"solved": true` and the seconds, and saves every evaluation return to JSON. Everything runs on a CPU from locked wheels. The full laptop comparison is `nice -n 19 .venv/bin/python measure.py --plan configs/final.json --output results/replication`.

## Where the time goes

Mostly evaluation. From profiling five seeds each of ARS, CEM and my PPO:

- Gymnasium evaluation takes 64–81% of a run for ARS and CEM on both tasks, and for PPO on Acrobot.
- Simulating the training environments takes 22–27% of an ARS or CEM run, so even free simulation would leave most of the wait.
- CartPole PPO is the exception, with about half its time in optimizer updates.
- An optional C batch of CartPole steps 8–13× faster than my NumPy one and matches Gymnasium's resets exactly. The timed runs above use NumPy.
- A JAX PPO on an L4 GPU passed all six of its runs, but slower: 26 s on CartPole and 97 s on Acrobot.

The profiles, import costs, C kernels and the fresh-seed replays are in [docs/systems.md](docs/systems.md). The timing protocol, baseline settings and environment notes are in the same folder.

Built on [Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), [CleanRL](https://github.com/vwxyzjn/cleanrl), [Gymnax](https://github.com/RobertTLange/gymnax) and [PureJaxRL](https://github.com/luchris429/purejaxrl). ARS is [Mania, Guy and Recht](https://arxiv.org/abs/1803.07055). MIT license.
