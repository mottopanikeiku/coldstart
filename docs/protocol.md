# Protocol and rerun instructions

The clock runs from Python launch to a policy's first passing evaluation. PPO, ARS and CEM share the clock and evaluator in [`control_clock/protocol.py`](../control_clock/protocol.py).

## Passing

Each evaluation runs 100 full Gymnasium episodes with greedy actions and averages the raw returns. Ties go to the lowest action index.

| Task | Passing mean | Episode cap | Time limit per seed |
| --- | ---: | ---: | ---: |
| CartPole-v1 | ≥ 475 | 500 steps | 120 s |
| Acrobot-v1 | ≥ −100 | 500 steps | 300 s |

Evaluation seeds are 1,000,000 to 1,000,099 and are reused at every checkpoint and for every method. Training and reset seeds stay below 1,000,000. LunarLander has code and a few development trials but no final comparison.

## The clock

The parent starts a monotonic timer right before spawning a fresh single-seed Python process and passes the timestamp to the worker. The interval covers interpreter startup, imports, environment and model construction, training, and every evaluation up to the first pass. Installing dependencies, saving the policy, writing the result file and process teardown are outside it.

Evaluation runs before training, then after the first complete update past each further 10,000 training transitions (a search generation counts as an update). Evaluation never touches parameters or training statistics, and always steps real Gymnasium environments. An evaluation that finishes after the deadline doesn't count. Workers that overrun by five seconds are killed.

A random initial policy can pass: batched PPO did on CartPole seed 4 and Acrobot seed 19. Nothing starts from pretrained weights.

## What ran

[`configs/final.json`](../configs/final.json) is the run list:

| Method | Seeds per task | Setup |
| --- | ---: | --- |
| ARS | 20 | Affine policy, 1M (CartPole) or 4M (Acrobot) transition horizon |
| CEM | 20 | Same policy and horizons |
| Batched CPU PPO | 20 | Zoo-shaped on CartPole, original batch and network on Acrobot |
| SB3 PPO | 5 | RL Baselines3 Zoo settings |
| CleanRL PPO | 5 | 4 environments, 500k requested transitions |

ARS perturbs an affine policy in symmetric pairs and scales its step by the return standard deviation. CEM fits a Gaussian to the best candidates. Both score each candidate on four common-seed training episodes. Details of the PPO baselines are in [baselines.md](baselines.md).

The settings came from short trials on a shared CPU, using seeds 0 to 2. The first CartPole PPO (one 32-unit layer) failed while ARS and CEM passed in under two seconds, so I rebuilt it with Zoo-like settings (8 environments, 64×64 networks, 20 epochs, linear schedules). On Acrobot, 1M-transition searches ran out of horizon; 4M fixed that. The evaluation seeds were used for those choices too, which is why [the saved policies](systems.md#saved-policies-on-new-seeds) were re-scored on fresh seeds.

Laptop runs used otherwise idle CPU windows, one seed at a time, at nice 19, with one thread each for Torch, OpenMP, OpenBLAS and MKL. [`results/machine.json`](../results/machine.json) describes the laptop. Each result file holds the settings, package versions, worker command, seeds and every evaluation return.

## Statistics

[`summarize.py`](../summarize.py) reports the pass fraction and the median and quartiles of solve time among passing runs. The JSON adds a Kaplan–Meier median that censors failures at their stop time. All 140 laptop runs passed.

## Install and test

```sh
uv sync --frozen --python 3.12
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q
```

The lockfile pins CPU Torch and installs wheels only. Optional extras: `box2d` (LunarLander, needs a C++ compiler), `jax` (CPU JAX and Gymnax tests), `bench` (EnvPool, needs system OpenGL libraries), `modal` (client for the paid GPU run). Add them with `uv sync --frozen --python 3.12 --extra jax`. Tests for missing extras skip.

## Repeat the laptop comparison

```sh
nice -n 19 .venv/bin/python measure.py --plan configs/final.json --output results/replication
nice -n 19 .venv/bin/python run.py --phase final --task CartPole-v1 --method cem --seeds 0:5 --output results/one-method
uv run python summarize.py --input results/replication --output results/replication-summary.json
```

Use an idle CPU. Every seed starts a new process, existing results are never overwritten, and `measure.py` waits for `--not-before` (an ISO timestamp) before any clock starts.

## JAX PPO on a GPU

`modal_app.py` runs JAX PPO on one Modal L4: seeds 0 to 2 per task, same thresholds, evaluator and time limits. It is a paid run and needs `modal setup`.

```sh
uv sync --frozen --python 3.12 --extra modal
.venv/bin/modal run modal_app.py --phase final --output results/gpu-replication
uv run python gpu_summary.py --input results/gpu-replication --output results/gpu-replication-summary.json
```

Imports, CUDA initialization, JIT compilation, weight transfer and CPU evaluation all count; provisioning and image build don't. All six runs passed: median 26.4 s on CartPole and 97.4 s on Acrobot ([summary](../results/gpu/summary.json)). Cloud cost was about $0.37 including two failed setup attempts ([breakdown](../results/gpu/cost.json)).
