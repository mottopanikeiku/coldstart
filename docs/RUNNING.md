# Running coldstart

The [README](../README.md#try-it) starts with one CartPole seed. For the measurement rules and settings, see the [protocol](PROTOCOL.md), [baselines](BASELINES.md), [environment checks](ENVIRONMENTS.md) and [development experiments](DEVELOPMENT.md).

## Install and test

```sh
uv sync --frozen --python 3.12
uv run ruff check .
uv run ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q
```

The lockfile installs CPU Torch. The base install uses wheels only. Tests check the algorithms, evaluation timing, environment dynamics and result summaries without running full benchmark groups.

Optional dependencies:

| Extra | Adds | Requirement |
|---|---|---|
| `box2d` | LunarLander and its dynamics tests | C++ compiler; SWIG comes from a pinned wheel |
| `jax` | CPU JAX and Gymnax tests | No GPU needed |
| `bench` | EnvPool systems comparison | System OpenGL libraries |
| `modal` | Client for the paid L4 comparison | Authenticated Modal account |

Use `uv sync --frozen --python 3.12 --extra box2d`, for example. Repeat `--extra` to combine extras. Tests for missing extras are skipped; `run.py` rejects LunarLander without Box2D. To run the JAX tests on CPU:

```sh
uv sync --frozen --python 3.12 --extra jax
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q tests/test_jax.py
```

CI tests the base install and README command, Box2D with the C batch, and CPU JAX with EnvPool. EnvPool's required system libraries are listed in [systems instructions](SYSTEMS.md#reproduce).

## Optional C batch

```sh
uv run python -m control_clock.native build
uv run python -m control_clock.native verify --seeds 1000000
```

The build uses the system C compiler and libm. The second command compares resets with Gymnasium on 1,010,006 seeds per task, including large and boundary seeds. CI runs it too. The ordinary tests use 3,306 seeds per task.

Without the compiled library, `make_vector(task, n, native=True)` uses the NumPy batch. The benchmark's `native` backend requires the library. Qualification runs use NumPy regardless; C is currently used by the separate systems benchmark. [Environment notes](ENVIRONMENTS.md#fused-c-batch) explain its reset and step implementation.

## Repeat the laptop comparison

Use an otherwise idle CPU. The original run used one numeric-library thread and nice 19; the launcher sets the thread environment for each worker. [results/machine.json](../results/machine.json) describes the machine.

```sh
nice -n 19 .venv/bin/python measure.py --plan configs/final.json --output results/replication
```

`measure.py` runs the plan's chunks sequentially. `--not-before` accepts a timezone-qualified ISO timestamp; any wait happens before the seed clocks start. Each chunk is bounded to less than 30 minutes at the task deadlines.

To run one chunk or one method instead:

```sh
nice -n 19 .venv/bin/python sweep.py --plan configs/final.json --chunk 0 --output results/single-chunk
nice -n 19 .venv/bin/python run.py --phase final --task CartPole-v1 --method cem --seeds 0:5 --output results/one-method
```

Every seed starts a new Python process. Existing seed results are never overwritten, so choose a new output directory for each replication. Batch commands are saved under `results/batches/` with a separate prefix for each output directory. The absolute monotonic timestamp in a worker command is generated anew by the launcher.

Development trials use the same tools with `configs/development.json` or `configs/tuning.json`:

```sh
nice -n 19 .venv/bin/python sweep.py --plan configs/development.json --chunk 0 --output results/development-replication
```

## Summarize results

```sh
uv run python summarize.py --input results/replication --output results/replication-summary.json
```

The script prints a Markdown table and saves the statistics, contributing file paths and per-seed outcomes. Median and quartiles use successful runs; the JSON also includes a Kaplan–Meier estimate. [Summary definitions](PROTOCOL.md#summary-statistics) explain failures and censoring.

## JAX PPO on an L4

This run bills a Modal account. It needs `modal setup` but no local GPU. The final command launches three new Python processes per task on one L4, requesting two host CPU cores and 4 GiB RAM:

```sh
uv sync --frozen --python 3.12 --extra modal
.venv/bin/modal run modal_app.py --phase final --output results/gpu-replication
uv run python gpu_summary.py --input results/gpu-replication --output results/gpu-replication-summary.json
```

The Modal app is `coldstart-gpu`. Use `--phase pilot` for seed 100 with a 90-second limit. `gpu_run.py` also runs a single seed from an environment with the pinned CUDA JAX dependencies installed, and rejects CPU-only execution.

The final runs passed 3/3 seeds on each task: 26.38 seconds median on CartPole and 97.43 on Acrobot ([summary](../results/gpu/summary.json)). JIT compilation, device initialization and Gymnasium evaluation are inside each process clock. Image construction, client preparation and container provisioning are outside it.

Those outside-clock costs totaled about 18 seconds for the final batch and 218 seconds for the pilot. The estimated cloud cost, including two failed setup attempts, was $0.3745 ([cost calculation](../results/gpu/cost.json)). The image uses `gpu-requirements.txt` with CUDA 12 wheels; the CPU lockfile does not fix the image's transitive dependencies.
