# Running the comparison

Design and evidence: [protocol](PROTOCOL.md), [baseline settings](BASELINES.md), [dynamics checks](ENVIRONMENTS.md), and [experiment log](LOG.md).

## Installation and checks

```sh
uv sync --frozen --python 3.12 --extra box2d
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

The lockfile installs CPU Torch; no GPU is used. The base install is wheels only. The `box2d` extra adds Box2D for LunarLander and the Box2D equivalence tests; its isolated build gets SWIG from its pinned PyPI wheel through uv's extra build dependency, and still needs a C++ compiler. Without that extra, the Box2D tests are skipped and `run.py` refuses LunarLander before starting a worker. The measured machine and compiler are in [results/machine.json](../results/machine.json). The pytest suite includes a small Box2D comparison and one update or generation of every CPU method to check that evaluation follows a complete update; it does not run benchmark cohorts. CI runs the same three check commands with the `box2d` extra. A second CI job installs without it and with `--no-build`, so nothing is compiled, runs the suite, and replays the README's first run, failing unless seed 0 passes. The JAX checks are skipped unless the `jax` extra is installed; a third CI job runs them with CPU JAX:

```sh
uv sync --frozen --python 3.12 --extra jax
JAX_PLATFORMS=cpu uv run pytest -q tests/test_jax.py
```

The `jax` extra holds the pins in `gpu-requirements.txt`, which the Modal image installs before adding CUDA 12 wheels; `tests/test_gpu_results.py` keeps the two lists equal. The lock fixes transitive versions for the CPU checks only; the Modal image resolves its own when it is built.

```sh
uv run python -m control_clock.native build
uv run python -m control_clock.native verify --seeds 1000000
```

The fused C batch for the systems benchmark is optional and is not an extra: the first command compiles `control_clock/classic.c` with the system C compiler, and the second checks its one-call reset against Gymnasium's per-seed resets, bit for bit, on 1,010,006 seeds. Without the library, its tests are skipped, the equivalence tests run on the NumPy batch only, and `make_vector(task, n, native=True)` returns the NumPy batch. CI builds it in the `test` and `jax` jobs and runs the 1M-seed check; the first-run job leaves it out and checks the fallback. [Environment notes](ENVIRONMENTS.md#fused-c-batch) describe it.

## Development

Each entry in [configs/development.json](../configs/development.json) states task, method, seed range, variant and time limit. Chunks are bounded so they can share the machine:

```sh
nice -n 19 .venv/bin/python sweep.py --plan configs/development.json --chunk 0 --output results/development-replication
```

Development timing is affected by other heavy work. It is kept as experiment history, not mixed into the main result. `docs/LOG.md` reports effects and unsuccessful attempts.

## Final measurements

Final measurements use an otherwise idle CPU, a single-thread numeric-library environment, and nice 19. `configs/final.json` is the exact seed plan. Its five-seed Acrobot batches are bounded by the task's fixed 300-second per-seed budget. Dependency installation takes place before measurement. For a replication, stop other CPU-heavy work first.

```sh
nice -n 19 .venv/bin/python sweep.py --plan configs/final.json --chunk 0 --output results/single-chunk
```

`measure.py` drives all chunks sequentially. `--not-before` accepts a timezone-qualified ISO timestamp; waiting is outside the seed clocks. Use `--output` for a new replication directory: committed raw results are never overwritten. Batch command logs live in `results/batches/`, with a separate prefix for a replication.

```sh
.venv/bin/python measure.py --plan configs/final.json --output results/replication
```

Every worker starts in a new process. The launcher refuses to overwrite an existing seed result; select a new output directory for a replication rather than deleting original data. For one method/task independently:

```sh
nice -n 19 .venv/bin/python run.py --phase final --task CartPole-v1 --method cem --seeds 0:5 --output results/one-method
```

Every raw JSON contains the worker command, launcher arguments, software versions, thread settings, full evaluation-return arrays, elapsed clock values, stop reason and unsuccessful seeds. The absolute monotonic start timestamp in a command is a trace value, not a reusable CLI argument; `run.py` generates it anew.

Published commands use repository-relative paths and omit host-specific orchestration prefixes. Early development executable paths were normalized to their equivalent `.venv/bin/python`; no measurements were altered. Fresh processes do not imply cold disk caches: package files can remain in the OS page cache across seeds. CPU frequency and ordinary OS background activity are not pinned.

## Tables

```sh
uv run python summarize.py --input results/final --output results/summary.json
```

The script prints Markdown and saves machine-readable statistics including the files that contributed to every row. Its median and interquartile range are conditional on success. The Kaplan–Meier output uses actual stop times, including training-step horizon stops, as censoring; this is descriptive because early horizon exhaustion need not be independent of eventual solve time. A method that never passes has no median solve time, not a zero or its timeout value.

Each cohort also contains a seed-sorted `records` list with `seed`, `solved`, `seconds` and `stop_reason`. `seconds` is time to first pass for a success and actual stop time for a failure (parent-observed elapsed time for a watchdog kill). It is never a fabricated solve time. The existing KM calculation caps censoring at the protocol wall limit; the per-seed plotting records retain actual stop times, including small deadline overshoots.

## Separate implementation and hardware comparison on a Modal L4

This comparison is paid. It does not need a local GPU or the JAX/CUDA dependencies on
the client. I use Modal client 1.5.3, an account already authenticated with
`modal setup`, and the pinned image in `modal_app.py`. The final invocation
runs three fresh processes per task on one L4, requesting two host CPU cores
and 4 GiB RAM. It keeps the 120/300-second per-seed limits and writes a different
directory from the laptop measurements:

```sh
uv sync --frozen --python 3.12 --extra modal
.venv/bin/modal run modal_app.py --phase final --output results/gpu-replication
uv run python gpu_summary.py --input results/gpu-replication --output results/gpu-replication-summary.json
```

The published runs launched the same client version through `uvx --from modal==1.5.3`
under the app name `control-clock-gpu`; the `modal` extra also locks the client's
dependencies, and the app is now named `coldstart-gpu`. Neither change alters the packages
the image installs or what the clock measures.

Use `--phase pilot` for the separate seed-100 development check with a
90-second limit. Select a new output directory each time; published records
are not overwritten. `gpu_run.py` can also run a single final-budget seed
from an environment with CUDA JAX and the pinned GPU dependencies installed.
It refuses to report a CPU run as a GPU comparison.

The clock begins at Python subprocess launch on the cloud host. JIT
compilation, device initialization and every full Gymnasium evaluation are
inside that clock. Image construction, Modal client preparation and container
provisioning are not; I state those separately from qualification times.
Driver and OS caches are not cleared, and correctness checks run on the GPU
before the seed processes. This is a fresh-process comparison, not a
power-cycled GPU or an estimate of an already compiled training service.

Cloud client preparation, image work, provisioning and shutdown are outside
each policy's process clock. Their combined overhead was about 18 seconds
for the final batch and 218 seconds for the pilot, not an extra charge added
to every seed. My total conservative cloud-cost estimate is $0.3745,
including two failed setup attempts; it is not an invoice
([cost and overhead](../results/gpu/cost.json)).
