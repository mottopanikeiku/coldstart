# Running the comparison

Design and evidence: [protocol](PROTOCOL.md), [baseline settings](BASELINES.md), [dynamics checks](ENVIRONMENTS.md), [experiment log](LOG.md), [independent review](COLD_REVIEW.md), and [one proposed next comparison](NEXT.md).

## Installation and checks

```sh
uv sync --frozen --python 3.12
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

The lockfile installs CPU Torch; no GPU is used. Box2D's isolated build gets SWIG from its pinned PyPI wheel through uv's extra build dependency. A C++ compiler is still required. The measured machine and compiler are in [results/machine.json](../results/machine.json). The pytest suite includes a small Box2D comparison and one update or generation of every CPU method to check that evaluation follows a complete update; it does not run benchmark cohorts. CI runs the same three check commands separately. The JAX checks are skipped in this environment; CI runs them in a second job with CPU JAX:

```sh
uv venv --python 3.12 .jax-venv
uv pip install --python .jax-venv/bin/python -r gpu-requirements.txt pytest==8.4.2
JAX_PLATFORMS=cpu .jax-venv/bin/python -m pytest -q tests/test_jax.py
```

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

## Extra comparison on a Modal L4

This comparison does not need a local GPU or the JAX/CUDA dependencies on
the client. I use Modal client 1.5.3, an account already authenticated with
`modal setup`, and the pinned image in `modal_app.py`. The final invocation
runs three fresh processes per task on one L4, requesting two host CPU cores
and 4 GiB RAM. It keeps the 120/300-second per-seed limits and writes a different
directory from the laptop measurements:

```sh
uvx --from modal==1.5.3 modal run modal_app.py --phase final --output results/gpu-replication
uv run python gpu_summary.py --input results/gpu-replication --output results/gpu-replication-summary.json
```

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
