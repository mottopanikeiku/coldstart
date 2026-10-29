"""Systems benchmark: environment speed, imports, evaluation and a phase decomposition.

    uv sync --frozen --python 3.12 --extra jax --extra bench
    .venv/bin/python -m control_clock.native build
    .venv/bin/python bench.py all --output results/bench --cpu 4 --reps 5
    .venv/bin/python bench_summary.py --input results/bench --output results/bench/summary.json

Every measurement runs in a fresh process pinned to one CPU with one numeric thread. Each
step writes raw per-call or per-run values; ``summarize`` derives every table from them.
JAX runs on the CPU here; ``--jax-platform`` passes another platform through unchanged.
``--backends`` limits the environment step to some backends. Every command records the
machine and load snapshots in machine.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from control_clock import native
from control_clock.backends import BACKENDS, supports
from run import source_revision

TASKS = ("CartPole-v1", "Acrobot-v1")
SIZES = (1, 8, 64, 512, 4096)
IMPORTS = (
    "pass",
    "numpy",
    "gymnasium",
    "control_clock.search",
    "torch",
    "control_clock.ppo",
    "stable_baselines3",
    "jax",
    "gymnax",
    "envpool",
)
REPO_METHODS = ("ars", "cem", "ppo-cpu")
LIMITS = {"CartPole-v1": 120, "Acrobot-v1": 300}
PACKAGES = (
    "numpy",
    "gymnasium",
    "torch",
    "stable-baselines3",
    "jax",
    "jaxlib",
    "gymnax",
    "envpool",
)


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot(cpu: int) -> dict:
    """Load and frequency now; the CPU this benchmark pins to and its SMT sibling."""
    base = f"/sys/devices/system/cpu/cpu{cpu}"
    return {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "load_average": list(os.getloadavg()),
        "pinned_cpu_mhz": int(read(f"{base}/cpufreq/scaling_cur_freq") or 0) / 1000,
        "memory_available_mib": next(
            int(line.split()[1]) // 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable")
        ),
    }


def machine(cpu: int) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    base = f"/sys/devices/system/cpu/cpu{cpu}"
    model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "cpu_model": model,
        "logical_cpus": os.cpu_count(),
        "pinned_cpu": cpu,
        "pinned_cpu_siblings": read(f"{base}/topology/thread_siblings_list"),
        "pinned_cpu_max_mhz": int(read(f"{base}/cpufreq/cpuinfo_max_freq") or 0) / 1000,
        "max_mhz_by_cpu": {
            index: int(read(f"/sys/devices/system/cpu/cpu{index}/cpufreq/cpuinfo_max_freq") or 0)
            / 1000
            for index in range(os.cpu_count())
        },
        "governor": read(f"{base}/cpufreq/scaling_governor"),
        "energy_performance_preference": read(f"{base}/cpufreq/energy_performance_preference"),
        "platform_profile": read("/sys/firmware/acpi/platform_profile"),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "packages": {name: installed(name) for name in PACKAGES},
        "numpy_simd": np.show_config(mode="dicts")["SIMD Extensions"],
        "threads": "OMP/MKL/OPENBLAS/NUMEXPR = 1; XLA single-threaded Eigen, 1 intra-op thread",
        "native": native.describe(),
        "source_commit": commit.stdout.strip() or None,
    }


def child_environment(platform_name: str) -> dict:
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JAX_PLATFORMS=platform_name,
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        PYTHONWARNINGS="ignore",
    )
    return environment


def launch(command: list[str], args, *, clocked=False, timeout=None) -> tuple[dict, dict]:
    """Run one pinned child; with clocked, append --started like run.py does."""
    context = snapshot(args.cpu)
    started = time.perf_counter()
    if clocked:
        command = [*command, "--started", repr(started)]
    process = subprocess.run(
        [os.path.relpath(sys.executable), *command],
        capture_output=True,
        text=True,
        env=child_environment(args.jax_platform),
        preexec_fn=lambda: os.sched_setaffinity(0, {args.cpu}),
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(f"{command} failed:\n{process.stderr}")
    context.update(parent_seconds=elapsed, command=command)
    return json.loads(process.stdout.splitlines()[-1]), context


def env(args) -> None:
    output = args.output / "env"
    output.mkdir(parents=True, exist_ok=True)
    order = random.Random(args.seed)
    for rep in range(args.reps):
        pairs = [(b, t) for t in TASKS for b in args.backends if supports(b, t)]
        order.shuffle(pairs)
        for backend, task in pairs:
            destination = output / f"{backend}__{task}__{rep}.json.gz"
            if destination.exists():
                continue
            sizes = list(args.sizes)
            order.shuffle(sizes)
            command = ["-m", "control_clock.systems", "env", "--backend", backend]
            command += ["--task", task, "--sizes", ",".join(map(str, sizes)), "--seed", str(rep)]
            result, context = launch(command, args)
            result.update(rep=rep, context=context)
            destination.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
            print(destination, "gate passed:", result["gate"]["passed"], flush=True)


def imports(args) -> None:
    destination = args.output / "imports.json"
    if destination.exists():
        return
    order = random.Random(args.seed)
    rows = []
    for rep in range(args.import_reps):
        modules = list(IMPORTS)
        order.shuffle(modules)
        for module in modules:
            command = ["-m", "control_clock.systems", "import", "--module", module]
            result, context = launch(command, args, clocked=True)
            result.update(rep=rep, context=context)
            rows.append(result)
    destination.write_text(json.dumps(rows, indent=1) + "\n")
    print(destination, len(rows), "processes")


def cohorts() -> list[tuple[str, str, str]]:
    plan = json.loads(Path("configs/final.json").read_text())
    entries = {
        (entry["task"], entry["method"], entry.get("variant", "default"))
        for chunk in plan["chunks"]
        for entry in chunk
        if entry["method"] in REPO_METHODS
    }
    return sorted(entries)


def decompose(args) -> None:
    output = args.output / "decompose"
    output.mkdir(parents=True, exist_ok=True)
    first, last = map(int, args.seeds.split(":"))
    # Recorded before any clock starts, as run.py does; saved policies cite it.
    source = source_revision()
    for task, method, variant in cohorts():
        for seed in range(first, last):
            # Alternate which variant goes first so slow drift cannot favour one of them.
            kinds = ("plain", "profiled") if seed % 2 == 0 else ("profiled", "plain")
            for kind in kinds:
                destination = output / f"{task}__{method}__{variant}__{seed:03}__{kind}.json"
                if destination.exists():
                    continue
                module = "control_clock.worker" if kind == "plain" else "control_clock.phases"
                command = ["-m", module, "--task", task, "--method", method, "--seed", str(seed)]
                command += ["--limit", str(LIMITS[task]), "--variant", variant]
                record, context = launch(command, args, clocked=True, timeout=LIMITS[task] + 30)
                record.update(task=task, method=method, variant=variant, seed=seed, kind=kind)
                record.update(context=context, source=source)
                destination.write_text(json.dumps(record) + "\n")
                print(destination, round(record["solve_seconds"] or -1, 3), flush=True)


def evaluation(args) -> None:
    """Time both evaluators on the seed-0 policies that the decomposition's plain runs saved."""
    destination = args.output / "evaluation.json"
    rows = []
    for task in TASKS:
        policies = [
            str(path)
            for path in sorted((args.output / "decompose").glob(f"{task}__*__000__plain.json"))
            if json.loads(path.read_text()).get("policy")
        ]
        command = ["-m", "control_clock.systems", "evaluation", "--task", task]
        command += ["--policies", ",".join(policies), "--repeats", str(args.repeats)]
        result, context = launch(command, args)
        result.update(context=context, policy_files=policies)
        rows.append(result)
    destination.write_text(json.dumps(rows, indent=1) + "\n")
    print(destination)


def run(args, steps) -> None:
    """Resumable: finished files are kept, and machine.json keeps its earlier snapshots."""
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "machine.json"
    if path.exists():
        record = json.loads(path.read_text())
        record["snapshots"]["resumed"] = snapshot(args.cpu)
    else:
        record = {"machine": machine(args.cpu), "snapshots": {"before": snapshot(args.cpu)}}
    for name in steps:
        STEPS[name](args)
        record["snapshots"][f"after {name}"] = snapshot(args.cpu)
        path.write_text(json.dumps(record, indent=2) + "\n")


STEPS = {"env": env, "imports": imports, "decompose": decompose, "evaluation": evaluation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", *STEPS))
    parser.add_argument("--output", type=Path, default=Path("results/bench"))
    parser.add_argument("--cpu", type=int, default=4, help="CPU to pin every child process to")
    parser.add_argument("--reps", type=int, default=5, help="Fresh processes per backend")
    parser.add_argument("--backends", type=lambda s: tuple(s.split(",")), default=BACKENDS)
    parser.add_argument("--import-reps", type=int, default=10)
    parser.add_argument("--sizes", type=lambda s: tuple(map(int, s.split(","))), default=SIZES)
    parser.add_argument("--seeds", default="0:5", help="Decomposition seeds, half-open")
    parser.add_argument("--repeats", type=int, default=5, help="Evaluations per policy")
    parser.add_argument("--seed", type=int, default=2026, help="Seeds the run order")
    parser.add_argument("--jax-platform", default="cpu")
    args = parser.parse_args()
    unknown = set(args.backends) - set(BACKENDS)
    if unknown:
        parser.error(f"unknown backends: {sorted(unknown)}")
    run(args, list(STEPS) if args.command == "all" else [args.command])


if __name__ == "__main__":
    main()
