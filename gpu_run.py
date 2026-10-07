"""Launch each JAX seed in a fresh process; never measure the notebook/client."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

LIMITS = {"CartPole-v1": 120, "Acrobot-v1": 300}


def worker(args):
    import jax

    from control_clock.jax_ppo import train
    from control_clock.protocol import Finished, RunContext

    if jax.default_backend() != "gpu":
        raise RuntimeError("This comparison requires a GPU; refusing a CPU timing")
    context = RunContext(args.task, args.seed, args.started, args.limit)
    reason = "error"
    try:
        train(context)
    except Finished as error:
        reason = str(error)
    finally:
        stopped = time.perf_counter()
        context.close()
    print(
        json.dumps(
            {
                "solved": context.solved,
                "solve_seconds": context.solve_seconds,
                "stop_seconds": stopped - args.started,
                "evaluation_policy": "deterministic argmax, lowest-index ties",
                "evaluation_seeds": [1_000_000, 1_000_099],
                "evaluation_episodes": 100,
                "evaluations": context.evaluations,
                "steps": context.steps,
                "configuration": context.configuration,
                "stop_reason": reason,
                "devices": [str(device) for device in jax.devices()],
                "device_kind": [device.device_kind for device in jax.devices()],
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("jax", "jaxlib", "flax", "optax", "gymnax", "gymnasium", "numpy")
                },
            }
        )
    )


def launch(task, seed, output, phase, limit=None):
    destination = Path(output) / f"{task}__jax-ppo__{seed:03}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    limit = LIMITS[task] if limit is None else limit
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    started = time.perf_counter()
    command = [
        sys.executable,
        "gpu_run.py",
        "--worker",
        "--task",
        task,
        "--seed",
        str(seed),
        "--started",
        repr(started),
        "--limit",
        str(limit),
    ]
    watchdog = False
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=limit + 5)
        except subprocess.TimeoutExpired:
            watchdog = True
            process.kill()
            stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    try:
        record = json.loads(stdout)
    except json.JSONDecodeError:
        record = {
            "solved": False,
            "solve_seconds": None,
            "evaluations": [],
            "stop_reason": "watchdog" if watchdog else "worker error",
        }
    record.update(
        task=task,
        seed=seed,
        method="jax-ppo",
        phase=phase,
        limit_seconds=limit,
        parent_seconds=elapsed,
        returncode=process.returncode,
        watchdog=watchdog,
        stderr=stderr,
        machine={"platform": platform.platform(), "python": platform.python_version()},
        hardware="Modal NVIDIA L4, 2 CPU cores, 4 GiB host RAM",
        thread_environment={
            key: environment[key]
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        # The monotonic start value and container interpreter path are not portable.
        command=["python", "gpu_run.py", "--worker", "--task", task, "--seed", str(seed)],
    )
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{task} seed {seed}: {record['stop_reason']} at {record['solve_seconds']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", choices=LIMITS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--started", type=float)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--output", type=Path, default=Path("results/gpu-replication"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.limit is not None:
            parser.error("use the fixed task budgets for standalone replication")
        launch(args.task, args.seed, args.output, "replication")


if __name__ == "__main__":
    main()
