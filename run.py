"""Spawn fresh single-seed processes and retain successes, timeouts and failures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

LIMITS = {"CartPole-v1": 120, "Acrobot-v1": 300, "LunarLander-v3": 300}
METHODS = ("ppo-cpu", "sb3-zoo", "cleanrl", "ars", "cem")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=LIMITS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seeds", default="0:20", help="Half-open seed range, e.g. 0:5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument(
        "--limit", type=float, help="Development only; final uses fixed task budget"
    )
    parser.add_argument("--variant", default="default", help="Explicit development configuration")
    args = parser.parse_args()
    if args.limit is not None and args.phase == "final":
        parser.error("final runs use the fixed protocol limit")
    first, last = map(int, args.seeds.split(":"))
    if first < 0 or last <= first or last > 10_000:
        parser.error("seed range must satisfy 0 <= first < last <= 10000")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"
    )
    for seed in range(first, last):
        destination = args.output / f"{args.task}__{args.method}__{args.variant}__{seed:03}.json"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite a seed result: {destination}")
        limit = args.limit if args.limit is not None else LIMITS[args.task]
        command = [
            os.path.relpath(sys.executable),
            "-m",
            "control_clock.worker",
            "--task",
            args.task,
            "--method",
            args.method,
            "--seed",
            str(seed),
            "--limit",
            str(limit),
            "--variant",
            args.variant,
        ]
        started = time.perf_counter()
        command += ["--started", repr(started)]
        timed_out = False
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=limit + 5)
            except subprocess.TimeoutExpired:
                timed_out = True
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
                "steps": None,
                "configuration": {},
                "stop_reason": "watchdog" if timed_out else "worker error",
            }
        record.update(
            task=args.task,
            method=args.method,
            variant=args.variant,
            seed=seed,
            phase=args.phase,
            limit_seconds=limit,
            parent_seconds=elapsed,
            returncode=process.returncode,
            watchdog=timed_out,
            stderr=stderr,
            command=command,
            launch_command=sys.argv,
            machine={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "processor": platform.processor(),
            },
            thread_environment={
                key: environment[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        )
        if record.get("solve_seconds") is not None and record["solve_seconds"] > limit:
            record.update(solved=False, solve_seconds=None, stop_reason="late evaluation")
        destination.write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "result": str(destination),
                    "solved": record["solved"],
                    "seconds": record["solve_seconds"],
                    "stop": record["stop_reason"],
                }
            ),
            flush=True,
        )
        if process.returncode != 0 and not timed_out:
            raise RuntimeError(f"worker failed; retained error record {destination}: {stderr}")


if __name__ == "__main__":
    main()
