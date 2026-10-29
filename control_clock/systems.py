"""Measurements that bench.py runs inside fresh, pinned processes; each prints one JSON line.

    python -m control_clock.systems env --backend numpy --task CartPole-v1 --sizes 1,64
    python -m control_clock.systems import --module torch --started <parent clock>
    python -m control_clock.systems evaluation --task CartPole-v1 --policies a.json,b.json

Clocks are time.perf_counter_ns. Only backend calls sit inside a timed region; action
generation and done-flag bookkeeping stay outside.
"""

from __future__ import annotations

import time

STARTED = time.perf_counter()

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

WARMUP = 32


def calls_for(backend: str, size: int) -> int:
    """Fixed workload per batch size: 8,192 calls at size 1, never fewer than 128 calls."""
    if backend == "gymnax-scan":
        return max(16, 1024 // size)
    return max(128, 8192 // size)


def measure_size(backend_name: str, task: str, size: int, seed: int) -> dict:
    import numpy as np

    from control_clock import backends

    clock = time.perf_counter_ns
    begin = clock()
    backend = backends.make(backend_name, task, size, seed)
    construct = clock() - begin
    begin = clock()
    backend.reset()
    reset = clock() - begin
    calls = calls_for(backend_name, size)
    per_call = backends.SCAN_LENGTH if backend_name == "gymnax-scan" else 1
    shape = (WARMUP + calls, per_call, size) if per_call > 1 else (WARMUP + calls, size)
    actions_count = {"CartPole-v1": 2, "Acrobot-v1": 3}[task]
    actions = np.random.default_rng(seed).integers(0, actions_count, size=shape)
    step_ns = np.zeros(calls, dtype=np.int64)
    reset_ns = np.zeros(calls, dtype=np.int64)
    done_counts = np.zeros(calls, dtype=np.int64)
    first_step = None
    previous_done = 0
    restarting = 0
    try:
        for index in range(WARMUP + calls):
            begin = clock()
            _, _, _, done = backend.step(actions[index])
            elapsed = clock() - begin
            if first_step is None:
                first_step = elapsed
            spent = 0
            if per_call == 1:
                count = int(done.sum())
                if backend.restart == "explicit" and count:
                    seeds = backend.reset_seeds(done)
                    begin = clock()
                    backend.reset_done(done, seeds)
                    spent = clock() - begin
            else:
                count = done
            if index >= WARMUP:
                timed = index - WARMUP
                step_ns[timed], reset_ns[timed], done_counts[timed] = elapsed, spent, count
                if backend.restart == "next-step":
                    restarting += previous_done
            previous_done = count
    finally:
        backend.close()
    return {
        "size": size,
        "calls": calls,
        "steps_per_call": per_call,
        "warmup_calls": WARMUP,
        "transitions": calls * per_call * size - restarting,
        "next_step_reset_slots": restarting,
        "construct_ns": construct,
        "full_reset_ns": reset,
        "first_step_ns": first_step,
        "step_ns": step_ns.tolist(),
        "reset_done_ns": reset_ns.tolist(),
        "done_counts": done_counts.tolist(),
    }


def env(args) -> dict:
    from control_clock import backends

    gate = backends.gate(args.backend, args.task)
    result = {"backend": args.backend, "task": args.task, "gate": gate, "sizes": []}
    if not gate["passed"]:
        return result
    for size in map(int, args.sizes.split(",")):
        result["sizes"].append(measure_size(args.backend, args.task, size, args.seed))
    return result


def import_module(args) -> dict:
    """Interpreter start to first statement, then the import alone, in one fresh process."""
    begin = time.perf_counter()
    if args.module != "pass":
        importlib.import_module(args.module)
    end = time.perf_counter()
    return {
        "module": args.module,
        "startup_seconds": STARTED - args.started,
        "import_seconds": end - begin,
        "modules_loaded": len(sys.modules),
    }


def zero_policy(task: str) -> dict:
    import numpy as np

    from control_clock.policies import affine

    size = {"CartPole-v1": 4, "Acrobot-v1": 6}[task]
    actions = {"CartPole-v1": 2, "Acrobot-v1": 3}[task]
    return affine(np.zeros((actions, size + 1)), np.ones(size))


def numpy_evaluate(task: str, policy, seeds) -> list[float]:
    """The same episodes on this repository's NumPy batch: a counterfactual evaluator."""
    import numpy as np

    from control_clock.vector_env import make_vector

    batch = make_vector(task, len(seeds))
    try:
        observations = batch.reset(list(seeds))
        returns = np.zeros(len(seeds))
        active = np.ones(len(seeds), dtype=bool)
        actions = np.zeros(len(seeds), dtype=np.int64)
        while active.any():
            indices = np.flatnonzero(active)
            actions[indices] = policy(observations[indices])
            observations, rewards, terminated, truncated = batch.step(actions, active=active)
            returns += rewards
            active &= ~(terminated | truncated)
    finally:
        batch.close()
    return returns.tolist()


def evaluation(args) -> dict:
    """Protocol evaluator (first call builds 100 environments) against the NumPy batch."""
    import math

    from control_clock.policies import build, replay
    from control_clock.protocol import EVAL_SEEDS, Finished, RunContext

    policies = {"initial zero weights": zero_policy(args.task)}
    for path in filter(None, args.policies.split(",")):
        record = json.loads(Path(path).read_text())
        name = f"{record['method']} seed {record['seed']} passing policy"
        policies[name] = record["policy"]["parameters"]
    rows = []
    for name, parameters in policies.items():
        policy = build(parameters)
        ctx = RunContext(args.task, 0, time.perf_counter(), math.inf)
        protocol_seconds, numpy_seconds = [], []
        try:
            for _ in range(args.repeats):
                begin = time.perf_counter()
                try:
                    ctx.checkpoint(policy, 0, force=True)
                except Finished:
                    pass
                protocol_seconds.append(time.perf_counter() - begin)
        finally:
            ctx.close()
        for _ in range(args.repeats):
            begin = time.perf_counter()
            returns = numpy_evaluate(args.task, policy, EVAL_SEEDS)
            numpy_seconds.append(time.perf_counter() - begin)
        protocol_returns = ctx.evaluations[-1]["returns"]
        observed = []

        def counted(observations, policy=policy, observed=observed):
            observed.append(len(observations))
            return policy(observations)

        returns_counted, _ = replay(args.task, counted, EVAL_SEEDS)
        if returns_counted != protocol_returns:
            raise RuntimeError("protocol evaluation is not repeatable")
        rows.append(
            {
                "policy": name,
                "kind": parameters["kind"],
                "mean_return": ctx.evaluations[-1]["mean_return"],
                "transitions": sum(observed),
                "policy_calls": len(observed),
                "protocol_seconds": protocol_seconds,
                "numpy_seconds": numpy_seconds,
                "returns_identical": returns == protocol_returns,
                "episodes_differing": sum(a != b for a, b in zip(returns, protocol_returns)),
            }
        )
    return {"task": args.task, "repeats": args.repeats, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    env_parser = commands.add_parser("env")
    env_parser.add_argument("--backend", required=True)
    env_parser.add_argument("--task", required=True)
    env_parser.add_argument("--sizes", required=True)
    env_parser.add_argument("--seed", type=int, default=0)
    import_parser = commands.add_parser("import")
    import_parser.add_argument("--module", required=True)
    import_parser.add_argument("--started", type=float, required=True)
    evaluation_parser = commands.add_parser("evaluation")
    evaluation_parser.add_argument("--task", required=True)
    evaluation_parser.add_argument("--policies", default="")
    evaluation_parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run = {"env": env, "import": import_module, "evaluation": evaluation}[args.command]
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
