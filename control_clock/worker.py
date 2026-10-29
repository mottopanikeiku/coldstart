"""Single-seed worker. Import costs occur after the parent starts its clock."""

import argparse
import importlib
import importlib.metadata
import json
import time
import traceback

from control_clock.protocol import EVAL_SEEDS, Finished, RunContext

MODULES = {
    "ppo-cpu": "control_clock.ppo",
    "sb3-zoo": "control_clock.library_baselines",
    "cleanrl": "control_clock.cleanrl_baseline",
    "ars": "control_clock.search",
    "cem": "control_clock.search",
}


def installed_version(name):
    """Box2D and SWIG are optional extras; record null when they are absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=MODULES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--limit", type=float, required=True)
    parser.add_argument("--variant", default="default")
    return parser.parse_args(argv)


def execute(args, *, save_policy=True) -> dict:
    ctx = RunContext(args.task, args.seed, args.started, args.limit)
    ctx.configuration.update(method=args.method, variant=args.variant)
    error = None
    try:
        if args.method in ("ppo-cpu", "sb3-zoo", "cleanrl"):
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        module = importlib.import_module(MODULES[args.method])
        module.train(ctx)
        stop_reason = "step horizon"
    except Finished as exception:
        stop_reason = str(exception)
    except Exception:
        error = traceback.format_exc()
        stop_reason = "worker error"
    stopped = time.perf_counter()
    ctx.close()
    policy = None
    if save_policy and ctx.solved and ctx.policy_parameters is not None:
        # After the clock: replay needs about a second, and the parent kills at limit + 5 s.
        if stopped - args.started > args.limit - 10:
            policy = {"error": "skipped: too close to the parent watchdog"}
        else:
            from control_clock.policies import save

            try:
                policy = save(ctx)
            except Exception:
                policy = {"error": traceback.format_exc()}
    versions = {
        name: installed_version(name)
        for name in ("numpy", "gymnasium", "torch", "stable-baselines3", "box2d-py", "swig")
    }
    return {
        "solved": ctx.solved,
        "solve_seconds": ctx.solve_seconds,
        "stop_seconds": stopped - args.started,
        "steps": ctx.steps,
        "evaluations": ctx.evaluations,
        "configuration": ctx.configuration,
        "stop_reason": stop_reason,
        "versions": versions,
        "error": error,
        "evaluation_policy": "deterministic argmax, lowest-index ties",
        "evaluation_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        "evaluation_episodes": len(EVAL_SEEDS),
        "policy": policy,
    }


def main():
    record = execute(parse())
    print(json.dumps(record))
    if record["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
