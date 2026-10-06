"""Single-seed worker. Import costs occur after the parent starts its clock."""

import argparse
import importlib
import importlib.metadata
import json
import time
import traceback

from control_clock.protocol import Finished, RunContext

MODULES = {
    "ppo-cpu": "control_clock.ppo",
    "sb3-zoo": "control_clock.library_baselines",
    "cleanrl": "control_clock.cleanrl_baseline",
    "ars": "control_clock.search",
    "cem": "control_clock.search",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=MODULES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--limit", type=float, required=True)
    parser.add_argument("--variant", default="default")
    args = parser.parse_args()
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
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "gymnasium", "torch", "stable-baselines3", "box2d-py", "swig")
    }
    print(
        json.dumps(
            {
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
                "evaluation_seeds": [1_000_000, 1_000_099],
                "evaluation_episodes": 100,
            }
        )
    )
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
