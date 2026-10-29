"""The normal single-seed worker with wall-clock phase accounting, for bench.py.

    python -m control_clock.phases --task CartPole-v1 --method ars --seed 0 --started <clock>

I wrap a few functions instead of editing the training code. Each wrapped call or interval
is a frame; nested frames get paths such as ``rollout/env_step``. On the main timeline:

- startup: parent clock start to this module's first statement (interpreter, site);
- imports: from there to the method's ``train`` (worker, Gymnasium, NumPy, Torch);
- setup: ``train`` entry to its first ``checkpoint`` (environments, model, optimizer);
- evaluation: every ``RunContext.checkpoint`` call, including building its 100 environments;
- rollout: ARS/CEM ``candidate_returns``; for PPO, the interval from a checkpoint to GAE;
- update: ARS/CEM ``ars_update``/``cem_update``; for PPO, the interval from GAE to the
  next checkpoint (all optimizer epochs);
- gae: PPO's ``advantages_reference``.

Inside rollouts, ``env_step``, ``env_reset`` and ARS/CEM ``policy`` calls are frames too.
Whatever no frame covers is reported as unattributed, so the parts add up to the protocol's
own solve clock by construction; the check is that unattributed time stays small.
"""

from __future__ import annotations

import time

ENTERED = time.perf_counter()

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

from control_clock import worker  # noqa: E402


class Recorder:
    def __init__(self):
        self.stack: list[tuple[str, float]] = []
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.train_entered: float | None = None

    def open(self, name: str) -> None:
        self.stack.append((name, time.perf_counter()))

    def close(self, name: str) -> None:
        if not self.stack or self.stack[-1][0] != name:
            return
        now = time.perf_counter()
        _, begin = self.stack.pop()
        path = "/".join([frame for frame, _ in self.stack] + [name])
        self.seconds[path] += now - begin
        self.calls[path] += 1

    def wrap(self, function, name: str, *, before=None, after=None):
        def wrapped(*args, **kwargs):
            if before:
                before()
            self.open(name)
            try:
                return function(*args, **kwargs)
            finally:
                self.close(name)
                if after:
                    after()

        return wrapped


def install(recorder: Recorder, method: str):
    """Import the method's modules and wrap the calls that delimit phases."""
    from control_clock import protocol, vector_env

    def end_setup():
        recorder.close("setup")

    def entered(train):
        def wrapped(ctx):
            recorder.train_entered = time.perf_counter()
            recorder.open("setup")
            return train(ctx)

        return wrapped

    batch = vector_env._Batch
    vector_env._ClassicBatch.step = recorder.wrap(vector_env._ClassicBatch.step, "env_step")
    batch.reset = recorder.wrap(batch.reset, "env_reset")
    batch.reset_done = recorder.wrap(batch.reset_done, "env_reset")
    if method in ("ars", "cem"):
        from control_clock import search as module

        module.candidate_returns = recorder.wrap(module.candidate_returns, "rollout")
        module.greedy_actions = recorder.wrap(module.greedy_actions, "policy")
        module.ars_update = recorder.wrap(module.ars_update, "update")
        module.cem_update = recorder.wrap(module.cem_update, "update")
        protocol.RunContext.checkpoint = recorder.wrap(
            protocol.RunContext.checkpoint, "evaluation", before=end_setup
        )
    elif method == "ppo-cpu":
        from control_clock import ppo as module

        def before_checkpoint():
            recorder.close("setup")
            recorder.close("update")

        module.advantages_reference = recorder.wrap(
            module.advantages_reference,
            "gae",
            before=lambda: recorder.close("rollout"),
            after=lambda: recorder.open("update"),
        )
        protocol.RunContext.checkpoint = recorder.wrap(
            protocol.RunContext.checkpoint,
            "evaluation",
            before=before_checkpoint,
            after=lambda: recorder.open("rollout"),
        )
    else:
        raise ValueError(f"no phase accounting for {method}")
    module.train = entered(module.train)


def decompose(record: dict, recorder: Recorder, started: float) -> dict:
    """Seconds per phase up to the end of the passing evaluation."""
    phases = {
        "startup": ENTERED - started,
        "imports": recorder.train_entered - ENTERED,
        **{path: seconds for path, seconds in sorted(recorder.seconds.items())},
    }
    total = record["solve_seconds"] if record["solved"] else record["stop_seconds"]
    top = sum(seconds for path, seconds in phases.items() if "/" not in path)
    return {
        "total_seconds": total,
        "phases": phases,
        "calls": dict(sorted(recorder.calls.items())),
        "unattributed_seconds": total - top,
    }


def main():
    args = worker.parse()
    recorder = Recorder()
    install(recorder, args.method)
    record = worker.execute(args, save_policy=False)
    # PPO opens its next rollout as the passing evaluation returns; that is after the clock.
    recorder.stack.clear()
    record["decomposition"] = decompose(record, recorder, args.started)
    print(json.dumps(record))
    if record["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
