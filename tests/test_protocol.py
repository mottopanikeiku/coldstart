import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from control_clock.protocol import EVAL_SEEDS, Finished, RunContext
from summarize import km_median, summarize


def test_training_seeds_disjoint():
    ctx = RunContext("CartPole-v1", 19, time.perf_counter(), 120)
    seeds = ctx.training_seeds(100_000)
    assert all(0 <= seed < min(EVAL_SEEDS) for seed in seeds)
    assert len(set(EVAL_SEEDS)) == 100


def test_initial_evaluation_includes_clock_offset():
    ctx = RunContext("CartPole-v1", 0, time.perf_counter() - 10, 120)
    try:
        ctx.checkpoint(lambda observations: np.zeros(len(observations), dtype=int), 0)
        assert len(ctx.evaluations) == 1
        entry = ctx.evaluations[0]
        assert len(entry["returns"]) == 100
        assert entry["elapsed_seconds"] >= 10 + entry["evaluation_seconds"]
        assert not entry["passes"]
        ctx.checkpoint(lambda observations: np.zeros(len(observations), dtype=int), 1)
        assert len(ctx.evaluations) == 1
    finally:
        ctx.close()


def test_time_limit():
    ctx = RunContext("CartPole-v1", 0, time.perf_counter() - 2, 1)
    with pytest.raises(Finished, match="time limit"):
        ctx.checkpoint(lambda observations: np.zeros(len(observations), dtype=int), 0)
    assert not ctx.solved


@pytest.mark.parametrize("finish_time,expected_success", [(0.5, True), (2.0, False)])
def test_passing_evaluation_must_finish_before_deadline(monkeypatch, finish_time, expected_success):
    from control_clock import protocol

    clock = [0.0]
    stepped = [0]

    class OneStepEnv:
        def reset(self, seed):
            assert seed in EVAL_SEEDS
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            stepped[0] += 1
            if stepped[0] == 100:
                clock[0] = finish_time
            return np.zeros(4, dtype=np.float32), 500.0, True, False, {}

        def close(self):
            pass

    monkeypatch.setattr(protocol.gym, "make", lambda task: OneStepEnv())
    monkeypatch.setattr(protocol.time, "perf_counter", lambda: clock[0])
    ctx = RunContext("CartPole-v1", 0, 0.0, 1.0)
    try:
        with pytest.raises(Finished, match="solved" if expected_success else "time limit"):
            ctx.checkpoint(lambda obs: np.zeros(len(obs), dtype=int), 0)
        assert ctx.solved is expected_success
        assert ctx.evaluations[0]["evaluation_seconds"] == finish_time
        assert ctx.solve_seconds == (finish_time if expected_success else None)
    finally:
        ctx.close()


def record(seed, solved, seconds):
    return dict(
        seed=seed,
        solved=solved,
        solve_seconds=seconds if solved else None,
        stop_seconds=seconds,
        parent_seconds=seconds + 0.01,
        limit_seconds=120,
        stop_reason="solved" if solved else "time limit",
    )


def test_summary_retains_failures():
    records = [record(0, True, 10), record(1, True, 20), record(2, False, 120)]
    summary = summarize(records)
    assert summary["seeds"] == 3
    assert summary["successes"] == 2
    assert summary["success_rate"] == 2 / 3
    assert summary["conditional_median_seconds"] == 15
    assert summary["conditional_q25_seconds"] == 12.5
    assert summary["conditional_q75_seconds"] == 17.5
    assert summary["km_median_seconds"] == 20


def test_summary_seed_records_are_sorted_and_keep_actual_stop_times():
    failed = record(2, False, 122.5)
    watchdog = record(3, False, 125)
    del watchdog["stop_seconds"]
    watchdog["stop_reason"] = "watchdog"
    records = [watchdog, failed, record(0, True, 10)]
    entries = summarize(records)["records"]
    assert [entry["seed"] for entry in entries] == [0, 2, 3]
    assert entries[0] == {"seed": 0, "solved": True, "seconds": 10, "stop_reason": "solved"}
    assert entries[1] == {"seed": 2, "solved": False, "seconds": 122.5, "stop_reason": "time limit"}
    assert entries[2]["seconds"] == watchdog["parent_seconds"]
    assert entries[2]["stop_reason"] == "watchdog"


def test_km_no_solved():
    assert km_median([record(0, False, 120)]) is None


def test_km_tied_events_before_censoring():
    assert km_median([record(0, True, 10), record(1, False, 10)]) == 10


def test_actual_parent_child_clock_includes_python_startup(tmp_path):
    # CPython imports sitecustomize before the worker module; a worker-local
    # timestamp taken after imports would incorrectly exclude this delay.
    (tmp_path / "sitecustomize.py").write_text("import time\ntime.sleep(0.08)\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "results"
    command = [
        sys.executable,
        "run.py",
        "--task",
        "CartPole-v1",
        "--method",
        "cem",
        "--seeds",
        "0:1",
        "--output",
        str(destination),
        "--phase",
        "development",
        "--limit",
        "0.001",
    ]
    subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True, timeout=15)
    record = json.loads((destination / "CartPole-v1__cem__default__000.json").read_text())
    assert record["stop_reason"] == "time limit"
    assert not record["solved"]
    assert record["evaluations"] == []
    assert record["stop_seconds"] >= 0.08
    assert record["parent_seconds"] >= record["stop_seconds"]
