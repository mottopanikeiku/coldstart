"""The systems benchmark gates, accounts for time and replays policies honestly."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bench_summary import summarize
from control_clock import backends, systems
from control_clock.phases import Recorder
from control_clock.policies import build, replay
from control_clock.protocol import EVAL_SEEDS
from replay import FRESH_SEEDS
from replay import summarize as summarize_replay

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = {"envpool": "envpool", "gymnax": "gymnax", "gymnax-scan": "gymnax"}


def available(backend):
    module = OPTIONAL.get(backend)
    return module is None or importlib.util.find_spec(module) is not None


CASES = [
    pytest.param(
        backend,
        task,
        marks=pytest.mark.skipif(not available(backend), reason="needs the jax or bench extra"),
    )
    for task in ("CartPole-v1", "Acrobot-v1")
    for backend in backends.BACKENDS
    if backends.supports(backend, task)
]


@pytest.mark.parametrize("backend, task", CASES)
def test_every_backend_passes_the_gate(backend, task):
    gate = backends.gate(backend, task, slots=16, steps=32)
    assert gate["passed"], gate
    assert gate["compared_transitions"] >= 16 * 8


def test_gate_rejects_changed_dynamics(monkeypatch):
    step = backends.NumpyBackend.step

    def drifting(self, actions):
        obs, reward, terminated, done = step(self, actions)
        return obs + 1e-3, reward, terminated, done

    monkeypatch.setattr(backends.NumpyBackend, "step", drifting)
    assert not backends.gate("numpy", "CartPole-v1", slots=8, steps=8)["passed"]


@pytest.mark.skipif(not available("gymnax"), reason="needs the jax extra")
def test_scan_gate_runs_the_fused_path_and_rejects_a_corrupted_scan(monkeypatch):
    gate = backends.gate("gymnax-scan", "CartPole-v1", slots=16, steps=16)
    assert gate["passed"] and gate["scan_passed"], gate
    fused, unfused = gate["scan_episode_ends"]
    assert fused == unfused > 0
    states = backends.GymnaxBackend.states
    monkeypatch.setattr(backends.GymnaxScanBackend, "states", lambda self: states(self) + 1e-3)
    assert not backends.gate("gymnax-scan", "CartPole-v1", slots=16, steps=16)["passed"]

    def broken(*args):
        raise RuntimeError("no fused backend")

    monkeypatch.setitem(backends.CLASSES, "gymnax-scan", broken)
    with pytest.raises(RuntimeError, match="no fused backend"):
        backends.gate("gymnax-scan", "CartPole-v1", slots=16, steps=16)


def test_decompose_records_source_revision_with_every_run(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import bench

    source = {"commit": "abc123", "dirty": False}
    monkeypatch.setattr(bench, "cohorts", lambda: [("CartPole-v1", "cem", "default")])
    monkeypatch.setattr(bench, "source_revision", lambda: source)
    monkeypatch.setattr(bench, "launch", lambda *args, **kwargs: ({"solve_seconds": 1.0}, {}))
    bench.decompose(SimpleNamespace(output=tmp_path, seeds="0:1"))
    records = [json.loads(path.read_text()) for path in (tmp_path / "decompose").glob("*.json")]
    assert len(records) == 2 and all(record["source"] == source for record in records)


@pytest.mark.parametrize("backend", ["numpy", "gymnasium-vector"])
def test_transitions_exclude_next_step_reset_slots(monkeypatch, backend):
    monkeypatch.setattr(systems, "calls_for", lambda backend, size: 200)
    row = systems.measure_size(backend, "CartPole-v1", 8, 0)
    assert len(row["step_ns"]) == len(row["done_counts"]) == 200
    assert sum(row["done_counts"]) > 0
    if backend == "numpy":
        assert row["transitions"] == 200 * 8 and row["next_step_reset_slots"] == 0
        assert sum(row["reset_done_ns"]) > 0
    else:
        assert row["transitions"] == 200 * 8 - row["next_step_reset_slots"]
        assert row["next_step_reset_slots"] > 0 and not any(row["reset_done_ns"])


def test_recorder_nests_frames_and_counts_calls():
    recorder = Recorder()
    inner = recorder.wrap(lambda: time.sleep(0.002), "env_step")
    outer = recorder.wrap(lambda: [inner() for _ in range(3)], "rollout")
    outer()
    assert recorder.calls == {"rollout/env_step": 3, "rollout": 1}
    assert recorder.seconds["rollout"] >= recorder.seconds["rollout/env_step"] >= 0.006
    recorder.open("update")
    recorder.close("rollout")  # Closing a frame that is not on top is ignored.
    assert recorder.stack[-1][0] == "update"


def launch(module, task, method, variant, seed):
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", module, "--task", task, "--method", method]
    command += ["--seed", str(seed), "--limit", "120", "--variant", variant]
    command += ["--started", repr(time.perf_counter())]
    output = subprocess.run(command, capture_output=True, text=True, env=environment, cwd=ROOT)
    assert output.returncode == 0, output.stderr
    return json.loads(output.stdout)


def returns(record):
    return [evaluation["returns"] for evaluation in record["evaluations"]]


@pytest.fixture(scope="module")
def plain():
    # CEM seed 5 passed CartPole at its second evaluation in the published run.
    return launch("control_clock.worker", "CartPole-v1", "cem", "default", 5)


PUBLISHED = ROOT / "results/final/CartPole-v1__cem__default__005.json"


def test_profiled_worker_reproduces_the_published_run_and_accounts_for_its_time(plain):
    published = json.loads(PUBLISHED.read_text())
    profiled = launch("control_clock.phases", "CartPole-v1", "cem", "default", 5)
    assert returns(plain) == returns(profiled) == returns(published)
    decomposition = profiled["decomposition"]
    phases = decomposition["phases"]
    assert {"startup", "imports", "setup", "rollout", "evaluation", "update"} <= set(phases)
    assert phases["rollout/env_step"] < phases["rollout"]
    assert abs(decomposition["unattributed_seconds"]) < 0.05 * decomposition["total_seconds"]
    assert decomposition["total_seconds"] == profiled["solve_seconds"]
    # The plain worker saves the passing policy after its clock; the profiled one does not.
    assert profiled["policy"] is None and plain["policy"]["evaluation_actions_sha256"]


def test_saved_policy_replays_in_a_new_process(plain):
    record = plain
    saved = record["policy"]
    policy = build(json.loads(json.dumps(saved["parameters"])))
    replayed, digest = replay("CartPole-v1", policy, EVAL_SEEDS)
    assert replayed == record["evaluations"][-1]["returns"]
    assert digest == saved["evaluation_actions_sha256"]
    assert saved["contract"]["gymnasium"] == "1.2.1"


def test_fresh_cohort_is_disjoint_from_training_and_protocol_seeds():
    # Training draws integers in [0, 1_000_000); the protocol evaluates 1_000_000..1_000_099.
    assert min(FRESH_SEEDS) >= 1_000_000 and not set(FRESH_SEEDS) & set(EVAL_SEEDS)
    assert len(FRESH_SEEDS) == 1000


def test_numpy_evaluator_matches_the_protocol_on_a_saved_policy(plain):
    record = json.loads(PUBLISHED.read_text())
    policy = build(plain["policy"]["parameters"])
    counterfactual = systems.numpy_evaluate("CartPole-v1", policy, EVAL_SEEDS)
    assert np.array_equal(counterfactual, record["evaluations"][-1]["returns"])


def test_benchmark_summary_recomputes_from_raw_files():
    saved = json.loads((ROOT / "results/bench/summary.json").read_text())
    assert summarize(ROOT / "results/bench") == saved
    assert all(row["gate_passed"] for row in saved["environments"].values())
    for cohort in saved["decomposition"].values():
        assert cohort["reproduced_published_evaluations"] == cohort["seeds"] == 5
        assert cohort["unattributed_share_max_abs"] < 0.001
    assert all(row["returns_identical"] for row in saved["evaluation"].values())


def test_replay_summary_recomputes_and_every_replay_matched():
    saved = json.loads((ROOT / "results/bench/replay.json").read_text())
    assert summarize_replay(saved["policies"]) == saved["summary"]
    assert len(saved["policies"]) == 30 and all(p["replay_matches"] for p in saved["policies"])
    for row in saved["policies"]:
        blocks = np.asarray(row["fresh_returns"]).reshape(10, 100).mean(axis=1)
        assert np.allclose(blocks, row["fresh_block_means"])
