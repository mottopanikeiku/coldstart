"""The GPU evidence stays distinct from the original laptop cohort."""

import json
from pathlib import Path

import numpy as np

from gpu_summary import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_summary_and_complete_cohort():
    directory = ROOT / "results/gpu/final"
    summary = json.loads((ROOT / "results/gpu/summary.json").read_text())
    assert summary == summarize(directory)
    for group in summary["groups"]:
        assert group["runs"] == 3
        assert [record["seed"] for record in group["records"]] == [0, 1, 2]
        assert group["solved"] == sum(record["solved"] for record in group["records"])
        for row in group["records"]:
            record = json.loads((directory / row["file"]).read_text())
            assert record["phase"] == "final"
            assert record["method"] == "jax-ppo"
            assert record["device_kind"] == ["NVIDIA L4"]
            assert record["evaluation_policy"] == "deterministic argmax, lowest-index ties"
            assert record["evaluation_seeds"] == [1_000_000, 1_000_099]
            assert record["evaluation_episodes"] == 100
            assert record["limit_seconds"] == (120 if record["task"] == "CartPole-v1" else 300)
            threshold = 475 if record["task"] == "CartPole-v1" else -100
            previous_time, previous_steps = 0, -1
            for evaluation in record["evaluations"]:
                assert len(evaluation["returns"]) == 100
                assert evaluation["mean_return"] == np.mean(evaluation["returns"])
                elapsed = evaluation["elapsed_seconds"]
                assert previous_time < elapsed <= record["parent_seconds"]
                assert 0 < evaluation["evaluation_seconds"] <= elapsed - previous_time
                assert previous_steps < evaluation["steps"]
                assert evaluation["passes"] == (
                    evaluation["mean_return"] >= threshold and elapsed <= record["limit_seconds"]
                )
                previous_time, previous_steps = elapsed, evaluation["steps"]
            passing = [evaluation for evaluation in record["evaluations"] if evaluation["passes"]]
            assert record["solved"] == bool(passing)
            assert record["solve_seconds"] == (passing[0]["elapsed_seconds"] if passing else None)
            if record["solved"]:
                assert record["returncode"] == 0
                assert not record["watchdog"]
                assert record["stop_reason"] == "solved"


def test_gpu_cost_estimate():
    for phase in ("pilot", "final"):
        run = json.loads((ROOT / f"results/gpu/{phase}/run.json").read_text())
        assert run["hardware"] == "NVIDIA L4"
        assert run["container_cost_estimate_usd"] == (
            run["container_seconds"] / 3600 * run["hourly_rate_usd"]
        )
        assert run["hourly_rate_usd"] == 0.7992 + 2 * 0.047160 + 4 * 0.007992
    cost = json.loads((ROOT / "results/gpu/cost.json").read_text())
    assert cost["total_all_in_cost_estimate_usd"] == round(
        sum(run["all_in_cost_estimate_usd"] for run in cost["runs"]), 4
    )
    assert cost["total_all_in_cost_estimate_usd"] <= cost["allocation_usd"]
    for run in cost["runs"]:
        if run["phase"] in ("pilot", "final"):
            assert run["client_image_and_provisioning_overhead_seconds"] == (
                run["modal_client_wall_minutes"] * 60 - run["container_seconds"]
            )
