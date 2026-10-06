import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from control_clock.protocol import BUDGETS, THRESHOLDS
from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]


def expected_cohort():
    plan = json.loads((ROOT / "configs/final.json").read_text())
    expected = set()
    for chunk in plan["chunks"]:
        worst_case_seconds = 0
        for item in chunk:
            first, last = map(int, item["seeds"].split(":"))
            worst_case_seconds += (last - first) * (BUDGETS[item["task"]] + 5)
            for seed in range(first, last):
                key = (item["task"], item["method"], item.get("variant", "default"), seed)
                assert key not in expected
                expected.add(key)
        assert worst_case_seconds < 1800
    return expected


def test_final_plan_has_required_cohorts_and_bounded_windows():
    grouped = defaultdict(list)
    for task, method, _, seed in expected_cohort():
        grouped[task, method].append(seed)
    for task in ("CartPole-v1", "Acrobot-v1"):
        for method in ("ppo-cpu", "ars", "cem", "sb3-zoo", "cleanrl"):
            count = 5 if method in ("sb3-zoo", "cleanrl") else 20
            assert sorted(grouped[task, method]) == list(range(count))


def test_raw_final_records_match_plan_and_endpoints():
    records = [json.loads(path.read_text()) for path in (ROOT / "results/final").glob("*.json")]
    actual = [(r["task"], r["method"], r["variant"], r["seed"]) for r in records]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected_cohort()
    for record in records:
        assert record["phase"] == "final"
        assert record["limit_seconds"] == BUDGETS[record["task"]]
        assert record["returncode"] == 0 or record["watchdog"]
        evaluations = record["evaluations"]
        for evaluation in evaluations:
            assert len(evaluation["returns"]) == 100
            assert np.isclose(np.mean(evaluation["returns"]), evaluation["mean_return"])
            expected_pass = (
                evaluation["mean_return"] >= THRESHOLDS[record["task"]]
                and evaluation["elapsed_seconds"] <= record["limit_seconds"]
            )
            assert evaluation["passes"] is expected_pass
        passing = [evaluation for evaluation in evaluations if evaluation["passes"]]
        if record["solved"]:
            assert len(passing) == 1 and passing[0] == evaluations[-1]
            assert record["solve_seconds"] == passing[0]["elapsed_seconds"]
            assert record["solve_seconds"] <= record["limit_seconds"]
            assert passing[0]["mean_return"] >= THRESHOLDS[record["task"]]
            assert sum(e["evaluation_seconds"] for e in evaluations) <= record["solve_seconds"]
        else:
            assert not passing
            assert record["solve_seconds"] is None


def test_summary_numbers_recompute_from_raw_records():
    saved = json.loads((ROOT / "results/summary.json").read_text())
    for group in saved.values():
        records = [json.loads((ROOT / name).read_text()) for name in group["files"]]
        recomputed = summarize(records)
        assert {key: value for key, value in group.items() if key != "files"} == recomputed
