"""Generate tables from raw records; never silently drop unsuccessful seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def km_median(records):
    """Kaplan–Meier median, with actual early stop times as censoring times."""
    observations = [
        (
            record["solve_seconds"]
            if record["solved"]
            else min(record.get("stop_seconds", record["parent_seconds"]), record["limit_seconds"]),
            record["solved"],
        )
        for record in records
    ]
    at_risk = len(observations)
    survival = 1.0
    for timestamp in sorted({item[0] for item in observations}):
        events = sum(time == timestamp and solved for time, solved in observations)
        censored = sum(time == timestamp and not solved for time, solved in observations)
        survival *= 1 - events / at_risk
        if survival <= 0.5:
            return float(timestamp)
        at_risk -= events + censored
    return None


def summarize(records):
    solved = [record["solve_seconds"] for record in records if record["solved"]]
    quantiles = np.quantile(solved, [0.25, 0.5, 0.75]).tolist() if solved else [None] * 3
    return {
        "seeds": len(records),
        "seed_ids": sorted(record["seed"] for record in records),
        "records": [
            {
                "seed": record["seed"],
                "solved": record["solved"],
                "seconds": (
                    record["solve_seconds"]
                    if record["solved"]
                    else record.get("stop_seconds", record["parent_seconds"])
                ),
                "stop_reason": record["stop_reason"],
            }
            for record in sorted(records, key=lambda item: item["seed"])
        ],
        "successes": len(solved),
        "success_rate": len(solved) / len(records),
        "conditional_q25_seconds": quantiles[0],
        "conditional_median_seconds": quantiles[1],
        "conditional_q75_seconds": quantiles[2],
        "km_median_seconds": km_median(records),
        "limit_seconds": sorted({record["limit_seconds"] for record in records}),
        "stop_reasons": {
            reason: sum(record["stop_reason"] == reason for record in records)
            for reason in sorted({record["stop_reason"] for record in records})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/final"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--phase", default="final", choices=("final", "development"))
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.input.glob("*.json")):
        record = json.loads(path.read_text())
        if record["phase"] != args.phase:
            raise ValueError(f"mixed phase in {path}")
        groups[(record["task"], record["method"], record["variant"])].append((path, record))
    result = {}
    print("| Task | Method | Solved | Median seconds [Q1, Q3], successes only | KM median |")
    print("|---|---|---:|---:|---:|")
    for (task, method, variant), items in sorted(groups.items()):
        records = [record for _, record in items]
        if len({record["seed"] for record in records}) != len(records):
            raise ValueError(f"duplicate seeds in {task}/{method}/{variant}")
        statistics = summarize(records)
        statistics["files"] = [str(path) for path, _ in items]
        result[f"{task}/{method}/{variant}"] = statistics
        median = statistics["conditional_median_seconds"]
        if median is None:
            timing = "—"
        else:
            timing = (
                f"{median:.2f} [{statistics['conditional_q25_seconds']:.2f}, "
                f"{statistics['conditional_q75_seconds']:.2f}]"
            )
        km = statistics["km_median_seconds"]
        km_text = "not reached" if km is None else f"{km:.2f}"
        print(
            f"| {task} | {method} ({variant}) | {len(solved_times(records))}/{len(records)} | "
            f"{timing} | {km_text} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def solved_times(records):
    return [record["solve_seconds"] for record in records if record["solved"]]


if __name__ == "__main__":
    main()
