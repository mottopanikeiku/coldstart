"""Summarize the extra GPU cohort without mixing it with laptop measurements."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize import km_median


def summarize(directory):
    records = [
        json.loads(path.read_text()) for path in sorted(directory.glob("*__jax-ppo__*.json"))
    ]
    groups = []
    for task in ("CartPole-v1", "Acrobot-v1"):
        cohort = [record for record in records if record["task"] == task]
        successful = [record["solve_seconds"] for record in cohort if record["solved"]]
        groups.append(
            {
                "task": task,
                "runs": len(cohort),
                "solved": len(successful),
                "median_seconds": float(np.median(successful)) if successful else None,
                "km_median_seconds": km_median(cohort),
                "quartiles_seconds": np.quantile(successful, [0.25, 0.75]).tolist()
                if successful
                else None,
                "records": [
                    {
                        "seed": record["seed"],
                        "solved": record["solved"],
                        "solve_seconds": record["solve_seconds"],
                        "parent_seconds": record["parent_seconds"],
                        "stop_reason": record["stop_reason"],
                        "last_mean_return": record["evaluations"][-1]["mean_return"]
                        if record["evaluations"]
                        else None,
                        "file": f"{task}__jax-ppo__{record['seed']:03}.json",
                    }
                    for record in cohort
                ],
            }
        )
    return {
        "hardware": "Modal L4; not the laptop table",
        "protocol": "fresh Python process through first passing 100-episode Gymnasium evaluation",
        "groups": groups,
        "run": json.loads((directory / "run.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(summarize(args.input), indent=2) + "\n")


if __name__ == "__main__":
    main()
