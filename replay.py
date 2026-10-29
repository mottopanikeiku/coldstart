"""Replay saved policies in a new process, then score them on episodes never used to select them.

    uv run python replay.py --input results/policies --output results/policies/replay.json

For every record with a saved policy, I check its contract against this environment and
replay the 100 protocol evaluation seeds. The replay must give the recorded returns and the
recorded digest of every action. Then I run 1,000 fresh episodes. Training draws seeds below
1,000,000 and the protocol evaluates 1,000,000 to 1,000,099, so the fresh seeds played no
part in training, stopping or any development choice.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from control_clock.policies import build, contract, replay

FRESH_SEEDS = tuple(range(2_000_000, 2_001_000))
BLOCK = 100


def check(record: dict) -> dict:
    saved = record["policy"]
    expected = contract(record["task"], saved["parameters"]["kind"])
    if saved["contract"] != expected:
        raise ValueError(f"contract differs: saved {saved['contract']}, here {expected}")
    policy = build(saved["parameters"])
    first, last = saved["evaluation_seeds"]
    returns, digest = replay(record["task"], policy, range(first, last + 1))
    passing = record["evaluations"][-1]
    if returns != passing["returns"] or digest != saved["evaluation_actions_sha256"]:
        raise ValueError("replay differs from the recorded passing evaluation")
    fresh, _ = replay(record["task"], policy, FRESH_SEEDS)
    blocks = np.asarray(fresh).reshape(-1, BLOCK).mean(axis=1)
    threshold = saved["contract"]["threshold"]
    return {
        "task": record["task"],
        "method": record["method"],
        "variant": record["variant"],
        "seed": record["seed"],
        "source": record.get("source"),
        "selection_mean": passing["mean_return"],
        "replay_matches": True,
        "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "fresh_mean": float(np.mean(fresh)),
        "fresh_passes": bool(np.mean(fresh) >= threshold),
        "fresh_block_means": blocks.tolist(),
        "fresh_blocks_passing": int((blocks >= threshold).sum()),
        "fresh_returns": fresh,
    }


def summarize(results: list[dict]) -> dict:
    cohorts = defaultdict(list)
    for result in results:
        cohorts[f"{result['task']}/{result['method']}/{result['variant']}"].append(result)
    summary = {}
    for key, rows in sorted(cohorts.items()):
        selection = [row["selection_mean"] for row in rows]
        fresh = [row["fresh_mean"] for row in rows]
        summary[key] = {
            "policies": len(rows),
            "replays_matching": sum(row["replay_matches"] for row in rows),
            "median_selection_mean": float(np.median(selection)),
            "median_fresh_mean": float(np.median(fresh)),
            "min_fresh_mean": float(np.min(fresh)),
            "fresh_passing": sum(row["fresh_passes"] for row in rows),
            "fresh_blocks_passing": sum(row["fresh_blocks_passing"] for row in rows),
            "fresh_blocks": sum(len(row["fresh_block_means"]) for row in rows),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*__*.json"))
    # bench.py's profiled runs skip saving; their plain twins carry the same policy.
    records = [json.loads(path.read_text()) for path in paths if "__profiled" not in path.name]
    saved = [r for r in records if r.get("policy") and "error" not in r["policy"]]
    if len(saved) != sum(record["solved"] for record in records):
        raise SystemExit("every solved record must carry a saved policy")
    results = [check(record) for record in saved]
    payload = {"summary": summarize(results), "policies": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for key, row in payload["summary"].items():
        print(key, json.dumps(row))


if __name__ == "__main__":
    main()
