"""Derive every systems-benchmark table from the raw files that bench.py wrote.

    uv run python bench_summary.py --input results/bench --output results/bench/summary.json

Throughput counts simulated transitions over time spent inside backend calls, including
explicit resets. Medians are over fresh processes (environments, imports) or seeds
(decomposition). The decomposition checks that every profiled run reproduces its plain run
and the published record exactly before reporting any phase.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def quartiles(values) -> dict:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3), "n": len(values)}


def environments(directory: Path) -> dict:
    runs = defaultdict(list)
    for path in sorted(directory.glob("*.json.gz")):
        result = json.loads(gzip.decompress(path.read_bytes()))
        runs[(result["task"], result["backend"])].append(result)
    summary = {}
    for (task, backend), results in sorted(runs.items()):
        gates = [result["gate"] for result in results]
        sizes = defaultdict(list)
        for result in results:
            for row in result["sizes"]:
                sizes[row["size"]].append(row)
        rows = {}
        for size, measured in sorted(sizes.items()):
            throughput = [
                row["transitions"] / ((sum(row["step_ns"]) + sum(row["reset_done_ns"])) / 1e9)
                for row in measured
            ]
            steps = np.concatenate([row["step_ns"] for row in measured]) / 1e3
            resets = sum(sum(row["reset_done_ns"]) for row in measured)
            restarted = sum(sum(row["done_counts"]) for row in measured)
            rows[str(size)] = {
                "transitions_per_second": quartiles(throughput),
                "call_microseconds_p50": float(np.percentile(steps, 50)),
                "call_microseconds_p99": float(np.percentile(steps, 99)),
                "steps_per_call": measured[0]["steps_per_call"],
                "explicit_reset_microseconds_per_slot": resets / restarted / 1e3
                if resets
                else None,
                "construct_milliseconds": quartiles([r["construct_ns"] / 1e6 for r in measured]),
                "full_reset_milliseconds": quartiles([r["full_reset_ns"] / 1e6 for r in measured]),
                "first_step_milliseconds": quartiles([r["first_step_ns"] / 1e6 for r in measured]),
                "processes": len(measured),
            }
        summary[f"{task}/{backend}"] = {
            "gate_passed": all(gate["passed"] for gate in gates),
            "gate_max_observation_difference": max(
                gate["max_observation_difference"] for gate in gates
            ),
            "gate_compared_transitions": gates[0]["compared_transitions"],
            "sizes": rows,
        }
    return summary


def imports(path: Path) -> dict:
    rows = defaultdict(list)
    for row in json.loads(path.read_text()):
        rows[row["module"]].append(row)
    return {
        module: {
            "import_seconds": quartiles([row["import_seconds"] for row in measured]),
            "startup_seconds": quartiles([row["startup_seconds"] for row in measured]),
            "modules_loaded": measured[0]["modules_loaded"],
        }
        for module, measured in rows.items()
    }


def evaluation(path: Path) -> dict:
    summary = {}
    for result in json.loads(path.read_text()):
        for row in result["rows"]:
            first, *repeats = row["protocol_seconds"]
            numpy_seconds = float(np.median(row["numpy_seconds"]))
            protocol_seconds = float(np.median(repeats))
            summary[f"{result['task']}/{row['policy']}"] = {
                "mean_return": row["mean_return"],
                "transitions": row["transitions"],
                "protocol_first_seconds": first,
                "protocol_repeat_seconds": protocol_seconds,
                "numpy_seconds": numpy_seconds,
                "protocol_microseconds_per_transition": protocol_seconds / row["transitions"] * 1e6,
                "numpy_microseconds_per_transition": numpy_seconds / row["transitions"] * 1e6,
                "returns_identical": row["returns_identical"],
                "episodes_differing": row["episodes_differing"],
            }
    return summary


def same_evaluations(a: dict, b: dict) -> bool:
    return [e["returns"] for e in a["evaluations"]] == [
        e["returns"] for e in b["evaluations"]
    ] and [e["steps"] for e in a["evaluations"]] == [e["steps"] for e in b["evaluations"]]


def decomposition(directory: Path, published: Path) -> dict:
    runs = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        runs[(record["task"], record["method"], record["variant"])].setdefault(record["seed"], {})[
            record["kind"]
        ] = record
    summary = {}
    for (task, method, variant), seeds in sorted(runs.items()):
        totals, overheads, shares, seconds, unattributed, parent_gaps = [], [], [], [], [], []
        reproduced = 0
        for seed, pair in sorted(seeds.items()):
            plain, profiled = pair["plain"], pair["profiled"]
            reference = json.loads(
                (published / f"{task}__{method}__{variant}__{seed:03}.json").read_text()
            )
            if not (plain["solved"] and profiled["solved"]):
                raise ValueError(f"{task} {method} seed {seed} did not solve")
            if not (same_evaluations(plain, profiled) and same_evaluations(plain, reference)):
                raise ValueError(f"{task} {method} seed {seed} did not reproduce its evaluations")
            reproduced += 1
            decomposition = profiled["decomposition"]
            total = decomposition["total_seconds"]
            totals.append(plain["solve_seconds"])
            overheads.append(total - plain["solve_seconds"])
            seconds.append(decomposition["phases"])
            shares.append({key: value / total for key, value in decomposition["phases"].items()})
            unattributed.append(decomposition["unattributed_seconds"] / total)
            parent_gaps.append(plain["context"]["parent_seconds"] - plain["stop_seconds"])
        names = sorted({name for phases in seconds for name in phases})
        summary[f"{task}/{method}/{variant}"] = {
            "seeds": len(seeds),
            "reproduced_published_evaluations": reproduced,
            "plain_solve_seconds": quartiles(totals),
            "profiling_overhead_seconds": quartiles(overheads),
            "unattributed_share_max_abs": float(np.max(np.abs(unattributed))),
            "parent_minus_stop_seconds": quartiles(parent_gaps),
            "phase_seconds": {
                name: float(np.median([phases.get(name, 0.0) for phases in seconds]))
                for name in names
            },
            "phase_share": {
                name: float(np.median([phases.get(name, 0.0) for phases in shares]))
                for name in names
            },
        }
    return summary


def summarize(directory: Path, published: Path = ROOT / "results/final") -> dict:
    return {
        "environments": environments(directory / "env"),
        "imports": imports(directory / "imports.json"),
        "evaluation": evaluation(directory / "evaluation.json"),
        "decomposition": decomposition(directory / "decompose", published),
    }


def tables(summary: dict) -> str:
    lines = ["| Task | Backend | " + " | ".join(f"B={s}" for s in ("1", "8", "64", "512", "4096"))]
    lines.append("|---|---|" + "---:|" * 5)
    for key, row in summary["environments"].items():
        task, backend = key.split("/")
        cells = [
            f"{row['sizes'][size]['transitions_per_second']['median'] / 1e6:.3g}"
            if size in row["sizes"]
            else ""
            for size in ("1", "8", "64", "512", "4096")
        ]
        lines.append(f"| {task} | {backend} | " + " | ".join(cells) + " |")
    lines += ["", "| Module | Import seconds, median [Q1, Q3] |", "|---|---:|"]
    for module, row in summary["imports"].items():
        q = row["import_seconds"]
        lines.append(f"| {module} | {q['median']:.3f} [{q['q1']:.3f}, {q['q3']:.3f}] |")
    lines += ["", "| Cohort | Solve s | " + "Phase shares |", "|---|---:|---|"]
    for key, row in summary["decomposition"].items():
        shares = ", ".join(
            f"{name} {share:.0%}" for name, share in row["phase_share"].items() if "/" not in name
        )
        lines.append(f"| {key} | {row['plain_solve_seconds']['median']:.2f} | {shares} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/bench"))
    parser.add_argument("--output", type=Path, default=Path("results/bench/summary.json"))
    args = parser.parse_args()
    summary = summarize(args.input)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(tables(summary))


if __name__ == "__main__":
    main()
