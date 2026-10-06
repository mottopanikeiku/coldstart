"""Run an explicit JSON experiment plan; a chunk contains bounded seed ranges."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--output", type=Path, help="Override the plan's output for a replication")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    chunk = plan["chunks"][args.chunk]
    for experiment in chunk:
        command = [
            sys.executable,
            "run.py",
            "--phase",
            plan["phase"],
            "--output",
            str(args.output) if args.output is not None else plan["output"],
            "--task",
            experiment["task"],
            "--method",
            experiment["method"],
            "--seeds",
            experiment["seeds"],
            "--variant",
            experiment.get("variant", "default"),
        ]
        if "limit" in experiment:
            command.extend(("--limit", str(experiment["limit"])))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
