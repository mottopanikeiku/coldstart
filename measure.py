"""Run bounded final batches at nice 19, optionally after a stated time."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/final.json"))
    parser.add_argument("--chunks", help="Half-open chunk range; default is all chunks")
    parser.add_argument("--not-before", help="ISO timestamp with time zone")
    parser.add_argument("--output", type=Path, help="New directory for a replication")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["phase"] != "final":
        parser.error("measurement driver accepts only final plans")
    if args.not_before:
        timestamp = datetime.fromisoformat(args.not_before)
        if timestamp.tzinfo is None:
            parser.error("--not-before needs an explicit timezone")
        time.sleep(max(0, timestamp.timestamp() - time.time()))
    first, last = map(int, args.chunks.split(":")) if args.chunks else (0, len(plan["chunks"]))
    if not 0 <= first < last <= len(plan["chunks"]):
        parser.error("chunk range is outside the plan")
    output = args.output if args.output is not None else Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    log_directory = output.parent / "batches"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_name = args.plan.stem if args.output is None else f"{args.plan.stem}__{output.name}"
    for index in range(first, last):
        destination = log_directory / f"{log_name}__{index:03}.json"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark batch log {destination}")
        command = [
            "nice",
            "-n",
            "19",
            os.path.relpath(sys.executable),
            "sweep.py",
            "--plan",
            str(args.plan),
            "--chunk",
            str(index),
        ]
        if args.output is not None:
            command.extend(("--output", str(args.output)))
        begin = datetime.now().astimezone().isoformat()
        result = subprocess.run(command, env=os.environ.copy())
        destination.write_text(
            json.dumps(
                {
                    "command": command,
                    "started_including_queue": begin,
                    "finished_including_queue": datetime.now().astimezone().isoformat(),
                    "returncode": result.returncode,
                    "note": "Worker clocks start immediately before each child spawn.",
                },
                indent=2,
            )
            + "\n"
        )
        result.check_returncode()


if __name__ == "__main__":
    main()
