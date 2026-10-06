import json
import sys
from types import SimpleNamespace

import pytest

import measure
import sweep


def plan_file(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "phase": "final",
                "output": str(tmp_path / "original"),
                "chunks": [[{"task": "CartPole-v1", "method": "cem", "seeds": "0:1"}]],
            }
        )
    )
    return path


def test_sweep_routes_replication_without_modifying_plan(monkeypatch, tmp_path):
    plan = plan_file(tmp_path)
    output = tmp_path / "replication"
    commands = []
    monkeypatch.setattr(
        sys, "argv", ["sweep.py", "--plan", str(plan), "--chunk", "0", "--output", str(output)]
    )
    monkeypatch.setattr(sweep.subprocess, "run", lambda command, **kwargs: commands.append(command))
    sweep.main()
    command = commands[0]
    assert command[command.index("--output") + 1] == str(output)
    assert json.loads(plan.read_text())["output"] == str(tmp_path / "original")


def test_measure_preserves_original_logs_and_uses_nice(monkeypatch, tmp_path):
    plan = plan_file(tmp_path)
    output = tmp_path / "replication"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, check_returncode=lambda: None)

    monkeypatch.setattr(sys, "argv", ["measure.py", "--plan", str(plan), "--output", str(output)])
    monkeypatch.setattr(measure.subprocess, "run", run)
    measure.main()
    command = commands[0]
    assert command[:3] == ["nice", "-n", "19"]
    assert command[command.index("--output") + 1] == str(output)
    log = tmp_path / "batches/plan__replication__000.json"
    assert json.loads(log.read_text())["command"] == command
    assert not (tmp_path / "original").exists()
    with pytest.raises(FileExistsError):
        measure.main()
    assert len(commands) == 1
