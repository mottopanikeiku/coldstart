"""Run GPU-only extra comparisons: modal run modal_app.py --phase pilot|final."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "gpu-requirements.txt"))
    .pip_install("jax[cuda12]==0.6.2", "pytest==8.4.2")
    .add_local_dir(
        ROOT,
        remote_path="/repo",
        ignore=[".git", ".venv", ".jax-venv", "__pycache__", ".pytest_cache", ".ruff_cache"],
    )
)
app = modal.App("control-clock-gpu")


def run_comparison(phase):
    import os
    import subprocess
    import sys
    import time

    os.chdir("/repo")
    sys.path.insert(0, "/repo")
    started = time.perf_counter()
    checks = subprocess.run(
        ["python", "-m", "pytest", "-q", "-s", "tests/test_jax.py"],
        capture_output=True,
        text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
    )
    if checks.returncode:
        raise RuntimeError(checks.stdout + checks.stderr)
    from gpu_run import launch

    records = []
    for task in ("CartPole-v1", "Acrobot-v1"):
        for seed in [100] if phase == "pilot" else range(3):
            record = launch(
                task,
                seed,
                f"/tmp/gpu-{phase}",
                phase,
                limit=90 if phase == "pilot" else None,
            )
            records.append(record)
    seconds = time.perf_counter() - started
    hourly_rate = 0.7992 + 2 * 0.047160 + 4 * 0.007992
    return {
        "phase": phase,
        "hardware": "NVIDIA L4",
        "cpu_cores": 2,
        "host_memory_gib": 4,
        "container_seconds": seconds,
        "container_cost_estimate_usd": seconds / 3600 * hourly_rate,
        "hourly_rate_usd": hourly_rate,
        "cost_scope": "GPU, requested host CPU and RAM; excludes image build and client startup",
        "checks": {"command": "python -m pytest -q -s tests/test_jax.py", "stdout": checks.stdout},
        "records": records,
    }


@app.function(image=image, gpu="L4", cpu=2, memory=4096, timeout=300, max_containers=1)
def pilot():
    return run_comparison("pilot")


@app.function(image=image, gpu="L4", cpu=2, memory=4096, timeout=1380, max_containers=1)
def final():
    return run_comparison("final")


@app.local_entrypoint()
def main(phase: str = "pilot", output: str = "results/gpu-replication"):
    if phase not in ("pilot", "final"):
        raise ValueError("phase must be pilot or final")
    payload = pilot.remote() if phase == "pilot" else final.remote()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for record in payload.pop("records"):
        path = destination / f"{record['task']}__jax-ppo__{record['seed']:03}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(record, indent=2) + "\n")
    metadata = destination / "run.json"
    if metadata.exists():
        raise FileExistsError(metadata)
    metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
