"""Fused C CartPole and Acrobot batches from classic.c, loaded through ctypes when built.

    .venv/bin/python -m control_clock.native build
    .venv/bin/python -m control_clock.native verify --seeds 1000000 --output gate.json

Building needs only the system C compiler (``cc``, or ``$CC``) and libm. One call resets any
number of slots and reproduces gymnasium.make(task).reset(seed=s) bit for bit; one call
validates and steps the whole batch with the NumPy batch's float64 operations. Without a
built library, ``make_vector(task, n, native=True)`` returns the NumPy batch.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from control_clock.vector_env import _ClassicBatch

SOURCE = Path(__file__).with_name("classic.c")
LIBRARY = SOURCE.with_name("libclassic.dylib" if sys.platform == "darwin" else "libclassic.so")
# No fused multiply-adds, and sin and cos stay separate libm calls instead of one sincos.
FLAGS = (
    "-O2",
    "-std=c11",
    "-shared",
    "-fPIC",
    "-ffp-contract=off",
    "-fno-builtin-sin",
    "-fno-builtin-cos",
    "-fno-builtin-sinf",
    "-fno-builtin-cosf",
)
TASKS = ("CartPole-v1", "Acrobot-v1")


class Unavailable(RuntimeError):
    """The library is not built, or was built from a different classic.c."""


def source_sha256(source: Path | None = None) -> str:
    return hashlib.sha256(Path(SOURCE if source is None else source).read_bytes()).hexdigest()


def build(source: Path | None = None, output: Path | None = None) -> Path:
    """Compile with the system compiler; the library records the digest of its source."""
    source = SOURCE if source is None else source
    output = LIBRARY if output is None else output
    compiler = os.environ.get("CC", "cc")
    command = [compiler, *FLAGS, f'-DCOLDSTART_SOURCE_SHA256="{source_sha256(source)}"']
    command += ["-o", str(output), str(source), "-lm"]
    subprocess.run(command, check=True)
    return Path(output)


@functools.cache
def _open(library: str, digest: str) -> ctypes.CDLL:
    if not Path(library).exists():
        raise Unavailable(f"{library} is not built; run python -m control_clock.native build")
    handle = ctypes.CDLL(library)
    handle.coldstart_source_sha256.restype = ctypes.c_char_p
    if handle.coldstart_source_sha256().decode() != digest:
        raise Unavailable(f"{library} was built from a different classic.c; rebuild it")
    pointer, size = ctypes.c_void_p, ctypes.c_int64
    for task in ("cartpole", "acrobot"):
        reset = getattr(handle, f"coldstart_{task}_reset")
        reset.argtypes = [size, *[pointer] * 6]
        reset.restype = None
        step = getattr(handle, f"coldstart_{task}_step")
        step.argtypes = [size, *[pointer] * 9, size]
        step.restype = ctypes.c_int
    handle.coldstart_numpy_trig_f32.argtypes = [size, pointer, pointer, ctypes.c_int]
    handle.coldstart_numpy_trig_f32.restype = None
    return handle


def load(library: Path | None = None, source: Path | None = None) -> ctypes.CDLL:
    """The library at LIBRARY, checked against SOURCE; raises Unavailable."""
    return _open(str(LIBRARY if library is None else library), source_sha256(source))


def available() -> bool:
    try:
        load()
    except Unavailable:
        return False
    return True


class _NativeBatch(_ClassicBatch):
    """The NumPy batch's interface and arrays; classic.c reads and writes them in place."""

    def __init__(self, task: str, num_envs: int, library: ctypes.CDLL):
        super().__init__(task, num_envs)
        name = task.split("-")[0].lower()
        self._reset_kernel = getattr(library, f"coldstart_{name}_reset")
        self._step_kernel = getattr(library, f"coldstart_{name}_step")
        self._mask_in = np.zeros(num_envs, dtype=bool)
        self._actions_in = np.zeros(num_envs, dtype=np.int64)
        self._active_in = np.zeros(num_envs, dtype=bool)
        self._outputs = (
            np.zeros(num_envs, dtype=np.float64),
            np.zeros(num_envs, dtype=bool),
            np.zeros(num_envs, dtype=bool),
        )
        # classic.c writes through these addresses, so every array is only filled in place.
        # Holding the arrays here keeps the memory alive even if an attribute is rebound.
        self._buffers = (self.state, self._obs, self.steps, self._done)
        self._buffers += (self._mask_in, self._actions_in, self._active_in, *self._outputs)
        self._shared = tuple(array.ctypes.data for array in self._buffers[:4])
        self._mask_address, self._actions_address, self._active_address = (
            array.ctypes.data for array in self._buffers[4:7]
        )
        self._output_addresses = tuple(array.ctypes.data for array in self._outputs)

    def _restart(self, mask_address, seeds: list[int]):
        try:
            seeds = np.array(seeds, dtype=np.uint64)
        except OverflowError:
            raise ValueError("seeds must be below 2**64 for the native batch") from None
        self._reset_kernel(self.num_envs, mask_address, seeds.ctypes.data, *self._shared)

    def reset(self, seeds: list[int]) -> np.ndarray:
        self._check_open()
        self._restart(None, self._seeds(seeds, self.num_envs))
        self._ready = True
        return self._observation()

    def reset_done(self, mask, seeds: list[int]) -> np.ndarray:
        """Reset selected slots, with one seed per True entry in index order."""
        self._check_ready()
        mask = np.asarray(mask)
        if mask.shape != (self.num_envs,) or mask.dtype.kind != "b":
            raise ValueError("mask must be a boolean array of shape (num_envs,)")
        seeds = self._seeds(seeds, int(np.count_nonzero(mask)))
        self._mask_in[...] = mask
        self._restart(self._mask_address, seeds)
        return self._observation()

    def step(self, actions, active=None):
        self._check_ready()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or actions.dtype.kind not in "iu":
            raise ValueError("actions must be an integer array of shape (num_envs,)")
        active_address = None
        if active is not None:
            active = np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype.kind != "b":
                raise ValueError("active must be a boolean array of shape (num_envs,)")
            self._active_in[...] = active
            active_address = self._active_address
        # Unsafe casting wraps unsigned values above 2**63 to negatives, which the kernel rejects.
        self._actions_in[...] = actions
        status = self._step_kernel(
            self.num_envs,
            self._actions_address,
            active_address,
            *self._shared,
            *self._output_addresses,
            self.max_steps,
        )
        if status == 1:
            raise ValueError(f"actions must lie in [0, {self.action_size})")
        if status == 2:
            raise RuntimeError("Reset finished active slots with reset_done before stepping")
        reward, terminated, truncated = (array.copy() for array in self._outputs)
        return self._observation(), reward, terminated, truncated


class CartPoleBatch(_NativeBatch):
    def __init__(self, num_envs: int, library: ctypes.CDLL | None = None):
        super().__init__("CartPole-v1", num_envs, load() if library is None else library)


class AcrobotBatch(_NativeBatch):
    def __init__(self, num_envs: int, library: ctypes.CDLL | None = None):
        super().__init__("Acrobot-v1", num_envs, load() if library is None else library)


def make(task: str, num_envs: int, library: ctypes.CDLL | None = None) -> _NativeBatch:
    """The fused batch; raises Unavailable instead of falling back."""
    if task == "CartPole-v1":
        return CartPoleBatch(num_envs, library)
    if task == "Acrobot-v1":
        return AcrobotBatch(num_envs, library)
    raise ValueError(f"No native batch for {task!r}")


def numpy_trig_f32(x: np.ndarray, cosine: bool, library: ctypes.CDLL | None = None):
    """classic.c's copy of NumPy's float32 cos (cosine=True) or sin."""
    library = load() if library is None else library
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    library.coldstart_numpy_trig_f32(x.size, x.ctypes.data, out.ctypes.data, int(cosine))
    return out


WIDE_SEEDS = (2**32 - 1, 2**32, 2**32 + 1, 2**63 - 1, 2**63, 2**64 - 1)


def gate_seeds(count: int, wide: int, seed: int = 2026) -> list[int]:
    """range(count), then fixed boundary seeds and random seeds that need two uint32 words."""
    random = np.random.default_rng(seed).integers(2**32, 2**64, size=wide, dtype=np.uint64)
    return [*range(count), *WIDE_SEEDS, *map(int, random)]


def reset_gate(task: str, seeds: list[int], library: ctypes.CDLL | None = None) -> dict:
    """One batched native reset against gymnasium.make(task).reset(seed=s) for every seed.

    Observations (float32) and internal states (float64; Acrobot's are float32 upstream) are
    compared as raw bits.
    """
    import gymnasium as gym

    batch = make(task, len(seeds), library)
    obs = batch.reset(seeds)
    state = batch.state
    reference = gym.make(task)
    expected_obs = np.empty_like(obs)
    expected_state = np.empty_like(state)
    try:
        for index, seed in enumerate(seeds):
            expected_obs[index], _ = reference.reset(seed=seed)
            expected_state[index] = reference.unwrapped.state
    finally:
        reference.close()
        batch.close()
    obs_rows = np.any(obs.view(np.uint32) != expected_obs.view(np.uint32), axis=1)
    state_rows = np.any(state.view(np.uint64) != expected_state.view(np.uint64), axis=1)
    differing = np.flatnonzero(obs_rows | state_rows)
    return {
        "task": task,
        "seeds": len(seeds),
        "observation_mismatches": int(obs_rows.sum()),
        "state_mismatches": int(state_rows.sum()),
        "first_mismatching_seeds": [seeds[index] for index in differing[:5]],
        "passed": differing.size == 0,
    }


def describe(library: Path | None = None) -> dict | None:
    """Provenance for benchmark records: digests, compiler and flags; None when unavailable."""
    library = LIBRARY if library is None else Path(library)
    try:
        load(library)
    except Unavailable:
        return None
    compiler = os.environ.get("CC", "cc")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True)
    return {
        "library": library.name,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_sha256": source_sha256(),
        "compiler": version.stdout.splitlines()[0] if version.stdout else None,
        "flags": list(FLAGS),
    }


def float32_trig_targets() -> dict:
    """The SIMD target NumPy dispatches float32 sin and cos to on this CPU."""
    from numpy.lib.introspect import opt_func_info

    info = opt_func_info(func_name="^(sin|cos)$", signature="float32")
    return {name: next(iter(rows.values()))["current"] for name, rows in sorted(info.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    verify = commands.add_parser("verify")
    verify.add_argument("--seeds", type=int, default=1_000_000, help="Contiguous seeds from 0")
    verify.add_argument("--wide", type=int, default=10_000, help="Extra random 64-bit seeds")
    verify.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        print(build())
        return
    from importlib.metadata import version

    seeds = gate_seeds(args.seeds, args.wide)
    record = {
        "command": ["python", "-m", "control_clock.native", *sys.argv[1:]],
        "contiguous_seeds": [0, args.seeds],
        "boundary_seeds": list(WIDE_SEEDS),
        "random_wide_seeds": args.wide,
        "numpy": np.__version__,
        "gymnasium": version("gymnasium"),
        "numpy_float32_trig_target": float32_trig_targets(),
        "native": describe(),
        "tasks": [reset_gate(task, seeds) for task in TASKS],
    }
    text = json.dumps(record, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if not all(row["passed"] for row in record["tasks"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
