"""The fused C batch: bitwise resets, bitwise steps, the NumPy fallback and corrupted kernels."""

import os
import shutil

import numpy as np
import pytest

from control_clock import backends, native
from control_clock.vector_env import AcrobotBatch, CartPoleBatch, make_vector

needs_library = pytest.mark.skipif(
    not native.available(), reason="needs python -m control_clock.native build"
)
needs_compiler = pytest.mark.skipif(
    shutil.which(os.environ.get("CC", "cc")) is None, reason="needs a C compiler"
)


@needs_library
@pytest.mark.parametrize("task", native.TASKS)
def test_one_batched_reset_matches_gymnasium_bit_for_bit(task):
    # The 1M-seed version of this gate is `python -m control_clock.native verify`.
    gate = native.reset_gate(task, native.gate_seeds(3000, 300))
    assert gate["passed"], gate
    assert gate["seeds"] == 3000 + len(native.WIDE_SEEDS) + 300


@needs_library
@pytest.mark.parametrize("cosine", [True, False])
def test_float32_trigonometry_matches_numpy_bit_for_bit(cosine):
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2**32, size=1_000_000, dtype=np.uint64).astype(np.uint32)
    special = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, 71476.0625, 117435.992, np.pi / 4]
    x = np.concatenate(
        [
            bits.view(np.float32),
            rng.uniform(-0.1, 0.1, size=200_000).astype(np.float32),
            rng.uniform(-2e5, 2e5, size=200_000).astype(np.float32),
            np.asarray(special, dtype=np.float32),
        ]
    )
    with np.errstate(invalid="ignore"):
        expected = (np.cos if cosine else np.sin)(x)
    actual = native.numpy_trig_f32(x, cosine)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@needs_library
@pytest.mark.parametrize("task", native.TASKS)
def test_long_trace_matches_the_numpy_batch_bit_for_bit(task):
    trace = backends.native_trace(task, slots=32, steps=1500, seed=11)
    assert trace["native_passed"] and trace["native_steps"] == 1500, trace
    assert trace["native_episode_ends"] > 0


@needs_library
def test_seeds_must_fit_in_64_bits():
    batch = native.make("CartPole-v1", 2)
    batch.reset([0, 2**64 - 1])
    with pytest.raises(ValueError, match="seeds"):
        batch.reset([0, 2**64])


@needs_library
def test_make_vector_uses_the_library_only_when_asked():
    assert isinstance(make_vector("CartPole-v1", 2, native=True), native.CartPoleBatch)
    assert isinstance(make_vector("Acrobot-v1", 2, native=True), native.AcrobotBatch)
    assert type(make_vector("CartPole-v1", 2)) is CartPoleBatch


def test_make_vector_falls_back_to_the_numpy_batch_without_the_library(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "LIBRARY", tmp_path / "libclassic.so")
    assert not native.available()
    assert type(make_vector("CartPole-v1", 2, native=True)) is CartPoleBatch
    assert type(make_vector("Acrobot-v1", 2, native=True)) is AcrobotBatch
    # The benchmark's native backend never falls back, so it cannot time NumPy by mistake.
    with pytest.raises(native.Unavailable, match="not built"):
        backends.make("native", "CartPole-v1", 2, 0)


@needs_compiler
def test_a_library_built_from_other_source_is_refused(tmp_path):
    library = native.build(native.SOURCE, tmp_path / "libclassic.so")
    assert native.load(library) is not None
    edited = tmp_path / "classic.c"
    edited.write_text(native.SOURCE.read_text() + "/* edited */\n")
    with pytest.raises(native.Unavailable, match="different classic.c"):
        native.load(library, edited)


# Each mutant changes one expression of classic.c. Reset mutants must fail the bitwise reset
# gate. Dynamics mutants must fail the benchmark gate, whose native_trace compares raw bytes
# with the NumPy batch, and leave resets intact. The unchanged copy must pass both.
MUTANTS = {
    "unchanged": ("CartPole-v1", None, "#define MULT_B 0x58f38dedu", "#define MULT_B 0x58f38dedu"),
    "seed hash multiplier": (
        "CartPole-v1",
        "reset",
        "#define MULT_B 0x58f38dedu",
        "#define MULT_B 0x58f38deeu",
    ),
    "52-bit doubles": (
        "Acrobot-v1",
        "reset",
        "(pcg64_next64(rng) >> 11)",
        "(pcg64_next64(rng) >> 12)",
    ),
    "libm cosf in the reset observation": (
        "Acrobot-v1",
        "reset",
        "o[0] = numpy_sincosf(s[0], 1);",
        "o[0] = cosf(s[0]);",
    ),
    "CartPole gravity": (
        "CartPole-v1",
        "bench",
        "#define CARTPOLE_GRAVITY 9.8",
        "#define CARTPOLE_GRAVITY 9.81",
    ),
    "Acrobot RK4 weight": (
        "Acrobot-v1",
        "bench",
        "2 * k3[j] + k4[j]",
        "2 * k3[j] + 1.0000001 * k4[j]",
    ),
    "Acrobot summation order": (
        "Acrobot-v1",
        "bench",
        "(k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])",
        "(k1[j] + (2 * k2[j] + 2 * k3[j]) + k4[j])",
    ),
}


@needs_compiler
@pytest.mark.parametrize("name", MUTANTS)
def test_gates_reject_a_corrupted_kernel(name, tmp_path, monkeypatch):
    task, catches, original, replacement = MUTANTS[name]
    text = native.SOURCE.read_text()
    assert text.count(original) == 1
    source = tmp_path / "classic.c"
    source.write_text(text.replace(original, replacement))
    library = native.build(source, tmp_path / "libclassic.so")
    monkeypatch.setattr(native, "SOURCE", source)
    monkeypatch.setattr(native, "LIBRARY", library)
    reset = native.reset_gate(task, native.gate_seeds(2000, 200))
    assert reset["passed"] is (catches != "reset"), reset
    if catches != "reset":
        gate = backends.gate("native", task, slots=16, steps=32)
        assert gate["passed"] is (catches is None), gate
