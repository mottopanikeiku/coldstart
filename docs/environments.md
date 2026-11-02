# Batch environments

[`vector_env.py`](../control_clock/vector_env.py) has NumPy batches of Gymnasium 1.2.1's default CartPole-v1 and Acrobot-v1, while LunarLander simply loops over `gymnasium.make` environments. The optional [C batch](#fused-c-batch) covers the same two classic tasks but is used only by the systems benchmark, and evaluation always uses real Gymnasium.

## Interface

`make_vector(task, num_envs, native=False)` returns an object with `reset(seeds)`, `step(actions, active=None)`, `reset_done(mask, seeds)` and `close()`. Observations are `float32` arrays of shape `(num_envs, observation_size)`, copied so later resets can't overwrite a rollout buffer.

- Resets are explicit: a terminal step returns its final observation, and stepping a finished slot raises.
- `active` is a boolean mask that leaves finished candidates waiting during policy search: inactive slots keep their state and return zero reward with both flags false.
- Episodes truncate at 500 steps (classic tasks) or 1,000 (LunarLander), and termination and truncation can both be true.
- Action shapes, integer values, masks and seed counts are validated.

## Numerical behavior

A classic reset draws four uniforms from a freshly seeded `np.random.default_rng(seed)`, matching Gymnasium's PCG64. CartPole resets in [−0.05, 0.05], integrates with explicit Euler in float64, and pays 1 per step including the terminal one. Acrobot, which uses the default equations with no torque noise, resets in [−0.1, 0.1] rounded to float32 and then promoted. It takes one RK4 step of 0.2, wraps angles into the inclusive range [−π, π], clips velocities to ±4π and ±9π, and rewards 0 once `-cos(θ1) - cos(θ1 + θ2) > 1`, else −1.

[`tests/test_equivalence.py`](../tests/test_equivalence.py) compares observations and states against Gymnasium with `atol=2e-6, rtol=2e-7`; rewards, termination and truncation must match exactly. It covers random action sequences, partial resets, active masks, failure thresholds, wrapping, velocity clipping, terminal rewards, episode limits and simultaneous termination and truncation.

## Fused C batch

[`classic.c`](../control_clock/classic.c) implements both tasks in plain C and is loaded through ctypes by [`native.py`](../control_clock/native.py), with the state held in NumPy arrays that C reads and writes in place. `python -m control_clock.native build` compiles it with `cc` (or `$CC`) and libm. The library stores its source hash so the loader can reject a mismatch, and without a build `native=True` falls back to NumPy.

To make one batched call reproduce Gymnasium's per-seed resets for any seed below 2^64, the C code follows NumPy 2.2.6's seeding: `SeedSequence` hashes the seed into four 32-bit words, `generate_state` draws four 64-bit words, PCG64 takes its state and stream from them, and each uniform is `(u64 >> 11) * 2**-53`. Acrobot then rounds to float32 and computes float32 sine and cosine by copying NumPy's AVX2/FMA3 path (Cody–Waite reduction plus polynomials). Steps evaluate the NumPy batch's float64 expressions in the same order with FMA contraction off.

[`tests/test_native.py`](../tests/test_native.py) checks resets against Gymnasium on 3,306 seeds per task, float32 sin/cos against NumPy on 1.4 million inputs, 1,500 steps of 32 slots bit for bit against the NumPy batch, and that six mutated copies of the C source (reset constants, float conversion, dynamics) are rejected. The command-line gate compared all 1,010,006 seeds per task ([result](../results/bench/native-reset-gate.json)):

```sh
uv run python -m control_clock.native build
uv run python -m control_clock.native verify --seeds 1000000
```

The C checks have only run on x86-64 Linux with glibc, and on a CPU where NumPy uses libm `sinf`/`cosf` instead, a reset could differ in the last bit.

## Gymnax in the GPU comparison

JAX PPO trains on Gymnax 0.0.9, which [`tests/test_jax.py`](../tests/test_jax.py) compares against Gymnasium on 100 fixed states per task and every action. Rewards and done flags match exactly away from time limits, and float32 state differences stay under `1e-4`. Gymnax still differs from Gymnasium in float32 internals, PRNG (different initial states for the same integer seed), a single `done` flag for termination and time limit, and Acrobot angle wrapping into [−π, π), but every qualification evaluation runs on Gymnasium.

## Sources and licenses

Dynamics, integration order, reset behavior and episode semantics are adapted
from the following pinned primary sources:

- [Gymnasium 1.2.1 CartPole](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/classic_control/cartpole.py),
  originally based on Rich Sutton's `pole.c`.
- [Gymnasium 1.2.1 Acrobot](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/classic_control/acrobot.py),
  including its RK4, inclusive wrap, velocity bounds and default book equations.
- [Gymnasium 1.2.1 seeding](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/utils/seeding.py)
  and [task registrations](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/__init__.py).
- [Gymnasium 1.2.1 LunarLander](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/gymnasium/envs/box2d/lunar_lander.py)
  is used directly through Gymnasium, not copied.

Gymnasium is MIT-licensed, and its Acrobot source also credits the RLPy
implementation, which is BSD-3-Clause and by Alborz Geramifard, Robert H. Klein,
Christoph Dann, William Dabney and Jonathan P. How, with Christoph Dann as author.
The NumPy batch layout, explicit-reset API, active masks, equivalence tests and
the C batch are additions in this repository, and both upstream notices are
retained below.

The C batch's seeding follows
[NumPy 2.2.6's random sources](https://github.com/numpy/numpy/tree/v2.2.6/numpy/random):
`bit_generator.pyx` (SeedSequence), `src/pcg64/` (PCG64), `_common.pxd`, `_generator.pyx`
and `src/distributions/distributions.c` (uniform), while its float32 sine and cosine follow
[`loops_trigonometric.dispatch.cpp`](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_trigonometric.dispatch.cpp).
NumPy is BSD-3-Clause ([`third_party/NumPy-LICENSE`](../third_party/NumPy-LICENSE)), but two
of the pieces used here carry their own MIT notices. SeedSequence derives from Melissa E.
O'Neill's C++11 `seed_seq` alternative, and `bit_generator.pyx` carries its notice
(Copyright (c) 2015 Melissa E. O'Neill, Copyright (c) 2019 NumPy Developers), retained in
[`third_party/SeedSequence-LICENSE`](../third_party/SeedSequence-LICENSE). PCG64 comes from
Melissa O'Neill's PCG library under the MIT license
([`third_party/PCG64-LICENSE`](../third_party/PCG64-LICENSE)).

### Gymnasium MIT notice

[Upstream license](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.1/LICENSE):

The MIT License

Copyright (c) 2016 OpenAI  
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

### Acrobot / RLPy BSD notice

[Upstream license](https://github.com/rlpy/rlpy/blob/master/LICENSE.txt):

Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, Christoph Dann,
William Dabney and Jonathan P. How  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of ACL nor the names of its contributors may be used to
endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
