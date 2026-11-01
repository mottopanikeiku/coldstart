# Batch environments

[`vector_env.py`](../control_clock/vector_env.py) implements NumPy batches of Gymnasium 1.2.1's default CartPole-v1 and Acrobot-v1. LunarLander uses a loop over `gymnasium.make` environments and Box2D physics. Evaluation always uses Gymnasium.

The optional [C batch](#fused-c-batch) implements the same two classic-control environments. Only the systems benchmark uses it.

## Interface and episode handling

`make_vector(task, num_envs, native=False)` exposes `num_envs`, `observation_size` and `action_size`, plus:

| Method | Behavior |
|---|---|
| `reset(seeds)` | One nonnegative Python integer seed per slot |
| `step(actions, active=None)` | Integer actions of shape `(num_envs,)`; returns observations, rewards, terminated and truncated |
| `reset_done(mask, seeds)` | Boolean mask and one seed per selected slot, in ascending slot order; returns the full observation batch |
| `close()` | Releases Gymnasium environments and prevents further use |

Observations have shape `(num_envs, observation_size)` and dtype `float32`; flags are boolean. Observation arrays are snapshots, so later resets cannot overwrite a rollout buffer's observations.

Resets are explicit. A terminal transition returns its final observation. Stepping a finished active slot raises an error. `reset_done` clears only the selected counters and done status; selected slots can be restarted before they finish. An empty selection accepts `[]`.

Policy search uses the boolean `active` mask to leave finished candidates waiting while others run. Inactive slots keep their observations, counters and internal done status. Their returned reward is zero and both flags are false. Omitting the mask steps every slot.

Truncation occurs at 500 steps for the classic tasks and 1,000 for LunarLander. Termination and truncation can both be true. The API validates action shapes, integer values and bounds, mask shapes/dtypes, and seed counts and values.

## Numerical behavior

Each classic reset draws four uniforms from a newly seeded `np.random.default_rng(seed)`, matching Gymnasium's PCG64. CartPole resets in `[-0.05, 0.05]`, uses explicit Euler integration with float64 state and rewards every step with one, including the terminal transition.

Acrobot resets in `[-0.1, 0.1]`. Gymnasium rounds its initial state to float32; the NumPy batch does the same, then promotes it to float64. Reset observations use float32 trigonometry; later observations are rounded from float64 dynamics.

Actions apply torque `−1`, `0` or `1`. The default book equations use one RK4 step of duration `0.2`, wrap angles into inclusive `[-pi, pi]` and clip velocities to `[-4*pi, 4*pi]` and `[-9*pi, 9*pi]`. Reward is zero when `-cos(theta1)-cos(theta1+theta2) > 1`, otherwise minus one. The API supports the default equations and reset ranges, without torque noise. LunarLander keeps Box2D's native precision.

[`tests/test_equivalence.py`](../tests/test_equivalence.py) compares observations and internal states with `atol=2e-6, rtol=2e-7`. The allowed discrepancy is `2e-6 + 2e-7 * abs(reference)`, under `8e-6` at Acrobot's largest velocity. Rewards, termination and truncation must match exactly.

The tests cover random action sequences, partial resets, active masks, failure thresholds, wrapping, velocity clipping, terminal rewards, full episode limits and simultaneous termination/truncation. Boundary fixtures set matching internal states in both implementations. Time-limit tests hold the state near equilibrium without resetting its elapsed counter.

```sh
uv run pytest tests/test_equivalence.py
```

## Fused C batch

[`classic.c`](../control_clock/classic.c) implements CartPole and Acrobot in plain C, loaded through ctypes by [`native.py`](../control_clock/native.py). NumPy arrays hold the state; C reads and writes them in place. One call checks actions and steps the batch. Another resets any set of slots and clears their counters.

`python -m control_clock.native build` compiles it with `cc` (or `$CC`) and libm. The library stores its source SHA-256; the loader rejects a build from a different `classic.c`. `make_vector(task, num_envs, native=True)` returns the C batch when built and NumPy otherwise. The benchmark requires the C library explicitly.

### Reset implementation

I followed NumPy 2.2.6's seeding code so one batched call reproduces Gymnasium's per-seed resets:

1. `SeedSequence` hashes the seed into four 32-bit words.
2. `generate_state` draws four 64-bit words.
3. PCG64 takes its 128-bit state and stream from those words.
4. Each uniform double uses `(u64 >> 11) * 2**-53`, then `low + (high - low) * d`.
5. Acrobot rounds to float32 and computes float32 cosine and sine.

Seeds must be below `2**64`. NumPy's AVX2/FMA3 float32 trigonometry uses Cody–Waite reduction and polynomial approximations. The C code copies that path, including a multiply-add fused by GCC in the manylinux wheel. The fused form matters at large magnitudes; both forms agree over Acrobot's reset range.

### Step implementation and tests

C evaluates the NumPy batch's float64 expressions in the same order, using Gymnasium's constants. The build disables multiply-add contraction and keeps `sin` and `cos` separate, matching NumPy's float64 libm calls.

[`tests/test_native.py`](../tests/test_native.py) checks:

- Reset observations and internal states against Gymnasium on 3,306 seeds per task. The larger CLI check matched all 1,010,006 seeds per task ([result](../results/bench/native-reset-gate.json)).
- Float32 sine and cosine against NumPy on 1.4 million inputs, including NaN, infinities and large inputs.
- 1,500 steps of 32 slots against the NumPy batch, including inactive slots and restarts. Observations, rewards, flags, counters and internal states must match bit for bit.
- Six modified copies of the C source, changing reset constants, float conversion or dynamics. The reset and step checks must reject them.

When C is built, the full Gymnasium equivalence suite runs on both batches. The C-to-Gymnasium step comparison uses the same tolerances as NumPy; C-to-NumPy traces match bit for bit.

The C checks have run on x86-64 Linux with glibc. Its float32 trigonometry assumes NumPy's AVX2/FMA3 path. A CPU using libm `sinf` and `cosf` instead may produce last-bit reset differences.

## Gymnax training in the separate L4 comparison

JAX PPO trains with Gymnax 0.0.9. [`test_jax.py`](../tests/test_jax.py) compares 100 fixed states per task and every action with Gymnasium 1.2.1. Rewards and done flags match exactly away from time limits. Float64 equations use `atol=2e-6, rtol=2e-7`; training-precision float32 uses `atol=1e-4, rtol=1e-5`. Near-limit Acrobot velocities had a float32 difference of `4.72e-5`.

The suite also checks reset ranges, the 500-step limit, CartPole's strict position threshold, terminal rewards and Acrobot's wrapping endpoint. Both tasks use the default physics and 500-step training episodes.

The main differences are:

- Gymnax uses float32 internal dynamics here; Gymnasium mostly uses float64.
- JAX's PRNG gives different initial states from NumPy's PCG64 for the same integer seed.
- Gymnax resets on the terminal step and combines termination and time limits into one `done`. The JAX PPO masks both in GAE.
- Gymnax wraps Acrobot angles into `[-pi, pi)`; Gymnasium includes both endpoints. At `+pi` the internal angles differ by a rotation, with equivalent physical positions up to rounding.

All qualification evaluations still use Gymnasium. Sources: [Gymnax](https://github.com/RobertTLange/gymnax), its [0.0.9 release](https://pypi.org/project/gymnax/0.0.9/), and the Gymnasium sources below.

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

Gymnasium is MIT-licensed. Its Acrobot source additionally credits the RLPy
implementation under BSD-3-Clause: Alborz Geramifard, Robert H. Klein,
Christoph Dann, William Dabney and Jonathan P. How; author Christoph Dann.
Both notices are retained below. The NumPy batch layout, explicit-reset API,
active masks, equivalence tests and the C batch are additions in this repository.

The C batch's seeding follows
[NumPy 2.2.6's random sources](https://github.com/numpy/numpy/tree/v2.2.6/numpy/random):
`bit_generator.pyx` (SeedSequence), `src/pcg64/` (PCG64), `_common.pxd`, `_generator.pyx`
and `src/distributions/distributions.c` (uniform). Its float32 sine and cosine follow
[`loops_trigonometric.dispatch.cpp`](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_trigonometric.dispatch.cpp).
NumPy is BSD-3-Clause ([`third_party/NumPy-LICENSE`](../third_party/NumPy-LICENSE)).
NumPy's SeedSequence derives from Melissa E. O'Neill's C++11 `seed_seq` alternative, and
`bit_generator.pyx` carries its own MIT notice (Copyright (c) 2015 Melissa E. O'Neill,
Copyright (c) 2019 NumPy Developers), retained in
[`third_party/SeedSequence-LICENSE`](../third_party/SeedSequence-LICENSE). Its PCG64 comes
from Melissa O'Neill's PCG library under the MIT license
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
