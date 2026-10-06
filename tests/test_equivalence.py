"""The slow oracle is pinned Gymnasium, not a second copy of the equations."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.classic_control.acrobot import wrap

from control_clock.vector_env import GymnasiumBatch, _wrap_angles, make_vector

# Float32 observations permit approximately one ULP; rewards/flags remain exact.
OBS_ATOL = 2e-6
OBS_RTOL = 2e-7
CLASSIC_TASKS = ["CartPole-v1", "Acrobot-v1"]
ALL_TASKS = [*CLASSIC_TASKS, "LunarLander-v3"]


def assert_observations(actual, expected):
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=OBS_ATOL, rtol=OBS_RTOL)


def assert_internal_states(batch, references):
    expected = np.stack([env.unwrapped.state for env in references])
    np.testing.assert_allclose(batch.state, expected, atol=OBS_ATOL, rtol=OBS_RTOL)


def assert_transition(actual, expected):
    assert_observations(actual[0], expected[0])
    for actual_array, expected_array in zip(actual[1:], expected[1:], strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert actual[2].dtype == actual[3].dtype == np.dtype(bool)


def reference_step(envs, actions):
    transitions = [env.step(int(action)) for env, action in zip(envs, actions, strict=True)]
    return tuple(np.asarray([transition[j] for transition in transitions]) for j in range(4))


@pytest.mark.parametrize("task", CLASSIC_TASKS)
@pytest.mark.parametrize("sequence_seed", [0, 42, 2026])
def test_many_seeded_random_sequences(task, sequence_seed):
    rng = np.random.default_rng(sequence_seed)
    batch = make_vector(task, 4)
    references = [gym.make(task) for _ in range(4)]
    try:
        seeds = rng.integers(0, 1_000_000, size=4).tolist()
        obs = batch.reset(seeds)
        expected = np.stack([env.reset(seed=s)[0] for env, s in zip(references, seeds)])
        assert_observations(obs, expected)
        assert_internal_states(batch, references)
        assert batch.state.dtype == np.float64
        for _ in range(600):
            actions = rng.integers(0, batch.action_size, size=4)
            actual = batch.step(actions)
            expected = reference_step(references, actions)
            assert_transition(actual, expected)
            assert_internal_states(batch, references)
            done = actual[2] | actual[3]
            if np.any(done):
                final = actual[0].copy()
                seeds = rng.integers(0, 1_000_000, size=int(done.sum())).tolist()
                for index, seed in zip(np.flatnonzero(done), seeds, strict=True):
                    expected[0][index] = references[index].reset(seed=seed)[0]
                reset_obs = batch.reset_done(done, seeds)
                assert_observations(reset_obs, expected[0])
                np.testing.assert_array_equal(reset_obs[~done], final[~done])
                assert_internal_states(batch, references)
                # Reset must not overwrite an already returned final observation.
                np.testing.assert_array_equal(actual[0], final)
    finally:
        batch.close()
        for env in references:
            env.close()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_numpy_or_fallback_against_gymnasium_batch(task):
    batch = make_vector(task, 3)
    reference = GymnasiumBatch(task, 3)
    rng = np.random.default_rng(51)
    try:
        assert_observations(batch.reset([11, 22, 33]), reference.reset([11, 22, 33]))
        for step in range(150):
            actions = rng.integers(0, batch.action_size, size=3)
            actual, expected = batch.step(actions), reference.step(actions)
            assert_transition(actual, expected)
            done = actual[2] | actual[3]
            if np.any(done):
                seeds = [step * 3 + int(i) for i in np.flatnonzero(done)]
                assert_observations(
                    batch.reset_done(done, seeds), reference.reset_done(done, seeds)
                )
    finally:
        batch.close()
        reference.close()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_partial_reset_and_seed_order(task):
    batch = make_vector(task, 4)
    reference = GymnasiumBatch(task, 4)
    try:
        batch.reset([10, 20, 30, 40])
        reference.reset([10, 20, 30, 40])
        transition = batch.step(np.array([0, 1, 0, 1]))
        assert_transition(transition, reference.step(np.array([0, 1, 0, 1])))
        before = transition[0]
        mask = np.array([False, True, False, True])
        actual = batch.reset_done(mask, [101, 202])
        assert_observations(actual, reference.reset_done(mask, [101, 202]))
        np.testing.assert_array_equal(actual[~mask], before[~mask])
        np.testing.assert_array_equal(batch.steps, [1, 0, 1, 0])
        np.testing.assert_array_equal(batch.reset_done(np.zeros(4, dtype=bool), []), actual)
        # Repeating identical per-slot seeds must reproduce reset exactly.
        np.testing.assert_array_equal(batch.reset_done(mask, [101, 202]), actual)
        actions = np.array([0, 1, 0, 1])
        assert_transition(batch.step(actions), reference.step(actions))
    finally:
        batch.close()
        reference.close()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_active_mask_against_independent_gymnasium(task):
    batch = make_vector(task, 3)
    references = [gym.make(task) for _ in range(3)]
    rng = np.random.default_rng(57)
    try:
        obs = batch.reset([17, 18, 19])
        expected_obs = np.stack([env.reset(seed=s)[0] for env, s in zip(references, [17, 18, 19])])
        elapsed = np.zeros(3, dtype=np.int64)
        done = np.zeros(3, dtype=bool)
        for step in range(100):
            active = (rng.random(3) > 0.4) & ~done
            if step == 0:
                active[:] = False
            actions = rng.integers(batch.action_size, size=3)
            rewards = np.zeros(3)
            terminated = np.zeros(3, dtype=bool)
            truncated = np.zeros(3, dtype=bool)
            for index in np.flatnonzero(active):
                transition = references[index].step(int(actions[index]))
                expected_obs[index] = transition[0]
                rewards[index] = transition[1]
                terminated[index] = transition[2]
                truncated[index] = transition[3]
            actual = batch.step(actions, active=active)
            assert_transition(actual, (expected_obs, rewards, terminated, truncated))
            np.testing.assert_array_equal(actual[0][~active], obs[~active])
            elapsed += active
            np.testing.assert_array_equal(batch.steps, elapsed)
            done |= terminated | truncated
            obs = actual[0]
        # Even finished slots can remain inactive indefinitely, without post-done steps.
        frozen = batch.step(np.zeros(3, dtype=int), active=np.zeros(3, dtype=bool))
        np.testing.assert_array_equal(frozen[0], obs)
        for array in frozen[1:]:
            assert not np.any(array)
    finally:
        batch.close()
        for env in references:
            env.close()


@pytest.mark.parametrize("task", CLASSIC_TASKS)
def test_full_500_step_time_limit(task):
    batch = make_vector(task, 1)
    reference = gym.make(task)
    try:
        batch.reset([1])
        reference.reset(seed=1)
        for elapsed in range(1, 501):
            # Isolate TimeLimit from failure dynamics. Acrobot stays down with zero torque;
            # CartPole state is set down each step WITHOUT resetting its elapsed counter.
            batch.state[0] = 0.0
            reference.unwrapped.state = np.zeros(4, dtype=np.float64)
            action = 0 if task == "CartPole-v1" else 1
            actual = batch.step(np.array([action]))
            expected = reference_step([reference], [action])
            assert_transition(actual, expected)
            assert not actual[2][0]
            assert actual[3][0] == (elapsed == 500)
        np.testing.assert_array_equal(batch.steps, [500])
        with pytest.raises(RuntimeError, match="finished"):
            batch.step(np.array([action]))
        batch.reset_done(np.array([True]), [2])
        assert batch.steps[0] == 0
        assert not batch.step(np.array([action]))[3][0]
    finally:
        batch.close()
        reference.close()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_time_limit_and_terminal_observations(task):
    batch = make_vector(task, 1)
    reference = gym.make(task)
    try:
        batch.reset([71])
        reference.reset(seed=71)
        cap = 1000 if task == "LunarLander-v3" else 500
        batch.steps[0] = cap - 1
        reference._elapsed_steps = cap - 1
        if task == "CartPole-v1":
            state = np.array([2.5, 0.0, 0.0, 0.0])
        elif task == "Acrobot-v1":
            state = np.array([np.pi, 0.0, 0.0, 0.0])
        else:
            state = None
            batch.envs[0]._elapsed_steps = cap - 1
        if state is not None:
            batch.state[0] = state
            reference.unwrapped.state = state.copy()
        actual = batch.step(np.array([1]))
        expected = reference_step([reference], [1])
        assert_transition(actual, expected)
        assert actual[3][0]
        if task in CLASSIC_TASKS:
            assert actual[2][0]  # termination and truncation are independent flags
            assert actual[1][0] == (1.0 if task == "CartPole-v1" else 0.0)
        final = actual[0].copy()
        batch.reset_done(np.array([True]), [72])
        np.testing.assert_array_equal(actual[0], final)
    finally:
        batch.close()
        reference.close()


@pytest.mark.parametrize("task", CLASSIC_TASKS)
def test_boundary_dynamics_against_reference(task):
    if task == "CartPole-v1":
        angle = 12 * 2 * np.pi / 360
        states = [
            [position, 0.0, theta, 0.0]
            for position in [-2.4, np.nextafter(-2.4, -np.inf), 2.4, np.nextafter(2.4, np.inf)]
            for theta in [-angle, np.nextafter(-angle, -np.inf), angle, np.nextafter(angle, np.inf)]
        ]
        states += [[0.0, 1.0, 0.1, -2.0], [0.0, -1.0, -0.1, 2.0]]
    else:
        states = [
            [theta1, theta2, velocity1, velocity2]
            for theta1 in [-np.pi, -np.pi / 2, np.pi / 2, np.pi]
            for theta2 in [-np.pi, 0.0, np.pi]
            for velocity1, velocity2 in [
                (0.0, 0.0),
                (4 * np.pi, 9 * np.pi),
                (-4 * np.pi, -9 * np.pi),
            ]
        ]
        states += [[2 * np.pi / 3 + delta, 0.0, 0.0, 0.0] for delta in [-1e-8, 0, 1e-8]]
    states = np.asarray(states, dtype=np.float64)
    batch = make_vector(task, len(states))
    references = [gym.make(task) for _ in states]
    clipped1 = clipped2 = False
    try:
        for action in range(batch.action_size):
            batch.reset(list(range(len(states))))
            batch.state[:] = states
            for env, state in zip(references, states, strict=True):
                env.reset(seed=0)
                env.unwrapped.state = state.copy()
            actions = np.full(len(states), action, dtype=int)
            assert_transition(batch.step(actions), reference_step(references, actions))
            if task == "Acrobot-v1":
                assert np.all(np.abs(batch.state[:, :2]) <= np.pi)
                clipped1 |= bool(np.any(np.abs(batch.state[:, 2]) == 4 * np.pi))
                clipped2 |= bool(np.any(np.abs(batch.state[:, 3]) == 9 * np.pi))
        if task == "Acrobot-v1":
            assert clipped1 and clipped2
    finally:
        batch.close()
        for env in references:
            env.close()


def test_inclusive_wrap_matches_gymnasium():
    angles = np.array(
        [
            -21 * np.pi,
            -np.pi,
            np.nextafter(-np.pi, -np.inf),
            0.0,
            np.pi,
            np.nextafter(np.pi, np.inf),
            21 * np.pi,
        ]
    )
    expected = np.array([wrap(value, -np.pi, np.pi) for value in angles])
    _wrap_angles(angles)
    np.testing.assert_array_equal(angles, expected)
    assert angles[1] == -np.pi and angles[4] == np.pi


@pytest.mark.parametrize("factory", [make_vector, GymnasiumBatch])
@pytest.mark.parametrize("task", ALL_TASKS)
def test_runtime_input_validation(factory, task):
    batch = factory(task, 2)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            batch.step(np.array([0, 0]))
        for seeds in [
            [],
            [1],
            [1, 2, 3],
            [1, -1],
            [1, 2.0],
            [1, True],
            (1, 2),
            [[1], 2],
            [np.int64(1), 2],
        ]:
            with pytest.raises(ValueError, match="seeds"):
                batch.reset(seeds)
        batch.reset([1, 2])
        for actions in [
            0,
            [[0, 0]],
            [0],
            [0, 0, 0],
            [0.0, 1.0],
            [True, False],
            [-1, 0],
            [0, batch.action_size],
        ]:
            with pytest.raises(ValueError, match="actions"):
                batch.step(actions)
        for mask in [True, [True], [[True, False]], [1, 0]]:
            with pytest.raises(ValueError, match="mask"):
                batch.reset_done(mask, [3])
        for active in [True, [True], [[True, False]], [1, 0]]:
            with pytest.raises(ValueError, match="active"):
                batch.step([0, 0], active=active)
        with pytest.raises(ValueError, match="seeds"):
            batch.reset_done([True, False], [3, 4])
        np.testing.assert_array_equal(batch.steps, [0, 0])
        batch.close()
        with pytest.raises(RuntimeError, match="closed"):
            batch.reset([1, 2])
    finally:
        batch.close()


@pytest.mark.parametrize("num_envs", [0, -1, 1.5, True])
def test_invalid_batch_size(num_envs):
    with pytest.raises(ValueError, match="num_envs"):
        make_vector("CartPole-v1", num_envs)


def test_unknown_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        make_vector("Unknown-v1", 1)
