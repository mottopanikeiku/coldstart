"""Fixed-state checks use installed Gymnax 0.0.9, not a guessed upstream revision."""

import importlib

import gymnasium as gym
import numpy as np
import pytest

jax = pytest.importorskip("jax")
gymnax = pytest.importorskip("gymnax")

jnp = importlib.import_module("jax.numpy")
AcrobotState = importlib.import_module("gymnax.environments.classic_control.acrobot").EnvState
CartPoleState = importlib.import_module("gymnax.environments.classic_control.cartpole").EnvState
ppo = importlib.import_module("control_clock.jax_ppo")
ActorCritic = ppo.ActorCritic
Transition = ppo.Transition
advantages = ppo.advantages
make_update = ppo.make_update
numpy_policy = ppo.numpy_policy


def cartpole(values, time=0, dtype=jnp.float32):
    return CartPoleState(
        time=time, **dict(zip(("x", "x_dot", "theta", "theta_dot"), map(dtype, values)))
    )


def acrobot(values, time=0, dtype=jnp.float32):
    return AcrobotState(
        time=time,
        **dict(
            zip(
                ("joint_angle1", "joint_angle2", "velocity_1", "velocity_2"),
                map(dtype, values),
            )
        ),
    )


@pytest.mark.parametrize("task", ["CartPole-v1", "Acrobot-v1"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_fixed_state_dynamics(task, precision):
    # Float64 checks the equations independently of the training precision.
    # Float32 Acrobot RK4 has larger roundoff at near-limit velocities.
    with jax.experimental.enable_x64(precision == "float64"):
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        env, params = gymnax.make(task)
        reference = gym.make(task).unwrapped
        rng = np.random.default_rng(2026)
        max_error = 0.0
        for _ in range(100):
            if task == "CartPole-v1":
                state = rng.uniform([-2.3, -3, -0.2, -4], [2.3, 3, 0.2, 4])
                gymnax_state = cartpole(state, dtype=dtype)
            else:
                state = rng.uniform([-3.1, -3.1, -12, -27], [3.1, 3.1, 12, 27])
                gymnax_state = acrobot(state, dtype=dtype)
            for action in range(reference.action_space.n):
                reference.state = state.copy()
                if task == "CartPole-v1":
                    reference.steps_beyond_terminated = None
                observation, reward, terminated, _, _ = reference.step(action)
                actual, _, actual_reward, done, _ = env.step_env(
                    jax.random.PRNGKey(1), gymnax_state, action, params
                )
                error = float(np.max(np.abs(np.asarray(actual) - observation)))
                max_error = max(max_error, error)
                atol, rtol = (2e-6, 2e-7) if precision == "float64" else (1e-4, 1e-5)
                np.testing.assert_allclose(actual, observation, atol=atol, rtol=rtol)
                assert float(actual_reward) == reward
                assert bool(done) == terminated
        print(f"{task} {precision}: maximum observation absolute error {max_error:.9g}")
        reference.close()


@pytest.mark.parametrize("task", ["CartPole-v1", "Acrobot-v1"])
def test_timeout_reset_and_terminal_rewards(task):
    env, params = gymnax.make(task)
    reference = gym.make(task)
    assert params.max_steps_in_episode == reference.spec.max_episode_steps == 500
    if task == "CartPole-v1":
        state = cartpole([0.0, 0.0, 0.0, 0.0], time=499)
        terminal_state = cartpole([2.4, 1.0, 0.0, 0.0])
        action, terminal_reward = 1, 1.0
        # Strict position/angle limits: exactly the boundary is not terminated.
        assert not bool(env.is_terminal(cartpole([2.4, 0.0, 0.0, 0.0]), params))
        assert bool(env.is_terminal(cartpole([2.401, 0.0, 0.0, 0.0]), params))
        low, high = -0.05, 0.05
    else:
        state = acrobot([0.0, 0.0, 0.0, 0.0], time=499)
        terminal_state = acrobot([jnp.pi, 0.0, 0.0, 0.0])
        action, terminal_reward = 1, 0.0
        low, high = -0.1, 0.1
    _, _, reward, done, _ = env.step_env(jax.random.PRNGKey(1), state, action, params)
    assert bool(done)
    assert float(reward) == (1.0 if task == "CartPole-v1" else -1.0)
    _, _, reward, done, _ = env.step_env(jax.random.PRNGKey(1), terminal_state, action, params)
    assert bool(done)
    assert float(reward) == terminal_reward
    _, reset_state = env.reset(jax.random.PRNGKey(42), params)
    fields = (
        [reset_state.x, reset_state.x_dot, reset_state.theta, reset_state.theta_dot]
        if task == "CartPole-v1"
        else [
            reset_state.joint_angle1,
            reset_state.joint_angle2,
            reset_state.velocity_1,
            reset_state.velocity_2,
        ]
    )
    assert np.all(np.asarray(fields) >= low) and np.all(np.asarray(fields) <= high)
    assert int(reset_state.time) == 0
    reference.close()


def test_gae_against_scalar_reference():
    rewards = np.array([[1, -1], [2, -1], [3, 0]], dtype=np.float32)
    values = np.array([[0.3, 0.1], [0.5, 0.3], [0.8, 0.6]], dtype=np.float32)
    dones = np.array([[False, False], [True, False], [False, True]])
    last = np.array([1.1, 0.2], dtype=np.float32)
    trajectory = Transition(
        None, None, None, jnp.array(values), jnp.array(rewards), jnp.array(dones)
    )
    actual, target = advantages(trajectory, jnp.array(last))
    expected = np.zeros_like(values)
    carry, next_value = np.zeros(2), last
    for index in range(2, -1, -1):
        mask = 1 - dones[index]
        delta = rewards[index] + 0.99 * next_value * mask - values[index]
        carry = delta + 0.99 * 0.95 * mask * carry
        expected[index] = carry
        next_value = values[index]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(target, expected + values, atol=1e-6)


def test_update_and_host_greedy_policy():
    env, params = gymnax.make("CartPole-v1")
    network = ActorCritic(2)
    initialize, update = make_update(network, env, params)
    runner = initialize(jax.random.PRNGKey(0))
    initial = runner[0].params
    runner = update(runner)
    jax.block_until_ready(runner)
    leaves = jax.tree.leaves(runner[0].params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    assert any(
        not np.array_equal(before, after) for before, after in zip(jax.tree.leaves(initial), leaves)
    )
    observations = np.random.default_rng(1).normal(size=(100, 4)).astype(np.float32)
    logits, _ = network.apply(runner[0].params, observations)
    np.testing.assert_array_equal(
        numpy_policy(runner[0].params)(observations), np.asarray(logits).argmax(-1)
    )


def test_acrobot_wrap_endpoint_difference():
    gymnax_wrap = importlib.import_module("gymnax.environments.classic_control.acrobot").wrap
    gymnasium_wrap = importlib.import_module("gymnasium.envs.classic_control.acrobot").wrap
    assert gymnasium_wrap(np.pi, -np.pi, np.pi) == np.pi
    np.testing.assert_allclose(gymnax_wrap(jnp.pi, -jnp.pi, jnp.pi), -np.pi, atol=1e-6)
