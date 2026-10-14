import time

import numpy as np
import pytest
import torch

from control_clock.ppo import advantages_reference
from control_clock.protocol import Finished, RunContext
from control_clock.search import ars_update, cem_update, greedy_actions


def test_gae_matches_numpy_reference_with_truncation():
    rng = np.random.default_rng(7)
    shape = (9, 4)
    rewards = rng.normal(size=shape).astype(np.float32)
    values = rng.normal(size=shape).astype(np.float32)
    next_values = rng.normal(size=shape).astype(np.float32)
    terminated = rng.random(shape) < 0.2
    done = terminated | (rng.random(shape) < 0.2)
    expected = np.zeros(shape, dtype=np.float32)
    for slot in range(shape[1]):
        carry = 0.0
        for index in reversed(range(shape[0])):
            delta = rewards[index, slot] - values[index, slot]
            if not terminated[index, slot]:
                delta += 0.99 * next_values[index, slot]
            carry = delta + (0 if done[index, slot] else 0.99 * 0.95 * carry)
            expected[index, slot] = carry
    actual = advantages_reference(
        *[torch.from_numpy(array) for array in (rewards, values, next_values, terminated, done)],
        0.99,
        0.95,
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)


def test_ars_update_matches_loop_reference():
    rng = np.random.default_rng(19)
    weights = rng.normal(size=(3, 7))
    directions = rng.normal(size=(8, 3, 7))
    positive, negative = rng.normal(size=(2, 8))
    selected = np.argsort(np.maximum(positive, negative))[-4:]
    scale = np.std(np.concatenate((positive[selected], negative[selected])))
    expected = weights.copy()
    for index in selected:
        expected += 0.2 / (4 * scale) * (positive[index] - negative[index]) * directions[index]
    np.testing.assert_allclose(
        ars_update(weights, directions, positive, negative, 4, 0.2), expected, atol=1e-14
    )
    np.testing.assert_array_equal(
        ars_update(weights, directions, np.ones(8), np.ones(8), 4, 0.2), weights
    )


def test_affine_policy_matches_scalar_reference():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(10, 3, 7))
    observations = rng.normal(size=(10, 6))
    scales = np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi])
    expected = [np.argmax(w @ np.append(obs / scales, 1)) for w, obs in zip(weights, observations)]
    np.testing.assert_array_equal(greedy_actions(weights, observations, scales), expected)
    np.testing.assert_array_equal(
        greedy_actions(np.zeros((3, 7)), observations, scales), np.zeros(10)
    )


def test_cem_update_matches_scalar_reference_and_stable_ties():
    rng = np.random.default_rng(41)
    mean = rng.normal(size=(3, 7))
    std = np.abs(rng.normal(size=(3, 7)))
    candidates = rng.normal(size=(16, 3, 7))
    scores = np.arange(16) % 4  # Ties retain ascending candidate order.
    indices = sorted(range(16), key=lambda i: (scores[i], i))[-4:]
    expected_mean = np.empty_like(mean)
    expected_std = np.empty_like(std)
    for action in range(3):
        for feature in range(7):
            values = [candidates[i, action, feature] for i in indices]
            average = sum(values) / len(values)
            variance = sum((value - average) ** 2 for value in values) / len(values)
            expected_mean[action, feature] = 0.5 * mean[action, feature] + 0.5 * average
            expected_std[action, feature] = max(
                0.05, 0.5 * std[action, feature] + 0.5 * np.sqrt(variance)
            )
    actual_mean, actual_std = cem_update(mean, std, candidates, scores, 4)
    np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-14)
    np.testing.assert_allclose(actual_std, expected_std, atol=1e-14)
    _, collapsed_std = cem_update(mean, np.zeros_like(std), np.ones_like(candidates), scores, 4)
    np.testing.assert_array_equal(collapsed_std, np.full_like(std, 0.05))


@pytest.mark.parametrize("variant", ["default", "zoo-shape"])
def test_cpu_ppo_checkpoint_follows_all_optimizer_epochs(monkeypatch, variant):
    from control_clock import ppo

    optimizer_steps = [0]
    checkpoints = []
    environments = []
    original_step = torch.optim.Adam.step
    original_make_vector = ppo.make_vector

    def recorded_step(optimizer, *args, **kwargs):
        optimizer_steps[0] += 1
        return original_step(optimizer, *args, **kwargs)

    def recorded_make_vector(task, num_envs):
        environments.append(original_make_vector(task, num_envs))
        return environments[-1]

    class WiringContext(RunContext):
        def checkpoint(self, policy, steps, *, force=False):
            assert policy(np.zeros((100, 4), dtype=np.float32)).shape == (100,)
            checkpoints.append((steps, optimizer_steps[0]))
            if steps:
                raise Finished("wiring test complete")

    monkeypatch.setattr(torch.optim.Adam, "step", recorded_step)
    monkeypatch.setattr(ppo, "make_vector", recorded_make_vector)
    ctx = WiringContext("CartPole-v1", 0, time.perf_counter(), 120)
    ctx.configuration.update(method="ppo-cpu", variant=variant)
    with pytest.raises(Finished, match="wiring test complete"):
        ppo.train(ctx)
    config = ctx.configuration
    batch = config["envs"] * config["steps"]
    updates_per_epoch = -(-batch // config["minibatch"])
    assert checkpoints == [(0, 0), (batch, config["epochs"] * updates_per_epoch)]
    assert environments[0]._closed


@pytest.mark.parametrize("method", ["ars", "cem"])
def test_search_checkpoint_follows_complete_generation(monkeypatch, method):
    from control_clock import search

    generations = []
    checkpoints = []
    environments = []
    original_returns = search.candidate_returns
    original_make_vector = search.make_vector

    def recorded_returns(ctx, weights, *args):
        scores, steps = original_returns(ctx, weights, *args)
        generations.append(steps)
        return scores, steps

    def recorded_make_vector(task, num_envs):
        environments.append(original_make_vector(task, num_envs))
        return environments[-1]

    class WiringContext(RunContext):
        def checkpoint(self, policy, steps, *, force=False):
            assert policy(np.zeros((100, 4))).shape == (100,)
            checkpoints.append((steps, list(generations)))
            if steps:
                raise Finished("wiring test complete")

    monkeypatch.setattr(search, "candidate_returns", recorded_returns)
    monkeypatch.setattr(search, "make_vector", recorded_make_vector)
    ctx = WiringContext("CartPole-v1", 0, time.perf_counter(), 120)
    ctx.configuration.update(method=method, variant="default")
    with pytest.raises(Finished, match="wiring test complete"):
        search.train(ctx)
    assert len(generations) == 1
    assert checkpoints == [(0, []), (generations[0], generations)]
    assert generations[0] >= ctx.configuration["population"] * 4
    assert environments[0]._closed
