import numpy as np
import torch

from control_clock.ppo import advantages_reference
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
