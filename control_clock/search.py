"""Reward-based black-box search over deterministic affine policies."""

from __future__ import annotations

import numpy as np

from control_clock.vector_env import make_vector


def greedy_actions(weights, observations, scales):
    features = np.concatenate(
        (observations / scales, np.ones((*observations.shape[:-1], 1))), axis=-1
    )
    return np.einsum("...ao,...o->...a", weights, features).argmax(axis=-1)


def candidate_returns(ctx, weights, env, scales, episodes, steps):
    """Common reset seeds within each population reduce score-comparison noise."""
    totals = np.zeros(len(weights))
    for _ in range(episodes):
        seed = ctx.training_seeds(1)[0]
        observations = env.reset([seed] * env.num_envs)
        active = np.ones(env.num_envs, dtype=bool)
        while active.any():
            ctx.check_time()
            actions = greedy_actions(weights, observations, scales)
            observations, rewards, terminated, truncated = env.step(actions, active=active)
            steps += int(active.sum())
            ctx.steps = steps
            totals += rewards
            active &= ~(terminated | truncated)
    return totals / episodes, steps


def ars_update(weights, directions, positive, negative, top, learning_rate):
    """ARS top-direction update; zero score variance leaves parameters unchanged."""
    selected = np.argsort(np.maximum(positive, negative))[-top:]
    score_std = np.std(np.concatenate((positive[selected], negative[selected])))
    if score_std < 1e-8:
        return weights.copy()
    differences = positive[selected] - negative[selected]
    return weights + learning_rate / (top * score_std) * np.einsum(
        "n,nao->ao", differences, directions[selected]
    )


def cem_update(mean, std, candidates, scores, elite_count):
    """Fit and smooth the elite Gaussian; stable ordering defines score ties."""
    elites = candidates[np.argsort(scores, kind="stable")[-elite_count:]]
    return (
        0.5 * mean + 0.5 * elites.mean(axis=0),
        np.maximum(0.05, 0.5 * std + 0.5 * elites.std(axis=0)),
    )


def train(ctx):
    method = ctx.configuration["method"]
    variant = ctx.configuration["variant"]
    if variant not in ("default", "more-episodes", "larger-population", "longer"):
        raise ValueError(f"unknown policy-search variant: {variant}")
    population = 64 if ctx.task != "LunarLander-v3" else 32
    episodes = 4
    if variant == "more-episodes":
        episodes = 8
    elif variant == "larger-population":
        population *= 2
    horizon = 4_000_000 if variant == "longer" else 1_000_000
    directions_count = population // 2
    env = make_vector(ctx.task, population)
    scales = np.ones(env.observation_size, dtype=np.float64)
    if ctx.task == "Acrobot-v1":
        scales[-2:] = [4 * np.pi, 9 * np.pi]
    rng = np.random.default_rng(ctx.seed)
    shape = (env.action_size, env.observation_size + 1)
    weights = np.zeros(shape)
    std = np.ones(shape)
    steps = 0
    ctx.configuration.update(
        algorithm=method.upper(),
        policy="affine deterministic argmax",
        population=population,
        training_episodes_per_candidate=episodes,
        horizon=horizon,
        scales=scales.tolist(),
        training_reset_seeds="common across candidates, fresh each episode",
    )
    if method == "ars":
        ctx.configuration.update(
            directions=directions_count,
            top=directions_count // 2,
            perturbation=0.5,
            learning_rate=0.2,
            normalization="fixed physical velocity scales",
        )
    else:
        ctx.configuration.update(
            elites=population // 8, initial_std=1.0, smoothing=0.5, minimum_std=0.05
        )

    def policy(observations):
        return greedy_actions(weights, observations, scales)

    try:
        ctx.checkpoint(policy, steps)
        while steps < horizon:
            if method == "ars":
                directions = rng.standard_normal((directions_count, *shape))
                candidates = np.concatenate(
                    (weights + 0.5 * directions, weights - 0.5 * directions)
                )
                scores, steps = candidate_returns(ctx, candidates, env, scales, episodes, steps)
                weights = ars_update(
                    weights,
                    directions,
                    scores[:directions_count],
                    scores[directions_count:],
                    directions_count // 2,
                    0.2,
                )
            else:
                candidates = rng.normal(weights, std, size=(population, *shape))
                scores, steps = candidate_returns(ctx, candidates, env, scales, episodes, steps)
                weights, std = cem_update(weights, std, candidates, scores, population // 8)
            ctx.checkpoint(policy, steps)
    finally:
        env.close()
