"""The common evaluation contract; deliberately always evaluates with Gymnasium."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np

THRESHOLDS = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0, "LunarLander-v3": 200.0}
BUDGETS = {"CartPole-v1": 120.0, "Acrobot-v1": 300.0, "LunarLander-v3": 300.0}
EVAL_SEEDS = tuple(range(1_000_000, 1_000_100))
EVAL_INTERVAL = 10_000
Policy = Callable[[np.ndarray], np.ndarray]


class Finished(Exception):
    """A success or a time limit ends a training run, not the parent batch."""


@dataclass
class RunContext:
    task: str
    seed: int
    started: float
    limit: float
    evaluations: list[dict] = field(default_factory=list)
    solved: bool = False
    solve_seconds: float | None = None
    steps: int = 0
    configuration: dict = field(default_factory=dict)
    _next_evaluation: int = 0
    _eval_envs: list = field(default_factory=list)
    _train_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._train_rng = np.random.default_rng(self.seed)

    def training_seeds(self, count: int) -> list[int]:
        return self._train_rng.integers(0, 1_000_000, size=count).tolist()

    def check_time(self):
        if time.perf_counter() - self.started >= self.limit:
            raise Finished("time limit")

    def checkpoint(self, policy: Policy, steps: int, *, force: bool = False):
        self.steps = int(steps)
        self.check_time()
        if not force and steps < self._next_evaluation:
            return
        if not self._eval_envs:
            self._eval_envs = [gym.make(self.task) for _ in EVAL_SEEDS]
        begin = time.perf_counter()
        observations = np.stack(
            [env.reset(seed=seed)[0] for env, seed in zip(self._eval_envs, EVAL_SEEDS)]
        )
        returns = np.zeros(len(EVAL_SEEDS), dtype=np.float64)
        active = np.ones(len(EVAL_SEEDS), dtype=bool)
        while active.any():
            self.check_time()
            indices = np.flatnonzero(active)
            actions = np.asarray(policy(observations[indices]), dtype=np.int64)
            if actions.shape != indices.shape:
                raise ValueError(f"policy action shape {actions.shape}, expected {indices.shape}")
            for index, action in zip(indices, actions):
                obs, reward, terminated, truncated, _ = self._eval_envs[index].step(int(action))
                observations[index] = obs
                returns[index] += reward
                active[index] = not (terminated or truncated)
        end = time.perf_counter()
        mean = float(returns.mean())
        entry = {
            "steps": int(steps),
            "elapsed_seconds": end - self.started,
            "evaluation_seconds": end - begin,
            "mean_return": mean,
            "returns": returns.tolist(),
            "passes": mean >= THRESHOLDS[self.task] and end - self.started <= self.limit,
        }
        self.evaluations.append(entry)
        self._next_evaluation = int(steps) + EVAL_INTERVAL
        if entry["passes"]:
            self.solved = True
            self.solve_seconds = entry["elapsed_seconds"]
            raise Finished("solved")
        self.check_time()

    def close(self):
        for env in self._eval_envs:
            env.close()
