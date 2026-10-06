"""SB3 PPO with the pinned RL Baselines3 Zoo settings in configs/zoo_ppo.json.

No PPO equations are reimplemented here. Instrumentation checks the clock during
rollouts/minibatches and evaluates only after SB3's entire PPO.train() returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from control_clock.protocol import RunContext


def load_settings(task: str) -> dict:
    """Return JSON-serializable effective settings, including upstream defaults."""
    path = Path(__file__).resolve().parents[1] / "configs" / "zoo_ppo.json"
    source = json.loads(path.read_text())
    task_settings = source["tasks"][task]
    ppo = source["ppo_defaults"] | task_settings["ppo"]
    normalization = None
    if task_settings["normalize"]:
        normalization = source["vec_normalize_defaults"] | {"gamma": ppo["gamma"]}
    return {
        "method": "sb3_ppo",
        "sources": source["sources"],
        "task": task,
        "policy": task_settings["policy"],
        "n_envs": task_settings["n_envs"],
        "total_timesteps": task_settings["n_timesteps"],
        "ppo": ppo,
        "policy_defaults": source["policy_defaults"],
        "vec_normalize": normalization,
        "training_environment": "Gymnasium + Monitor + DummyVecEnv",
        "autoreset": "SB3 same-step reset",
        "checkpoint": "initial, then immediately after PPO.train returns",
        "evaluation": "deterministic; frozen observation statistics; raw rewards",
        "learn": {"log_interval": None, "progress_bar": False, "reset_num_timesteps": True},
        "torch_threads": 1,
    }


def linear_schedule(initial: float):
    """Zoo's lin_ schedule: initial value times SB3 progress remaining."""

    def schedule(progress_remaining: float) -> float:
        return initial * progress_remaining

    return schedule


def train(ctx: RunContext) -> None:
    """Train the library comparator; propagate Finished to the common harness."""
    settings = load_settings(ctx.task)
    ctx.configuration.update(settings)
    ctx.check_time()

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class ClockCallback(BaseCallback):
        def _on_step(self) -> bool:
            ctx.steps = self.num_timesteps
            ctx.check_time()
            return True

    class ClockRolloutBuffer(RolloutBuffer):
        def get(self, batch_size=None):
            for batch in super().get(batch_size):
                ctx.check_time()
                yield batch

    class CheckpointPPO(PPO):
        def train(self) -> None:
            ctx.check_time()
            super().train()
            ctx.checkpoint(policy, self.num_timesteps)

    # SB3 seeds VecEnv lanes as seed + index. Bound the base before any reset.
    n_envs = settings["n_envs"]
    seed_modulus = 1_000_000 - n_envs + 1
    model_seed = ctx.seed % seed_modulus
    reset_seed = ctx.training_seeds(1)[0] % seed_modulus
    ctx.configuration.update(
        {
            "model_seed": model_seed,
            "training_reset_seeds": list(range(reset_seed, reset_seed + n_envs)),
            "action_space_seed": model_seed,
            "seed_mapping": "model: run_seed % (1000000-n_envs+1); reset: context draw % same",
            "rollout_buffer_class": "ClockRolloutBuffer (unchanged SB3 batches; clock check only)",
        }
    )
    kwargs = settings["ppo"].copy()
    for key in ("learning_rate", "clip_range"):
        value = kwargs[key]
        if isinstance(value, str):
            kwargs[key] = linear_schedule(float(value.removeprefix("lin_")))

    environments = []
    env = None
    try:
        for _ in range(n_envs):
            ctx.check_time()
            environments.append(gym.make(ctx.task))
            environments[-1] = Monitor(environments[-1])
        env = DummyVecEnv([lambda item=item: item for item in environments])
        if settings["vec_normalize"] is not None:
            env = VecNormalize(env, **settings["vec_normalize"])
        model = CheckpointPPO(
            settings["policy"],
            env,
            seed=model_seed,
            rollout_buffer_class=ClockRolloutBuffer,
            **kwargs,
        )
        # Override SB3's pending model-seed reset with protocol training seeds.
        env.seed(reset_seed)

        def policy(observations: np.ndarray) -> np.ndarray:
            if isinstance(env, VecNormalize):
                # normalize_obs does NOT update obs_rms/ret_rms, even in training mode.
                observations = env.normalize_obs(observations)
            actions, _ = model.predict(observations, deterministic=True)
            return np.asarray(actions, dtype=np.int64).reshape(-1)

        ctx.checkpoint(policy, 0, force=True)
        model.learn(settings["total_timesteps"], callback=ClockCallback(), **settings["learn"])
    finally:
        if env is not None:
            env.close()
        else:
            for item in environments:
                item.close()
