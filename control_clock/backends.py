"""Environment backends for the systems benchmark, behind one host-side interface.

Every backend steps a batch with host NumPy actions and returns host NumPy observations,
rewards and done flags, which is what a training loop with a host policy needs. Backends
differ in how finished slots restart:

- explicit: the caller resets done slots with ``reset_done`` (this repository's batches);
- same-step: ``step`` already returned the reset observation (Gymnasium sync, Gymnax);
- next-step: the next ``step`` spends the slot on a reset and ignores its action
  (Gymnasium's vector CartPole, EnvPool). Those slots are not counted as transitions.

``states`` returns float64 internal states for the equivalence gate. EnvPool exposes only
observations, so its states are rebuilt from float32 observations.
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("numpy", "gymnasium", "gymnasium-vector", "envpool", "gymnax", "gymnax-scan")
SCAN_LENGTH = 64


def _acrobot_state(obs: np.ndarray) -> np.ndarray:
    obs = obs.astype(np.float64)
    angles = np.arctan2(obs[:, [1, 3]], obs[:, [0, 2]])
    return np.concatenate((angles, obs[:, 4:6]), axis=1)


class NumpyBackend:
    """This repository's explicit-reset batches."""

    restart = "explicit"

    def __init__(self, task, num_envs, seed):
        from control_clock.vector_env import make_vector

        self.env = make_vector(task, num_envs)
        self._seeds = np.random.default_rng(seed).integers(0, 1_000_000, size=1 << 16).tolist()
        self._next = 0

    def _take(self, count):
        if self._next + count > len(self._seeds):
            self._next = 0
        seeds = self._seeds[self._next : self._next + count]
        self._next += count
        return seeds

    def reset(self):
        self.env.reset(self._take(self.env.num_envs))

    def step(self, actions):
        obs, reward, terminated, truncated = self.env.step(actions)
        return obs, reward, terminated, terminated | truncated

    def reset_seeds(self, done):
        """Untimed: choose seeds for the slots reset_done will restart."""
        return self._take(int(done.sum()))

    def reset_done(self, done, seeds):
        self.env.reset_done(done, seeds)

    def states(self):
        return self.env.state.copy()

    def close(self):
        self.env.close()


class GymnasiumBackend:
    """Gymnasium's SyncVectorEnv over gymnasium.make, resetting on the same step."""

    restart = "same-step"

    def __init__(self, task, num_envs, seed):
        import gymnasium as gym
        from gymnasium.vector import AutoresetMode

        self.env = gym.make_vec(
            task,
            num_envs=num_envs,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
        )
        self.seed = seed

    def reset(self):
        self.env.reset(seed=self.seed)

    def step(self, actions):
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        return obs, reward, terminated, terminated | truncated

    def states(self):
        return np.stack([np.asarray(env.unwrapped.state, np.float64) for env in self.env.envs])

    def close(self):
        self.env.close()


class GymnasiumVectorBackend(GymnasiumBackend):
    """Gymnasium's own NumPy-vectorized CartPole (no Acrobot equivalent exists)."""

    restart = "next-step"

    def __init__(self, task, num_envs, seed):
        import gymnasium as gym

        if task != "CartPole-v1":
            raise ValueError("Gymnasium 1.2.1 has a vector entry point only for CartPole")
        self.env = gym.make_vec(task, num_envs=num_envs, vectorization_mode="vector_entry_point")
        self.seed = seed

    def states(self):
        return np.asarray(self.env.unwrapped.state, np.float64).T.copy()


class EnvPoolBackend:
    """EnvPool's C++ environments, synchronous, one worker thread."""

    restart = "next-step"

    def __init__(self, task, num_envs, seed):
        import envpool

        self.task = task
        self.env = envpool.make(
            task,
            env_type="gymnasium",
            num_envs=num_envs,
            batch_size=num_envs,
            num_threads=1,
            seed=seed,
        )
        self._obs = None

    def reset(self):
        self._obs, _ = self.env.reset()

    def step(self, actions):
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        self._obs = obs
        return obs, reward, terminated, terminated | truncated

    def states(self):
        if self.task == "CartPole-v1":
            return self._obs.astype(np.float64)
        return _acrobot_state(self._obs)

    def close(self):
        self.env.close()


class GymnaxBackend:
    """Gymnax under jit(vmap): one dispatch per step, results copied to the host."""

    restart = "same-step"

    def __init__(self, task, num_envs, seed):
        import gymnax
        import jax

        self.jax = jax
        self.task = task
        env, params = gymnax.make(task)
        self._reset = jax.jit(
            lambda key: jax.vmap(env.reset, in_axes=(0, None))(
                jax.random.split(key, num_envs), params
            )
        )

        def step(key, state, actions):
            key, sub = jax.random.split(key)
            obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(sub, num_envs), state, actions, params
            )
            return key, state, obs, reward, done

        self._step = jax.jit(step)
        self._key = jax.random.PRNGKey(seed)
        self._state = None

    def reset(self):
        self._key, key = self.jax.random.split(self._key)
        _, self._state = self.jax.block_until_ready(self._reset(key))

    def step(self, actions):
        self._key, self._state, obs, reward, done = self._step(self._key, self._state, actions)
        obs, reward, done = np.asarray(obs), np.asarray(reward), np.asarray(done)
        return obs, reward, None, done

    def states(self):
        state = self._state
        if self.task == "CartPole-v1":
            fields = (state.x, state.x_dot, state.theta, state.theta_dot)
        else:
            fields = (state.joint_angle1, state.joint_angle2, state.velocity_1, state.velocity_2)
        return np.stack([np.asarray(field, np.float64) for field in fields], axis=1)

    def close(self):
        pass


class GymnaxScanBackend(GymnaxBackend):
    """Gymnax with SCAN_LENGTH steps fused in one lax.scan; nothing returns to the host.

    One call takes a (SCAN_LENGTH, num_envs) action block and reports only the episode
    completions as a count, so it measures on-device simulation, not a host training loop.
    """

    def __init__(self, task, num_envs, seed):
        super().__init__(task, num_envs, seed)
        jax = self.jax
        step = self._step

        def block(key, state, actions):
            def body(carry, action):
                key, state = carry
                key, state, _, _, done = step(key, state, action)
                return (key, state), done.sum()

            (key, state), done = jax.lax.scan(body, (key, state), actions)
            return key, state, done.sum()

        self._block = jax.jit(block)

    def step(self, actions):
        self._key, self._state, done = self._block(self._key, self._state, actions)
        return None, None, None, int(done)


CLASSES = {
    "numpy": NumpyBackend,
    "gymnasium": GymnasiumBackend,
    "gymnasium-vector": GymnasiumVectorBackend,
    "envpool": EnvPoolBackend,
    "gymnax": GymnaxBackend,
    "gymnax-scan": GymnaxScanBackend,
}


def make(name, task, num_envs, seed):
    return CLASSES[name](task, num_envs, seed)


def supports(name, task):
    return not (name == "gymnasium-vector" and task != "CartPole-v1")


def gate(name, task, *, slots=64, steps=64, seed=7, atol=1e-4):
    """Step the backend and Gymnasium from the backend's own reset states with the same actions.

    Compares observations, rewards and termination until each slot's first termination.
    Same-step backends replace a terminal observation with the reset observation, so on
    that step only reward and termination are compared.
    Returns the largest observation difference and the number of compared transitions.
    """
    import gymnasium as gym

    backend_name = "gymnax" if name == "gymnax-scan" else name
    backend = make(backend_name, task, slots, seed)
    references = [gym.make(task).unwrapped for _ in range(slots)]
    actions = np.random.default_rng(seed).integers(
        0, references[0].action_space.n, size=(steps, slots)
    )
    worst, compared, mismatches = 0.0, 0, 0
    try:
        backend.reset()
        for reference, state in zip(references, backend.states(), strict=True):
            reference.reset(seed=0)
            reference.state = state.copy()
        live = np.ones(slots, dtype=bool)
        for row in actions:
            obs, reward, terminated, done = backend.step(row)
            for index in np.flatnonzero(live):
                expected = references[index].step(int(row[index]))
                ref_done = expected[2] or expected[3]
                ended = bool(done[index])
                if not (ended and backend.restart == "same-step"):
                    worst = max(worst, float(np.max(np.abs(obs[index] - expected[0]))))
                if float(reward[index]) != expected[1] or ended != ref_done:
                    mismatches += 1
                compared += 1
                if ended or ref_done:
                    live[index] = False
            if backend.restart == "explicit" and done.any():
                backend.reset_done(done, backend.reset_seeds(done))
            if not live.any():
                break
    finally:
        backend.close()
        for reference in references:
            reference.close()
    return {
        "passed": mismatches == 0 and worst <= atol,
        "max_observation_difference": worst,
        "atol": atol,
        "compared_transitions": compared,
        "reward_or_done_mismatches": mismatches,
    }
