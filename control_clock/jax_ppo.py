"""JAX PPO using vmap environments and scan rollouts, following PureJaxRL's design.

This is a separate GPU comparison, not a replacement for the laptop experiment.

Inspired by Chris Lu's PureJaxRL (2023), licensed under Apache-2.0:
https://github.com/luchris429/purejaxrl (license in third_party/PureJaxRL-LICENSE).
I changed the structure to expose per-update checkpoints and host-side evaluation;
this is not an unchanged run of the upstream implementation.
"""

from typing import NamedTuple

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

NUM_ENVS = 128
NUM_STEPS = 128
MINIBATCHES = 4
EPOCHS = 4
CONFIG = {
    "num_envs": NUM_ENVS,
    "num_steps": NUM_STEPS,
    "minibatches": MINIBATCHES,
    "epochs": EPOCHS,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "max_gradient_norm": 0.5,
    "hidden_sizes": [64, 64],
    "anneal_learning_rate": False,
    "timeout_bootstrap": False,
}


class ActorCritic(nn.Module):
    actions: int

    @nn.compact
    def __call__(self, observations):
        outputs = []
        for name, size, scale in (("actor", self.actions, 0.01), ("critic", 1, 1.0)):
            x = observations
            for index in range(2):
                x = nn.tanh(
                    nn.Dense(
                        64,
                        kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        name=f"{name}_{index}",
                    )(x)
                )
            outputs.append(
                nn.Dense(size, kernel_init=nn.initializers.orthogonal(scale), name=f"{name}_out")(x)
            )
        return outputs[0], outputs[1].squeeze(-1)


class Transition(NamedTuple):
    observation: jax.Array
    action: jax.Array
    log_probability: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def log_probability(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]


def advantages(trajectory, last_value):
    def step(carry, transition):
        previous, next_value = carry
        mask = 1 - transition.done
        delta = transition.reward + 0.99 * next_value * mask - transition.value
        advantage = delta + 0.99 * 0.95 * mask * previous
        return (advantage, transition.value), advantage

    _, result = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value), trajectory, reverse=True
    )
    return result, result + trajectory.value


def numpy_policy(parameters):
    """Snapshot GPU weights once; common Gymnasium evaluation runs on the host."""
    parameters = jax.device_get(parameters)["params"]

    def policy(observations):
        x = np.asarray(observations, dtype=np.float32)
        for index in range(2):
            layer = parameters[f"actor_{index}"]
            x = np.tanh(x @ layer["kernel"] + layer["bias"])
        layer = parameters["actor_out"]
        return np.argmax(x @ layer["kernel"] + layer["bias"], axis=-1)

    return policy


def make_update(network, env, env_params):
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step_env = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def initialize(key):
        key, model_key, env_key = jax.random.split(key, 3)
        parameters = network.init(model_key, jnp.zeros((1, env.obs_shape[0])))
        optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(0.0003, eps=1e-5))
        train = TrainState.create(apply_fn=network.apply, params=parameters, tx=optimizer)
        observations, states = reset(jax.random.split(env_key, NUM_ENVS), env_params)
        return train, states, observations, key

    @jax.jit
    def update(runner):
        def collect(carry, _):
            train, states, observations, key = carry
            key, action_key, env_key = jax.random.split(key, 3)
            logits, values = network.apply(train.params, observations)
            actions = jax.random.categorical(action_key, logits)
            next_obs, states, rewards, dones, _ = step_env(
                jax.random.split(env_key, NUM_ENVS), states, actions, env_params
            )
            transition = Transition(
                observations, actions, log_probability(logits, actions), values, rewards, dones
            )
            return (train, states, next_obs, key), transition

        runner, trajectory = jax.lax.scan(collect, runner, None, length=NUM_STEPS)
        train, states, observations, key = runner
        _, last_value = network.apply(train.params, observations)
        gae, targets = advantages(trajectory, last_value)
        batch = jax.tree.map(
            lambda x: x.reshape((NUM_ENVS * NUM_STEPS,) + x.shape[2:]),
            (trajectory, gae, targets),
        )

        def epoch(carry, _):
            train, key = carry
            key, shuffle_key = jax.random.split(key)
            order = jax.random.permutation(shuffle_key, NUM_ENVS * NUM_STEPS)
            minibatches = jax.tree.map(
                lambda x: x[order].reshape((MINIBATCHES, -1) + x.shape[1:]), batch
            )

            def optimize(train, minibatch):
                transitions, advantage, target = minibatch

                def loss(parameters):
                    logits, value = network.apply(parameters, transitions.observation)
                    ratio = jnp.exp(
                        log_probability(logits, transitions.action) - transitions.log_probability
                    )
                    normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    actor = -jnp.minimum(
                        ratio * normalized, jnp.clip(ratio, 0.8, 1.2) * normalized
                    ).mean()
                    clipped = transitions.value + jnp.clip(value - transitions.value, -0.2, 0.2)
                    critic = (
                        0.5 * jnp.maximum((value - target) ** 2, (clipped - target) ** 2).mean()
                    )
                    log_probs = jax.nn.log_softmax(logits)
                    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
                    return actor + 0.5 * critic - 0.01 * entropy

                gradients = jax.grad(loss)(train.params)
                return train.apply_gradients(grads=gradients), None

            train, _ = jax.lax.scan(optimize, train, minibatches)
            return (train, key), None

        (train, key), _ = jax.lax.scan(epoch, (train, key), None, length=EPOCHS)
        return train, states, observations, key

    return initialize, update


def train(context):
    env, env_params = gymnax.make(context.task)
    network = ActorCritic(env.num_actions)
    initialize, update = make_update(network, env, env_params)
    context.configuration.update(CONFIG, training_environment="gymnax", evaluation="gymnasium")
    runner = initialize(jax.random.PRNGKey(context.seed))
    jax.block_until_ready(runner)
    context.checkpoint(numpy_policy(runner[0].params), 0, force=True)
    steps = 0
    while True:
        context.check_time()
        runner = update(runner)
        jax.block_until_ready(runner)
        steps += NUM_ENVS * NUM_STEPS
        context.checkpoint(numpy_policy(runner[0].params), steps)
