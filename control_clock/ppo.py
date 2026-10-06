"""Small CPU PPO: batched NumPy dynamics and larger tensor minibatches."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from control_clock.vector_env import make_vector

CONFIGS = {
    "CartPole-v1": dict(
        envs=16,
        steps=32,
        width=32,
        layers=1,
        epochs=8,
        minibatch=512,
        lr=0.002,
        gamma=0.98,
        lam=0.8,
        entropy=0.0,
        horizon=100_000,
    ),
    "Acrobot-v1": dict(
        envs=64,
        steps=128,
        width=64,
        layers=2,
        epochs=4,
        minibatch=1024,
        lr=0.002,
        gamma=0.99,
        lam=0.95,
        entropy=0.0,
        horizon=1_000_000,
    ),
    "LunarLander-v3": dict(
        envs=16,
        steps=256,
        width=64,
        layers=2,
        epochs=4,
        minibatch=512,
        lr=0.0007,
        gamma=0.999,
        lam=0.98,
        entropy=0.01,
        horizon=1_000_000,
    ),
}


class ActorCritic(nn.Module):
    def __init__(self, observations, actions, width, layers):
        super().__init__()

        def network(output, gain):
            parts = []
            size = observations
            for _ in range(layers):
                linear = nn.Linear(size, width)
                nn.init.orthogonal_(linear.weight, np.sqrt(2))
                nn.init.zeros_(linear.bias)
                parts.extend((linear, nn.Tanh()))
                size = width
            linear = nn.Linear(size, output)
            nn.init.orthogonal_(linear.weight, gain)
            nn.init.zeros_(linear.bias)
            return nn.Sequential(*parts, linear)

        self.actor = network(actions, 0.01)
        self.critic = network(1, 1.0)

    def forward(self, observations):
        return self.actor(observations), self.critic(observations).squeeze(-1)


def advantages_reference(rewards, values, next_values, terminated, done, gamma, lam):
    """GAE: bootstrap truncations, but never propagate across episode boundaries."""
    advantages = torch.empty_like(rewards)
    last = torch.zeros(rewards.shape[1])
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_values[index] * (~terminated[index]) - values[index]
        last = delta + gamma * lam * (~done[index]) * last
        advantages[index] = last
    return advantages


def train(ctx):
    config = dict(CONFIGS[ctx.task])
    variant = ctx.configuration["variant"]
    if variant == "small-batch":
        config.update(envs=16, minibatch=256)
    elif variant == "wide":
        config.update(width=128, layers=2)
    elif variant == "zoo-shape":
        if ctx.task != "CartPole-v1":
            raise ValueError("zoo-shape is a CartPole configuration")
        config.update(envs=8, minibatch=256, width=64, layers=2, epochs=20, lr=0.001)
    elif variant == "gym-dynamics":
        pass
    elif variant != "default":
        raise ValueError(f"unknown PPO variant: {variant}")
    ctx.configuration.update(
        config,
        algorithm="PPO",
        clip=0.2,
        vf=0.5,
        grad_norm=0.5,
        adam_eps=1e-5,
        value_clip=False,
        advantage_normalization="batch",
        lr_schedule="linear" if variant == "zoo-shape" else "constant",
        clip_schedule="linear" if variant == "zoo-shape" else "constant",
        truncated_bootstrap=True,
    )
    rng = np.random.default_rng(ctx.seed)
    torch.manual_seed(ctx.seed)
    if variant == "gym-dynamics":
        from control_clock.vector_env import GymnasiumBatch

        env = GymnasiumBatch(ctx.task, config["envs"])
    else:
        env = make_vector(ctx.task, config["envs"])
    agent = ActorCritic(env.observation_size, env.action_size, config["width"], config["layers"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["lr"], eps=1e-5)
    # Bounded observation features, not fitted on evaluation data.
    scales = np.ones(env.observation_size, dtype=np.float32)
    if ctx.task == "Acrobot-v1":
        scales[-2:] = [4 * np.pi, 9 * np.pi]
    ctx.configuration.update(observation_scales=scales.tolist(), dynamics=type(env).__name__)

    def tensor(obs):
        return torch.from_numpy(np.asarray(obs, dtype=np.float32) / scales)

    def policy(obs):
        with torch.no_grad():
            return agent.actor(tensor(obs)).argmax(dim=-1).numpy()

    n, length = config["envs"], config["steps"]
    shape = (length, n)
    observations = torch.empty((*shape, env.observation_size))
    actions = torch.empty(shape, dtype=torch.long)
    logprobs, rewards, values, next_values = (torch.empty(shape) for _ in range(4))
    terminated = torch.empty(shape, dtype=torch.bool)
    dones = torch.empty(shape, dtype=torch.bool)
    steps = 0
    try:
        next_obs = env.reset(ctx.training_seeds(n))
        ctx.checkpoint(policy, steps)
        while steps < config["horizon"]:
            for t in range(length):
                ctx.check_time()
                obs_tensor = tensor(next_obs)
                observations[t] = obs_tensor
                with torch.no_grad():
                    logits, value = agent(obs_tensor)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    actions[t] = action
                    logprobs[t] = distribution.log_prob(action)
                    values[t] = value
                final_obs, reward, term, trunc = env.step(action.numpy())
                with torch.no_grad():
                    next_values[t] = agent.critic(tensor(final_obs)).squeeze(-1)
                rewards[t] = torch.from_numpy(np.asarray(reward, dtype=np.float32))
                terminated[t] = torch.from_numpy(term)
                done = term | trunc
                dones[t] = torch.from_numpy(done)
                next_obs = (
                    env.reset_done(done, ctx.training_seeds(int(done.sum())))
                    if done.any()
                    else final_obs
                )
                steps += n
                ctx.steps = steps
            advantages = advantages_reference(
                rewards, values, next_values, terminated, dones, config["gamma"], config["lam"]
            )
            returns = advantages + values
            flat_obs = observations.reshape(-1, env.observation_size)
            flat_actions = actions.flatten()
            flat_logprobs = logprobs.flatten()
            flat_returns = returns.flatten()
            flat_advantages = advantages.flatten()
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (
                flat_advantages.std() + 1e-8
            )
            batch_size = length * n
            progress = 1.0 - steps / config["horizon"]
            if variant == "zoo-shape":
                optimizer.param_groups[0]["lr"] = config["lr"] * progress
            clip = 0.2 * progress if variant == "zoo-shape" else 0.2
            for _ in range(config["epochs"]):
                order = rng.permutation(batch_size)
                for start in range(0, batch_size, config["minibatch"]):
                    ctx.check_time()
                    indices = order[start : start + config["minibatch"]]
                    logits, new_values = agent(flat_obs[indices])
                    distribution = Categorical(logits=logits)
                    ratio = (
                        distribution.log_prob(flat_actions[indices]) - flat_logprobs[indices]
                    ).exp()
                    advantage = flat_advantages[indices]
                    loss_policy = torch.maximum(
                        -advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip)
                    ).mean()
                    loss_value = 0.5 * (new_values - flat_returns[indices]).square().mean()
                    loss = (
                        loss_policy
                        + 0.5 * loss_value
                        - config["entropy"] * distribution.entropy().mean()
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
            ctx.checkpoint(policy, steps)
    finally:
        env.close()
