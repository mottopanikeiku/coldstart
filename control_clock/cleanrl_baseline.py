"""Discrete PPO adapted from CleanRL ppo.py, not a new PPO implementation.

Copyright (c) 2019 CleanRL developers. MIT license: third_party/CleanRL-LICENSE.
Source: https://github.com/vwxyzjn/cleanrl/blob/
e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py
Changes: callable CPU-only entry point, protocol seeding/evaluation/clock,
Gymnasium 1.2 SAME_STEP reset compatibility, no CLI/TensorBoard/video/printing.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

from control_clock.protocol import RunContext

SOURCE_COMMIT = "e421c2e50b81febf639fced51a69e2602593d50d"


@dataclass(frozen=True)
class Args:
    """Pinned upstream algorithm defaults; device/logging are documented below."""

    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None


def make_agent(envs):
    """The upstream separate 64x64 Tanh actor/critic and initialization order."""
    import torch
    import torch.nn as nn
    from torch.distributions.categorical import Categorical

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            observation_size = int(np.prod(envs.single_observation_space.shape))
            self.critic = nn.Sequential(
                layer_init(nn.Linear(observation_size, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(observation_size, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )

        def get_value(self, x):
            return self.critic(x)

        def get_action_and_value(self, x, action=None):
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    return Agent()


def train(ctx: RunContext) -> None:
    """Run pinned CleanRL PPO; all environments close on Finished or failure."""
    args = Args()
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    model_seed = ctx.seed % 1_000_000
    reset_seeds = ctx.training_seeds(args.num_envs)
    ctx.configuration.update(
        {
            "method": "cleanrl_ppo",
            "task": ctx.task,
            "sources": {
                "cleanrl": {
                    "commit": SOURCE_COMMIT,
                    "url": f"https://github.com/vwxyzjn/cleanrl/blob/{SOURCE_COMMIT}/cleanrl/ppo.py",
                },
            },
            "ppo": asdict(args),
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "num_iterations": num_iterations,
            "effective_total_timesteps": num_iterations * batch_size,
            "policy": {
                "actor": [64, 64],
                "critic": [64, 64],
                "activation": "Tanh",
                "initialization": "orthogonal; hidden sqrt(2), actor output .01, critic output 1",
                "bias_initialization": 0.0,
            },
            "optimizer": {
                "class": "torch.optim.Adam",
                "eps": 1e-5,
                "betas": [0.9, 0.999],
                "weight_decay": 0,
            },
            "device": "cpu",
            "torch_threads": 1,
            "torch_deterministic": True,
            "model_seed": model_seed,
            "training_reset_seeds": reset_seeds,
            "seed_mapping": "model: run_seed % 1000000; reset: context training_seeds(4)",
            "training_environment": "Gymnasium + RecordEpisodeStatistics + SyncVectorEnv",
            "autoreset_mode": "SAME_STEP",
            "timeout_bootstrapping": False,
            "observation_normalization": False,
            "reward_normalization": False,
            "evaluation": "deterministic argmax; raw Gymnasium rewards",
            "checkpoint": "initial, then after all PPO epochs/minibatches in each iteration",
            "tensorboard": False,
            "capture_video": False,
            "track": False,
        }
    )
    ctx.check_time()

    import torch
    import torch.nn as nn
    import torch.optim as optim

    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cpu")

    environments = []
    envs = None
    try:
        for _ in range(args.num_envs):
            ctx.check_time()
            environments.append(gym.make(ctx.task))
            environments[-1] = gym.wrappers.RecordEpisodeStatistics(environments[-1])
        envs = gym.vector.SyncVectorEnv(
            [lambda item=item: item for item in environments],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        if not isinstance(envs.single_action_space, gym.spaces.Discrete):
            raise ValueError("CleanRL ppo.py supports only discrete action spaces")
        agent = make_agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
        logprobs = torch.zeros((args.num_steps, args.num_envs))
        rewards = torch.zeros((args.num_steps, args.num_envs))
        dones = torch.zeros((args.num_steps, args.num_envs))
        values = torch.zeros((args.num_steps, args.num_envs))

        def policy(observations: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                logits = agent.actor(torch.as_tensor(observations, dtype=torch.float32))
                return logits.argmax(dim=-1).cpu().numpy()

        ctx.checkpoint(policy, 0, force=True)
        global_step = 0
        next_obs, _ = envs.reset(seed=reset_seeds)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, num_iterations + 1):
            ctx.check_time()
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            for step in range(args.num_steps):
                ctx.check_time()
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
                ctx.steps = global_step
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(next_done).to(device)

            # Preserve upstream masking: truncations, like terminations, do not bootstrap.
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_inds = np.arange(batch_size)

            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    ctx.check_time()
                    mb_inds = b_inds[start : start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            ctx.checkpoint(policy, global_step)
    finally:
        if envs is not None:
            envs.close()
        else:
            for item in environments:
                item.close()
