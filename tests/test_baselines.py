import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from control_clock.cleanrl_baseline import Args, make_agent
from control_clock.library_baselines import linear_schedule, load_settings
from control_clock.protocol import Finished, RunContext


def test_zoo_task_settings_and_schedules():
    cart = load_settings("CartPole-v1")
    acrobot = load_settings("Acrobot-v1")
    lunar = load_settings("LunarLander-v3")
    assert (cart["n_envs"], cart["ppo"]["n_steps"], cart["ppo"]["batch_size"]) == (8, 32, 256)
    assert cart["ppo"]["n_epochs"] == 20
    assert cart["ppo"]["learning_rate"] == "lin_0.001"
    assert cart["vec_normalize"] is None
    assert acrobot["vec_normalize"]["norm_obs"]
    assert acrobot["vec_normalize"]["norm_reward"]
    assert acrobot["ppo"]["gae_lambda"] == 0.94
    assert lunar["ppo"]["gamma"] == 0.999
    assert lunar["ppo"]["ent_coef"] == 0.01
    assert linear_schedule(0.001)(0.5) == 0.0005


def test_sb3_evaluation_normalization_is_frozen():
    env = VecNormalize(DummyVecEnv([lambda: gym.make("Acrobot-v1")]))
    try:
        env.reset()
        count = env.obs_rms.count
        mean = env.obs_rms.mean.copy()
        variance = env.obs_rms.var.copy()
        observations = np.ones((100, 6), dtype=np.float32)
        normalized = env.normalize_obs(observations)
        assert normalized.shape == observations.shape
        assert env.obs_rms.count == count
        np.testing.assert_array_equal(env.obs_rms.mean, mean)
        np.testing.assert_array_equal(env.obs_rms.var, variance)
    finally:
        env.close()


def test_cleanrl_defaults_and_same_step_reset():
    args = Args()
    assert args.num_envs == 4 and args.num_steps == 128
    assert args.total_timesteps // 512 * 512 == 499712
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1", max_episode_steps=1)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        observations, _ = envs.reset(seed=[7])
        agent = make_agent(envs)
        with torch.no_grad():
            actions = agent.actor(torch.as_tensor(observations)).argmax(-1).numpy()
        _, rewards, _, truncated, infos = envs.step(actions)
        assert rewards[0] == 1
        assert truncated[0]
        assert "final_obs" in infos
        _, rewards, _, truncated, _ = envs.step(actions)
        # NEXT_STEP would instead ignore this action and return zero reward.
        assert rewards[0] == 1
        assert truncated[0]
    finally:
        envs.close()


def test_sb3_checkpoint_follows_all_optimizer_epochs(monkeypatch):
    import time

    from stable_baselines3 import PPO

    from control_clock.library_baselines import train

    events = []
    optimizer_steps = [0]
    original_train = PPO.train
    original_step = torch.optim.Adam.step

    def recorded_step(optimizer, *args, **kwargs):
        optimizer_steps[0] += 1
        return original_step(optimizer, *args, **kwargs)

    def recorded_train(model):
        events.append("train begin")
        original_train(model)
        assert model._n_updates == 20
        events.append("train end")

    class WiringContext(RunContext):
        def checkpoint(self, policy, steps, *, force=False):
            assert policy(np.zeros((100, 4), dtype=np.float32)).shape == (100,)
            if steps == 0:
                events.append("initial checkpoint")
                return
            assert steps == 256
            assert events[-1] == "train end"
            assert optimizer_steps[0] == 20
            events.append("updated checkpoint")
            raise Finished("wiring test complete")

    monkeypatch.setattr(torch.optim.Adam, "step", recorded_step)
    monkeypatch.setattr(PPO, "train", recorded_train)
    ctx = WiringContext("CartPole-v1", 0, time.perf_counter(), 120)
    with pytest.raises(Finished, match="wiring test complete"):
        train(ctx)
    assert events == ["initial checkpoint", "train begin", "train end", "updated checkpoint"]


def test_cleanrl_checkpoint_follows_all_optimizer_epochs(monkeypatch):
    import time

    from control_clock.cleanrl_baseline import train

    args = Args()
    optimizer_steps = [0]
    closed = [0]
    checkpoints = []
    original_step = torch.optim.Adam.step
    original_close = gym.vector.SyncVectorEnv.close

    def recorded_step(optimizer, *step_args, **kwargs):
        optimizer_steps[0] += 1
        return original_step(optimizer, *step_args, **kwargs)

    def recorded_close(envs, *close_args, **kwargs):
        closed[0] += 1
        return original_close(envs, *close_args, **kwargs)

    class WiringContext(RunContext):
        def checkpoint(self, policy, steps, *, force=False):
            assert policy(np.zeros((100, 4), dtype=np.float32)).shape == (100,)
            checkpoints.append((steps, optimizer_steps[0]))
            if steps:
                raise Finished("wiring test complete")

    monkeypatch.setattr(torch.optim.Adam, "step", recorded_step)
    monkeypatch.setattr(gym.vector.SyncVectorEnv, "close", recorded_close)
    ctx = WiringContext("CartPole-v1", 0, time.perf_counter(), 120)
    with pytest.raises(Finished, match="wiring test complete"):
        train(ctx)
    batch = args.num_envs * args.num_steps
    assert checkpoints == [(0, 0), (batch, args.update_epochs * args.num_minibatches)]
    assert closed[0] == 1
