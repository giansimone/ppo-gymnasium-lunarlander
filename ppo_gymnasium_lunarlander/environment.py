"""
Module to create and manage the Lunar Lander environment.
"""
from typing import Tuple

import gymnasium as gym


def make_env(env_id: str, render_mode=None) -> Tuple[gym.Env, int, int]:
    "Create and wrap environment."
    env = gym.make(env_id, render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    return env, state_dim, action_dim


def make_vec_env(
    env_id: str,
    num_envs: int,
    render_mode=None
) -> Tuple[gym.vector.VectorEnv, int, int]:
    "Create and wrap environment."
    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="async",
        render_mode=render_mode,
    )

    env_state_dim = envs.single_observation_space.shape[0]
    env_action_dim = envs.single_action_space.n

    return envs, env_state_dim, env_action_dim
