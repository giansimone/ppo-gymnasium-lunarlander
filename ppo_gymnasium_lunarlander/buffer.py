"""
Rollout buffer for storing experiences in PPO algorithm.
"""
from typing import Tuple

import torch
import numpy as np


class RolloutBuffer:
    """A simple rollout buffer for PPO."""
    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        state_dim: int,
        action_dim: int,
    ):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = torch.zeros((n_steps, num_envs, state_dim))
        self.actions = torch.zeros((n_steps, num_envs), dtype=torch.int64)
        self.log_probs = torch.zeros((n_steps, num_envs))
        self.rewards = torch.zeros((n_steps, num_envs))
        self.values = torch.zeros((n_steps, num_envs))
        self.dones = torch.zeros((n_steps, num_envs), dtype=torch.int64)

        self.step_idx = 0


    def clear(self):
        """Clears the buffer."""
        self.step_idx = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: np.ndarray,
        value: torch.Tensor,
        done: np.ndarray,
    ):
        """Adds a new experience to the buffer."""
        if self.step_idx >= self.n_steps:
            raise IndexError("Rollout buffer is full.")

        self.states[self.step_idx] = state
        self.actions[self.step_idx] = action
        self.log_probs[self.step_idx] = log_prob
        self.rewards[self.step_idx] = torch.tensor(reward, dtype=torch.float32)
        self.values[self.step_idx] = value
        self.dones[self.step_idx] = torch.tensor(done, dtype=torch.int64)

        self.step_idx += 1

    def get(self) -> Tuple[torch.Tensor, ...]:
        """Returns all experiences from the buffer."""
        if self.step_idx != self.n_steps:
            print(f"Warning: Buffer not full. Only {self.step_idx} steps collected.")

        states = self.states.reshape(-1, self.state_dim)
        actions = self.actions.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        values = self.values.reshape(-1)

        return states, actions, log_probs, self.rewards, values, self.dones
