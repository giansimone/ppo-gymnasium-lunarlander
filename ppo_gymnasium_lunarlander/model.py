"""
Actor-Critic Network for PPO agent in Lunar Lander environment.
"""
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic Network for discrete action space."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, action_dim)

        self.critic_head = nn.Linear(hidden_dim, 1)


    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = self.body(state)

        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        state_value = self.critic_head(x)

        return action_probs, state_value

    def act(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action based on the current policy."""
        action_probs, state_value = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        action_log_prob = dist.log_prob(action)

        return action.detach(), action_log_prob.detach(), state_value.detach()

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action log probabilities and state value."""
        action_probs, state_value = self.forward(state)

        dist = Categorical(action_probs)

        action_log_prob = dist.log_prob(action)

        dist_entropy = dist.entropy()

        return action_log_prob, state_value, dist_entropy
