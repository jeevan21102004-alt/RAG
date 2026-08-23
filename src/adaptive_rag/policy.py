from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalPolicy(nn.Module):
    """A small neural-network policy for the retrieval decision problem.

    Input: 5-dimensional state from state.py
    Output: probability of ANSWER (index 0) and SEARCH (index 1)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(state))
        logits = self.fc2(hidden)
        return F.softmax(logits, dim=-1)

    def action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities for a state tensor."""
        return self.forward(state)

    def sample_action(self, state: torch.Tensor) -> tuple[int, float]:
        """Sample an action from the policy's probability distribution.

        Returns (action, log_prob_of_action).
        Action mapping: 0 = ANSWER, 1 = SEARCH.
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.item()