"""Policy network for RL configuration selection (Phase 4).

A small PyTorch MLP mapping the optimization-state vector to a
probability distribution over configuration action IDs::

    Input (state dim)
        -> Linear(state_dim, hidden_dim)
        -> ReLU
        -> Linear(hidden_dim, n_actions)   # logits
        -> Softmax                          # action probabilities

For the pilot space the output dimension is **175**.  The policy outputs
an ACTION ID only — never retrieval parameters directly.

Invalid-action masking: :func:`mask_invalid_actions` sets the logits of
already-evaluated configurations to ``-inf`` before the softmax, so their
probabilities become exactly zero.  The environment still validates every
action independently; the mask is an efficiency aid, not the only guard.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimization_state import OPTIMIZATION_STATE_DIM

DEFAULT_HIDDEN_DIM: int = 128


def mask_invalid_actions(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask invalid action logits to ``-inf`` so softmax yields zeros.

    Parameters
    ----------
    logits:
        Tensor of shape ``(batch, n_actions)`` or ``(n_actions,)``.
    valid_mask:
        Boolean tensor of the same shape where ``True`` marks *valid*
        (still selectable) actions.
    """
    mask = valid_mask.to(dtype=torch.bool, device=logits.device)
    masked = logits.clone()
    masked[~mask] = float("-inf")
    return masked


class ConfigurationPolicy(nn.Module):
    """MLP policy over configuration action IDs."""

    def __init__(
        self,
        n_actions: int,
        state_dim: int = OPTIMIZATION_STATE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, n_actions)``."""
        hidden = F.relu(self.fc1(state))
        return self.fc2(hidden)

    def probabilities(
        self,
        state: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Action probabilities with optional invalid-action masking."""
        logits = self.forward(state)
        if valid_mask is not None:
            logits = mask_invalid_actions(logits, valid_mask)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float]:
        """Select one action.

        Returns ``(action_id, log_probability_of_action)``.  In
        ``deterministic`` mode the highest-probability valid action is
        chosen (used at evaluation time); otherwise the action is sampled
        from the masked distribution (used during training).
        """
        probs = self.probabilities(state, valid_mask)
        if deterministic:
            action_id = int(torch.argmax(probs, dim=-1).item())
        else:
            action_id = int(torch.multinomial(probs, num_samples=1).item())
        log_prob = torch.log(probs[0, action_id] + 1e-12)
        return action_id, float(log_prob.item())