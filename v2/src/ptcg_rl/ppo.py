"""Small PPO update routine for externally collected CABT trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .policy import CandidateActorCritic


@dataclass
class PPOBatch:
    states: Tensor          # [B, STATE_DIM]
    options: Tensor         # [B, MAX_OPTIONS, OPTION_DIM]
    masks: Tensor           # [B, MAX_OPTIONS]
    actions: Tensor         # [B] -- single-selection samples only for v1
    old_log_probs: Tensor   # [B]
    returns: Tensor         # [B]
    advantages: Tensor      # [B]


def ppo_update(model: CandidateActorCritic, optimizer: torch.optim.Optimizer, batch: PPOBatch, clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01) -> dict[str, float]:
    logits, values = model(batch.states, batch.options, batch.masks)
    distribution = torch.distributions.Categorical(logits=logits)
    log_probs = distribution.log_prob(batch.actions)
    ratio = torch.exp(log_probs - batch.old_log_probs)
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std(unbiased=False) + 1e-8)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, batch.returns)
    entropy = distribution.entropy().mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach())}
