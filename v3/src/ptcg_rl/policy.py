"""Candidate-scoring actor--critic for CABT's changing legal action sets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .features import OPTION_DIM, STATE_DIM, encode_observation, encode_options


class CandidateActorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128, state_dim: int = STATE_DIM, option_dim: int = OPTION_DIM) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.option_encoder = nn.Sequential(nn.Linear(option_dim, hidden_dim), nn.Tanh())
        self.logit_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, state: Tensor, options: Tensor, option_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Score padded options. Shapes: state=[B,S], options=[B,A,O]."""
        encoded_state = self.state_encoder(state)
        encoded_option = self.option_encoder(options)
        repeated_state = encoded_state.unsqueeze(1).expand(-1, options.shape[1], -1)
        logits = self.logit_head(torch.cat((repeated_state, encoded_option), dim=-1)).squeeze(-1)
        if option_mask is not None:
            logits = logits.masked_fill(~option_mask.bool(), torch.finfo(logits.dtype).min)
        return logits, self.value_head(encoded_state).squeeze(-1)


@torch.inference_mode()
def select_action(model: CandidateActorCritic, observation: Mapping[str, Any] | Any, device: str = "cpu") -> list[int]:
    """Choose a conservative legal action set from CABT's supplied candidates."""
    select = observation.get("select") if isinstance(observation, Mapping) else getattr(observation, "select", None)
    if select is None:
        raise ValueError("Deck selection is handled by the Kaggle agent wrapper, not the policy.")
    get = (lambda name, default: select.get(name, default)) if isinstance(select, Mapping) else (lambda name, default: getattr(select, name, default))
    option_count = len(get("option", []))
    minimum = int(get("minCount", 1) or 0)
    maximum = int(get("maxCount", minimum) or minimum)
    if option_count < minimum:
        raise ValueError("CABT supplied fewer candidates than select.minCount.")
    if minimum == 0:
        return []
    state = torch.from_numpy(encode_observation(observation)).to(device).unsqueeze(0)
    options = torch.from_numpy(encode_options(observation)).to(device).unsqueeze(0)
    logits, _ = model(state, options)
    chosen = torch.topk(logits[0], k=min(minimum, maximum, option_count)).indices.tolist()
    return [int(index) for index in chosen]


@torch.inference_mode()
def sample_single_action(model: CandidateActorCritic, observation: Mapping[str, Any] | Any, device: str = "cpu") -> tuple[int, float, float]:
    """Sample one action and return (index, log-probability, state value).

    PPO v1 deliberately learns only decision points with exactly one required
    choice. CABT is still advanced at all other decisions by `select_action`.
    """
    select = observation.get("select") if isinstance(observation, Mapping) else getattr(observation, "select", None)
    if select is None:
        raise ValueError("Deck selection is not an RL action.")
    get = (lambda name, default: select.get(name, default)) if isinstance(select, Mapping) else (lambda name, default: getattr(select, name, default))
    if int(get("minCount", 0) or 0) != 1 or int(get("maxCount", 0) or 0) != 1:
        raise ValueError("sample_single_action requires a single-choice CABT decision.")
    state = torch.from_numpy(encode_observation(observation)).to(device).unsqueeze(0)
    options = torch.from_numpy(encode_options(observation)).to(device).unsqueeze(0)
    logits, value = model(state, options)
    distribution = torch.distributions.Categorical(logits=logits[0])
    action = distribution.sample()
    return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())
