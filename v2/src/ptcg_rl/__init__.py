"""Minimal components for a legal-option PTCG reinforcement-learning agent."""

from .features import OPTION_DIM, STATE_DIM, encode_observation, encode_options
from .policy import CandidateActorCritic, select_action

__all__ = [
    "OPTION_DIM",
    "STATE_DIM",
    "CandidateActorCritic",
    "encode_observation",
    "encode_options",
    "select_action",
]
