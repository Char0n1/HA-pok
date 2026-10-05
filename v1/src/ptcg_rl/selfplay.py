"""Direct-CABT self-play collection without requiring kaggle-environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features import encode_observation, encode_options
from .policy import CandidateActorCritic, sample_single_action, select_action


@dataclass
class Decision:
    state: np.ndarray
    options: np.ndarray
    action: int
    old_log_prob: float
    value: float
    actor_index: int
    return_value: float = 0.0


def _field(source: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return source.get(key, default)


def collect_game(model: CandidateActorCritic, sdk_path: Path, deck: list[int], max_steps: int = 2_000, device: str = "cpu") -> list[Decision]:
    """Play one game, retaining valid single-choice decisions for PPO v1."""
    import sys

    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))
    from cg.game import battle_finish, battle_select, battle_start

    observation, _ = battle_start(deck, deck)
    if observation is None:
        raise RuntimeError("CABT rejected the supplied deck.")
    decisions: list[Decision] = []
    try:
        for _ in range(max_steps):
            current = _field(observation, "current", {})
            if _field(current, "result", -1) != -1:
                winner = int(_field(current, "result", -1))
                for decision in decisions:
                    decision.return_value = 1.0 if decision.actor_index == winner else -1.0
                return decisions
            select = _field(observation, "select", {})
            minimum = int(_field(select, "minCount", 0) or 0)
            maximum = int(_field(select, "maxCount", 0) or 0)
            if minimum == maximum == 1:
                action, old_log_prob, value = sample_single_action(model, observation, device)
                decisions.append(Decision(encode_observation(observation), encode_options(observation), action, old_log_prob, value, int(_field(current, "yourIndex", 0))))
                selected = [action]
            else:
                selected = select_action(model, observation, device)
            observation = battle_select(selected)
    finally:
        battle_finish()
    raise RuntimeError(f"CABT did not finish within {max_steps} selections.")


def save_decisions(decisions: list[Decision], output: Path) -> None:
    """Store variable-length legal action sets in a torch-readable file."""
    if not decisions:
        raise ValueError("No single-choice decisions were collected.")
    max_options = max(len(item.options) for item in decisions)
    option_dim = decisions[0].options.shape[1]
    options = np.zeros((len(decisions), max_options, option_dim), dtype=np.float32)
    masks = np.zeros((len(decisions), max_options), dtype=bool)
    for row, item in enumerate(decisions):
        options[row, : len(item.options)] = item.options
        masks[row, : len(item.options)] = True
    returns = np.array([item.return_value for item in decisions], dtype=np.float32)
    payload = {
        "states": torch.from_numpy(np.stack([item.state for item in decisions])),
        "options": torch.from_numpy(options),
        "masks": torch.from_numpy(masks),
        "actions": torch.tensor([item.action for item in decisions], dtype=torch.long),
        "old_log_probs": torch.tensor([item.old_log_prob for item in decisions], dtype=torch.float32),
        "returns": torch.from_numpy(returns),
        "values": torch.tensor([item.value for item in decisions], dtype=torch.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
