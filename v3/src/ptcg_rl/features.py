"""Feature extraction from the raw CABT observation dictionary.

The functions deliberately consume dictionaries rather than SDK dataclasses so
they can be used both in Kaggle's `agent(obs_dict)` callback and in local data
collection. Unknown or newly-added SDK fields are ignored safely.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

CARD_BUCKETS = 16
GLOBAL_DIM = 12
PLAYER_SCALARS = 10
STATE_DIM = GLOBAL_DIM + 2 * (PLAYER_SCALARS + 3 * CARD_BUCKETS)
OPTION_DIM = 80


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value: Any, scale: float = 1.0) -> float:
    try:
        return float(value) / scale
    except (TypeError, ValueError):
        return 0.0


def _card_id(value: Any) -> int:
    if value is None:
        return 0
    candidate = _get(value, "id", _get(value, "cardId", 0))
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return 0


def _bucket(values: list[Any]) -> np.ndarray:
    result = np.zeros(CARD_BUCKETS, dtype=np.float32)
    for card in values or []:
        card_id = _card_id(card)
        if card_id:
            result[card_id % CARD_BUCKETS] += 1.0
    return np.clip(result / 4.0, 0.0, 1.0)


def _pokemon_scalars(pokemon: Any) -> tuple[float, float]:
    if pokemon is None:
        return 0.0, 0.0
    hp = _number(_get(pokemon, "hp", 0), 400.0)
    energies = _get(pokemon, "energies", []) or []
    return hp, min(len(energies) / 8.0, 1.0)


def _player_features(player: Any) -> np.ndarray:
    active = (_get(player, "active", []) or [None])
    active = active[0] if active else None
    active_hp, active_energy = _pokemon_scalars(active)
    bench = _get(player, "bench", []) or []
    bench_hp = sum(_pokemon_scalars(item)[0] for item in bench) / 5.0
    bench_energy = sum(_pokemon_scalars(item)[1] for item in bench) / 5.0
    scalar = np.array(
        [
            _number(_get(player, "handCount", len(_get(player, "hand", []) or [])), 20.0),
            _number(_get(player, "deckCount", 0), 60.0),
            len(_get(player, "prize", []) or []) / 6.0,
            len(_get(player, "discard", []) or []) / 60.0,
            len(bench) / 5.0,
            active_hp,
            active_energy,
            bench_hp,
            bench_energy,
            sum(bool(_get(player, flag, False)) for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused")) / 5.0,
        ],
        dtype=np.float32,
    )
    cards = _get(player, "hand", []) or []
    discard = _get(player, "discard", []) or []
    in_play = [active] + list(bench)
    return np.concatenate((scalar, _bucket(cards), _bucket(discard), _bucket(in_play)))


def encode_observation(observation: Mapping[str, Any] | Any) -> np.ndarray:
    """Encode only information that the agent can receive from CABT."""
    current = _get(observation, "current")
    if current is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    players = _get(current, "players", []) or []
    your_index = int(_get(current, "yourIndex", 0) or 0)
    own = players[your_index] if len(players) > your_index else {}
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if len(players) > opponent_index else {}
    global_features = np.array(
        [
            _number(_get(current, "turn", 0), 30.0),
            _number(_get(current, "turnActionCount", 0), 20.0),
            float(your_index),
            _number(_get(current, "firstPlayer", -1) + 1, 2.0),
            float(bool(_get(current, "supporterPlayed", False))),
            float(bool(_get(current, "stadiumPlayed", False))),
            float(bool(_get(current, "energyAttached", False))),
            float(bool(_get(current, "retreated", False))),
            float(_get(current, "result", -1) not in (None, -1)),
            len(_get(current, "looking", []) or []) / 10.0,
            float(_get(current, "stadium", None) is not None),
            min(len(_get(observation, "logs", []) or []) / 20.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate((global_features, _player_features(own), _player_features(opponent)))


def _hash_feature(vector: np.ndarray, index: int, value: Any) -> None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return
    vector[index + abs(number) % 8] = 1.0


def encode_options(observation: Mapping[str, Any] | Any) -> np.ndarray:
    """Return one fixed-width vector for every currently legal candidate."""
    select = _get(observation, "select")
    options = _get(select, "option", []) if select is not None else []
    result = np.zeros((len(options), OPTION_DIM), dtype=np.float32)
    fields = ("type", "area", "inPlayArea", "playerIndex", "context", "cardId", "attackId", "index", "inPlayIndex")
    for row, option in enumerate(options):
        for field_index, field in enumerate(fields):
            _hash_feature(result[row], field_index * 8, _get(option, field, 0))
        card_id = _card_id(option)
        if card_id:
            result[row, 72 + card_id % 8] = 1.0
    return result
