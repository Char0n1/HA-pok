"""Kaggle submission agent: NumPy inference for the trained PPO-v1 policy."""

import os

import numpy as np

CARD_BUCKETS = 16
STATE_DIM = 128
OPTION_DIM = 80


def get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def number(value, scale=1.0):
    try:
        return float(value) / scale
    except (TypeError, ValueError):
        return 0.0


def card_id(value):
    if value is None:
        return 0
    try:
        return int(get(value, "id", get(value, "cardId", 0)))
    except (TypeError, ValueError):
        return 0


def bucket(cards):
    result = np.zeros(CARD_BUCKETS, dtype=np.float32)
    for card in cards or []:
        ident = card_id(card)
        if ident:
            result[ident % CARD_BUCKETS] += 1.0
    return np.clip(result / 4.0, 0.0, 1.0)


def pokemon_scalars(pokemon):
    if pokemon is None:
        return 0.0, 0.0
    return number(get(pokemon, "hp", 0), 400.0), min(len(get(pokemon, "energies", []) or []) / 8.0, 1.0)


def player_features(player):
    active_list = get(player, "active", []) or []
    active = active_list[0] if active_list else None
    active_hp, active_energy = pokemon_scalars(active)
    bench = get(player, "bench", []) or []
    scalar = np.array([
        number(get(player, "handCount", len(get(player, "hand", []) or [])), 20.0), number(get(player, "deckCount", 0), 60.0),
        len(get(player, "prize", []) or []) / 6.0, len(get(player, "discard", []) or []) / 60.0, len(bench) / 5.0,
        active_hp, active_energy, sum(pokemon_scalars(item)[0] for item in bench) / 5.0,
        sum(pokemon_scalars(item)[1] for item in bench) / 5.0,
        sum(bool(get(player, flag, False)) for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused")) / 5.0,
    ], dtype=np.float32)
    return np.concatenate((scalar, bucket(get(player, "hand", []) or []), bucket(get(player, "discard", []) or []), bucket([active] + list(bench))))


def encode_state(obs):
    current = get(obs, "current")
    if current is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    players = get(current, "players", []) or []
    your_index = int(get(current, "yourIndex", 0) or 0)
    own = players[your_index] if len(players) > your_index else {}
    opponent_index = 1 - your_index
    opponent = players[opponent_index] if len(players) > opponent_index else {}
    glob = np.array([
        number(get(current, "turn", 0), 30.0), number(get(current, "turnActionCount", 0), 20.0), float(your_index),
        number(get(current, "firstPlayer", -1) + 1, 2.0), float(bool(get(current, "supporterPlayed", False))),
        float(bool(get(current, "stadiumPlayed", False))), float(bool(get(current, "energyAttached", False))),
        float(bool(get(current, "retreated", False))), float(get(current, "result", -1) not in (None, -1)),
        len(get(current, "looking", []) or []) / 10.0, float(get(current, "stadium") is not None),
        min(len(get(obs, "logs", []) or []) / 20.0, 1.0),
    ], dtype=np.float32)
    return np.concatenate((glob, player_features(own), player_features(opponent)))


def encode_options(obs):
    select = get(obs, "select", {})
    options = get(select, "option", []) or []
    result = np.zeros((len(options), OPTION_DIM), dtype=np.float32)
    fields = ("type", "area", "inPlayArea", "playerIndex", "context", "cardId", "attackId", "index", "inPlayIndex")
    for row, option in enumerate(options):
        for field_index, field in enumerate(fields):
            try:
                value = int(get(option, field, 0))
                result[row, field_index * 8 + abs(value) % 8] = 1.0
            except (TypeError, ValueError):
                pass
        ident = card_id(option)
        if ident:
            result[row, 72 + ident % 8] = 1.0
    return result


WEIGHTS = None


def resource_path(filename):
    """Find an uploaded file in both local and Kaggle raw-exec contexts.

    Kaggle loads competition agents with `exec`, where `__file__` is not
    guaranteed to exist. The documented agent directory is the fallback.
    """
    for path in (filename, os.path.join("/kaggle_simulations/agent", filename)):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Missing submission resource: " + filename)


def load_weights():
    global WEIGHTS
    if WEIGHTS is None:
        WEIGHTS = np.load(resource_path("model.npz"))
    return WEIGHTS


def linear(x, name):
    weights = load_weights()
    return x @ weights[name + ".weight"].T + weights[name + ".bias"]


def choose(obs):
    select = get(obs, "select")
    options = get(select, "option", []) or []
    minimum = int(get(select, "minCount", 0) or 0)
    maximum = int(get(select, "maxCount", minimum) or minimum)
    if minimum == 0:
        return []
    state = encode_state(obs)
    candidate = encode_options(obs)
    state_embedding = np.tanh(linear(np.tanh(linear(state, "state_encoder.0")), "state_encoder.2"))
    option_embedding = np.tanh(linear(candidate, "option_encoder.0"))
    repeated_state = np.repeat(state_embedding[None, :], len(options), axis=0)
    score_hidden = np.tanh(linear(np.concatenate((repeated_state, option_embedding), axis=1), "logit_head.0"))
    logits = linear(score_hidden, "logit_head.2").reshape(-1)
    count = min(minimum, maximum, len(options))
    return np.argsort(-logits, kind="stable")[:count].astype(int).tolist()


def read_deck():
    with open(resource_path("deck.csv"), encoding="utf-8") as file:
        return [int(line.strip()) for line in file if line.strip()]


def agent(obs_dict):
    if get(obs_dict, "select") is None:
        return read_deck()
    return choose(obs_dict)
