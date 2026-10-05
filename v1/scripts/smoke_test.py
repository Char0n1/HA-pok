"""Non-destructive checks for the official SDK and the first RL model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_rl.features import OPTION_DIM, STATE_DIM, encode_observation, encode_options
from ptcg_rl.policy import CandidateActorCritic, select_action
from ptcg_rl.ppo import PPOBatch, ppo_update


def synthetic_observation() -> dict:
    card = {"id": 12, "hp": 220, "energies": [1, 1]}
    player = {"active": [card], "bench": [], "hand": [{"id": 31}], "handCount": 1, "deckCount": 52, "prize": [None] * 6, "discard": [], "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False}
    return {
        "logs": [],
        "current": {"turn": 1, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False, "retreated": False, "result": -1, "stadium": None, "looking": None, "players": [player, player]},
        "select": {"minCount": 1, "maxCount": 1, "option": [{"type": 13, "attackId": 3}, {"type": 14}]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    args = parser.parse_args()
    sdk_path = args.competition_root / "sample_submission" / "sample_submission" / "sample_submission"
    sys.path.insert(0, str(sdk_path))
    from cg.api import all_card_data  # Imported only after the official SDK path is configured.
    from cg.game import battle_finish, battle_start

    catalogue = all_card_data()
    assert catalogue, "Official CABT card catalogue could not be loaded."
    deck = [int(value) for value in (sdk_path / "deck.csv").read_text(encoding="utf-8").split()]
    observation, _ = battle_start(deck, deck)
    assert observation is not None, "CABT could not start a game with the supplied sample deck."
    try:
        state = encode_observation(observation)
        options = encode_options(observation)
        assert state.shape == (STATE_DIM,)
        assert options.shape[1] == OPTION_DIM and len(options) > 0
        model = CandidateActorCritic()
        chosen = select_action(model, observation)
        assert len(chosen) == 1 and 0 <= chosen[0] < len(options)
    finally:
        battle_finish()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch = PPOBatch(
        states=torch.from_numpy(state).repeat(4, 1),
        options=torch.from_numpy(options).unsqueeze(0).repeat(4, 1, 1),
        masks=torch.ones((4, len(options)), dtype=torch.bool),
        actions=torch.zeros(4, dtype=torch.long),
        old_log_probs=torch.zeros(4),
        returns=torch.tensor([1.0, -1.0, 0.5, -0.5]),
        advantages=torch.tensor([1.0, -1.0, 0.5, -0.5]),
    )
    metrics = ppo_update(model, optimizer, batch)
    print(f"CABT cards: {len(catalogue)}")
    print(f"state={state.shape}; options={options.shape}; selected={chosen}; ppo={metrics}")


if __name__ == "__main__":
    main()
