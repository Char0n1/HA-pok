"""Collect one local CABT self-play game for the PPO data pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_rl.policy import CandidateActorCritic
from ptcg_rl.selfplay import collect_game, save_decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "selfplay_v1.pt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    sdk_path = args.competition_root / "sample_submission" / "sample_submission" / "sample_submission"
    deck = [int(value) for value in (sdk_path / "deck.csv").read_text(encoding="utf-8").split()]
    model = CandidateActorCritic().eval()
    decisions = collect_game(model, sdk_path, deck)
    save_decisions(decisions, args.output)
    payload = torch.load(args.output, weights_only=True)
    payload['behavior_state_dict'] = model.state_dict()
    torch.save(payload, args.output)
    print(f"Saved {len(decisions)} PPO-v1 decisions to {args.output}")


if __name__ == "__main__":
    main()
