"""Train the candidate actor--critic from collected CABT self-play decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_rl.policy import CandidateActorCritic
from ptcg_rl.ppo import PPOBatch, ppo_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, default=ROOT / "artifacts" / "selfplay_v1.pt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "policy_v1.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    rollout = torch.load(args.rollout, map_location="cpu", weights_only=True)
    model = CandidateActorCritic()
    if 'behavior_state_dict' not in rollout:
        raise ValueError('Rollout lacks its behavior policy. Recollect; random initialization is not valid PPO.')
    model.load_state_dict(rollout['behavior_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    advantages = rollout["returns"] - rollout["values"]
    batch = PPOBatch(
        states=rollout["states"].float(),
        options=rollout["options"].float(),
        masks=rollout["masks"].bool(),
        actions=rollout["actions"].long(),
        old_log_probs=rollout["old_log_probs"].float(),
        returns=rollout["returns"].float(),
        advantages=advantages.float(),
    )
    metrics: dict[str, float] = {}
    for _ in range(args.epochs):
        metrics = ppo_update(model, optimizer, batch)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epochs": args.epochs, "metrics": metrics}, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}; final metrics={metrics}")


if __name__ == "__main__":
    main()
