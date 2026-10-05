"""Stream CABT self-play into PPO updates without retaining all games on disk."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_rl.policy import CandidateActorCritic
from ptcg_rl.ppo import PPOBatch, ppo_update
from ptcg_rl.selfplay import Decision, collect_game


def make_batch(decisions: list[Decision]) -> PPOBatch:
    max_options = max(len(item.options) for item in decisions)
    option_dim = decisions[0].options.shape[1]
    options = np.zeros((len(decisions), max_options, option_dim), dtype=np.float32)
    masks = np.zeros((len(decisions), max_options), dtype=bool)
    for row, item in enumerate(decisions):
        options[row, : len(item.options)] = item.options
        masks[row, : len(item.options)] = True
    returns = torch.tensor([item.return_value for item in decisions], dtype=torch.float32)
    values = torch.tensor([item.value for item in decisions], dtype=torch.float32)
    return PPOBatch(
        states=torch.from_numpy(np.stack([item.state for item in decisions])),
        options=torch.from_numpy(options),
        masks=torch.from_numpy(masks),
        actions=torch.tensor([item.action for item in decisions], dtype=torch.long),
        old_log_probs=torch.tensor([item.old_log_prob for item in decisions], dtype=torch.float32),
        returns=returns,
        advantages=returns - values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--rollout-decisions", type=int, default=1_024)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "policy_10k_selfplay.pt")
    args = parser.parse_args()
    if args.games < 1 or args.rollout_decisions < 1 or args.ppo_epochs < 1:
        raise ValueError("games, rollout-decisions, and ppo-epochs must be positive.")

    torch.manual_seed(args.seed)
    sdk_path = args.competition_root / "sample_submission" / "sample_submission" / "sample_submission"
    deck = [int(value) for value in (sdk_path / "deck.csv").read_text(encoding="utf-8").split()]
    model = CandidateActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    buffer: list[Decision] = []
    all_metrics: list[dict[str, float]] = []
    wins = [0, 0]
    decision_count = 0
    started = time.perf_counter()

    def update_if_ready(force: bool = False) -> None:
        nonlocal buffer
        if not buffer or (not force and len(buffer) < args.rollout_decisions):
            return
        batch = make_batch(buffer)
        for _ in range(args.ppo_epochs):
            all_metrics.append(ppo_update(model, optimizer, batch))
        buffer = []

    for game_number in range(1, args.games + 1):
        model.eval()
        episode = collect_game(model, sdk_path, deck)
        model.train()
        buffer.extend(episode)
        decision_count += len(episode)
        winner = episode[0].actor_index if episode[0].return_value > 0 else 1 - episode[0].actor_index
        wins[winner] += 1
        update_if_ready()
        if game_number % 1_000 == 0 or game_number == args.games:
            elapsed = time.perf_counter() - started
            print(
                f"games={game_number}/{args.games} decisions={decision_count} "
                f"games_per_sec={game_number / elapsed:.2f} wins={wins} updates={len(all_metrics)}",
                flush=True,
            )

    update_if_ready(force=True)
    elapsed = time.perf_counter() - started
    mean_metrics = {key: float(np.mean([item[key] for item in all_metrics])) for key in all_metrics[0]} if all_metrics else {}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "games": args.games,
            "decisions": decision_count,
            "wins": wins,
            "elapsed_seconds": elapsed,
            "mean_metrics": mean_metrics,
        },
        args.checkpoint,
    )
    print(f"Saved {args.checkpoint}; elapsed={elapsed:.1f}s; mean_metrics={mean_metrics}", flush=True)


if __name__ == "__main__":
    main()
