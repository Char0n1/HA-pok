"""Verify the built archive structure and run one full local CABT game."""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, default=Path("submission"))
    parser.add_argument("--archive", type=Path, default=Path("submission/submission.tar.gz"))
    args = parser.parse_args()
    args.competition_root = args.competition_root.resolve()
    args.submission_dir = args.submission_dir.resolve()
    args.archive = args.archive.resolve()
    expected = {"main.py", "deck.csv", "model.npz"}
    with tarfile.open(args.archive, "r:gz") as archive:
        actual = {member.name for member in archive.getmembers() if member.isfile()}
        for member in archive.getmembers():
            if member.name not in expected or not member.isfile():
                raise RuntimeError('Unexpected archive member')
            if archive.extractfile(member).read() != (args.submission_dir / member.name).read_bytes():
                raise RuntimeError('Archive differs from the tested submission directory')
    if actual != expected:
        raise RuntimeError(f"Archive files must be exactly {sorted(expected)}, got {sorted(actual)}")

    # Reproduce Kaggle's raw exec agent loader: do not inject __file__.
    original_cwd = Path.cwd()
    os.chdir(args.submission_dir)
    module_globals: dict[str, object] = {"__name__": "submission_agent"}
    try:
        exec(compile((args.submission_dir / "main.py").read_text(encoding="utf-8"), "main.py", "exec"), module_globals)
    finally:
        os.chdir(original_cwd)

    sdk_path = args.competition_root / "sample_submission" / "sample_submission" / "sample_submission"
    sys.path.insert(0, str(sdk_path))
    from cg.game import battle_finish, battle_select, battle_start

    os.chdir(args.submission_dir)
    deck = module_globals["read_deck"]()
    if len(deck) != 60:
        raise RuntimeError(f"deck.csv contains {len(deck)} cards, not 60")
    observation, _ = battle_start(deck, deck)
    if observation is None:
        raise RuntimeError("CABT rejected deck.csv")
    try:
        steps = 0
        while observation["current"]["result"] == -1 and steps < 2_000:
            action = module_globals["agent"](observation)
            select = observation["select"]
            if not select["minCount"] <= len(action) <= select["maxCount"]:
                raise RuntimeError(f"Invalid action length at step {steps}: {len(action)}")
            if len(action) != len(set(action)) or any(index < 0 or index >= len(select["option"]) for index in action):
                raise RuntimeError(f"Invalid action indices at step {steps}: {action}")
            observation = battle_select(action)
            steps += 1
        if observation["current"]["result"] == -1:
            raise RuntimeError("Agent did not finish a game within 2,000 selections")
    finally:
        battle_finish()
        os.chdir(original_cwd)
    print(f"Submission verified: files={sorted(actual)}, deck=60, game_steps={steps}, result={observation['current']['result']}")


if __name__ == "__main__":
    main()
