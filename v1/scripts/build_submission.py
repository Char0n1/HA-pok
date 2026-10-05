"""Export the PyTorch checkpoint to a minimal Kaggle .tar.gz submission."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_rl.policy import CandidateActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "policy_10k_selfplay.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission")
    parser.add_argument("--archive", type=Path, default=ROOT / "submission" / "submission.tar.gz")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CandidateActorCritic()
    model.load_state_dict(checkpoint["state_dict"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "submission_template" / "main.py", args.output_dir / "main.py")
    source_deck = args.competition_root / "sample_submission" / "sample_submission" / "sample_submission" / "deck.csv"
    shutil.copy2(source_deck, args.output_dir / "deck.csv")
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    np.savez_compressed(args.output_dir / "model.npz", **arrays)
    with tarfile.open(args.archive, "w:gz") as archive:
        for filename in ("main.py", "deck.csv", "model.npz"):
            archive.add(args.output_dir / filename, arcname=filename)
    print(f"Created {args.archive} ({args.archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
