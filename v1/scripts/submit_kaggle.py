"""Build, validate, and optionally upload the PTCG Playground submission."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "the-pokemon-company-ptcg-ai-battle-challenge-playground"


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "access_token").is_file() or (kaggle_dir / "kaggle.json").is_file():
        return True
    # Kaggle CLI 2.x can retain an OAuth session in its credential backend
    # without exposing a token file. Ask the CLI for non-secret config only.
    probe = subprocess.run(
        [sys.executable, "-m", "kaggle", "config", "view"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and "auth_method: OAUTH" in probe.stdout


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and submit the verified PTCG PPO agent.")
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "policy_10k_selfplay.pt")
    parser.add_argument("--message", default="PPO v1: 10k self-play")
    parser.add_argument("--submit", action="store_true", help="Upload after successful local validation.")
    args = parser.parse_args()
    archive = ROOT / "submission" / "submission.tar.gz"
    run([sys.executable, str(ROOT / "scripts" / "build_submission.py"), "--competition-root", str(args.competition_root), "--checkpoint", str(args.checkpoint)])
    run([sys.executable, str(ROOT / "scripts" / "verify_submission.py"), "--competition-root", str(args.competition_root), "--submission-dir", str(ROOT / "submission"), "--archive", str(archive)])
    if not args.submit:
        print(f"Pipeline complete. Valid archive: {archive}")
        print("Upload intentionally skipped. Re-run with --submit after Kaggle authentication.")
        return
    if not has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials are missing. Run `python -m kaggle auth login` yourself, "
            "or configure KAGGLE_API_TOKEN, then re-run this command with --submit."
        )
    run([sys.executable, "-m", "kaggle", "competitions", "submit", COMPETITION, "-f", str(archive), "-m", args.message])


if __name__ == "__main__":
    main()
