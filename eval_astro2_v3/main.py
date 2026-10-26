"""NLA 5/90 research pipeline - Stage 1: data pipeline + randomness diagnostics.

Usage:
    python main.py                      # interactive game menu
    python main.py --game 1             # skip the menu
    python main.py --data-dir ./data --out-dir ./outputs --n-sims 2000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import DEFAULT_SEED, GAMES
from data_pipeline import load_game
from diagnostics import run_all, save_report, summarise
from log_setup import setup_logging


def prompt_game() -> str:
    print("\nSelect a game:")
    for k, (name, fname) in GAMES.items():
        print(f"  {k}: {name:<9} ({fname})")
    while True:
        choice = input("Enter number: ").strip()
        if choice in GAMES:
            return choice
        print(f"  '{choice}' is not valid. Choose one of {', '.join(GAMES)}.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=list(GAMES), help="game number (skips the menu)")
    ap.add_argument("--data-dir", default=".", help="folder holding the CSVs (default: current folder)")
    ap.add_argument("--out-dir", default="outputs", help="folder for logs and reports")
    ap.add_argument("--n-sims", type=int, default=2000, help="Monte Carlo simulations for calibrated tests")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    key = args.game or prompt_game()
    name = GAMES[key][0]

    out_dir = Path(args.out_dir)
    setup_logging(out_dir / "logs", name)
    log = logging.getLogger("nla.main")
    log.info("Game selected: %s | data_dir=%s | seed=%d | n_sims=%d", name, Path(args.data_dir).resolve(), args.seed, args.n_sims)

    try:
        game = load_game(key, args.data_dir, out_dir)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1

    results = run_all(game.X, game.df, seed=args.seed, n_sims=args.n_sims)
    table, verdict = summarise(results)
    tp, _ = save_report(results, table, verdict, out_dir, name, len(game.df))

    log.info("\n%s\n\n%s", table.to_string(index=False, float_format=lambda x: f"{x:.4f}"), verdict)
    log.info("Stage 1 complete. Full report: %s", tp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
