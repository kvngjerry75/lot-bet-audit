#!/usr/bin/env python3
"""
wheel_generator.py: covering-design "wheel" over the top-V numbers of the Stage 4 play-ensemble.

Goal: a set of tickets such that if ANY 2 of the V core numbers are among the drawn numbers, at least one ticket
contains both of them. (A covering design C(V, k, 2): every pair of core numbers appears together in some ticket.)

Usage
  python wheel_generator.py --json outputs/stage4_results_daywa_blind2_XXXX.json        # reads p_play, takes top 8
  python wheel_generator.py --json outputs/                                              # newest stage4_results_*.json
  python wheel_generator.py --numbers 7 29 18 8 28 3 12 33 --n-balls 39                  # manual
Options
  --size 0        core pool size V (0 = automatic: 12 for 90 balls, 10 for 39 balls, else 8)
  --ticket 2      numbers per ticket k (2 = pair tickets, up to 5)
  --price 1.0     price of ONE ticket (flat)
  --pair-price    price per PAIR instead: a k-number ticket costs C(k,2) x price (typical for "perm" style betting)
  --payout 240    optional payout multiple per winning pair, for break-even maths

Be clear about what this does and does not do:
  * It guarantees COVERAGE: if >=2 core numbers are drawn, some ticket holds that pair.
  * It does NOT guarantee that 2 core numbers will be drawn. That probability is printed below and is the real limit.
  * Wheeling changes how the same bets are packaged. It does not change the expected value of each pair bet.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Covering design
# --------------------------------------------------------------------------- #
def default_core_size(n: int) -> int:
    """Core pool size by game: 12 for a 90-ball pool, 10 for a 39-ball pool, 8 otherwise."""
    return {90: 12, 39: 10}.get(int(n), 8)


def schonheim(v: int, k: int, t: int = 2) -> int:
    """Lower bound on the number of k-subsets needed to cover all pairs of v points (Schonheim bound)."""
    return math.ceil(v / k * math.ceil((v - 1) / (k - 1)))


def greedy_cover(v: int, k: int, rng: random.Random):
    pairs = {frozenset(p) for p in itertools.combinations(range(v), 2)}
    tickets = []
    allk = list(itertools.combinations(range(v), k))
    while pairs:
        rng.shuffle(allk)
        best, best_gain = None, -1
        for c in allk:
            g = sum(1 for p in itertools.combinations(c, 2) if frozenset(p) in pairs)
            if g > best_gain:
                best, best_gain = c, g
                if g == math.comb(k, 2):
                    break
        tickets.append(best)
        for p in itertools.combinations(best, 2):
            pairs.discard(frozenset(p))
    return tickets


def prune(tickets, v):
    need = {frozenset(p) for p in itertools.combinations(range(v), 2)}
    keep = list(tickets)
    for t in sorted(tickets, key=lambda c: 0):
        rest = [c for c in keep if c != t]
        cov = {frozenset(p) for c in rest for p in itertools.combinations(c, 2)}
        if cov >= need:
            keep = rest
    return keep


def min_cover(v: int, k: int, restarts: int = 300, seed: int = 1):
    if k == 2:
        return [tuple(p) for p in itertools.combinations(range(v), 2)], True
    if k >= v:
        return [tuple(range(v))], True
    rng = random.Random(seed)
    lb = schonheim(v, k)
    best = None
    for _ in range(restarts):
        c = prune(greedy_cover(v, k, rng), v)
        if best is None or len(c) < len(best):
            best = c
            if len(best) == lb:
                break
    return best, len(best) == lb


def verify(tickets, v) -> bool:
    cov = {frozenset(p) for c in tickets for p in itertools.combinations(c, 2)}
    return cov >= {frozenset(p) for p in itertools.combinations(range(v), 2)}


# --------------------------------------------------------------------------- #
# Probabilities (hypergeometric: 5 numbers drawn from n)
# --------------------------------------------------------------------------- #
def p_at_least(n: int, drawn: int, core: int, m: int) -> float:
    return sum(math.comb(core, j) * math.comb(n - core, drawn - j) for j in range(m, min(core, drawn) + 1)) / math.comb(n, drawn)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def load_json(path: str, size: int):
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("stage4_results_*.json"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise SystemExit(f"no stage4_results_*.json in {p}")
        p = files[-1]
    data = json.loads(p.read_text())
    nxt = data["next"]
    prob = nxt.get("p_play") or nxt.get("p")
    if prob is None:
        raise SystemExit("JSON has no p_play / p array (re-run the updated stage4_astronomical.py)")
    n = int(data.get("N", len(prob)))
    size = size or default_core_size(n)
    order = sorted(range(len(prob)), key=lambda i: -prob[i])[:size]
    return [i + 1 for i in order], n, str(p), nxt.get("date")


def print_wheel(core, n, src, date, a):
    v, k = len(core), a.ticket
    if not 2 <= k <= 5 or k > v:
        raise SystemExit("--ticket must be between 2 and min(5, core size)")

    tickets_idx, optimal = min_cover(v, k)
    assert verify(tickets_idx, v), "internal error: cover is incomplete"
    tickets = [sorted(core[i] for i in t) for t in tickets_idx]
    tickets.sort()

    unit = a.price * (math.comb(k, 2) if a.pair_price else 1)
    cost = len(tickets) * unit
    pairs_total = math.comb(v, 2)
    lb = schonheim(v, k) if k > 2 else pairs_total

    print("=" * 70)
    print(f"WHEEL | source: {src}" + (f" | draw date {date}" if date else ""))
    print("=" * 70)
    print(f"Core ({v}): {core}")
    print(f"Ticket size {k}: {len(tickets)} tickets" + (f" (provably minimal: meets the lower bound {lb})" if optimal
          else f" (best found; theoretical lower bound {lb})"))
    print()
    for i, t in enumerate(tickets, 1):
        print(f"  {i:>3}. " + "  ".join(f"{x:>2}" for x in t))
    print()
    print(f"Cost: {len(tickets)} tickets x {unit:g} = {cost:g}" +
          ("  (priced per pair inside each ticket)" if a.pair_price else "  (flat price per ticket)"))
    print(f"GUARANTEE (coverage only): if any 2 of the {v} core numbers are drawn, at least one ticket contains both.")
    print(f"  verified by exhaustive check: all {pairs_total} core pairs are covered.")
    if k == 2:
        print(f"  {pairs_total} pair tickets cannot be reduced: each ticket covers exactly one pair.")
    else:
        flat = pairs_total * (a.price if not a.pair_price else a.price)
        print(f"  Compare with plain pair tickets: {pairs_total} tickets. This wheel uses {len(tickets)} bigger tickets;")
        if a.pair_price:
            print(f"  priced per pair it costs {cost:g} vs {flat:g} for the plain pairs, because bigger tickets repeat pairs.")
        print("  A ticket with several drawn core pairs pays once per matching pair if your game pays per pair; check the rules.")
    print()

    if n:
        pd = a.draw
        p2 = p_at_least(n, pd, v, 2)
        p_pair = pd * (pd - 1) / (n * (n - 1))
        print(f"ODDS for a {pd}-of-{n} game if the machine is fair")
        print(f"  P(a specific pair is drawn)                  = {100 * p_pair:.3f}%  (1 in {1 / p_pair:.0f})")
        print(f"  P(at least 2 of your {v} core numbers drawn)  = {100 * p2:.2f}%  <- the wheel can only win in these draws")
        print(f"  P(at least 3 of {v} core drawn)               = {100 * p_at_least(n, pd, v, 3):.2f}%")
        print(f"  P(none or only 1 core number drawn)         = {100 * (1 - p2):.2f}%  <- the whole wheel loses")
        exp_pairs = pairs_total * p_pair
        print(f"  expected winning core pairs per draw          = {exp_pairs:.3f} (= {pairs_total} x {100 * p_pair:.3f}%)")
        if a.payout > 0:
            ret = exp_pairs * a.payout * a.price
            print(f"  at a {a.payout:g}x payout per winning pair: expected return {ret:.2f} on a cost of {cost:g} "
                  f"({100 * ret / cost:.0f}% of stake) if the machine is fair")
            print(f"  break-even payout per pair = {1 / p_pair:.0f}x for single-pair tickets")
        else:
            print(f"  break-even: a winning pair must pay more than {1 / p_pair:.0f}x its stake (pass --payout to compare)")
        print("  Choosing the numbers cleverly changes these odds only if the machine is biased towards them.")
    else:
        print("(pass --n-balls <pool size> to print the odds; the JSON input supplies it automatically)")
    print("-" * 70)
    print("Note: ticket-size 5 'core-only' wheels are what covering designs mean. Tickets with exactly 2 core numbers")
    print("plus 3 filler numbers would not cover the pairs, so they are not generated.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="stage4_results_*.json (or a folder containing them)")
    ap.add_argument("--numbers", type=int, nargs="+", help="core numbers, in priority order")
    ap.add_argument("--n-balls", type=int, default=0, help="pool size of the game (needed for odds with --numbers)")
    ap.add_argument("--size", type=int, default=0,
                    help="core size; 0 = automatic from the pool size (90 -> 12, 39 -> 10, else 8)")
    ap.add_argument("--ticket", type=int, default=2, help="numbers per ticket (2..5)")
    ap.add_argument("--price", type=float, default=1.0)
    ap.add_argument("--pair-price", action="store_true", help="price = C(k,2) x price per ticket")
    ap.add_argument("--payout", type=float, default=0.0, help="payout multiple per winning pair (for break-even)")
    ap.add_argument("--draw", type=int, default=5, help="numbers drawn per draw")
    a = ap.parse_args()

    if a.numbers:
        core, n, src, date = a.numbers[:(a.size or default_core_size(a.n_balls))], a.n_balls, "manual", None
    elif a.json:
        core, n, src, date = load_json(a.json, a.size)
    else:
        raise SystemExit("give --json <file|folder> or --numbers ...")
    if len(set(core)) != len(core):
        raise SystemExit("core numbers must be distinct")
    print_wheel(core, n, src, date, a)


if __name__ == "__main__":
    main()
