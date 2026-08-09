#!/usr/bin/env python3
"""
test_sd_benchmark.py

Profile each phase of the single-dummy pipeline (deal generation, DD solving,
result assembly) in isolation and together, with per-core CPU monitoring.

The output shows exactly where time goes and how much batching across PBNs
would improve throughput -- guiding the optimisation of
estimate_sd_trick_distributions_for_df().

Usage:
    python test_sd_benchmark.py [--num-pbns N] [--sd-productions P] [--seed S]
"""

import argparse
import random
import sys
import pathlib
import threading
import time

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import polars as pl
from endplay.types import Deal
from endplay.dealer import generate_deals

from mlBridge.dds_ddss import DDSS_AVAILABLE
if DDSS_AVAILABLE:
    from mlBridge import dds_ddss
from mlBridge.mlBridgeAugmentLib import (
    estimate_sd_trick_distributions,
    solve_dd_for_deals,
    deal_generation_constraints,
)

RANKS = "AKQJT98765432"
SAMPLE_INTERVAL = 0.01  # 10ms -- fast enough to capture short phases


# ---------------------------------------------------------------------------
# CPU monitor (same as test_ddss_vs_endplay.py)
# ---------------------------------------------------------------------------

class CpuMonitor:
    def __init__(self):
        self._samples: list[list[float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _worker(self):
        psutil.cpu_percent(percpu=True)
        while not self._stop.wait(SAMPLE_INTERVAL):
            self._samples.append(psutil.cpu_percent(percpu=True))

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict | None:
        if not self._samples:
            return None
        nc = len(self._samples[0])
        ns = len(self._samples)
        per_core = [sum(s[c] for s in self._samples) / ns for c in range(nc)]
        avg = sum(per_core) / nc
        active = sum(1 for a in per_core if a > 50)
        return {"avg": avg, "active": active, "n_cores": nc, "n_samples": ns,
                "per_core": sorted(per_core, reverse=True)}


class _NullMonitor:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass
    def summary(self):
        return None


def cpu_monitor():
    return CpuMonitor() if PSUTIL_AVAILABLE else _NullMonitor()


def fmt_cpu(s: dict | None) -> str:
    if s is None:
        return "(no CPU data -- phase too short to sample)"
    cores = s["per_core"]
    line = f"avg {s['avg']:4.0f}%  active {s['active']:>2}/{s['n_cores']}  ({s['n_samples']} samples)"
    row1 = " ".join(f"{v:3.0f}" for v in cores[:16])
    if len(cores) > 16:
        row2 = " ".join(f"{v:3.0f}" for v in cores[16:])
        line += f"\n{'':>20}cores: [{row1}]\n{'':>20}       [{row2}]"
    else:
        line += f"  cores: [{row1}]"
    return line


# ---------------------------------------------------------------------------
# Deal generation helpers
# ---------------------------------------------------------------------------

def generate_random_pbns(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    deck = list(range(52))
    pbns = []
    for _ in range(n):
        rng.shuffle(deck)
        hands = [[] for _ in range(4)]
        for i, card in enumerate(deck):
            hands[i % 4].append(card)
        parts = []
        for hand in hands:
            suit_cards = {s: [] for s in range(4)}
            for c in sorted(hand):
                suit_cards[c // 13].append(RANKS[c % 13])
            parts.append(".".join("".join(suit_cards[s]) for s in range(4)))
        pbns.append("N:" + " ".join(parts))
    return pbns


def make_predeal_strings(pbn: str):
    """Return (ns_predeal, ew_predeal) with two seats masked."""
    s = pbn[2:].split()
    ns = list(s)
    ns[1] = '...'
    ns[3] = '...'
    ew = list(s)
    ew[0] = '...'
    ew[2] = '...'
    return 'N:' + ' '.join(ns), 'N:' + ' '.join(ew)


def gen_deals_for_predeal(predeal_string: str, produce: int, seed: int = 42):
    predeal = Deal(predeal_string)
    deals_t = generate_deals(
        deal_generation_constraints,
        predeal=predeal,
        swapping=0,
        show_progress=False,
        produce=produce,
        seed=seed,
        max_attempts=1000000,
        env={},
        strict=True,
    )
    return list(deals_t)


def postprocess_sd_results(deals, dd_tables):
    """Mimics the DataFrame + value_counts work in estimate_sd_trick_distributions."""
    schema = {'SD_Deal': pl.String}
    schema.update({f'SD_Tricks_{d}_{s}': pl.UInt8 for s in 'SHDCN' for d in 'NESW'})

    data_rows = []
    for deal, t in zip(deals, dd_tables):
        row = [deal.to_pbn()]
        row.extend([s for d in t.to_list() for s in d])
        data_rows.append(row)

    df = pl.DataFrame(data_rows, schema=schema, orient='row')

    ns_ew_rows = {}
    for d in 'NESW':
        for s in 'SHDCN':
            col = f'SD_Tricks_{d}_{s}'
            vc = dict(df[col].value_counts(normalize=True).rows())
            ns_ew_rows[(d, s)] = [vc.get(i, 0.0) for i in range(14)]
    return ns_ew_rows


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase2_baseline(pbns: list[str], produce: int):
    """Run the current per-PBN serial pipeline."""
    print("=" * 70)
    print("Phase 2: BASELINE (current per-PBN serial pipeline)")
    print("=" * 70)
    total_dd_solves = len(pbns) * produce * 2

    with cpu_monitor() as mon:
        t0 = time.perf_counter()
        for pbn in pbns:
            estimate_sd_trick_distributions(pbn, produce)
        elapsed = time.perf_counter() - t0

    cpu = mon.summary()
    per_pbn = elapsed / len(pbns)
    print(f"  {len(pbns)} PBNs x {produce} productions x 2 sides = {total_dd_solves} DD solves")
    print(f"  Total: {elapsed:.2f}s  Per-PBN: {per_pbn:.3f}s  DD-solves/s: {total_dd_solves/elapsed:.0f}")
    print(f"  CPU: {fmt_cpu(cpu)}")
    print()
    return {"label": "Baseline (serial)", "elapsed": elapsed, "dd_solves": total_dd_solves, "cpu": cpu}


def phase3_breakdown(pbns: list[str], produce: int, seed: int):
    """Time each component independently."""
    print("=" * 70)
    print("Phase 3: COMPONENT BREAKDOWN")
    print("=" * 70)

    # --- Deal generation ---
    all_generated: list[tuple[str, str, list]] = []  # (side, pbn, deals)
    with cpu_monitor() as mon_gen:
        t0 = time.perf_counter()
        for pbn in pbns:
            ns_pre, ew_pre = make_predeal_strings(pbn)
            ns_deals = gen_deals_for_predeal(ns_pre, produce, seed)
            ew_deals = gen_deals_for_predeal(ew_pre, produce, seed)
            all_generated.append(("NS", pbn, ns_deals))
            all_generated.append(("EW", pbn, ew_deals))
        t_gen = time.perf_counter() - t0
    cpu_gen = mon_gen.summary()

    # --- DD solving (serial, one small batch per predeal) ---
    all_tables = []
    with cpu_monitor() as mon_solve:
        t0 = time.perf_counter()
        for side, pbn, deals in all_generated:
            tables = solve_dd_for_deals(deals)
            all_tables.append((side, pbn, deals, tables))
        t_solve = time.perf_counter() - t0
    cpu_solve = mon_solve.summary()

    # --- Post-processing ---
    with cpu_monitor() as mon_post:
        t0 = time.perf_counter()
        for side, pbn, deals, tables in all_tables:
            postprocess_sd_results(deals, tables)
        t_post = time.perf_counter() - t0
    cpu_post = mon_post.summary()

    total = t_gen + t_solve + t_post
    print(f"  Deal generation:  {t_gen:7.2f}s ({t_gen/total*100:4.0f}%)  CPU: {fmt_cpu(cpu_gen)}")
    print(f"  DD solving:       {t_solve:7.2f}s ({t_solve/total*100:4.0f}%)  CPU: {fmt_cpu(cpu_solve)}")
    print(f"  Post-processing:  {t_post:7.2f}s ({t_post/total*100:4.0f}%)  CPU: {fmt_cpu(cpu_post)}")
    print(f"  Sum:              {total:7.2f}s")
    print()

    return {
        "gen": {"elapsed": t_gen, "cpu": cpu_gen},
        "solve": {"elapsed": t_solve, "cpu": cpu_solve},
        "post": {"elapsed": t_post, "cpu": cpu_post},
        "all_generated": all_generated,
    }


def phase4_batched(all_generated: list, produce: int):
    """Batch all generated deals into one large solve call."""
    print("=" * 70)
    print("Phase 4: BATCHED DD SOLVE (all generated deals in one call)")
    print("=" * 70)

    combined_deals = []
    batch_map = []  # (start_idx, count, side, pbn) for regrouping
    for side, pbn, deals in all_generated:
        start = len(combined_deals)
        combined_deals.extend(deals)
        batch_map.append((start, len(deals), side, pbn))

    total_deals = len(combined_deals)

    with cpu_monitor() as mon:
        t0 = time.perf_counter()
        all_tables = solve_dd_for_deals(combined_deals)
        t_solve = time.perf_counter() - t0
    cpu = mon.summary()

    # Verify we can regroup and post-process
    with cpu_monitor() as mon_post:
        t0_post = time.perf_counter()
        for start, count, side, pbn in batch_map:
            sub_deals = combined_deals[start:start + count]
            sub_tables = all_tables[start:start + count]
            postprocess_sd_results(sub_deals, sub_tables)
        t_post = time.perf_counter() - t0_post
    cpu_post = mon_post.summary()

    rate = total_deals / t_solve if t_solve > 0 else 0
    print(f"  {total_deals} deals solved in one batch: {t_solve:.2f}s ({rate:.0f} deals/s)")
    print(f"  CPU: {fmt_cpu(cpu)}")
    print(f"  Post-processing after regroup: {t_post:.2f}s  CPU: {fmt_cpu(cpu_post)}")
    print()

    return {
        "label": "Batched DD solve",
        "solve_elapsed": t_solve,
        "post_elapsed": t_post,
        "total_deals": total_deals,
        "cpu": cpu,
    }


def phase5_summary(baseline, breakdown, batched, num_pbns, produce, projection_pbns=747_583):
    """Print comparison table and projection."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    b_elapsed = baseline["elapsed"]
    b_rate = baseline["dd_solves"] / b_elapsed if b_elapsed > 0 else 0

    # Batched estimate: generation + batched solve + batched post-processing
    batched_total = breakdown["gen"]["elapsed"] + batched["solve_elapsed"] + batched["post_elapsed"]
    batched_dd_solves = batched["total_deals"]
    batched_rate = batched_dd_solves / batched["solve_elapsed"] if batched["solve_elapsed"] > 0 else 0

    serial_solve = breakdown["solve"]["elapsed"]
    speedup_solve = serial_solve / batched["solve_elapsed"] if batched["solve_elapsed"] > 0 else 0
    speedup_total = b_elapsed / batched_total if batched_total > 0 else 0

    print(f"  {'Metric':<40} {'Serial':>12} {'Batched':>12} {'Speedup':>10}")
    print("  " + "-" * 76)
    print(f"  {'DD solving time':<40} {serial_solve:>11.2f}s {batched['solve_elapsed']:>11.2f}s {speedup_solve:>9.1f}x")
    print(f"  {'Total pipeline time':<40} {b_elapsed:>11.2f}s {batched_total:>11.2f}s {speedup_total:>9.1f}x")
    print(f"  {'DD solves/s':<40} {b_rate:>12.0f} {batched_rate:>12.0f}")
    print(f"  {'Per-PBN time':<40} {b_elapsed/num_pbns:>11.3f}s {batched_total/num_pbns:>11.3f}s")
    print()

    cpu_serial = baseline["cpu"]["avg"] if baseline["cpu"] else 0
    cpu_batched = batched["cpu"]["avg"] if batched["cpu"] else 0
    print(f"  {'CPU utilisation (DD solve phase)':<40} {cpu_serial:>11.0f}% {cpu_batched:>11.0f}%")

    if projection_pbns > 0:
        proj_serial_h = (b_elapsed / num_pbns) * projection_pbns / 3600
        proj_batched_h = (batched_total / num_pbns) * projection_pbns / 3600
        print()
        print(f"  Projected for {projection_pbns:,} PBNs:")
        print(f"    Current (serial):  {proj_serial_h:6.1f} hours")
        print(f"    Batched pipeline:  {proj_batched_h:6.1f} hours  ({speedup_total:.1f}x faster)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark single-dummy pipeline phases")
    parser.add_argument("--num-pbns", type=int, default=50,
                        help="Number of source PBNs to benchmark (default: 50)")
    parser.add_argument("--sd-productions", type=int, default=10,
                        help="Deals generated per side per PBN (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not DDSS_AVAILABLE:
        print("ERROR: ddss DLL not available.")
        sys.exit(1)

    num_pbns = args.num_pbns
    produce = args.sd_productions
    seed = args.seed

    print(f"Single-Dummy Pipeline Benchmark")
    print(f"  PBNs={num_pbns}  productions={produce}  seed={seed}")
    print(f"  Total DD solves per run: {num_pbns * produce * 2}")
    print()

    # --- Phase 1: generate source PBNs ---
    print("Phase 1: Generating random source PBNs...")
    pbns = generate_random_pbns(num_pbns, seed)
    print(f"  Generated {len(pbns)} PBNs.\n")

    # --- Phase 2: baseline ---
    baseline = phase2_baseline(pbns, produce)

    # --- Phase 3: component breakdown ---
    breakdown = phase3_breakdown(pbns, produce, seed)

    # --- Phase 4: batched ---
    batched = phase4_batched(breakdown["all_generated"], produce)

    # --- Phase 5: summary ---
    phase5_summary(baseline, breakdown, batched, num_pbns, produce)


if __name__ == "__main__":
    main()
