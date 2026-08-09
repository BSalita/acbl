#!/usr/bin/env python3
"""
test_ddss_vs_endplay.py

Validate that the ddss ctypes wrapper produces identical DD tables to endplay,
and benchmark performance across a wide range of batch sizes.

Endplay is solved ONCE for all deals (the reference truth) and cached.
Only ddss is called at each batch size to exercise its chunking logic.
This avoids ~80% of redundant DLL calls vs solving both engines per batch,
keeping CPU cores saturated and completing much faster.

Usage:
    python test_ddss_vs_endplay.py [--num-deals N]
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

from endplay.dds import calc_all_tables
from endplay.types import Deal

from mlBridge.dds_ddss import DDSS_AVAILABLE
if DDSS_AVAILABLE:
    from mlBridge import dds_ddss


BATCH_SIZES = [-1, 0, 1, 40, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999, 1000, 1001, 1040, 2000]

RANKS = "AKQJT98765432"
SUITS = "SHDC"

SAMPLE_INTERVAL = 0.05  # 50 ms between per-core CPU samples


class CpuMonitor:
    """Sample per-core CPU% from a background thread while DLL runs.

    ctypes calls release the GIL, so Python threads can sample freely.
    Use as a context manager around the code to monitor.
    """

    def __init__(self):
        self._samples: list[list[float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _worker(self):
        psutil.cpu_percent(percpu=True)  # prime the internal delta
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
        n_cores = len(self._samples[0])
        n_samples = len(self._samples)
        per_core_avg = [
            sum(s[c] for s in self._samples) / n_samples
            for c in range(n_cores)
        ]
        overall_avg = sum(per_core_avg) / n_cores
        active = sum(1 for a in per_core_avg if a > 50)
        return {
            "per_core": sorted(per_core_avg, reverse=True),
            "avg": overall_avg,
            "active": active,
            "n_cores": n_cores,
            "n_samples": n_samples,
        }


class _NullMonitor:
    """No-op stand-in when psutil is not installed."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def summary(self):
        return None


def cpu_monitor():
    return CpuMonitor() if PSUTIL_AVAILABLE else _NullMonitor()


def fmt_cpu(summary: dict | None) -> str:
    """One-line per-core CPU summary, cores sorted high-to-low."""
    if summary is None:
        return "(no CPU data -- install psutil)"
    cores = summary["per_core"]
    avg = summary["avg"]
    active = summary["active"]
    n = summary["n_cores"]
    samp = summary["n_samples"]
    core_str = " ".join(f"{v:3.0f}" for v in cores)
    return (
        f"avg {avg:4.0f}%  active {active:>2}/{n}  "
        f"({samp} samples)  cores: [{core_str}]"
    )


def generate_random_deals(n: int) -> list[Deal]:
    """Generate n random deals by shuffling a 52-card deck."""
    deals = []
    deck = list(range(52))
    for _ in range(n):
        random.shuffle(deck)
        hands = [[] for _ in range(4)]
        for i, card in enumerate(deck):
            hands[i % 4].append(card)
        parts = []
        for hand in hands:
            suit_cards = {s: [] for s in range(4)}
            for c in sorted(hand):
                suit_cards[c // 13].append(RANKS[c % 13])
            parts.append(".".join("".join(suit_cards[s]) for s in range(4)))
        pbn = "N:" + " ".join(parts)
        deals.append(Deal(pbn))
    return deals


def solve_endplay(deals: list[Deal]) -> list[list[list[int]]]:
    """Solve via endplay in batches of 40 and return flattened tables."""
    results = []
    for b in range(0, len(deals), 40):
        batch = deals[b:b + 40]
        tables = calc_all_tables(batch)
        results.extend(tables)
    return [rt.to_list() for rt in results]


def solve_ddss(pbn_strings: list[str]) -> list[list[list[int]]]:
    """Solve via ddss wrapper and return flattened tables."""
    results = dds_ddss.calc_all_tables_pbnx(pbn_strings)
    return [rt.to_list() for rt in results]


def compare_tables(endplay_tables, ddss_tables, deal_index: int) -> list[str]:
    """Return a list of mismatch descriptions, empty if identical."""
    strains = "SHDCN"
    hands = "NESW"
    mismatches = []
    for si in range(5):
        for hi in range(4):
            ev = endplay_tables[si][hi]
            dv = ddss_tables[si][hi]
            if ev != dv:
                mismatches.append(
                    f"  deal {deal_index}: {hands[hi]}/{strains[si]} "
                    f"endplay={ev} ddss={dv}"
                )
    return mismatches


def run_test(pbn_strings: list[str], ep_ref: list, batch_size: int):
    """Test a specific batch size: ddss correctness + timing + CPU.

    Uses cached endplay reference results instead of re-solving.
    """
    n = batch_size
    header = f"batch_size={n:>5}"

    if n < 0:
        print(f"{header}  SKIP (invalid negative size)")
        return None
    if n == 0:
        print(f"{header}  SKIP (zero deals)")
        return None
    if n > len(pbn_strings):
        print(f"{header}  SKIP (only {len(pbn_strings)} deals available)")
        return None

    with cpu_monitor() as mon:
        t0 = time.perf_counter()
        ddss_tables = solve_ddss(pbn_strings[:n])
        t_ddss = time.perf_counter() - t0

    cpu_sum = mon.summary()

    all_mismatches = []
    for i in range(n):
        all_mismatches.extend(compare_tables(ep_ref[i], ddss_tables[i], i))

    status = "PASS" if not all_mismatches else "FAIL"
    ddss_rate = n / t_ddss if t_ddss > 0 else float("inf")

    print(
        f"{header}  {status}  "
        f"ddss={t_ddss:7.3f}s ({ddss_rate:7.0f}/s)  "
        f"CPU: {fmt_cpu(cpu_sum)}"
    )

    if all_mismatches:
        for m in all_mismatches[:20]:
            print(m)
        if len(all_mismatches) > 20:
            print(f"  ... and {len(all_mismatches) - 20} more mismatches")

    return {
        "batch_size": n,
        "status": status,
        "ddss_time": t_ddss,
        "mismatches": len(all_mismatches),
        "cpu": cpu_sum,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate ddss vs endplay DD results")
    parser.add_argument("--num-deals", type=int, default=2000,
                        help="Number of random deals to generate (default: 2000)")
    args = parser.parse_args()

    if not DDSS_AVAILABLE:
        print("ERROR: ddss DLL is not available. Cannot run comparison tests.")
        print("       Ensure the DLL exists at the expected path.")
        sys.exit(1)

    num_deals = args.num_deals
    print(f"Generating {num_deals} random deals...")
    deals = generate_random_deals(num_deals)
    pbn_strings = [d.to_pbn() for d in deals]
    print(f"Generated {len(deals)} deals.\n")

    # --- Phase 1: endplay reference (solved once, reused for all batch sizes) ---
    print(f"Solving all {num_deals} deals with endplay (reference)...")
    with cpu_monitor() as ep_mon:
        t0 = time.perf_counter()
        ep_ref = solve_endplay(deals)
        t_ep = time.perf_counter() - t0
    ep_cpu = ep_mon.summary()
    ep_rate = num_deals / t_ep if t_ep > 0 else float("inf")
    print(f"  endplay: {t_ep:.3f}s ({ep_rate:.0f} deals/s)")
    print(f"  CPU: {fmt_cpu(ep_cpu)}\n")

    # --- Phase 2: ddss correctness + timing at each batch size ---
    print(f"{'Batch':>16}  {'Result':>6}  {'ddss':>26}")
    print("-" * 60)

    results = []
    for bs in BATCH_SIZES:
        effective = min(bs, len(deals)) if bs >= 0 else bs
        r = run_test(pbn_strings, ep_ref, effective if bs >= 0 else bs)
        if r is not None:
            results.append(r)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    skip_count = len(BATCH_SIZES) - len(results)
    print(f"  Tests run: {len(results)},  passed: {pass_count},  failed: {fail_count},  skipped: {skip_count}")

    if results:
        total_ddss = sum(r["ddss_time"] for r in results)
        total_deals_ddss = sum(r["batch_size"] for r in results)
        ddss_agg_rate = total_deals_ddss / total_ddss if total_ddss > 0 else float("inf")

        print(f"\n  Endplay reference:  {t_ep:.2f}s  ({ep_rate:.0f} deals/s) -- solved once for {num_deals} deals")
        print(f"  Total ddss solves:  {total_ddss:.2f}s  ({ddss_agg_rate:.0f} deals/s) over {total_deals_ddss} deals ({len(results)} batch sizes)")

        # Apples-to-apples: ddss at full num_deals vs endplay at full num_deals
        full_run = next((r for r in results if r["batch_size"] == num_deals), None)
        if full_run:
            speedup = t_ep / full_run["ddss_time"] if full_run["ddss_time"] > 0 else float("inf")
            print(f"\n  Full-batch speedup ({num_deals} deals): ddss {full_run['ddss_time']:.2f}s vs endplay {t_ep:.2f}s = {speedup:.2f}x")


if __name__ == "__main__":
    main()
