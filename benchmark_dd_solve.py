#!/usr/bin/env python3
"""
benchmark_dd_solve.py

Focused DD benchmark that runs in a subprocess so each invocation loads
a fresh copy of the ddss DLL.  Outputs JSON to stdout for the harness.

Usage (called by benchmark_improvements.py, not directly):
    python benchmark_dd_solve.py [--num-deals N] [--iterations N] [--seed S]
"""

import argparse
import json
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
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

from mlBridge.dds_ddss import DDSS_AVAILABLE
if DDSS_AVAILABLE:
    from mlBridge import dds_ddss

RANKS = "AKQJT98765432"
SAMPLE_INTERVAL = 0.05


def generate_pbn_strings(n: int, seed: int) -> list[str]:
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


class _CpuSampler:
    def __init__(self):
        self._samples: list[list[float]] = []
        self._stop = threading.Event()

    def _worker(self):
        psutil.cpu_percent(percpu=True)
        while not self._stop.wait(SAMPLE_INTERVAL):
            self._samples.append(psutil.cpu_percent(percpu=True))

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        self._thread = t
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def result(self) -> dict:
        if not self._samples:
            return {"avg": -1, "active": -1, "n_cores": -1, "n_samples": 0, "per_core": []}
        nc = len(self._samples[0])
        ns = len(self._samples)
        per_core = [sum(s[c] for s in self._samples) / ns for c in range(nc)]
        avg = sum(per_core) / nc
        active = sum(1 for a in per_core if a > 50)
        return {
            "avg": round(avg, 1),
            "active": active,
            "n_cores": nc,
            "n_samples": ns,
            "per_core": [round(v, 1) for v in sorted(per_core, reverse=True)],
        }


class _NullSampler:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass
    def result(self):
        return {"avg": -1, "active": -1, "n_cores": -1, "n_samples": 0, "per_core": []}


def run_benchmark(pbns: list[str], iterations: int) -> dict:
    times = []
    cpu_results = []
    for _ in range(iterations):
        sampler = _CpuSampler() if _PSUTIL else _NullSampler()
        with sampler:
            t0 = time.perf_counter()
            dds_ddss.calc_all_tables_pbnx(pbns)
            elapsed = time.perf_counter() - t0
        times.append(elapsed)
        cpu_results.append(sampler.result())

    best = min(times)
    median = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    rate_best = len(pbns) / best if best > 0 else 0
    rate_median = len(pbns) / median if median > 0 else 0

    avg_cpu = sum(c["avg"] for c in cpu_results if c["avg"] >= 0) / max(1, sum(1 for c in cpu_results if c["avg"] >= 0))
    avg_active = sum(c["active"] for c in cpu_results if c["active"] >= 0) / max(1, sum(1 for c in cpu_results if c["active"] >= 0))
    n_cores = cpu_results[0]["n_cores"] if cpu_results else -1
    best_cpu = cpu_results[times.index(best)] if cpu_results else {}

    return {
        "num_deals": len(pbns),
        "iterations": iterations,
        "times": [round(t, 4) for t in times],
        "best_s": round(best, 4),
        "median_s": round(median, 4),
        "mean_s": round(mean, 4),
        "rate_best": round(rate_best, 1),
        "rate_median": round(rate_median, 1),
        "cpu_avg_pct": round(avg_cpu, 1),
        "cpu_active_cores": round(avg_active, 1),
        "cpu_total_cores": n_cores,
        "cpu_per_core_best": best_cpu.get("per_core", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-deals", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not DDSS_AVAILABLE:
        json.dump({"error": "ddss DLL not available"}, sys.stdout)
        sys.exit(1)

    pbns = generate_pbn_strings(args.num_deals, args.seed)

    # warmup
    dds_ddss.calc_all_tables_pbnx(pbns[:200])

    result = run_benchmark(pbns, args.iterations)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
