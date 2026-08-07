"""Replay evaluation *through the running system*, and attribute the gap.

The offline number in `state/offline.json` was produced in one process, on warm and complete data,
with no budget. This scores the same users on the same metric by asking the orchestrator, over
HTTP, one request at a time — which is the only way the losses this drill teaches can show up at
all. None of them are visible to a model-level evaluation.
"""

from __future__ import annotations

import json
import statistics
import time

import httpx
import numpy as np
from thilirec import evaluate_topk

from .state import read_chaos, reset_chaos, set_chaos, state_dir

ORCH = "http://127.0.0.1:8800"
K = 10


def _users(n: int, seed: int = 0) -> np.ndarray:
    """A fixed random sample. Fixed because every row of the attribution table must be scored on
    the same users, or the differences are sampling noise wearing a lab coat."""
    test_mask = np.load(state_dir() / "test_mask.npy")
    eligible = np.flatnonzero(test_mask.any(axis=1))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=min(n, eligible.size), replace=False))


def run(users: np.ndarray, k: int = K, n_candidates: int = 200) -> dict:
    """Send one request per user and score the answers."""
    test_mask = np.load(state_dir() / "test_mask.npy")
    recs, lat, notes = [], [], []
    with httpx.Client(timeout=30.0) as c:
        for u in users:
            t0 = time.perf_counter()
            r = c.get(f"{ORCH}/recommend",
                      params={"user_id": int(u), "k": k, "n_candidates": n_candidates}).json()
            lat.append((time.perf_counter() - t0) * 1000)
            recs.append(r["items"])
            notes.extend(r["notes"])

    metrics = evaluate_topk(recs, test_mask[users], k=k)
    lat.sort()
    metrics["p50_ms"] = round(statistics.median(lat), 2)
    metrics["p99_ms"] = round(lat[int(len(lat) * 0.99) - 1], 2)
    metrics["empty"] = sum(1 for r in recs if not r)
    return metrics


# Each row is (label, chaos switches). Applied one at a time from a clean baseline so the
# attribution is per-cause; the final row turns them all on together, which is NOT the sum —
# the losses overlap, and saying so is lesson 10.
SCENARIOS = [
    ("through the system, nothing wrong", {}),
    ("stale index (last batch run)", {"stale_index": True}),
    ("feature skew (ranker row drift)", {"feature_skew": True}),
    ("rules filter applied", {"rules_filter": True}),
    ("ranker timeouts 20%, fail open", {"ranker_timeout_rate": 0.2, "failure_policy": "open"}),
    ("ranker timeouts 20%, fail closed", {"ranker_timeout_rate": 0.2, "failure_policy": "closed"}),
    ("fetch capped at 300", {"max_fetch": 300}),
    ("all of them at once", {"stale_index": True, "feature_skew": True, "rules_filter": True,
                             "ranker_timeout_rate": 0.2, "max_fetch": 300}),
]


def offline_on(users: np.ndarray, k: int = K) -> dict:
    """The offline baseline, scored on exactly the users the harness replays."""
    test_mask = np.load(state_dir() / "test_mask.npy")
    recs = json.load(open(state_dir() / "offline_recs.json"))
    return evaluate_topk([recs[u] for u in users], test_mask[users], k=k)


def attribution(n_users: int = 800, k: int = K, n_candidates: int = 200) -> list[dict]:
    users = _users(n_users)
    offline = offline_on(users, k=k)
    base = offline[f"recall@{k}"]

    print(f"{len(users)} users, k={k}, {n_candidates} candidates\n")
    print(f"{'scenario':<38} {'recall@10':>10} {'ndcg@10':>9} {'p50':>7} {'p99':>8} {'vs offline':>11}")
    print("-" * 90)
    print(f"{'OFFLINE (in-process, warm)':<38} {base:>10.4f} "
          f"{offline[f'ndcg@{k}']:>9.4f} {'-':>7} {'-':>8} {'-':>11}")

    rows = []
    for label, switches in SCENARIOS:
        reset_chaos()
        if switches:
            set_chaos(**switches)
        m = run(users, k=k, n_candidates=n_candidates)
        delta = 100 * (m[f"recall@{k}"] - base) / base if base else float("nan")
        rows.append({"scenario": label, "switches": switches, **m, "vs_offline_pct": delta})
        print(f"{label:<38} {m[f'recall@{k}']:>10.4f} {m[f'ndcg@{k}']:>9.4f} "
              f"{m['p50_ms']:>7.1f} {m['p99_ms']:>8.1f} {delta:>10.1f}%")
    reset_chaos()

    json.dump({"offline": offline, "rows": rows},
              open(state_dir() / "attribution.json", "w"), indent=2)
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--candidates", type=int, default=200)
    a = ap.parse_args()
    attribution(n_users=a.users, n_candidates=a.candidates)
