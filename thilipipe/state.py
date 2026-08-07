"""Shared on-disk state: artifacts the services read, and the chaos switches.

Every service reads this directory. That is deliberate — in production these would be a model
registry, a feature store and a config service, and the failure modes this drill teaches all come
from those three things being updated on *different cadences*. One directory with explicit files
makes the cadence visible instead of hiding it behind infrastructure.
"""

from __future__ import annotations

import json
import os
import pathlib

STATE_DIR = pathlib.Path(os.environ.get("THILIPIPE_STATE", "state")).resolve()

# --- chaos switches -----------------------------------------------------------------------
# One JSON file, re-read on every request. Re-reading per request is wasteful and correct: the
# learner flips a switch in a notebook cell and the running services must pick it up without a
# restart, or the drill becomes "edit config, restart six processes, lose your place".
CHAOS_FILE = STATE_DIR / "chaos.json"

DEFAULT_CHAOS = {
    # Serve item vectors from the previous batch run instead of the current model.
    "stale_index": False,
    # The feature store returns a user row from before the last reindex, so the ranker scores
    # the wrong user. This is embedding-table row drift, not a made-up failure.
    "feature_skew": False,
    # Apply the hand-written eligibility rules.
    "rules_filter": False,
    # Fraction of ranker calls that exceed the budget and get dropped by the orchestrator.
    "ranker_timeout_rate": 0.0,
    # Orchestrator policy when the ranker fails: "open" serves unranked candidates, "closed"
    # serves nothing.
    "failure_policy": "open",
    # Cap on how many candidates the index may be asked for, before seen-item exclusion.
    "max_fetch": None,
}


def state_dir() -> pathlib.Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def read_chaos() -> dict:
    cfg = dict(DEFAULT_CHAOS)
    if CHAOS_FILE.exists():
        try:
            cfg.update(json.loads(CHAOS_FILE.read_text()))
        except json.JSONDecodeError:
            pass          # a half-written file must not take the whole system down
    return cfg


def set_chaos(**kwargs) -> dict:
    """Flip switches from a notebook. Unknown keys raise rather than silently doing nothing."""
    unknown = set(kwargs) - set(DEFAULT_CHAOS)
    if unknown:
        raise KeyError(f"unknown chaos switch(es): {sorted(unknown)}")
    cfg = read_chaos()
    cfg.update(kwargs)
    state_dir()
    CHAOS_FILE.write_text(json.dumps(cfg, indent=2))
    return cfg


def reset_chaos() -> dict:
    state_dir()
    CHAOS_FILE.write_text(json.dumps(DEFAULT_CHAOS, indent=2))
    return dict(DEFAULT_CHAOS)
