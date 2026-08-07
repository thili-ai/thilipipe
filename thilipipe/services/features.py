"""Feature store: resolves a request's user into the row the ranker was trained against.

Tiny, and the most dangerous service in the system. The ranker is an embedding table indexed by
an integer; that integer is *not* the user's identity, it is a position assigned by whichever
training run built the table. When a retrain reorders users and the store is not updated in the
same deploy, every lookup silently returns a coherent, confident, wrong row.

Nothing throws. The recommendations stay plausible. This is what train/serve skew looks like
when the "feature" is an id.
"""

from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI

from ..state import read_chaos, state_dir

app = FastAPI(title="features")

S = state_dir()
META = json.loads((S / "meta.json").read_text())
N_USERS = int(META["n_users"])

# The drift a retrain would introduce: a deterministic reshuffle of a slice of the id space.
# Deterministic so the drill's numbers reproduce; a slice rather than everything because a real
# reindex usually appends and re-sorts rather than permuting the whole table.
_rng = np.random.default_rng(0)
DRIFT = np.arange(N_USERS)
_slice = DRIFT[: int(N_USERS * 0.30)].copy()
_rng.shuffle(_slice)
DRIFT[: int(N_USERS * 0.30)] = _slice


@app.get("/health")
def health() -> dict:
    return {"ok": True, "users": N_USERS}


@app.get("/features/{user_id}")
def features(user_id: int) -> dict:
    cfg = read_chaos()
    row = int(DRIFT[user_id]) if cfg["feature_skew"] else int(user_id)
    return {"user_id": user_id, "ranker_row": row, "skewed": row != user_id}
