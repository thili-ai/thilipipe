"""Candidate generation: the two-tower retriever behind an ANN index.

Serves the shortlist. Owns the index, and therefore owns the staleness: the index is a artifact
built by a batch job on its own cadence, and this service has no way to know how old it is. That
is the point of the `stale_index` switch — nothing here errors, logs, or looks wrong.
"""

from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from ..state import read_chaos, state_dir

app = FastAPI(title="candidates")

S = state_dir()
META = json.loads((S / "meta.json").read_text())
ITEM_FRESH = np.load(S / "item_vectors.npy")
USER_FRESH = np.load(S / "user_vectors.npy")
ITEM_STALE = np.load(S / "item_vectors_stale.npy")
USER_STALE = np.load(S / "user_vectors_stale.npy")
TRAIN_MASK = np.load(S / "train_mask.npy")


class Req(BaseModel):
    user_id: int
    n: int = 200


class Resp(BaseModel):
    items: list[int]
    stale: bool
    fetched: int
    capped: bool


@app.get("/health")
def health() -> dict:
    return {"ok": True, "items": int(ITEM_FRESH.shape[0])}


@app.post("/candidates", response_model=Resp)
def candidates(req: Req) -> Resp:
    cfg = read_chaos()
    item_v, user_v = (
        (ITEM_STALE, USER_STALE) if cfg["stale_index"] else (ITEM_FRESH, USER_FRESH)
    )

    seen = TRAIN_MASK[req.user_id]
    # Over-fetch: asking for n when the user has seen m of them returns n items you cannot use.
    # The honest fetch is n + (items this user has seen); production caps it, and the cap is
    # what turns a heavy user's request into a short list rather than a slow one.
    n_fetch = req.n + int(seen.sum())
    capped = False
    if cfg["max_fetch"] is not None and n_fetch > cfg["max_fetch"]:
        n_fetch, capped = int(cfg["max_fetch"]), True
    n_fetch = min(n_fetch, item_v.shape[0])

    scores = item_v @ user_v[req.user_id]
    idx = np.argpartition(-scores, kth=min(n_fetch, len(scores)) - 1)[:n_fetch]
    idx = idx[np.argsort(-scores[idx])]
    idx = idx[~seen[idx]][: req.n]

    return Resp(items=idx.tolist(), stale=bool(cfg["stale_index"]),
                fetched=int(n_fetch), capped=capped)
