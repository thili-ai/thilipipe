"""Ranking: drill 2's NeuMF, scoring the shortlist retrieval handed over.

Reused rather than reinvented, because the argument is about cost. This is the exact network that
could not be run over the full catalog; putting retrieval in front of it is what makes it
affordable. Whether it is worth the hop is a question this drill measures rather than assumes —
the two-tower drill already found it barely helps at 50 candidates and hurts at 200.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..state import read_chaos, state_dir

app = FastAPI(title="rank")

S = state_dir()
META = json.loads((S / "meta.json").read_text())

from thilirec.ncf import _NeuMFNet  # noqa: E402  (needs META first)

NET = _NeuMFNet(int(META["n_users"]), int(META["n_items"]), int(META["n_factors"]), (64, 32, 16))
NET.load_state_dict(torch.load(S / "ranker.pt", map_location="cpu"))
NET.eval()


def _times_out(user_id: int, rate: float) -> bool:
    """Deterministic per user, so a run replicates.

    An unseeded random() here makes every replay a different experiment: the drill's own first
    measurement of fail-open moved between -1.1% and -3.6% across runs, which is larger than
    several of the effects the course teaches. Hashing the *user* (not the ranker row) also keeps
    the timing population identical when other switches change it, so scenarios stay comparable.
    """
    if rate <= 0:
        return False
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 10_000
    return h < rate * 10_000


class Req(BaseModel):
    ranker_row: int
    items: list[int]
    user_id: int = -1


class Resp(BaseModel):
    items: list[int]


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/rank", response_model=Resp)
def rank(req: Req) -> Resp:
    cfg = read_chaos()
    # A ranker that exceeds its budget is indistinguishable, to the orchestrator, from one that
    # is down. Returning 503 rather than sleeping keeps the drill's replay fast while producing
    # the same decision at the caller.
    if _times_out(req.user_id, float(cfg["ranker_timeout_rate"])):
        raise HTTPException(status_code=503, detail="ranker budget exceeded")

    if not req.items:
        return Resp(items=[])
    u = torch.full((len(req.items),), req.ranker_row, dtype=torch.int64)
    i = torch.tensor(req.items, dtype=torch.int64)
    with torch.no_grad():
        scores = NET(u, i).numpy()
    order = np.argsort(-scores)
    return Resp(items=[req.items[j] for j in order])
