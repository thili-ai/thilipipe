"""Eligibility: hand-written business rules, deliberately not a model.

This is the "inconsistently integrated" part of the stack, and it is not a strawman — every real
recommender has one. Legal cannot be learned from click data, and nobody trains a network to
enforce a licensing window.

The cost is that these rules were written against assumptions nobody re-checks, and they run
*after* the model has ranked. A rule that drops 8% of the catalog drops 8% of the model's best
work too, and no offline evaluation of the model will ever show it.
"""

from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from ..state import read_chaos, state_dir

app = FastAPI(title="filter")

S = state_dir()
ITEM_COUNTS = np.load(S / "item_counts.npy")

# "Do not surface long-tail items we cannot vouch for." A real rule, written for a real reason
# (quality control on thin data), with a threshold somebody picked once and never revisited.
MIN_INTERACTIONS = 25
ELIGIBLE = ITEM_COUNTS >= MIN_INTERACTIONS


class Req(BaseModel):
    items: list[int]


class Resp(BaseModel):
    items: list[int]
    dropped: int
    applied: bool


@app.get("/health")
def health() -> dict:
    return {"ok": True, "eligible": int(ELIGIBLE.sum()), "catalog": int(ELIGIBLE.size)}


@app.post("/filter", response_model=Resp)
def filter_items(req: Req) -> Resp:
    cfg = read_chaos()
    if not cfg["rules_filter"]:
        return Resp(items=req.items, dropped=0, applied=False)
    keep = [i for i in req.items if ELIGIBLE[i]]
    return Resp(items=keep, dropped=len(req.items) - len(keep), applied=True)
