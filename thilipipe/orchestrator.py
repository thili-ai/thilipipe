"""The orchestrator: fan-out, a latency budget, and a policy for when a stage fails.

The only service with no model in it, and the one that decides what the user actually sees. Every
interesting decision in a production recommender lives here rather than in any of the models:
how long to wait, what to do when a stage times out, and whether a degraded answer beats none.
"""

from __future__ import annotations

import time

import httpx
from fastapi import FastAPI

from .state import read_chaos

app = FastAPI(title="orchestrator")

CANDIDATES = "http://127.0.0.1:8801"
FEATURES = "http://127.0.0.1:8802"
FILTER = "http://127.0.0.1:8803"
RANK = "http://127.0.0.1:8804"

BUDGET_S = 0.100          # the whole request, end to end
_client = httpx.Client(timeout=BUDGET_S)


@app.get("/health")
def health() -> dict:
    out = {}
    for name, url in (("candidates", CANDIDATES), ("features", FEATURES),
                      ("filter", FILTER), ("rank", RANK)):
        try:
            out[name] = _client.get(f"{url}/health", timeout=2.0).json()
        except Exception as exc:                                  # noqa: BLE001
            out[name] = {"ok": False, "error": str(exc)[:80]}
    return out


@app.get("/recommend")
def recommend(user_id: int, k: int = 10, n_candidates: int = 200) -> dict:
    cfg = read_chaos()
    t0 = time.perf_counter()
    stages: dict[str, float] = {}
    notes: list[str] = []

    def mark(name: str, start: float) -> None:
        stages[name] = round((time.perf_counter() - start) * 1000, 2)

    t = time.perf_counter()
    cand = _client.post(f"{CANDIDATES}/candidates",
                        json={"user_id": user_id, "n": n_candidates}).json()
    mark("candidates", t)
    items = cand["items"]
    if cand["stale"]:
        notes.append("stale_index")
    if cand["capped"]:
        notes.append("fetch_capped")

    t = time.perf_counter()
    feat = _client.get(f"{FEATURES}/features/{user_id}").json()
    mark("features", t)
    if feat["skewed"]:
        notes.append("feature_skew")

    t = time.perf_counter()
    filt = _client.post(f"{FILTER}/filter", json={"items": items}).json()
    mark("filter", t)
    items = filt["items"]
    if filt["applied"] and filt["dropped"]:
        notes.append(f"filtered_{filt['dropped']}")

    # The ranker is the only stage allowed to fail. Retrieval failing means no answer exists;
    # ranking failing means we still have a shortlist in retrieval order, which is a real —
    # if worse — recommendation. Which of those to serve is the policy decision.
    ranked, t = None, time.perf_counter()
    try:
        r = _client.post(f"{RANK}/rank", json={"ranker_row": feat["ranker_row"], "items": items})
        if r.status_code == 200:
            ranked = r.json()["items"]
        else:
            notes.append("ranker_failed")
    except Exception:                                             # noqa: BLE001
        notes.append("ranker_failed")
    mark("rank", t)

    if ranked is None:
        if cfg["failure_policy"] == "closed":
            # Serve nothing rather than something unranked. Defensible — an unranked list is
            # not what the product promised — and it is measured as the total miss it is.
            items, notes = [], notes + ["failed_closed"]
        else:
            notes.append("failed_open")
    else:
        items = ranked

    return {
        "user_id": user_id,
        "items": items[:k],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "stages": stages,
        "notes": notes,
    }
