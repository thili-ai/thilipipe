"""Train everything this drill needs, from scratch, in this drill.

There is deliberately NO runtime dependency on the two-tower drill's artifacts. A learner can
unlock this course in a fresh container having never opened the previous one, and the platform
will not warn them — `prereq` in meta.yml is not consumed by the ingest, the DB or the UI. So a
cross-course dependency on a file left on disk would fail with an unexplained FileNotFoundError.

It is also better teaching. The offline-to-online gap has to be measured on one system by one
learner, or it is two unrelated numbers subtracted.

    python -m thilipipe.bootstrap --epochs 8
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from thilirec import Popularity, evaluate_topk, leave_last_n_out, to_interactions
from thilitower import (
    ExactIndex, RetrievalEngine, TwoTowerModel, build_item_content,
    download_ml1m, load_ml1m, load_ml1m_movies, train_two_tower,
)

from .state import state_dir

K = 10


def load(n_test: int = 5):
    path = download_ml1m()
    ratings = load_ml1m(path)
    train_df, test_df = leave_last_n_out(ratings, n=n_test)
    users = np.sort(ratings["user_id"].unique())
    items = np.sort(ratings["item_id"].unique())
    train = to_interactions(train_df, users, items)
    test = to_interactions(test_df, users, items)
    content = build_item_content(load_ml1m_movies(path), train.item_ids)
    return ratings, train_df, train, test, content


def _split_recent(train_df, frac: float = 0.10):
    """Split the training interactions into 'what the last batch job saw' and 'what arrived since'.

    This is what makes the stale-index lesson real rather than simulated. The stale index is not
    corrupted or randomised — it is a correct index, built by a correct batch job, that simply
    ran before the most recent 10% of interactions existed.
    """
    cut = train_df["timestamp"].quantile(1.0 - frac)
    return train_df[train_df["timestamp"] <= cut], train_df[train_df["timestamp"] > cut]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--ranker-epochs", type=int, default=8)
    args = ap.parse_args()

    out = state_dir()
    ratings, train_df, train, test, content = load()
    print(f"{train.n_users:,} users x {train.n_items:,} items, "
          f"{train.n_observed:,} train / {test.n_observed:,} test")

    # --- the current model: everything the batch job would have by now -------------------
    t0 = time.time()
    model = TwoTowerModel(
        train.n_users, train.n_items, content.features, embed_dim=args.embed_dim,
    )
    train_two_tower(model, train, epochs=args.epochs, use_logq=True)
    print(f"two-tower trained in {time.time() - t0:.0f}s")

    engine = RetrievalEngine(model, ExactIndex()).build()

    # --- the previous batch run: same code, older data -----------------------------------
    old_df, recent_df = _split_recent(train_df)
    old = to_interactions(old_df, train.user_ids, train.item_ids)
    stale_model = TwoTowerModel(
        train.n_users, train.n_items, content.features, embed_dim=args.embed_dim,
    )
    train_two_tower(stale_model, old, epochs=args.epochs, use_logq=True)
    stale_engine = RetrievalEngine(stale_model, ExactIndex()).build()
    print(f"stale index built from {len(old_df):,} of {len(train_df):,} interactions "
          f"({len(recent_df):,} arrived after it ran)")

    # --- the ranker: drill 2's NeuMF, reused ---------------------------------------------
    from thilirec.ncf import NeuMF
    ranker = NeuMF(n_factors=16, n_negatives=4, epochs=args.ranker_epochs, batch_size=1024)
    ranker.fit(train, verbose=False)
    print("ranker trained")

    # --- artifacts the services read ------------------------------------------------------
    np.save(out / "item_vectors.npy", engine.item_vectors)
    np.save(out / "user_vectors.npy", engine.user_vectors)
    np.save(out / "item_vectors_stale.npy", stale_engine.item_vectors)
    np.save(out / "user_vectors_stale.npy", stale_engine.user_vectors)
    np.save(out / "train_mask.npy", train.mask)
    np.save(out / "test_mask.npy", test.mask)
    np.save(out / "item_counts.npy", train.item_counts())
    torch.save(ranker.net.state_dict(), out / "ranker.pt")
    json.dump(
        {"n_users": int(train.n_users), "n_items": int(train.n_items),
         "n_factors": 16, "embed_dim": args.embed_dim, "k": K},
        open(out / "meta.json", "w"), indent=2,
    )

    # --- the offline baseline: warm, complete, in-process ---------------------------------
    pop = Popularity().fit(train)
    pop_recs = [
        np.argsort(-np.where(train.mask[u], -np.inf, pop.predict()[u]))[:K].tolist()
        for u in range(train.n_users)
    ]
    offline_pop = evaluate_topk(pop_recs, test.mask, k=K)
    offline_retrieval = engine.evaluate(train.mask, test.mask, k=K)

    # THE baseline for the attribution, and it must be the *same pipeline* the services run —
    # retrieval AND ranking, over the same candidate depth. Scoring retrieval-only offline
    # against retrieval+ranking online measures the ranker, not the system, and makes the
    # plumbing look free. (It did: the first run of this harness reported the live system as
    # 1.1% BETTER than "offline", which is the ranker leaking into the comparison.)
    from thilitower import neumf_ranker
    all_users = np.arange(train.n_users)
    offline_recs = engine.recommend(
        all_users, k=K, exclude=train.mask, ranker=neumf_ranker(ranker), n_candidates=200,
    )
    offline = evaluate_topk(offline_recs, test.mask, k=K)
    # Saved per user so the harness can score the *same sample* it replays. Comparing a
    # 6,040-user offline number against an 800-user online one measures the sample, not the
    # system — and the difference is the same order of magnitude as the effects we are hunting.
    json.dump([list(map(int, r)) for r in offline_recs],
              open(out / "offline_recs.json", "w"))

    print("\n=== offline baselines (in-process, warm, complete) ===")
    print("popularity          ", "  ".join(f"{k} {v:.4f}" for k, v in offline_pop.items()))
    print("two-tower only      ", "  ".join(f"{k} {v:.4f}" for k, v in offline_retrieval.items()))
    print("two-tower + ranker  ", "  ".join(f"{k} {v:.4f}" for k, v in offline.items()),
          "  <- the attribution baseline")

    json.dump(
        {"offline": offline, "offline_retrieval_only": offline_retrieval,
         "offline_popularity": offline_pop},
        open(out / "offline.json", "w"), indent=2,
    )
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
