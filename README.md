# thilipipe

The system *around* a recommendation model — four services, an orchestrator with a latency budget,
a batch job, and a set of switches that inject the failures every deployed recommender actually has.

> ⚠️ **Educational reference, not production.** Five uvicorn processes on localhost is not a
> distributed system. It is the smallest thing that still has process boundaries, serialization,
> timeouts and independent update cadences — which is where the recommendation quality goes. The
> latencies it measures are a floor, not an estimate: there is no network here.

Most recommender code you will read ends at a trained model and an offline metric. Production does
not. The retrieval layer of a real system is microservices stitched together with neural networks
inconsistently integrated — a hand-written rules filter beside a learned ranker beside a nightly
batch job, each deployed on its own cadence, none of them agreeing about what time it is.

This library lets you build that, then break it on purpose and measure what it costs.

## Built lesson-by-lesson in a Learn Fast course

Reference implementation for **[Ship a Production Retrieval System](https://learnfast.thili.ai)**
(`lfrecsys-lfpipeline`).

| Lesson | You build | File |
|---|---|---|
| 1 | Train your own two-tower and get your own offline baseline | `bootstrap.py` |
| 2 | The five processes, and the boundary between them | `run.py`, `services/` |
| 3 | The latency budget, stage by stage | `orchestrator.py` |
| 4–5 | The feature store, and the id drift that skews it | `services/features.py` |
| 6 | The stale index — a correct index built before the data existed | `services/candidates.py` |
| 7 | Hand-written eligibility rules beside a learned ranker | `services/filter.py` |
| 8 | Timeouts, and fail-open vs fail-closed | `orchestrator.py`, `services/rank.py` |
| 9 | Over-fetch under seen-item exclusion | `services/candidates.py` |
| 10–11 | Attribute the offline→online gap, then fix what is cheap | `harness.py`, `chaos` switches |

There is **no runtime dependency on the two-tower course.** `bootstrap.py` trains everything from
scratch in minutes, so this runs in a fresh container whether or not you did the previous one.

## Quickstart

```bash
pip install -e .                       # plus thilirec and thilitower
python -m thilipipe.bootstrap          # trains, ~2 min on CPU, writes state/
python -m thilipipe.run &              # five services on 127.0.0.1:8800-8804
python -m thilipipe.harness --users 800
```

```python
from thilipipe import set_chaos, reset_chaos

set_chaos(stale_index=True)            # the services pick it up on the next request
reset_chaos()
```

## What's inside

| File | What it is |
|---|---|
| `bootstrap.py` | Trains the two-tower, a deliberately stale second copy, and the NeuMF ranker |
| `services/candidates.py` | Two-tower retrieval + over-fetch + the stale index |
| `services/features.py` | Resolves a user to the ranker's embedding row — and drifts it |
| `services/filter.py` | Hand-written eligibility rules, not a model |
| `services/rank.py` | NeuMF, and a budget it sometimes exceeds |
| `orchestrator.py` | Fan-out, the latency budget, and the failure policy |
| `harness.py` | Replays users through the live system and attributes the gap |
| `state.py` | The shared artifacts and the chaos switches |

## License

MIT © thili.ai
