"""thilipipe — the system around a recommendation model.

The reference implementation for the Learn Fast course *Ship a Production Retrieval System*.
Four services, an orchestrator with a latency budget, a batch job, and a set of switches that
inject the failures every deployed recommender actually has.

⚠️  Teaching-grade, not production. Five uvicorn processes on localhost is not a distributed
    system; it is the smallest thing that still has process boundaries, serialization, timeouts
    and independent update cadences — which is where the quality goes.

Depends on `thilitower` (retriever) and `thilirec` (metrics, ranker) so every number here is on
the same scale as the earlier courses in the track.
"""

from .state import DEFAULT_CHAOS, read_chaos, reset_chaos, set_chaos, state_dir

__version__ = "0.1.0"

__all__ = ["state_dir", "read_chaos", "set_chaos", "reset_chaos", "DEFAULT_CHAOS"]
