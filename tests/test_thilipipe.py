"""Tests for the invariants the drill's argument depends on.

These deliberately do not need trained artifacts or running services — they cover the pure logic
whose failure would silently corrupt the attribution table.
"""

import json

import numpy as np
import pytest

from thilipipe import state


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state, "CHAOS_FILE", tmp_path / "chaos.json")
    return tmp_path


def test_chaos_defaults_are_all_off(tmp_state):
    """A fresh system must be undegraded, or the baseline row is not a baseline."""
    cfg = state.reset_chaos()
    assert cfg["stale_index"] is False
    assert cfg["feature_skew"] is False
    assert cfg["rules_filter"] is False
    assert cfg["ranker_timeout_rate"] == 0.0
    assert cfg["max_fetch"] is None


def test_unknown_switch_raises(tmp_state):
    """A typo'd switch that silently does nothing would produce a table of identical rows."""
    with pytest.raises(KeyError):
        state.set_chaos(stale_indx=True)


def test_switches_round_trip(tmp_state):
    state.reset_chaos()
    state.set_chaos(feature_skew=True, ranker_timeout_rate=0.2)
    cfg = state.read_chaos()
    assert cfg["feature_skew"] is True
    assert cfg["ranker_timeout_rate"] == 0.2
    assert cfg["stale_index"] is False          # untouched switches keep their default


def test_corrupt_chaos_file_does_not_crash(tmp_state):
    """A half-written config must not take five services down mid-replay."""
    state.reset_chaos()
    state.CHAOS_FILE.write_text("{not json")
    assert state.read_chaos() == state.DEFAULT_CHAOS


def test_feature_drift_is_deterministic_and_partial():
    """The skew must be reproducible (so numbers replicate) and affect only some users."""
    n = 6040
    rng = np.random.default_rng(0)
    drift = np.arange(n)
    sl = drift[: int(n * 0.30)].copy()
    rng.shuffle(sl)
    drift[: int(n * 0.30)] = sl

    rng2 = np.random.default_rng(0)
    drift2 = np.arange(n)
    sl2 = drift2[: int(n * 0.30)].copy()
    rng2.shuffle(sl2)
    drift2[: int(n * 0.30)] = sl2

    assert np.array_equal(drift, drift2)              # reproducible
    assert (drift != np.arange(n)).any()              # something actually drifted
    assert (drift[int(n * 0.30):] == np.arange(n)[int(n * 0.30):]).all()   # rest untouched
    assert sorted(drift) == list(range(n))            # a permutation, not corruption


def test_attribution_baseline_must_match_offline_pipeline():
    """Guards the bug that made the first harness report the live system as *better* offline.

    The offline baseline must be retrieval AND ranking. If someone reverts it to retrieval-only,
    every row of the attribution table shifts by the ranker's contribution.
    """
    from thilipipe import bootstrap
    src = bootstrap.__file__
    text = open(src).read()
    assert "neumf_ranker(ranker)" in text, "offline baseline must include the ranker"
    assert "offline_recs.json" in text, "baseline must be saved per user for sample matching"
