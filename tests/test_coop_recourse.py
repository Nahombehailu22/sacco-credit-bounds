import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import coop_recourse as cr  # noqa: E402


def test_ownership_reduces_acting_and_cutoff_shift():
    rows = [cr.simulate(k, "naive") for k in (0.0, 0.5, 1.0)]
    assert rows[0]["share_acting"] > rows[1]["share_acting"] > rows[2]["share_acting"]
    assert rows[0]["cutoff_shift"] > rows[1]["cutoff_shift"] > rows[2]["cutoff_shift"]


def test_ownership_raises_naive_validity_at_every_noise_level():
    for r in (0.1, 0.3, 0.5):
        assert cr.simulate(1.0, "naive", rescore=r)["recourse_validity"] > \
               cr.simulate(0.0, "naive", rescore=r)["recourse_validity"]


def test_durable_advice_beats_naive_but_fewer_act():
    for k in (0.0, 1.0):
        n, d = cr.simulate(k, "naive"), cr.simulate(k, "durable")
        assert d["recourse_validity"] > n["recourse_validity"]
        assert d["share_acting"] < n["share_acting"]


def test_gaming_raises_pool_default_rate():
    r = cr.simulate(0.0, "naive")
    assert r["pd_approved"] > r["pd_approved_round0"]
