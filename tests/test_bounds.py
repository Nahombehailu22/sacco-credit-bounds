import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bounds as B  # noqa: E402


def test_independent_closed_form():
    assert B.pluto_tasche_independent(50, 0.9) == pytest.approx(1 - 0.1 ** (1 / 50))
    assert B.pluto_tasche_independent(50, 0.9) == pytest.approx(0.0450, abs=1e-4)


def test_independent_with_defaults_matches_binomial():
    p = B.pluto_tasche_independent(100, 0.9, k=2)
    assert binom.cdf(2, 100, p) == pytest.approx(0.1, abs=1e-8)


def test_one_factor_reduces_to_independent():
    assert B.pluto_tasche_one_factor(50, 0.0) == pytest.approx(B.pluto_tasche_independent(50))


def test_one_factor_increases_with_correlation():
    vals = [B.pluto_tasche_one_factor(50, r) for r in (0.0, 0.03, 0.08, 0.16, 0.3)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_one_factor_matches_monte_carlo():
    rng = np.random.default_rng(0)
    p, rho, n = 0.05, 0.12, 50
    z = rng.standard_normal(400_000)
    pz = B._cond_pd(p, rho, z)
    mc = np.mean((1 - pz) ** n)
    assert B.prob_no_default_one_factor(p, n, rho) == pytest.approx(mc, rel=5e-3)


def test_beta_binomial_reduces_and_increases():
    assert B.beta_binomial(50, 0.0) == pytest.approx(B.pluto_tasche_independent(50))
    assert B.beta_binomial(50, 0.01) < B.beta_binomial(50, 0.05)


def test_bayes_uniform_closed_form():
    # posterior Beta(1, n + 1): upper quantile solves (1 - p)^(n + 1) = 1 - conf
    assert B.bayes_uniform(50) == pytest.approx(1 - 0.1 ** (1 / 51))


def test_bounds_shrink_with_more_loans():
    assert B.pluto_tasche_independent(500) < B.pluto_tasche_independent(50)
    assert B.pluto_tasche_one_factor(500, 0.12) < B.pluto_tasche_one_factor(50, 0.12)
