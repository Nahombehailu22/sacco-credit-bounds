"""
Upper confidence bounds on the probability of default (PD) for a loan book with
zero (or very few) observed defaults.

These are standard methods, collected here so they can be run on a small
cooperative book:

* pluto_tasche_independent: Pluto and Tasche (2005), single grade, independent
  defaults. With zero defaults in n loans the bound solves (1 - p)^n = 1 - conf.
* pluto_tasche_one_factor: the same bound when defaults share one Gaussian
  factor (Vasicek model) with asset correlation rho. Defaults are conditionally
  independent given the factor, so P(no default) = E_Z[(1 - p(Z))^n].
* beta_binomial: defaults are binomial given a PD drawn from a Beta distribution
  whose intra-class correlation is rho. P(no default) = B(a, b + n) / B(a, b).
* bayes_uniform: posterior upper quantile under a uniform prior,
  posterior Beta(k + 1, n - k + 1).

References
- Pluto, K. and Tasche, D. (2005). Thinking positively. Risk, 18(8), 72-78.
- Vasicek, O. (2002). The distribution of loan portfolio value. Risk, 15(12).
- Kiefer, N. M. (2009). Default estimation for low-default portfolios.
  Journal of Empirical Finance, 16(1), 164-173.
- Tasche, D. (2013). Bayesian estimation of probabilities of default for low
  default portfolios. Journal of Risk Management in Financial Institutions, 6(3).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln
from scipy.stats import beta as beta_dist
from scipy.stats import binom, norm

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(200)  # nodes for E[f(Z)], Z ~ N(0, 1)
_GH_W = _GH_W / _GH_W.sum()


def pluto_tasche_independent(n: int, conf: float = 0.9, k: int = 0) -> float:
    """Largest PD consistent with seeing at most k defaults in n independent loans."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k == 0:
        return 1 - (1 - conf) ** (1 / n)
    return brentq(lambda p: binom.cdf(k, n, p) - (1 - conf), 1e-12, 1 - 1e-12)


def _cond_pd(p: float, rho: float, z: np.ndarray) -> np.ndarray:
    return norm.cdf((norm.ppf(p) - math.sqrt(rho) * z) / math.sqrt(1 - rho))


def prob_no_default_one_factor(p: float, n: int, rho: float) -> float:
    if rho == 0:
        return (1 - p) ** n
    return float(np.sum(_GH_W * (1 - _cond_pd(p, rho, _GH_X)) ** n))


def pluto_tasche_one_factor(n: int, rho: float, conf: float = 0.9) -> float:
    """Zero-default bound when defaults share one Gaussian factor with correlation rho."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    return brentq(lambda p: prob_no_default_one_factor(p, n, rho) - (1 - conf), 1e-10, 0.999)


def prob_no_default_beta_binomial(p: float, n: int, rho: float) -> float:
    if rho == 0:
        return (1 - p) ** n
    s = (1 - rho) / rho
    a, b = p * s, (1 - p) * s
    return math.exp(betaln(a, b + n) - betaln(a, b))


def beta_binomial(n: int, rho: float, conf: float = 0.9) -> float:
    """Zero-default bound when the PD is Beta-distributed with intra-class correlation rho."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    return brentq(lambda p: prob_no_default_beta_binomial(p, n, rho) - (1 - conf), 1e-10, 0.999)


def bayes_uniform(n: int, k: int = 0, conf: float = 0.9) -> float:
    """Upper conf-quantile of the posterior PD under a uniform prior."""
    return float(beta_dist.ppf(conf, k + 1, n - k + 1))


def table(n: int = 50, conf: float = 0.9) -> list[dict]:
    rows = [dict(method="Pluto-Tasche, independent", param="", bound=pluto_tasche_independent(n, conf)),
            dict(method="Bayes, uniform prior", param="", bound=bayes_uniform(n, 0, conf))]
    for rho in (0.03, 0.04, 0.12, 0.16):
        rows.append(dict(method="Pluto-Tasche, one factor", param=f"rho={rho}",
                         bound=pluto_tasche_one_factor(n, rho, conf)))
    for rho in (0.01, 0.05):
        rows.append(dict(method="Beta-binomial", param=f"rho={rho}", bound=beta_binomial(n, rho, conf)))
    return rows


if __name__ == "__main__":
    import json
    from pathlib import Path

    n, conf = 50, 0.9
    rows = table(n, conf)
    print(f"Upper bounds on PD, {n} loans, zero defaults, {conf:.0%} confidence")
    for r in rows:
        print(f"  {r['method']:28s} {r['param']:10s} {r['bound']:.2%}")
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "bounds.json").write_text(json.dumps(dict(n=n, conf=conf, rows=rows), indent=2))
    print("Wrote results/bounds.json")
