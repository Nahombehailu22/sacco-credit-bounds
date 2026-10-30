"""
Toy simulation: does recourse advice stay valid when borrowers own part of the lender?

Setting (follows the competitive setting of Ceccon et al., arXiv 2509.22102, in a
simplified one-round form; all parameters are invented):
- N applicants per round. True creditworthiness q ~ N(0, 1). Default probability
  PD(q) = sigmoid(A - B_SLOPE * q). The lender sees a score x = q + noise.
- The classifier is fixed. Each round the lender approves the top K applicants by
  score, so the cutoff is recomputed from the scores it sees (competition for K slots).
- Round 0: the cutoff t0 admits the top K. Each rejected applicant is told a target
  score. Naive advice: reach t0. Durable advice: reach t_d, the cutoff that results
  once everyone who follows the durable advice has acted (a fixed point).
- An applicant acts if the loan's value beats the cost of moving plus his own
  expected loss:  VALUE - kappa * PD(q) * LOSS - C * gap^2 > 0.
  kappa is the share of a default's loss the borrower bears himself (through his
  savings and shares); kappa = 0 is an arm's-length lender.
  Acting is gaming here: it raises the score without changing q.
- Round 1: applicants who acted reach their target, then every score is
  re-measured with noise RESCORE, and the lender again approves the top K.
- Recourse validity: the share of applicants who acted on the advice who are
  approved in round 1. The approved pool's default rate is also reported.

Outputs: results/coop_recourse.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N = 100_000
K_SHARE = 0.5    # approval slots as a share of applicants
NOISE = 0.6      # score noise
A, B_SLOPE = -3.2, 1.0
VALUE = 1.0      # value of getting the loan
LOSS = 6.0       # loss given default, same units
C = 1.0          # cost of moving the score, per unit gap squared
RESCORE = 0.3    # noise when scores are re-measured in round 1
SEED = 20261030
KAPPAS = [0.0, 0.25, 0.5, 1.0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _round1(x0, target, acts, k, rescore, rng):
    x1 = np.where(acts, np.maximum(x0, target), x0) + rescore * rng.standard_normal(len(x0))
    cut = np.partition(x1, -k)[-k]
    return x1 >= cut, cut


def simulate(kappa: float, advice: str = "naive", seed: int = SEED, loss: float = LOSS,
             rescore: float = RESCORE) -> dict:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(N)
    pd_ = sigmoid(A - B_SLOPE * q)
    x0 = q + NOISE * rng.standard_normal(N)
    k = int(K_SHARE * N)
    t0 = float(np.partition(x0, -k)[-k])
    net_value = VALUE - kappa * pd_ * loss
    rejected = x0 < t0

    def acts_for(target):
        gap = np.maximum(target - x0, 0)
        return rejected & (net_value - C * gap ** 2 > 0)

    if advice == "naive":
        target = t0
    elif advice == "durable":
        # fixed point: the target equals the round-1 cutoff produced when followers aim at it
        target = t0
        for _ in range(100):
            sub = np.random.default_rng(seed + 1)
            _, cut = _round1(x0, target, acts_for(target), k, rescore, sub)
            if abs(cut - target) < 1e-4:
                break
            target = 0.5 * target + 0.5 * cut
    else:
        raise ValueError(advice)

    acts = acts_for(target)
    approved, cut = _round1(x0, target, acts, k, rescore, np.random.default_rng(seed + 2))
    validity = float(approved[acts].mean()) if acts.any() else float("nan")
    return dict(kappa=kappa, advice=advice, target_minus_t0=float(target - t0),
                share_acting=float(acts.mean()), cutoff_shift=float(cut - t0),
                recourse_validity=validity,
                pd_approved=float(pd_[approved].mean()),
                pd_approved_round0=float(pd_[~rejected].mean()))


def main():
    rows = [simulate(k, a) for a in ("naive", "durable") for k in KAPPAS]
    sens = [dict(rescore=r, **{f"validity_{a}_k{k}": simulate(k, a, rescore=r)["recourse_validity"]
                               for a in ("naive", "durable") for k in (0.0, 1.0)})
            for r in (0.1, 0.3, 0.5)]
    seeds = [simulate(1.0, "naive", seed=s)["recourse_validity"] for s in range(SEED, SEED + 5)]

    print("Top-K lender, one round. kappa = share of his own default loss a borrower bears.")
    print(f"{'advice':>8} {'kappa':>6} {'acting':>7} {'cutoff shift':>13} {'valid':>7} {'PD approved':>12}")
    for r in rows:
        print(f"{r['advice']:>8} {r['kappa']:6.2f} {r['share_acting']:7.1%} {r['cutoff_shift']:13.3f} "
              f"{r['recourse_validity']:7.1%} {r['pd_approved']:12.2%}")
    print(f"Round-0 approved pool PD: {rows[0]['pd_approved_round0']:.2%}")
    print("Sensitivity of validity to re-measurement noise:")
    for s in sens:
        print("  " + ", ".join(f"{k}={v:.1%}" if k != "rescore" else f"rescore={v}" for k, v in s.items()))
    print(f"Naive validity at kappa=1 over 5 seeds: {min(seeds):.1%} to {max(seeds):.1%}")

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    params = dict(N=N, K_SHARE=K_SHARE, NOISE=NOISE, A=A, B_SLOPE=B_SLOPE, VALUE=VALUE,
                  LOSS=LOSS, C=C, RESCORE=RESCORE, SEED=SEED)
    (out / "coop_recourse.json").write_text(json.dumps(
        dict(params=params, rows=rows, rescore_sensitivity=sens, naive_k1_5_seeds=seeds), indent=2))
    print("Wrote results/coop_recourse.json")


if __name__ == "__main__":
    main()
