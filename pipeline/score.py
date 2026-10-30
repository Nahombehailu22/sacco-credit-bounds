#!/usr/bin/env python3
"""
Interpretable scorecard on the SYNTHETIC member book (run generate.py first).
Binned features -> weight of evidence -> logistic regression -> 300-850 band.
Outcome: at least one late-payment penalty posting in the window (not default).
Writes results/scored_members.csv and results/results.json.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from generate import SNAPSHOT, ec_text, ec_iso
from datetime import date

HERE = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(7)
NA = "No loan"  # loan features of non-borrowers are undefined: neutral, never learned
WOE_PRIOR = 0.5
NEUTRAL_NOLOAN = False
WOE_CAP = 0.45  # cap on |WoE| per bin (winsorising). Also a calibration knob: see CALIBRATED SCALE below
BANDS = [("A", 750, 850), ("B", 700, 749), ("C", 650, 699), ("D", 600, 649), ("E", 300, 599)]
FEATURES = ["monthly_savings", "membership_months", "share_count", "loan_amount",
            "loan_term", "on_time_ratio"]
LABELS = {"monthly_savings": "Monthly savings (ETB)", "membership_months": "Months of membership",
          "share_count": "Share holdings", "loan_amount": "Loan amount (ETB)",
          "loan_term": "Loan term (months)", "on_time_ratio": "On-time repayment ratio"}
# CALIBRATED SCALE (read this). Three knots map the log-odds of "no penalty" to points:
# the riskiest member -> 300, the 6th-percentile member -> 750, the safest -> 850.
# The knots were set on this seed-fixed synthetic book so that about 94% of members land in
# band A, the concentration published for the real book. So the band shares below are
# a calibration target met by construction, and the scale is NOT a points-to-double-the-odds scale.
KNOTS_O = [-0.30, 0.25, 4.50]
KNOTS_S = [300, 750, 850]
ZERO_DEFAULT_BOUND = 1 - 0.1 ** (1 / 50)


def gini(y_bad, risk):
    """Gini = 2 * AUC - 1. Higher risk should mean more likely bad."""
    return 2 * roc_auc_score(y_bad, risk) - 1


def ks(y_bad, risk):
    """Kolmogorov-Smirnov statistic: largest gap between the bad and good score CDFs."""
    fpr, tpr, _ = roc_curve(y_bad, risk)
    return float(np.max(tpr - fpr))


# ---- binning ---------------------------------------------------------------
def make_spec(tr):
    """Bin edges learned on the training rows only."""
    spec = {}
    for f in ["monthly_savings", "membership_months", "share_count"]:
        e = np.unique(np.quantile(tr[f], [.25, .5, .75]))
        spec[f] = list(e)
    b = tr[tr.loan_amount > 0]
    spec["loan_amount"] = list(np.unique(np.quantile(b.loan_amount, [1 / 3, 2 / 3])))
    spec["loan_term"] = [12, 24]
    spec["on_time_ratio"] = [0.85, 1.0]
    return spec


def bin_of(f, v, spec):
    e = spec[f]
    if f in ("loan_amount", "loan_term") and v == 0:
        return NA
    if f == "on_time_ratio":
        if v == -2:
            return NA
        if v == -1:
            return "No history yet"
        return "<0.85" if v < 0.85 else ("0.85 to <1.00" if v < 1.0 else "1.00")
    i = int(np.searchsorted(e, v, side="left"))
    lo = "" if i == 0 else f">{e[i - 1]:,.0f}"
    hi = "" if i == len(e) else f"<={e[i]:,.0f}"
    return f"{lo} and {hi}".strip().replace(" and ", " to ") if lo and hi else (hi or lo)


def binned(df, spec):
    return pd.DataFrame({f: [bin_of(f, v, spec) for v in df[f]] for f in FEATURES}, index=df.index)


def woe_tables(b, good):
    out = {}
    for f in FEATURES:
        keep = (b[f] != NA).values if NEUTRAL_NOLOAN else np.ones(len(b), bool)
        G, B = good[keep].sum(), (1 - good[keep]).sum()
        t = pd.DataFrame({"bin": b[f][keep], "good": good[keep]}).groupby("bin").good.agg(["sum", "count"])
        t["bad"] = t["count"] - t["sum"]
        a = WOE_PRIOR  # pseudo-count per bin: shrinks the weight of evidence of bins with no events
        t["woe"] = np.log(((t["sum"] + a) / (G + a * len(t))) / ((t["bad"] + a) / (B + a * len(t))))
        t["woe"] = t["woe"].clip(-WOE_CAP, WOE_CAP)
        t["iv"] = ((t["sum"] / G) - (t["bad"] / B)) * t["woe"]
        out[f] = t
    return out


def apply_woe(b, wt):
    return np.column_stack([b[f].map(wt[f]["woe"]).fillna(0.0).values for f in FEATURES])


def fit_signed_logit(X, y, C=1.0):
    """L2 logistic regression on WoE inputs with coefficients constrained to be >= 0.
    WoE already points the right way, so a negative weight would only be an artefact of
    features overlapping. The constraint keeps every bin's points readable."""
    n, k = X.shape

    def f(p):
        w, b = p[:k], p[k]
        z = X @ w + b
        ll = np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)
        pr = 1 / (1 + np.exp(-z))
        gz = pr - y
        return C * ll.sum() + 0.5 * w @ w, np.append(C * X.T @ gz + w, C * gz.sum())

    r = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                 bounds=[(0, None)] * k + [(None, None)])
    return r.x[:k], r.x[k]


class Scorecard:
    def fit(self, tr):
        self.spec = make_spec(tr)
        b = binned(tr, self.spec)
        good = 1 - tr.bad.values
        self.wt = woe_tables(b, good)
        X = apply_woe(b, self.wt)
        self.w, self.b0 = fit_signed_logit(X, good)
        return self

    def logodds(self, df):
        return apply_woe(binned(df, self.spec), self.wt) @ self.w + self.b0

    def score(self, df):
        return np.interp(self.logodds(df), KNOTS_O, KNOTS_S)


def band_of(s):
    for n, lo, hi in BANDS:
        if s >= lo:
            return n
    return "E"


def main():
    raw = pd.read_csv(HERE / "synthetic_member_book.csv")
    led = pd.read_csv(HERE / "synthetic_ledger.csv")
    assert (raw.data_label == "SYNTHETIC").all()
    df = raw.copy()
    df["bad"] = (df.penalty_postings > 0).astype(int)
    df["monthly_savings"] = df.monthly_savings_etb
    df["loan_amount"] = df.loan_amount_etb
    df["loan_term"] = df.loan_term_months
    df["on_time_ratio"] = np.where(df.loan_amount == 0, -2.0, df.on_time_ratio_hist.fillna(-1.0))
    df["borrower"] = (df.loan_amount > 0).astype(int)
    sc = df[df.scorable == 1].copy()

    # ---- fit and in-sample ---------------------------------------------------
    card = Scorecard().fit(sc)
    sc["score"] = card.score(sc)
    sc["pd"] = 1 / (1 + np.exp(card.logodds(sc)))
    sc["band"] = sc.score.map(band_of)
    risk = -card.logodds(sc)  # higher = riskier (unclipped, so ties from clipping do not matter)
    g_all = gini(sc.bad, risk)
    bor = sc.borrower == 1
    g_bor = gini(sc.bad[bor], risk[bor])
    boots = []
    bi = np.where(bor)[0]
    yb, rb = sc.bad.values[bi], risk[bi]
    for _ in range(2000):
        s = RNG.integers(0, len(bi), len(bi))
        if 0 < yb[s].sum() < len(s):
            boots.append(gini(yb[s], rb[s]))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # ---- 5-fold cross-validated (WoE and bins refit inside each fold) --------
    oof = np.zeros(len(sc))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=11).split(sc, sc.bad):
        m = Scorecard().fit(sc.iloc[tr_i])
        oof[te_i] = -m.logodds(sc.iloc[te_i])
    cv_all = gini(sc.bad, oof)
    cv_bor = gini(sc.bad[bor], oof[bor.values])

    # ---- bands ---------------------------------------------------------------
    band_rows = []
    for n, lo, hi in BANDS:
        s = sc[sc.band == n]
        band_rows.append(dict(band=n, lo=lo, hi=hi, n=int(len(s)), share=len(s) / len(sc),
                              pen_members=int(s.bad.sum()),
                              pen_rate=float(s.bad.mean()) if len(s) else None,
                              borrowers=int(s.borrower.sum())))

    # ---- portfolio risk and concentration -----------------------------------
    loans = sc[sc.borrower == 1]
    out = loans.loan_outstanding_etb
    par = float(out[loans.bad == 1].sum() / out.sum())
    par30 = float(out[loans.max_days_overdue_at_snapshot > 30].sum() / out.sum())
    top5 = float(out.nlargest(5).sum() / out.sum())
    hhi = float(((out / out.sum()) ** 2).sum())
    total_sav = float(sc.savings_balance_etb.sum())
    cover = float((loans.collateral_value_etb / loans.loan_amount).min())
    largest_band = max(band_rows, key=lambda r: r["share"])

    iv = {f: float(card.wt[f]["iv"].sum()) for f in FEATURES}
    coefs = {f: float(c) for f, c in zip(FEATURES, card.w)}

    fpr_a, tpr_a, _ = roc_curve(sc.bad, risk)
    fpr_b, tpr_b, _ = roc_curve(yb, rb)

    res = dict(
        label="SYNTHETIC DATA. Not the pilot SACCO's records. Not the real model result.",
        members=int(len(df)), scored=int(len(sc)), borrowers=int(bor.sum()),
        penalty_members=int(sc.bad.sum()), penalty_borrowers=int(sc.bad[bor].sum()), penalty_postings=int(df.penalty_postings.sum()),
        defaults=int(df.default_flag.sum()),
        gini_all=g_all, gini_borrowers=g_bor, ks_all=ks(sc.bad, risk), ks_borrowers=ks(yb, rb), gini_borrowers_boot95=ci,
        gini_all_cv=cv_all, gini_borrowers_cv=cv_bor,
        bands=band_rows, par=par, par30=par30, top5=top5, hhi=hhi,
        loans_outstanding=float(out.sum()), savings_total=total_sav,
        loans_to_savings=float(out.sum() / total_sav), min_collateral_cover=cover,
        largest_band=largest_band["band"], largest_band_share=largest_band["share"],
        zero_default_bound=ZERO_DEFAULT_BOUND, iv=iv, coef=coefs,
        intercept=float(card.b0),
    )
    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    keep = ["member_id", "member_name", "score", "band", "pd", "bad", "borrower"]
    sc[keep].to_csv(HERE / "scored_members.csv", index=False)

    # ---- console report -------------------------------------------------------
    L = "=" * 72
    print(L)
    print("SYNTHETIC DATA. Generated by generate.py from summary statistics.")
    print("These are NOT the pilot SACCO's member records and NOT the real model result.")
    print(L)
    print(f"Records {len(df)}, scored {len(sc)}, borrowers {int(bor.sum())}, "
          f"formal defaults {res['defaults']}")
    print(f"Outcome: >=1 late-payment penalty. {res['penalty_postings']} postings, "
          f"{res['penalty_members']} members. Not default.")
    print(f"\nGini, all {len(sc)} scored members, in-sample:   {g_all:.3f}")
    print(f"  5-fold cross-validated:                       {cv_all:.3f}")
    print(f"KS, all scored members, in-sample:             {res['ks_all']:.3f}")
    print(f"Gini, the {int(bor.sum())} borrowers only, in-sample:      {g_bor:.3f}"
          f"  (bootstrap 95% interval {ci[0]:.2f} to {ci[1]:.2f})")
    print(f"  5-fold cross-validated:                       {cv_bor:.3f}")
    print("\nShare of scored members by band:")
    for r in band_rows:
        pr = "n/a" if r["pen_rate"] is None else f"{r['pen_rate']:.1%}"
        print(f"  {r['band']} ({r['lo']}-{r['hi']}): {r['n']:4d} members  {r['share']:6.1%}"
              f"   with penalty: {r['pen_members']:3d}  rate {pr}")
    print(f"\nPortfolio at risk (outstanding of borrowers with a penalty): {par:.1%}")
    print(f"PAR30 (installment more than 30 days overdue at snapshot): {par30:.1%}")
    print(f"Top-5 borrowers share of outstanding: {top5:.1%}   HHI {hhi:.3f}")
    print(f"Largest band: {largest_band['band']} with {largest_band['share']:.1%} of members")
    print(f"\nZero-default bound, 50 loans, 90% confidence, Pluto-Tasche one grade:")
    print(f"  1 - 0.1^(1/50) = {ZERO_DEFAULT_BOUND:.2%}  (upper bound on PD, not an estimate)")
    n_cap = int((sc.score >= 849.5).sum())
    nb = sc[sc.borrower == 0]
    print(f"\nNon-borrowers: score range {nb.score.min():.0f} to {nb.score.max():.0f}, "
          f"{n_cap} at the 850 cap. {int(nb.bad.sum())} of them were penalised (late savings deposits).")
    print("CALIBRATION WARNING: generate.py knobs, the WoE cap and the score-scale knots were tuned so this")
    print("run lands near the published Gini and band concentration. The match is by construction.")
    print("It shows the pipeline runs end to end. It is not evidence that the model works.")
    print("\nWeight-of-evidence coefficients (on 'no penalty'):")
    for f in FEATURES:
        print(f"  {LABELS[f]:28s} coef {coefs[f]:+.2f}   IV {iv[f]:.2f}")
    print("  (coefficients constrained >= 0; a 0.00 means the feature added nothing once the others were in)")
    print(L)

    print("Wrote results/results.json and results/scored_members.csv")


if __name__ == "__main__":
    main()
