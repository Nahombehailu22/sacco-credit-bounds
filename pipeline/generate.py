#!/usr/bin/env python3
"""
SYNTHETIC SACCO member book. Not real member data.

Built only from published summary statistics of a single-office SACCO book:
  490 member records, 374 scorable, 50 members with loans, fully collateralised,
  zero formal defaults, 115 late-payment penalty postings across 31 members.
Everything else (names, savings, dates, loan sizes, repayment behaviour) is
invented with a seeded random generator. No real record is read, copied or imitated.

Simulation assumptions (invented here, not facts about any real SACCO):
  * Snapshot date is 31 Mar 2026. Penalties are levied from 1 Oct 2025 (the "outcome
    window"). Instalments paid 1 to 7 days late before that date carry no penalty.
  * on_time_ratio is measured on instalments due BEFORE the window, so the feature
    comes before the outcome it is later used to rank.
  * Flat 12% yearly interest and a 100 ETB share price are placeholders.

Outputs: synthetic_member_book.csv (CRMS-style member export),
         synthetic_ledger.csv (savings and loan ledger), generate_summary.json
"""
import csv
import json
import math
import os
from datetime import date, timedelta

import numpy as np

SEED = 20261028
SNAPSHOT = date(2026, 3, 31)
WINDOW_START = date(2025, 10, 1)
N_MEMBERS, N_SCORABLE, N_BORROWERS = 490, 374, 50
N_PEN_MEMBERS, N_PEN_POSTINGS = 31, 115
N_NEW, N_DORMANT, N_INCOMPLETE = 38, 54, 24  # sums to 490 - 374 = 116
SHARE_PRICE = 100
FLAT_RATE = 0.12
GRACE_DAYS = 7
# Calibration knobs. They were tuned so that the scoring run lands near the published
# summary statistics (all-member Gini about 0.75, borrower-only Gini about 0.3 to 0.4,
# about 94% of members in the top band). The match is therefore BY CONSTRUCTION. It is
# not evidence that a model works. See score.py for the same warning.
N_PEN_NB = 9       # of the 31 penalised members, how many have no loan (late savings deposits)
NB_SLOPE = 2.5     # how strongly low savings and short tenure raise late-deposit risk
Z_NOISE = 2.35     # noise in the hidden lateness driver for borrowers
HIST_FACTOR = 0.4  # how much pre-window lateness reflects the hidden driver
SAV_EFFECT = 0.6   # how much low monthly savings raises a borrower's lateness risk
Q_POWER = 1.0      # how sharply penalties concentrate on the riskiest borrowers

EC_MONTHS = ["Meskerem", "Tikimt", "Hidar", "Tahsas", "Tir", "Yekatit", "Megabit",
             "Miazia", "Ginbot", "Sene", "Hamle", "Nehase", "Pagume"]


def greg_to_ec(d):
    """Gregorian date -> (year, month, day) in the Ethiopian calendar (JDN method)."""
    jdn = d.toordinal() + 1721425
    r = (jdn - 1723856) % 1461
    n = r % 365 + 365 * (r // 1460)
    year = 4 * ((jdn - 1723856) // 1461) + r // 365 - r // 1460
    return year, n // 30 + 1, n % 30 + 1


def ec_iso(d):
    y, m, dd = greg_to_ec(d)
    return f"{y:04d}-{m:02d}-{dd:02d}"


def ec_text(d):
    y, m, dd = greg_to_ec(d)
    return f"{EC_MONTHS[m - 1]} {dd}, {y}"


def add_months(d, n):
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return date(y, m + 1, min(d.day, 28))


def months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month - (1 if a.day > b.day else 0)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out, exist_ok=True)
    os.chdir(out)
    rng = np.random.default_rng(SEED)

    # ---- member groups -------------------------------------------------------
    kinds = (["ok"] * N_SCORABLE + ["new"] * N_NEW + ["dormant"] * N_DORMANT
             + ["incomplete"] * N_INCOMPLETE)
    assert len(kinds) == N_MEMBERS
    kinds = list(rng.permutation(kinds))

    members = []
    for i, kind in enumerate(kinds, start=1):
        if kind == "new":
            T = int(rng.integers(0, 3))
        elif kind == "dormant":
            T = int(rng.integers(12, 41))
        else:
            T = 3 + int(rng.beta(1.2, 1.6) * 38)  # 3..40 months
            T = min(T, 40)
        reg = add_months(date(2026, 3, int(rng.integers(1, 29))), -T)
        monthly = float(np.clip(round(rng.lognormal(math.log(600), 0.6) / 50) * 50, 100, 5000))
        shares = max(1, int(round(rng.lognormal(math.log(6), 0.7) * (monthly / 600) ** 0.4)))
        members.append(dict(idx=i, kind=kind, T=T, reg=reg, monthly=monthly, shares=shares,
                            loan=None))

    # ---- deposits ------------------------------------------------------------
    for m in members:
        first = date(m["reg"].year, m["reg"].month, 25)
        if m["reg"].day > 25:
            first = add_months(first, 1)
        last_off = 0
        if m["kind"] == "dormant":
            last_off = int(rng.integers(7, min(18, m["T"] - 2) + 1))
        last = add_months(date(2026, 3, 25), -last_off)
        dep, d = [], first
        while d <= last and d <= SNAPSHOT:
            dep.append(d)
            d = add_months(d, 1)
        m["deposits"] = dep

    # ---- borrowers -----------------------------------------------------------
    elig = [m for m in members if m["kind"] == "ok" and m["T"] >= 12]
    pick = rng.choice(len(elig), size=N_BORROWERS, replace=False)
    borrowers = [elig[j] for j in pick]
    terms = [6, 9, 12, 18, 24, 36]
    tprob = [.08, .10, .30, .20, .22, .10]
    for m in borrowers:
        term = int(rng.choice(terms, p=tprob))
        k = int(rng.integers(1, min(term, 30, m["T"] - 3) + 1))
        amount = float(np.clip(round(rng.lognormal(math.log(45000), 0.7) / 1000) * 1000,
                               10000, 300000))
        disb = add_months(date(2026, 3, int(rng.integers(1, 29))), -k)
        m["loan"] = dict(amount=amount, term=term, k=k, disb=disb,
                         collateral=round(amount * rng.uniform(1.0, 1.5) / 1000) * 1000,
                         installment=round(amount * (1 + FLAT_RATE * term / 12) / term))
    # latent propensity to pay late (hidden driver of the simulation)
    lts = np.array([math.log(m["loan"]["amount"] / max(m["monthly"] * len(m["deposits"]), 1))
                    for m in borrowers])
    zt = np.array([m["loan"]["term"] for m in borrowers], dtype=float)
    zten = np.array([m["T"] for m in borrowers], dtype=float)
    sv = np.array([math.log(m["monthly"]) for m in borrowers])
    z = (-SAV_EFFECT * (sv - sv.mean()) / sv.std() + 0.7 * (lts - lts.mean()) / lts.std() + 0.4 * (zt - zt.mean()) / zt.std()
         - 0.3 * (zten - zten.mean()) / zten.std() + rng.normal(0, Z_NOISE, len(borrowers)))
    q = 1 / (1 + np.exp(-(-1.0 + z)))

    # pre-window lateness (inside the grace period: no penalty)
    for m, qi in zip(borrowers, q):
        L = m["loan"]
        due = [add_months(L["disb"], j) for j in range(1, L["k"] + 1)]
        L["due"] = due
        L["hist_idx"] = [j for j, d in enumerate(due) if d < WINDOW_START]
        L["win_idx"] = [j for j, d in enumerate(due) if d >= WINDOW_START]
        n_late = int(rng.binomial(len(L["hist_idx"]), HIST_FACTOR * qi)) if L["hist_idx"] else 0
        L["hist_late"] = set(rng.choice(L["hist_idx"], size=n_late, replace=False).tolist()) \
            if n_late else set()
        L["q"] = float(qi)

    # penalised members: weighted draws without replacement, then 115 postings.
    # Most are borrowers (late instalments). A few are non-borrowers who paid savings
    # deposits late (assumption of this simulation: the SACCO penalises late deposits too).
    n_pen_b = N_PEN_MEMBERS - N_PEN_NB
    keys = np.array([rng.random() ** (1 / (m["loan"]["q"] ** Q_POWER)) for m in borrowers])
    pen_members = [borrowers[j] for j in np.argsort(-keys)[:n_pen_b]]
    for m in members:  # late-deposit candidates: scorable, no loan, enough deposits in the window
        win_dep = [i for i, d in enumerate(m["deposits"])
                   if d >= WINDOW_START and (SNAPSHOT - d).days >= 10]
        m["nb_idx"] = win_dep if (m["kind"] == "ok" and not m["loan"] and m["T"] >= 6) else []
    nb = [m for m in members if m["nb_idx"]]
    zs = np.array([math.log(m["monthly"]) for m in nb])
    zt = np.array([m["T"] for m in nb], dtype=float)
    zz = 0.6 * (zs - zs.mean()) / zs.std() + 0.4 * (zt - zt.mean()) / zt.std()
    nbw = np.exp(-NB_SLOPE * zz)
    nkeys = np.array([rng.random() ** (1 / w) for w in nbw])
    pen_members += [nb[j] for j in np.argsort(-nkeys)[:N_PEN_NB]]
    def cap_of(m):
        return 2 * (len(m["loan"]["win_idx"]) if m["loan"] else len(m["nb_idx"]))
    def weight_of(m):
        return m["loan"]["q"] if m["loan"] else 0.35
    cap = {m["idx"]: cap_of(m) for m in pen_members}
    assert sum(cap.values()) >= N_PEN_POSTINGS, "caps too small for the target"
    cnt = {m["idx"]: 1 for m in pen_members}
    while sum(cnt.values()) < N_PEN_POSTINGS:
        open_ = [m for m in pen_members if cnt[m["idx"]] < cap[m["idx"]]]
        pw = np.array([weight_of(m) for m in open_])
        cnt[open_[int(rng.choice(len(open_), p=pw / pw.sum()))]["idx"]] += 1

    # ---- ledger --------------------------------------------------------------
    ledger = []

    def post(m, d, typ, desc, debit=0.0, credit=0.0, **kw):
        ledger.append(dict(member_id=f"SYN-{m['idx']:04d}", date=d, type=typ, desc=desc,
                           debit=debit, credit=credit, **kw))

    for m in members:
        if m["kind"] != "incomplete":
            post(m, m["reg"], "SHARE_PURCHASE", f"{m['shares']} shares at {SHARE_PRICE} ETB",
                 credit=m["shares"] * SHARE_PRICE)
        m["pen_n"], m["pen_total"] = 0, 0.0
        L = m["loan"]
        late_dep, extra_dep = {}, {}
        if not L and cnt.get(m["idx"], 0):
            pen = cnt[m["idx"]]
            n_late = min(pen, len(m["nb_idx"]))
            late_dep = {i: int(rng.integers(GRACE_DAYS + 1, 25))
                        for i in rng.choice(m["nb_idx"], size=n_late, replace=False).tolist()}
            extra_dep = {i: 1 for i in late_dep}
            for _ in range(pen - n_late):
                i = int(rng.choice(sorted(late_dep)))
                while extra_dep[i] >= 2:
                    i = int(rng.choice(sorted(late_dep)))
                extra_dep[i] += 1
        pen_dep = max(10, int(round(0.02 * m["monthly"] / 5) * 5))
        for i, d in enumerate(m["deposits"]):
            paid = min(d + timedelta(days=late_dep.get(i, 0)), SNAPSHOT)
            post(m, paid, "SAVINGS_DEPOSIT", "Monthly savings deposit"
                 + (f" ({late_dep[i]} days late)" if i in late_dep else ""), credit=m["monthly"])
            for p in range(extra_dep.get(i, 0)):
                pdte = min(d + timedelta(days=GRACE_DAYS + 1 + (0 if p == 0 else
                           int(rng.integers(5, 12)))), SNAPSHOT)
                post(m, pdte, "PENALTY_POSTING", "Late-payment penalty, savings deposit",
                     debit=pen_dep)
                m["pen_n"] += 1
                m["pen_total"] += pen_dep
        if not L:
            m["outstanding"] = 0.0
            continue
        post(m, L["disb"], "LOAN_DISBURSEMENT", f"Loan {L['term']} months, collateral held",
             debit=L["amount"])
        principal = L["amount"] / L["term"]
        paid_principal = 0.0
        pen = cnt.get(m["idx"], 0)
        late_win = set()
        extra = {}
        if pen:
            n_late = min(pen, len(L["win_idx"]))
            late_win = set(rng.choice(L["win_idx"], size=n_late, replace=False).tolist())
            extra = {j: 1 for j in late_win}
            for _ in range(pen - n_late):
                j = int(rng.choice(sorted(late_win)))
                while extra[j] >= 2:
                    j = int(rng.choice(sorted(late_win)))
                extra[j] += 1
        pen_amt = max(20, int(round(0.02 * L["installment"] / 5) * 5))
        ontime_hist = 0
        max_over = 0
        for j, due in enumerate(L["due"]):
            if j in late_win:
                delay = int(rng.integers(GRACE_DAYS + 1, 31))
            elif j in L["hist_late"]:
                delay = int(rng.integers(1, GRACE_DAYS + 1))
            else:
                delay = 0
            if j in L["hist_idx"] and delay == 0:
                ontime_hist += 1
            pay = due + timedelta(days=delay)
            if pay > SNAPSHOT:
                max_over = max(max_over, (SNAPSHOT - due).days)
            if pay <= SNAPSHOT:
                paid_principal += principal
                post(m, pay, "LOAN_REPAYMENT", f"Instalment {j + 1} of {L['term']}"
                     + (f" ({delay} days late)" if delay else ""),
                     credit=L["installment"], principal=principal)
            for p in range(extra.get(j, 0)):
                pdte = min(due + timedelta(days=GRACE_DAYS + 1 + (0 if p == 0 else
                           int(rng.integers(7, 21)))), SNAPSHOT)
                post(m, pdte, "PENALTY_POSTING", f"Late-payment penalty, instalment {j + 1}",
                     debit=pen_amt)
                m["pen_n"] += 1
                m["pen_total"] += pen_amt
        m["outstanding"] = round(L["amount"] - paid_principal, 0)
        L["max_over"] = max_over
        L["hist_n"] = len(L["hist_idx"])
        L["on_time"] = (ontime_hist / L["hist_n"]) if L["hist_n"] else None
    assert sum(m["pen_n"] for m in members) == N_PEN_POSTINGS
    assert sum(1 for m in members if m["pen_n"]) == N_PEN_MEMBERS

    # running balances
    order = {"SHARE_PURCHASE": 0, "LOAN_DISBURSEMENT": 1, "SAVINGS_DEPOSIT": 2,
             "LOAN_REPAYMENT": 3, "PENALTY_POSTING": 4}
    ledger.sort(key=lambda r: (r["member_id"], r["date"], order[r["type"]]))
    bal = {}
    for r in ledger:
        s, l = bal.get(r["member_id"], (0.0, 0.0))
        if r["type"] == "SAVINGS_DEPOSIT":
            s += r["credit"]
        elif r["type"] == "LOAN_DISBURSEMENT":
            l += r["debit"]
        elif r["type"] == "LOAN_REPAYMENT":
            l -= r["principal"]
        bal[r["member_id"]] = (s, l)
        r["savings_balance"], r["loan_balance"] = round(s), round(l)

    # ---- write CSVs ----------------------------------------------------------
    cols = ["member_id", "member_name", "registration_date_gc", "registration_date_ec",
            "membership_months", "monthly_savings_etb", "savings_balance_etb", "share_count",
            "share_capital_etb", "loan_id", "loan_amount_etb", "loan_term_months",
            "loan_disbursed_gc", "loan_disbursed_ec", "loan_outstanding_etb",
            "collateral_value_etb", "hist_installments_due", "on_time_ratio_hist",
            "max_days_overdue_at_snapshot", "penalty_postings", "penalty_total_etb", "default_flag", "scorable",
            "exclusion_reason", "data_label"]
    reasons = {"new": "MEMBER_UNDER_3_MONTHS", "dormant": "NO_DEPOSIT_6_MONTHS",
               "incomplete": "INCOMPLETE_RECORD"}
    with open("synthetic_member_book.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for m in sorted(members, key=lambda x: x["idx"]):
            L = m["loan"]
            bal_s = m["monthly"] * len(m["deposits"])
            wr.writerow({
                "member_id": f"SYN-{m['idx']:04d}",
                "member_name": f"Synthetic Member {m['idx']:04d}",
                "registration_date_gc": m["reg"].isoformat(),
                "registration_date_ec": ec_iso(m["reg"]),
                "membership_months": m["T"],
                "monthly_savings_etb": int(m["monthly"]),
                "savings_balance_etb": int(bal_s),
                "share_count": "" if m["kind"] == "incomplete" else m["shares"],
                "share_capital_etb": "" if m["kind"] == "incomplete" else m["shares"] * SHARE_PRICE,
                "loan_id": f"SYN-L{m['idx']:04d}" if L else "",
                "loan_amount_etb": int(L["amount"]) if L else 0,
                "loan_term_months": L["term"] if L else 0,
                "loan_disbursed_gc": L["disb"].isoformat() if L else "",
                "loan_disbursed_ec": ec_iso(L["disb"]) if L else "",
                "loan_outstanding_etb": int(m["outstanding"]),
                "collateral_value_etb": int(L["collateral"]) if L else 0,
                "hist_installments_due": L["hist_n"] if L else 0,
                "on_time_ratio_hist": ("" if (not L or L["on_time"] is None)
                                       else round(L["on_time"], 4)),
                "max_days_overdue_at_snapshot": L["max_over"] if L else 0,
                "penalty_postings": m["pen_n"],
                "penalty_total_etb": int(m["pen_total"]),
                "default_flag": 0,
                "scorable": 1 if m["kind"] == "ok" else 0,
                "exclusion_reason": reasons.get(m["kind"], ""),
                "data_label": "SYNTHETIC",
            })
    lcols = ["member_id", "txn_date_gc", "txn_date_ec", "type", "description", "debit_etb",
             "credit_etb", "savings_balance_etb", "loan_balance_etb", "data_label"]
    with open("synthetic_ledger.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(lcols)
        for r in ledger:
            wr.writerow([r["member_id"], r["date"].isoformat(), ec_iso(r["date"]), r["type"],
                         r["desc"], int(r["debit"]), int(r["credit"]), r["savings_balance"],
                         r["loan_balance"], "SYNTHETIC"])

    summary = dict(data="SYNTHETIC", seed=SEED, snapshot_gc=SNAPSHOT.isoformat(),
                   snapshot_ec=ec_text(SNAPSHOT), members=len(members),
                   scorable=sum(1 for m in members if m["kind"] == "ok"),
                   borrowers=sum(1 for m in members if m["loan"]),
                   penalty_postings=sum(m["pen_n"] for m in members),
                   penalty_members=sum(1 for m in members if m["pen_n"]), defaults=0,
                   ledger_rows=len(ledger))
    with open("generate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SYNTHETIC DATA. Not real member records.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
