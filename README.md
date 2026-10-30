# sacco-credit-bounds

Credit scoring and zero-default PD bounds for small savings and credit cooperatives (SACCOs), with a synthetic member book so anyone can run and check the code.

Author: Nahom Behailu Geremew, Amani Solutions, Addis Ababa.

## Why this exists

Ethiopian SACCOs keep years of savings and repayment records, but their loan books are small and often fully collateralised. A book like that can have zero formal defaults, so the usual default model has nothing to learn from, and a lender or insurer still needs an honest upper bound on the probability of default.

This repository has four parts:

1. `pipeline/generate.py` builds a synthetic member book and ledger in a CRMS-style export format (CRMS is the Ethiopian government's cooperative platform).
2. `pipeline/score.py` fits an interpretable scorecard (binned features, weight of evidence, logistic regression) and reports Gini and KS, in-sample and 5-fold cross-validated, overall and for borrowers only.
3. `pipeline/bounds.py` computes upper bounds on PD when a book has no defaults: Pluto and Tasche with independent defaults, the one-factor (Vasicek) version, a beta-binomial version, and a Bayesian bound under a uniform prior.
4. `pipeline/coop_recourse.py` is a first toy simulation of score gaming when borrowers own part of the lender (see below).

## No member data

Every record here is synthetic and labelled `SYNTHETIC`. The generator uses only aggregate figures from a real pilot book (490 member records, 374 scorable, 50 borrowers, zero formal defaults, 115 late-payment penalty postings across 31 members). Names, balances, dates, loan sizes and behaviour are invented with a fixed seed. No real record is read, copied or imitated.

The generator's knobs were tuned so the scoring run lands near the pilot's published summary statistics. That match is by construction. The pipeline shows the method runs end to end. It is not evidence that the model works.

## Run it

```
pip install -r requirements.txt
make all     # generate the book, score it, compute the bounds
make test    # 17 tests
```

Outputs go to `results/`. The generator is deterministic, so every run gives the same files.

## Results on the synthetic book

Outcome: at least one late-payment penalty in the outcome window (not default, since the book has none).

| Measure | In-sample | 5-fold CV |
|---|---|---|
| Gini, all 374 scored members | 0.76 | 0.68 |
| Gini, the 50 borrowers only | 0.30 (bootstrap 95%: -0.03 to 0.60) | 0.12 |
| KS, all scored members | 0.64 | |

The gap between the two Gini rows is the main lesson: most of the all-member Gini separates borrowers from non-borrowers, and among the 50 members who actually borrowed the signal is weak and uncertain. 94.4% of scored members fall in one band.

## Zero-default bounds, 50 loans, 90% confidence

| Method | Parameter | Upper bound on PD |
|---|---|---|
| Pluto-Tasche, independent defaults | | 4.50% |
| Bayes, uniform prior | | 4.41% |
| Pluto-Tasche, one factor | rho = 0.03 | 5.22% |
| Pluto-Tasche, one factor | rho = 0.04 | 5.48% |
| Pluto-Tasche, one factor | rho = 0.12 | 7.79% |
| Pluto-Tasche, one factor | rho = 0.16 | 9.15% |
| Beta-binomial | rho = 0.01 | 5.54% |
| Beta-binomial | rho = 0.05 | 9.02% |

Basel's retail asset correlations run from 0.03 to 0.16, so for a book of 50 loans the honest range is roughly 4.5% to 9%. The bound depends far more on the correlation assumption than on the scorecard.

## A first toy simulation: recourse when borrowers own the lender

In a SACCO the borrowers are also owners, so a member who games a score to get a loan bears part of the loss if it goes bad. `coop_recourse.py` adds that to a simplified, one-round version of the competitive recourse setting in Ceccon et al. (arXiv 2509.22102): a fixed classifier, and a lender that approves the top K applicants each round. Rejected applicants get advice on what score to reach and act if the loan's value beats a quadratic cost plus kappa times their own expected loss (kappa is the share of the loss a borrower bears himself; kappa = 0 is an arm's-length lender). Acting raises the score without changing creditworthiness. In round 1 every score is re-measured with noise and the lender again takes the top K. Validity is the share of those who acted who are approved in round 1.

Naive advice says "reach the old cutoff". Durable advice says "reach the cutoff that results once everyone following this advice has acted" (a fixed point).

With the invented parameters in the file (half of applicants approved, loss given default 6 times the loan's value, re-measurement noise 0.3):

| Advice | kappa | Share acting | Cutoff shift | Validity | Default rate of approved |
|---|---|---|---|---|---|
| naive | 0 | 30.5% | 0.19 | 26.0% | 2.92% |
| naive | 1 | 23.8% | 0.16 | 29.8% | 2.74% |
| durable | 0 | 22.3% | 0.31 | 49.6% | 2.95% |
| durable | 1 | 18.1% | 0.23 | 49.6% | 2.72% |

Before anyone acted, the approved pool's default rate was 2.53%.

What it shows, with every parameter invented:
- Ownership makes fewer members act and raises the validity of naive advice by about 4 points. The direction holds at re-measurement noise of 0.1, 0.3 and 0.5 and across seeds; the size depends on the noise (13% to 15% at 0.1, 34% to 38% at 0.5).
- Durable advice roughly doubles validity, but fewer members act on it. Its validity of about 50% is largely by construction: the target sits at the expected cutoff and the noise is symmetric. It is not evidence that durable advice works in practice.
- Gaming still raises the approved pool's default rate with or without ownership, because the lender here only counts slots.

## Open questions

- How should bounds be pooled across hundreds of small cooperatives that mostly have zero defaults and share one economic factor?
- Can an out-of-time test say anything useful when only a few dozen members borrow in each period?
- What recourse advice stays valid after the lender re-fits, once borrowers own the lender?

## References

- Pluto, K. and Tasche, D. (2005). Thinking positively. Risk, 18(8), 72-78.
- Vasicek, O. (2002). The distribution of loan portfolio value. Risk, 15(12).
- Kiefer, N. M. (2009). Default estimation for low-default portfolios. Journal of Empirical Finance, 16(1), 164-173.
- Tasche, D. (2013). Bayesian estimation of probabilities of default for low default portfolios. Journal of Risk Management in Financial Institutions, 6(3).

## License

MIT
