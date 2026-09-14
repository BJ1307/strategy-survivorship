# Next round: benchmarking the noise model against standard alternatives

**Status: a design, written before anything was run. Nothing in this document is
implemented, no model here has been fitted, and no number below is a result.** It is
written so the comparison is fixed in advance and cannot be tuned after the fact.

## The question

The incumbent noise model is a log-variance AR(1) stochastic-volatility kernel with an
optional jump component. It has never been compared with anything. Two standard
alternatives would answer the obvious question — *is this family doing any work?* —
and one further comparison answers a question the recovery experiment forced on us.

Three families, on the S&P 500:

| family | shape parameters | what it can and cannot produce |
|---|---|---|
| **iid Student-t** | ν | fat tails, **no** volatility clustering: every \|ε\| autocorrelation is zero in population |
| **GARCH(1,1)-t** | ω, α, β, ν | fat tails and clustering with a deterministic variance recursion |
| **log-variance AR(1) SV** (incumbent) | A, ρ (grid), κ, λ, K if jumps are on | fat tails and clustering with a stochastic variance |

## The part that is easy to get wrong

A family and the procedure used to fit it are **two different things**, and the
recovery experiment showed the procedure is not innocent: on twenty pseudo-histories
where the truth was inside the grid, the existing screen-then-re-score routine selected
the exact truth in 5 of 20 and 4 of 20. A head-to-head in which the SV family is fitted
by that routine and GARCH-t is fitted by maximum likelihood would conflate "this family
is worse" with "this estimator is worse", and would do so in a direction that
flatters GARCH.

So the round reports a **grid of (family × estimator)**, not a league table of families:

| | its natural estimator | the incumbent's simulated-moment routine |
|---|---|---|
| iid-t | MLE on the training window | grid over ν, same screen-then-re-score |
| GARCH(1,1)-t | MLE on the training window | grid over (α, β, ν) with variance targeting |
| SV (incumbent) | — (no MLE is available for it here) | the existing routine, unchanged |

Six cells, one of which is empty. Reading down a column compares families at a fixed
estimator. Reading across a row compares estimators at a fixed family. The SV row has
one cell, which is itself a finding to state plainly: the incumbent has no likelihood
we can evaluate, so it can never appear in the left column, and any conclusion drawn
from that column alone does not transfer to it.

## What is held common

Everything that is not the object of comparison:

- **One frozen return sample, drawn once.** Every method reads the same vector, taken
  from `market_returns.sample` under `RETURN_BOUNDARY_CONVENTION.md` and then frozen.
  Under that convention the S&P training window is **1,259** returns, not the 1,258 of
  the earlier stages, because the first day now reaches back one close. Parameters
  estimated under the older convention are to be labelled as such and **must not be
  described as re-estimated on this sample**.
- **Training window** 2017-01-01..2021-12-31 (1,259 S&P returns under the frozen
  convention). **Evaluation window** 2022-01-01..2023-12-31 (501 returns), parameters
  frozen at the training-window estimates, nothing re-fitted.
- **2024-2025 is not read.** Not for a statistic, not for a plot, not for a check.
- **Scale.** One annualised scale σ̂ is estimated from the training window by the
  existing estimator and imposed on all three families, exactly as the recovery
  experiment does. For GARCH this means **variance targeting**: ω is set so the
  unconditional variance equals σ̂²/252 rather than being freely estimated. The
  alternative — letting each family choose its own scale — makes the comparison partly
  about scale, and the question here is about shape. If variance targeting is dropped,
  that must be stated and the scale reported separately.
- **Innovations.** The t innovations are standardised to unit variance (scale by
  √((ν−2)/ν)), so ν controls tail shape and not scale. ν is constrained above 2; a
  fitted ν at the boundary is reported, not silently clipped.
- **Evaluation metrics**, fixed here and not changed afterwards:
  1. the existing eight-component objective J (3 RV21 log quantiles, 5 centred-|r| ACF
     lags) — the fitting criterion, so it is the *least* informative of the three;
  2. the energy score on the same eight standardised components with equal weights;
  3. the five **unfitted** diagnostics already in use — `tail_below_m3_count`,
     `tail_above_p3_count`, `acf_sqret_lag1`, `acf_sqret_lag5`, `acf_sqret_lag21`.
  Metric 3 carries the weight. A family that wins on J alone has won on the thing it
  was fitted to.
- **Simulation budget and seeds.** Equal path counts per family, common random numbers
  across families within a replicate, and seeds derived by `blake2b` — never Python's
  `hash()`, which is per-process randomised and made stage 4 non-reproducible once.
- **Path length** equal to the real window, so nothing gains from a longer sample.

## What would count as evidence, and what would not

Stated now, so it cannot be chosen later.

- **Against the incumbent:** GARCH(1,1)-t beats it on the five unfitted diagnostics,
  under a matched estimator, on the frozen 2022-2023 window, by more than the
  Monte-Carlo standard error of the paired difference.
- **For the incumbent:** the reverse, on the same terms.
- **Neither:** iid-t doing badly on the |r| autocorrelations. Its population value there
  is zero by construction; confirming that is a check that the pipeline works, not a
  finding about the incumbent.
- **Neither:** any count of how often a real value lands inside a simulated range. A
  range is a spread across simulated paths, not a confidence interval, and landing
  inside one does not make a model correct — the same rule the cross-asset round
  already operates under.
- **Neither:** an information criterion. The SV fit has no likelihood here, so AIC or
  BIC cannot be computed across the three families, and comparing them on a criterion
  two of the three can evaluate would be a comparison of what is computable.
- **Expected in advance:** the three families differ in how many parameters they search
  (1, 3-4, 2 on a 40-cell grid). Out-of-sample evaluation on a frozen window does not
  neutralise that. So each estimator also gets a **recovery check of its own** — simulate
  from a known member of the family, refit, and report how often the truth is recovered
  — before its cross-family result is read.

## Cost, and what is deliberately out of scope

Roughly the scale of one recovery run: three families, two estimators where both
apply, the existing path budget. It does not require re-fitting any other market.

Out of scope for that round, and to stay out unless asked: re-fitting the fifteen other
objects, re-running the sequential monitor, touching the 2024-2025 hold-out, and
widening the SV grid. The benchmark is about the **noise side only** — none of these
three families carries a known Sharpe ratio, which is the label the project is actually
built on, and a better noise model does not supply one.
