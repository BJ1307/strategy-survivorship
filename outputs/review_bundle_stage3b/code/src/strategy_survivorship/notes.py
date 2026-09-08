"""Findings and limitations carried into the generated report.

Kept in source (rather than hand-edited into the markdown) so that the report
stays a *generated* artefact: rerunning the pipeline reproduces it byte for byte.
"""

from __future__ import annotations

# Problems found while building / self-reviewing Stage 1 and the fix applied.
FIXED_ISSUES: tuple[tuple[str, str], ...] = (
    (
        "Student-t scale confused with standard deviation",
        "A t(nu) with scale a has variance a^2 * nu/(nu-2). Using scale = 1 would "
        "have given the Student-t detector observation noise of variance nu/(nu-2) "
        "= 5/3 in z-space, i.e. a different (easier) problem from the Gaussian "
        "detector's. Fixed by a_nu = sqrt((nu-2)/nu), and pinned by a unit-variance "
        "test.",
    ),
    (
        "Rolling detectors were credited with days they cannot see",
        "A 252-day trailing window is undefined for the first 251 days. Those days "
        "are held as NaN, excluded from the calibration minimum and unable to "
        "alarm, instead of being back-filled or silently treated as an expanding "
        "window.",
    ),
    (
        "Threshold quantile could overshoot the FAR budget",
        "Taking the empirical alpha-quantile of the per-path minima with a "
        "non-strict alarm rule lets the achieved calibration FAR land above alpha "
        "when there are ties. Fixed by pairing a strict '<' alarm rule with the "
        "floor(alpha*N)-th order statistic, which is conservative by construction, "
        "and asserting achieved <= target in the run.",
    ),
    (
        "Censored paths were indistinguishable from day-504 alarms",
        "Both give min(tau, H) = 504. The first-passage table now stores the alarm "
        "flag and a -1 sentinel separately from the truncated time, so a genuine "
        "day-504 crossing and a never-crossing path stay distinguishable.",
    ),
    (
        "Median detection time conditional on detection",
        "Reporting the median only over detected paths flatters detectors that "
        "rarely fire. The median is computed over all invalid paths, and reported "
        "as 'not reached within the horizon' when fewer than half alarm by H.",
    ),
    (
        "Wilson intervals were the only stated uncertainty",
        "The pointwise Wilson interval answers 'given THIS frozen threshold, how "
        "noisy is the rate measured on 5,000 fresh test paths'. That was the only "
        "uncertainty reported, so the report said nothing about how much the whole "
        "pipeline moves when the calibration sample is redrawn. Stage 1.1 answers "
        "the second question exactly -- the threshold is an order statistic, so its "
        "true false-alarm probability is Beta(j, N+1-j) with j = floor(alpha*N)+1 -- "
        "and the report now keeps the two questions separate instead of treating "
        "one as a bound on the other.",
    ),
    (
        "The stopping rule for the replication study was invalid",
        "The replication count was raised from 20 to 100 with the stated reason "
        "that at 20 'two of the eight combinations returned an inflation factor "
        "below 1' and at 100 'all eight reject'. Deciding when to stop sampling by "
        "looking at significance biases the result towards significance, and an "
        "empirical sd landing below a reference value in a small sample is ordinary "
        "sampling noise, not evidence that the theory is wrong. Both the rule and "
        "the reasoning are withdrawn. Stage 1.1 replaces the study with the exact "
        "Beta / Beta-binomial law and keeps the 100 replications only as a one-off "
        "cross-check (it agrees, max |z| = 1.57); the replication stage is no longer "
        "part of the default run.",
    ),
    (
        "An unverified causal attribution about the rolling detectors",
        "The report stated that the rolling detectors lag 'because of the one-year "
        "start-up delay and the equal weighting inside the window'. No experiment "
        "isolated that: establishing it needs a control in which every detector may "
        "only alarm from day 252 and each is calibrated independently. The claim is "
        "deleted; only what the known-vol control actually shows is kept.",
    ),
    (
        "The Gaussian ranking was stated without its scope",
        "Gaussian leads only under the matched Gaussian DGP and the decision rules "
        "as implemented. Stage 1.1's random-failure-time diagnostic shows the "
        "ranking reverses once a strategy is valid first and fails later, so the "
        "Stage 1 comparison is now explicitly scoped and cross-references that "
        "result.",
    ),
)

# What Stage 1 does *not* establish.
LIMITATIONS: tuple[str, ...] = (
    "The benchmark DGP is fixed-volatility iid Gaussian. Fat tails, "
    "time-varying volatility, jumps and valid-then-decaying strategies are out of "
    "scope for this stage, so no conclusion here transfers to them.",
    "The daily volatility is *known* to every detector. This is an idealised "
    "condition that favours all four detectors, and the known-vol rolling control "
    "exists precisely to show how much of the rolling detectors' handicap is the "
    "volatility estimate.",
    "The two Bayesian detectors update from day 1 while the two rolling detectors "
    "are silent until day 252. Part of the measured gap is this difference in "
    "operating regime, but no experiment here isolates how much, so the cause is "
    "left open rather than attributed.",
    "Every Stage 1 result assumes the state is CONSTANT over the whole horizon. "
    "Stage 1.1 shows the detector ranking reverses once the strategy is valid "
    "first and fails later, so the Stage 1 ranking should not be carried into "
    "any decaying-strategy setting.",
    "Both Bayesian detectors assume the true Sharpe is exactly 0 or exactly 1, and "
    "under this DGP that assumption is exactly right -- the true alternative is "
    "literally one of the two hypotheses, and the Gaussian detector's noise law is "
    "correct as well. The trailing Sharpe assumes neither. A meaningful part of "
    "the measured gap is therefore correct specification, which is not free in "
    "practice; a continuous or three-state prior is deferred to a later stage.",
    "Thresholds are calibrated on a finite (5,000-path) calibration sample, so "
    "the frozen threshold's true false-alarm probability is a random variable. "
    "Its exact law is Beta(j, N+1-j) with j = floor(alpha*N)+1; see Stage 1.1 "
    "section 3. Note E[p_FA] = j/(N+1) sits slightly ABOVE the nominal alpha.",
    "The reported Wilson intervals are pointwise and conditional on the frozen "
    "threshold: they cover Monte-Carlo noise in one rate at one day, not the "
    "whole time curve simultaneously, and not the spread induced by redrawing "
    "the calibration sample. That second spread is a different question with its "
    "own exact answer (Stage 1.1 section 3); neither bounds the other.",
    "The Student-t detector is evaluated only under a Gaussian DGP here, where it "
    "is mis-specified by construction. Down-weighting one outlier is not a "
    "solution to persistent stochastic volatility, and nothing in this stage tests "
    "that claim.",
    "The single-shock diagnostic uses one pre-registered path. It explains update "
    "behaviour and nothing else; it contributes no detection or false-alarm "
    "estimate.",
)
