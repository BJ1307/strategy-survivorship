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
        "The pointwise Wilson interval covers Monte-Carlo noise in one rate on one "
        "test set and nothing else, so the report had no magnitude for the "
        "uncertainty contributed by the calibration draw itself. Added a "
        "replication study (module `robustness`, run as part of the pipeline) that "
        "repeats simulate -> calibrate -> freeze -> measure under independent root "
        "seeds, decomposes the variance and tests the excess over the binomial "
        "term. Section 5.4 reports the result; the Wilson widths are now labelled "
        "as a lower bound.",
    ),
    (
        "The first version of that replication study was underpowered",
        "At 20 replications the estimated standard deviations carried a ~16% "
        "relative standard error, wide enough that two of the eight combinations "
        "returned an inflation factor below 1 -- impossible in expectation, since "
        "total variance cannot fall below the binomial term. Raising the default to "
        "100 replications (~7% relative SE) resolved it: all eight combinations now "
        "reject 'no calibration excess'. The lesson is recorded because the "
        "under-powered version would have supported a wrong sentence in the report.",
    ),
    (
        "The known-volatility control's answer was computed but never stated",
        "The control exists to separate 'estimating sigma' from the rest of the "
        "rolling detectors' handicap, and the numbers showed the two rolling rows "
        "are nearly identical. That conclusion -- volatility estimation costs "
        "almost nothing at this sample size, the handicap is the 252-day start-up "
        "delay -- was missing from the report and has been added.",
    ),
    (
        "Comparison-fairness caveat was incomplete",
        "The report warned that the Bayesian detectors update from day 1 while the "
        "rolling ones wait a year, but omitted the larger advantage: under this DGP "
        "the Bayesian likelihood is exactly correct and the true alternative S=1 is "
        "literally one of its two hypotheses, while the trailing Sharpe knows "
        "neither. Both halves of the caveat are now stated.",
    ),
    (
        "The calibration-FAR column could be read as a result",
        "With N = 5,000 and alpha in {0.05, 0.15}, floor(alpha*N)/N equals alpha "
        "exactly, so that column is a mechanical property of the threshold rule, "
        "not evidence that anything works. The report now says so at the point of "
        "use.",
    ),
    (
        "Figure 1 drew one model's threshold beside two models' alarms",
        "The panel plotted only the Gaussian alarm thresholds while marking the "
        "first alarm of both Bayesian detectors, each computed against its own "
        "threshold, so the Student-t marker sat visibly below the drawn line. Each "
        "detector's threshold is now drawn in its own colour.",
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
    "operating regime, not evidence that the Bayesian recursion is intrinsically "
    "better.",
    "Both Bayesian detectors assume the true Sharpe is exactly 0 or exactly 1, and "
    "under this DGP that assumption is exactly right -- the true alternative is "
    "literally one of the two hypotheses, and the Gaussian detector's noise law is "
    "correct as well. The trailing Sharpe assumes neither. A meaningful part of "
    "the measured gap is therefore correct specification, which is not free in "
    "practice; a continuous or three-state prior is deferred to a later stage.",
    "Thresholds are calibrated on a finite (5,000-path) calibration sample. "
    "Empirical control on that sample is not a guarantee about the population "
    "false-alarm rate; the independent test FAR can and does land slightly either "
    "side of the target. The replication study in section 5.4 quantifies this: "
    "the realised FAR scatters with a standard deviation about 1.3-1.6 times the "
    "binomial SE of a single measurement.",
    "The reported Wilson intervals are pointwise and are therefore a LOWER bound "
    "on the real uncertainty. They cover Monte-Carlo noise in one rate at one day "
    "-- not the whole time curve simultaneously, and not the calibration draw. "
    "Use the inflated standard deviation from section 5.4 when judging whether a "
    "realised FAR sits on target.",
    "The Student-t detector is evaluated only under a Gaussian DGP here, where it "
    "is mis-specified by construction. Down-weighting one outlier is not a "
    "solution to persistent stochastic volatility, and nothing in this stage tests "
    "that claim.",
    "The single-shock diagnostic uses one pre-registered path. It explains update "
    "behaviour and nothing else; it contributes no detection or false-alarm "
    "estimate.",
)
