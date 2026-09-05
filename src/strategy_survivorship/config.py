"""Central configuration for Stage 1.

Every experiment parameter used by the Stage 1 benchmark lives here.  Nothing in
``simulate``/``detectors``/``evaluate``/``plots`` invents a parameter of its own.

Units convention (see ``theory.md``)
------------------------------------
* ``D``            trading days per year (integer).
* ``sigma_annual`` annualised volatility of the daily excess-return series.
* ``sigma_daily``  = ``sigma_annual / sqrt(D)``.
* ``S``            *annualised* Sharpe ratio of the data-generating process.
* daily drift      = ``S * sigma_annual / D``  =>  daily Sharpe ``S / sqrt(D)``.

Returns are interpreted as strategy excess returns *already net of costs*; Stage 1
does not deduct a second cost layer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Order in which independent random streams are spawned from the root
# SeedSequence.  This order is part of the reproducibility contract: changing it
# changes every downstream number.  Never reorder, only append.
# Appending to this tuple is safe: SeedSequence.spawn(n) hands child i the key
# (i,), so a longer spawn leaves the first four children bit-identical and every
# Stage 1 number reproduces unchanged.  Never reorder, only append.
STREAM_ORDER: tuple[str, ...] = (
    "calibration_valid",
    "test_valid",
    "test_invalid",
    "diagnostic",
    # --- Stage 1.1 ---
    "prob_time_valid",
    "prob_time_invalid",
    "switch_fixed",
    "switch_random",
    "switch_matched_cal",
)


@dataclass(frozen=True)
class Stage1Config:
    """Immutable parameter set for the Stage 1 benchmark."""

    # --- calendar / horizon -------------------------------------------------
    trading_days_per_year: int = 252
    horizon_days: int = 504  # H, the monitoring horizon (2 years)

    # --- data generating process -------------------------------------------
    sigma_annual: float = 0.10
    sharpe_valid: float = 1.0  # annualised Sharpe of the "valid" state
    sharpe_invalid: float = 0.0  # annualised Sharpe of the "invalid" state

    # --- detectors ----------------------------------------------------------
    prior_valid: float = 0.5  # initial P(valid) for both Bayesian detectors
    student_t_df: float = 5.0  # nu, fixed for Stage 1
    rolling_window: int = 252  # trailing window length in trading days
    rolling_ddof: int = 1  # sample std uses ddof=1 (documented, not implicit)

    # --- calibration / evaluation ------------------------------------------
    far_targets: tuple[float, ...] = (0.05, 0.15)  # cumulative FAR budget over H
    n_calibration: int = 5000  # valid paths used *only* to pick thresholds
    n_test_valid: int = 5000  # independent valid paths (false-alarm measurement)
    n_test_invalid: int = 5000  # independent invalid paths (detection measurement)
    eval_horizons: tuple[int, ...] = (126, 252, 504)  # half-year / 1y / 2y
    # 100 replications put the relative SE of an estimated sd near 7%, enough to
    # separate 'no calibration excess' from a 25% inflation. 20 was not.
    # The Beta / Beta-binomial law in `analytic_calibration` is exact, so this
    # replication study is now an optional cross-check, not a default step.
    n_calibration_replications: int = 0
    wilson_z: float = 1.959963984540054  # two-sided 95%

    # --- single-shock diagnostic (NOT part of the benchmark) ---------------
    shock_day: int = 252  # 1-based trading day the shock is added on
    shock_in_daily_sigma: float = 8.0  # +/- this many sigma_daily
    diagnostic_path_index: int = 0  # fixed path id, chosen before looking at it

    # --- Stage 1.1 diagnostics ---------------------------------------------
    # Probability-time: how long belief takes to move.  These horizons are a
    # separate diagnostic and do NOT inherit the 504-day false-alarm budget.
    prob_time_paths: int = 5000
    prob_time_report_days: tuple[int, ...] = (126, 252, 504)
    prob_time_long_days: tuple[int, ...] = (1260, 2520)  # 5y, 10y
    prob_thresholds: tuple[float, ...] = (0.80, 0.90, 0.95)
    prob_coverage_target: float = 0.80
    brier_days: tuple[int, ...] = (252, 504)
    sensitivity_sharpe: tuple[float, ...] = (1.0, 0.6)

    # Random failure time.  T = last valid trading day; every path is watched for
    # `switch_post_window` further days, so the post-failure window is identical
    # for every T.
    switch_fixed_T: tuple[int, ...] = (0, 252, 756, 1260)
    switch_fixed_paths: int = 2000
    switch_random_paths: int = 5000
    switch_random_T_max: int = 1260
    switch_post_window: int = 504
    switch_post_horizons: tuple[int, ...] = (126, 252, 504)
    # Declared before the run so the random group cannot be re-binned to taste.
    switch_T_bin_edges: tuple[int, ...] = (0, 252, 504, 756, 1008, 1261)
    # Matched pre-failure false-alarm diagnostic: a common nominal budget is
    # not a common realised pre-failure alarm rate, so each detector is
    # re-calibrated on independent valid-only paths to hit this rate.
    switch_matched_pre_fa: float = 0.15
    switch_matched_cal_paths: int = 2000

    # --- reproducibility ----------------------------------------------------
    root_seed: int = 20260905

    # --- bookkeeping --------------------------------------------------------
    label: str = "stage1_benchmark"
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Benchmark DGP is fixed-volatility iid Gaussian only.",
            "Student-t detector is deliberately mis-specified under this DGP.",
        )
    )

    # --- derived ------------------------------------------------------------
    @property
    def D(self) -> int:
        return self.trading_days_per_year

    @property
    def sigma_daily(self) -> float:
        """sigma_ann / sqrt(D)."""
        return self.sigma_annual / math.sqrt(self.trading_days_per_year)

    @property
    def sqrt_D(self) -> float:
        return math.sqrt(self.trading_days_per_year)

    @property
    def student_t_scale(self) -> float:
        """a_nu = sqrt((nu-2)/nu), the scale giving unit *variance* in z-space."""
        nu = self.student_t_df
        if nu <= 2:
            raise ValueError("student_t_df must exceed 2 for a finite variance")
        return math.sqrt((nu - 2.0) / nu)

    @property
    def prior_log_odds(self) -> float:
        p = self.prior_valid
        if not 0.0 < p < 1.0:
            raise ValueError("prior_valid must lie strictly inside (0, 1)")
        return math.log(p / (1.0 - p))

    def daily_drift(self, sharpe_annual: float) -> float:
        """S * sigma_ann / D."""
        return sharpe_annual * self.sigma_annual / self.trading_days_per_year

    def validate(self) -> None:
        if self.horizon_days < self.rolling_window:
            raise ValueError("horizon shorter than the rolling window")
        if max(self.eval_horizons) > self.horizon_days:
            raise ValueError("eval horizon beyond the monitoring horizon")
        if not all(0.0 < a < 1.0 for a in self.far_targets):
            raise ValueError("FAR targets must lie in (0, 1)")
        _ = self.student_t_scale, self.prior_log_odds

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        d["far_targets"] = list(self.far_targets)
        d["eval_horizons"] = list(self.eval_horizons)
        d["derived"] = {
            "sigma_daily": self.sigma_daily,
            "sqrt_D": self.sqrt_D,
            "student_t_scale_a_nu": self.student_t_scale,
            "prior_log_odds": self.prior_log_odds,
            "daily_drift_valid": self.daily_drift(self.sharpe_valid),
            "daily_drift_invalid": self.daily_drift(self.sharpe_invalid),
            "stream_order": list(STREAM_ORDER),
        }
        return d

    @property
    def switch_max_days(self) -> int:
        """Longest path any switching group needs: latest T plus the post window."""
        return max(max(self.switch_fixed_T), self.switch_random_T_max) + self.switch_post_window

    def smoke(self) -> "Stage1Config":
        """A tiny variant used to check the pipeline runs before the full scale."""
        from dataclasses import replace

        return replace(
            self,
            n_calibration=400,
            n_test_valid=400,
            n_test_invalid=400,
            prob_time_paths=200,
            switch_fixed_paths=150,
            switch_random_paths=200,
            label="stage1_smoke",
        )


DEFAULT = Stage1Config()
