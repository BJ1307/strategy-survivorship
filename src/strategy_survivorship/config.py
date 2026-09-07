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
    # Stage 2A fix: T must not be drawn from the same SeedSequence that
    # generates the return noise.
    "switch_random_T",
    # Stage 2A item 4: independent always-valid paths for validating the
    # frozen continuation-false-alarm thresholds out of sample.
    "switch_contfa_test",
    # Stage 2A: one parent, whose children are assigned per scenario in a
    # documented order (see stage2a.scenario_streams).
    "stage2a",
    # Stage 2B gets its own parent: every method compared this round runs on
    # new shared test paths, so no Stage 2A draw is reused.
    "stage2b",
    "stage2b_persistence_control",
    "stage2b_bootstrap",
    # Stage 2C: its own parents for calibration, covered tests, stress
    # tests and the sensitivity bootstrap.
    "stage2c_calibration",
    "stage2c_test",
    "stage2c_stress",
    "stage2c_bootstrap",
    # Stage 2D: persistent SV and isolated jumps together
    "stage2d_calibration",
    "stage2d_test",
    "stage2d_bootstrap",
    "stage2d_shock",
    # Stage 2E: fresh data for the 2x2 ablation
    "stage2e_calibration",
    "stage2e_test",
    "stage2e_bootstrap",
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
    switch_matched_cal_paths: int = 4000
    # Second control: match the false-alarm rate INSIDE the evaluation window
    # rather than the cumulative pre-failure one. A scalar threshold cannot do
    # both, so both are reported. The common level is set to the highest any
    # detector family can actually reach, which for the Bayesian detectors
    # shrinks sharply with T.
    switch_matched_survival_floor: float = 0.5
    switch_contfa_test_paths: int = 5000
    # A per-T common level makes each T internally fair but not comparable
    # ACROSS T. This fixed level is feasible for every T, so the T-trend it
    # produces is apples-to-apples.
    switch_fixed_continuation_fa: float = 0.02

    # --- Stage 2A: realistic-noise stress scenarios -------------------------
    # A research stress setting, fixed before any test result was seen. Not
    # fitted to a market and not claimed to reproduce one. One feature at a
    # time; no full combination grid this round.
    noise_scenarios: tuple[str, ...] = ("gaussian", "student_t", "stoch_vol", "jump")
    noise_student_t_df: float = 5.0        # nu
    noise_sv_rho: float = 0.98             # log-variance AR(1) persistence
    noise_sv_amplitude: float = 1.0        # 0 collapses to Gaussian
    noise_jump_lambda_annual: float = 2.0  # expected jumps per YEAR
    noise_jump_kappa: float = 5.0          # sd of ONE jump, in daily-shock units
    n_noise_calibration: int = 5000
    n_noise_test_valid: int = 5000
    n_noise_test_invalid: int = 5000

    # --- Stage 2B: variance forecast from past returns only -----------------
    # lambda is a pre-fixed simple baseline (RiskMetrics), NOT searched.
    ewma_lambda: float = 0.94
    ewma_student_t_df: float = 5.0
    # numerical guard only; reported when it fires, never tuned on test results
    ewma_variance_floor_factor: float = 1e-8
    # Stage 2E: truncation constant for the variance update. Fixed at 4
    # before running and NOT searched.
    ewma_truncation_c: float = 4.0
    n_stage2b_calibration: int = 5000
    n_stage2b_test_valid: int = 5000
    n_stage2b_test_invalid: int = 5000
    # persistence control: same one-day marginal and unconditional variance as the
    # SV scenario, with the latent persistence removed.
    sv_control_rho: float = 0.0
    bootstrap_reps: int = 2000
    n_info_days: tuple[int, ...] = (126, 252, 504)

    # --- Stage 2C: unified thresholds with a calibration-error buffer -------
    # Coverage set: five full path-generating laws. Each entry is
    # (key, base noise model, ((cfg field, value), ...)).
    stage2c_coverage: tuple = (
        ("gaussian", "gaussian", ()),
        ("student_t", "student_t", ()),
        ("sv_rho098", "stoch_vol", ()),
        ("jump_k5", "jump", ()),
        ("sv_rho0", "stoch_vol", (("noise_sv_rho", 0.0),)),
    )
    # Stress scenarios deliberately OUTSIDE the coverage set. They use the frozen
    # unified thresholds and are not part of this round's probability guarantee.
    stage2c_stress: tuple = (
        ("sv_rho090", "stoch_vol", (("noise_sv_rho", 0.90),)),
        ("sv_amp15", "stoch_vol", (("noise_sv_amplitude", 1.5),)),
        ("jump_k8", "jump", (("noise_jump_kappa", 8.0),)),
    )
    stage2c_calibration_paths: int = 10000
    stage2c_test_paths: int = 10000        # per role (valid / invalid)
    stage2c_stress_paths: int = 5000       # per role
    # delta is the CALIBRATION failure budget: the probability that the
    # calibration procedure fails to bound the true FAR. It is NOT the strategy
    # false-alarm budget alpha, and NOT a test confidence level.
    stage2c_delta: float = 0.05
    stage2c_report_days: tuple[int, ...] = (63, 126, 252, 504)
    stage2c_bootstrap_reps: int = 2000

    # --- Stage 2D: SV and jumps combined ------------------------------------
    # rho = 0.98 and lambda = 2/yr are held fixed; only (A, kappa) vary.
    # (key, SV amplitude A, jump kappa, in the original guarantee coverage?)
    stage2d_scenarios: tuple = (
        ("gaussian_ctrl", 0.0, 0.0, True),
        ("sv_ctrl", 1.0, 0.0, True),
        ("jump_ctrl", 0.0, 5.0, True),
        ("sv_jump", 1.0, 5.0, False),        # main experiment
        ("sv_jump_big", 1.0, 8.0, False),    # stress
    )
    stage2d_calibration_paths: int = 10000
    stage2d_test_paths: int = 10000
    stage2d_bootstrap_reps: int = 2000
    stage2d_shock_day: int = 126
    stage2d_shock_sigmas: float = 8.0
    stage2d_shock_path_index: int = 0

    # --- Stage 2E ------------------------------------------------------------
    stage2e_calibration_paths: int = 10000
    stage2e_test_paths: int = 10000
    stage2e_bootstrap_reps: int = 2000
    # window (in trading days after a jump) for the false-alarm timing split
    stage2e_jump_window: int = 20

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
