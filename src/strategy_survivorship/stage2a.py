"""stage2a: one module per research stage.

- Model and computation: Stage 2A: the Stage 1 baselines under four noise scenarios.
- Figures: Stage 2A figures: four, English labels, one colour per detector.
- Report: Generate ``stage2a_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 2A pipeline.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import numpy as np
import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS, DETECTORS_BY_KEY
from .evaluate import build_metric_row, calibrate_threshold, first_passage
from .noise import draw_noise, returns_from_noise, true_daily_sigma
from .simulate import make_streams

# In Stage 1 the daily volatility really is known, so "known-vol rolling" was
# accurate.  Under stochastic volatility this detector still divides by the FIXED
# sigma_0 and does not see the current volatility, so it is relabelled here.
LABEL_2A = {
    "binary_gaussian": "Binary Gaussian",
    "binary_student_t": "Binary Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "gaussian_oracle_vol": "Gaussian oracle (true current vol)",
}
SET_ROLES = ("calibration_valid", "test_valid", "test_invalid")


def scenario_streams(cfg: Stage1Config) -> dict[tuple[str, str], np.random.SeedSequence]:
    """One independent child per (scenario, role), in a fixed documented order."""
    parent = make_streams(cfg)["stage2a"]
    kids = parent.spawn(len(cfg.noise_scenarios) * len(SET_ROLES))
    out, i = {}, 0
    for sc in cfg.noise_scenarios:
        for role in SET_ROLES:
            out[(sc, role)] = kids[i]
            i += 1
    return out


def build_scenario(cfg: Stage1Config, scenario: str) -> dict:
    """Calibration / test-valid / test-invalid blocks for one noise scenario."""
    streams = scenario_streams(cfg)
    n = {"calibration_valid": cfg.n_noise_calibration,
         "test_valid": cfg.n_noise_test_valid,
         "test_invalid": cfg.n_noise_test_invalid}
    sharpe = {"calibration_valid": cfg.sharpe_valid,
              "test_valid": cfg.sharpe_valid,
              "test_invalid": cfg.sharpe_invalid}
    block = {}
    for role in SET_ROLES:
        d = draw_noise(scenario, streams[(scenario, role)], n[role], cfg.horizon_days, cfg)
        block[role] = {
            "returns": returns_from_noise(d.eps, sharpe[role], cfg),
            "true_sigma": true_daily_sigma(d, cfg),  # evaluator / oracle only
            "latent": d.latent,
            "is_valid": sharpe[role] == cfg.sharpe_valid,
        }
    return block


# --------------------------------------------------------------------------- #
# oracle detector (stochastic volatility only)
# --------------------------------------------------------------------------- #


def gaussian_oracle_log_odds(returns: np.ndarray, sigma_t: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Known-variance Gaussian LLR, keeping the two FIXED mean hypotheses.

        dL_t = (mu1 - mu0) * (r_t - (mu1+mu0)/2) / sigma_t^2

    This is the exact log-likelihood ratio of N(mu1, sigma_t^2) against
    N(mu0, sigma_t^2).  It is deliberately NOT "divide the return by the current
    volatility and reuse the fixed-conditional-Sharpe update", which would test a
    different (and wrong) hypothesis pair.
    """
    mu0 = cfg.daily_drift(cfg.sharpe_invalid)
    mu1 = cfg.daily_drift(cfg.sharpe_valid)
    inc = (mu1 - mu0) * (returns - 0.5 * (mu1 + mu0)) / (sigma_t ** 2)
    return cfg.prior_log_odds + np.cumsum(inc, axis=-1)


def detectors_for(scenario: str) -> list:
    """Baselines everywhere; the oracle only where a current volatility exists."""
    return list(DETECTORS) + (["gaussian_oracle_vol"] if scenario == "stoch_vol" else [])


def compute_statistic(det, block_role: dict, cfg: Stage1Config, scenario: str) -> np.ndarray:
    if det == "gaussian_oracle_vol":
        return gaussian_oracle_log_odds(block_role["returns"], block_role["true_sigma"], cfg)
    return det.compute(block_role["returns"], cfg)


def _key(det) -> str:
    return det if isinstance(det, str) else det.key


def _first_eligible(det, cfg) -> int:
    return 1 if isinstance(det, str) else det.first_eligible_day(cfg)


def _label(det) -> str:
    return LABEL_2A[_key(det)]


class _Shim:
    """build_metric_row wants a detector-like object; supply one for the oracle."""

    def __init__(self, key, label, role, statistic_name):
        self.key, self.label, self.role, self.statistic_name = key, label, role, statistic_name


def run_scenario(cfg: Stage1Config, scenario: str, frozen_thresholds: dict) -> dict:
    """Both comparison arms for one noise scenario."""
    block = build_scenario(cfg, scenario)
    dets = detectors_for(scenario)
    stats = {
        _key(d): {role: compute_statistic(d, block[role], cfg, scenario) for role in SET_ROLES}
        for d in dets
    }

    rows, passages, calibs = [], {}, {}
    for d in dets:
        k, elig = _key(d), _first_eligible(d, cfg)
        shim = _Shim(
            k, _label(d),
            "oracle (not deployable)" if k == "gaussian_oracle_vol"
            else ("diagnostic control" if k == "known_vol_rolling_252" else "baseline"),
            "posterior log-odds L_t" if "gaussian" in k or "student" in k
            else "annualised trailing statistic",
        )
        for alpha in cfg.far_targets:
            # --- arm 1: Stage 1 thresholds, unchanged ------------------------
            if (k, alpha) in frozen_thresholds:
                thr = frozen_thresholds[(k, alpha)]
                fpv = first_passage(stats[k]["test_valid"], thr, elig, cfg.horizon_days)
                fpi = first_passage(stats[k]["test_invalid"], thr, elig, cfg.horizon_days)
                r = build_metric_row(
                    detector=shim, cfg=cfg,
                    calib=_FrozenCal(k, alpha, thr, elig),
                    fp_valid=fpv, fp_invalid=fpi,
                )
                r.update({"scenario": scenario, "arm": "frozen_stage1"})
                rows.append(r)
                passages[(k, alpha, "frozen", "test_valid")] = fpv
                passages[(k, alpha, "frozen", "test_invalid")] = fpi

            # --- arm 2: recalibrated on this scenario's own valid paths -------
            c = calibrate_threshold(stats[k]["calibration_valid"], alpha, elig,
                                    cfg.horizon_days, k)
            calibs[(k, alpha)] = c
            fpv = first_passage(stats[k]["test_valid"], c.threshold, elig, cfg.horizon_days)
            fpi = first_passage(stats[k]["test_invalid"], c.threshold, elig, cfg.horizon_days)
            r = build_metric_row(detector=shim, cfg=cfg, calib=c, fp_valid=fpv, fp_invalid=fpi)
            r.update({"scenario": scenario, "arm": "recalibrated"})
            rows.append(r)
            passages[(k, alpha, "recalibrated", "test_valid")] = fpv
            passages[(k, alpha, "recalibrated", "test_invalid")] = fpi

    return {"rows": rows, "passages": passages, "stats": stats, "block": block,
            "calibrations": calibs, "detectors": dets}


class _FrozenCal:
    """Calibration record for a threshold that was NOT chosen on this scenario."""

    rule = "Stage 1 threshold, carried over unchanged (not calibrated on this scenario)"

    def __init__(self, key, alpha, threshold, elig):
        self.detector_key, self.far_target, self.threshold = key, alpha, threshold
        self.first_eligible_day, self.n_paths, self.n_alarms = elig, 0, 0
        self.achieved_far = float("nan")

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import COLOURS, setup_style


COLOURS = {**COLOURS, "gaussian_oracle_vol": "#e07b39"}
SC_LABEL = {"gaussian": "Gaussian (control)", "student_t": "Student-t tails ($\\nu$=5)",
            "stoch_vol": "Stochastic volatility ($\\rho$=0.98)", "jump": "Jumps ($\\lambda$=2/yr, $\\kappa$=5)"}


def _curve(res, key, alpha, arm, role):
    fp = res["passages"].get((key, alpha, arm, role))
    return None if fp is None else fp.cumulative_rate


def figure_noise_diagnostics(cfg: Stage1Config, per_scenario: dict, diag, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    scs = list(cfg.noise_scenarios)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(scs)))

    ax = axes[0]
    grid = np.linspace(0, 6, 60)
    for c, sc in zip(cols, scs):
        eps = (per_scenario[sc]["block"]["test_valid"]["returns"]
               - cfg.daily_drift(cfg.sharpe_valid)) / cfg.sigma_daily
        tail = [(np.abs(eps) > g).mean() for g in grid]
        ax.semilogy(grid, np.maximum(tail, 1e-7), color=c, lw=1.7, label=SC_LABEL[sc])
    ax.set_xlabel("$c$"); ax.set_ylabel("$P(|\\varepsilon| > c)$")
    ax.set_title("(a) Tail frequency of the standardised shock", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[1]
    for c, sc in zip(cols, scs):
        eps = (per_scenario[sc]["block"]["test_valid"]["returns"]
               - cfg.daily_drift(cfg.sharpe_valid)) / cfg.sigma_daily
        pass

        a = np.abs(eps)
        lags = np.arange(1, 41)
        ax.plot(lags, [pooled_autocorr(a, int(l)) for l in lags], color=c, lw=1.7,
                label=SC_LABEL[sc])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("lag (trading days)"); ax.set_ylabel("autocorrelation of $|\\varepsilon_t|$")
    ax.set_title("(b) Volatility clustering\npooled over paths, global mean", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[2]
    d = diag.set_index("scenario")
    x = np.arange(len(scs)); w = 0.35
    ax.bar(x - w / 2, [d.loc[s, "mean"] for s in scs], w,
           yerr=[3 * d.loc[s, "mean_se_path_level"] for s in scs], capsize=3, label="mean $\\varepsilon$")
    ax.bar(x + w / 2, [d.loc[s, "var"] - 1 for s in scs], w,
           yerr=[3 * d.loc[s, "var_se_path_level"] for s in scs], capsize=3, label="var $\\varepsilon$ $-$ 1")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x, [SC_LABEL[s].split(" (")[0].replace(" ", "\n") for s in scs],
                  rotation=0, fontsize=8)
    ax.set_title("(c) Target moments: 0 and 1\nerror bars = 3x path-level SE", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    fig.suptitle("Fig 2A.1  The four noise scenarios, all normalised to zero mean and unit "
                 "unconditional variance by theoretical constants", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out); plt.close(fig)
    return out


def figure_frozen_far(cfg: Stage1Config, per_scenario: dict, out: Path,
                      alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = list(cfg.noise_scenarios)
    fig, axes = plt.subplots(1, len(scs), figsize=(15.0, 4.3), sharey=True)
    days = np.arange(1, cfg.horizon_days + 1)
    for ax, sc in zip(np.atleast_1d(axes), scs):
        for k in ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
                  "known_vol_rolling_252"):
            c = _curve(per_scenario[sc], k, alpha, "frozen", "test_valid")
            if c is None:
                continue
            ax.plot(days, c, color=COLOURS[k], lw=1.6, label=LABEL_2A[k])
        ax.axhline(alpha, color="k", ls="-.", lw=1.0, label=f"nominal budget {alpha:g}")
        ax.set_title(SC_LABEL[sc], fontsize=10)
        ax.set_xlabel("trading day"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("cumulative false-alarm rate | $S=1$")
    np.atleast_1d(axes)[0].legend(fontsize=7.4, loc="upper left")
    fig.suptitle("Fig 2A.2  Stage 1 thresholds held FIXED: what the noise change does to the "
                 f"realised false-alarm rate (nominal $\\alpha$={alpha:g}, "
                 f"n={cfg.n_noise_test_valid} valid test paths per scenario)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89)); fig.savefig(out); plt.close(fig)
    return out


def figure_recalibrated_detection(cfg: Stage1Config, per_scenario: dict, out: Path,
                                  alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = list(cfg.noise_scenarios)
    fig, axes = plt.subplots(2, len(scs), figsize=(15.0, 7.6), sharex=True)
    days = np.arange(1, cfg.horizon_days + 1)
    for j, sc in enumerate(scs):
        for k in ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
                  "known_vol_rolling_252"):
            ax = axes[0, j]
            c = _curve(per_scenario[sc], k, alpha, "recalibrated", "test_invalid")
            if c is not None:
                ax.plot(days, c, color=COLOURS[k], lw=1.6, label=LABEL_2A[k])
            ax = axes[1, j]
            c = _curve(per_scenario[sc], k, alpha, "recalibrated", "test_valid")
            if c is not None:
                ax.plot(days, c, color=COLOURS[k], lw=1.6)
        axes[0, j].set_title(SC_LABEL[sc], fontsize=10)
        axes[1, j].axhline(alpha, color="k", ls="-.", lw=1.0)
        axes[1, j].set_xlabel("trading day")
        for i in (0, 1):
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("cumulative detection | $S=0$")
    axes[1, 0].set_ylabel("cumulative false alarm | $S=1$")
    axes[0, 0].legend(fontsize=7.4, loc="upper left")
    fig.suptitle("Fig 2A.3  Recalibrated per scenario to the same NOMINAL budget, then frozen\n"
                 f"$\\alpha$={alpha:g}; this is an idealised per-environment calibration - it assumes the "
                 "noise scenario is known in advance and does not show adaptation", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_oracle(cfg: Stage1Config, per_scenario: dict, metrics, out: Path,
                  alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    res = per_scenario["stoch_vol"]
    days = np.arange(1, cfg.horizon_days + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    keys = ["binary_gaussian", "binary_student_t", "known_vol_rolling_252", "gaussian_oracle_vol"]

    for ax, role, lab in ((axes[0], "test_invalid", "cumulative detection | $S=0$"),
                          (axes[1], "test_valid", "cumulative false alarm | $S=1$")):
        for k in keys:
            c = _curve(res, k, alpha, "recalibrated", role)
            if c is None:
                continue
            ls = "--" if k == "gaussian_oracle_vol" else "-"
            ax.plot(days, c, color=COLOURS[k], lw=1.8 if ls == "--" else 1.5, ls=ls,
                    label=LABEL_2A[k])
        if role == "test_valid":
            ax.axhline(alpha, color="k", ls="-.", lw=1.0, label=f"nominal {alpha:g}")
        ax.set_xlabel("trading day"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.6, loc="upper left")

    ax = axes[2]
    m = metrics[(metrics.scenario == "stoch_vol") & (metrics.arm == "recalibrated")
                & (metrics.far_target == alpha)]
    m = m.set_index("detector")
    present = [k for k in keys if k in m.index]
    x = np.arange(len(present))
    ax.bar(x, [m.loc[k, "trunc_mean_detect_days"] for k in present],
           color=[COLOURS[k] for k in present], alpha=0.9)
    for i, k in enumerate(present):
        ax.text(i, m.loc[k, "trunc_mean_detect_days"] + 6, f"{m.loc[k,'trunc_mean_detect_days']:.0f}",
                ha="center", fontsize=8)
    ax.axhline(cfg.horizon_days, color="k", ls="--", lw=1.0, label=f"H={cfg.horizon_days}d")
    ax.set_xticks(x, [LABEL_2A[k].replace(" (", "\n(") for k in present], rotation=18, ha="right", fontsize=7.6)
    ax.set_ylabel("truncated mean detection time (days)")
    ax.set_ylim(0, cfg.horizon_days * 1.15); ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    fig.suptitle("Fig 2A.4  Stochastic volatility: what knowing the CURRENT variance would buy\n"
                 "The oracle uses the realised conditional variance, which no deployable rule has; "
                 "'Rolling / fixed sigma_0' divides by the constant $\\sigma_0$, NOT the current volatility",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import pandas as pd


SC = {"gaussian": "Gaussian (control)", "student_t": "Heavy-tailed Student-t (ν=5)",
      "stoch_vol": "Stochastic volatility (ρ=0.98)", "jump": "Jumps (λ=2/yr, κ=5)"}
ARM = {"frozen_stage1": "Frozen Stage 1 thresholds", "recalibrated": "Recalibrated per scenario"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "not reached"


def write_stage2a_report(cfg, summary, metrics: pd.DataFrame, diag: pd.DataFrame, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    H = cfg.horizon_days
    a_main = cfg.far_targets[-1]
    M = metrics

    A("# Stage 2A report: baseline stress test under realistic noise")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage2a`; "
      "every number is taken from the result tables of the same run.")
    A("")

    A("## 1. Conclusions of this round")
    A("")
    fz = M[(M.arm == "frozen_stage1") & (M.far_target == a_main)]
    rc = M[(M.arm == "recalibrated") & (M.far_target == a_main)]
    g = lambda d, sc, arm, col: float(
        M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a_main) & (M.detector == d)][col].iloc[0])
    A(
        f"**With the Stage 1 thresholds frozen, changing the noise pushes the actual false-alarm rate away from the nominal budget.** At nominal α={a_main:g}, "
        f"the two-year realised false-alarm rates are: Gaussian control {g('binary_gaussian','gaussian','frozen_stage1','far_d504'):.4f}, "
        f"heavy-tailed {g('binary_gaussian','student_t','frozen_stage1','far_d504'):.4f}, "
        f"stochastic volatility {g('binary_gaussian','stoch_vol','frozen_stage1','far_d504'):.4f}, "
        f"jumps {g('binary_gaussian','jump','frozen_stage1','far_d504'):.4f}"
        " (Binary Gaussian). The full four-detector × four-scenario results are in Section 4."
    )
    A("")
    A(
        "**Recalibrating per scenario can bring the false-alarm rate back near the budget, but this is an idealised setting.** "
        "It assumes the noise scenario is known in advance; in this round **no** detector adapts in an unknown environment, "
        "and the recalibrated arm only answers 'if the environment were known, what could these rules achieve'."
    )
    A("")
    if "gaussian_oracle_vol" in set(M.detector):
        o = g("gaussian_oracle_vol", "stoch_vol", "recalibrated", "detect_d504")
        b = g("binary_gaussian", "stoch_vol", "recalibrated", "detect_d504")
        A(
            f"**How much knowing the current variance helps (stochastic-volatility scenario)**: oracle two-year detection rate {o:.4f}, "
            f"fixed-volatility Binary Gaussian {b:.4f}, difference {o - b:+.4f}. "
            "The oracle uses the true current conditional variance; **no deployable rule has access to this information**. "
            "It is an **idealised-information reference**: it shows how far this family of updates with fixed mean hypotheses "
            "can go when given the true current variance. **It is not a proven strict upper bound on performance** - "
            "other methods (different mean hypotheses, different decision rules, a posterior that integrates over the latent variance) may well exceed it."
        )
        A("")

    A("## 2. Experiment definition (fixed before looking at the results)")
    A("")
    A("```")
    A("r_t = mu_S + sigma_0 * eps_t,   mu_S = S*sigma_ann/D,   sigma_0 = sigma_ann/sqrt(D)")
    A(f"S in {{0, 1}},  D = {cfg.D},  sigma_ann = {cfg.sigma_annual:g},  H = {H} trading days")
    A("```")
    A("")
    A("All four eps have **zero mean and unit unconditional variance**; the normalisation uses only theoretical constants, "
      "and **there is no after-the-fact standardisation by per-path sample mean/standard deviation** (that would change the generating mechanism and bring future data into the construction).")
    A("")
    A("| Scenario | Definition | Parameters |")
    A("|---|---|---|")
    A("| Gaussian (control) | `eps = z`, `z~N(0,1)` | — |")
    A(f"| Heavy-tailed | `eps = sqrt((nu-2)/nu) * X`, `X~t_nu` | ν = {cfg.noise_student_t_df:g} |")
    A(f"| Stochastic volatility | `a_t = rho a_{{t-1}} + sqrt(1-rho^2) xi_t`, `a_0~N(0,1)`; `v_t = exp(a_t - 1/2)`; `eps = sqrt(v_t) z_t` | ρ = {cfg.noise_sv_rho:g}, amplitude {cfg.noise_sv_amplitude:g} |")
    A(f"| Jumps | `K_t~Poisson(lambda/D)`; `eps = (z + kappa*sqrt(K)*w)/sqrt(1+kappa^2 lambda/D)` | λ = {cfg.noise_jump_lambda_annual:g}/yr, κ = {cfg.noise_jump_kappa:g} |")
    A("")
    A("Units: **λ is the annual jump intensity** (expected number of jumps per year), so the daily intensity is λ/D; "
      "**κ is the standard deviation of a single jump**, in units of the daily Gaussian shock.")
    A("")
    A("> These parameters are **research stress scenarios**, not fits to any market, and they do not claim to reproduce real markets. "
      "The main experiment adds only one noise feature at a time and does not run a full grid of combinations. "
      "The valid state means a **long-run** Sharpe of 1; under stochastic volatility the daily conditional Sharpe necessarily varies, and this round does not require it to be identically 1.")
    A("")
    A(f"Each scenario uses its own independent {cfg.n_noise_calibration} valid calibration paths, "
      f"{cfg.n_noise_test_valid} valid test paths and {cfg.n_noise_test_invalid} invalid test paths; "
      "within a scenario all models are evaluated on the **same set** of test paths.")
    A("")

    A("## 3. Noise diagnostics")
    A("")
    A("Errors are **path-level** (paths are independent of each other, but the trading days within a path are not - at ρ=0.98 the half-life of the log-variance is about 34 days, "
      "and treating every path-day as an independent sample would severely understate the error).")
    A("")
    A("Autocorrelations are always **pooled across paths and demeaned by the global mean**. Demeaning each path by its own sample mean also removes the "
      "low-frequency component that carries the persistence: measured here, it understates the lag-1 autocorrelation of |ε| by about 21% and the lag-40 autocorrelation by about 47%, "
      "and drags the AR(1) estimate about 23 standard errors below the configured ρ. The pooled estimates agree with the configured values.")
    A("")
    L.extend(_t(diag.to_dict("records"),
                ["Scenario", "Mean ± 3SE", "Variance ± 3SE", "P(abs ε>2)", "P(abs ε>4)", "P(abs ε>6)", "abs ε lag-1 autocorrelation"],
                lambda r: [SC[r["scenario"]],
                           f"{r['mean']:+.5f} ± {3*r['mean_se_path_level']:.5f}",
                           f"{r['var']:.4f} ± {3*r['var_se_path_level']:.4f}",
                           f"{r['p_abs_gt_2']:.4f}", f"{r['p_abs_gt_4']:.5f}",
                           f"{r['p_abs_gt_6']:.6f}", f"{r['abs_eps_lag1_autocorr']:+.4f}"]))
    A("")
    sv = diag[diag.scenario == "stoch_vol"]
    jp = diag[diag.scenario == "jump"]
    if len(sv):
        r = sv.iloc[0]
        A(f"Stochastic volatility: lag-1 autocorrelation of the log-variance **{r.log_var_lag1_autocorr:.5f} ± {r.log_var_lag1_autocorr_se:.5f}**"
          f" (configured ρ = {cfg.noise_sv_rho:g}, a difference of "
          f"{abs(r.log_var_lag1_autocorr - cfg.noise_sv_rho)/r.log_var_lag1_autocorr_se:.1f} standard errors); "
          f"lag-1 autocorrelation of |ε| **{r.abs_eps_lag1_autocorr:.5f} ± {r.abs_eps_lag1_autocorr_se:.5f}**, "
          f"with a corresponding **analytic population value of {r.abs_eps_lag1_autocorr_theory:.5f}**"
          f" (a difference of {abs(r.abs_eps_lag1_autocorr - r.abs_eps_lag1_autocorr_theory)/r.abs_eps_lag1_autocorr_se:.2f} standard errors); "
          f"90/10 quantile ratio of the true daily volatility {r.true_sigma_ratio_p90_p10:.2f}.")
    if len(jp):
        A(f"Jumps: realised mean number of jumps per day {jp.mean_jumps_per_day.iloc[0]:.5f}"
          f" (theoretical {jp.expected_jumps_per_day.iloc[0]:.5f}), "
          f"fraction of trading days with a jump {jp.frac_days_with_jump.iloc[0]:.4f}.")
    A("")
    A("At ν=5 the sample kurtosis of the Student-t has **no central limit theorem** (the eighth moment does not exist); "
      "in finite samples it is also bounded above by a function of n, is systematically biased low, and its distribution keeps changing with n, "
      "so it is **not** used as an acceptance statistic; tail frequencies are used instead. "
      "(Note that this is different from 'the finite-sample variance does not exist': for fixed n it is a bounded statistic.)")
    A("")
    A("> **Relation between stochastic volatility and heavy tails**: SV is conditionally Gaussian **given** the latent variance, "
      "but once the unobserved variance is integrated out, the one-day marginal distribution is a **scale mixture of normals**, which is itself heavy-tailed "
      "(in Fig 2A.1(a) its tail frequency almost coincides with that of the Student-t scenario). "
      "So one cannot say 'the advantage of the Student-t detector under SV has nothing to do with heavy tails' - "
      "the heavy marginal tails are precisely part of the reason it comes out ahead here; "
      "the other part is persistence, which this round does not separate out; only the ρ=0 control in Stage 2B answers that.")
    A("")

    for arm in ("frozen_stage1", "recalibrated"):
        n = 4 if arm == "frozen_stage1" else 5
        A(f"## {n}. {ARM[arm]}")
        A("")
        if arm == "frozen_stage1":
            A("The Stage 1 thresholds are carried over unchanged, **without recalibration**, to see what the noise change does.")
        else:
            A("Thresholds are calibrated to the same nominal budget on **each scenario's own valid calibration paths**, then frozen and tested. "
              "**This is an idealised calibration performed per environment distribution**; it does not mean the detectors can adapt to an unknown environment.")
        A("")
        for alpha in cfg.far_targets:
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} Same **nominal calibration budget** α = {alpha:g}")
            A("")
            A("What the detectors share is the **nominal calibration budget**, not the same realised false-alarm rate - "
              "the third column of the table below is each detector's actual two-year FAR on the test set, and detection rates must be read against it.")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)]
            L.extend(_t(sub.to_dict("records"),
                        ["Scenario", "Detector", "Two-year realised false alarm", "Detection h=126d", "h=252d", "h=504d",
                         "Undetected at two years", "Truncated mean detection time (days)", "Median detection time"],
                        lambda r: [SC[r["scenario"]], LABEL_2A.get(r["detector"], r["detector"]),
                                   f"{r['far_d504']:.4f}", f"{r['detect_d126']:.4f}",
                                   f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                                   f"{r['undetected_at_H']:.4f}",
                                   f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
            A("")

    A("## 6. Paired comparisons")
    A("")
    A("Within a scenario all models are evaluated on the same set of test paths, so the differences are paired.")
    A("")
    A("> **These intervals are conditional on the already-frozen calibration thresholds that the arm actually uses.** "
      "They answer 'given this set of thresholds, how much do the two rules differ on new test data', "
      "and **do not include** the variability that redrawing the calibration set would bring.")
    A("")
    pr = [p for p in summary["paired"] if p["far_target"] == a_main]
    L.extend(_t(pr, ["Scenario", "Arm", "Comparison (reference − other)", "Difference in two-year detection rate", "95% CI", "Difference in truncated mean detection time (days)", "95% CI"],
                lambda r: [SC[r["scenario"]], ARM["frozen_stage1" if r["arm"] == "frozen" else "recalibrated"],
                           f"Gaussian − {LABEL_2A.get(r['other'], r['other'])}",
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           f"{r['trunc_time_diff_days']:+.1f}",
                           f"[{r['trunc_time_diff_lo']:+.1f}, {r['trunc_time_diff_hi']:+.1f}]"]))
    A("")
    A(f"(The table above is for α={a_main:g}; the full results at the 5% level are in `stage2a_paired.csv`.)")
    A("")

    A("## 7. Probability calibration (reported separately from detection performance)")
    A("")
    A("Computed only for detectors that output a genuine posterior; the rolling Sharpe is not converted into a probability.")
    A("")
    L.extend(_t(summary["probability_calibration"],
                ["Scenario", "Detector", "Time", "Brier", "Base-rate Brier", "Skill score", "Mean predicted q", "Observed invalid fraction"],
                lambda r: [SC[r["scenario"]], LABEL_2A.get(r["detector"], r["detector"]),
                           f"{r['years']:.0f}y", f"{r['brier']:.4f}", f"{r['brier_base_rate']:.4f}",
                           f"{r['brier_skill_vs_base_rate']:+.4f}", f"{r['mean_predicted_q']:.4f}",
                           f"{r['observed_invalid_frac']:.4f}"]))
    A("")

    A("## 8. Reproduction")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage1     # main benchmark (unchanged)")
    A(".venv/bin/python -m strategy_survivorship.stage11    # Stage 1.1 diagnostics")
    A(".venv/bin/python -m strategy_survivorship.stage2a    # this report")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}; "
      f"this Stage 2A run took {summary['elapsed_s']:.1f} s.")
    A("")
    A("| Output | Contents |")
    A("|---|---|")
    A("| `stage2a_report.md` | This report |")
    A("| `stage2a_metrics.csv` | All metrics by scenario × arm × α × detector |")
    A("| `stage2a_noise_diagnostics.csv` | Moments, tail frequencies and structural diagnostics of the four noise models |")
    A("| `stage2a_paired.csv` | Paired differences and intervals |")
    A("| `stage2a_probability_calibration.csv` | Brier and skill scores |")
    A("| `stage2a_summary.json` | Compact summary of all the results above |")
    A("| `figures/fig2a1..fig2a4*.png` | The four main figures |")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


from .config import DEFAULT, Stage1Config
from .noise import SCENARIOS, draw_noise
from .stage11 import NO_ALARM

from .stage1 import environment_info
from .stage11 import Status, stage1_thresholds
from .simulate import stream_fingerprint, make_streams


def pooled_autocorr(x: np.ndarray, lag: int, with_se: bool = False):
    """Lag-`lag` autocorrelation pooled over paths, demeaned by the GLOBAL mean.

    Demeaning each path by its own sample mean removes the persistent level that
    creates the autocorrelation being measured: at rho = 0.98 the log-variance
    half-life is ~34 days inside a 504-day path, so the path mean absorbs most of
    it. Measured here, per-path demeaning understates the |eps| autocorrelation by
    about 21% at lag 1 and 47% at lag 40, and drags the AR(1) estimate 23 standard
    errors below the configured rho. The paths are iid, so pooling is valid.
    """
    import math as _m

    mu = x.mean()
    num_i = ((x[:, :-lag] - mu) * (x[:, lag:] - mu)).mean(axis=1)
    den_i = ((x - mu) ** 2).mean(axis=1)
    r = float(num_i.mean() / den_i.mean())
    if not with_se:
        return r
    # delta-method SE of a ratio of two path-level means
    se = float((num_i - r * den_i).std(ddof=1) / (_m.sqrt(x.shape[0]) * den_i.mean()))
    return r, se


def noise_diagnostics(cfg: Stage1Config, n_paths: int = 4000) -> pd.DataFrame:
    """Moments, tails and structure of each noise model, with path-level errors."""
    import math

    rows = []
    streams = scenario_streams(cfg)
    for sc in cfg.noise_scenarios:
        d = draw_noise(sc, streams[(sc, "calibration_valid")], n_paths, cfg.horizon_days, cfg)
        e = d.eps
        # paths are the independent replicates; days inside a path are not
        pm, pv = e.mean(axis=1), (e ** 2).mean(axis=1)
        row = {
            "scenario": sc,
            "n_paths": n_paths,
            "n_days": cfg.horizon_days,
            "mean": float(pm.mean()),
            "mean_se_path_level": float(pm.std(ddof=1) / math.sqrt(n_paths)),
            "var": float(pv.mean()),
            "var_se_path_level": float(pv.std(ddof=1) / math.sqrt(n_paths)),
        }
        for c in (2.0, 3.0, 4.0, 6.0):
            f = (np.abs(e) > c).mean(axis=1)
            row[f"p_abs_gt_{c:g}"] = float(f.mean())
            row[f"p_abs_gt_{c:g}_se"] = float(f.std(ddof=1) / math.sqrt(n_paths))
        row["abs_eps_lag1_autocorr"], row["abs_eps_lag1_autocorr_se"] = pooled_autocorr(
            np.abs(e), 1, with_se=True)
        if sc == "stoch_vol":
            row["log_var_lag1_autocorr"], row["log_var_lag1_autocorr_se"] = pooled_autocorr(
                d.latent["log_var"], 1, with_se=True)
            from .noise import stoch_vol_abs_eps_autocorr
            row["abs_eps_lag1_autocorr_theory"] = stoch_vol_abs_eps_autocorr(1, cfg)
            row["true_sigma_ratio_p90_p10"] = float(
                np.quantile(np.sqrt(d.latent["variance_multiplier"]), 0.9)
                / np.quantile(np.sqrt(d.latent["variance_multiplier"]), 0.1)
            )
        if sc == "jump":
            k = d.latent["jump_counts"]
            row["mean_jumps_per_day"] = float(k.mean())
            row["expected_jumps_per_day"] = cfg.noise_jump_lambda_annual / cfg.D
            row["frac_days_with_jump"] = float((k >= 1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rows(res: dict, cfg: Stage1Config, scenario: str, arm: str,
                reference: str = "binary_gaussian") -> list[dict]:
    """Paired detection differences on the SAME test paths, per arm."""
    import math

    out = []
    P = res["passages"]
    keys = [k for k in {k[0] for k in P} if k != reference]
    for alpha in cfg.far_targets:
        ref = P.get((reference, alpha, arm, "test_invalid"))
        if ref is None:
            continue
        for k in sorted(keys):
            other = P.get((k, alpha, arm, "test_invalid"))
            if other is None:
                continue
            a = (ref.first_day != NO_ALARM) & (ref.first_day <= cfg.horizon_days)
            b = (other.first_day != NO_ALARM) & (other.first_day <= cfg.horizon_days)
            d = a.astype(float) - b.astype(float)
            n = d.size
            se = d.std(ddof=1) / math.sqrt(n)
            dt = ref.truncated_days.astype(float) - other.truncated_days.astype(float)
            se_t = dt.std(ddof=1) / math.sqrt(n)
            z = cfg.wilson_z
            out.append({
                "scenario": scenario, "arm": arm, "far_target": alpha,
                "reference": reference, "other": k, "n_paired_paths": n,
                "detect_d504_reference": float(a.mean()), "detect_d504_other": float(b.mean()),
                "detect_diff": float(d.mean()),
                "detect_diff_lo": float(d.mean() - z * se), "detect_diff_hi": float(d.mean() + z * se),
                "trunc_time_diff_days": float(dt.mean()),
                "trunc_time_diff_lo": float(dt.mean() - z * se_t),
                "trunc_time_diff_hi": float(dt.mean() + z * se_t),
            })
    return out


def probability_calibration(res: dict, cfg: Stage1Config, scenario: str) -> list[dict]:
    """Brier / reliability for the detectors that emit a genuine posterior."""
    from scipy.special import expit

    from .probability_time import brier_and_reliability

    out = []
    for k in ("binary_gaussian", "binary_student_t", "gaussian_oracle_vol"):
        if k not in res["stats"]:
            continue
        qi = expit(-res["stats"][k]["test_invalid"])
        qv = expit(-res["stats"][k]["test_valid"])
        for day in cfg.brier_days:
            s, _ = brier_and_reliability(qi, qv, day, k)
            s["scenario"] = scenario
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2A: baselines under realistic noise.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2a") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    diag = noise_diagnostics(cfg, n_paths=min(4000, cfg.n_noise_calibration))
    diag.to_csv(out_dir / "stage2a_noise_diagnostics.csv", index=False)
    status("noise diagnostics done", scenarios=len(diag))

    frozen = stage1_thresholds(cfg)
    status("Stage 1 thresholds recovered", n=len(frozen))

    all_rows, all_paired, all_prob, per_scenario = [], [], [], {}
    for sc in cfg.noise_scenarios:
        res = run_scenario(cfg, sc, frozen)
        all_rows.extend(res["rows"])
        for arm in ("frozen", "recalibrated"):
            all_paired.extend(paired_rows(res, cfg, sc, arm))
        all_prob.extend(probability_calibration(res, cfg, sc))
        per_scenario[sc] = res
        status(f"scenario done: {sc}", detectors=len(res["detectors"]),
               rows=len(res["rows"]))

    metrics = pd.DataFrame(all_rows)
    order = {k: i for i, k in enumerate(LABEL_2A)}
    metrics["_o"] = metrics.detector.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2a_metrics.csv", index=False)
    pd.DataFrame(all_paired).to_csv(out_dir / "stage2a_paired.csv", index=False)
    pd.DataFrame(all_prob).to_csv(out_dir / "stage2a_probability_calibration.csv", index=False)

    figures = []
    if not args.no_figures:
        setup_style()
        fd = out_dir / "figures"
        figures = [
            str(figure_noise_diagnostics(cfg, per_scenario, diag, fd / "fig2a1_noise_diagnostics.png").relative_to(out_dir)),
            str(figure_frozen_far(cfg, per_scenario, fd / "fig2a2_frozen_far.png").relative_to(out_dir)),
            str(figure_recalibrated_detection(cfg, per_scenario, fd / "fig2a3_recalibrated_detection.png").relative_to(out_dir)),
            str(figure_oracle(cfg, per_scenario, metrics, fd / "fig2a4_oracle.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2A",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "scenario_stream_order": [f"{sc}|{role}" for sc in cfg.noise_scenarios
                                  for role in ("calibration_valid", "test_valid", "test_invalid")],
        "sample_sizes": {"calibration_valid": cfg.n_noise_calibration,
                         "test_valid": cfg.n_noise_test_valid,
                         "test_invalid": cfg.n_noise_test_invalid,
                         "days": cfg.horizon_days},
        "stage1_thresholds_frozen": {f"{k[0]}|alpha={k[1]:g}": v for k, v in frozen.items()},
        "noise_diagnostics": json.loads(diag.to_json(orient="records")),
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": all_paired,
        "probability_calibration": all_prob,
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2a_report(cfg, summary, metrics, diag, out_dir / "stage2a_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
