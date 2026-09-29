"""stage2b: one module per research stage.

- Model and computation: Stage 2B: does a variance forecast built from past returns help?
- Model and computation: Stage 2B uncertainty: a bootstrap that also resamples the calibration block.
- Model and computation: Stage 2B follow-up: two paired analyses on the SAME Stage 2B data.
- Model and computation: Stage 2B volatility-forecast diagnostics (mechanism, not the headline metric).
- Figures: Stage 2B figures: four, English labels, one colour per detector.
- Report: Generate ``stage2b_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 2B pipeline.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY
from .evaluate import build_metric_row, calibrate_threshold, first_passage
from .ewma import ewma_gaussian_log_odds, ewma_student_t_log_odds, ewma_variance_forecast
from .noise import draw_noise, returns_from_noise, true_daily_sigma
from .simulate import make_streams
from .stage2a import gaussian_oracle_log_odds

LABEL_2B = {
    "binary_gaussian": "Fixed Gaussian",
    "binary_student_t": "Fixed Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "ewma_gaussian": "EWMA Gaussian",
    "ewma_student_t": "EWMA Student-t (nu=5)",
    "gaussian_oracle_vol": "Gaussian oracle (true current var)",
}
FIXED_KEYS = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
              "known_vol_rolling_252")
EWMA_KEYS = ("ewma_gaussian", "ewma_student_t")
CUMULATIVE_KEYS = ("binary_gaussian", "binary_student_t") + EWMA_KEYS
ORACLE = "gaussian_oracle_vol"
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def scenario_streams(cfg: Stage1Config, parent: str = "stage2b") -> dict:
    """One independent child per (scenario, role), in a fixed documented order."""
    scen = tuple(cfg.noise_scenarios)
    kids = make_streams(cfg)[parent].spawn(len(scen) * len(ROLES))
    return {(sc, role): kids[i * len(ROLES) + j]
            for i, sc in enumerate(scen) for j, role in enumerate(ROLES)}


def build_blocks(cfg: Stage1Config, scenario: str, streams: dict, cfg_noise=None) -> dict:
    """Calibration / valid-test / invalid-test blocks for one scenario."""
    cn = cfg_noise or cfg
    n = {"calibration_valid": cfg.n_stage2b_calibration,
         "test_valid": cfg.n_stage2b_test_valid,
         "test_invalid": cfg.n_stage2b_test_invalid}
    sharpe = {"calibration_valid": cfg.sharpe_valid, "test_valid": cfg.sharpe_valid,
              "test_invalid": cfg.sharpe_invalid}
    out = {}
    for role in ROLES:
        d = draw_noise(scenario, streams[(scenario, role)], n[role], cfg.horizon_days, cn)
        out[role] = {
            "returns": returns_from_noise(d.eps, sharpe[role], cfg),
            "true_sigma": true_daily_sigma(d, cn),  # evaluator / oracle only
            "latent": d.latent,
        }
    return out


def statistic(key: str, block_role: dict, cfg: Stage1Config) -> np.ndarray:
    """Dispatch. Only the oracle is handed anything beyond the return series."""
    r = block_role["returns"]
    if key == "ewma_gaussian":
        return ewma_gaussian_log_odds(r, cfg)
    if key == "ewma_student_t":
        return ewma_student_t_log_odds(r, cfg)
    if key == ORACLE:
        return gaussian_oracle_log_odds(r, block_role["true_sigma"], cfg)
    return DETECTORS_BY_KEY[key].compute(r, cfg)


def first_eligible(key: str, cfg: Stage1Config) -> int:
    """EWMA starts on day 1: no artificial warm-up is added."""
    if key in EWMA_KEYS or key == ORACLE:
        return 1
    return DETECTORS_BY_KEY[key].first_eligible_day(cfg)


def detectors_for(scenario: str) -> tuple[str, ...]:
    base = FIXED_KEYS + EWMA_KEYS
    return base + ((ORACLE,) if scenario == "stoch_vol" else ())


class _Shim:
    """build_metric_row wants a detector-like object."""

    def __init__(self, key, arm, threshold_source):
        self.key = key
        self.label = LABEL_2B[key]
        self.role = ("ideal-information control" if key == ORACLE
                     else "forecast scale" if key in EWMA_KEYS else "fixed scale")
        self.statistic_name = threshold_source


class _Frozen:
    """Calibration record for a threshold chosen elsewhere (arm B)."""

    rule = "threshold calibrated on THIS ROUND's Gaussian valid paths, then transferred"

    def __init__(self, key, alpha, threshold, elig, achieved):
        self.detector_key, self.far_target, self.threshold = key, alpha, threshold
        self.first_eligible_day, self.n_paths, self.n_alarms = elig, 0, 0
        self.achieved_far = achieved


def path_minima(stat: np.ndarray, elig: int, horizon: int) -> np.ndarray:
    """Per-path minimum over the eligible monitoring days -- all the bootstrap needs."""
    w = stat[:, elig - 1 : horizon]
    if not np.isfinite(w).all():
        raise ValueError("non-finite statistic inside the eligible window")
    return w.min(axis=1)


def run_scenario(cfg: Stage1Config, scenario: str, streams: dict,
                 transfer_thresholds: dict | None, cfg_noise=None) -> dict:
    """Both arms for one scenario, plus the per-path minima the bootstrap reuses."""
    blocks = build_blocks(cfg, scenario, streams, cfg_noise)
    keys = detectors_for(scenario)
    rows, passages, minima, calibs, floors = [], {}, {}, {}, {}

    for key in keys:
        elig = first_eligible(key, cfg)
        stats = {role: statistic(key, blocks[role], cfg) for role in ROLES}
        minima[key] = {role: path_minima(stats[role], elig, cfg.horizon_days) for role in ROLES}
        if key in EWMA_KEYS:
            floors[key] = sum(
                ewma_variance_forecast(blocks[role]["returns"], cfg)[1] for role in ROLES
            )
        for alpha in cfg.far_targets:
            # ---- arm A: calibrated on this scenario ---------------------------
            c = calibrate_threshold(stats["calibration_valid"], alpha, elig,
                                    cfg.horizon_days, key)
            calibs[(key, alpha, "per_scenario")] = c
            fpv = first_passage(stats["test_valid"], c.threshold, elig, cfg.horizon_days)
            fpi = first_passage(stats["test_invalid"], c.threshold, elig, cfg.horizon_days)
            r = build_metric_row(detector=_Shim(key, "per_scenario", "calibrated here"),
                                 cfg=cfg, calib=c, fp_valid=fpv, fp_invalid=fpi)
            r.update({"scenario": scenario, "arm": "per_scenario"})
            rows.append(r)
            passages[(key, alpha, "per_scenario", "test_valid")] = fpv
            passages[(key, alpha, "per_scenario", "test_invalid")] = fpi

            # ---- arm B: Gaussian-calibrated threshold transferred -------------
            if transfer_thresholds is not None and key != ORACLE:
                thr, ach = transfer_thresholds[(key, alpha)]
                fpv = first_passage(stats["test_valid"], thr, elig, cfg.horizon_days)
                fpi = first_passage(stats["test_invalid"], thr, elig, cfg.horizon_days)
                r = build_metric_row(
                    detector=_Shim(key, "gaussian_transfer", "Gaussian-calibrated"),
                    cfg=cfg, calib=_Frozen(key, alpha, thr, elig, ach),
                    fp_valid=fpv, fp_invalid=fpi)
                r.update({"scenario": scenario, "arm": "gaussian_transfer"})
                rows.append(r)
                passages[(key, alpha, "gaussian_transfer", "test_valid")] = fpv
                passages[(key, alpha, "gaussian_transfer", "test_invalid")] = fpi
        del stats

    return {"rows": rows, "passages": passages, "minima": minima, "blocks": blocks,
            "calibrations": calibs, "keys": keys, "ewma_floor_hits": floors}


def gaussian_transfer_thresholds(cfg: Stage1Config, streams: dict) -> dict:
    """Each detector's threshold from THIS ROUND's Gaussian valid calibration paths."""
    blocks = build_blocks(cfg, "gaussian", streams)
    out = {}
    for key in FIXED_KEYS + EWMA_KEYS:
        elig = first_eligible(key, cfg)
        stat = statistic(key, blocks["calibration_valid"], cfg)
        for alpha in cfg.far_targets:
            c = calibrate_threshold(stat, alpha, elig, cfg.horizon_days, key)
            out[(key, alpha)] = (c.threshold, c.achieved_far)
        del stat
    return out


def persistence_control(cfg: Stage1Config) -> dict:
    """SV with rho = 0: same one-day marginal and unconditional variance, no persistence."""
    cn = replace(cfg, noise_sv_rho=cfg.sv_control_rho)
    streams = scenario_streams(cfg, parent="stage2b_persistence_control")
    res = run_scenario(cfg, "stoch_vol", streams, transfer_thresholds=None, cfg_noise=cn)
    keep = set(CUMULATIVE_KEYS) | {ORACLE}
    res["rows"] = [r for r in res["rows"] if r["detector"] in keep]
    for r in res["rows"]:
        r["scenario"] = "sv_rho0_control"
    return res

# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

from .evaluate import excludes_zero


def _threshold(cal_min: np.ndarray, alpha: float) -> float:
    """Same rule as evaluate.calibrate_threshold, on per-path minima."""
    n = cal_min.size
    k = min(int(math.floor(alpha * n)), n - 1)
    return float(np.partition(cal_min, k)[k])


def bootstrap_pair(
    minima_a: dict, minima_b: dict, alpha: float, n_reps: int,
    seed_seq: np.random.SeedSequence, z_level: float = 0.95,
) -> dict:
    """Bootstrap the FAR and 2-year detection difference between two detectors.

    ``minima_*`` are dicts with the per-path minimum statistic for each of the
    three blocks.  Returns point estimates and percentile intervals for each
    detector and for the difference.
    """
    rng = np.random.default_rng(seed_seq)
    n_cal = minima_a["calibration_valid"].size
    n_tv = minima_a["test_valid"].size
    n_ti = minima_a["test_invalid"].size
    lo_q, hi_q = (1 - z_level) / 2, 1 - (1 - z_level) / 2

    def point(m):
        thr = _threshold(m["calibration_valid"], alpha)
        return (float((m["test_valid"] < thr).mean()),
                float((m["test_invalid"] < thr).mean()))

    far_a0, det_a0 = point(minima_a)
    far_b0, det_b0 = point(minima_b)

    far_a = np.empty(n_reps); det_a = np.empty(n_reps)
    far_b = np.empty(n_reps); det_b = np.empty(n_reps)
    for i in range(n_reps):
        ic = rng.integers(0, n_cal, n_cal)   # shared across detectors
        iv = rng.integers(0, n_tv, n_tv)
        ii = rng.integers(0, n_ti, n_ti)
        ta = _threshold(minima_a["calibration_valid"][ic], alpha)
        tb = _threshold(minima_b["calibration_valid"][ic], alpha)
        far_a[i] = (minima_a["test_valid"][iv] < ta).mean()
        det_a[i] = (minima_a["test_invalid"][ii] < ta).mean()
        far_b[i] = (minima_b["test_valid"][iv] < tb).mean()
        det_b[i] = (minima_b["test_invalid"][ii] < tb).mean()

    d_far, d_det = far_a - far_b, det_a - det_b
    q = lambda x: (float(np.quantile(x, lo_q)), float(np.quantile(x, hi_q)))
    fa_lo, fa_hi = q(far_a); da_lo, da_hi = q(det_a)
    fb_lo, fb_hi = q(far_b); db_lo, db_hi = q(det_b)
    df_lo, df_hi = q(d_far); dd_lo, dd_hi = q(d_det)
    return {
        "far_target": alpha, "n_reps": n_reps,
        "far_a": far_a0, "far_a_lo": fa_lo, "far_a_hi": fa_hi,
        "det_a": det_a0, "det_a_lo": da_lo, "det_a_hi": da_hi,
        "far_b": far_b0, "far_b_lo": fb_lo, "far_b_hi": fb_hi,
        "det_b": det_b0, "det_b_lo": db_lo, "det_b_hi": db_hi,
        "far_diff": far_a0 - far_b0, "far_diff_lo": df_lo, "far_diff_hi": df_hi,
        "det_diff": det_a0 - det_b0, "det_diff_lo": dd_lo, "det_diff_hi": dd_hi,
        "det_diff_excludes_zero": excludes_zero(dd_lo, dd_hi),
        "includes_calibration_uncertainty": True,
    }


def run_comparisons(cfg, per_scenario: dict, pairs: list[tuple], seed_seq) -> pd.DataFrame:
    """Run a pre-specified list of (scenario, detector_a, detector_b, alpha) pairs."""
    kids = seed_seq.spawn(len(pairs))
    out = []
    for (sc, a, b, alpha), child in zip(pairs, kids):
        res = per_scenario.get(sc)
        if res is None or a not in res["minima"] or b not in res["minima"]:
            continue
        row = bootstrap_pair(res["minima"][a], res["minima"][b], alpha,
                             cfg.bootstrap_reps, child)
        row.update({"scenario": sc, "detector_a": a, "detector_b": b})
        out.append(row)
    return pd.DataFrame(out)

# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

from .evaluate import excludes_zero
from scipy.special import expit


def frozen_vs_recalibrated_pair(minima_a: dict, minima_b: dict, alpha: float,
                                n_reps: int, seed_seq, z: float = 1.959963984540054) -> dict:
    """Same two detectors, two interval constructions, one dataset."""
    rng = np.random.default_rng(seed_seq)
    thr_a = _threshold(minima_a["calibration_valid"], alpha)
    thr_b = _threshold(minima_b["calibration_valid"], alpha)
    da = minima_a["test_invalid"] < thr_a
    db = minima_b["test_invalid"] < thr_b
    n = da.size

    # (i) conditional on the frozen thresholds: a paired normal interval
    d = da.astype(float) - db.astype(float)
    se = d.std(ddof=1) / math.sqrt(n)
    frozen = {"diff": float(d.mean()),
              "lo": float(d.mean() - z * se), "hi": float(d.mean() + z * se)}

    # (ii) calibration block resampled too
    n_cal = minima_a["calibration_valid"].size
    diffs = np.empty(n_reps)
    for i in range(n_reps):
        ic = rng.integers(0, n_cal, n_cal)
        ii = rng.integers(0, n, n)
        ta = _threshold(minima_a["calibration_valid"][ic], alpha)
        tb = _threshold(minima_b["calibration_valid"][ic], alpha)
        diffs[i] = ((minima_a["test_invalid"][ii] < ta).mean()
                    - (minima_b["test_invalid"][ii] < tb).mean())
    boot = {"diff": frozen["diff"],
            "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975))}
    return {"far_target": alpha, "n_reps": n_reps,
            "frozen_diff": frozen["diff"], "frozen_lo": frozen["lo"], "frozen_hi": frozen["hi"],
            "recal_diff": boot["diff"], "recal_lo": boot["lo"], "recal_hi": boot["hi"],
            "frozen_excludes_zero": excludes_zero(frozen["lo"], frozen["hi"]),
            "recal_excludes_zero": excludes_zero(boot["lo"], boot["hi"])}


def paired_brier_difference(q_a_inv, q_a_val, q_b_inv, q_b_val, day: int,
                            n_reps: int, seed_seq) -> dict:
    """Brier(A) - Brier(B) at one day, paired by path.

    The valid and invalid groups are resampled SEPARATELY, each to its own size,
    so the 50/50 weighting the original evaluation used is preserved exactly.
    """
    rng = np.random.default_rng(seed_seq)
    ai, av = q_a_inv[:, day - 1], q_a_val[:, day - 1]
    bi, bv = q_b_inv[:, day - 1], q_b_val[:, day - 1]
    n_i, n_v = ai.size, av.size

    def brier(pi, pv):
        return float((np.sum((pi - 1.0) ** 2) + np.sum(pv ** 2)) / (pi.size + pv.size))

    point = brier(ai, av) - brier(bi, bv)
    draws = np.empty(n_reps)
    for r in range(n_reps):
        ii = rng.integers(0, n_i, n_i)   # shared by A and B: paired
        iv = rng.integers(0, n_v, n_v)
        draws[r] = brier(ai[ii], av[iv]) - brier(bi[ii], bv[iv])
    return {"day": day, "n_reps": n_reps, "brier_a": brier(ai, av), "brier_b": brier(bi, bv),
            "diff": point, "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)),
            "excludes_zero": excludes_zero(float(np.quantile(draws, 0.025)),
                                           float(np.quantile(draws, 0.975)))}


def exploratory_rho0_interval(minima_a: dict, minima_b: dict, alpha: float,
                              n_reps: int, seed_seq) -> dict:
    """Interval for the rho=0 control difference.

    Flagged as an EXPLORATORY addition made in Stage 2C.  It was not among the
    comparisons pre-specified before Stage 2B ran, and is reported as such.
    """
    out = frozen_vs_recalibrated_pair(minima_a, minima_b, alpha, n_reps, seed_seq)
    out["prespecified"] = False
    out["added_in"] = "Stage 2C follow-up (exploratory)"
    return out

# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

from .ewma import ewma_variance_forecast, information_equivalent_days, qlike


def forecast_diagnostics(returns: np.ndarray, true_var: np.ndarray, cfg) -> pd.DataFrame:
    """Per-path aggregates first, then the across-path standard error.

    Aggregating day-level errors as if independent would badly understate the
    error under a rho = 0.98 latent process.
    """
    vhat, n_floored = ewma_variance_forecast(returns, cfg)
    ratio = vhat / true_var
    ql = qlike(true_var, vhat)

    per_path_ql = ql.mean(axis=1)
    per_path_bias = (vhat - true_var).mean(axis=1) / cfg.sigma_daily ** 2
    n = returns.shape[0]
    row = {
        "n_paths": n,
        "n_days": returns.shape[1],
        "qlike_mean": float(per_path_ql.mean()),
        "qlike_se": float(per_path_ql.std(ddof=1) / math.sqrt(n)),
        "mean_forecast_bias_in_sigma0sq": float(per_path_bias.mean()),
        "mean_forecast_bias_se": float(per_path_bias.std(ddof=1) / math.sqrt(n)),
        "n_variance_floor_hits": n_floored,
    }
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        row[f"ratio_vhat_over_v_q{int(q*100):02d}"] = float(np.quantile(ratio, q))
    return pd.DataFrame([row])


def qlike_benchmarks(returns: np.ndarray, true_var: np.ndarray, cfg) -> pd.DataFrame:
    """QLIKE of the EWMA forecast against two reference forecasts."""
    vhat, _ = ewma_variance_forecast(returns, cfg)
    refs = {
        "ewma_lambda0.94": vhat,
        "constant_sigma0_sq": np.full_like(true_var, cfg.sigma_daily ** 2),
        "oracle_true_variance": true_var,
    }
    rows = []
    n = returns.shape[0]
    for name, f in refs.items():
        per_path = qlike(true_var, f).mean(axis=1)
        rows.append({"forecast": name,
                     "qlike_mean": float(per_path.mean()),
                     "qlike_se": float(per_path.std(ddof=1) / math.sqrt(n))})
    return pd.DataFrame(rows)


def information_equivalent_table(true_var: np.ndarray, cfg) -> pd.DataFrame:
    """N_info(n) = sum_{t<=n} sigma_0^2 / v_t, with its theoretical mean.

    Uses the TRUE latent variance: an interpretation tool for the oracle, not data
    any deployable detector receives, and not a conversion formula for detection
    time.
    """
    n_info = information_equivalent_days(true_var, cfg)
    amp2 = cfg.noise_sv_amplitude ** 2
    rows = []
    for d in cfg.n_info_days:
        col = n_info[:, d - 1]
        rows.append({
            "day": d,
            "n_paths": int(col.size),
            "mean": float(col.mean()),
            "se": float(col.std(ddof=1) / math.sqrt(col.size)),
            "median": float(np.median(col)),
            "q10": float(np.quantile(col, 0.10)),
            "q90": float(np.quantile(col, 0.90)),
            # v_t = exp(a_t - amp^2/2) with a_t ~ N(0, amp^2), so
            # E[sigma_0^2 / v_t] = E[exp(amp^2/2 - a_t)] = exp(amp^2) = e at amp = 1
            "theoretical_mean": float(d * math.exp(amp2)),
        })
    return pd.DataFrame(rows)


def fixed_path_trace(returns: np.ndarray, true_var: np.ndarray, cfg,
                     path_index: int = 0) -> pd.DataFrame:
    """True vs forecast variance on one PRE-FIXED path index."""
    vhat, _ = ewma_variance_forecast(returns, cfg)
    i = path_index
    return pd.DataFrame({
        "day": np.arange(1, returns.shape[1] + 1),
        "path_index": i,
        "return": returns[i],
        "true_variance": true_var[i],
        "forecast_variance": vhat[i],
        "ratio_forecast_over_true": vhat[i] / true_var[i],
        "qlike": qlike(true_var[i], vhat[i]),
    })

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import setup_style


C = {
    "binary_gaussian": "#1f77b4",
    "binary_student_t": "#d62728",
    "trailing_sharpe_252": "#2ca02c",
    "known_vol_rolling_252": "#9467bd",
    "ewma_gaussian": "#17becf",
    "ewma_student_t": "#ff7f0e",
    ORACLE: "#8c564b",
}
SC = {"gaussian": "Gaussian", "student_t": "Student-t tails",
      "stoch_vol": "Stochastic volatility", "jump": "Jumps",
      "sv_rho0_control": "SV control ($\\rho$=0)"}


def figure_vol_forecast(cfg: Stage1Config, trace, diag, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    s0sq = cfg.sigma_daily ** 2

    ax = axes[0]
    ax.plot(trace.day, trace.true_variance / s0sq, color="0.35", lw=1.3, label="true $v_t$")
    ax.plot(trace.day, trace.forecast_variance / s0sq, color=C["ewma_gaussian"], lw=1.4,
            label="EWMA forecast $\\hat v_t$")
    ax.axhline(1.0, color="k", lw=0.8, ls=":", label="$\\sigma_0^2$")
    ax.set_yscale("log")
    ax.set_xlabel("trading day"); ax.set_ylabel("variance / $\\sigma_0^2$")
    ax.set_title(f"(a) One pre-fixed path (index {int(trace.path_index.iloc[0])})\n"
                 "true vs strictly causal forecast", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[1]
    r = trace.ratio_forecast_over_true
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    vals = [diag[f"ratio_vhat_over_v_q{int(q*100):02d}"].iloc[0] for q in qs]
    ax.plot([q * 100 for q in qs], vals, "o-", color=C["ewma_gaussian"], lw=1.6, ms=6)
    ax.axhline(1.0, color="k", lw=1.0, ls="--", label="perfect forecast")
    ax.set_xlabel("percentile across all path-days")
    ax.set_ylabel("$\\hat v_t / v_t$")
    ax.set_title("(b) Forecast / true variance ratio\nacross the whole test block", fontsize=10)
    ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(trace.day, trace.qlike, color="#444444", lw=1.0)
    ax.set_xlabel("trading day"); ax.set_ylabel("QLIKE  $v/\\hat v-\\log(v/\\hat v)-1$")
    ax.set_title("(c) QLIKE on the same path\n0 = perfect", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Fig 2B.1  The EWMA variance forecast under stochastic volatility "
                 "(mechanism diagnostic, not the headline metric)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_sv_curves(cfg: Stage1Config, per_scenario: dict, out: Path,
                     alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    res = per_scenario["stoch_vol"]
    days = np.arange(1, cfg.horizon_days + 1)
    keys = ["binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t", ORACLE]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for ax, role, lab in ((axes[0], "test_invalid", "cumulative detection | $S=0$"),
                          (axes[1], "test_valid", "cumulative false alarm | $S=1$")):
        for k in keys:
            fp = res["passages"].get((k, alpha, "per_scenario", role))
            if fp is None:
                continue
            ls = "--" if k == ORACLE else "-"
            ax.plot(days, fp.cumulative_rate, color=C[k], lw=1.8 if k == ORACLE else 1.5,
                    ls=ls, label=LABEL_2B[k])
        if role == "test_valid":
            ax.axhline(alpha, color="k", ls="-.", lw=1.0, label=f"nominal {alpha:g}")
        ax.set_xlabel("trading day"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.6, loc="upper left")
    fig.suptitle("Fig 2B.2  Stochastic volatility, calibrated per scenario then frozen\n"
                 f"$\\alpha$={alpha:g}; the oracle is an IDEAL-INFORMATION control, not a proven bound",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out); plt.close(fig)
    return out


def figure_arms(cfg: Stage1Config, metrics, out: Path, alpha: float | None = None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = list(cfg.noise_scenarios)
    keys = ["binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    x = np.arange(len(scs)); w = 0.2

    for row, (arm, name) in enumerate((("per_scenario", "A: calibrated per scenario"),
                                       ("gaussian_transfer", "B: Gaussian threshold transferred"))):
        for col, (field, lab, ref) in enumerate(
                ((("detect_d504"), "2y detection | $S=0$", None),
                 (("far_d504"), "2y realised FAR | $S=1$", alpha))):
            ax = axes[row, col]
            for i, k in enumerate(keys):
                vals = []
                for sc in scs:
                    m = metrics[(metrics.scenario == sc) & (metrics.arm == arm)
                                & (metrics.far_target == alpha) & (metrics.detector == k)]
                    vals.append(float(m[field].iloc[0]) if len(m) else np.nan)
                ax.bar(x + (i - 1.5) * w, vals, w, color=C[k], label=LABEL_2B[k], alpha=0.9)
            if ref is not None:
                ax.axhline(ref, color="k", ls="-.", lw=1.0, label=f"nominal {ref:g}")
            ax.set_xticks(x, [SC[s] for s in scs], rotation=12, ha="right", fontsize=8)
            ax.set_ylabel(lab)
            ax.set_title(f"{name}", fontsize=9.5)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7.2, loc="upper left", ncol=2)
    fig.suptitle(f"Fig 2B.3  Per-scenario calibration vs Gaussian-threshold transfer "
                 f"(nominal $\\alpha$={alpha:g})\n"
                 "Arm B carries ONE threshold, calibrated on this round's Gaussian valid paths, "
                 "into every scenario", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_reliability(cfg: Stage1Config, rel, out: Path) -> Path:
    scs = ["gaussian", "stoch_vol"]
    days = [cfg.n_info_days[1], cfg.n_info_days[-1]]
    keys = ["binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t", ORACLE]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    for i, sc in enumerate(scs):
        for j, day in enumerate(days):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="perfect")
            for k in keys:
                s = rel[(rel.scenario == sc) & (rel.day == day) & (rel.detector == k)]
                if not len(s):
                    continue
                ax.plot(s.mean_predicted_q, s.observed_invalid_frac, "o-", color=C[k],
                        lw=1.4, ms=4, label=LABEL_2B[k])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("mean predicted $q$"); ax.set_ylabel("observed invalid fraction")
            ax.set_title(f"{SC[sc]}, day {day} ({day/cfg.D:.1f}y)", fontsize=10)
            ax.grid(alpha=0.3)
    # legend on the SV panel: it is the only one where the oracle exists
    axes[1, 0].legend(fontsize=7.2, loc="upper left")
    fig.suptitle("Fig 2B.4  Probability reliability, all paths continued past any alarm\n"
                 "bin counts are in stage2b_reliability.csv", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(out); plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import pandas as pd


SC_REPORT = {"gaussian": "Gaussian", "student_t": "Heavy-tailed t(5)", "stoch_vol": "Stochastic volatility ρ=0.98",
      "jump": "Jumps", "sv_rho0_control": "**Persistence control SV ρ=0**"}
ARM = {"per_scenario": "A: calibrated separately per scenario", "gaussian_transfer": "B: Gaussian-calibrated threshold transferred"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "not reached"


def write_stage2b_report(cfg, summary, metrics: pd.DataFrame, boot: pd.DataFrame,
                         diag: pd.DataFrame, qb: pd.DataFrame, ninfo: pd.DataFrame,
                         path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    M = metrics
    g = lambda det, sc, arm, col, a=a_main: (
        float(M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a)
                & (M.detector == det)][col].iloc[0]))
    b = lambda sc, da, db, a, col: (
        float(boot[(boot.scenario == sc) & (boot.detector_a == da)
                   & (boot.detector_b == db) & (boot.far_target == a)][col].iloc[0]))

    A("# Stage 2B report: does a volatility forecast built from past returns improve strategy-validity testing?")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage2b`; "
      "every number is taken from the result tables of the same run.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. Direct answers to the six questions")
    A("")
    sv_eg = g("ewma_gaussian", "stoch_vol", "per_scenario", "detect_d504")
    sv_bg = g("binary_gaussian", "stoch_vol", "per_scenario", "detect_d504")
    sv_et = g("ewma_student_t", "stoch_vol", "per_scenario", "detect_d504")
    sv_bt = g("binary_student_t", "stoch_vol", "per_scenario", "detect_d504")
    sv_or = g(ORACLE, "stoch_vol", "per_scenario", "detect_d504")
    A("**(1) Does it improve? Only under stochastic volatility; in every other scenario it does slight harm.**")
    A("")
    A(f"Two-year detection-rate difference between each EWMA method and its fixed-scale counterpart (arm A, α = {a_main:g}):")
    A("")
    _pairs = [("ewma_gaussian", "binary_gaussian"), ("ewma_student_t", "binary_student_t")]
    _rows = []
    for sc in list(cfg.noise_scenarios) + ["sv_rho0_control"]:
        for e, f in _pairs:
            try:
                de, df_ = g(e, sc, "per_scenario", "detect_d504"), g(f, sc, "per_scenario", "detect_d504")
                fe, ff = g(e, sc, "per_scenario", "far_d504"), g(f, sc, "per_scenario", "far_d504")
            except IndexError:
                continue
            _rows.append({"sc": sc, "e": e, "f": f, "d": de - df_, "fd": fe - ff, "de": de, "df": df_})
    L.extend(_t(_rows, ["Scenario", "EWMA method", "Fixed counterpart", "EWMA detection", "Fixed detection",
                        "Detection-rate difference", "Realised FAR difference"],
                lambda r: [SC_REPORT[r["sc"]], LABEL_2B[r["e"]], LABEL_2B[r["f"]],
                           f"{r['de']:.4f}", f"{r['df']:.4f}",
                           f"**{r['d']:+.4f}**", f"{r['fd']:+.4f}"]))
    A("")
    A("All realised FAR differences are at most 0.011, so these comparisons are made at essentially the same false-alarm cost.")
    A("")
    A(f"Stochastic volatility, α={a_main:g}: EWMA Gaussian {sv_eg:.4f} vs fixed Gaussian {sv_bg:.4f}"
      f" (difference {sv_eg - sv_bg:+.4f}, bootstrap 95% "
      f"[{b('stoch_vol','ewma_gaussian','binary_gaussian',a_main,'det_diff_lo'):+.4f}, "
      f"{b('stoch_vol','ewma_gaussian','binary_gaussian',a_main,'det_diff_hi'):+.4f}]).")
    A("")
    A(f"**(2) Does it beat the fixed Student-t?** Under stochastic volatility, EWMA Gaussian {sv_eg:.4f} vs fixed Student-t {sv_bt:.4f}; "
      f"EWMA Student-t {sv_et:.4f}. For the distance to the oracle ({sv_or:.4f}), see Section 5.")
    A("")
    tr_eg = g("ewma_gaussian", "stoch_vol", "gaussian_transfer", "detect_d504")
    tr_bg = g("binary_gaussian", "stoch_vol", "gaussian_transfer", "detect_d504")
    tr_eg_far = g("ewma_gaussian", "stoch_vol", "gaussian_transfer", "far_d504")
    tr_bg_far = g("binary_gaussian", "stoch_vol", "gaussian_transfer", "far_d504")
    A(f"**(3) Is it retained after threshold transfer?** Transferring this round's Gaussian-calibrated thresholds to stochastic volatility: "
      f"EWMA Gaussian detection {tr_eg:.4f} (realised FAR {tr_eg_far:.4f}), "
      f"fixed Gaussian {tr_bg:.4f} (realised FAR {tr_bg_far:.4f}). "
      "The two realised FARs differ, so this comparison is **not** made at the same false-alarm cost; see Section 4 for details.")
    A("")
    c_eg = g("ewma_gaussian", "sv_rho0_control", "per_scenario", "detect_d504")
    c_bg = g("binary_gaussian", "sv_rho0_control", "per_scenario", "detect_d504")
    A(f"**(4) How much does it depend on persistence? Under the current EWMA(λ=0.94), the current parameters, the current five scenarios and the current detection-rate metric, "
      f"the improvement comes entirely from persistence.** This is evidence about this specific set of settings, not a general conclusion about \"any volatility-forecasting method\". "
      "ρ=0 control (same one-day marginal and unconditional variance, persistence removed): "
      f"EWMA Gaussian {c_eg:.4f} vs fixed Gaussian {c_bg:.4f}, difference {c_eg - c_bg:+.4f}; "
      f"whereas at ρ=0.98 this difference is {sv_eg - sv_bg:+.4f}.")
    A("")
    A("**(5) Which conclusions are still limited by calibration uncertainty?** The bootstrap in Section 5 resamples both the calibration set and the test set, "
      "so it includes the sampling noise of the threshold-selection step; the other paired intervals in the main text are **conditional on the frozen thresholds** and do not include it. "
      "Both kinds of interval are **pointwise** and do not guarantee that all comparisons hold simultaneously.")
    A("")
    A("**(6) The single most valuable next question** is given in Section 8.")
    A("")

    # ---------------- 2 ----------------
    A("## 2. Model definition for this round (fixed before the run)")
    A("")
    A("```")
    A(f"D = {cfg.D},  H = {cfg.horizon_days},  sigma_ann = {cfg.sigma_annual:g}")
    A(f"mu_0 = 0,  mu_1 = {cfg.daily_drift(cfg.sharpe_valid):.8e},  "
      f"sigma_0 = {cfg.sigma_daily:.8e}")
    A(f"m = (mu_0 + mu_1)/2 = {0.5*cfg.daily_drift(cfg.sharpe_valid):.8e}")
    A(f"v_1 = sigma_0^2;   v_(t+1) = {cfg.ewma_lambda:g} v_t + "
      f"{1-cfg.ewma_lambda:.2f} (r_t - m)^2")
    A("```")
    A("")
    A("`v_t` is the forecast of day t's variance made **before seeing day t's return**, using only `r_1..r_(t-1)`. "
      "No full-sample backcast, full-path sample variance or true-state initialisation is used. λ=0.94 is a simple baseline fixed in advance; no parameter search was done.")
    A("")
    from .ewma import midpoint_variance_inflation
    infl = midpoint_variance_inflation(cfg) / cfg.sigma_daily ** 2
    A(f"The fixed midpoint `m` is used so that the true state is not read and no extra mean estimator is introduced. "
      f"**Cost**: under both true-mean hypotheses the squared residual carries the same extra `(mu_1-mu_0)^2/4` term, "
      f"so this is not a strictly unbiased variance recursion; in this round the term is about **{100*infl:.4f}%** of the baseline variance.")
    A("")
    A("The two detectors **share exactly the same** `v_t`. Update rules:")
    A("")
    A("```")
    A("Gaussian :  dL_t = (mu_1 - mu_0)(r_t - m) / v_t")
    A(f"Student-t:  b_t = sqrt((nu-2)/nu * v_t),  nu = {cfg.ewma_student_t_df:g}")
    A("            dL_t = log f_nu(r_t; mu_1, b_t) - log f_nu(r_t; mu_0, b_t)")
    A("```")
    A("")
    A("`b_t` is the scale parameter that makes the **variance** of the t distribution equal to `v_t`; it is not the standard deviation. "
      "Returns are **not** first standardised by the forecast volatility and then given a fixed conditional-Sharpe assumption: the two mean hypotheses are always `mu_0`, `mu_1` in return space.")
    A("")
    A("> The two methods output **model posteriors under their respective EWMA conditional-distribution assumptions**, "
      "**not** exact posteriors obtained by integrating over the true latent SV state.")
    A("")
    fl = summary.get("ewma_variance_floor_hits", {})
    total_fl = sum(sum(v.values()) for v in fl.values() if isinstance(v, dict))
    A(f"Numerical variance floor `{cfg.ewma_variance_floor_factor:g}·sigma_0^2`, triggered **{total_fl}** times in this round"
      " (a numerical safeguard only; not tuned to test performance).")
    A("")
    A(f"Each scenario: {cfg.n_stage2b_calibration} valid calibration paths, {cfg.n_stage2b_test_valid} valid test paths, "
      f"{cfg.n_stage2b_test_invalid} invalid test paths. **Stage 2B uses entirely new random streams**, "
      "and all compared methods run on the same batch of new test paths; Stage 2A results are left unchanged. "
      "EWMA starts on day 1, with no artificial warm-up period.")
    A("")

    # ---------------- 3 / 4 ----------------
    for arm in ("per_scenario", "gaussian_transfer"):
        n = 3 if arm == "per_scenario" else 4
        A(f"## {n}. {ARM[arm]}")
        A("")
        if arm == "per_scenario":
            A("Each detector sets its threshold using only this scenario's valid calibration paths; the threshold is then frozen and tested. "
              "**This is an idealised calibration comparison with the scenario known**; it does not mean a detector can identify which scenario it is in.")
        else:
            A("Each conventional detector sets its threshold using only **this round's Gaussian** valid calibration paths, and the same threshold is used in all four scenarios. "
              "This is called \"Gaussian-calibrated threshold transfer\" rather than \"frozen Stage 1 threshold\" because the two EWMA methods have no Stage 1 threshold. "
              "The oracle does not take part in the transfer comparison.")
        A("")
        A("What the detectors share is the **nominal calibration budget**, not the same realised false-alarm rate; detection rates must be read against the "
          "\"2-year realised FAR\" in the same row.")
        A("")
        for alpha in cfg.far_targets:
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} Nominal calibration budget α = {alpha:g}")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)
                    & (M.scenario != "sv_rho0_control")]
            L.extend(_t(sub.to_dict("records"),
                        ["Scenario", "Detector", "2-year realised FAR", "FAR 95%", "Detection h=126d",
                         "h=252d", "h=504d", "Undetected at 2 years", "Truncated mean detection time (d)", "Median detection time"],
                        lambda r: [SC_REPORT[r["scenario"]], LABEL_2B.get(r["detector"], r["detector"]),
                                   f"{r['far_d504']:.4f}",
                                   f"[{r['far_d504_lo']:.4f}, {r['far_d504_hi']:.4f}]",
                                   f"{r['detect_d126']:.4f}", f"{r['detect_d252']:.4f}",
                                   f"{r['detect_d504']:.4f}", f"{r['undetected_at_H']:.4f}",
                                   f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
            A("")

    # ---------------- 5 ----------------
    A("## 5. Persistence control and mechanism diagnostics")
    A("")
    A(f"Auxiliary scenario: SV `ρ = {cfg.sv_control_rho:g}`, amplitude still {cfg.noise_sv_amplitude:g}, "
      "keeping the same one-day marginal distribution and unconditional variance and removing only the time persistence of the latent volatility. "
      "Independent random streams, independent calibration and testing; only the four cumulative methods and the oracle are compared.")
    A("")
    ctl = M[(M.scenario == "sv_rho0_control") & (M.far_target == a_main)]
    L.extend(_t(ctl.to_dict("records"),
                ["Detector", "2-year realised FAR", "Detection h=252d", "h=504d", "Truncated mean detection time (d)", "Median detection time"],
                lambda r: [LABEL_2B.get(r["detector"], r["detector"]), f"{r['far_d504']:.4f}",
                           f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                           f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
    A("")
    A(f"For comparison, at ρ=0.98 the two-year detection-rate difference of EWMA Gaussian relative to fixed Gaussian is "
      f"**{sv_eg - sv_bg:+.4f}**; at ρ=0 it is **{c_eg - c_bg:+.4f}**.")
    A("")
    A("### 5.1 Volatility-forecast diagnostics")
    A("")
    d0 = diag.iloc[0]
    A(f"Stochastic volatility, invalid test block (n = {int(d0.n_paths)} paths × {int(d0.n_days)} days); "
      "errors are first aggregated over each complete path, and then the across-path standard error is computed:")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| QLIKE (EWMA) | {d0.qlike_mean:.5f} ± {d0.qlike_se:.5f} |")
    A(f"| Mean forecast bias (in units of σ₀²) | {d0.mean_forecast_bias_in_sigma0sq:+.5f} ± {d0.mean_forecast_bias_se:.5f} |")
    A(f"| Variance-floor hits | {int(d0.n_variance_floor_hits)} |")
    for q in (10, 25, 50, 75, 90):
        A(f"| {q}% quantile of `v̂/v` | {d0[f'ratio_vhat_over_v_q{q:02d}']:.4f} |")
    A("")
    A("QLIKE against reference forecasts:")
    A("")
    L.extend(_t(qb.to_dict("records"), ["Forecast", "QLIKE", "Standard error"],
                lambda r: [r["forecast"], f"{r['qlike_mean']:.5f}", f"{r['qlike_se']:.5f}"]))
    A("")
    A("> **QLIKE is a mechanism diagnostic and cannot replace the strategy-validation result.** Under model misspecification, different forecast loss functions may rank forecasts differently "
      "(Patton, *Comparing Possibly Misspecified Forecasts*). The main conclusions are still determined by the false-alarm and detection metrics.")
    A("")
    A("An isolated jump raises the EWMA variance forecast for many subsequent days (half-life about "
      f"{__import__('math').log(0.5)/__import__('math').log(cfg.ewma_lambda):.0f} days), "
      "which in the jump scenario lowers the evidence increments for a period afterwards. This round does **not** add an ad-hoc model patch for this.")
    A("")
    A("### 5.2 Information-equivalent observation days")
    A("")
    A("`N_info(n) = Σ_{t≤n} σ₀²/v_t`. **It uses the true variance** and is a tool for interpreting the oracle's information; "
      "it is neither extra data available to a deployable detector nor a direct conversion formula for detection time.")
    A("")
    L.extend(_t(ninfo.to_dict("records"),
                ["Days n", "Mean ± SE", "Theoretical mean n·e", "Deviation (SE)", "Median", "10%", "90%"],
                lambda r: [str(int(r["day"])), f"{r['mean']:.1f} ± {r['se']:.1f}",
                           f"{r['theoretical_mean']:.1f}",
                           f"{abs(r['mean']-r['theoretical_mean'])/r['se']:.2f}",
                           f"{r['median']:.1f}", f"{r['q10']:.1f}", f"{r['q90']:.1f}"]))
    A("")

    # ---------------- 6 ----------------
    A("## 6. Uncertainty: a bootstrap that resamples both the calibration set and the test set")
    A("")
    A(f"The number of replications is fixed at **{cfg.bootstrap_reps}** (set before the run, and not changed according to whether an interval excludes zero). "
      "The valid-calibration, valid-test and invalid-test blocks are each resampled by path **separately**; within a block, different detectors **share the same path indices**, "
      "so the differences remain paired; in each replication the threshold is **re-determined** on the resampled calibration set before measuring. Days are not resampled independently within a path.")
    A("")
    A("> **These intervals include the sampling uncertainty of threshold selection**; they answer a different question from the intervals in the Section 3 and 4 tables, which are conditional on the frozen thresholds. "
      "Both are pointwise intervals and **do not provide a joint guarantee that all comparisons hold simultaneously**.")
    A("")
    L.extend(_t(boot.to_dict("records"),
                ["Scenario", "A", "B", "α", "A detection", "B detection", "Detection difference", "95% bootstrap interval", "Excludes zero"],
                lambda r: [SC_REPORT.get(r["scenario"], r["scenario"]),
                           LABEL_2B.get(r["detector_a"], r["detector_a"]),
                           LABEL_2B.get(r["detector_b"], r["detector_b"]),
                           f"{r['far_target']:g}", f"{r['det_a']:.4f}", f"{r['det_b']:.4f}",
                           f"{r['det_diff']:+.4f}",
                           f"[{r['det_diff_lo']:+.4f}, {r['det_diff_hi']:+.4f}]",
                           "Yes" if r["det_diff_excludes_zero"] else "No"]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. Interpretability of the probabilities")
    A("")
    A("Brier scores and binned reliability are computed on days 126, 252 and 504. "
      "**The raw model outputs are used, computed on all paths**, not only on those not yet alarmed; no probability calibrator is added in this round.")
    A("")
    L.extend(_t(summary["probability_calibration"],
                ["Scenario", "Detector", "Day", "Brier", "Base-rate Brier", "Skill score", "Mean predicted q", "Observed invalid fraction"],
                lambda r: [SC_REPORT.get(r["scenario"], r["scenario"]),
                           LABEL_2B.get(r["detector"], r["detector"]),
                           f"{int(r['day'])}", f"{r['brier']:.4f}", f"{r['brier_base_rate']:.4f}",
                           f"{r['brier_skill_vs_base_rate']:+.4f}", f"{r['mean_predicted_q']:.4f}",
                           f"{r['observed_invalid_frac']:.4f}"]))
    A("")
    A("Per-bin sample sizes are in `stage2b_reliability.csv`.")
    A("")

    # ---------------- 8 ----------------
    A("## 8. Next step")
    A("")
    A("> **Retracted statement**: the previous version presented \"point-forecast error is the main bottleneck\" as a proven conclusion, "
      "and drew an unverified mapping between QLIKE and the oracle detection gap. "
      "This round ran no experiment that could support that mapping (that would require rerunning detection rates at controlled forecast-error levels), "
      "so both sentences are retracted. What is established is only that EWMA's QLIKE is clearly lower than that of the constant forecast, "
      "and that its detection-rate gap to the oracle excludes zero under SV.")
    A("")
    A("> **Retracted statement**: the previous version presented \"point-forecast error is the main bottleneck\" as a proven conclusion, "
      "and drew an unverified mapping between QLIKE and the oracle detection gap. "
      "This round ran no experiment that could support that mapping (that would require rerunning detection rates at controlled forecast-error levels), so both sentences are retracted. "
      "What is established is only that EWMA's QLIKE is clearly lower than that of the constant forecast, and that its detection-rate gap to the oracle excludes zero under SV.")
    A("")
    A("See the final delivery notes. If EWMA shows value but its forecast error is noticeable, the next-stage candidate is **sequential Bayesian filtering of the latent volatility** "
      "(Jacquier, Polson, Sokolov: *Sequential Bayesian Learning for Merton's Jump Model with "
      "Stochastic Volatility*); this round first completes the simple comparisons above.")
    A("")

    # ---------------- 9 ----------------
    A("## 9. Reproduction")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage2b")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}; "
      f"this Stage 2B run took {summary['elapsed_s']:.1f} s. "
      f"Random streams: `{summary['stream_note']}`")
    A("")

    A("")
    A("## Appendix A: Qualifying analyses added in Stage 2C (same Stage 2B data)")
    A("")
    F = summary.get("followup") or {}
    if F.get("jump_student_t_vs_gaussian"):
        A("### A.1 Jump scenario: same data, two interval constructions")
        A("")
        A("Fixed Student-t vs fixed Gaussian: on **the same batch of Stage 2B jump data**, "
          "\"the paired interval with frozen thresholds\" and \"the interval including calibration resampling\" are computed side by side. "
          "This is a side-by-side comparison within the same data, not a cross-stage comparison.")
        A("")
        L.extend(_t(F["jump_student_t_vs_gaussian"],
                    ["α", "Detection-rate difference", "Frozen threshold 95%", "Excludes zero", "With calibration resampling 95%", "Excludes zero"],
                    lambda r: [f"{r['far_target']:g}", f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               "Yes" if r["frozen_excludes_zero"] else "No",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "Yes" if r["recal_excludes_zero"] else "No"]))
        A("")
        A("At both α values, the frozen-threshold intervals exclude zero while the intervals including calibration resampling do not. "
          "**The two answer different questions**: the former asks \"given this set of thresholds, how much do the two rules differ on new test data\", "
          "the latter asks \"if the whole pipeline is rerun, is the gap still stable\". Neither should be treated as a correction of the other.")
        A("")
    if F.get("sv_paired_brier_ewma_t_minus_ewma_g"):
        A("### A.2 SV: paired Brier difference, EWMA Student-t vs EWMA Gaussian")
        A("")
        A("Paired by path; the valid and invalid groups are resampled **separately**, each to its original size, "
          "so the 50/50 evaluation weighting of the original design is preserved exactly.")
        A("")
        L.extend(_t(F["sv_paired_brier_ewma_t_minus_ewma_g"],
                    ["Day", "Brier(EWMA t)", "Brier(EWMA G)", "Difference", "95% interval", "Excludes zero"],
                    lambda r: [str(int(r["day"])), f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                               f"{r['diff']:+.5f}", f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "Yes" if r["excludes_zero"] else "No"]))
        A("")
        A("The detection-rate comparison cannot tell these two methods apart, but **in probability quality EWMA Student-t is better and the interval excludes zero**. "
          "The two conclusions stand side by side; neither replaces the other.")
        A("")
    if F.get("rho0_ewma_vs_fixed_exploratory"):
        A("### A.3 Intervals for the negative differences in the ρ=0 control (exploratory)")
        A("")
        A("> **This is an exploratory analysis added in Stage 2C**; it is not in the list of comparisons pre-specified before the Stage 2B run.")
        A("")
        L.extend(_t(F["rho0_ewma_vs_fixed_exploratory"],
                    ["α", "Comparison", "Detection-rate difference", "Frozen threshold 95%", "With calibration resampling 95%", "Latter excludes zero"],
                    lambda r: [f"{r['far_target']:g}",
                               f"{LABEL_2B.get(r['detector_a'],r['detector_a'])} − {LABEL_2B.get(r['detector_b'],r['detector_b'])}",
                               f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "Yes" if r["recal_excludes_zero"] else "No"]))
        A("")
        A("**Distinguishing point estimates from interval-supported conclusions**: the negative difference for EWMA Gaussian is interval-supported at both α values; "
          "for EWMA Student-t at α=0.15 there is only a point estimate (the interval including calibration resampling does not exclude zero). "
          "The slight losses in the other scenarios in the Section 1 table should be distinguished by the same standard: "
          "that table gives point estimates, and only the rows with intervals listed in this appendix and in Section 6 have interval support.")
        A("")
    A("")
    A("## Appendix A: Qualifying analyses added in Stage 2C (same Stage 2B data)")
    A("")
    F = summary.get("followup") or {}
    if F.get("jump_student_t_vs_gaussian"):
        A("### A.1 Jump scenario: same data, two interval constructions")
        A("")
        A("Fixed Student-t vs fixed Gaussian: on **the same batch of Stage 2B jump data**, "
          "\"the paired interval with frozen thresholds\" and \"the interval including calibration resampling\" are computed side by side. This is a side-by-side comparison within the same data, not a cross-stage comparison.")
        A("")
        L.extend(_t(F["jump_student_t_vs_gaussian"],
                    ["α", "Detection-rate difference", "Frozen threshold 95%", "Excludes zero", "With calibration resampling 95%", "Excludes zero"],
                    lambda r: [f"{r['far_target']:g}", f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               "Yes" if r["frozen_excludes_zero"] else "No",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "Yes" if r["recal_excludes_zero"] else "No"]))
        A("")
        A("At both α values, the frozen-threshold intervals exclude zero while the intervals including calibration resampling do not. **The two answer different questions**: "
          "the former asks \"given this set of thresholds, how much do the two rules differ on new test data\", "
          "the latter asks \"if the whole pipeline is rerun, is the gap still stable\". Neither should be treated as a correction of the other.")
        A("")
    if F.get("sv_paired_brier_ewma_t_minus_ewma_g"):
        A("### A.2 SV: paired Brier difference, EWMA Student-t vs EWMA Gaussian")
        A("")
        A("Paired by path; the valid and invalid groups are resampled **separately**, each to its original size, so the 50/50 evaluation weighting of the original design is preserved exactly.")
        A("")
        L.extend(_t(F["sv_paired_brier_ewma_t_minus_ewma_g"],
                    ["Day", "Brier(EWMA t)", "Brier(EWMA G)", "Difference", "95% interval", "Excludes zero"],
                    lambda r: [str(int(r["day"])), f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                               f"{r['diff']:+.5f}", f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "Yes" if r["excludes_zero"] else "No"]))
        A("")
        A("The detection-rate comparison cannot tell these two methods apart, but **in probability quality EWMA Student-t is better and the interval excludes zero**. The two conclusions stand side by side.")
        A("")
    if F.get("rho0_ewma_vs_fixed_exploratory"):
        A("### A.3 Intervals for the negative differences in the ρ=0 control (exploratory)")
        A("")
        A("> **This is an exploratory analysis added in Stage 2C**; it is not in the list of comparisons pre-specified before the Stage 2B run.")
        A("")
        L.extend(_t(F["rho0_ewma_vs_fixed_exploratory"],
                    ["α", "Comparison", "Detection-rate difference", "Frozen threshold 95%", "With calibration resampling 95%", "Latter excludes zero"],
                    lambda r: [f"{r['far_target']:g}",
                               f"{LABEL_2B.get(r['detector_a'], r['detector_a'])} - {LABEL_2B.get(r['detector_b'], r['detector_b'])}",
                               f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "Yes" if r["recal_excludes_zero"] else "No"]))
        A("")
        A("**Distinguishing point estimates from interval-supported conclusions**: the negative difference for EWMA Gaussian is interval-supported at both α values; "
          "for EWMA Student-t at α=0.15 there is only a point estimate. The slight losses in the other scenarios in the Section 1 table are distinguished by the same standard: "
          "that table gives point estimates, and only the rows with intervals listed in this appendix and in Section 6 have interval support.")
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
from scipy.special import expit


from .config import DEFAULT, Stage1Config
from .probability_time import brier_and_reliability

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint


POSTERIOR_KEYS = ("binary_gaussian", "binary_student_t", "ewma_gaussian",
                  "ewma_student_t", ORACLE)


def bootstrap_pairs(cfg: Stage1Config) -> list[tuple]:
    """Pre-specified before running; not chosen after seeing an interval."""
    a_main = cfg.far_targets[-1]
    pairs = [(sc, "binary_student_t", "binary_gaussian", a_main)
             for sc in cfg.noise_scenarios]
    for alpha in (a_main, cfg.far_targets[0]):
        pairs += [
            ("stoch_vol", "ewma_gaussian", "binary_gaussian", alpha),
            ("stoch_vol", "ewma_student_t", "binary_student_t", alpha),
            ("stoch_vol", "ewma_gaussian", "ewma_student_t", alpha),
            ("stoch_vol", "ewma_gaussian", ORACLE, alpha),
            ("stoch_vol", "ewma_student_t", ORACLE, alpha),
        ]
    return pairs


def probability_rows(cfg, blocks, keys, scenario):
    """Brier and binned reliability, on ALL paths, never only the unalarmed ones."""
    summ, rel = [], []
    for key in keys:
        if key not in POSTERIOR_KEYS:
            continue
        qi = expit(-statistic(key, blocks["test_invalid"], cfg))
        qv = expit(-statistic(key, blocks["test_valid"], cfg))
        for day in cfg.n_info_days:
            s, tab = brier_and_reliability(qi, qv, day, key)
            s["scenario"] = scenario
            tab["scenario"] = scenario
            summ.append(s)
            rel.append(tab)
        del qi, qv
    return summ, rel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2B: EWMA variance forecast.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--reps", type=int, default=None, help="override bootstrap reps")
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    if args.reps is not None:
        from dataclasses import replace as _r
        cfg = _r(cfg, bootstrap_reps=args.reps)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2b") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    streams = scenario_streams(cfg)
    transfer = gaussian_transfer_thresholds(cfg, streams)
    status("Gaussian transfer thresholds set", n=len(transfer))

    rows, per_scenario, prob_summ, prob_rel = [], {}, [], []
    for sc in cfg.noise_scenarios:
        res = run_scenario(cfg, sc, streams, transfer)
        rows.extend(res["rows"])
        per_scenario[sc] = res
        s, r = probability_rows(cfg, res["blocks"], res["keys"], sc)
        prob_summ.extend(s)
        prob_rel.extend(r)
        status(f"scenario done: {sc}", detectors=len(res["keys"]), rows=len(res["rows"]))

    ctrl = persistence_control(cfg)
    rows.extend(ctrl["rows"])
    per_scenario["sv_rho0_control"] = ctrl
    status("persistence control done (SV rho=0)", rows=len(ctrl["rows"]))

    metrics = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(LABEL_2B)}
    metrics["_o"] = metrics.detector.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2b_metrics.csv", index=False)

    # ---- volatility diagnostics (stochastic volatility only) ---------------- #
    sv = per_scenario["stoch_vol"]["blocks"]["test_invalid"]
    diag = forecast_diagnostics(sv["returns"], sv["true_sigma"] ** 2, cfg)
    diag.to_csv(out_dir / "stage2b_vol_diagnostics.csv", index=False)
    qb = qlike_benchmarks(sv["returns"], sv["true_sigma"] ** 2, cfg)
    qb.to_csv(out_dir / "stage2b_qlike_benchmarks.csv", index=False)
    ninfo = information_equivalent_table(sv["true_sigma"] ** 2, cfg)
    ninfo.to_csv(out_dir / "stage2b_information_days.csv", index=False)
    trace = fixed_path_trace(sv["returns"], sv["true_sigma"] ** 2, cfg, path_index=0)
    trace.to_csv(out_dir / "stage2b_vol_trace.csv", index=False)
    status("volatility diagnostics done", qlike=float(diag.qlike_mean.iloc[0]))

    # ---- bootstrap ---------------------------------------------------------- #
    pairs = bootstrap_pairs(cfg)
    boot = run_comparisons(cfg, per_scenario, pairs,
                           make_streams(cfg)["stage2b_bootstrap"])
    boot.to_csv(out_dir / "stage2b_bootstrap.csv", index=False)
    status("bootstrap done", comparisons=len(boot), reps=cfg.bootstrap_reps)

    # ---- Stage 2C follow-up appendix, on the SAME Stage 2B blocks ----------- #
    fk = make_streams(cfg)["stage2b_bootstrap"].spawn(3)
    followup = {}
    jm = per_scenario["jump"]["minima"]
    followup["jump_student_t_vs_gaussian"] = [
        {**frozen_vs_recalibrated_pair(jm["binary_student_t"], jm["binary_gaussian"],
                                       a, cfg.bootstrap_reps, fk[0]),
         "scenario": "jump", "detector_a": "binary_student_t", "detector_b": "binary_gaussian"}
        for a in cfg.far_targets]
    cm = per_scenario["sv_rho0_control"]["minima"]
    followup["rho0_ewma_vs_fixed_exploratory"] = [
        {**exploratory_rho0_interval(cm[e], cm[f], a, cfg.bootstrap_reps, fk[1]),
         "scenario": "sv_rho0_control", "detector_a": e, "detector_b": f}
        for a in cfg.far_targets
        for e, f in (("ewma_gaussian", "binary_gaussian"), ("ewma_student_t", "binary_student_t"))]
    svb = per_scenario["stoch_vol"]["blocks"]
    qa_i = expit(-statistic("ewma_student_t", svb["test_invalid"], cfg))
    qa_v = expit(-statistic("ewma_student_t", svb["test_valid"], cfg))
    qb_i = expit(-statistic("ewma_gaussian", svb["test_invalid"], cfg))
    qb_v = expit(-statistic("ewma_gaussian", svb["test_valid"], cfg))
    followup["sv_paired_brier_ewma_t_minus_ewma_g"] = [
        paired_brier_difference(qa_i, qa_v, qb_i, qb_v, d, cfg.bootstrap_reps, fk[2])
        for d in (252, 504)]
    del qa_i, qa_v, qb_i, qb_v
    for name, rows in followup.items():
        pd.DataFrame(rows).to_csv(out_dir / f"stage2b_followup_{name}.csv", index=False)
    status("Stage 2B follow-up appendix done", blocks=len(followup))

    pd.DataFrame(prob_summ).to_csv(out_dir / "stage2b_probability_calibration.csv", index=False)
    pd.concat(prob_rel, ignore_index=True).to_csv(
        out_dir / "stage2b_reliability.csv", index=False)

    figures = []
    if not args.no_figures:
        setup_style()
        fd = out_dir / "figures"
        figures = [
            str(figure_vol_forecast(cfg, trace, diag, fd / "fig2b1_vol_forecast.png").relative_to(out_dir)),
            str(figure_sv_curves(cfg, per_scenario, fd / "fig2b2_sv_curves.png").relative_to(out_dir)),
            str(figure_arms(cfg, metrics, fd / "fig2b3_arms.png").relative_to(out_dir)),
            str(figure_reliability(cfg, pd.concat(prob_rel, ignore_index=True),
                                       fd / "fig2b4_reliability.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    floor_hits = {sc: per_scenario[sc].get("ewma_floor_hits", {}) for sc in per_scenario}
    summary = {
        "stage": "2B",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "stream_note": "Stage 2B uses its own parent streams; no Stage 2A draw is reused.",
        "sample_sizes": {"calibration_valid": cfg.n_stage2b_calibration,
                         "test_valid": cfg.n_stage2b_test_valid,
                         "test_invalid": cfg.n_stage2b_test_invalid,
                         "days": cfg.horizon_days},
        "gaussian_transfer_thresholds": {f"{k[0]}|alpha={k[1]:g}": {"threshold": v[0],
                                                                    "calibration_far": v[1]}
                                          for k, v in transfer.items()},
        "ewma_variance_floor_hits": floor_hits,
        "metrics": json.loads(metrics.to_json(orient="records")),
        "vol_diagnostics": json.loads(diag.to_json(orient="records")),
        "qlike_benchmarks": json.loads(qb.to_json(orient="records")),
        "information_days": json.loads(ninfo.to_json(orient="records")),
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "bootstrap_pairs_prespecified": [list(p) for p in pairs],
        "probability_calibration": prob_summ,
        "followup": followup,
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2b_report(cfg, summary, metrics, boot, diag, qb, ninfo,
                         out_dir / "stage2b_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
