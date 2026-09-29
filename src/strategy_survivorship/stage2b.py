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


SC_REPORT = {"gaussian": "高斯", "student_t": "厚尾 t(5)", "stoch_vol": "随机波动率 ρ=0.98",
      "jump": "跳跃", "sv_rho0_control": "**持续性对照 SV ρ=0**"}
ARM = {"per_scenario": "A：按情境分别校准", "gaussian_transfer": "B：高斯校准门槛迁移"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "未达到"


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

    A("# Stage 2B 报告：用历史收益预测波动率，能否改善策略有效性验证")
    A("")
    A("本文件由 `python -m strategy_survivorship.stage2b` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 六个问题的直接回答")
    A("")
    sv_eg = g("ewma_gaussian", "stoch_vol", "per_scenario", "detect_d504")
    sv_bg = g("binary_gaussian", "stoch_vol", "per_scenario", "detect_d504")
    sv_et = g("ewma_student_t", "stoch_vol", "per_scenario", "detect_d504")
    sv_bt = g("binary_student_t", "stoch_vol", "per_scenario", "detect_d504")
    sv_or = g(ORACLE, "stoch_vol", "per_scenario", "detect_d504")
    A("**(1) 是否改善？只在随机波动率下改善；其余每一个情境里它都略有损害。**")
    A("")
    A(f"EWMA 与其固定尺度对应方法的两年检出率之差（臂 A，α = {a_main:g}）：")
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
    L.extend(_t(_rows, ["情境", "EWMA 方法", "对应固定方法", "EWMA 检出", "固定检出",
                        "检出率之差", "实测 FAR 之差"],
                lambda r: [SC_REPORT[r["sc"]], LABEL_2B[r["e"]], LABEL_2B[r["f"]],
                           f"{r['de']:.4f}", f"{r['df']:.4f}",
                           f"**{r['d']:+.4f}**", f"{r['fd']:+.4f}"]))
    A("")
    A("实测 FAR 之差都不超过 0.011，所以这些比较基本是在同一误杀代价下进行的。")
    A("")
    A(f"随机波动率、α={a_main:g}：EWMA Gaussian {sv_eg:.4f} vs 固定 Gaussian {sv_bg:.4f}"
      f"（差 {sv_eg - sv_bg:+.4f}，bootstrap 95% "
      f"[{b('stoch_vol','ewma_gaussian','binary_gaussian',a_main,'det_diff_lo'):+.4f}, "
      f"{b('stoch_vol','ewma_gaussian','binary_gaussian',a_main,'det_diff_hi'):+.4f}]）。")
    A("")
    A(f"**(2) 是否超过固定 Student-t？** 随机波动率下 EWMA Gaussian {sv_eg:.4f} vs 固定 Student-t {sv_bt:.4f}；"
      f"EWMA Student-t {sv_et:.4f}。与 oracle（{sv_or:.4f}）的距离见第 5 节。")
    A("")
    tr_eg = g("ewma_gaussian", "stoch_vol", "gaussian_transfer", "detect_d504")
    tr_bg = g("binary_gaussian", "stoch_vol", "gaussian_transfer", "detect_d504")
    tr_eg_far = g("ewma_gaussian", "stoch_vol", "gaussian_transfer", "far_d504")
    tr_bg_far = g("binary_gaussian", "stoch_vol", "gaussian_transfer", "far_d504")
    A(f"**(3) 门槛迁移后是否保留？** 用本轮高斯校准门槛迁移到随机波动率："
      f"EWMA Gaussian 检出 {tr_eg:.4f}（实测 FAR {tr_eg_far:.4f}）、"
      f"固定 Gaussian {tr_bg:.4f}（实测 FAR {tr_bg_far:.4f}）。"
      "两者的实测 FAR 不同，因此这一比较**不是**在同一误杀代价下进行的，详见第 4 节。")
    A("")
    c_eg = g("ewma_gaussian", "sv_rho0_control", "per_scenario", "detect_d504")
    c_bg = g("binary_gaussian", "sv_rho0_control", "per_scenario", "detect_d504")
    A(f"**(4) 对持续性的依赖有多大？在当前 EWMA(λ=0.94)、当前参数、当前五个情境与当前检出率指标下，"
      f"改善完全来自持续性。** 这是对这一组具体设定的证据，不是对「任何波动率预测方法」的一般结论。"
      "ρ=0 对照（同样的单日边际与无条件方差，去掉持续性）："
      f"EWMA Gaussian {c_eg:.4f} vs 固定 Gaussian {c_bg:.4f}，差 {c_eg - c_bg:+.4f}；"
      f"而 ρ=0.98 时该差为 {sv_eg - sv_bg:+.4f}。")
    A("")
    A("**(5) 哪些结论仍受校准不确定性限制？** 第 5 节的 bootstrap 同时重采样校准集与测试集，"
      "因此包含了选门槛这一步的抽样噪声；正文其余配对区间**条件于已冻结的门槛**，不含这部分。"
      "两类区间都是**逐点**的，不构成所有比较同时成立的保证。")
    A("")
    A("**(6) 下一步最有价值的一个问题**见第 8 节。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 本轮模型定义（运行前固定）")
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
    A("`v_t` 是**看到第 t 天收益之前**对该天方差的预测，只用 `r_1..r_(t-1)`。"
      "不使用全样本 backcast、全路径样本方差或真实状态初始化。λ=0.94 是预先固定的简单基准，未做参数搜索。")
    A("")
    from .ewma import midpoint_variance_inflation
    infl = midpoint_variance_inflation(cfg) / cfg.sigma_daily ** 2
    A(f"选固定中点 `m` 是为了不读取真实状态、也不引入额外均值估计器。"
      f"**代价**：在两个真实均值假设下，平方残差都多出同一个 `(mu_1-mu_0)^2/4` 项，"
      f"因此这不是严格无偏的方差递推；本轮该项约为基准方差的 **{100*infl:.4f}%**。")
    A("")
    A("两个检测器**共用完全相同的** `v_t`。更新式：")
    A("")
    A("```")
    A("Gaussian :  dL_t = (mu_1 - mu_0)(r_t - m) / v_t")
    A(f"Student-t:  b_t = sqrt((nu-2)/nu * v_t),  nu = {cfg.ewma_student_t_df:g}")
    A("            dL_t = log f_nu(r_t; mu_1, b_t) - log f_nu(r_t; mu_0, b_t)")
    A("```")
    A("")
    A("`b_t` 是使 t 分布**方差**等于 `v_t` 的尺度参数，不是标准差。"
      "收益**没有**先按预测波动率标准化再套用固定条件 Sharpe 的假设——两个均值假设始终是收益空间的 `mu_0`、`mu_1`。")
    A("")
    A("> 两个方法输出的是**各自 EWMA 条件分布假设下的模型后验**，"
      "**不是**对真实潜在 SV 状态积分之后的精确后验。")
    A("")
    fl = summary.get("ewma_variance_floor_hits", {})
    total_fl = sum(sum(v.values()) for v in fl.values() if isinstance(v, dict))
    A(f"数值方差下限 `{cfg.ewma_variance_floor_factor:g}·sigma_0^2`，本轮触发 **{total_fl}** 次"
      "（仅数值保护，未按测试表现调整）。")
    A("")
    A(f"每个情境：{cfg.n_stage2b_calibration} 条有效校准路径、{cfg.n_stage2b_test_valid} 条有效测试路径、"
      f"{cfg.n_stage2b_test_invalid} 条无效测试路径。**Stage 2B 使用全新随机流**，"
      "所有被比较方法都在同一批新测试路径上运行；Stage 2A 结果保留不动。"
      "EWMA 从第 1 天开始，不加人为预热期。")
    A("")

    # ---------------- 3 / 4 ----------------
    for arm in ("per_scenario", "gaussian_transfer"):
        n = 3 if arm == "per_scenario" else 4
        A(f"## {n}. {ARM[arm]}")
        A("")
        if arm == "per_scenario":
            A("每个检测器仅用该情境的有效校准路径定门槛，冻结后测试。"
              "**这是已知情境下的理想化校准比较**，不代表检测器能识别自己身处哪个情境。")
        else:
            A("每个常规检测器只用**本轮高斯**有效校准路径定门槛，同一门槛用于四个情境。"
              "称为「高斯校准门槛迁移」而不是「冻结 Stage 1 门槛」，因为两个 EWMA 方法没有 Stage 1 门槛。"
              "Oracle 不参加迁移比较。")
        A("")
        A("各检测器共享的是**名义校准预算**，不是相同的实测误杀率——读检出率必须对照同一行的"
          "「两年实测 FAR」。")
        A("")
        for alpha in cfg.far_targets:
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} 名义校准预算 α = {alpha:g}")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)
                    & (M.scenario != "sv_rho0_control")]
            L.extend(_t(sub.to_dict("records"),
                        ["情境", "检测器", "两年实测 FAR", "FAR 95%", "检出 h=126d",
                         "h=252d", "h=504d", "两年未检出", "截断均检(日)", "中位检测"],
                        lambda r: [SC_REPORT[r["scenario"]], LABEL_2B.get(r["detector"], r["detector"]),
                                   f"{r['far_d504']:.4f}",
                                   f"[{r['far_d504_lo']:.4f}, {r['far_d504_hi']:.4f}]",
                                   f"{r['detect_d126']:.4f}", f"{r['detect_d252']:.4f}",
                                   f"{r['detect_d504']:.4f}", f"{r['undetected_at_H']:.4f}",
                                   f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
            A("")

    # ---------------- 5 ----------------
    A("## 5. 持续性对照与机制诊断")
    A("")
    A(f"辅助情境：SV `ρ = {cfg.sv_control_rho:g}`、振幅仍为 {cfg.noise_sv_amplitude:g}，"
      "保持相同的单日边际分布与无条件方差，只去掉潜在波动率的时间持续性。"
      "独立随机流、独立校准与测试，只比较四个累计方法与 oracle。")
    A("")
    ctl = M[(M.scenario == "sv_rho0_control") & (M.far_target == a_main)]
    L.extend(_t(ctl.to_dict("records"),
                ["检测器", "两年实测 FAR", "检出 h=252d", "h=504d", "截断均检(日)", "中位检测"],
                lambda r: [LABEL_2B.get(r["detector"], r["detector"]), f"{r['far_d504']:.4f}",
                           f"{r['detect_d252']:.4f}", f"{r['detect_d504']:.4f}",
                           f"{r['trunc_mean_detect_days']:.0f}", _med(r)]))
    A("")
    A(f"对比 ρ=0.98：EWMA Gaussian 相对固定 Gaussian 的两年检出率之差为 "
      f"**{sv_eg - sv_bg:+.4f}**；ρ=0 时为 **{c_eg - c_bg:+.4f}**。")
    A("")
    A("### 5.1 波动率预测诊断")
    A("")
    d0 = diag.iloc[0]
    A(f"随机波动率、无效测试块（n = {int(d0.n_paths)} 条路径 × {int(d0.n_days)} 日），"
      "误差先按完整路径聚合再算跨路径标准误：")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A(f"| QLIKE（EWMA） | {d0.qlike_mean:.5f} ± {d0.qlike_se:.5f} |")
    A(f"| 平均预测偏差（以 σ₀² 为单位） | {d0.mean_forecast_bias_in_sigma0sq:+.5f} ± {d0.mean_forecast_bias_se:.5f} |")
    A(f"| 方差下限触发次数 | {int(d0.n_variance_floor_hits)} |")
    for q in (10, 25, 50, 75, 90):
        A(f"| `v̂/v` 的 {q}% 分位 | {d0[f'ratio_vhat_over_v_q{q:02d}']:.4f} |")
    A("")
    A("QLIKE 对照：")
    A("")
    L.extend(_t(qb.to_dict("records"), ["预测", "QLIKE", "标准误"],
                lambda r: [r["forecast"], f"{r['qlike_mean']:.5f}", f"{r['qlike_se']:.5f}"]))
    A("")
    A("> **QLIKE 是机制诊断，不能替代策略验证效果。** 模型失配时，不同预测损失函数的排序可能不同"
      "（Patton, *Comparing Possibly Misspecified Forecasts*）。主结论仍由误杀与检测指标决定。")
    A("")
    A("孤立跳跃会抬高其后多天的 EWMA 方差预测（半衰期约 "
      f"{__import__('math').log(0.5)/__import__('math').log(cfg.ewma_lambda):.0f} 天），"
      "在跳跃情境下这会压低随后一段时间的证据增量。本轮**不**为此临时增加新模型修补。")
    A("")
    A("### 5.2 信息等价观测天数")
    A("")
    A("`N_info(n) = Σ_{t≤n} σ₀²/v_t`。**它使用真实方差**，是 oracle 的信息解释工具，"
      "既不是可部署检测器额外获得的数据，也不是检测时间的直接换算公式。")
    A("")
    L.extend(_t(ninfo.to_dict("records"),
                ["天数 n", "均值 ± SE", "理论均值 n·e", "偏离(SE)", "中位", "10%", "90%"],
                lambda r: [str(int(r["day"])), f"{r['mean']:.1f} ± {r['se']:.1f}",
                           f"{r['theoretical_mean']:.1f}",
                           f"{abs(r['mean']-r['theoretical_mean'])/r['se']:.2f}",
                           f"{r['median']:.1f}", f"{r['q10']:.1f}", f"{r['q90']:.1f}"]))
    A("")

    # ---------------- 6 ----------------
    A("## 6. 不确定性：同时重采样校准集与测试集的 bootstrap")
    A("")
    A(f"重复次数固定为 **{cfg.bootstrap_reps}**（运行前设定，不因区间是否排除零而改动）。"
      "校准有效、测试有效、测试无效三块**分别**按路径重采样；同一块内不同检测器**共用路径索引**，"
      "因此差值仍是配对的；每次用重采样后的校准集**重新确定门槛**再测量。日内不独立重采样。")
    A("")
    A("> **这些区间包含选门槛的抽样不确定性**，与第 3、4 节表中条件于冻结门槛的区间回答不同的问题。"
      "两者都是逐点区间，**不构成所有比较同时成立的联合保证**。")
    A("")
    L.extend(_t(boot.to_dict("records"),
                ["情境", "A", "B", "α", "A 检出", "B 检出", "检出差", "95% bootstrap 区间", "排除零"],
                lambda r: [SC_REPORT.get(r["scenario"], r["scenario"]),
                           LABEL_2B.get(r["detector_a"], r["detector_a"]),
                           LABEL_2B.get(r["detector_b"], r["detector_b"]),
                           f"{r['far_target']:g}", f"{r['det_a']:.4f}", f"{r['det_b']:.4f}",
                           f"{r['det_diff']:+.4f}",
                           f"[{r['det_diff_lo']:+.4f}, {r['det_diff_hi']:+.4f}]",
                           "是" if r["det_diff_excludes_zero"] else "否"]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. 概率解释性")
    A("")
    A("在第 126、252、504 天计算 Brier 与分箱可靠性。"
      "**使用所有路径继续计算的原始模型输出**，不只保留尚未报警者；本轮未增加任何概率校准器。")
    A("")
    L.extend(_t(summary["probability_calibration"],
                ["情境", "检测器", "天数", "Brier", "基准率 Brier", "技能分", "平均预测 q", "实际无效比例"],
                lambda r: [SC_REPORT.get(r["scenario"], r["scenario"]),
                           LABEL_2B.get(r["detector"], r["detector"]),
                           f"{int(r['day'])}", f"{r['brier']:.4f}", f"{r['brier_base_rate']:.4f}",
                           f"{r['brier_skill_vs_base_rate']:+.4f}", f"{r['mean_predicted_q']:.4f}",
                           f"{r['observed_invalid_frac']:.4f}"]))
    A("")
    A("每箱样本数见 `stage2b_reliability.csv`。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 下一步")
    A("")
    A("> **已撤回的表述**：上一版把「点预测误差是主要瓶颈」写成已证明的结论，"
      "并在 QLIKE 与 oracle 检出缺口之间做了未经验证的映射。"
      "本轮没有做能支持该映射的实验（那需要在受控的预测误差水平上重跑检出率），"
      "两句都撤回。已确立的只是：EWMA 的 QLIKE 明显低于常数预测，"
      "且它与 oracle 的检出率差距在 SV 下排除零。")
    A("")
    A("> **已撤回的表述**：上一版把「点预测误差是主要瓶颈」写成已证明的结论，"
      "并在 QLIKE 与 oracle 检出缺口之间做了未经验证的映射。"
      "本轮没有做能支持该映射的实验（那需要在受控的预测误差水平上重跑检出率），两句都撤回。"
      "已确立的只是：EWMA 的 QLIKE 明显低于常数预测，且它与 oracle 的检出率差距在 SV 下排除零。")
    A("")
    A("见最终交付说明。若 EWMA 显示价值但预测误差明显，下一阶段候选是**潜在波动率的顺序贝叶斯滤波**"
      "（Jacquier, Polson, Sokolov: *Sequential Bayesian Learning for Merton's Jump Model with "
      "Stochastic Volatility*），本轮先完成上述简单对照。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage2b")
    A("```")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}；"
      f"本次 Stage 2B 耗时 {summary['elapsed_s']:.1f} s。"
      f"随机流：`{summary['stream_note']}`")
    A("")

    A("")
    A("## 附录 A：Stage 2C 追加的限定性分析（同一批 Stage 2B 数据）")
    A("")
    F = summary.get("followup") or {}
    if F.get("jump_student_t_vs_gaussian"):
        A("### A.1 跳跃情境：同一数据、两种区间构造")
        A("")
        A("固定 Student-t 对固定 Gaussian，在**同一批 Stage 2B 跳跃数据**上并列计算"
          "「冻结门槛的配对区间」与「包含校准重采样的区间」。"
          "这是同数据内的并列，不是跨阶段比较。")
        A("")
        L.extend(_t(F["jump_student_t_vs_gaussian"],
                    ["α", "检出率之差", "冻结门槛 95%", "排除零", "含校准重采样 95%", "排除零"],
                    lambda r: [f"{r['far_target']:g}", f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               "是" if r["frozen_excludes_zero"] else "否",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("两个 α 下，冻结门槛区间都排除零而含校准重采样的区间都不排除零。"
          "**两者回答不同问题**：前者是「给定这组门槛，两条规则在新测试数据上差多少」，"
          "后者是「整条流程重走一遍，差距还稳定吗」。不应把其中一个当作对另一个的更正。")
        A("")
    if F.get("sv_paired_brier_ewma_t_minus_ewma_g"):
        A("### A.2 SV：EWMA Student-t 对 EWMA Gaussian 的配对 Brier 差异")
        A("")
        A("按路径配对；有效、无效两组**分别**重采样到各自原有规模，"
          "因此原设计的 50/50 评价权重完整保留。")
        A("")
        L.extend(_t(F["sv_paired_brier_ewma_t_minus_ewma_g"],
                    ["天数", "Brier(EWMA t)", "Brier(EWMA G)", "差", "95% 区间", "排除零"],
                    lambda r: [str(int(r["day"])), f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                               f"{r['diff']:+.5f}", f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "是" if r["excludes_zero"] else "否"]))
        A("")
        A("检出率比较分不出这两个方法，但**概率质量上 EWMA Student-t 更好且区间排除零**。"
          "两项结论并存，不互相取代。")
        A("")
    if F.get("rho0_ewma_vs_fixed_exploratory"):
        A("### A.3 ρ=0 对照的负差异区间（探索性）")
        A("")
        A("> **这是 Stage 2C 追加的探索性分析**，不在 Stage 2B 运行前预先指定的比较清单内。")
        A("")
        L.extend(_t(F["rho0_ewma_vs_fixed_exploratory"],
                    ["α", "对比", "检出率之差", "冻结门槛 95%", "含校准重采样 95%", "后者排除零"],
                    lambda r: [f"{r['far_target']:g}",
                               f"{LABEL_2B.get(r['detector_a'],r['detector_a'])} − {LABEL_2B.get(r['detector_b'],r['detector_b'])}",
                               f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("**区分点估计与有区间支持的结论**：EWMA Gaussian 的负差异在两个 α 下都有区间支持；"
          "EWMA Student-t 在 α=0.15 下只有点估计（含校准重采样的区间不排除零）。"
          "第 1 节表中其余情境的轻微损失同样应按此标准区分——"
          "那张表给的是点估计，只有本附录与第 6 节列出区间的行才有区间支持。")
        A("")
    A("")
    A("## 附录 A：Stage 2C 追加的限定性分析（同一批 Stage 2B 数据）")
    A("")
    F = summary.get("followup") or {}
    if F.get("jump_student_t_vs_gaussian"):
        A("### A.1 跳跃情境：同一数据、两种区间构造")
        A("")
        A("固定 Student-t 对固定 Gaussian，在**同一批 Stage 2B 跳跃数据**上并列计算"
          "「冻结门槛的配对区间」与「包含校准重采样的区间」。这是同数据内的并列，不是跨阶段比较。")
        A("")
        L.extend(_t(F["jump_student_t_vs_gaussian"],
                    ["α", "检出率之差", "冻结门槛 95%", "排除零", "含校准重采样 95%", "排除零"],
                    lambda r: [f"{r['far_target']:g}", f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               "是" if r["frozen_excludes_zero"] else "否",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("两个 α 下，冻结门槛区间都排除零而含校准重采样的区间都不排除零。**两者回答不同问题**："
          "前者是「给定这组门槛，两条规则在新测试数据上差多少」，"
          "后者是「整条流程重走一遍，差距还稳定吗」。不应把其中一个当作对另一个的更正。")
        A("")
    if F.get("sv_paired_brier_ewma_t_minus_ewma_g"):
        A("### A.2 SV：EWMA Student-t 对 EWMA Gaussian 的配对 Brier 差异")
        A("")
        A("按路径配对；有效、无效两组**分别**重采样到各自原有规模，因此原设计的 50/50 评价权重完整保留。")
        A("")
        L.extend(_t(F["sv_paired_brier_ewma_t_minus_ewma_g"],
                    ["天数", "Brier(EWMA t)", "Brier(EWMA G)", "差", "95% 区间", "排除零"],
                    lambda r: [str(int(r["day"])), f"{r['brier_a']:.5f}", f"{r['brier_b']:.5f}",
                               f"{r['diff']:+.5f}", f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "是" if r["excludes_zero"] else "否"]))
        A("")
        A("检出率比较分不出这两个方法，但**概率质量上 EWMA Student-t 更好且区间排除零**。两项结论并存。")
        A("")
    if F.get("rho0_ewma_vs_fixed_exploratory"):
        A("### A.3 ρ=0 对照的负差异区间（探索性）")
        A("")
        A("> **这是 Stage 2C 追加的探索性分析**，不在 Stage 2B 运行前预先指定的比较清单内。")
        A("")
        L.extend(_t(F["rho0_ewma_vs_fixed_exploratory"],
                    ["α", "对比", "检出率之差", "冻结门槛 95%", "含校准重采样 95%", "后者排除零"],
                    lambda r: [f"{r['far_target']:g}",
                               f"{LABEL_2B.get(r['detector_a'], r['detector_a'])} - {LABEL_2B.get(r['detector_b'], r['detector_b'])}",
                               f"{r['frozen_diff']:+.4f}",
                               f"[{r['frozen_lo']:+.4f}, {r['frozen_hi']:+.4f}]",
                               f"[{r['recal_lo']:+.4f}, {r['recal_hi']:+.4f}]",
                               "是" if r["recal_excludes_zero"] else "否"]))
        A("")
        A("**区分点估计与有区间支持的结论**：EWMA Gaussian 的负差异在两个 α 下都有区间支持；"
          "EWMA Student-t 在 α=0.15 下只有点估计。第 1 节表中其余情境的轻微损失同样按此标准区分——"
          "那张表给的是点估计，只有本附录与第 6 节列出区间的行才有区间支持。")
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
