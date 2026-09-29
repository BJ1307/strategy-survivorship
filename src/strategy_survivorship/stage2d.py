"""stage2d: one module per research stage.

- Model and computation: Stage 2D: persistent stochastic volatility AND isolated jumps, together.
- Model and computation: Stage 2D uncertainty: frozen-threshold paired intervals, and a bootstrap that
- Figures: Stage 2D figures. Drawn from the CSVs on disk, so redrawing can never
- Report: Generate ``stage2d_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 2D pipeline.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Stage1Config
from .evaluate import wilson_interval
from .noise import (diffusive_daily_variance, draw_noise, returns_from_noise,
                    total_daily_variance_given_vol, true_daily_sigma)
from .simulate import make_streams
from .stage2c import (LABEL_2C, LATE_STARTERS, METHODS, buffered_rank, first_alarm_day,
                      first_eligible, metric_row, path_minima, rank_via_beta, statistic)

LABEL_2D = dict(LABEL_2C)
SCEN_LABEL = {
    "gaussian_ctrl": "Gaussian control (A=0, k=0)",
    "sv_ctrl": "SV control (A=1, k=0)",
    "jump_ctrl": "Jump control (A=0, k=5)",
    "sv_jump": "SV + jumps (A=1, k=5)",
    "sv_jump_big": "SV + larger jumps (A=1, k=8)",
}
COMBINED = ("sv_jump", "sv_jump_big")
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def scenario_cfg(cfg: Stage1Config, A: float, kappa: float) -> Stage1Config:
    """Only (A, kappa) move; rho and lambda stay at the frozen values."""
    return replace(cfg, noise_sv_amplitude=float(A), noise_jump_kappa=float(kappa))


def specs(cfg: Stage1Config):
    return [tuple(s) for s in cfg.stage2d_scenarios]


def streams_for(cfg: Stage1Config, parent: str) -> dict:
    kids = make_streams(cfg)[parent].spawn(len(specs(cfg)) * len(ROLES))
    return {(sp[0], role): kids[i * len(ROLES) + j]
            for i, sp in enumerate(specs(cfg)) for j, role in enumerate(ROLES)}


def make_block(cfg: Stage1Config, spec, seed_seq, n_paths: int, sharpe: float) -> dict:
    key, A, kappa, _ = spec
    cn = scenario_cfg(cfg, A, kappa)
    d = draw_noise("sv_jump", seed_seq, n_paths, cfg.horizon_days, cn)
    # Both latent variances are stored for scoring and diagnostics only; no
    # detector reads them, and neither is a real-time forecast.
    return {"returns": returns_from_noise(d.eps, sharpe, cfg),
            "diffusive_variance": diffusive_daily_variance(d, cn),
            "total_variance_given_vol": total_daily_variance_given_vol(d, cn),
            "latent": d.latent}


def load_frozen_thresholds(cfg: Stage1Config, path: Path) -> dict:
    """Read the Stage 2C unified thresholds exactly as they were written."""
    t = pd.read_csv(path)
    return {(r.method, float(r.far_target)): float(r.unified_threshold)
            for r in t.itertuples()}


def calibrate_diagnostic(cfg: Stage1Config) -> dict:
    """Per-scenario thresholds with the same buffer as Stage 2C.

    J is again methods x scenarios x budgets = 60, so delta/J matches Stage 2C's
    per-comparison budget and the ranks come out the same at the same N.
    """
    sp = specs(cfg)
    st = streams_for(cfg, "stage2d_calibration")
    n = cfg.stage2d_calibration_paths
    J = len(METHODS) * len(sp) * len(cfg.far_targets)
    dj = cfg.stage2c_delta / J
    ranks = {a: {"buffered": buffered_rank(n, a, dj), "beta_route": rank_via_beta(n, a, dj)}
             for a in cfg.far_targets}
    thr, rows = {}, []
    for spec in sp:
        key = spec[0]
        blk = make_block(cfg, spec, st[(key, "calibration_valid")], n, cfg.sharpe_valid)
        for m in METHODS:
            mins = np.sort(path_minima(statistic(m, blk["returns"], cfg),
                                       first_eligible(m, cfg), cfg.horizon_days))
            for a in cfg.far_targets:
                k = ranks[a]["buffered"]
                thr[(m, key, a)] = float(mins[k - 1])
                rows.append({"method": m, "scenario": key, "far_target": a, "rank": k,
                             "threshold": float(mins[k - 1]), "n_calibration_paths": n})
        del blk
    return {"thresholds": thr, "ranks": ranks, "J": J, "delta_per_comparison": dj,
            "table": pd.DataFrame(rows)}


def evaluate(cfg: Stage1Config, spec, seed_seq, transfer: dict, diagnostic: dict, days) -> tuple:
    """Both arms for one scenario, plus per-path minima for the bootstrap."""
    key = spec[0]
    kids = seed_seq.spawn(2)
    bv = make_block(cfg, spec, kids[0], cfg.stage2d_test_paths, cfg.sharpe_valid)
    bi = make_block(cfg, spec, kids[1], cfg.stage2d_test_paths, cfg.sharpe_invalid)
    rows, minima, qrows, trunc = [], {}, [], {}
    from scipy.special import expit

    for m in METHODS:
        elig = first_eligible(m, cfg)
        sv = statistic(m, bv["returns"], cfg)
        si = statistic(m, bi["returns"], cfg)
        minima[m] = {"test_valid": path_minima(sv, elig, cfg.horizon_days),
                     "test_invalid": path_minima(si, elig, cfg.horizon_days)}
        if m in ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"):
            for state, st_ in (("valid", sv), ("invalid", si)):
                q = expit(-st_[:, [d - 1 for d in days]])
                for j, d in enumerate(days):
                    col = q[:, j]
                    qrows.append({"scenario": key, "method": m, "true_state": state, "day": d,
                                  "n_paths": int(col.size), "mean_q": float(col.mean()),
                                  "median_q": float(np.median(col)),
                                  "q10": float(np.quantile(col, 0.10)),
                                  "q90": float(np.quantile(col, 0.90))})
        for a in cfg.far_targets:
            for arm, thr in (("transfer", transfer[(m, a)]),
                             ("diagnostic", diagnostic["thresholds"][(m, key, a)])):
                tv = first_alarm_day(sv, thr, elig, cfg.horizon_days)
                ti = first_alarm_day(si, thr, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "method": m, "arm": arm, "far_target": a,
                       "threshold": thr, "first_eligible_day": elig,
                       "far_alarms": far["n_alarms"], "far_lo": lo, "far_hi": hi,
                       "in_stage2c_coverage": bool(spec[3])}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                    dl, dh = wilson_interval(det[f"n_alarms_d{d}"], det["n_paths"], cfg.wilson_z)
                    row[f"detect_d{d}_lo"], row[f"detect_d{d}_hi"] = dl, dh
                trunc[(m, arm, a)] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
                row.update({"trunc_mean_detect_days": det["trunc_mean_days"],
                            "trunc_mean_detect_se": det["trunc_mean_se"],
                            "median_detect_days": det["median_days"],
                            "median_detect_note": det["median_note"],
                            "undetected_at_H": det["undetected_at_H"],
                            "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]})
                rows.append(row)
        del sv, si
    return rows, minima, qrows, bv, bi, trunc


def shock_diagnostic(cfg: Stage1Config) -> pd.DataFrame:
    """One pre-fixed VALID SV path, with +/- 8 sigma_0 added on a single day.

    An artificial perturbation used to read the update mechanism. It is not part
    of the benchmark and contributes no detection or false-alarm estimate.
    """
    from .ewma import (ewma_gaussian_increments, ewma_student_t_increments,
                       ewma_variance_forecast)

    spec = next(s for s in specs(cfg) if s[0] == "sv_jump")
    cn = scenario_cfg(cfg, spec[1], spec[2])
    idx = cfg.stage2d_shock_path_index
    d = draw_noise("sv_jump", make_streams(cfg)["stage2d_shock"], idx + 1,
                   cfg.horizon_days, cn)
    base = returns_from_noise(d.eps, cfg.sharpe_valid, cfg)[idx].copy()
    day0 = cfg.stage2d_shock_day - 1
    amp = cfg.stage2d_shock_sigmas * cfg.sigma_daily
    variants = {"base": base, "plus": base.copy(), "minus": base.copy()}
    variants["plus"][day0] += amp
    variants["minus"][day0] -= amp

    frames = []
    for name, r in variants.items():
        r2 = r[None, :]
        v, _ = ewma_variance_forecast(r2, cfg)
        gi = ewma_gaussian_increments(r2, cfg, v)[0]
        ti = ewma_student_t_increments(r2, cfg, v)[0]
        frames.append(pd.DataFrame({
            "day": np.arange(1, cfg.horizon_days + 1), "variant": name, "return": r,
            "forecast_variance": v[0], "forecast_var_over_sigma0sq": v[0] / cfg.sigma_daily ** 2,
            "diffusive_variance": diffusive_daily_variance(d, cn)[idx],
            "total_variance_given_vol": total_daily_variance_given_vol(d, cn)[idx],
            "ewma_gaussian_increment": gi, "ewma_student_t_increment": ti,
            "ewma_gaussian_cum": cfg.prior_log_odds + np.cumsum(gi),
            "ewma_student_t_cum": cfg.prior_log_odds + np.cumsum(ti)}))
    return pd.concat(frames, ignore_index=True)

# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

from .evaluate import excludes_zero
from .stage2c import METHODS, first_eligible, path_minima, statistic


def paired_frozen(minima: dict, thresholds: dict, ma: str, mb: str, scenario: str,
                  arm: str, alpha: float, cfg) -> dict:
    """Paired difference on the same test paths, thresholds held fixed.

    Covers TEST sampling only. Realised FARs are carried alongside so a higher
    detection rate bought with a higher false-alarm rate is visible.
    """
    z = cfg.wilson_z
    a_det = minima[ma]["test_invalid"] < thresholds[ma]
    b_det = minima[mb]["test_invalid"] < thresholds[mb]
    a_far = minima[ma]["test_valid"] < thresholds[ma]
    b_far = minima[mb]["test_valid"] < thresholds[mb]
    d = a_det.astype(float) - b_det.astype(float)
    se = d.std(ddof=1) / math.sqrt(d.size)
    lo, hi = float(d.mean() - z * se), float(d.mean() + z * se)
    return {"scenario": scenario, "arm": arm, "far_target": alpha,
            "method_a": ma, "method_b": mb, "n_paired_paths": int(d.size),
            "detect_a": float(a_det.mean()), "detect_b": float(b_det.mean()),
            "far_a": float(a_far.mean()), "far_b": float(b_far.mean()),
            "detect_diff": float(d.mean()), "detect_diff_lo": lo, "detect_diff_hi": hi,
            "detect_excludes_zero": excludes_zero(lo, hi),
            "interval_covers": "test sampling only, thresholds frozen"}


def paired_truncated_time(minima_tau: dict, ma: str, mb: str, cfg, z=None) -> dict:
    """Paired difference in truncated detection time, frozen thresholds."""
    z = cfg.wilson_z if z is None else z
    a, b = minima_tau[ma], minima_tau[mb]
    d = a.astype(float) - b.astype(float)
    se = d.std(ddof=1) / math.sqrt(d.size)
    return {"trunc_time_diff_days": float(d.mean()),
            "trunc_time_diff_lo": float(d.mean() - z * se),
            "trunc_time_diff_hi": float(d.mean() + z * se)}


def run_bootstrap(cfg, diag: dict, minima: dict, pairs, seed_seq) -> pd.DataFrame:
    """Resample the diagnostic calibration block AND the test block.

    Each replicate redraws the scenario's calibration minima, re-picks the
    buffered threshold at the frozen rank, then redraws the test paths. Methods
    share path indices inside a block, so the difference stays paired. This
    describes sampling variability; it is not the finite-sample FAR guarantee.
    """
    rng = np.random.default_rng(seed_seq)
    st = streams_for(cfg, "stage2d_calibration")
    n_cal = cfg.stage2d_calibration_paths
    reps = cfg.stage2d_bootstrap_reps
    out = []

    # calibration minima per (scenario, method), recomputed once
    cal_min = {}
    for spec in specs(cfg):
        key = spec[0]
        blk = make_block(cfg, spec, st[(key, "calibration_valid")], n_cal, cfg.sharpe_valid)
        for m in METHODS:
            cal_min[(key, m)] = path_minima(statistic(m, blk["returns"], cfg),
                                            first_eligible(m, cfg), cfg.horizon_days)
        del blk

    for spec in specs(cfg):
        key = spec[0]
        for alpha in cfg.far_targets:
            k = diag["ranks"][alpha]["buffered"]
            tm = minima[key]
            n_v = tm[METHODS[0]]["test_valid"].size
            n_i = tm[METHODS[0]]["test_invalid"].size
            for ma, mb in pairs:
                d_det = np.empty(reps)
                for r in range(reps):
                    ic = rng.integers(0, n_cal, n_cal)          # shared by both methods
                    ii = rng.integers(0, n_i, n_i)
                    ta = float(np.partition(cal_min[(key, ma)][ic], k - 1)[k - 1])
                    tb = float(np.partition(cal_min[(key, mb)][ic], k - 1)[k - 1])
                    d_det[r] = ((tm[ma]["test_invalid"][ii] < ta).mean()
                                - (tm[mb]["test_invalid"][ii] < tb).mean())
                ta0 = float(np.partition(cal_min[(key, ma)], k - 1)[k - 1])
                tb0 = float(np.partition(cal_min[(key, mb)], k - 1)[k - 1])
                point = float((tm[ma]["test_invalid"] < ta0).mean()
                              - (tm[mb]["test_invalid"] < tb0).mean())
                lo, hi = float(np.quantile(d_det, 0.025)), float(np.quantile(d_det, 0.975))
                out.append({"scenario": key, "arm": "diagnostic", "far_target": alpha,
                            "method_a": ma, "method_b": mb, "n_reps": reps,
                            "detect_diff": point, "detect_diff_lo": lo, "detect_diff_hi": hi,
                            "excludes_zero": excludes_zero(lo, hi),
                            "covers": "calibration + test sampling"})
    return pd.DataFrame(out)

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import setup_style
from .stage2c import LATE_STARTERS, METHODS


C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "known_vol_rolling_252": "#9467bd",
     "ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e"}
SHORT = {"gaussian_ctrl": "Gaussian\n(A=0,k=0)", "sv_ctrl": "SV\n(A=1,k=0)",
         "jump_ctrl": "Jump\n(A=0,k=5)", "sv_jump": "SV+jump\n(A=1,k=5)",
         "sv_jump_big": "SV+big jump\n(A=1,k=8)"}
ARM = {"transfer": "Stage 2C threshold transferred", "diagnostic": "calibrated per scenario"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage2d_metrics.csv")
    sh = pd.read_csv(out_dir / "stage2d_shock.csv")
    q = pd.read_csv(out_dir / "stage2d_failure_probability.csv")
    fd = out_dir / "figures"
    outs = [figure_dgp(cfg, fd / "fig2d1_dgp.png"),
            figure_arms(cfg, m, fd / "fig2d2_arms.png"),
            figure_main(cfg, m, q, fd / "fig2d3_main.png"),
            figure_shock(cfg, sh, fd / "fig2d4_shock.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def figure_dgp(cfg, out: Path) -> Path:
    from .noise import draw_noise
    pass

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    ss = np.random.SeedSequence(20260906)
    cols = plt.cm.viridis(np.linspace(0.08, 0.88, len(specs(cfg))))
    grid = np.linspace(0, 8, 70)
    for col, spec in zip(cols, specs(cfg)):
        key, A, kap, _ = spec
        d = draw_noise("sv_jump", ss, 2000, cfg.horizon_days, scenario_cfg(cfg, A, kap))
        e = d.eps
        axes[0].semilogy(grid, np.maximum([(np.abs(e) > g).mean() for g in grid], 1e-7),
                         color=col, lw=1.6, label=SHORT[key].replace("\n", " "))
        a = np.abs(e)
        c = a - a.mean()
        lags = np.arange(1, 41)
        den = (c ** 2).sum()
        axes[1].plot(lags, [(c[:, :-l] * c[:, l:]).sum() / den for l in lags],
                     color=col, lw=1.6)
        if key in ("jump_ctrl", "sv_jump", "sv_jump_big"):
            k = d.latent["jump_counts"]
            quiet = k == 0
            axes[2].bar(SHORT[key].replace("\n", " "), float(e[quiet].std()), color=col, alpha=0.9)
    axes[2].axhline(1.0, color="k", ls="--", lw=1.0, label="unit unconditional sd")
    for kap, lab in ((5.0, "k=5"), (8.0, "k=8")):
        th = 1 / np.sqrt(1 + kap ** 2 * cfg.noise_jump_lambda_annual / cfg.D)
        axes[2].axhline(th, color="0.4", ls=":", lw=1.0)
    axes[0].set_xlabel("$c$"); axes[0].set_ylabel("$P(|\\varepsilon|>c)$")
    axes[0].set_title("(a) Tail frequency", fontsize=10); axes[0].legend(fontsize=7.2)
    axes[1].set_xlabel("lag (trading days)"); axes[1].set_ylabel("autocorr of $|\\varepsilon_t|$")
    axes[1].set_title("(b) Volatility clustering (pooled)", fontsize=10)
    axes[2].set_ylabel("sd of $\\varepsilon$ on jump-free days")
    axes[2].set_title("(c) Ordinary days get QUIETER as $\\kappa$ grows\n"
                      "dotted: theoretical $1/\\sqrt{1+\\kappa^2\\lambda/D}$", fontsize=10)
    axes[2].legend(fontsize=7.4)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 2D.1  The combined DGP: persistent SV and independent additive jumps\n"
                 "all five settings have zero mean and unit unconditional variance by theoretical constants",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out


def figure_arms(cfg, m, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    scs = [s for s in SHORT]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.0))
    x = np.arange(len(scs)); w = 0.13
    for row, arm in enumerate(("transfer", "diagnostic")):
        for i, meth in enumerate(METHODS):
            far, det = [], []
            for sc in scs:
                r = m[(m.scenario == sc) & (m.arm == arm) & (m.far_target == alpha)
                      & (m.method == meth)]
                far.append(float(r.far_d504.iloc[0]) if len(r) else np.nan)
                det.append(float(r.detect_d504.iloc[0]) if len(r) else np.nan)
            off = (i - (len(METHODS) - 1) / 2) * w
            axes[row, 0].bar(x + off, far, w, color=C[meth], label=LABEL_2D[meth], alpha=0.9)
            axes[row, 1].bar(x + off, det, w, color=C[meth], alpha=0.9)
        axes[row, 0].axhline(alpha, color="k", ls="-.", lw=1.1)
        axes[row, 0].set_ylabel("realised 2y FAR | $S=1$")
        axes[row, 1].set_ylabel("2y detection | $S=0$")
        axes[row, 0].set_title(f"({'ab'[row]}) {ARM[arm]} — false alarms", fontsize=10)
        axes[row, 1].set_title(f"({'cd'[row]}) {ARM[arm]} — detection", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks(x, [SHORT[s] for s in scs], fontsize=7.6)
            ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=6, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Fig 2D.2  Two threshold conditions, five scenarios (nominal $\\alpha$={alpha:g})\n"
                 "the two combined scenarios are OUTSIDE the Stage 2C coverage set, so the transferred "
                 "threshold carries no guarantee there", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90)); fig.savefig(out); plt.close(fig)
    return out


def figure_main(cfg, m, q, out: Path, alpha=None) -> Path:
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    days = list(cfg.stage2c_report_days)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    sub = m[(m.scenario == "sv_jump") & (m.arm == "diagnostic") & (m.far_target == alpha)]
    for meth in METHODS:
        r = sub[sub.method == meth]
        if not len(r):
            continue
        ls = "--" if meth in LATE_STARTERS else "-"
        axes[0].plot(days, [float(r[f"detect_d{d}"].iloc[0]) for d in days], "o" + ls,
                     color=C[meth], lw=1.6, ms=5, label=LABEL_2D[meth])
        axes[1].plot(days, [float(r[f"far_d{d}"].iloc[0]) for d in days], "o" + ls,
                     color=C[meth], lw=1.6, ms=5)
    axes[1].axhline(alpha, color="k", ls="-.", lw=1.1)
    axes[0].set_ylabel("cumulative detection | $S=0$")
    axes[1].set_ylabel("cumulative false alarm | $S=1$")
    axes[0].set_title("(a) SV+jump: detection", fontsize=10)
    axes[1].set_title("(b) SV+jump: realised false alarms", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("trading day"); ax.set_xticks(days); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.2, loc="upper left")

    ax = axes[2]
    qq = q[(q.scenario == "sv_jump") & (q.true_state == "invalid")]
    for meth in ("binary_gaussian", "binary_student_t", "ewma_gaussian", "ewma_student_t"):
        s = qq[qq.method == meth].sort_values("day")
        if not len(s):
            continue
        ax.plot(s.day, s.median_q, "o-", color=C[meth], lw=1.6, ms=5, label=LABEL_2D[meth])
        ax.fill_between(s.day, s.q10, s.q90, color=C[meth], alpha=0.10, lw=0)
    ax.set_xlabel("trading day"); ax.set_xticks(days); ax.set_ylim(0, 1)
    ax.set_ylabel("failure probability $q$ | $S=0$")
    ax.set_title("(c) $q$ on ALL paths, median and 10-90%\n"
                 "(paths continue past any alarm)", fontsize=10)
    ax.legend(fontsize=7.2, loc="upper left"); ax.grid(alpha=0.3)
    fig.suptitle(f"Fig 2D.3  Main combined scenario, per-scenario calibration, $\\alpha$={alpha:g}\n"
                 "dashed markers: the 252d rolling methods cannot start before day 252",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out


def figure_shock(cfg, sh, out: Path) -> Path:
    d0 = cfg.stage2d_shock_day
    lo, hi = d0 - 10, d0 + 90
    cols = {"base": "#333333", "plus": "#1a9850", "minus": "#d73027"}
    lab = {"base": "unshocked", "plus": f"+{cfg.stage2d_shock_sigmas:g}$\\sigma_0$",
           "minus": f"-{cfg.stage2d_shock_sigmas:g}$\\sigma_0$"}
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    for v, c in cols.items():
        s = sh[sh.variant == v]
        w = s[(s.day >= lo) & (s.day <= hi)]
        axes[0].plot(w.day, w.forecast_var_over_sigma0sq, color=c, lw=1.5, label=lab[v])
        axes[1].plot(w.day, w.ewma_gaussian_increment, color=c, lw=1.2, label=lab[v])
        axes[2].plot(w.day, w.ewma_student_t_increment, color=c, lw=1.2)
    base_true = sh[sh.variant == "base"]
    bt = base_true[(base_true.day >= lo) & (base_true.day <= hi)]
    axes[0].plot(bt.day, bt.diffusive_variance / cfg.sigma_daily ** 2, color="0.6", ls=":",
                 lw=1.3, label="latent diffusive var")
    axes[0].plot(bt.day, bt.total_variance_given_vol / cfg.sigma_daily ** 2, color="0.6",
                 ls="-.", lw=1.0, label="latent total var (jump averaged)")
    for ax in axes:
        ax.axvline(d0, color="0.5", lw=1.0); ax.grid(alpha=0.3); ax.set_xlabel("trading day")
    axes[0].set_yscale("log"); axes[0].set_ylabel("$\\hat v_t/\\sigma_0^2$")
    axes[0].set_title(f"(a) Variance forecast after a single shock on day {d0}", fontsize=10)
    axes[0].legend(fontsize=7.4)
    axes[1].set_ylabel("one-step increment"); axes[1].set_title("(b) EWMA Gaussian", fontsize=10)
    axes[1].legend(fontsize=7.4)
    axes[2].set_ylabel("one-step increment"); axes[2].set_title("(c) EWMA Student-t", fontsize=10)
    fig.suptitle("Fig 2D.4  Single-shock diagnostic on ONE pre-fixed valid SV+jump path\n"
                 "an artificial perturbation used to read the update mechanism; NOT part of the benchmark",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out); plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

import math
from pathlib import Path

import pandas as pd

from .stage2c import LATE_STARTERS, METHODS


SC = {"gaussian_ctrl": "高斯对照 (A=0, κ=0)", "sv_ctrl": "SV 对照 (A=1, κ=0)",
      "jump_ctrl": "跳跃对照 (A=0, κ=5)", "sv_jump": "**SV+跳跃 (A=1, κ=5)**",
      "sv_jump_big": "**SV+较大跳跃 (A=1, κ=8)**"}
ARM_REPORT = {"transfer": "第一组：Stage 2C 冻结门槛迁移", "diagnostic": "第二组：按情境独立校准（诊断）"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(r):
    v = r.get("median_detect_days", "")
    return f"{int(v)}d" if v not in ("", None) and v == v else "未达到"


def write_stage2d_report(cfg, summary, metrics, diag, boot, brier, shock, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    days = list(cfg.stage2c_report_days)
    M = metrics
    g = lambda meth, sc, arm, col, a=a_main: float(
        M[(M.scenario == sc) & (M.arm == arm) & (M.far_target == a) & (M.method == meth)][col].iloc[0])

    A("# Stage 2D 报告：持续性随机波动率与孤立跳跃同时存在时的检测")
    A("")
    A("本文件由 `python -m strategy_survivorship.stage2d` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # -------- 1 --------
    A("## 1. 主问题的回答")
    A("")
    sc = "sv_jump"
    et = g("ewma_student_t", sc, "diagnostic", "detect_d504")
    eg = g("ewma_gaussian", sc, "diagnostic", "detect_d504")
    bt = g("binary_student_t", sc, "diagnostic", "detect_d504")
    bg = g("binary_gaussian", sc, "diagnostic", "detect_d504")
    A(f"**在按情境独立校准的诊断条件下（α={a_main:g}，主组合情境 SV+跳跃）**，两年检出率："
      f"EWMA Student-t {et:.4f}、EWMA Gaussian {eg:.4f}、固定 Student-t {bt:.4f}、"
      f"固定 Gaussian {bg:.4f}。相对各自的固定尺度对应方法，差为 "
      f"**{et - bt:+.4f}**（Student-t 家族）与 **{eg - bg:+.4f}**（Gaussian 家族）。")
    A("")
    tet = g("ewma_student_t", sc, "transfer", "detect_d504")
    tft = g("ewma_student_t", sc, "transfer", "far_d504")
    A(f"**在直接迁移 Stage 2C 冻结门槛的条件下**，EWMA Student-t 的两年检出率为 {tet:.4f}，"
      f"但其实测两年 FAR 为 **{tft:.4f}**（名义预算 {a_main:g}）。"
      f"{'**该规则不满足预算，因此其较高的检出率不能称为提升。**' if tft > a_main else '实测 FAR 在预算内。'}")
    A("")
    A("详细分解见第 4–6 节；两个组合情境**不属于** Stage 2C 保证的覆盖范围。")
    A("")

    # -------- 2 --------
    A("## 2. 组合 DGP（运行前固定）")
    A("")
    A("```")
    A("r_t = mu_S + sigma_0 eps_t")
    A("a_t = rho a_{t-1} + sqrt(1-rho^2) xi_t,   a_1 ~ N(0,1)")
    A("v_t = exp(A a_t - A^2/2)")
    A("K_t ~ Poisson(lambda/D)")
    A("eps_t = ( sqrt(v_t) z_t + kappa sqrt(K_t) w_t ) / sqrt(1 + kappa^2 lambda/D)")
    A("```")
    A("")
    A(f"`xi, z, w, K` 的基础随机来源相互独立；`rho = {cfg.noise_sv_rho:g}`、"
      f"`lambda = {cfg.noise_jump_lambda_annual:g}`/年、`D = {cfg.D}`、`H = {cfg.horizon_days}`、"
      f"年化波动率基准 {cfg.sigma_annual:g}，有效/无效 Sharpe 为 1/0，**平均收益不随波动改变**。")
    A("")
    A("**跳跃是独立加性成分**：它的幅度**不**乘以当天的背景波动，"
      "所以落在平静日上的跳跃相对更大。这是本轮要检验的设计选择。")
    A("")
    A("| 情境 | SV 振幅 A | 跳跃幅度 κ | 在 Stage 2C 覆盖集内 |")
    A("|---|---|---|---|")
    for s in summary["scenarios"]:
        A(f"| {SC[s['key']]} | {s['A']:g} | {s['kappa']:g} | "
          f"{'是' if s['in_stage2c_coverage'] else '**否**'} |")
    A("")
    A("**归一化只用理论常数**：`a_t` 边际为 `N(0,1)`，故 `E[v_t] = exp(A^2/2 - A^2/2) = 1`；"
      "两个分子项独立，方差分别为 1 与 `kappa^2 lambda/D`，因此对**任意** `(A, kappa)` 都有 "
      "`E[eps]=0`、`Var(eps)=1`。**不做任何按整条路径的样本均值或方差标准化。**")
    A("")
    quiet = {k: 1 / math.sqrt(1 + k ** 2 * cfg.noise_jump_lambda_annual / cfg.D) for k in (0, 5, 8)}
    A(f"> **κ 越大不等于任务必然更难。** 单位方差常数同时压低了无跳跃日的扩散尺度。"
      f"**跨全部无跳跃日的总体尺度**为 σ₀/c（因 E[v]=1）：κ=0 时 {quiet[0]:.4f}、"
      f"κ=5 时 {quiet[5]:.4f}、κ=8 时 {quiet[8]:.4f}。"
      "也就是说 κ=8 的普通交易日整体上比 κ=0 **更安静**，单日信噪比反而更高。"
      "**但这是总体尺度，不是某个具体潜在状态下的尺度**——"
      "给定 v_t 时的条件尺度是 `σ₀√v_t/c`，在 A=1 下其 90/10 分位比超过 3。"
      "本轮报告不预设 κ 的单调效应。")
    A("")
    A("**两个潜在方差的命名与定义**（评分与诊断用，**任何检测器都看不到，也都不是实时预测**）：")
    A("")
    A("```")
    A("扩散方差（给定 v_t 且当日无跳跃）      : sigma_0^2 * v_t / c^2")
    A("给定 v_t、未观察当日跳跃的总方差        : sigma_0^2 * (v_t + kappa^2 lambda/D) / c^2")
    A("c^2 = 1 + kappa^2 lambda/D")
    A("```")
    A("")
    A("> **已修正**：上一版的 `true_daily_sigma` 返回 `sigma_0 sqrt(v_t)`，"
      "漏掉了组合归一化 `c`，也没有区分是否包含跳跃风险。"
      "已逐一检查全部调用点：该量在 Stage 2D 中**只写不读**，"
      "唯一的消费者是单次冲击诊断里的一列与图 2D.4 的一条参考线；"
      "Stage 2A/2B 使用无跳跃的 SV 生成器（c=1），其 oracle 输入与全部数值**不受影响**（已重跑验证）。"
      "**因此主分数、oracle、概率评分均保持原值，只有诊断被修正。**")
    A("")
    A(f"每个情境：{cfg.stage2d_calibration_paths} 条有效校准路径、"
      f"{cfg.stage2d_test_paths} 条有效测试路径、{cfg.stage2d_test_paths} 条无效测试路径；"
      "校准与测试独立，同一数据块内各检测器共用收益路径。"
      "检测器只接收收益与公共模型配置，**不获得真实状态、潜在波动、跳跃标记或情境参数**。")
    A("")

    # -------- 3 --------
    A("## 3. 两个比较条件")
    A("")
    A(f"**第一组（门槛迁移）**：直接读取 Stage 2C 冻结的统一门槛"
      f"（来源 `{summary['transfer_thresholds_source']}`），五种情境中原样使用，"
      "**不根据本轮测试结果调整**。三个对照对应既有覆盖分布；两个组合情境不属于原保证范围。")
    A("")
    A(f"**第二组（诊断校准）**：复用同一校准误差缓冲方法，分别为五种情境校准门槛"
      f"（J = {summary['diagnostic_J']}，秩 "
      + "、".join(f"α={float(a):g}→k={v['buffered']}"
                   for a, v in summary["diagnostic_ranks"].items())
      + "）。")
    A("")
    A("> **第二组具有环境信息**：它知道自己身处哪个情境。它的作用是判断第一组的变化"
      "来自门槛迁移还是模型在该环境中的检测能力，**不能包装成未知环境下的部署结果**。")
    A("")
    A("> **保证范围要分开说。** 两个组合情境**不在** Stage 2C 迁移保证的覆盖集内，"
      "所以第一组在这两个情境中没有概率保证。"
      "但第二组是**按本轮五个情境自己校准**的，它有**自己的**有限样本保证："
      "在这五种路径生成规律及其假设下，以至少 "
      f"{1-cfg.stage2c_delta:.0%} 的概率同时把各自的真实两年 FAR 控制在目标以内。"
      "**不能笼统说所有规则都没有保证。**")
    A("")

    # -------- 4 --------
    for arm in ("transfer", "diagnostic"):
        n = 4 if arm == "transfer" else 5
        A(f"## {n}. {ARM_REPORT[arm]}")
        A("")
        for alpha in cfg.far_targets:
            tag = "主结果" if alpha == a_main else "敏感性"
            A(f"### {n}.{list(cfg.far_targets).index(alpha)+1} α = {alpha:g}（{tag}）")
            A("")
            sub = M[(M.arm == arm) & (M.far_target == alpha)].to_dict("records")
            L.extend(_t(sub,
                        ["情境", "方法", "两年实测 FAR", "FAR 95%", "超预算"]
                        + [f"检出 {d}d" for d in days]
                        + ["截断均检(日)", "中位检测", "两年未检出"],
                        lambda r: [SC[r["scenario"]], LABEL_2D[r["method"]],
                                   f"{r['far_d504']:.4f}",
                                   f"[{r['far_lo']:.4f}, {r['far_hi']:.4f}]",
                                   "**是**" if r["far_d504"] > alpha else "否"]
                                  + [f"{r[f'detect_d{d}']:.4f}" for d in days]
                                  + [f"{r['trunc_mean_detect_days']:.0f}", _med(r),
                                     f"{r['undetected_at_H']:.4f}"]))
            A("")
        if arm == "transfer":
            A(f"> 第 {days[0]} 天与第 {days[1]} 天，两个 252 日滚动方法的检出率与误杀率"
              f"结构性为 0：它们要到第 {cfg.rolling_window} 天才能启动，这是窗口设置而非性能。")
            A("")

    # -------- 6 --------
    A("## 6. 主要比较与不确定性")
    A("")
    A("在两个组合情境中，运行前预先指定三组比较："
      "EWMA Student-t 对固定 Student-t、EWMA Gaussian 对固定 Gaussian、两个 EWMA 之间。")
    A("")
    A("**冻结门槛的配对区间**（同一批测试路径，仅覆盖测试抽样），并列实测 FAR：")
    A("")
    pr = [r for r in summary["paired"] if r["far_target"] == a_main]
    L.extend(_t(pr, ["情境", "条件", "A", "B", "A 检出", "B 检出", "A FAR", "B FAR",
                     "检出差", "95% 区间", "排除零"],
                lambda r: [SC[r["scenario"]], "迁移" if r["arm"] == "transfer" else "诊断",
                           LABEL_2D[r["method_a"]], LABEL_2D[r["method_b"]],
                           f"{r['detect_a']:.4f}", f"{r['detect_b']:.4f}",
                           f"{r['far_a']:.4f}", f"{r['far_b']:.4f}",
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "是" if r["detect_excludes_zero"] else "否"]))
    A("")
    A(f"（α={a_alt:g} 档见 `stage2d_paired.csv`。）")
    A("")
    A(f"**含校准抽样的敏感性区间**（{cfg.stage2d_bootstrap_reps} 次；每次重采样该情境的诊断校准块、"
      "按冻结的秩重新取门槛，再重采样测试块；块内各方法共用路径索引）：")
    A("")
    bb = boot[(boot.far_target == a_main) & (boot.scenario.isin(COMBINED))].to_dict("records")
    L.extend(_t(bb, ["情境", "A", "B", "检出差", "95% 敏感性区间", "排除零"],
                lambda r: [SC[r["scenario"]], LABEL_2D[r["method_a"]], LABEL_2D[r["method_b"]],
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("> **两类区间覆盖的随机性不同**：配对区间只覆盖测试抽样，敏感性区间还包含选门槛这一步。"
      "两者都是逐点区间，不构成所有比较同时成立的保证。")
    A("")
    A("**两个 EWMA 方法的配对 Brier 差异**（EWMA Student-t 减 EWMA Gaussian；"
      "有效/无效两组分别重采样，保持 50/50 评价权重）：")
    A("")
    L.extend(_t(brier.to_dict("records"),
                ["情境", "天数", "Brier(EWMA t)", "Brier(EWMA G)", "差", "95% 区间", "排除零"],
                lambda r: [SC[r["scenario"]], str(int(r["day"])), f"{r['brier_a']:.5f}",
                           f"{r['brier_b']:.5f}", f"{r['diff']:+.5f}",
                           f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("> Brier 改善**不等同于**所有概率区间都校准准确；它是一个综合评分，"
      "分箱可靠性需要单独看。")
    A("")

    # -------- 7 --------
    A("## 7. 全部路径的失效概率分布")
    A("")
    A("报警后仍继续计算原始分数用于概率诊断，但**报警规则仍按第一次穿越计分**。")
    A("")
    fp = [r for r in summary["failure_probability"] if r["scenario"] == "sv_jump"]
    L.extend(_t(sorted(fp, key=lambda r: (r["method"], r["true_state"], r["day"])),
                ["方法", "真实状态", "天数", "均值 q", "中位 q", "10%", "90%"],
                lambda r: [LABEL_2D[r["method"]], "有效" if r["true_state"] == "valid" else "无效",
                           str(int(r["day"])), f"{r['mean_q']:.4f}", f"{r['median_q']:.4f}",
                           f"{r['q10']:.4f}", f"{r['q90']:.4f}"]))
    A("")
    A("（上表为主组合情境；全部情境见 `stage2d_failure_probability.csv`。）")
    A("")

    # -------- 8 --------
    A("## 8. 单次冲击诊断")
    A("")
    d0 = cfg.stage2d_shock_day
    A(f"在**一条预先固定的有效 SV+跳跃路径**（路径编号 {cfg.stage2d_shock_path_index}）上，"
      f"于第 {d0} 日分别加入 ±{cfg.stage2d_shock_sigmas:g}σ₀ 的单次收益冲击，"
      "保留未修改路径作对照；其余各日收益完全相同。")
    A("")
    A("> **这是人为扰动诊断，不属于主 benchmark**，不贡献任何检出率或误杀率估计。")
    A("")
    b = shock[shock.variant == "base"].set_index("day")
    p_ = shock[shock.variant == "plus"].set_index("day")
    mn = shock[shock.variant == "minus"].set_index("day")
    A("| 量 | 未修改 | +8σ₀ | −8σ₀ |")
    A("|---|---|---|---|")
    A(f"| 冲击当日 EWMA Gaussian 增量 | {b.ewma_gaussian_increment[d0]:+.5f} | "
      f"{p_.ewma_gaussian_increment[d0]:+.5f} | {mn.ewma_gaussian_increment[d0]:+.5f} |")
    A(f"| 冲击当日 EWMA Student-t 增量 | {b.ewma_student_t_increment[d0]:+.5f} | "
      f"{p_.ewma_student_t_increment[d0]:+.5f} | {mn.ewma_student_t_increment[d0]:+.5f} |")
    for h in (1, 5, 21, 63):
        d = d0 + h
        if d in b.index:
            A(f"| 第 {d0}+{h} 日预测方差 / σ₀² | {b.forecast_var_over_sigma0sq[d]:.4f} | "
              f"{p_.forecast_var_over_sigma0sq[d]:.4f} | {mn.forecast_var_over_sigma0sq[d]:.4f} |")
    A(f"| 第 {cfg.horizon_days} 日累计证据 (Gaussian) | {b.ewma_gaussian_cum.iloc[-1]:+.4f} | "
      f"{p_.ewma_gaussian_cum.iloc[-1]:+.4f} | {mn.ewma_gaussian_cum.iloc[-1]:+.4f} |")
    A(f"| 第 {cfg.horizon_days} 日累计证据 (Student-t) | {b.ewma_student_t_cum.iloc[-1]:+.4f} | "
      f"{p_.ewma_student_t_cum.iloc[-1]:+.4f} | {mn.ewma_student_t_cum.iloc[-1]:+.4f} |")
    A("")
    lam = cfg.ewma_lambda
    extra = (1 - lam) * (cfg.stage2d_shock_sigmas * cfg.sigma_daily) ** 2 / cfg.sigma_daily ** 2
    hl = math.log(0.5) / math.log(lam)
    A(f"**验算冲击引入的方差预测差异如何衰减**：冲击把当日平方残差抬高，"
      f"次日预测方差因此多出约 `(1-λ)·(8σ₀)²/σ₀² = {extra:.3f}`（以 σ₀² 为单位，忽略中点项），"
      f"此后按 λ={lam:g} 每日衰减，半衰期 `ln0.5/lnλ = {hl:.1f}` 个交易日。"
      "实测差值见上表与 `stage2d_shock.csv`。")
    A("")
    A("Student-t 在**冲击当日**把证据增量压得比 Gaussian 小得多，"
      "但**随后各日**它与 Gaussian 共用同一个被抬高的方差预测，因此同样受到影响。"
      "**本轮不把这一现象写成主要性能瓶颈**：单条人为扰动路径不足以支持那样的结论。")
    A("")

    # -------- 8.5 follow-up --------
    A("## 9. 追加证据（follow-up）")
    A("")
    A("> 以下为 Stage 2E 阶段追加，**不是运行前预先指定的发现**；使用的是同一批 Stage 2D 数据。")
    A("")
    A("### 9.1 截断检测时间的配对差异（冻结门槛，仅覆盖测试抽样）")
    A("")
    tt = [r for r in summary.get("paired_time", []) if r["far_target"] == a_main]
    if tt:
        L.extend(_t(tt, ["情境", "条件", "A", "B", "A 截断均检", "B 截断均检", "差(日)", "95% 区间"],
                    lambda r: [SC[r["scenario"]], "迁移" if r["arm"] == "transfer" else "诊断",
                               LABEL_2D[r["method_a"]], LABEL_2D[r["method_b"]],
                               f"{r['trunc_mean_a']:.1f}", f"{r['trunc_mean_b']:.1f}",
                               f"{r['trunc_time_diff_days']:+.2f}",
                               f"[{r['trunc_time_diff_lo']:+.2f}, {r['trunc_time_diff_hi']:+.2f}]"]))
        A("")
        A(f"（α={a_alt:g} 档见 `stage2d_paired_time.csv`。）")
        A("")
    A("### 9.2 配对 Brier：EWMA Student-t 对固定 Student-t")
    A("")
    fb = [r for r in summary["brier"] if r.get("followup")]
    if fb:
        L.extend(_t(fb, ["情境", "天数", "Brier(EWMA t)", "Brier(固定 t)", "差", "95% 区间", "排除零"],
                    lambda r: [SC[r["scenario"]], str(int(r["day"])), f"{r['brier_a']:.5f}",
                               f"{r['brier_b']:.5f}", f"{r['diff']:+.5f}",
                               f"[{r['lo']:+.5f}, {r['hi']:+.5f}]",
                               "是" if r["excludes_zero"] else "否"]))
        A("")
    A("> **Brier 改善与概率校准改善是两件事**：Brier 是分辨力与校准的合成评分，"
      "分箱可靠性单独见 `stage2d_reliability.csv`（含每箱样本数）。"
      "全部概率诊断使用**所有路径**，不因报警而删去后续概率。")
    A("")

    # -------- 9 --------
    A("## 10. 复现")
    A("")
    A("```bash")
    A(".venv/bin/python -m pytest")
    A(".venv/bin/python -m strategy_survivorship.stage2d                # 正式实验")
    A(".venv/bin/python -m strategy_survivorship.stage2d --figures-only # 仅重绘")
    A("```")
    A("")
    A("`--figures-only` 从磁盘上的 CSV 重绘，**不会重跑实验**，"
      "因此调整图例不可能覆盖正式的重采样结果。")
    A("")
    e = summary["environment"]
    A(f"Python {e['python']} / NumPy {e['numpy']} / SciPy {e['scipy']}；"
      f"本次耗时 {summary['elapsed_s']:.1f} s。")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


from .config import DEFAULT
from .evaluate import excludes_zero

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2c import METHODS


MAIN_PAIRS = (("ewma_student_t", "binary_student_t"),
              ("ewma_gaussian", "binary_gaussian"),
              ("ewma_student_t", "ewma_gaussian"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2D: SV and jumps together.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw from the CSVs already on disk; never re-runs the experiment")
    ap.add_argument("--stage2c-thresholds", type=Path,
                    default=Path("outputs/stage2c_unified_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2d_calibration_paths=800, stage2d_test_paths=800,
                 stage2d_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2d") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        # Plotting is deliberately separable so a legend tweak can never overwrite
        # a formal resampling result.
        setup_style()
        figs = draw_all(cfg, out_dir)
        print("figures redrawn from disk:", figs)
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)

    transfer = load_frozen_thresholds(cfg, args.stage2c_thresholds)
    status("Stage 2C unified thresholds loaded", n=len(transfer),
           source=str(args.stage2c_thresholds))

    diag = calibrate_diagnostic(cfg)
    diag["table"].to_csv(out_dir / "stage2d_thresholds.csv", index=False)
    status("diagnostic thresholds frozen", scenarios=len(specs(cfg)),
           paths=cfg.stage2d_calibration_paths, J=diag["J"])

    rows, minima, qrows, blocks, trunc = [], {}, [], {}, {}
    tst = make_streams(cfg)["stage2d_test"].spawn(len(specs(cfg)))
    for spec, kid in zip(specs(cfg), tst):
        rr, mm, qq, bv, bi, tt = evaluate(cfg, spec, kid, transfer, diag, days)
        rows.extend(rr); minima[spec[0]] = mm; qrows.extend(qq)
        blocks[spec[0]] = (bv, bi); trunc[spec[0]] = tt
        status(f"scenario tested: {spec[0]}", rows=len(rr))

    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2d_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage2d_failure_probability.csv", index=False)

    # ---- paired differences under the frozen thresholds --------------------- #
    paired = []
    for sc in COMBINED:
        for a in cfg.far_targets:
            for arm in ("transfer", "diagnostic"):
                thr = {m: float(metrics[(metrics.scenario == sc) & (metrics.arm == arm)
                                        & (metrics.far_target == a)
                                        & (metrics.method == m)].threshold.iloc[0])
                       for m in METHODS}
                for ma, mb in MAIN_PAIRS:
                    paired.append(paired_frozen(minima[sc], thr, ma, mb, sc, arm, a, cfg))
    pd.DataFrame(paired).to_csv(out_dir / "stage2d_paired.csv", index=False)

    # follow-up: paired TRUNCATED DETECTION TIME differences, frozen thresholds
    tt_rows = []
    for sc in COMBINED:
        for a in cfg.far_targets:
            for arm in ("transfer", "diagnostic"):
                tau = {m: trunc[sc][(m, arm, a)] for m in METHODS}
                for ma, mb in MAIN_PAIRS:
                    r = paired_truncated_time(tau, ma, mb, cfg)
                    r.update({"scenario": sc, "arm": arm, "far_target": a,
                              "method_a": ma, "method_b": mb,
                              "trunc_mean_a": float(tau[ma].mean()),
                              "trunc_mean_b": float(tau[mb].mean()),
                              "n_paired_paths": int(tau[ma].size),
                              "interval_covers": "test sampling only, thresholds frozen",
                              "followup": True})
                    tt_rows.append(r)
    pd.DataFrame(tt_rows).to_csv(out_dir / "stage2d_paired_time.csv", index=False)
    status("paired (frozen threshold) differences done",
           detect_rows=len(paired), time_rows=len(tt_rows))

    # ---- bootstrap including calibration resampling ------------------------- #
    boot = run_bootstrap(cfg, diag, minima, MAIN_PAIRS,
                         make_streams(cfg)["stage2d_bootstrap"])
    boot.to_csv(out_dir / "stage2d_bootstrap.csv", index=False)
    status("bootstrap done", comparisons=len(boot), reps=cfg.stage2d_bootstrap_reps)

    # ---- paired Brier between the two EWMA methods -------------------------- #
    from scipy.special import expit

    from .probability_time import brier_and_reliability
    from .stage2b import paired_brier_difference
    from .stage2c import statistic as stat2c

    BRIER_PAIRS = (("ewma_student_t", "ewma_gaussian", False),
                   ("ewma_student_t", "binary_student_t", True))   # True -> follow-up
    brier, rel = [], []
    bk = make_streams(cfg)["stage2d_bootstrap"].spawn(
        len(COMBINED) * len(days) * len(BRIER_PAIRS))
    i = 0
    for sc in COMBINED:
        bv, bi = blocks[sc]
        q = {}
        for m in ("ewma_student_t", "ewma_gaussian", "binary_student_t"):
            q[(m, "inv")] = expit(-stat2c(m, bi["returns"], cfg))
            q[(m, "val")] = expit(-stat2c(m, bv["returns"], cfg))
            for d in days:
                s_, tab = brier_and_reliability(q[(m, "inv")], q[(m, "val")], d, m)
                tab["scenario"] = sc
                rel.append(tab)
        for ma, mb, fu in BRIER_PAIRS:
            for d in days:
                r = paired_brier_difference(q[(ma, "inv")], q[(ma, "val")],
                                            q[(mb, "inv")], q[(mb, "val")], d,
                                            cfg.stage2d_bootstrap_reps, bk[i]); i += 1
                r.update({"scenario": sc, "a": ma, "b": mb, "followup": fu})
                brier.append(r)
        del q
    pd.DataFrame(brier).to_csv(out_dir / "stage2d_brier.csv", index=False)
    pd.concat(rel, ignore_index=True).to_csv(out_dir / "stage2d_reliability.csv", index=False)
    status("paired Brier and reliability done", brier_rows=len(brier))

    shock = shock_diagnostic(cfg)
    shock.to_csv(out_dir / "stage2d_shock.csv", index=False)
    status("shock diagnostic done", days=int(shock.day.max()))

    figures = []
    if not args.no_figures:
        setup_style()
        figures = draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2D",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "scenarios": [{"key": s[0], "A": s[1], "kappa": s[2], "in_stage2c_coverage": s[3]}
                      for s in specs(cfg)],
        "sample_sizes": {"calibration_valid": cfg.stage2d_calibration_paths,
                         "test_valid": cfg.stage2d_test_paths,
                         "test_invalid": cfg.stage2d_test_paths, "days": cfg.horizon_days},
        "transfer_thresholds_source": str(args.stage2c_thresholds),
        "transfer_thresholds": {f"{k[0]}|alpha={k[1]:g}": v for k, v in transfer.items()},
        "diagnostic_J": diag["J"], "diagnostic_ranks": {str(a): v for a, v in diag["ranks"].items()},
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": paired,
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "brier": brier,
        "paired_time": tt_rows,
        "failure_probability": qrows,
        "report_days": list(days),
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage2d_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2d_report(cfg, summary, metrics, diag, boot, pd.DataFrame(brier), shock,
                         out_dir / "stage2d_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
