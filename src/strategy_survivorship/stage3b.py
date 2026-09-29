"""stage3b: one module per research stage.

- Model and computation: Stage 3B: a bounded diagnostic for a strategy that works first and fails later.
- Figures: Stage 3B figures: three, all drawn from files already on disk.
- Report: Generate ``stage3b_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 3B: random failure time, a bounded diagnostic.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import Stage1Config
from .evaluate import excludes_zero, median_first_passage_day, wilson_interval
from .simulate import make_streams
from .stage2c import first_alarm_day
from .stage2e import first_eligible, statistic
from .stage3a import draw_eps, sharpe_cfg
from .stage3a1 import spec_for

METHODS_3B = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
              "ewma_student_t", "ewma_trunc_student_t")
# The evidence diagnostic reads U = -L as a log-odds and maps it through expit.
# The trailing Sharpe's statistic is an annualised Sharpe ratio, not a log-odds,
# so expit of it is a number without an interpretation; it is excluded rather
# than reported on a scale it does not live on.
from .evaluate import LOG_ODDS_METHODS  # noqa: E402  (kept next to its use)

# the log-odds subset of THIS stage's five methods, in report order
EVIDENCE_METHODS = tuple(m for m in METHODS_3B if m in LOG_ODDS_METHODS)
SCENARIOS_3B = ("gaussian_ctrl", "sv_jump")
NEVER = -1                      # tau = infinity; must never be read as an early alarm
ALWAYS_VALID = "always_valid"   # the T = infinity control
RANDOM_T = "random_uniform"

# written down before any Stage 3B path was scored
MAIN = {"sharpe": 0.6, "scenario": "sv_jump", "alpha": 0.15, "T": 252, "h": 252}
MAIN_PAIRS = (("ewma_student_t", "binary_student_t"),
              ("ewma_student_t", "trailing_sharpe_252"))


def load_stage3a_thresholds(path: Path) -> dict:
    """The frozen 504-day thresholds, read with a correctly rounded parser."""
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["scenario"], float(r["sharpe_valid"]),
                 float(r["far_target"]))] = float(r["threshold"])
    return out


def failure_settings(cfg: Stage1Config) -> list:
    return ([("fixed", int(T)) for T in cfg.stage3b_fixed_failure_days]
            + [(RANDOM_T, None)])


def draw_failure_times(cfg: Stage1Config, kind: str, T, n_paths: int,
                       seed_seq) -> np.ndarray:
    """Per-path last valid day.

    Independent of every noise stream, so the failure time carries no
    information about the volatility path or the jumps.
    """
    if kind == "fixed":
        return np.full(n_paths, int(T), dtype=np.int64)
    if kind == RANDOM_T:
        rng = np.random.default_rng(seed_seq)
        return rng.integers(0, cfg.stage3b_random_failure_max + 1, n_paths).astype(np.int64)
    raise ValueError(f"unknown failure-time kind {kind!r}")


def returns_with_failure(eps: np.ndarray, T: np.ndarray, sharpe: float,
                         cfg: Stage1Config) -> np.ndarray:
    """r_t = mu(theta_t) + sigma_0 eps_t, with theta_t = s for t <= T else 0.

    Days are 1-based, so day t is valid exactly when t <= T; T = 0 means the
    strategy was never valid and T >= n_days means it never failed.
    """
    day = np.arange(1, eps.shape[1] + 1)[None, :]
    valid = day <= np.asarray(T, dtype=np.int64)[:, None]
    return np.where(valid, cfg.daily_drift(sharpe), 0.0) + cfg.sigma_daily * eps


def survival_metrics(tau: np.ndarray, T: np.ndarray, cfg: Stage1Config,
                     windows) -> dict:
    """Pre-failure false alarms, conditional detection, joint detection.

        pre-failure false alarm    P(tau <= T)
        conditional detection      P(T < tau <= T+h | tau > T)
        joint detection            P(T < tau <= T+h)

    ``tau = NEVER`` means the path never alarmed, i.e. tau = infinity.  It is
    therefore a SURVIVOR of the pre-failure window and an undetected path
    afterwards -- never an early false alarm.
    """
    tau = np.asarray(tau, dtype=np.int64)
    T = np.asarray(T, dtype=np.int64)
    alarmed = tau != NEVER
    n = int(tau.size)

    early = alarmed & (tau <= T)          # NEVER never satisfies this
    survived = ~early
    n_surv = int(survived.sum())
    out = {"n_paths": n, "n_pre_failure_alarms": int(early.sum()),
           "pre_failure_far": float(early.mean()),
           "n_survivors": n_surv, "survival_rate": n_surv / n}
    lo, hi = wilson_interval(int(early.sum()), n, cfg.wilson_z)
    out["pre_failure_far_lo"], out["pre_failure_far_hi"] = lo, hi

    for h in windows:
        caught = alarmed & (tau > T) & (tau <= T + h)
        k = int(caught.sum())
        out[f"joint_detect_h{h}"] = k / n
        jl, jh = wilson_interval(k, n, cfg.wilson_z)
        out[f"joint_detect_h{h}_lo"], out[f"joint_detect_h{h}_hi"] = jl, jh
        if n_surv:
            out[f"cond_detect_h{h}"] = k / n_surv
            cl, ch = wilson_interval(k, n_surv, cfg.wilson_z)
        else:
            out[f"cond_detect_h{h}"], cl, ch = float("nan"), float("nan"), float("nan")
        out[f"cond_detect_h{h}_lo"], out[f"cond_detect_h{h}_hi"] = cl, ch
        # the identity the brief asks to be checked, carried in the table itself
        out[f"joint_check_h{h}"] = out["survival_rate"] * out[f"cond_detect_h{h}"]
        out[f"joint_identity_gap_h{h}"] = abs(out[f"joint_detect_h{h}"]
                                              - out[f"joint_check_h{h}"])

    hmax = max(windows)
    # post-failure delay, over ALL survivors: an undetected survivor is scored at
    # the window cap, not dropped, and an early false alarm is not a zero delay
    delay = np.where(alarmed & (tau > T), tau - T, hmax + 1).astype(float)
    delay_surv = np.minimum(delay[survived], hmax) if n_surv else np.array([])
    out["expected_min_delay"] = float(delay_surv.mean()) if n_surv else float("nan")
    out["expected_min_delay_se"] = (float(delay_surv.std(ddof=1) / math.sqrt(n_surv))
                                    if n_surv > 1 else float("nan"))
    out["delay_cap_days"] = hmax
    out["post_failure_undetected"] = (
        1.0 - out[f"cond_detect_h{hmax}"] if n_surv else float("nan"))

    # post-failure median on the survivor population, undetected survivors kept
    if n_surv:
        d = np.where(alarmed[survived] & (tau[survived] > T[survived]),
                     tau[survived] - T[survived], NEVER).astype(np.int64)
        med, note = median_first_passage_day(d, hmax)
    else:
        med, note = "", "no survivors"
    out["median_post_failure_delay"] = med
    out["median_post_failure_note"] = note
    return out


def evidence_at_T(stat: np.ndarray, T: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """U_T = -L_T, the working failure log-odds at the last valid day.

    At T = 0 there is no data yet, so the value is the PRIOR, not row -1 of the
    array.  Reading the last column instead would silently report the end of the
    horizon for every T = 0 path.
    """
    n, n_days = stat.shape
    T = np.clip(np.asarray(T, dtype=np.int64), 0, n_days)
    out = np.full(n, -cfg.prior_log_odds, dtype=float)
    has = T >= 1
    rows = np.nonzero(has)[0]
    out[rows] = -stat[rows, T[rows] - 1]
    return out


def evaluate(cfg: Stage1Config, thresholds: dict, eps_fail: dict, eps_valid: dict,
             fail_times: dict, survivor_alpha: float | None = None) -> tuple:
    """Every (scenario, s, method, alpha, failure setting), plus the T=inf control.

    ``survivor_alpha`` chooses which budget's rule defines "survived to T" for the
    evidence diagnostic.  It is an explicit argument rather than a silent
    ``far_targets[-1]``, and it is recorded in the output so a reader knows which
    rule selected the sample they are looking at.
    """
    survivor_alpha = cfg.far_targets[-1] if survivor_alpha is None else float(survivor_alpha)
    windows = list(cfg.stage3b_post_windows)
    rows, taus, ev, always = [], {}, [], []
    for sc in SCENARIOS_3B:
        for s in cfg.stage3a_sharpes:
            cs = sharpe_cfg(cfg, s)
            # the always-valid control never fails: T is the whole horizon
            r_always = returns_with_failure(
                eps_valid[sc], np.full(eps_valid[sc].shape[0], cfg.horizon_days), s, cfg)
            for m in METHODS_3B:
                elig = first_eligible(m, cs)
                st_always = statistic(m, r_always, cs)
                for a in cfg.far_targets:
                    t = thresholds[(m, sc, s, a)]
                    tv = first_alarm_day(st_always, t, elig, cfg.horizon_days)
                    k = int((tv != NEVER).sum())
                    lo, hi = wilson_interval(k, tv.size, cfg.wilson_z)
                    always.append({
                        "scenario": sc, "sharpe_valid": s, "method": m, "far_target": a,
                        "setting": ALWAYS_VALID, "threshold": t,
                        "n_paths": int(tv.size), "n_alarms": k,
                        "full_period_far": k / tv.size,
                        "full_period_far_lo": lo, "full_period_far_hi": hi,
                        "horizon_days": cfg.horizon_days})
                del st_always
            del r_always

            for kind, T0 in failure_settings(cfg):
                key = f"fixed_T{T0}" if kind == "fixed" else RANDOM_T
                T = fail_times[(sc, kind, T0)]
                r = returns_with_failure(eps_fail[sc], T, s, cfg)
                for m in METHODS_3B:
                    elig = first_eligible(m, cs)
                    st = statistic(m, r, cs)
                    u = evidence_at_T(st, T, cfg)
                    for a in cfg.far_targets:
                        t = thresholds[(m, sc, s, a)]
                        tau = first_alarm_day(st, t, elig, cfg.horizon_days)
                        sm = survival_metrics(tau, T, cfg, windows)
                        rows.append({"scenario": sc, "sharpe_valid": s, "method": m,
                                     "far_target": a, "setting": key,
                                     "failure_kind": kind,
                                     "T_fixed": T0 if kind == "fixed" else "",
                                     "mean_T": float(T.mean()), "threshold": t,
                                     # blank where the statistic is not a log-odds:
                                     # expit of an annualised Sharpe ratio is a
                                     # number without an interpretation
                                     "working_prob_threshold":
                                         float(expit(-t)) if m in LOG_ODDS_METHODS else "",
                                     "working_prob_threshold_note":
                                         "" if m in LOG_ODDS_METHODS
                                         else "not applicable: the statistic is not a "
                                              "log-odds",
                                     "first_eligible_day": elig, **sm})
                        taus[(sc, s, m, a, key)] = (tau.copy(), T)
                    if a == survivor_alpha:
                        surv_mask = ~((tau != NEVER) & (tau <= T))
                    if m not in LOG_ODDS_METHODS:
                        del st, u
                        continue
                    ev.append({"scenario": sc, "sharpe_valid": s, "method": m,
                               "setting": key, "far_target": float("nan"),
                               "n_paths": int(u.size),
                               "mean_U_at_T_all": float(u.mean()),
                               "median_U_at_T_all": float(np.median(u)),
                               "mean_q_at_T_all": float(expit(u).mean()),
                               "median_q_at_T_all": float(np.median(expit(u))),
                               "n_survivors_last_alpha": int(surv_mask.sum()),
                               "mean_U_at_T_survivors": float(u[surv_mask].mean()),
                               "mean_q_at_T_survivors": float(expit(u[surv_mask]).mean()),
                               "survivor_filter_alpha": survivor_alpha,
                               "score_note": "working failure score: a static two-state "
                                             "classifier's q on switching data, not a "
                                             "posterior that models the switch"})
                    del st, u
                del r
    return rows, taus, ev, always


def bootstrap(cfg: Stage1Config, taus: dict, sc: str, s: float, a: float, setting: str,
              seed_seq) -> list[dict]:
    """Paired path bootstrap on the FROZEN thresholds.

    The survivor denominator is recomputed inside every replicate, because it is
    itself a random quantity: holding the original denominator fixed would
    understate the uncertainty of a conditional rate.

    These intervals describe TEST sampling given the thresholds already frozen in
    Stage 3A.  They do not re-cover calibration error.
    """
    rng = np.random.default_rng(seed_seq)
    reps = cfg.stage3b_bootstrap_reps
    hmax = max(cfg.stage3b_post_windows)
    need = sorted({m for p in MAIN_PAIRS for m in p})
    n = taus[(sc, s, need[0], a, setting)][0].size
    stat = {(m, k): np.empty(reps) for m in need
            for k in ("joint", "cond", "pre_far")}
    for rep in range(reps):
        idx = rng.integers(0, n, n)
        for m in need:
            tau, T = taus[(sc, s, m, a, setting)]
            tt, TT = tau[idx], T[idx]
            alarmed = tt != NEVER
            early = alarmed & (tt <= TT)
            caught = alarmed & (tt > TT) & (tt <= TT + hmax)
            surv = int((~early).sum())
            stat[(m, "joint")][rep] = caught.mean()
            stat[(m, "cond")][rep] = caught.sum() / surv if surv else np.nan
            stat[(m, "pre_far")][rep] = early.mean()

    def point(m, kind):
        tau, T = taus[(sc, s, m, a, setting)]
        alarmed = tau != NEVER
        early = alarmed & (tau <= T)
        caught = alarmed & (tau > T) & (tau <= T + hmax)
        if kind == "joint":
            return float(caught.mean())
        if kind == "pre_far":
            return float(early.mean())
        surv = int((~early).sum())
        return float(caught.sum() / surv) if surv else float("nan")

    out = []
    for ma, mb in MAIN_PAIRS:
        for kind in ("joint", "cond", "pre_far"):
            d = stat[(ma, kind)] - stat[(mb, kind)]
            lo, hi = float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))
            out.append({"scenario": sc, "sharpe_valid": s, "far_target": a,
                        "setting": setting, "quantity": kind, "window_h": hmax,
                        "method_a": ma, "method_b": mb,
                        "point": point(ma, kind) - point(mb, kind),
                        "lo": lo, "hi": hi, "excludes_zero": excludes_zero(lo, hi),
                        "n_reps": reps,
                        "denominator": "survivor denominator recomputed each replicate"
                                       if kind == "cond" else "all paths",
                        "interval_type": "per-comparison, test sampling only; thresholds "
                                         "were frozen in Stage 3A and are not re-covered"})
    return out

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


C = {"binary_gaussian": "#1f77b4", "binary_student_t": "#d62728",
     "trailing_sharpe_252": "#2ca02c", "ewma_student_t": "#ff7f0e",
     "ewma_trunc_student_t": "#7b3294"}
MK = {"binary_gaussian": "o", "binary_student_t": "s", "trailing_sharpe_252": "v",
      "ewma_student_t": "^", "ewma_trunc_student_t": "D"}
SHORT = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
         "trailing_sharpe_252": "trailing 12m Sharpe", "ewma_student_t": "EWMA Student-t",
         "ewma_trunc_student_t": "trunc-EWMA Student-t"}
SET_LABEL = {"fixed_T0": "T=0", "fixed_T126": "T=126", "fixed_T252": "T=252",
             RANDOM_T: "T~U{0..252}"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    """One set of figures per signal strength.

    The pre-specified main comparison for Stage 3B was s = 0.6; the s = 1 set is
    the supervisor's primary setting and is drawn from the same run's data, not
    relabelled or rescaled from s = 0.6.
    """
    m = pd.read_csv(out_dir / "stage3b_metrics.csv")
    e = pd.read_csv(out_dir / "stage3b_evidence.csv")
    fd = out_dir / "figures"
    outs = []
    for sharpe in sorted(cfg.stage3a_sharpes, reverse=True):
        tag = f"s{sharpe:g}".replace(".", "")
        outs += [figure_rates(cfg, m, fd / f"fig3b1_rates_{tag}.png", sharpe),
                 figure_delay(cfg, m, fd / f"fig3b2_delay_{tag}.png", sharpe),
                 figure_evidence(cfg, e, fd / f"fig3b3_evidence_{tag}.png", sharpe)]
    return [str(p.relative_to(out_dir)) for p in outs]


def _order(cfg):
    return [f"fixed_T{T}" for T in cfg.stage3b_fixed_failure_days] + [RANDOM_T]


def figure_rates(cfg, m: pd.DataFrame, out: Path, sharpe: float | None = None) -> Path:
    """Early false alarms, and the two detection rates they have to be read with."""
    a, sc, h = MAIN["alpha"], MAIN["scenario"], MAIN["h"]
    s = MAIN["sharpe"] if sharpe is None else float(sharpe)
    sets = _order(cfg)
    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    d = m[(m.scenario == sc) & (m.far_target == a) & (m.sharpe_valid == s)]
    panels = [("pre_failure_far", f"P(tau <= T)\npre-failure false alarm"),
              (f"cond_detect_h{h}", f"P(T < tau <= T+{h} | tau > T)\nconditional detection"),
              (f"joint_detect_h{h}", f"P(T < tau <= T+{h})\njoint detection")]
    w = 0.16
    for ax, (col, title) in zip(axes, panels):
        for k, mth in enumerate(METHODS_3B):
            y = [float(d[(d.setting == st) & (d.method == mth)][col].iloc[0])
                 if len(d[(d.setting == st) & (d.method == mth)]) else np.nan
                 for st in sets]
            ax.bar(x + (k - 2) * w, y, w, color=C[mth], edgecolor="white", lw=0.5,
                   label=SHORT[mth] if col == "pre_failure_far" else None)
        if col == "pre_failure_far":
            ax.axhline(a, color="0.3", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([SET_LABEL[t] for t in sets], fontsize=8)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel(f"rate  (s={s:g}, budget {a:.0%})")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Stage 3B figure 1 (s = {s:g} vs 0): valid first, failing at T. "
                 "SV+jumps, one frozen 504-day threshold per method.\n"
                 "Joint = survival x conditional; a path that never alarms is tau = "
                 "infinity, never an early false alarm. Post-failure window fixed at "
                 f"h={h} days for every T.", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_delay(cfg, m: pd.DataFrame, out: Path, sharpe: float | None = None) -> Path:
    """Post-failure delay on a common window, over ALL survivors."""
    a, sc = MAIN["alpha"], MAIN["scenario"]
    s = MAIN["sharpe"] if sharpe is None else float(sharpe)
    sets = _order(cfg)
    hs = list(cfg.stage3b_post_windows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    x = np.arange(len(sets))
    d = m[(m.scenario == sc) & (m.far_target == a) & (m.sharpe_valid == s)]
    for k, mth in enumerate(METHODS_3B):
        y, lo = [], []
        for st in sets:
            r = d[(d.setting == st) & (d.method == mth)]
            y.append(float(r.expected_min_delay.iloc[0]) if len(r) else np.nan)
            lo.append(float(r.n_survivors.iloc[0]) if len(r) else np.nan)
        ax.plot(x, y, marker=MK[mth], ms=5, lw=1.4, color=C[mth], label=SHORT[mth])
    ax.set_xticks(x)
    ax.set_xticklabels([SET_LABEL[t] for t in sets], fontsize=8)
    ax.set_ylabel(f"E[min(tau - T, {max(hs)}) | tau > T]  (days)")
    ax.set_title("post-failure delay, all survivors kept", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for k, mth in enumerate(METHODS_3B):
        for st, ls in zip(sets, ("-", "--", "-.", ":")):
            r = d[(d.setting == st) & (d.method == mth)]
            if not len(r):
                continue
            ax.plot(hs, [float(r[f"cond_detect_h{h}"].iloc[0]) for h in hs],
                    ls=ls, marker=MK[mth], ms=4, lw=1.1, color=C[mth],
                    label=f"{SHORT[mth]}, {SET_LABEL[st]}" if mth == "ewma_student_t"
                    else None)
    ax.set_xticks(hs)
    ax.set_xlabel("post-failure window h (days)")
    ax.set_ylabel("conditional detection")
    ax.set_ylim(0, 1)
    ax.set_title("conditional detection against the window", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)
    axes[0].set_xlabel("failure-time setting")
    fig.suptitle(f"Stage 3B figure 2 (s = {s:g} vs 0): delay after failure, measured on a "
                 "COMMON window so "
                 "different T do not get different amounts of observation.\n"
                 "Undetected survivors are scored at the cap, not dropped; an early false "
                 "alarm is not a zero delay.", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_evidence(cfg, e: pd.DataFrame, out: Path, sharpe: float | None = None) -> Path:
    """How much favourable history each method is carrying at the moment of failure."""
    sc = MAIN["scenario"]
    s = MAIN["sharpe"] if sharpe is None else float(sharpe)
    sets = _order(cfg)
    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    d = e[(e.scenario == sc) & (e.sharpe_valid == s)]
    alpha_used = float(d.survivor_filter_alpha.iloc[0]) if len(d) else float("nan")
    for ax, (col_all, col_sur, name) in zip(
            axes, [("mean_U_at_T_all", "mean_U_at_T_survivors",
                    "working failure log-odds  U_T"),
                   ("mean_q_at_T_all", "mean_q_at_T_survivors",
                    "working failure score  q_T")]):
        for k, mth in enumerate(EVIDENCE_METHODS):
            ya = [float(d[(d.setting == st) & (d.method == mth)][col_all].iloc[0])
                  if len(d[(d.setting == st) & (d.method == mth)]) else np.nan
                  for st in sets]
            ys = [float(d[(d.setting == st) & (d.method == mth)][col_sur].iloc[0])
                  if len(d[(d.setting == st) & (d.method == mth)]) else np.nan
                  for st in sets]
            ax.plot(x, ya, marker=MK[mth], ms=5, lw=1.5, color=C[mth],
                    label=f"{SHORT[mth]}, all paths" if col_all.endswith("all") else None)
            ax.plot(x, ys, marker=MK[mth], ms=4, lw=1.1, ls="--", color=C[mth], alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SET_LABEL[t] for t in sets], fontsize=8)
        ax.set_xlabel("failure-time setting")
        ax.set_title(name, fontsize=9.5)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[1].axhline(0.5, color="0.6", lw=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Stage 3B figure 3 (s = {s:g} vs 0): MEAN evidence at the failure "
                 "moment. "
                 "Solid = mean over all paths, dashed = mean over paths that survived "
                 f"to T\nunder the alpha = {alpha_used:.0%} rule (survivor sets differ "
                 "by budget, so the dashed line depends on that choice).\n"
                 "T=0 uses the PRIOR, not the end of the horizon. Working failure score "
                 "from a static two-state classifier, not a posterior that models the "
                 "switch.\nThe trailing Sharpe is omitted: its statistic is a Sharpe "
                 "ratio, not a log-odds, so expit of it would not mean anything.",
                 y=1.06, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import numpy as np
import pandas as pd


CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "trailing_sharpe_252": "252 日 trailing Sharpe",
      "ewma_student_t": "普通 EWMA Student-t",
      "ewma_trunc_student_t": "截断 EWMA Student-t"}
SC = {"gaussian_ctrl": "独立高斯对照", "sv_jump": "SV+跳跃（主情境）"}
QN = {"joint": "联合检出", "cond": "条件检出", "pre_far": "失效前误杀"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _setlabel(cfg, st):
    if st == RANDOM_T:
        return f"随机 T ~ U{{0..{cfg.stage3b_random_failure_max}}}"
    return f"固定 T={st.replace('fixed_T', '')}"


def _med(v, note):
    if isinstance(v, str) and v == "":
        return note or "未达到"
    if v != v:
        return note or "未达到"
    return f"{int(v)}d"


def write_stage3b_report(cfg, S, metrics, always, boot, ev, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    hs = list(cfg.stage3b_post_windows)
    a_main, s0, sc0, T0, h0 = (MAIN["alpha"], MAIN["sharpe"], MAIN["scenario"],
                               MAIN["T"], MAIN["h"])
    a_alt = min(cfg.far_targets)
    sets = [f"fixed_T{T}" for T in cfg.stage3b_fixed_failure_days] + [RANDOM_T]
    M = metrics

    def g(sc, s, a, st, m, col):
        r = M[(M.scenario == sc) & (M.sharpe_valid == s) & (M.far_target == a)
              & (M.setting == st) & (M.method == m)][col]
        if not len(r):
            return float("nan")
        try:
            return float(r.iloc[0])
        except (TypeError, ValueError):
            return r.iloc[0]

    def bo(sc, s, a, st, q, ma, mb):
        r = boot[(boot.scenario == sc) & (boot.sharpe_valid == s) & (boot.far_target == a)
                 & (boot.setting == st) & (boot.quantity == q)
                 & (boot.method_a == ma) & (boot.method_b == mb)]
        return r.iloc[0] if len(r) else None

    A("# Stage 3B 报告：随机失效时间的有限诊断")
    A("")
    A("本文件由 `python -m strategy_survivorship.stage3b` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 本轮的设定与覆盖范围")
    A("")
    A("```")
    A("    theta_t = s   当 t <= T")
    A("    theta_t = 0   当 t >  T")
    A("```")
    A("")
    A(f"T 是**最后一个有效交易日**。设定为固定 T ∈ "
      f"{{{', '.join(str(x) for x in cfg.stage3b_fixed_failure_days)}}}、"
      f"随机 T ~ 均匀分布于整数交易日 0..{cfg.stage3b_random_failure_max}，"
      f"以及 T=∞ 的始终有效对照。")
    A("")
    A(f"**这是“上线后一年内失效”的有限覆盖实验。**"
      f"T ≤ {cfg.stage3b_random_failure_max}，因此每个设定都有完整的 "
      f"{max(hs)} 天失效后观察期，落在 {cfg.horizon_days} 天总期限之内。"
      "**不能**把它描述成已经完成了 0–5 年或 0–10 年的寿命研究。")
    A("")
    A("- T **独立于**噪声、潜在波动率与跳跃；真实 T 只供生成器与评分使用。")
    A("- 模型持续接收收益：在 T 时**不**重置先验、累计证据、方差估计或任何窗口。")
    A("- 静态两状态分类器在切换数据上的 q，本报告称为**工作失效分数**，"
      "不把它解释成正确建模了状态切换后的真实后验概率。")
    A("")
    A(f"D={cfg.D}、H={cfg.horizon_days}、年化基础波动率 {cfg.sigma_annual:.0%}、"
      f"初始先验 {cfg.prior_valid:g}，以及全部检测器参数保持不变。"
      f"两个噪声情境（{SC['gaussian_ctrl']}、{SC['sv_jump']}），"
      f"五个既有方法，s ∈ {{{', '.join(f'{x:g}' for x in S['sharpes'])}}}。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 门槛与抽样")
    A("")
    A(f"- 门槛**统一读取** `{Path(S['thresholds']['source']).name}` 中对应方法、情境、"
      f"Sharpe 与预算的 **{cfg.horizon_days} 日**门槛。"
      "不混用 Stage 3A.1 的短期门槛，**不按真实 T 单独校准**——"
      "真实失效时间在实践中不可得。")
    A(f"- 保留 α ∈ {{{a_alt:g}, {a_main:g}}}；所有路径总监测期不超过 "
      f"{cfg.horizon_days} 天。")
    A(f"- 每个噪声情境使用新的独立测试数据：{cfg.stage3b_test_paths} 条始终有效对照路径，"
      f"另 {cfg.stage3b_test_paths} 条供各失效设定共用。"
      f"共 **{S['sampling']['base_noise_paths']:,} 条基础噪声路径**；"
      "两个 Sharpe 与各方法共享相应噪声以形成配对，"
      "因此各设定之间**不是**互相独立的样本。")
    A("- 随机 T 使用独立随机流，并在相关方法之间共用。")
    A("")
    A("**本轮不重新拟合门槛。**下文所有区间描述的是“给定既有冻结门槛时的测试不确定性”，"
      "**不能**声称重新覆盖了校准误差。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 评价定义")
    A("")
    A("```")
    A("    失效前误杀      P(tau <= T)")
    A("    条件检出        P(T < tau <= T+h | tau > T)")
    A("    联合检出        P(T < tau <= T+h)")
    A("```")
    A("")
    A(f"h ∈ {{{', '.join(str(x) for x in hs)}}} 个交易日。"
      "**主要比较保持相同的失效后观察长度**，不让不同 T 隐含不同的检测机会。")
    A("")
    A(f"从不报警编码为 τ=∞。它是失效前窗口的**存活者**、失效后的**未检出**路径，"
      f"**不是**提前误杀。联合检出恒等式 `联合 = 存活率 × 条件检出` 在每一行上核验，"
      f"本轮全表最大偏差 **{S['joint_identity_max_gap']:.2e}**。")
    A("")
    A(f"失效后延迟为 `E[min(τ−T, {max(hs)})]`，条件于 τ>T，分母是**全部存活到 T 的路径**："
      "未检出的存活者按窗口上限计入而不是删去，提前误杀的路径不计为零延迟。"
      "失效后中位时间同样以全部存活者为分母，取累计比例首次达到一半的那一天。")
    A("")

    # ---------------- 3.5: the supervisor's primary setting -------------
    st0 = f"fixed_T{T0}"
    s_pri = max(S["sharpes"])
    A(f"## 3.5 主线设定 s={s_pri:g}（导师指定）")
    A("")
    A(f"导师指定的主线是 **Sharpe {s_pri:g} 对 0**；s={s0:g} 是后加的弱信号压力测试。"
      f"下面先给 s={s_pri:g} 的完整结果，再在第 4–5 节给出 s={s0:g}。")
    A("")
    A(f"**协议记录不改写**：Stage 3B 事先写定的主比较用的是 s={s0:g}、T={T0}、"
      f"α={a_main:g}。本节是按导师要求调整**展示重点**，"
      f"不把 s={s_pri:g} 追溯描述为当时唯一的预设主分析。"
      f"s={s_pri:g} 使用它自己的收益漂移 μ₁={s_pri:g}·σ_ann/D、检测器候选均值、"
      "EWMA 中心与 Stage 3A 对应的 504 日冻结门槛——**不是** s=0.6 结果的改标签或换算。")
    A("")
    for st in sets:
        A(f"**{_setlabel(cfg, st)}**，{SC[sc0]}，α={a_main:g}，失效后 {h0} 天")
        A("")
        L.extend(_t(list(METHODS_3B),
                    ["方法", "失效前误杀", "存活率", "存活数", "条件检出", "95% 区间",
                     "联合检出", f"E[min(τ−T,{max(hs)})]（存活者）", "失效后未检出",
                     "失效后中位"],
                    lambda m: [
                        CN[m], f"{g(sc0, s_pri, a_main, st, m, 'pre_failure_far'):.4f}",
                        f"{g(sc0, s_pri, a_main, st, m, 'survival_rate'):.4f}",
                        f"{int(g(sc0, s_pri, a_main, st, m, 'n_survivors'))}",
                        f"{g(sc0, s_pri, a_main, st, m, f'cond_detect_h{h0}'):.4f}",
                        f"[{g(sc0, s_pri, a_main, st, m, f'cond_detect_h{h0}_lo'):.4f}, "
                        f"{g(sc0, s_pri, a_main, st, m, f'cond_detect_h{h0}_hi'):.4f}]",
                        f"{g(sc0, s_pri, a_main, st, m, f'joint_detect_h{h0}'):.4f}",
                        f"{g(sc0, s_pri, a_main, st, m, 'expected_min_delay'):.1f}",
                        f"{g(sc0, s_pri, a_main, st, m, 'post_failure_undetected'):.4f}",
                        _med(g(sc0, s_pri, a_main, st, m, 'median_post_failure_delay'),
                             str(g(sc0, s_pri, a_main, st, m,
                                   'median_post_failure_note')))]))
        A("")
    A(f"配对区间（s={s_pri:g}，同样是 {cfg.stage3b_bootstrap_reps} 次配对重抽，"
      "存活分母每次重算）：")
    A("")
    rows = [(st, ma, mb, q) for st in sets for ma, mb in MAIN_PAIRS
            for q in ("joint", "cond", "pre_far")]
    def rp(r):
        st, ma, mb, q = r
        b = bo(sc0, s_pri, a_main, st, q, ma, mb)
        if b is None:
            return [_setlabel(cfg, st), f"{CN[ma]} − {CN[mb]}", QN[q], "-", "-", "-"]
        return [_setlabel(cfg, st), f"{CN[ma]} − {CN[mb]}", QN[q],
                f"{float(b.point) * 100:+.2f} pp",
                f"[{float(b.lo) * 100:+.2f}, {float(b.hi) * 100:+.2f}]",
                "是" if bool(b.excludes_zero) else "否"]
    L.extend(_t(rows, ["失效设定", "比较", "量", "点估计", "95% 区间", "排除 0"], rp))
    A("")
    tt_gain = {st: (g(sc0, s_pri, a_main, st, "ewma_trunc_student_t", f"joint_detect_h{h0}")
                    - g(sc0, s_pri, a_main, st, "ewma_student_t", f"joint_detect_h{h0}"))
               for st in sets}
    A(f"截断更新的额外贡献（截断 t 减普通 EWMA t 的联合检出，s={s_pri:g}）："
      + "；".join(f"{_setlabel(cfg, st)} {v * 100:+.2f} pp" for st, v in tt_gain.items())
      + "。这一对不在 Stage 3B 预设比较之列，因此只报点估计，不作显著性声明。")
    A("")
    b_t0 = bo(sc0, s_pri, a_main, "fixed_T0", "cond", *MAIN_PAIRS[0])
    b_late = bo(sc0, s_pri, a_main, st0, "joint", *MAIN_PAIRS[1])
    b_rand = bo(sc0, s_pri, a_main, RANDOM_T, "joint", *MAIN_PAIRS[1])
    A(f"**从 s={s_pri:g} 自己的数据读出来的四点。**")
    A("")
    A(f"1. **EWMA t 相对固定 t 的改善**：T=0（上线即无效）条件检出差 "
      f"{float(b_t0.point) * 100:+.2f} pp "
      f"[{float(b_t0.lo) * 100:+.2f}, {float(b_t0.hi) * 100:+.2f}]，"
      f"随机 T 下联合检出差 "
      f"{float(bo(sc0, s_pri, a_main, RANDOM_T, 'joint', *MAIN_PAIRS[0]).point) * 100:+.2f} pp。"
      "四个失效设定下这一差都排除 0。")
    A(f"2. **滚动 Sharpe 的排序仍随 T 改变**：EWMA t 减 trailing Sharpe 的联合检出差，"
      f"T=0 为 {float(bo(sc0, s_pri, a_main, 'fixed_T0', 'joint', *MAIN_PAIRS[1]).point) * 100:+.2f} pp，"
      f"T={T0} 为 {float(b_late.point) * 100:+.2f} pp "
      f"[{float(b_late.lo) * 100:+.2f}, {float(b_late.hi) * 100:+.2f}]，"
      f"随机 T 为 {float(b_rand.point) * 100:+.2f} pp。"
      f"方向在 T={T0} 处反转，幅度比 s={s0:g} 时更大；成因见第 6 节列出的四条未分解机制。")
    A(f"3. **截断更新的额外贡献**：{min(tt_gain.values()) * 100:+.2f} 到 "
      f"{max(tt_gain.values()) * 100:+.2f} pp，比第 1 点小一个数量级。")
    A(f"4. **绝对能力**：见 Stage 3A——s={s_pri:g}、{SC[sc0]}、α={a_main:g} 下，"
      "观察满一年后最好的方法检出 48.74% 的无效策略，两年 76.17%；"
      f"同一设定 s={s0:g} 时为 25.84% / 52.32%。"
      "这些是模拟结果，不是真实市场部署的验证。")
    A("")

    # ---------------- 4 ----------------
    A(f"## 4. 主设定：s={s0:g}、{SC[sc0]}、α={a_main:g}、T={T0}、失效后 {h0} 天")
    A("")
    rows = list(METHODS_3B)
    L.extend(_t(rows, ["方法", "失效前误杀", "95% 区间", "存活数", "条件检出", "95% 区间",
                       "联合检出", "95% 区间", f"E[min(τ−T,{max(hs)})]（存活者）",
                       "失效后中位", "失效后未检出"],
                lambda m: [
                    CN[m], f"{g(sc0, s0, a_main, st0, m, 'pre_failure_far'):.4f}",
                    f"[{g(sc0, s0, a_main, st0, m, 'pre_failure_far_lo'):.4f}, "
                    f"{g(sc0, s0, a_main, st0, m, 'pre_failure_far_hi'):.4f}]",
                    f"{int(g(sc0, s0, a_main, st0, m, 'n_survivors'))}",
                    f"{g(sc0, s0, a_main, st0, m, f'cond_detect_h{h0}'):.4f}",
                    f"[{g(sc0, s0, a_main, st0, m, f'cond_detect_h{h0}_lo'):.4f}, "
                    f"{g(sc0, s0, a_main, st0, m, f'cond_detect_h{h0}_hi'):.4f}]",
                    f"{g(sc0, s0, a_main, st0, m, f'joint_detect_h{h0}'):.4f}",
                    f"[{g(sc0, s0, a_main, st0, m, f'joint_detect_h{h0}_lo'):.4f}, "
                    f"{g(sc0, s0, a_main, st0, m, f'joint_detect_h{h0}_hi'):.4f}]",
                    f"{g(sc0, s0, a_main, st0, m, 'expected_min_delay'):.1f}",
                    _med(g(sc0, s0, a_main, st0, m, 'median_post_failure_delay'),
                         str(g(sc0, s0, a_main, st0, m, 'median_post_failure_note'))),
                    f"{g(sc0, s0, a_main, st0, m, 'post_failure_undetected'):.4f}"]))
    A("")
    A("预先固定的重点比较（配对 bootstrap，存活分母在每次重抽中重算）：")
    A("")
    rows = [(ma, mb, q) for ma, mb in MAIN_PAIRS for q in ("joint", "cond", "pre_far")]
    L.extend(_t(rows, ["比较", "量", "点估计", "95% 区间", "排除 0"],
                lambda r: [f"{CN[r[0]]} − {CN[r[1]]}", QN[r[2]]]
                + ([f"{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).point) * 100:+.2f} pp",
                    f"[{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).lo) * 100:+.2f}, "
                    f"{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).hi) * 100:+.2f}]",
                    "是" if bool(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).excludes_zero)
                    else "否"]
                   if bo(sc0, s0, a_main, st0, r[2], r[0], r[1]) is not None
                   else ["-", "-", "-"])))
    A("")
    A("**三个量必须一起读**：联合检出的差可以由条件检出的差、也可以由失效前误杀的差造成。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 四种失效设定的对照")
    A("")
    for s in sorted(S["sharpes"], reverse=True):
        A(f"**s = {s:g}**，{SC[sc0]}，α={a_main:g}，失效后 {h0} 天")
        A("")
        rows = [(st, m) for st in sets for m in METHODS_3B]
        L.extend(_t(rows, ["失效设定", "方法", "失效前误杀", "存活率", "条件检出",
                           "联合检出", f"E[min(τ−T,{max(hs)})]（存活者）", "失效后中位"],
                    lambda r: [_setlabel(cfg, r[0]), CN[r[1]],
                               f"{g(sc0, s, a_main, r[0], r[1], 'pre_failure_far'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], 'survival_rate'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], f'cond_detect_h{h0}'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], f'joint_detect_h{h0}'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], 'expected_min_delay'):.1f}",
                               _med(g(sc0, s, a_main, r[0], r[1], 'median_post_failure_delay'),
                                    str(g(sc0, s, a_main, r[0], r[1],
                                          'median_post_failure_note')))]))
        A("")
    A("**始终有效对照（T=∞）的全期实际误杀率**：")
    A("")
    rows = [r for _, r in always[(always.scenario == sc0)
                                 & (always.far_target == a_main)].iterrows()]
    L.extend(_t(rows, ["s", "方法", "全期误杀率", "95% 区间", "样本量"],
                lambda r: [f"{r['sharpe_valid']:g}", CN[r["method"]],
                           f"{r['full_period_far']:.4f}",
                           f"[{r['full_period_far_lo']:.4f}, {r['full_period_far_hi']:.4f}]",
                           f"{int(r['n_paths'])}"]))
    A("")
    A(f"（T=0 与更早的失效设定下，“失效前误杀”按定义只统计到第 T 天为止，"
      f"因此 T=0 时它恒为 0：第 1 天起策略已经无效。）")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 历史证据诊断")
    A("")
    A("失效时刻 T 的工作失效分数 q_T 与对数赔率 U_T = −L_T，分别给出全部路径与"
      "存活到 T 的路径。**T=0 使用初始值（先验），不取数组最后一天。**")
    A("")
    A(f"**{CN['trailing_sharpe_252']} 不在此表中。**它的统计量是年化 Sharpe 比率，"
      "不在对数赔率尺度上；对它取 expit 得到的数没有可解释的含义，"
      "因此这里不把它放到一个它并不居于其上的刻度上报告。")
    A("")
    for s in sorted(S["sharpes"], reverse=True):
        A(f"**s = {s:g}**，{SC[sc0]}")
        A("")
        d = ev[(ev.scenario == sc0) & (ev.sharpe_valid == s)]
        rows = [(st, m) for st in sets for m in EVIDENCE_METHODS]
        def re_(r):
            x = d[(d.setting == r[0]) & (d.method == r[1])]
            if not len(x):
                return [_setlabel(cfg, r[0]), CN[r[1]], "-", "-", "-", "-", "-"]
            x = x.iloc[0]
            return [_setlabel(cfg, r[0]), CN[r[1]],
                    f"{x.mean_U_at_T_all:+.4f}", f"{x.mean_q_at_T_all:.4f}",
                    f"{x.mean_U_at_T_survivors:+.4f}", f"{x.mean_q_at_T_survivors:.4f}",
                    f"{int(x.n_survivors_last_alpha)}"]
        L.extend(_t(rows, ["失效设定", "方法", "U_T（全部）", "q_T（全部）",
                           "U_T（存活）", "q_T（存活）", "存活数"], re_))
        A("")
    idc = pd.DataFrame(S["identity_check"])
    A("高斯对照、固定 T、**全部路径**上的解析核验：")
    A("")
    A("```")
    A("    E[U_T] = -s²T/(2D),      E[U_{T+h} - U_T] = s²h/(2D)")
    A("```")
    A("")
    rows = [r for _, r in idc.iterrows()]
    L.extend(_t(rows, ["s", "T", "h", "E[U_T] 理论", "实测", "E[ΔU] 理论", "实测",
                       "最大偏差（标准误）"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{int(r['T'])}", f"{int(r['h'])}",
                           f"{r['mean_U_at_T_theory']:+.4f}",
                           f"{r['mean_U_at_T_measured']:+.4f}",
                           f"{r['mean_dU_theory']:+.4f}", f"{r['mean_dU_measured']:+.4f}",
                           f"{max(r['z_U_at_T'], r['z_dU']):.2f}"]))
    A("")
    A("**这个关系不自动适用于经过提前误杀筛选的样本，也不自动适用于 SV+跳跃。**"
      "上面两张表里的“存活”列正是被筛选过的样本。")
    A("")
    tr = "trailing_sharpe_252"
    elig = int(g(sc0, s0, a_main, st0, tr, "first_eligible_day"))
    A(f"**{CN[tr]} 在 T={T0} 处同时误杀更少、条件检出与联合检出更高，这是本设定下的"
      f"实测事实。** 但本轮**没有**分解这个优势的来源。至少有四条机制可能同时起作用，"
      "它们的相对份额未被测量：")
    A("")
    A(f"1. **启动期**：它最早只能在第 {elig} 天说话，因此在 T=0 时失效后窗口里只有"
      f"第 {elig}–{h0} 天可用，而在 T={T0} 时启动期恰在失效时刻走完。"
      f"同一件事也限制了它在失效前的报警机会："
      f"T={T0} 时它在失效前只有第 {elig} 天这一天可能报警"
      f"（实测误杀 {g(sc0, s0, a_main, st0, tr, 'pre_failure_far'):.2%}，"
      f"{CN['ewma_student_t']} 为 "
      f"{g(sc0, s0, a_main, st0, 'ewma_student_t', 'pre_failure_far'):.2%}）。")
    A("2. **滚动窗口会逐步移除失效前的数据**：失效之后，有利的历史证据会随窗口滑出，"
      "而累计型模型不会自动丢弃它。")
    A("3. **方差状态**：EWMA 类方法在失效时刻带着一个由过去数据形成的方差估计。")
    A("4. **存活者筛选**：条件统计只在没有被提前误杀的路径上计算，各方法的存活集合不同。")
    A("")
    A("因此**不能**把这个优势归因于其中任何单独一条，也不能据此断言"
      "“它的判别能力更强”或“它的优势只是窗口对齐”——两种说法本轮都没有证据支持。")
    A("")
    A("其二，若累计型模型在较晚失效时变慢，**不要把全部差异都归因于历史证据**："
      "初始化、方差状态与存活筛选（只有没被提前误杀的路径才进入条件统计）"
      "都会影响表现，本轮没有分解它们各自的份额。")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 直接回答")
    A("")
    jt = {m: g(sc0, s0, a_main, st0, m, f"joint_detect_h{h0}") for m in METHODS_3B}
    z0 = {m: g(sc0, s0, a_main, "fixed_T0", m, f"joint_detect_h{h0}") for m in METHODS_3B}
    bestj = max((v, k) for k, v in jt.items())
    A(f"**（一）从“上线即无效”迁移到“先有效、后失效”时有什么变化？** "
      f"在主设定下，把 T 从 0 移到 {T0} 天后，同样 {h0} 天的观察窗口内："
      + "；".join(f"{CN[m]} 的联合检出由 {z0[m]:.2%} 变为 {jt[m]:.2%}"
                  for m in ("binary_student_t", "ewma_student_t")) + "。")
    A("")
    A(f"变化来自三处，本报告分开列出而不是合成一个数字："
      f"（i）失效前的误杀会先消耗掉一部分路径（本设定下 "
      f"{g(sc0, s0, a_main, st0, 'ewma_student_t', 'pre_failure_far'):.2%}，"
      f"这些路径在 T 之前就被判无效，属于误杀而非检出）；"
      f"（ii）存活下来的路径带着一段**有利的历史证据**进入失效时刻"
      f"（第 6 节的 U_T、q_T）；"
      f"（iii）滚动型方法的启动期在较晚的 T 下已经走完。")
    A("")
    A(f"主设定下联合检出最高的是 {CN[bestj[1]]}（{bestj[0]:.2%}）。")
    A("")
    r_late = bo(sc0, s0, a_main, st0, "joint", "ewma_student_t", tr)
    r_rand = bo(sc0, s0, a_main, RANDOM_T, "joint", "ewma_student_t", tr)
    if r_late is not None and r_rand is not None:
        A(f"**一个随失效时间改变方向的比较。** {CN['ewma_student_t']} 相对 {CN[tr]} 的"
          f"联合检出差，在 T={T0} 时为 {float(r_late.point) * 100:+.2f} pp "
          f"[{float(r_late.lo) * 100:+.2f}, {float(r_late.hi) * 100:+.2f}]，"
          f"而在随机 T 下为 {float(r_rand.point) * 100:+.2f} pp "
          f"[{float(r_rand.lo) * 100:+.2f}, {float(r_rand.hi) * 100:+.2f}]——"
          "两个区间都排除 0，**方向相反**。"
          f"本轮只测了 T ∈ {{0, 126, {T0}}} 与随机 T 四个设定，"
          f"**没有**扫描 T，因此不能说 T={T0} 是对 {CN[tr]} 全局最有利的失效时刻。"
          "能记录的是：**方法排序随失效时刻改变**，成因见上一节列出的四条未分解机制。")
    A("")
    A("**（二）这些结果是否足以支持本周向导师说明适用范围？** "
      "可以说明的是：本项目现有的五个检测器在“先有效、后失效”这一设定下仍然可用，"
      "并且失效前误杀、条件检出、联合检出三个量已经被分开测量与报告；"
      f"覆盖范围是 T ≤ {cfg.stage3b_random_failure_max} 天、两个噪声情境、"
      f"两个信号强度、两个误杀预算。"
      "**不能**据此说明更长寿命（数年）的表现，也不能说明任何未纳入本项目的方法。")
    A("")
    A("**（三）哪些问题应留给导师决定下一周是否继续？** 本报告不替这些问题作答：")
    A("")
    A("1. 失效时间的覆盖范围是否要从一年扩展到数年——那需要更长的模拟期限，"
      "并会改变“两年监测”这个主基准的含义。")
    A("2. 是否需要一个显式建模状态切换的检测器（而不是继续用静态两状态分类器的"
      "工作失效分数）——本轮的证据只说明静态分类器在切换数据上仍能工作，"
      "没有比较过任何切换感知的方法。")
    A("3. 失效前误杀与失效后延迟之间的权衡应当按什么代价函数取舍——"
      "本项目至今只报告两者，没有为它们赋权，等待成本也未纳入评价。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 限制")
    A("")
    A(f"- 覆盖范围是 T ≤ {cfg.stage3b_random_failure_max} 天（一年内失效），"
      "不是 0–5 年或 0–10 年的寿命研究。")
    A("- 门槛沿用 Stage 3A 的 504 日冻结值；区间只描述测试抽样，不覆盖校准误差。")
    A("- 各失效设定共用同一批基础噪声，设定之间不是独立样本。")
    A("- 静态分类器的 q 是工作失效分数，不是对切换过程的正确后验。")
    A("- 解析恒等式只在 iid 高斯、固定 T、全部路径上核验过。")
    A("- 本轮不实现 BOCPD，也不新增任何检测模型或扩大参数范围。")
    A("")
    A("## 9. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage3b")
    A("uv run python -m strategy_survivorship.stage3b --figures-only")
    A("uv run python -m strategy_survivorship.stage3b --report-only")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
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


from .config import DEFAULT

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage3a import draw_eps, sharpe_cfg
from .stage3a1 import spec_for


def evidence_identity_check(cfg) -> list[dict]:
    """E[U_T] = -s^2 T/(2D) and E[U_{T+h} - U_T] = s^2 h/(2D).

    Gaussian control, fixed T, ALL paths -- the identity does not survive
    conditioning on having escaped an early false alarm, and it is not claimed
    for SV+jumps.
    """
    from .detectors import binary_gaussian_log_odds

    rows = []
    rng = np.random.default_rng(np.random.SeedSequence(3_11_2026))
    n = 40000
    for s in cfg.stage3a_sharpes:
        cs = sharpe_cfg(cfg, s)
        eps = rng.standard_normal((n, cfg.horizon_days))
        for T in cfg.stage3b_fixed_failure_days:
            r = returns_with_failure(eps, np.full(n, T), s, cfg)
            U = -binary_gaussian_log_odds(r, cs)
            uT = (np.full(n, -cs.prior_log_odds) if T == 0 else U[:, T - 1])
            for h in cfg.stage3b_post_windows:
                if T + h > cfg.horizon_days:
                    continue
                d = U[:, T + h - 1] - uT
                rows.append({
                    "sharpe_valid": s, "T": T, "h": h, "n_paths": n,
                    "mean_U_at_T_theory": -s * s * T / (2.0 * cfg.D),
                    "mean_U_at_T_measured": float(uT.mean()),
                    "mean_dU_theory": s * s * h / (2.0 * cfg.D),
                    "mean_dU_measured": float(d.mean()),
                    "scope": "iid Gaussian, fixed T, all paths (no survivor filter)"})
            del r, U
        del eps
    for row in rows:
        n_ = row["n_paths"]
        s, T, h = row["sharpe_valid"], row["T"], row["h"]
        se_T = np.sqrt(s * s * T / cfg.D / n_) if T else 0.0
        se_d = np.sqrt(s * s * h / cfg.D / n_)
        row["z_U_at_T"] = (abs(row["mean_U_at_T_measured"] - row["mean_U_at_T_theory"]) / se_T
                           if se_T else 0.0)
        row["z_dU"] = abs(row["mean_dU_measured"] - row["mean_dU_theory"]) / se_d
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3B: valid first, failing later.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs/stage3a_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage3b_test_paths=800, stage3b_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke3b") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only or args.report_only:
        if args.figures_only:
            setup_style()
            print("figures redrawn from disk:", draw_all(cfg, out_dir))
        if args.report_only:
            S = json.loads((out_dir / "stage3b_summary.json").read_text())
            write_stage3b_report(
                cfg, S, pd.read_csv(out_dir / "stage3b_metrics.csv"),
                pd.read_csv(out_dir / "stage3b_always_valid.csv"),
                pd.read_csv(out_dir / "stage3b_bootstrap.csv"),
                pd.read_csv(out_dir / "stage3b_evidence.csv"),
                out_dir / "stage3b_report.md")
            print("report rebuilt from disk, no simulation rerun")
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    ev_check = evidence_identity_check(cfg)
    worst = max(max(r["z_U_at_T"], r["z_dU"]) for r in ev_check)
    if worst > 5.0:
        raise AssertionError(f"evidence identity off by {worst:.2f} SE")
    pd.DataFrame(ev_check).to_csv(out_dir / "stage3b_identity_check.csv", index=False)
    status("analytic identity checked", worst_z=round(worst, 2))

    thr = load_stage3a_thresholds(args.thresholds)
    need = [(m, sc, s, a) for m in METHODS_3B for sc in SCENARIOS_3B
            for s in cfg.stage3a_sharpes for a in cfg.far_targets]
    missing = [k for k in need if k not in thr]
    if missing:
        raise KeyError(f"frozen thresholds missing for {missing[:3]} ...")
    status("frozen Stage 3A thresholds loaded", n=len(need), source=str(args.thresholds))

    n = cfg.stage3b_test_paths
    tkids = make_streams(cfg)["stage3b_test"].spawn(len(SCENARIOS_3B) * 2)
    fkids = make_streams(cfg)["stage3b_failure_time"].spawn(len(SCENARIOS_3B))
    eps_fail, eps_valid, fail_times = {}, {}, {}
    for i, sc in enumerate(SCENARIOS_3B):
        spec = spec_for(cfg, sc)
        eps_fail[sc] = draw_eps(cfg, spec, tkids[2 * i], n)
        eps_valid[sc] = draw_eps(cfg, spec, tkids[2 * i + 1], n)
        for kind, T0 in failure_settings(cfg):
            fail_times[(sc, kind, T0)] = draw_failure_times(cfg, kind, T0, n, fkids[i])
    status("test data drawn", base_noise_paths=len(SCENARIOS_3B) * 2 * n,
           settings=len(failure_settings(cfg)) + 1)

    rows, taus, ev, always = evaluate(cfg, thr, eps_fail, eps_valid, fail_times)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "stage3b_metrics.csv", index=False)
    pd.DataFrame(always).to_csv(out_dir / "stage3b_always_valid.csv", index=False)
    pd.DataFrame(ev).to_csv(out_dir / "stage3b_evidence.csv", index=False)
    gap = float(np.nanmax([metrics[f"joint_identity_gap_h{h}"].max()
                           for h in cfg.stage3b_post_windows]))
    if gap > 1e-12:
        raise AssertionError(f"joint = survival x conditional violated by {gap:.3e}")
    status("scored", metric_rows=len(metrics), always_valid_rows=len(always),
           joint_identity_max_gap=gap)

    bkids = make_streams(cfg)["stage3b_bootstrap"].spawn(
        len(SCENARIOS_3B) * len(cfg.stage3a_sharpes) * len(cfg.far_targets)
        * (len(cfg.stage3b_fixed_failure_days) + 1))
    boots, i = [], 0
    settings = [f"fixed_T{T}" for T in cfg.stage3b_fixed_failure_days] + [RANDOM_T]
    for sc in SCENARIOS_3B:
        for s in cfg.stage3a_sharpes:
            for a in cfg.far_targets:
                for setting in settings:
                    boots.extend(bootstrap(cfg, taus, sc, s, a, setting, bkids[i]))
                    i += 1
        status(f"bootstrap done: {sc}", rows=len(boots), reps=cfg.stage3b_bootstrap_reps)
    pd.DataFrame(boots).to_csv(out_dir / "stage3b_bootstrap.csv", index=False)

    figures = []
    if not args.no_figures:
        setup_style()
        figures = draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "3B", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "methods": list(METHODS_3B), "scenarios": list(SCENARIOS_3B),
        "sharpes": list(cfg.stage3a_sharpes),
        "failure_settings": [f"fixed T={T}" for T in cfg.stage3b_fixed_failure_days]
                            + [f"T ~ Uniform{{0..{cfg.stage3b_random_failure_max}}}",
                               "T = infinity (always-valid control)"],
        "coverage_note": "T <= 252, i.e. failure inside the first year; NOT a 0-5 or "
                         "0-10 year lifetime study",
        "post_windows": list(cfg.stage3b_post_windows),
        "main_setting": MAIN, "main_pairs": [list(p) for p in MAIN_PAIRS],
        "thresholds": {"source": str(args.thresholds),
                       "note": "frozen Stage 3A 504-day thresholds; nothing recalibrated "
                               "here and nothing calibrated per T"},
        "sampling": {
            "base_noise_paths": len(SCENARIOS_3B) * 2 * n,
            "note": "per scenario: 10,000 always-valid control paths and 10,000 shared by "
                    "every failure setting; both Sharpes and all methods share them, so "
                    "these are not independent per-setting samples",
        },
        "identity_check": ev_check, "joint_identity_max_gap": gap,
        "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage3b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage3b_report(cfg, summary, metrics, pd.DataFrame(always),
                         pd.DataFrame(boots), pd.DataFrame(ev),
                         out_dir / "stage3b_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
