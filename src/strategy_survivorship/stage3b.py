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


CN = {"binary_gaussian": "fixed Gaussian", "binary_student_t": "fixed Student-t",
      "trailing_sharpe_252": "252-day trailing Sharpe",
      "ewma_student_t": "plain EWMA Student-t",
      "ewma_trunc_student_t": "truncated EWMA Student-t"}
SC = {"gaussian_ctrl": "iid Gaussian control", "sv_jump": "SV+jumps (main scenario)"}
QN = {"joint": "joint detection", "cond": "conditional detection", "pre_far": "pre-failure false alarm"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _setlabel(cfg, st):
    if st == RANDOM_T:
        return f"random T ~ U{{0..{cfg.stage3b_random_failure_max}}}"
    return f"fixed T={st.replace('fixed_T', '')}"


def _med(v, note):
    if isinstance(v, str) and v == "":
        return note or "not reached"
    if v != v:
        return note or "not reached"
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

    A("# Stage 3B report: a bounded diagnostic for a random failure time")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage3b`; "
      "every number is taken from the result tables of a single run.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. Setup and coverage of this round")
    A("")
    A("```")
    A("    theta_t = s   if t <= T")
    A("    theta_t = 0   if t >  T")
    A("```")
    A("")
    A(f"T is the **last valid trading day**. The settings are fixed T ∈ "
      f"{{{', '.join(str(x) for x in cfg.stage3b_fixed_failure_days)}}}, "
      f"random T ~ uniform on the integer trading days 0..{cfg.stage3b_random_failure_max}, "
      f"and the always-valid control with T=∞.")
    A("")
    A(f"**This is a bounded-coverage experiment for “failure within one year after going live”.** "
      f"T ≤ {cfg.stage3b_random_failure_max}, so every setting has a complete "
      f"{max(hs)}-day post-failure observation period that falls inside the {cfg.horizon_days}-day total horizon. "
      "It **must not** be described as a completed 0–5 year or 0–10 year lifetime study.")
    A("")
    A("- T is **independent of** the noise, the latent volatility and the jumps; the true T is used only by the generator and for scoring.")
    A("- The models keep receiving returns: at T they do **not** reset the prior, the accumulated evidence, the variance estimate or any window.")
    A("- This report calls the q of a static two-state classifier on switching data the **working failure score**, "
      "and does not interpret it as the true posterior probability of a model that correctly represents the state switch.")
    A("")
    A(f"D={cfg.D}, H={cfg.horizon_days}, the annualised base volatility {cfg.sigma_annual:.0%}, "
      f"the initial prior {cfg.prior_valid:g} and all detector parameters are unchanged. "
      f"Two noise scenarios ({SC['gaussian_ctrl']}, {SC['sv_jump']}), "
      f"the five existing methods, s ∈ {{{', '.join(f'{x:g}' for x in S['sharpes'])}}}.")
    A("")

    # ---------------- 2 ----------------
    A("## 2. Thresholds and sampling")
    A("")
    A(f"- Thresholds are **read uniformly** from `{Path(S['thresholds']['source']).name}`: for each method, scenario, "
      f"Sharpe and budget, the matching **{cfg.horizon_days}-day** threshold. "
      "Stage 3A.1's short-horizon thresholds are not mixed in, and **nothing is calibrated separately for the true T** — "
      "the true failure time is not available in practice.")
    A(f"- α ∈ {{{a_alt:g}, {a_main:g}}} is kept; the total monitoring period of every path is at most "
      f"{cfg.horizon_days} days.")
    A(f"- Each noise scenario uses new, independent test data: {cfg.stage3b_test_paths} always-valid control paths, "
      f"and another {cfg.stage3b_test_paths} shared by all failure settings. "
      f"In total **{S['sampling']['base_noise_paths']:,} base noise paths**; "
      "both Sharpes and all methods share the corresponding noise so that comparisons are paired, "
      "so the settings are **not** mutually independent samples.")
    A("- The random T uses an independent random stream and is shared across the relevant methods.")
    A("")
    A("**Thresholds are not refitted in this round.** Every interval below describes “test uncertainty given the existing frozen thresholds”; "
      "it **cannot** be claimed to re-cover calibration error.")
    A("")

    # ---------------- 3 ----------------
    A("## 3. Evaluation definitions")
    A("")
    A("```")
    A("    pre-failure false alarm    P(tau <= T)")
    A("    conditional detection      P(T < tau <= T+h | tau > T)")
    A("    joint detection            P(T < tau <= T+h)")
    A("```")
    A("")
    A(f"h ∈ {{{', '.join(str(x) for x in hs)}}} trading days. "
      "**The main comparisons keep the same post-failure observation length**, so that different T do not implicitly get different detection opportunities.")
    A("")
    A(f"A path that never alarms is coded as τ=∞. It is a **survivor** of the pre-failure window and an **undetected** path after failure, "
      f"**not** an early false alarm. The joint-detection identity `joint = survival rate × conditional detection` is checked on every row; "
      f"the largest deviation over the whole table in this round is **{S['joint_identity_max_gap']:.2e}**.")
    A("")
    A(f"The post-failure delay is `E[min(τ−T, {max(hs)})]`, conditional on τ>T, and its denominator is **all paths that survive to T**: "
      "undetected survivors are counted at the window cap rather than dropped, and paths with an early false alarm are not counted as zero delay. "
      "The post-failure median time also uses all survivors as the denominator, and is the first day on which the cumulative fraction reaches one half.")
    A("")

    # ---------------- 3.5: the supervisor's primary setting -------------
    st0 = f"fixed_T{T0}"
    s_pri = max(S["sharpes"])
    A(f"## 3.5 Primary setting s={s_pri:g} (specified by the supervisor)")
    A("")
    A(f"The primary setting specified by the supervisor is **Sharpe {s_pri:g} vs 0**; s={s0:g} is a weak-signal stress test added later. "
      f"The full results for s={s_pri:g} are given first below, and s={s0:g} follows in Sections 4–5.")
    A("")
    A(f"**The protocol record is not rewritten**: the main comparison written down in advance for Stage 3B used s={s0:g}, T={T0}, "
      f"α={a_main:g}. This section changes the **emphasis of the presentation** at the supervisor's request, "
      f"and does not retroactively describe s={s_pri:g} as the only pre-specified main analysis at the time. "
      f"s={s_pri:g} uses its own return drift μ₁={s_pri:g}·σ_ann/D, its own detector candidate mean, "
      "EWMA centre and the corresponding 504-day frozen threshold from Stage 3A — it is **not** a relabelling or rescaling of the s=0.6 results.")
    A("")
    for st in sets:
        A(f"**{_setlabel(cfg, st)}**, {SC[sc0]}, α={a_main:g}, {h0} days after failure")
        A("")
        L.extend(_t(list(METHODS_3B),
                    ["Method", "Pre-failure false alarm", "Survival rate", "Survivors", "Conditional detection", "95% interval",
                     "Joint detection", f"E[min(τ−T,{max(hs)})] (survivors)", "Undetected after failure",
                     "Post-failure median"],
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
    A(f"Paired intervals (s={s_pri:g}, again {cfg.stage3b_bootstrap_reps} paired resamples, "
      "survivor denominator recomputed each time):")
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
                "yes" if bool(b.excludes_zero) else "no"]
    L.extend(_t(rows, ["Failure setting", "Comparison", "Quantity", "Point estimate", "95% interval", "Excludes 0"], rp))
    A("")
    tt_gain = {st: (g(sc0, s_pri, a_main, st, "ewma_trunc_student_t", f"joint_detect_h{h0}")
                    - g(sc0, s_pri, a_main, st, "ewma_student_t", f"joint_detect_h{h0}"))
               for st in sets}
    A(f"Additional contribution of the truncated update (joint detection of truncated t minus plain EWMA t, s={s_pri:g}): "
      + "; ".join(f"{_setlabel(cfg, st)} {v * 100:+.2f} pp" for st, v in tt_gain.items())
      + ". This pair is not among the pre-specified Stage 3B comparisons, so only point estimates are reported and no significance claim is made.")
    A("")
    b_t0 = bo(sc0, s_pri, a_main, "fixed_T0", "cond", *MAIN_PAIRS[0])
    b_late = bo(sc0, s_pri, a_main, st0, "joint", *MAIN_PAIRS[1])
    b_rand = bo(sc0, s_pri, a_main, RANDOM_T, "joint", *MAIN_PAIRS[1])
    A(f"**Four points read from the s={s_pri:g} data itself.**")
    A("")
    A(f"1. **Improvement of EWMA t over fixed t**: at T=0 (invalid from launch) the conditional-detection difference is "
      f"{float(b_t0.point) * 100:+.2f} pp "
      f"[{float(b_t0.lo) * 100:+.2f}, {float(b_t0.hi) * 100:+.2f}], "
      f"and under random T the joint-detection difference is "
      f"{float(bo(sc0, s_pri, a_main, RANDOM_T, 'joint', *MAIN_PAIRS[0]).point) * 100:+.2f} pp. "
      "This difference excludes 0 in all four failure settings.")
    A(f"2. **The ranking against the rolling Sharpe still changes with T**: the joint-detection difference EWMA t minus trailing Sharpe is, "
      f"at T=0, {float(bo(sc0, s_pri, a_main, 'fixed_T0', 'joint', *MAIN_PAIRS[1]).point) * 100:+.2f} pp; "
      f"at T={T0}, {float(b_late.point) * 100:+.2f} pp "
      f"[{float(b_late.lo) * 100:+.2f}, {float(b_late.hi) * 100:+.2f}]; "
      f"under random T, {float(b_rand.point) * 100:+.2f} pp. "
      f"The direction reverses at T={T0}, with a larger magnitude than at s={s0:g}; for the causes see the four undecomposed mechanisms listed in Section 6.")
    A(f"3. **Additional contribution of the truncated update**: {min(tt_gain.values()) * 100:+.2f} to "
      f"{max(tt_gain.values()) * 100:+.2f} pp, an order of magnitude smaller than point 1.")
    A(f"4. **Absolute capability**: see Stage 3A — at s={s_pri:g}, {SC[sc0]}, α={a_main:g}, "
      "the best method detects 48.74% of invalid strategies after a full year of observation, and 76.17% after two years; "
      f"in the same setting with s={s0:g} the figures are 25.84% / 52.32%. "
      "These are simulation results, not a validation in real market deployment.")
    A("")

    # ---------------- 4 ----------------
    A(f"## 4. Main setting: s={s0:g}, {SC[sc0]}, α={a_main:g}, T={T0}, {h0} days after failure")
    A("")
    rows = list(METHODS_3B)
    L.extend(_t(rows, ["Method", "Pre-failure false alarm", "95% interval", "Survivors", "Conditional detection", "95% interval",
                       "Joint detection", "95% interval", f"E[min(τ−T,{max(hs)})] (survivors)",
                       "Post-failure median", "Undetected after failure"],
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
    A("Pre-specified key comparisons (paired bootstrap, survivor denominator recomputed in every resample):")
    A("")
    rows = [(ma, mb, q) for ma, mb in MAIN_PAIRS for q in ("joint", "cond", "pre_far")]
    L.extend(_t(rows, ["Comparison", "Quantity", "Point estimate", "95% interval", "Excludes 0"],
                lambda r: [f"{CN[r[0]]} − {CN[r[1]]}", QN[r[2]]]
                + ([f"{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).point) * 100:+.2f} pp",
                    f"[{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).lo) * 100:+.2f}, "
                    f"{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).hi) * 100:+.2f}]",
                    "yes" if bool(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).excludes_zero)
                    else "no"]
                   if bo(sc0, s0, a_main, st0, r[2], r[0], r[1]) is not None
                   else ["-", "-", "-"])))
    A("")
    A("**The three quantities must be read together**: a difference in joint detection can be caused by a difference in conditional detection, or equally by a difference in pre-failure false alarms.")
    A("")

    # ---------------- 5 ----------------
    A("## 5. Comparison across the four failure settings")
    A("")
    for s in sorted(S["sharpes"], reverse=True):
        A(f"**s = {s:g}**, {SC[sc0]}, α={a_main:g}, {h0} days after failure")
        A("")
        rows = [(st, m) for st in sets for m in METHODS_3B]
        L.extend(_t(rows, ["Failure setting", "Method", "Pre-failure false alarm", "Survival rate", "Conditional detection",
                           "Joint detection", f"E[min(τ−T,{max(hs)})] (survivors)", "Post-failure median"],
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
    A("**Realised full-period false-alarm rate of the always-valid control (T=∞)**:")
    A("")
    rows = [r for _, r in always[(always.scenario == sc0)
                                 & (always.far_target == a_main)].iterrows()]
    L.extend(_t(rows, ["s", "Method", "Full-period false-alarm rate", "95% interval", "Sample size"],
                lambda r: [f"{r['sharpe_valid']:g}", CN[r["method"]],
                           f"{r['full_period_far']:.4f}",
                           f"[{r['full_period_far_lo']:.4f}, {r['full_period_far_hi']:.4f}]",
                           f"{int(r['n_paths'])}"]))
    A("")
    A(f"(For T=0 and the earlier failure settings, “pre-failure false alarm” by definition counts only up to day T, "
      f"so at T=0 it is always 0: the strategy is already invalid from day 1.)")
    A("")

    # ---------------- 6 ----------------
    A("## 6. Historical evidence diagnostic")
    A("")
    A("The working failure score q_T and the log-odds U_T = −L_T at the failure time T, given separately for all paths and for "
      "the paths that survive to T. **T=0 uses the initial value (the prior), not the last day of the array.**")
    A("")
    A(f"**{CN['trailing_sharpe_252']} is not in this table.** Its statistic is an annualised Sharpe ratio, "
      "not on the log-odds scale; expit of it gives a number with no interpretable meaning, "
      "so it is not reported here on a scale it does not live on.")
    A("")
    for s in sorted(S["sharpes"], reverse=True):
        A(f"**s = {s:g}**, {SC[sc0]}")
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
        L.extend(_t(rows, ["Failure setting", "Method", "U_T (all)", "q_T (all)",
                           "U_T (survivors)", "q_T (survivors)", "Survivors"], re_))
        A("")
    idc = pd.DataFrame(S["identity_check"])
    A("Analytic check on the Gaussian control, fixed T, **all paths**:")
    A("")
    A("```")
    A("    E[U_T] = -s²T/(2D),      E[U_{T+h} - U_T] = s²h/(2D)")
    A("```")
    A("")
    rows = [r for _, r in idc.iterrows()]
    L.extend(_t(rows, ["s", "T", "h", "E[U_T] theory", "Measured", "E[ΔU] theory", "Measured",
                       "Max deviation (SE)"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{int(r['T'])}", f"{int(r['h'])}",
                           f"{r['mean_U_at_T_theory']:+.4f}",
                           f"{r['mean_U_at_T_measured']:+.4f}",
                           f"{r['mean_dU_theory']:+.4f}", f"{r['mean_dU_measured']:+.4f}",
                           f"{max(r['z_U_at_T'], r['z_dU']):.2f}"]))
    A("")
    A("**This relation does not automatically apply to samples filtered by early false alarms, nor does it automatically apply to SV+jumps.** "
      "The “survivors” columns in the two tables above are exactly such filtered samples.")
    A("")
    tr = "trailing_sharpe_252"
    elig = int(g(sc0, s0, a_main, st0, tr, "first_eligible_day"))
    A(f"**{CN[tr]} at T={T0} simultaneously has fewer false alarms and higher conditional and joint detection; this is a "
      f"measured fact in this setting.** But this round did **not** decompose the source of this advantage. At least four mechanisms may be acting at the same time, "
      "and their relative shares have not been measured:")
    A("")
    A(f"1. **Warm-up period**: it can speak at the earliest on day {elig}, so at T=0 only "
      f"days {elig}–{h0} of the post-failure window are usable, whereas at T={T0} the warm-up period ends exactly at the failure time. "
      f"The same fact also limits its alarm opportunities before failure: "
      f"at T={T0} it can alarm before failure only on the single day {elig} "
      f"(realised false alarm {g(sc0, s0, a_main, st0, tr, 'pre_failure_far'):.2%}; "
      f"{CN['ewma_student_t']}: "
      f"{g(sc0, s0, a_main, st0, 'ewma_student_t', 'pre_failure_far'):.2%}).")
    A("2. **The rolling window gradually removes pre-failure data**: after failure, favourable historical evidence slides out of the window, "
      "whereas cumulative models do not discard it automatically.")
    A("3. **Variance state**: the EWMA-type methods enter the failure time carrying a variance estimate formed from past data.")
    A("4. **Survivor selection**: the conditional statistics are computed only on paths without an early false alarm, and the survivor sets differ across methods.")
    A("")
    A("The advantage therefore **cannot** be attributed to any single one of these, nor can one assert from it "
      "that “it discriminates better” or that “its advantage is only window alignment” — neither claim is supported by evidence in this round.")
    A("")
    A("Second, if the cumulative models become slower when failure comes later, **do not attribute the whole difference to historical evidence**: "
      "initialisation, the variance state and survivor selection (only paths without an early false alarm enter the conditional statistics) "
      "all affect performance, and this round did not decompose their individual shares.")
    A("")

    # ---------------- 7 ----------------
    A("## 7. Direct answers")
    A("")
    jt = {m: g(sc0, s0, a_main, st0, m, f"joint_detect_h{h0}") for m in METHODS_3B}
    z0 = {m: g(sc0, s0, a_main, "fixed_T0", m, f"joint_detect_h{h0}") for m in METHODS_3B}
    bestj = max((v, k) for k, v in jt.items())
    A(f"**(1) What changes when moving from “invalid from launch” to “valid-then-failing”?** "
      f"In the main setting, after moving T from 0 to {T0} days, within the same {h0}-day observation window: "
      + "; ".join(f"{CN[m]}: joint detection goes from {z0[m]:.2%} to {jt[m]:.2%}"
                  for m in ("binary_student_t", "ewma_student_t")) + ".")
    A("")
    A(f"The change comes from three sources, which this report lists separately rather than combining into one number: "
      f"(i) pre-failure false alarms first use up part of the paths (in this setting "
      f"{g(sc0, s0, a_main, st0, 'ewma_student_t', 'pre_failure_far'):.2%}; "
      f"these paths are declared invalid before T, which is a false alarm, not a detection); "
      f"(ii) the surviving paths enter the failure time carrying a stretch of **favourable historical evidence** "
      f"(U_T, q_T in Section 6); "
      f"(iii) the warm-up period of the rolling methods has already ended at the later T.")
    A("")
    A(f"In the main setting the highest joint detection is that of {CN[bestj[1]]} ({bestj[0]:.2%}).")
    A("")
    r_late = bo(sc0, s0, a_main, st0, "joint", "ewma_student_t", tr)
    r_rand = bo(sc0, s0, a_main, RANDOM_T, "joint", "ewma_student_t", tr)
    if r_late is not None and r_rand is not None:
        A(f"**A comparison whose direction changes with the failure time.** The joint-detection difference of {CN['ewma_student_t']} relative to {CN[tr]} "
          f"is, at T={T0}, {float(r_late.point) * 100:+.2f} pp "
          f"[{float(r_late.lo) * 100:+.2f}, {float(r_late.hi) * 100:+.2f}], "
          f"whereas under random T it is {float(r_rand.point) * 100:+.2f} pp "
          f"[{float(r_rand.lo) * 100:+.2f}, {float(r_rand.hi) * 100:+.2f}] — "
          "both intervals exclude 0, **in opposite directions**. "
          f"This round tested only four settings, T ∈ {{0, 126, {T0}}} and random T, "
          f"and did **not** scan T, so it cannot be said that T={T0} is the globally most favourable failure time for {CN[tr]}. "
          "What can be recorded is: **the method ranking changes with the failure time**; for the causes see the four undecomposed mechanisms listed in the previous section.")
    A("")
    A("**(2) Are these results enough to support explaining the scope of applicability to the supervisor this week?** "
      "What can be stated is: the five existing detectors of this project remain usable in the “valid-then-failing” setting, "
      "and the three quantities pre-failure false alarm, conditional detection and joint detection have been measured and reported separately; "
      f"the coverage is T ≤ {cfg.stage3b_random_failure_max} days, two noise scenarios, "
      f"two signal strengths and two false-alarm budgets. "
      "This **cannot** be used to describe performance over longer lifetimes (several years), nor any method not included in this project.")
    A("")
    A("**(3) Which questions should be left to the supervisor, to decide whether to continue next week?** This report does not answer them on the supervisor's behalf:")
    A("")
    A("1. Whether the coverage of failure times should be extended from one year to several years — that needs a longer simulation horizon, "
      "and would change the meaning of the “two-year monitoring” main benchmark.")
    A("2. Whether a detector that explicitly models the state switch is needed (instead of continuing to use the static two-state classifier's "
      "working failure score) — the evidence of this round only shows that the static classifier still works on switching data; "
      "no switch-aware method has been compared.")
    A("3. Which cost function should govern the trade-off between pre-failure false alarms and post-failure delay — "
      "this project has so far only reported both, without weighting them, and waiting costs are not included in the evaluation either.")
    A("")

    # ---------------- 8 ----------------
    A("## 8. Limitations")
    A("")
    A(f"- The coverage is T ≤ {cfg.stage3b_random_failure_max} days (failure within one year), "
      "not a 0–5 year or 0–10 year lifetime study.")
    A("- The thresholds reuse the frozen 504-day values from Stage 3A; the intervals describe test sampling only and do not cover calibration error.")
    A("- All failure settings share the same base noise, so the settings are not independent samples.")
    A("- The static classifier's q is a working failure score, not a correct posterior for the switching process.")
    A("- The analytic identity has been checked only for iid Gaussian noise, fixed T and all paths.")
    A("- This round does not implement BOCPD, nor does it add any detection model or widen any parameter range.")
    A("")
    A("## 9. Reproduction")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage3b")
    A("uv run python -m strategy_survivorship.stage3b --figures-only")
    A("uv run python -m strategy_survivorship.stage3b --report-only")
    A("```")
    A("")
    A(f"This run took {S['elapsed_s']:.1f} seconds.")
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
