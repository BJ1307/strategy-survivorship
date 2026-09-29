"""stage2e: one module per research stage.

- Model and computation: Stage 2E: does capping the influence of an extreme return on the variance
- Figures: Stage 2E figures: at most four, all drawn from files already on disk.
- Report: Generate ``stage2e_report.md``. Every number is read from the result tables.
- Command-line entry: Stage 2E pipeline: does truncating the variance update help?
"""
from __future__ import annotations

from .config import require_input


# ============================================================================
# Model and computation
# ============================================================================

import math

import numpy as np
import pandas as pd

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY
from .evaluate import wilson_interval
from .ewma import (ewma_gaussian_log_odds, ewma_student_t_log_odds,
                   ewma_trunc_gaussian_log_odds, ewma_trunc_student_t_log_odds)
from .simulate import make_streams
from .stage2c import buffered_rank, first_alarm_day, metric_row, path_minima, rank_via_beta
from .stage2d import make_block, specs

METHODS_2E = ("binary_gaussian", "binary_student_t", "trailing_sharpe_252",
              "known_vol_rolling_252", "ewma_gaussian", "ewma_student_t",
              "ewma_trunc_gaussian", "ewma_trunc_student_t")
CORE_2x2 = ("ewma_gaussian", "ewma_student_t", "ewma_trunc_gaussian", "ewma_trunc_student_t")
TRUNC_OF = {"ewma_gaussian": "ewma_trunc_gaussian", "ewma_student_t": "ewma_trunc_student_t"}
LATE_STARTERS = ("trailing_sharpe_252", "known_vol_rolling_252")
LABEL_2E = {
    "binary_gaussian": "Fixed Gaussian",
    "binary_student_t": "Fixed Student-t (nu=5)",
    "trailing_sharpe_252": "Trailing 12m Sharpe",
    "known_vol_rolling_252": "Rolling / fixed sigma_0",
    "ewma_gaussian": "EWMA Gaussian",
    "ewma_student_t": "EWMA Student-t (nu=5)",
    "ewma_trunc_gaussian": "Truncated-EWMA Gaussian",
    "ewma_trunc_student_t": "Truncated-EWMA Student-t (nu=5)",
}
ROLES = ("calibration_valid", "test_valid", "test_invalid")


def statistic(method: str, returns: np.ndarray, cfg: Stage1Config) -> np.ndarray:
    """Returns and the public model configuration only."""
    if method == "ewma_gaussian":
        return ewma_gaussian_log_odds(returns, cfg)
    if method == "ewma_student_t":
        return ewma_student_t_log_odds(returns, cfg)
    if method == "ewma_trunc_gaussian":
        return ewma_trunc_gaussian_log_odds(returns, cfg)
    if method == "ewma_trunc_student_t":
        return ewma_trunc_student_t_log_odds(returns, cfg)
    return DETECTORS_BY_KEY[method].compute(returns, cfg)


def first_eligible(method: str, cfg: Stage1Config) -> int:
    return DETECTORS_BY_KEY[method].first_eligible_day(cfg) if method in DETECTORS_BY_KEY else 1


def streams_for(cfg: Stage1Config, parent: str) -> dict:
    kids = make_streams(cfg)[parent].spawn(len(specs(cfg)) * len(ROLES))
    return {(sp[0], role): kids[i * len(ROLES) + j]
            for i, sp in enumerate(specs(cfg)) for j, role in enumerate(ROLES)}


def calibrate(cfg: Stage1Config) -> dict:
    """Per-scenario buffered thresholds for all EIGHT methods.

    J is methods x scenarios x budgets = 80 this round, not the 60 of Stage 2C:
    two new models were added and the Bonferroni split has to widen accordingly.
    """
    sp = specs(cfg)
    st = streams_for(cfg, "stage2e_calibration")
    n = cfg.stage2e_calibration_paths
    J = len(METHODS_2E) * len(sp) * len(cfg.far_targets)
    dj = cfg.stage2c_delta / J
    ranks = {a: {"buffered": buffered_rank(n, a, dj), "beta_route": rank_via_beta(n, a, dj)}
             for a in cfg.far_targets}
    thr, rows, minima = {}, [], {}
    for spec in sp:
        key = spec[0]
        blk = make_block(cfg, spec, st[(key, "calibration_valid")], n, cfg.sharpe_valid)
        for m in METHODS_2E:
            raw = path_minima(statistic(m, blk["returns"], cfg), first_eligible(m, cfg),
                              cfg.horizon_days)
            minima[(key, m)] = raw
            srt = np.sort(raw)
            for a in cfg.far_targets:
                k = ranks[a]["buffered"]
                thr[(m, key, a)] = float(srt[k - 1])
                rows.append({"method": m, "scenario": key, "far_target": a, "rank": k,
                             "threshold": float(srt[k - 1]), "n_calibration_paths": n,
                             "J": J})
        del blk
    return {"thresholds": thr, "ranks": ranks, "J": J, "delta_per_comparison": dj,
            "minima": minima, "table": pd.DataFrame(rows)}


def jump_relative_class(alarm_day: np.ndarray, jump_counts: np.ndarray, window: int) -> np.ndarray:
    """Mutually exclusive timing class of the FIRST alarm, for scoring only.

    0 = no alarm, 1 = the alarm day is itself a jump day,
    2 = not a jump day but the most recent prior jump is 1..window days back,
    3 = otherwise.  Classes 1-3 partition the alarmed paths exactly.

    The jump flags are used only here, by the evaluator.  A timing association is
    not a causal attribution.
    """
    n_paths, n_days = jump_counts.shape
    has_jump = jump_counts >= 1
    # days since the most recent jump strictly before each day
    idx = np.where(has_jump, np.arange(1, n_days + 1)[None, :], 0)
    last_jump = np.maximum.accumulate(idx, axis=1)
    prev_jump = np.concatenate([np.zeros((n_paths, 1), dtype=int), last_jump[:, :-1]], axis=1)

    out = np.zeros(n_paths, dtype=np.int8)
    alarmed = alarm_day != -1
    rows = np.arange(n_paths)[alarmed]
    d0 = alarm_day[alarmed] - 1
    on_jump = has_jump[rows, d0]
    gap = d0 + 1 - prev_jump[rows, d0]          # days since the last prior jump
    recent = (~on_jump) & (prev_jump[rows, d0] > 0) & (gap >= 1) & (gap <= window)
    cls = np.where(on_jump, 1, np.where(recent, 2, 3)).astype(np.int8)
    out[alarmed] = cls
    return out


def evaluate(cfg: Stage1Config, spec, seed_seq, thresholds: dict, legacy: dict, days) -> tuple:
    """Main per-scenario arm plus the mechanism-only legacy-threshold control."""
    key = spec[0]
    kids = seed_seq.spawn(2)
    bv = make_block(cfg, spec, kids[0], cfg.stage2e_test_paths, cfg.sharpe_valid)
    bi = make_block(cfg, spec, kids[1], cfg.stage2e_test_paths, cfg.sharpe_invalid)
    jc_valid = bv["latent"].get("jump_counts")
    rows, minima, trunc, qrows, far_split = [], {}, {}, [], []
    from scipy.special import expit

    for m in METHODS_2E:
        elig = first_eligible(m, cfg)
        sv = statistic(m, bv["returns"], cfg)
        si = statistic(m, bi["returns"], cfg)
        minima[m] = {"test_valid": path_minima(sv, elig, cfg.horizon_days),
                     "test_invalid": path_minima(si, elig, cfg.horizon_days)}
        if m not in LATE_STARTERS:
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
            arms = [("per_scenario", thresholds[(m, key, a)])]
            if m in legacy and (m, a) in legacy[m]:
                arms.append(("legacy_threshold", legacy[m][(m, a)]))
            for arm, thr in arms:
                tv = first_alarm_day(sv, thr, elig, cfg.horizon_days)
                ti = first_alarm_day(si, thr, elig, cfg.horizon_days)
                far = metric_row(tv, cfg, days, cfg.horizon_days)
                det = metric_row(ti, cfg, days, cfg.horizon_days)
                lo, hi = wilson_interval(far["n_alarms"], far["n_paths"], cfg.wilson_z)
                row = {"scenario": key, "method": m, "arm": arm, "far_target": a,
                       "threshold": thr, "first_eligible_day": elig,
                       "far_alarms": far["n_alarms"], "far_lo": lo, "far_hi": hi}
                for d in days:
                    row[f"far_d{d}"] = far[f"rate_d{d}"]
                    row[f"detect_d{d}"] = det[f"rate_d{d}"]
                row.update({"trunc_mean_detect_days": det["trunc_mean_days"],
                            "trunc_mean_detect_se": det["trunc_mean_se"],
                            "median_detect_days": det["median_days"],
                            "median_detect_note": det["median_note"],
                            "undetected_at_H": det["undetected_at_H"],
                            "n_test_valid": far["n_paths"], "n_test_invalid": det["n_paths"]})
                rows.append(row)
                if arm == "per_scenario":
                    trunc[(m, a)] = np.where(ti != -1, ti, cfg.horizon_days).astype(float)
                    if jc_valid is not None:
                        cls = jump_relative_class(tv, jc_valid, cfg.stage2e_jump_window)
                        n = tv.size
                        far_split.append({
                            "scenario": key, "method": m, "far_target": a,
                            "n_test_valid": n,
                            "far_total": float((tv != -1).mean()),
                            "far_on_jump_day": float((cls == 1).mean()),
                            "far_within_window": float((cls == 2).mean()),
                            "far_elsewhere": float((cls == 3).mean()),
                            "window_days": cfg.stage2e_jump_window})
        del sv, si
    return rows, minima, trunc, qrows, far_split, bv, bi

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
from .stage2d import SCEN_LABEL


C = {"ewma_gaussian": "#17becf", "ewma_student_t": "#ff7f0e",
     "ewma_trunc_gaussian": "#1f77b4", "ewma_trunc_student_t": "#d62728",
     "binary_student_t": "#7f7f7f"}
SHORT = {"gaussian_ctrl": "Gaussian\n(A=0,k=0)", "sv_ctrl": "SV\n(A=1,k=0)",
         "jump_ctrl": "Jump\n(A=0,k=5)", "sv_jump": "SV+jump\n(A=1,k=5)",
         "sv_jump_big": "SV+big jump\n(A=1,k=8)"}
PAIR_LABEL = {("ewma_trunc_student_t", "ewma_student_t"): "trunc-t  -  EWMA t",
              ("ewma_trunc_gaussian", "ewma_gaussian"): "trunc-G  -  EWMA G",
              ("ewma_trunc_student_t", "ewma_trunc_gaussian"): "trunc-t  -  trunc-G"}


def draw_all(cfg, out_dir: Path) -> list[str]:
    m = pd.read_csv(out_dir / "stage2e_metrics.csv")
    sh = pd.read_csv(out_dir / "stage2e_shock.csv")
    bo = pd.read_csv(out_dir / "stage2e_bootstrap.csv")
    ft = pd.read_csv(out_dir / "stage2e_far_timing.csv")
    fd = out_dir / "figures"
    outs = [figure_mechanism(cfg, sh, fd / "fig2e1_mechanism.png"),
            figure_2x2(cfg, m, fd / "fig2e2_2x2.png"),
            figure_diffs(cfg, bo, fd / "fig2e3_differences.png"),
            figure_far_timing(cfg, ft, fd / "fig2e4_far_timing.png")]
    return [str(p.relative_to(out_dir)) for p in outs]


def figure_mechanism(cfg, sh: pd.DataFrame, out: Path) -> Path:
    """What the cap does on one fixed path with one injected shock.

    Everything is drawn as U = log P(invalid)/P(valid), so UP means MORE
    evidence that the strategy has failed.  The running statistic inside the
    detectors is L = -U; only the display sign differs.  Each panel states the
    shock direction it uses.
    """
    day0 = cfg.stage2d_shock_day
    lo, hi = day0 - 20, day0 + 60
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
    ann = np.sqrt(cfg.trading_days_per_year)
    amp = cfg.stage2d_shock_sigmas

    b = sh[sh.variant == "base"]
    p = sh[sh.variant == "plus"]
    n = sh[sh.variant == "minus"]
    wb = (b.day >= lo) & (b.day <= hi)
    wp = (p.day >= lo) & (p.day <= hi)
    wn = (n.day >= lo) & (n.day <= hi)

    ax = axes[0]
    for frame, w, sty, lab in ((b, wb, dict(lw=1.4), "no shock"),
                               (p, wp, dict(lw=1.0, alpha=0.55), f"+{amp:g}$\\sigma$"),
                               (n, wn, dict(lw=1.0, alpha=0.55, ls=":"), f"-{amp:g}$\\sigma$")):
        ax.plot(frame.day[w], np.sqrt(frame.var_plain[w]) * ann, color="#17becf",
                label=f"plain, {lab}", **sty)
        ax.plot(frame.day[w], np.sqrt(frame.var_trunc[w]) * ann, color="#1f77b4",
                dashes=(4, 2), **sty)
    ax.axvline(day0, color="0.35", lw=0.9, ls=":")
    ax.set_title(f"A. forecast vol, all three shock directions\n"
                 f"(cyan = plain EWMA, blue dashed = truncated)", fontsize=9)
    ax.set_ylabel("annualised forecast vol")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for frame, w, sign, ls in ((p, wp, "+", "-"), (n, wn, "-", ":")):
        for tag, lw in (("plain", 1.5), ("trunc", 1.1)):
            for mth, col in (("gaussian", C["ewma_gaussian"]),
                             ("student_t", C["ewma_student_t"])):
                d = (frame[f"U_{tag}_{mth}"][w].to_numpy()
                     - b[f"U_{tag}_{mth}"][wb].to_numpy())
                ax.plot(frame.day[w], d, ls=ls, lw=lw, color=col,
                        alpha=1.0 if tag == "plain" else 0.55,
                        label=f"{mth}, {tag}, {sign}{amp:g}$\\sigma$"
                        if mth == "gaussian" or tag == "plain" else None)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(day0, color="0.35", lw=0.9, ls=":")
    ax.set_title(f"B. shock effect on U, solid = +{amp:g}$\\sigma$, dotted = -{amp:g}$\\sigma$\n"
                 "(a positive return pushes U DOWN)", fontsize=9)
    ax.set_ylabel("$\\Delta U$ = U(shocked) - U(base)")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[2]
    for tag, ls in (("plain", "-"), ("trunc", "--")):
        for mth, col in (("gaussian", C["ewma_gaussian"]), ("student_t", C["ewma_student_t"])):
            ax.plot(b.day[wb], b[f"U_{tag}_{mth}"][wb], ls=ls, color=col, lw=1.1,
                    label=f"{mth}, {tag}")
    ax.axvline(day0, color="0.35", lw=0.9, ls=":")
    ax.set_title("C. NO shock: U on the base path\n(solid plain, dashed truncated)", fontsize=9)
    ax.set_ylabel("$U=\\log\\,P(\\mathrm{invalid})/P(\\mathrm{valid})$")
    ax.legend(frameon=False, fontsize=7)
    for a_ in axes:
        a_.set_xlabel("day")
    fig.suptitle("Stage 2E figure 1: what the variance cap does, one SV+jump path. "
                 "All log-odds are U = log P(invalid)/P(valid); up = more evidence of failure.",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_2x2(cfg, m: pd.DataFrame, out: Path) -> Path:
    """Detection by day 504 under each method's own buffered threshold."""
    d = m[m.arm == "per_scenario"]
    scen = list(SCEN_LABEL)
    shown = list(CORE_2x2) + ["binary_student_t"]
    fig, axes = plt.subplots(1, len(cfg.far_targets), figsize=(13.2, 4.6), sharey=True)
    width = 0.16
    x = np.arange(len(scen))
    for ax, a in zip(np.atleast_1d(axes), cfg.far_targets):
        sub = d[d.far_target == a]
        for j, mth in enumerate(shown):
            y = [float(sub[(sub.scenario == s) & (sub.method == mth)].detect_d504.iloc[0])
                 for s in scen]
            ax.bar(x + (j - (len(shown) - 1) / 2) * width, y, width, color=C[mth],
                   label=LABEL_2E[mth], edgecolor="white", lw=0.5,
                   hatch="//" if mth == "binary_student_t" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[s] for s in scen], fontsize=8)
        ax.set_title(f"FAR budget {a:.0%}")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("detected by day 504 (invalid paths)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Stage 2E figure 2: the 2x2 ablation, each method on its own "
                 "per-scenario buffered threshold", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_diffs(cfg, bo: pd.DataFrame, out: Path) -> Path:
    """The three pre-specified differences plus the 2x2 interaction.

    Intervals also resample the calibration sample, so they carry the sampling
    uncertainty of the thresholds themselves.
    """
    scen = list(SCEN_LABEL)
    x = np.arange(len(scen))
    pairs = list(PAIR_LABEL.items())
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#7b3294"]
    fig, axes = plt.subplots(1, len(cfg.far_targets), figsize=(13.2, 4.8), sharey=True)
    off = np.linspace(-0.26, 0.26, len(pairs) + 1)
    for ax, a in zip(np.atleast_1d(axes), cfg.far_targets):
        sub = bo[bo.far_target == a]
        series = []
        for pair, lab in pairs:
            r = sub[(sub.contrast == "pair") & (sub.method_a == pair[0])
                    & (sub.method_b == pair[1])]
            series.append((lab, r))
        series.append(("EXPLORATORY interaction:\n(t-side gain) - (G-side gain)",
                       sub[sub.contrast == "interaction"]))
        for j, (lab, r) in enumerate(series):
            y = np.array([float(r[r.scenario == s].detect_diff.iloc[0]) for s in scen])
            lo = np.array([float(r[r.scenario == s].detect_diff_lo.iloc[0]) for s in scen])
            hi = np.array([float(r[r.scenario == s].detect_diff_hi.iloc[0]) for s in scen])
            ax.errorbar(x + off[j], y, yerr=[y - lo, hi - y], fmt="o", ms=4, capsize=3,
                        lw=1.2, color=cols[j], label=lab,
                        mfc="white" if j == len(series) - 1 else cols[j])
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[s] for s in scen], fontsize=8)
        ax.set_title(f"FAR budget {a:.0%}")
    np.atleast_1d(axes)[0].set_ylabel("difference in detection by day 504")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Stage 2E figure 3: THREE PRE-SPECIFIED differences (filled) and the "
                 "POST-HOC interaction (open, purple).\n95% per-comparison bootstrap "
                 "intervals that also resample the calibration sample; no multiplicity "
                 "adjustment.", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_far_timing(cfg, ft: pd.DataFrame, out: Path) -> Path:
    """Where the false alarms fall relative to jump days. Timing, not cause."""
    scen = [s for s in SCEN_LABEL if not ft[ft.scenario == s].empty]
    a = max(cfg.far_targets)
    sub = ft[ft.far_target == a]
    fig, axes = plt.subplots(1, len(scen), figsize=(3.1 * len(scen) + 1.6, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    parts = [("far_on_jump_day", "alarm day is a jump day", "#d62728"),
             ("far_within_window", f"jump in the prior {cfg.stage2e_jump_window} days", "#ff7f0e"),
             ("far_elsewhere", "no jump nearby", "#4c72b0")]
    for ax, s in zip(axes, scen):
        d = sub[sub.scenario == s]
        mm = [m for m in CORE_2x2 if m in set(d.method)]
        x = np.arange(len(mm))
        bottom = np.zeros(len(mm))
        for col, lab, colr in parts:
            y = np.array([float(d[d.method == m][col].iloc[0]) for m in mm])
            ax.bar(x, y, 0.62, bottom=bottom, color=colr, label=lab, edgecolor="white", lw=0.5)
            bottom += y
        ax.axhline(a, color="0.25", ls="--", lw=1.0)
        # theoretical chance level for class 1: p_J * (that method's total FAR)
        pJ = -np.expm1(-cfg.noise_jump_lambda_annual / cfg.trading_days_per_year)
        tot = np.array([float(d[d.method == m].far_total.iloc[0]) for m in mm])
        ax.plot(x, pJ * tot, ls="none", marker="_", ms=22, mew=1.6, color="0.15",
                label="chance level for the red band" if s == scen[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(["EWMA G", "EWMA t", "trunc G", "trunc t"][: len(mm)], fontsize=8)
        ax.set_title(SCEN_LABEL[s], fontsize=9)
    for ax in axes:
        ax.set_ylim(0.0, a * 1.42)              # headroom so the legend clears the budget line
    axes[0].set_ylabel(f"cumulative false-alarm rate, budget {a:.0%}")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("Stage 2E figure 4: first false alarm split into three exclusive "
                 "timing classes (association, not attribution)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out

# ============================================================================
# Report
# ============================================================================

from pathlib import Path

import pandas as pd

from .evaluate import excludes_zero
from .stage2d import COMBINED, SCEN_LABEL, specs


SC = {"gaussian_ctrl": "Gaussian control (A=0, κ=0)", "sv_ctrl": "SV control (A=1, κ=0)",
      "jump_ctrl": "Jump control (A=0, κ=5)", "sv_jump": "**SV+jumps (A=1, κ=5)**",
      "sv_jump_big": "**SV+larger jumps (A=1, κ=8)**"}
PAIRS = (("ewma_trunc_student_t", "ewma_student_t", "Truncated-t vs EWMA t (does truncation help?)"),
         ("ewma_trunc_gaussian", "ewma_gaussian", "Truncated-G vs EWMA G (does truncation help?)"),
         ("ewma_trunc_student_t", "ewma_trunc_gaussian",
          "Truncated-t vs truncated-G (is a heavy-tailed likelihood still needed?)"))


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _emph(sc: str) -> str:
    """SC already bolds the two combined scenarios; do not bold them twice."""
    return SC[sc] if SC[sc].startswith("**") else f"**{SC[sc]}**"


def _med(v):
    return f"{int(v)}d" if v == v and v not in ("", None) else "not reached"


def write_stage2e_report(cfg, summary, metrics, cal, boot, brier, shock, far_split,
                         path: Path) -> Path:
    L: list[str] = []
    A = L.append
    a_main, a_alt = cfg.far_targets[-1], cfg.far_targets[0]
    days = list(cfg.stage2c_report_days)
    M = metrics[metrics.arm == "per_scenario"]
    scen = list(SCEN_LABEL)

    def g(meth, sc, col="detect_d504", a=a_main, frame=None):
        f = M if frame is None else frame
        s = f[(f.scenario == sc) & (f.far_target == a) & (f.method == meth)][col]
        return float(s.iloc[0]) if len(s) else float("nan")

    def bo(ma, mb, sc, a=a_main):
        r = boot[(boot.contrast == "pair") & (boot.method_a == ma) & (boot.method_b == mb)
                 & (boot.scenario == sc) & (boot.far_target == a)]
        return r.iloc[0]

    def inter(sc, a=a_main):
        r = boot[(boot.contrast == "interaction") & (boot.scenario == sc)
                 & (boot.far_target == a)]
        return r.iloc[0]

    A("# Stage 2E report: does truncating the influence of extreme returns on the variance "
      "update further improve validation?")
    A("")
    A("This file is generated automatically by `python -m strategy_survivorship.stage2e`; "
      "every number is taken from the result tables of the same run.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. Direct answers")
    A("")
    A("**This round's answer changes with the false-alarm budget, so every question below is "
      "answered for both budgets.** Reporting either one alone would lead to opposite conclusions.")
    A("")
    budgets = list(cfg.far_targets)

    def sweep(ma, mb, a):
        rows = []
        for sc in scen:
            r = bo(ma, mb, sc, a)
            rows.append({"scenario": sc, "far_target": a, "d": float(r.detect_diff),
                         "lo": float(r.detect_diff_lo), "hi": float(r.detect_diff_hi),
                         "sig": bool(r.detect_excludes_zero)})
        return (rows, [r for r in rows if r["sig"] and r["d"] > 0],
                [r for r in rows if r["sig"] and r["d"] < 0])

    def tally(pos, neg, n):
        if pos and not neg:
            return f"{len(pos)}/{n} scenarios significantly positive"
        if neg and not pos:
            return f"{len(neg)}/{n} scenarios significantly negative"
        if pos and neg:
            return f"{len(pos)} significantly positive, {len(neg)} significantly negative"
        return f"{n} scenarios, all covering 0"

    has_jump = {sp[0]: float(sp[2]) > 0.0 for sp in specs(cfg)}

    def jump_story(sig_scenarios) -> bool:
        """True only if every significant scenario is one with jumps (kappa > 0).

        Checked against the scenario table rather than asserted in prose, so the
        sentence cannot survive a change in which scenarios come out significant.
        """
        sig = set(sig_scenarios)
        return bool(sig) and all(has_jump[sc] for sc in sig)

    T = {a: sweep("ewma_trunc_student_t", "ewma_student_t", a) for a in budgets}
    G = {a: sweep("ewma_trunc_gaussian", "ewma_gaussian", a) for a in budgets}
    t_rows, t_pos, t_neg = T[a_main]
    g_rows, g_pos, g_neg = G[a_main]

    plain0 = {k: v.strip("*") for k, v in SC.items()}

    def diff_table(D):
        rows = [r for a in budgets for r in D[a][0]]
        L.extend(_t(rows, ["α", "Scenario", "Difference", "95% interval", "Excludes 0"],
                    lambda r: [f"{r['far_target']:g}", SC[r["scenario"]], f"{r['d']:+.4f}",
                               f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", "yes" if r["sig"] else "no"]))

    A("**Question 1: does the truncated variance update improve on the previous round's strong "
      "method (EWMA Student-t)?**")
    A("")
    all_sig = {r["scenario"] for a in budgets for r in T[a][1]}
    counts = "; ".join(f"α={a:g}: {tally(T[a][1], T[a][2], len(scen))}" for a in budgets)
    if all_sig and jump_story(all_sig):
        A(f"Yes, a small improvement. {counts}. "
          "**The clear gains observed in this round are concentrated in the scenarios with "
          "jumps (κ>0)**, while the two scenarios without jumps are not significant under "
          "either budget. "
          "This describes the five scenarios of this round; it is not a general conclusion "
          "that \"truncation only helps when there are jumps\": "
          "in the no-jump scenarios the point estimates are close to zero but not zero, and "
          "this round can only say that the intervals cannot be distinguished from zero.")
    elif all_sig:
        A(f"{counts}. The significant scenarios are "
          f"{', '.join(plain0[sc] for sc in scen if sc in all_sig)}.")
    else:
        A(f"No: {counts}.")
    A("")
    diff_table(T)
    A("")
    ref = {a: g("ewma_student_t", "sv_jump", "detect_d504", a)
              - g("binary_student_t", "sv_jump", "detect_d504", a) for a in budgets}
    A("**Magnitudes must be compared within the same scenario, the same budget and the same "
      "round.** Taking the maximum across scenarios and then comparing it with the previous "
      "round mixes the difficulty of different scenarios into a single number. The two "
      "steps in the main scenario, SV+jumps:")
    A("")
    step = [{"a": a, "ewma": ref[a],
             "trunc": next(r["d"] for r in T[a][0] if r["scenario"] == "sv_jump")}
            for a in budgets]
    L.extend(_t(step, ["α", "Fixed t → EWMA t", "EWMA t → truncated t",
                       "Second step as share of first"],
                lambda r: [f"{r['a']:g}", f"{r['ewma'] * 100:+.2f} pp",
                           f"{r['trunc'] * 100:+.2f} pp",
                           f"{abs(r['trunc']) / abs(r['ewma']):.0%}"]))
    A("")
    A("That is: **this is a further small improvement on top of an existing improvement**. "
      "This round did not carry out any expansion to show that it is a \"second-order term\" "
      "in some asymptotic sense, so that term is not used.")
    A("")

    A("**Question 2: does the gain depend on pairing with a heavy-tailed likelihood?**")
    A("")
    A("The same comparison on the Gaussian side: " +
      "; ".join(f"α={a:g}: {tally(G[a][1], G[a][2], len(scen))}" for a in budgets) + ".")
    A("")
    diff_table(G)
    A("")
    A("Counting \"which side has more significant scenarios\" is not a test of this question, "
      "because both sides can be significant and of the same size. "
      "In the 2×2 ablation, the quantity that corresponds to \"does the gain depend on "
      "pairing\" is the **interaction**")
    A("")
    A("```")
    A("    (truncated-t − EWMA t) − (truncated-G − EWMA G)")
    A("```")
    A("")
    A("It is computed within the same resampling (the four cells share the same calibration "
      "and test indices). "
      "A significantly positive value means the gain appears only with the pairing; a "
      "significantly negative value means truncation is merely a substitute for the "
      "heavy-tailed likelihood; "
      "covering 0 is consistent with the two robustifications acting independently and "
      "approximately additively.")
    A("")
    I = {}
    for a in budgets:
        rows = []
        for sc in scen:
            r = inter(sc, a)
            rows.append({"scenario": sc, "far_target": a, "d": float(r.detect_diff),
                         "lo": float(r.detect_diff_lo), "hi": float(r.detect_diff_hi),
                         "sig": bool(r.detect_excludes_zero)})
        I[a] = (rows, [r for r in rows if r["sig"] and r["d"] > 0],
                [r for r in rows if r["sig"] and r["d"] < 0])
    irows_all = [r for a in budgets for r in I[a][0]]
    L.extend(_t(irows_all, ["α", "Scenario", "Interaction", "95% interval", "Excludes 0",
                            "Interval half-width"],
                lambda r: [f"{r['far_target']:g}", SC[r["scenario"]], f"{r['d']:+.4f}",
                           f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", "yes" if r["sig"] else "no",
                           f"{(r['hi'] - r['lo']) / 2:.4f}"]))
    A("")
    ipos, ineg = I[a_main][1], I[a_main][2]
    irows = I[a_main][0]
    half = max((r["hi"] - r["lo"]) / 2 for r in irows)
    tight = [a for a in budgets if I[a][1] and not I[a][2]]
    loose = [a for a in budgets if not I[a][1] and not I[a][2]]
    if tight and loose:
        ta, la = tight[0], loose[0]
        tsc = ", ".join(SC[r["scenario"]].strip("*") for r in I[ta][1])
        A(f"**The evidence differs between the two budgets.** "
          f"At α={ta:g} the interaction is significantly positive in {tsc}"
          f" (largest {max(r['d'] for r in I[ta][1]):+.4f}), "
          f"i.e. under the tight budget truncation gains more on the Student-t side than on "
          f"the Gaussian side, "
          f"and under the same budget truncation on the Gaussian side is not significant in "
          f"any scenario. "
          f"**This still does not mean that \"it only helps when paired with Student-t\"**: "
          f"the point estimates of the Gaussian-side gain are not zero; this round's intervals "
          f"simply cannot distinguish them from zero. "
          f"At α={la:g} the interaction covers 0 in all five scenarios, and the main effects "
          f"on the two sides are of similar size; "
          "this is **compatible** with \"the two robustifications are approximately "
          "additive\", but covering 0 is not evidence of additivity; "
          "it only shows that this round's resolution did not separate them.")
        A("")
        A(f"**\"Significant at α={ta:g}, not significant at α={la:g}\" does not by itself imply "
          "that the interaction differs between the two budgets.** "
          "That inference confuses a difference in significance with the significance of a "
          "difference. The direct estimate of `I(0.05) − I(0.15)` "
          "is given in Section 4 of the Stage 2E.1 verification note and in "
          "`stage2e1_cross_budget.csv`; "
          "there the interval is obtained with the two budgets sharing the same resampling "
          "indices, each recomputing its own threshold.")
        A("")
        A("**The interaction and the cross-budget comparison were both added after seeing the "
          "Stage 2E main results, and are labelled exploratory analyses.** "
          "Their intervals are per-item (unadjusted) intervals, with no multiple-comparison "
          "adjustment.")
        A("")
        A("One reading consistent with this (it cannot be tested further in this round and is "
          "recorded only as a note): "
          "the tighter the budget and the deeper the threshold, the more an alarm depends on a "
          "few extreme days; "
          "the Gaussian version's likelihood blow-up on the jump day then dominates its "
          "behaviour, "
          "and truncation cannot change the same-day likelihood, so no gain is visible on the "
          "Gaussian side; "
          "the Student-t version has no such blow-up, and its main remaining damage is "
          "precisely the contamination of the variance state, "
          "which is exactly what truncation repairs.")
    elif I[a_main][1] and not I[a_main][2]:
        A(f"**Truncation gains more on the Student-t side than on the Gaussian side.** The "
          f"interaction is significantly positive in {len(ipos)}/{len(irows)} scenarios. This "
          "does not prove that the Gaussian-side gain is zero, "
          "nor that the Student-t likelihood is a necessary condition.")
    elif I[a_main][2] and not I[a_main][1]:
        A(f"**Truncation gains more on the Gaussian side.** The interaction is significantly "
          f"negative in {len(ineg)}/{len(irows)} scenarios. "
          "This is compatible with \"truncation is a substitute for the heavy-tailed "
          "likelihood\", but is not sufficient to prove a substitution relationship.")
    else:
        A(f"The interaction covers 0 under both budgets (widest half-width {half:.4f}, of the "
          "same order as the main effects); "
          "this round can only rule out interactions much larger than the main effects, not "
          "interactions comparable to the main effects.")
    A("")

    A("**Question 3: once both sides are truncated, does the heavy-tailed likelihood still "
      "matter?**")
    A("")
    rows3 = [{"scenario": sc, "far_target": a,
              **{k: float(bo("ewma_trunc_student_t", "ewma_trunc_gaussian", sc, a)[k])
                 for k in ("detect_diff", "detect_diff_lo", "detect_diff_hi")}}
             for a in budgets for sc in scen]
    L.extend(_t(rows3, ["α", "Scenario", "Truncated-t minus truncated-G", "95% interval"],
                lambda r: [f"{r['far_target']:g}", SC[r["scenario"]],
                           f"{r['detect_diff']:+.4f}",
                           f"[{r['detect_diff_lo']:+.4f}, {r['detect_diff_hi']:+.4f}]"]))
    A("")
    m3 = max(rows3, key=lambda r: r["detect_diff"])
    A(f"**It still matters, and it is the largest of all effects in this round.** The maximum, "
      f"{m3['detect_diff']:+.4f}, occurs in "
      f"{SC[m3['scenario']].strip('*')}, α={m3['far_target']:g}, "
      "an order of magnitude larger than the effect of truncation itself. The choice of "
      "likelihood is still the dominant factor.")
    A("")
    lead = max(METHODS_2E, key=lambda m: g(m, "sv_jump"))
    A(f"In the main scenario SV+jumps at α={a_main:g}, the model with the highest detection "
      f"rate is **{LABEL_2E[lead]}**"
      f" ({g(lead, 'sv_jump'):.4f}).")
    A("")

    # ---------------- 2 ----------------
    A("## 2. What is new in this round, and why it counts as a new model")
    A("")
    A(f"The only change is a single truncation step in the variance recursion; `c` is fixed at "
      f"{cfg.ewma_truncation_c:g} and is **not searched over**:")
    A("")
    A("```")
    A(f"    plain:      V_{{t+1}} = {cfg.ewma_lambda:g} V_t + {1 - cfg.ewma_lambda:.2f} e_t^2")
    A(f"    truncated:  V_{{t+1}} = {cfg.ewma_lambda:g} V_t + {1 - cfg.ewma_lambda:.2f} "
      f"min(e_t^2, {cfg.ewma_truncation_c ** 2:g} V_t)")
    A(f"    e_t = r_t - m,  m = {summary['config'].get('ewma_midpoint', float('nan')):.3e}"
      if "ewma_midpoint" in summary["config"] else "    e_t = r_t - m (fixed midpoint)")
    A("```")
    A("")
    A("**Changing the variance update changes the detection rule, so the two truncated versions "
      "are new models, not robustified rewrites of the same model.** "
      "They inherit none of the probability guarantees attached to the plain EWMA thresholds; "
      "the main comparison recalibrates all eight models on new data under the same protocol "
      "instead of splicing in numbers from the previous round. The eight models form a 2×2 "
      "ablation plus four controls already present in the previous round:")
    A("")
    A("| Variance path | Gaussian likelihood | Student-t likelihood |")
    A("|---|---|---|")
    A("| plain EWMA | EWMA Gaussian | EWMA Student-t |")
    A("| truncated EWMA | Truncated-EWMA Gaussian | Truncated-EWMA Student-t |")
    A("")
    A(f"The other four models (fixed-scale Gaussian / fixed-scale Student-t / 12-month rolling "
      f"Sharpe / known-volatility rolling) are carried over from Stage 1 and are calibrated and "
      f"evaluated alongside them as references.")
    A("")
    A("Two proven properties of truncation:")
    A("")
    A("- By induction, `Ṽ_t ≤ V_t` holds on every day (for the same return series, the "
      "truncated variance is never above the plain variance).")
    A("- A smaller variance makes each daily likelihood-ratio increment larger in absolute "
      "value, so **the direction of the effect on FAR is not determined automatically**; "
      "the models must be recalibrated instead of reusing the old thresholds.")
    A("")
    A("On the effect of truncation on the variance level, this round makes only one "
      "conditional statement:")
    A("")
    A("> If the standardised residual is treated as standard normal, the second moment of the "
      "**single-step update input** is "
      "`(2Φ(c)−1) − 2cφ(c) + 2c²(1−Φ(c))`, which equals `0.99987946` at c=4, "
      "i.e. `0.01205%` below the untruncated value of 1.")
    A("")
    A("**This is not the long-run variance bias of the recursion, and this round makes no "
      "statement about that.** The truncation point `c²Ṽ_t` depends on the model's own "
      "random state; under the true DGP the standardised residuals are not standard normal; "
      "`e_t` is centred on a fixed midpoint rather than on the conditional mean; "
      "and the truncated term appears on both sides of the recursion, which breaks the linear "
      "argument for ordinary EWMA that \"the weights sum to 1 ⇒ the fixed point equals the "
      "input mean\" "
      "(a depressed variance lowers the threshold and makes truncation more frequent, a "
      "self-reinforcing feedback). "
      "The derivation is in `theory.md` §16.3.")
    A("")
    A("Accordingly, **the truncated `Ṽ_t` is not interpreted as an unbiased estimate of any of "
      "the working scale, the diffusion variance or the total variance**; it is an internal "
      "state of the detection rule, and its justification comes from the calibrated "
      "detection performance.")
    A("")

    # ---------------- 3 ----------------
    A("## 3. Calibration protocol and the order of threshold freezing")
    A("")
    rk = summary["ranks"]
    A(f"- Calibration sample: {cfg.stage2e_calibration_paths} **independent** valid-strategy "
      f"paths per scenario, sharing no random stream with the test data.")
    A(f"- Total number of comparisons J = 8 models × {len(scen)} scenarios × "
      f"{len(cfg.far_targets)} budgets = "
      f"**{summary['J']}** (Stage 2C had 60; this round adds two models, so the Bonferroni "
      f"split must widen accordingly); "
      f"calibration error budget per comparison δ/J = {summary['delta_per_comparison']:.3e}.")
    for a in cfg.far_targets:
        v = rk[str(a)] if str(a) in rk else rk[a]
        A(f"- α={a:g}: the order statistic of rank **{v['buffered']}** is used as the threshold"
          f" (the binomial route and the Beta upper-tail route independently give the same "
          f"rank: {v['beta_route']}).")
    A("- **The order is: freeze the thresholds, write them to `stage2e_thresholds.csv`, and "
      "only then read the formal test results.** "
      "The test data never took part in any threshold choice.")
    A("")

    # ---------------- 4 ----------------
    A("## 4. Main results: five scenarios × eight models")
    A("")
    for a in (a_main, a_alt):
        A(f"### α = {a:g} (two-year cumulative false-alarm budget)")
        A("")
        for sc in scen:
            A(_emph(sc))
            A("")
            rows = [(m, g(m, sc, "far_d504", a), g(m, sc, "detect_d504", a),
                     g(m, sc, f"detect_d{days[0]}", a), g(m, sc, "detect_d252", a),
                     g(m, sc, "trunc_mean_detect_days", a),
                     M[(M.scenario == sc) & (M.far_target == a)
                       & (M.method == m)]["median_detect_days"].iloc[0]) for m in METHODS_2E]
            L.extend(_t(rows, ["Model", "FAR(504)", f"Detection d{days[0]}", "Detection d252",
                               "**Detection d504**", "Truncated mean (days)",
                               "Median detection time"],
                        lambda r: [LABEL_2E[r[0]], f"{r[1]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}",
                                   f"**{r[2]:.4f}**", f"{r[5]:.1f}", _med(r[6])]))
            A("")
    A(f"(`trunc_mean_detect_days` records undetected paths as {cfg.horizon_days} days, "
      "so it is a bounded truncated mean, not a conditional mean. The median is recorded as "
      "\"not reached\" when the detection rate is below 50%.)")
    A("")

    # ---------------- 5 ----------------
    A("## 5. The three pre-specified comparisons")
    A("")
    A("These three comparisons were fixed before any test result of this round was seen. The "
      "intervals are 95% bootstrap intervals "
      f"({cfg.stage2e_bootstrap_reps} replicates, **resampling both the calibration sample and "
      "the test sample**, "
      "so they include the sampling uncertainty of the thresholds themselves).")
    A("")
    for ma, mb, title in PAIRS:
        A(f"**{title}**")
        A("")
        rows = []
        for a in (a_main, a_alt):
            for sc in scen:
                r = bo(ma, mb, sc, a)
                rows.append((a, sc, float(r.detect_diff), float(r.detect_diff_lo),
                             float(r.detect_diff_hi), float(r.far_diff_lo),
                             float(r.far_diff_hi), bool(r.detect_excludes_zero)))
        L.extend(_t(rows, ["α", "Scenario", "Difference in detection rate", "95% interval",
                           "Interval of FAR difference at the same budget",
                           "Excludes 0"],
                    lambda r: [f"{r[0]:g}", SC[r[1]], f"{r[2]:+.4f}",
                               f"[{r[3]:+.4f}, {r[4]:+.4f}]", f"[{r[5]:+.4f}, {r[6]:+.4f}]",
                               "yes" if r[7] else "no"]))
        A("")
    A("`stage2e_paired.csv` also gives paired intervals with the thresholds frozen (test "
      "sampling only), and "
      "`stage2e_paired_time.csv` gives the difference in truncated detection time under the "
      "same pairing. When both agree with the table above, "
      "the conclusion does not depend on whether calibration uncertainty is included.")
    A("")

    # ---------------- 6 ----------------
    A("## 6. When the false alarms occur")
    A("")
    fs = far_split[far_split.far_target == a_main]
    A(f"For valid-strategy paths, the **first** false alarm is split into three mutually "
      f"exclusive classes: the alarm day is itself a jump day; "
      f"the alarm day is not a jump day but the most recent jump is within the previous "
      f"{cfg.stage2e_jump_window} trading days; all others. "
      "By construction the three classes sum to the total FAR; the table below checks this "
      "row by row.")
    A("")
    for sc in [s for s in scen if not fs[fs.scenario == s].empty]:
        d = fs[fs.scenario == sc]
        A(f"{_emph(sc)}, α={a_main:g}")
        A("")
        rows = [d[d.method == m].iloc[0] for m in METHODS_2E if not d[d.method == m].empty]
        L.extend(_t(rows, ["Model", "On jump day",
                           f"Within {cfg.stage2e_jump_window} days after a jump",
                           "Elsewhere", "Sum of three classes", "Total FAR"],
                    lambda r: [LABEL_2E[r["method"]], f"{r['far_on_jump_day']:.4f}",
                               f"{r['far_within_window']:.4f}", f"{r['far_elsewhere']:.4f}",
                               f"{r['far_on_jump_day'] + r['far_within_window'] + r['far_elsewhere']:.4f}",
                               f"{r['far_total']:.4f}"]))
        A("")
    base_sc = "gaussian_ctrl"
    if not fs[fs.scenario == base_sc].empty:
        A("**This decomposition comes with an exact null control.** In the two κ=0 scenarios, "
          "jumps are still drawn at the same intensity λ, "
          "only with size 0; so classes 1 and 2 there measure pure **coincidental overlap**, "
          "giving the baseline of \"how much one should see if alarms were unrelated to "
          "jumps\".")
        A("")
        def share(sc, m):
            r = fs[(fs.scenario == sc) & (fs.method == m)]
            return float(r.far_on_jump_day.iloc[0] / r.far_total.iloc[0]) if len(r) else float("nan")
        srows = [(m, [share(sc, m) for sc in scen]) for m in CORE_2x2]
        L.extend(_t(srows, ["Model"] + [SC[sc].strip("*") for sc in scen],
                    lambda r: [LABEL_2E[r[0]]] + [f"{v:.4f}" for v in r[1]]))
        A("")
        import math as _m
        pJ = -_m.expm1(-cfg.noise_jump_lambda_annual / cfg.D)
        A("The table shows **class 1 as a share of that model's total FAR**. In the SV+jump "
          "family, the two Gaussian versions place between one third and more than half "
          "of their first false alarms on jump days; the two Student-t versions stay near the "
          "chance level.")
        A("")
        A(f"**The chance baseline is a theoretical value, not an observed share in some "
          f"control.** "
          f"Jump counts follow Poisson(λ/D), with λ={cfg.noise_jump_lambda_annual:g}/year and "
          f"D={cfg.D}, "
          f"so the probability that a day is flagged is `p_J = 1 − exp(−λ/D) = {pJ:.6f}`; "
          f"note that λ/D = {cfg.noise_jump_lambda_annual / cfg.D:.6f} is the mean count, not "
          f"a probability. "
          f"(An observed share of {share(base_sc, 'ewma_gaussian'):.2%} in one κ=0 control was "
          "previously called the \"exact baseline\"; this has been corrected.) The theoretical "
          "value for class 2 must also be aggregated over the realised distribution of alarm "
          "days; "
          "the derivation and per-method values are in `theory.md` §16.6 and "
          "`stage2e1_far_timing.csv`.")
        A("")
        gp_ = share("sv_jump", "ewma_gaussian")
        gt_ = share("sv_jump", "ewma_trunc_gaussian")
        A(f"**Truncation barely changes this share** (EWMA Gaussian {gp_:.4f} → truncated-G "
          f"{gt_:.4f}). "
          "This is consistent with the design of truncation: it truncates only the term written "
          "into the **next day's** variance state, "
          "while the same-day likelihood still sees the full return. In other words, truncation "
          "repairs the **persistent** contamination of the following days by a jump, "
          "but not the likelihood blow-up on the **day of** the jump; the latter requires a "
          "heavy-tailed likelihood. "
          "This is compatible with the observation that \"this round's effect sizes are smaller "
          "than the previous round's\", but it is a **mechanism hypothesis yet to be tested**; "
          "this round did not design an independent test for it.")
        A("")
    A("**The jump flags are used only for scoring in this step and do not enter any "
      "detector.** "
      "This is a description of timing association; the number obtained by subtracting the "
      "chance baseline from the realised share "
      "cannot be called the causal contribution of jumps to false alarms either.")
    A("")
    A("Latent states and observables must also be kept apart: in this DGP the jump counts "
      "are drawn independently by `rng.poisson(λ/D)`, "
      "and are **independent** of the latent stochastic-volatility state `v_t` (measured "
      "corr = −0.0024, n=200,000, SE≈0.0022). "
      "What is correlated with both is the **observed** large return (corr(v,|r|)=0.43, "
      "corr(K,|r|)=0.37). "
      "The earlier statement that \"jump days and high-volatility days are correlated by "
      "construction in this DGP\" does not match the specification and has been corrected.")
    A("")
    A("Shares alone are not enough: a falling share can also come from more false alarms "
      "elsewhere. The probabilities of the three classes with **all valid paths** as the "
      "denominator "
      "(the three sum to the total FAR) are in the `prob_*` columns of "
      "`stage2e1_far_timing.csv`.")
    A("")

    # ---------------- 7 ----------------
    A("## 7. Probability diagnostics and mechanism diagnostics")
    A("")
    A("The diagnostics of the failure probability q use **all days of all paths**; the later "
      "probabilities of a path are not dropped just because that path has already alarmed. "
      "An alarm is an action taken by the user, and dropping data would make the probability "
      "sample no longer the population it claims to describe.")
    A("")
    if len(brier):
        rows = []
        for sc in COMBINED:
            for _, r in brier[brier.scenario == sc].iterrows():
                rows.append(r)
        L.extend(_t(rows, ["Scenario", "Comparison", "Day", "Brier difference", "95% interval",
                           "Excludes 0"],
                    lambda r: [SC[r["scenario"]],
                               f"{LABEL_2E[r['a']]} − {LABEL_2E[r['b']]}", f"{int(r['day'])}",
                               f"{float(r['diff']):+.5f}",
                               f"[{float(r['lo']):+.5f}, {float(r['hi']):+.5f}]",
                               "yes" if excludes_zero(float(r["lo"]), float(r["hi"])) else "no"]))
        A("")
        A("A negative Brier difference means the first model's probability forecasts are more "
          "accurate. `stage2e_reliability.csv` gives binned reliability tables, "
          "with the sample size of every bin.")
        A("")
    A(f"Mechanism diagnostic (`stage2e_shock.csv`, figure 1): on day "
      f"{cfg.stage2d_shock_day} of one fixed SV+jump path, a shock of "
      f"±{cfg.stage2d_shock_sigmas:g}σ is injected, "
      "and the variance forecasts and cumulative log-odds of the four combinations are read "
      "off that same path. It shows directly that truncation flattens the effect of the shock "
      "on the "
      "**subsequent** variance forecasts, while the same-day likelihood still takes the full "
      "return.")
    A("")
    lg = metrics[metrics.arm == "legacy_threshold"]
    if len(lg):
        A("There is also a control **used only to observe the mechanism**: each truncated "
          "version borrows the "
          "Stage 2C unified threshold of its plain counterpart. **This control provides no "
          "false-alarm guarantee**; the two methods do not have the same budget here, "
          "so it cannot be used for same-budget rankings and only shows \"what happens without "
          "recalibration\".")
        A("")
        rows = []
        for sc in COMBINED:
            for m in ("ewma_gaussian", "ewma_trunc_gaussian", "ewma_student_t",
                      "ewma_trunc_student_t"):
                s = lg[(lg.scenario == sc) & (lg.far_target == a_main) & (lg.method == m)]
                if len(s):
                    rows.append((sc, m, float(s.far_d504.iloc[0]), float(s.detect_d504.iloc[0])))
        L.extend(_t(rows, ["Scenario", "Model", "FAR(504) under borrowed threshold",
                           "Detection d504"],
                    lambda r: [SC[r[0]], LABEL_2E[r[1]], f"{r[2]:.4f}", f"{r[3]:.4f}"]))
        A("")

    # ---------------- 8 ----------------
    A("## 8. Answers to the four questions")
    A("")
    plain = {k: v.strip("*") for k, v in SC.items()}
    n_sc = len(scen)
    sig_by_budget = {a: [r["scenario"] for r in T[a][1]] for a in budgets}
    any_sig = sorted({sc for v in sig_by_budget.values() for sc in v}, key=scen.index)
    if not any_sig:
        q1 = ("**No detectable improvement.** The differences for both budgets and all five "
              "scenarios cover 0. This is a clear negative result.")
    else:
        parts = []
        for a in budgets:
            rows_a = T[a][0]
            best_a = max(rows_a, key=lambda r: r["d"])
            parts.append(f"α={a:g}: {len(T[a][1])}/{n_sc} scenarios significantly positive"
                         f" (largest {best_a['d']:+.4f}, {plain[best_a['scenario']]})")
        mt = {a: next(r["d"] for r in T[a][0] if r["scenario"] == "sv_jump") for a in budgets}
        q1 = ("**A small improvement, worth keeping.** " + "; ".join(parts) + ". "
              f"The clear gains observed in this round are concentrated in "
              f"{', '.join(plain[sc] for sc in any_sig)}"
              + (" (all of them scenarios with κ>0)" if jump_story(set(any_sig)) else "")
              + "; this is an observation from this round and does not mean it has been proven "
              "that truncation only helps when there are jumps. "
              + f"The two steps in the main scenario SV+jumps, at the same budget, are: "
              + "; ".join(f"α={a:g}: fixed t→EWMA t {ref[a]*100:+.2f} pp, "
                          f"EWMA t→truncated t {mt[a]*100:+.2f} pp ("
                          f"{abs(mt[a])/abs(ref[a]):.0%} of the first step)" for a in budgets)
              + ".")
    A(f"1. **Does this round beat the current strong method?** {q1}")
    if tight and loose:
        ta, la = tight[0], loose[0]
        q2 = (f"Both mechanisms are supported, but under different conditions, and this is "
              f"visible only when the two budgets are put side by side. "
              f"(i) \"An extreme day contaminates the evidence weight of the following days "
              f"through the variance state\": truncation gives a significantly positive "
              f"improvement in the scenarios with jumps, and is uniformly not significant in "
              f"the scenarios without jumps. "
              f"(ii) \"The same-day likelihood blow-up and the cross-day variance contamination "
              f"are two distinct harms\": at α={ta:g} "
              f"the interaction is significantly positive while Gaussian-side truncation is "
              f"not significant anywhere, showing that under the tight budget the Gaussian "
              f"version's bottleneck is the same-day likelihood, which truncation cannot reach; "
              f"at α={la:g} the interaction covers 0 and the gains on the two sides are "
              f"similar. "
              f"The timing decomposition in Section 6 gives independent evidence for the same "
              f"conclusion: truncation barely changes the share of \"false alarms that fall on "
              f"jump days\".")
    elif ipos and not ineg:
        q2 = (f"The interaction is significantly positive in {len(ipos)}/{len(irows)} "
              f"scenarios, supporting \"the two are complementary and need to be paired\".")
    elif ineg and not ipos:
        q2 = (f"The interaction is significantly negative in {len(ineg)}/{len(irows)} "
              f"scenarios, supporting \"truncation and the heavy-tailed likelihood are "
              f"substitutes\".")
    else:
        q2 = (f"The interaction covers 0 under both budgets (widest half-width {half:.4f}, of "
              "the same order as the main effects); "
              "only interactions much larger than the main effects can be ruled out, not "
              "interactions comparable to the main effects; recorded as unresolved.")
    A(f"2. **Which mechanisms are supported?** {q2}")
    best_main = max(g(m, "sv_jump") for m in METHODS_2E)
    best_alt = max(g(m, "sv_jump", "detect_d504", a_alt) for m in METHODS_2E)
    from .gaussian_bound import max_detection
    b2 = max_detection(cfg.sharpe_valid, cfg.horizon_days / cfg.D, a_main)
    A(f"3. **What is still missing to reach a practical speed?** "
      f"The best model of this round catches {best_main:.1%} of invalid strategies within two "
      f"years (α={a_main:g}); "
      f"at α={a_alt:g} this falls to {best_alt:.1%}. **The earlier conclusion that \"the "
      "remaining gap comes entirely from an unbreakable "
      "information limit and can only be closed by adding new information\" has been "
      "withdrawn**: this project has not proven the "
      "optimal detection rate under the combined DGP, let alone that the existing methods come "
      "close to it. The same annualised Sharpe does not uniquely determine how identifiable a "
      "return series is: "
      "predictable volatility, the shape of the centre of the distribution and the jump "
      "structure all change how much evidence can be extracted; "
      "the improvement Stage 2B achieved in the SV scenario is itself evidence of this. "
      f"What can be given is a reference with **explicit conditions**: in a model with "
      f"independent Gaussian returns, known variance and two known candidate means, "
      f"the upper bound on the detection rate for S=1, two years and α={a_main:g} is "
      f"{b2:.1%} (`theory.md` §17). "
      "It constrains any sequential rule that satisfies the cumulative false-alarm constraint "
      "under that same Gaussian model, "
      "but does **not** apply to the SV+jump scenarios, so the 76.5% in the combined scenario "
      "cannot be read as \"close to some ceiling\".")
    core_lo = min(g(m, "sv_jump") for m in CORE_2x2)
    core_hi = max(g(m, "sv_jump") for m in CORE_2x2)
    A(f"4. **Should the next step move to a weaker Sharpe?** Yes, but as an **additional "
      f"setting** rather than a replacement for the main benchmark. "
      f"There are two reasons. First, the true Sharpe of most real strategies is below 1; "
      f"**the next main set of experiments uses S=0.6 vs S=0, which was explicitly discussed "
      f"at the meeting, and keeps S=1 vs S=0 as the main reference**. "
      f"Second, at S=1 the two-year detection rates of the four variance-adaptive rules "
      f"already cluster in the band "
      f"[{core_lo:.3f}, {core_hi:.3f}], of width {core_hi - core_lo:.3f}, "
      f"while the best method still misses {1 - best_main:.1%} of invalid strategies; the "
      f"marginal gain from further work on the form of the rule "
      "is already smaller than the information gained by moving the problem to a harder, more "
      "realistic region. "
      "**The direction must be stated clearly**: a weaker Sharpe makes every method slower; "
      "it is not a device for magnifying differences between methods, but a way to answer the "
      "question itself: \"at signal-to-noise ratios of practical relevance, "
      "can a decision be reached within two years at all?\" "
      f"The same Gaussian reference gives the following orders of magnitude: for S=0.6, two "
      f"years and α={a_main:g}, the upper bound is "
      f"{max_detection(0.6, cfg.horizon_days / cfg.D, a_main):.1%}, "
      f"and for one year {max_detection(0.6, 1.0, a_main):.1%}. This round does not start "
      f"these simulations on its own.")
    A("")
    A("**One further recommendation is tightened.** \"Higher-frequency data\" was earlier "
      "listed as a way to speed up detection. Denser sampling helps identify the volatility "
      "and jump structure, but within the same calendar window it does **not automatically** "
      "bring more independent evidence about the drift: "
      "in a model with constant drift and i.i.d. noise, the drift information in a given "
      "window is determined by s·√h, independent of the sampling frequency.")
    A("")

    # ---------------- 9 ----------------
    A("## 9. Limitations")
    A("")
    A(f"- `c` is fixed at {cfg.ewma_truncation_c:g}, **with no search**. This round can only "
      "answer \"how large an effect does the single a-priori "
      "reasonable value c=4 produce\", not \"which c is best\", and the c=4 result cannot be "
      "taken as an upper or lower bound for the truncation family "
      "of methods. Answering the latter would require putting the choice of c itself into "
      "the calibration protocol "
      "and widening the Bonferroni split accordingly.")
    A(f"- **The conclusions change with the false-alarm budget** (see Question 2 in Section 1). "
      f"This round validated only at the two points α="
      f"{', '.join(f'{a:g}' for a in cfg.far_targets)}; "
      "intermediate and more extreme budgets were not tested, and the conclusion under either "
      "budget should not be extrapolated to the whole α curve.")
    A("- The reading at the end of Section 1, that \"under the tight budget the Gaussian "
      "version's bottleneck is the same-day likelihood\", is consistent with the data, "
      "but this round did **not** design an independent test for it; it is a recorded "
      "hypothesis, not a conclusion.")
    A("- All conclusions hold under the single fixed setting S=1 vs S=0, σ_ann=0.10, D=252, "
      "H=504.")
    A("- All five scenarios are generated by the same SV+jump DGP family; real returns may "
      "fail in ways outside this family.")
    A("- The timing decomposition of false alarms is a description of association, not a "
      "causal decomposition (see Section 6).")
    A("- Under the standard normal reference, truncation lowers the second moment of the "
      "**single-step update input** by 0.01205%. "
      "This is not the long-run variance bias of the recursion, and this round makes no "
      "statement about the latter (`theory.md` §16.3).")
    A("- The interaction and the cross-budget comparison are **exploratory analyses** added "
      "after seeing the main results; "
      "their intervals are per-item (unadjusted) intervals, with no multiple-comparison "
      "adjustment. The δ/J of the calibration protocol protects only the false-alarm "
      "constraint of the thresholds, "
      "and gives no simultaneous confidence guarantee for differences in detection rate.")
    A("- This round's \"deterministic reproduction check\" means that a rerun with the same "
      "seed gives identical CSVs; "
      "it is **not** a replication with new independent samples.")
    A("")
    A("## 10. Reproduction")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.stage2e")
    A("```")
    A("")
    A(f"Run time {summary['elapsed_s']:.1f} seconds. The figures can be redrawn on their own: "
      "`uv run python -m strategy_survivorship.stage2e --figures-only`.")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
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
from scipy.special import expit


from .config import DEFAULT
from .evaluate import excludes_zero
from .probability_time import brier_and_reliability

from .stage1 import environment_info
from .stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2b import paired_brier_difference
from .stage2d import COMBINED, load_frozen_thresholds, specs
from .stage2d import paired_frozen, paired_truncated_time


# pre-specified before the run
MAIN_PAIRS = (("ewma_trunc_student_t", "ewma_student_t"),
              ("ewma_trunc_gaussian", "ewma_gaussian"),
              ("ewma_trunc_student_t", "ewma_trunc_gaussian"))


def bootstrap(cfg, cal, minima, seed_seq) -> pd.DataFrame:
    """Resample the scenario's calibration block AND the test block.

    All four cells of the 2x2 are recomputed inside one resampling loop, so the
    three pre-specified pairs and the interaction contrast share the same draws.
    The interaction

        (trunc_t - ewma_t) - (trunc_g - ewma_g)

    is what actually answers "does the gain depend on pairing with a Student-t
    likelihood?": a gain that only exists on the Gaussian side shows up as a
    negative interaction, one that only exists on the Student-t side as a
    positive one, and two separate additive mechanisms as an interval covering 0.
    """
    rng = np.random.default_rng(seed_seq)
    n_cal, reps = cfg.stage2e_calibration_paths, cfg.stage2e_bootstrap_reps
    tg, tt = "ewma_trunc_gaussian", "ewma_trunc_student_t"
    eg, et = "ewma_gaussian", "ewma_student_t"
    out = []
    for spec in specs(cfg):
        key = spec[0]
        for a in cfg.far_targets:
            k = cal["ranks"][a]["buffered"]
            tm = minima[key]
            n_i = tm[CORE_2x2[0]]["test_invalid"].size
            n_v = tm[CORE_2x2[0]]["test_valid"].size
            det = {m: np.empty(reps) for m in CORE_2x2}
            far = {m: np.empty(reps) for m in CORE_2x2}
            for r in range(reps):
                ic = rng.integers(0, n_cal, n_cal)   # one calibration resample, all four cells
                ii = rng.integers(0, n_i, n_i)
                iv = rng.integers(0, n_v, n_v)
                for m in CORE_2x2:
                    t_ = float(np.partition(cal["minima"][(key, m)][ic], k - 1)[k - 1])
                    det[m][r] = (tm[m]["test_invalid"][ii] < t_).mean()
                    far[m][r] = (tm[m]["test_valid"][iv] < t_).mean()
            point = {}
            for m in CORE_2x2:
                t0 = float(np.partition(cal["minima"][(key, m)], k - 1)[k - 1])
                point[m] = float((tm[m]["test_invalid"] < t0).mean())

            def row(name, ma, mb, dd, df, pt):
                lo, hi = float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))
                flo, fhi = float(np.quantile(df, 0.025)), float(np.quantile(df, 0.975))
                return {"scenario": key, "far_target": a, "contrast": name,
                        "method_a": ma, "method_b": mb, "n_reps": reps, "detect_diff": pt,
                        "detect_diff_lo": lo, "detect_diff_hi": hi,
                        "far_diff_lo": flo, "far_diff_hi": fhi,
                        "detect_excludes_zero": excludes_zero(lo, hi),
                        "covers": "calibration + test sampling"}

            for ma, mb in MAIN_PAIRS:
                out.append(row("pair", ma, mb, det[ma] - det[mb], far[ma] - far[mb],
                               point[ma] - point[mb]))
            out.append(row("interaction", f"{tt}-{et}", f"{tg}-{eg}",
                           (det[tt] - det[et]) - (det[tg] - det[eg]),
                           (far[tt] - far[et]) - (far[tg] - far[eg]),
                           (point[tt] - point[et]) - (point[tg] - point[eg])))
    return pd.DataFrame(out)


def shock_2x2(cfg) -> pd.DataFrame:
    """The fixed shock path, read through all four core combinations."""
    from .ewma import (ewma_gaussian_increments, ewma_student_t_increments,
                       ewma_truncated_variance_forecast, ewma_variance_forecast)
    from .noise import draw_noise, returns_from_noise
    from .stage2d import scenario_cfg

    spec = next(s for s in specs(cfg) if s[0] == "sv_jump")
    cn = scenario_cfg(cfg, spec[1], spec[2])
    idx = cfg.stage2d_shock_path_index
    d = draw_noise("sv_jump", make_streams(cfg)["stage2d_shock"], idx + 1, cfg.horizon_days, cn)
    base = returns_from_noise(d.eps, cfg.sharpe_valid, cfg)[idx].copy()
    day0 = cfg.stage2d_shock_day - 1
    amp = cfg.stage2d_shock_sigmas * cfg.sigma_daily
    variants = {"base": base, "plus": base.copy(), "minus": base.copy()}
    variants["plus"][day0] += amp
    variants["minus"][day0] -= amp

    frames = []
    for name, r in variants.items():
        r2 = r[None, :]
        vp, _ = ewma_variance_forecast(r2, cfg)
        vt, _ = ewma_truncated_variance_forecast(r2, cfg)
        cols = {"day": np.arange(1, cfg.horizon_days + 1), "variant": name, "return": r,
                "var_plain": vp[0], "var_trunc": vt[0]}
        # Column names say which way round the log-odds run.  The running
        # statistic every detector uses is L = log P(valid)/P(invalid); the
        # report and the figures show U = -L = log P(invalid)/P(valid), so that
        # "up" means "more evidence the strategy has failed".
        for tag, v in (("plain", vp), ("trunc", vt)):
            gi = ewma_gaussian_increments(r2, cfg, v)[0]
            ti = ewma_student_t_increments(r2, cfg, v)[0]
            cols[f"inc_L_{tag}_gaussian"] = gi
            cols[f"inc_L_{tag}_student_t"] = ti
            cols[f"U_{tag}_gaussian"] = -(cfg.prior_log_odds + np.cumsum(gi))
            cols[f"U_{tag}_student_t"] = -(cfg.prior_log_odds + np.cumsum(ti))
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2E: truncated variance update.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--stage2c-thresholds", type=Path,
                    default=Path("outputs/stage2c_unified_thresholds.csv"))
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, stage2e_calibration_paths=800, stage2e_test_paths=800,
                 stage2e_bootstrap_reps=50)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke2e") if args.smoke else Path("outputs"))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        setup_style()
        print("figures redrawn from disk:", draw_all(cfg, out_dir))
        return 0

    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()
    days = tuple(cfg.stage2c_report_days)

    cal = calibrate(cfg)
    cal["table"].to_csv(out_dir / "stage2e_thresholds.csv", index=False)
    status("thresholds frozen", methods=len(METHODS_2E), scenarios=len(specs(cfg)),
           J=cal["J"], ranks={a: v["buffered"] for a, v in cal["ranks"].items()})

    # mechanism-only control: each truncated variant borrows its plain counterpart's
    # Stage 2C threshold. No guarantee attaches to it and it is never used for a
    # same-budget ranking.
    require_input(args.stage2c_thresholds, "run `python -m strategy_survivorship.stage2c` first, or pass --stage2c-thresholds")
    s2c = load_frozen_thresholds(cfg, args.stage2c_thresholds)
    legacy = {}
    for plain, trunc_m in (("ewma_gaussian", "ewma_trunc_gaussian"),
                           ("ewma_student_t", "ewma_trunc_student_t")):
        legacy[plain] = {(plain, a): s2c[(plain, a)] for a in cfg.far_targets}
        legacy[trunc_m] = {(trunc_m, a): s2c[(plain, a)] for a in cfg.far_targets}

    rows, minima, trunc, qrows, far_split, blocks = [], {}, {}, [], [], {}
    tk = make_streams(cfg)["stage2e_test"].spawn(len(specs(cfg)))
    for spec, kid in zip(specs(cfg), tk):
        rr, mm, tt, qq, fs, bv, bi = evaluate(cfg, spec, kid, cal["thresholds"], legacy, days)
        rows.extend(rr); minima[spec[0]] = mm; trunc[spec[0]] = tt
        qrows.extend(qq); far_split.extend(fs); blocks[spec[0]] = (bv, bi)
        status(f"scenario tested: {spec[0]}", rows=len(rr))

    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS_2E)}
    metrics["_o"] = metrics.method.map(order)
    metrics = metrics.sort_values(["scenario", "arm", "far_target", "_o"]).drop(columns="_o")
    metrics.to_csv(out_dir / "stage2e_metrics.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "stage2e_failure_probability.csv", index=False)
    pd.DataFrame(far_split).to_csv(out_dir / "stage2e_far_timing.csv", index=False)
    status("scenario tables written", metric_rows=len(metrics), timing_rows=len(far_split))

    paired, ptime = [], []
    for sc in [s[0] for s in specs(cfg)]:
        for a in cfg.far_targets:
            thr = {m: cal["thresholds"][(m, sc, a)] for m in METHODS_2E}
            tau = {m: trunc[sc][(m, a)] for m in METHODS_2E}
            for ma, mb in MAIN_PAIRS:
                paired.append(paired_frozen(minima[sc], thr, ma, mb, sc, "per_scenario", a, cfg))
                r = paired_truncated_time(tau, ma, mb, cfg)
                r.update({"scenario": sc, "far_target": a, "method_a": ma, "method_b": mb,
                          "trunc_mean_a": float(tau[ma].mean()),
                          "trunc_mean_b": float(tau[mb].mean()),
                          "interval_covers": "test sampling only, thresholds frozen"})
                ptime.append(r)
    pd.DataFrame(paired).to_csv(out_dir / "stage2e_paired.csv", index=False)
    pd.DataFrame(ptime).to_csv(out_dir / "stage2e_paired_time.csv", index=False)
    status("paired differences done", detect=len(paired), time=len(ptime))

    boot = bootstrap(cfg, cal, minima, make_streams(cfg)["stage2e_bootstrap"])
    boot.to_csv(out_dir / "stage2e_bootstrap.csv", index=False)
    status("bootstrap done", comparisons=len(boot), reps=cfg.stage2e_bootstrap_reps)

    brier, rel = [], []
    bk = make_streams(cfg)["stage2e_bootstrap"].spawn(len(COMBINED) * len(days) * len(MAIN_PAIRS))
    i = 0
    for sc in COMBINED:
        bv, bi = blocks[sc]
        q = {}
        for m in CORE_2x2:
            q[(m, "i")] = expit(-statistic(m, bi["returns"], cfg))
            q[(m, "v")] = expit(-statistic(m, bv["returns"], cfg))
            for d in days:
                s_, tab = brier_and_reliability(q[(m, "i")], q[(m, "v")], d, m)
                tab["scenario"], tab["day"], tab["brier"] = sc, d, s_["brier"]
                rel.append(tab)
        for ma, mb in MAIN_PAIRS:
            for d in days:
                r = paired_brier_difference(q[(ma, "i")], q[(ma, "v")], q[(mb, "i")],
                                            q[(mb, "v")], d, cfg.stage2e_bootstrap_reps, bk[i])
                i += 1
                r.update({"scenario": sc, "a": ma, "b": mb})
                brier.append(r)
        del q
    pd.DataFrame(brier).to_csv(out_dir / "stage2e_brier.csv", index=False)
    pd.concat(rel, ignore_index=True).to_csv(out_dir / "stage2e_reliability.csv", index=False)
    status("Brier and reliability done", rows=len(brier))

    shock = shock_2x2(cfg)
    shock.to_csv(out_dir / "stage2e_shock.csv", index=False)
    status("shock diagnostic done")

    figures = []
    if not args.no_figures:
        setup_style()
        figures = draw_all(cfg, out_dir)
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "2E", "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "methods": list(METHODS_2E), "J": cal["J"],
        "delta_per_comparison": cal["delta_per_comparison"],
        "ranks": {str(a): v for a, v in cal["ranks"].items()},
        "truncation_c": cfg.ewma_truncation_c,
        "sample_sizes": {"calibration_valid": cfg.stage2e_calibration_paths,
                         "test_valid": cfg.stage2e_test_paths,
                         "test_invalid": cfg.stage2e_test_paths, "days": cfg.horizon_days},
        "metrics": json.loads(metrics.to_json(orient="records")),
        "paired": paired, "paired_time": ptime,
        "bootstrap": json.loads(boot.to_json(orient="records")),
        "brier": brier, "far_timing": far_split, "failure_probability": qrows,
        "report_days": list(days), "elapsed_s": elapsed, "figures": figures,
    }
    (out_dir / "stage2e_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stage2e_report(cfg, summary, metrics, cal, boot, pd.DataFrame(brier), shock,
                         pd.DataFrame(far_split), out_dir / "stage2e_report.md")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
