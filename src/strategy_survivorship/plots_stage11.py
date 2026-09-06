"""Stage 1.1 figures.  English labels, one colour per detector, four figures.

Shaded bands are always *path* quantiles (10th-90th percentile across simulated
strategies) and are labelled as such -- they are the spread of outcomes, not a
confidence interval for an estimate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .plots import COLOURS, setup_style  # one shared palette across all stages

T_COLOURS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
YEAR_TICKS = [(126, "0.5y"), (252, "1y"), (504, "2y"), (1260, "5y"), (2520, "10y")]


def _year_axis(ax, n_days: int, log: bool = False) -> None:
    """Label the x axis in years.

    On a log axis the scale must be set FIRST: switching to log afterwards
    rebuilds the locator and silently discards these ticks.  Minor ticks are
    cleared for the same reason.
    """
    if log:
        ax.set_xscale("log")
    ticks = [(d, lab) for d, lab in YEAR_TICKS if d <= n_days]
    for d, _ in ticks:
        ax.axvline(d, color="0.85", lw=0.7, zorder=0)
    ax.set_xticks([d for d, _ in ticks], [lab for _, lab in ticks])
    ax.set_xticks([], minor=True)


# --------------------------------------------------------------------------- #


def figure_failure_probability(cfg: Stage1Config, pt: dict, out: Path) -> Path:
    q = pt["q_curves"]
    n_days = pt["n_days"]
    days = np.arange(1, n_days + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)

    for ax, state, title in (
        (axes[0], "invalid", "True state: INVALID (S = 0)"),
        (axes[1], "valid", "True state: VALID (S = 1)"),
    ):
        for key in ("binary_gaussian", "binary_student_t"):
            lo, med, hi = q[(key, state)]
            ax.plot(days, med, color=COLOURS[key], lw=1.6,
                    label=f"{key.replace('_', ' ')} — median")
            ax.fill_between(days, lo, hi, color=COLOURS[key], alpha=0.16, lw=0,
                            label=f"{key.replace('_', ' ')} — 10–90% of paths")
        ax.axhline(0.5, color="0.5", lw=0.9, ls="-.")
        for b in cfg.prob_thresholds:
            ax.axhline(b, color="0.35", lw=0.7, ls=":")
            ax.text(n_days * 0.985, b + 0.012, f"b={b:g}", fontsize=7.5, ha="right", color="0.35")
        _year_axis(ax, n_days, log=True)
        ax.set_xlim(1, n_days)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time since go-live (log scale)")
    axes[0].set_ylabel("failure probability  $q_n = P(\\theta=0 \\mid r_{1:n})$")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.93)

    fig.suptitle(
        "Fig 1.1  Failure probability over time, fixed true state\n"
        f"n = {cfg.prob_time_paths} paths per state; bands are the 10th–90th percentile ACROSS PATHS, "
        "not a confidence interval. Diagnostic horizon 10y carries no 504-day false-alarm budget.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_threshold_hits(cfg: Stage1Config, pt: dict, out: Path) -> Path:
    cur = pt["curves"]
    n_days = pt["n_days"]
    main_b = 0.90
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)

    for ax, state, title in (
        (axes[0], "invalid", "True state: INVALID (S = 0)"),
        (axes[1], "valid", "True state: VALID (S = 1) — these are false convictions"),
    ):
        for key in ("binary_gaussian", "binary_student_t"):
            for b in cfg.prob_thresholds:
                s = cur[(cur.detector == key) & (cur.true_state == state) & (cur.threshold_b == b)]
                is_main = abs(b - main_b) < 1e-9
                ax.plot(
                    s.day, s.cumulative_reached, color=COLOURS[key],
                    lw=1.8 if is_main else 0.9,
                    alpha=1.0 if is_main else 0.42,
                    ls="-" if is_main else "--",
                    label=f"{key.replace('_', ' ')}, b={b:g}" + (" (main)" if is_main else ""),
                )
        ax.axhline(0.5, color="0.5", lw=0.9, ls="-.")
        _year_axis(ax, n_days, log=True)
        ax.set_xlim(1, n_days)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("time since go-live (log scale)")
    axes[0].set_ylabel("fraction of paths that have reached $q_n \\geq b$")
    axes[0].legend(loc="upper left", fontsize=7.4, framealpha=0.93, ncol=2)

    fig.suptitle(
        "Fig 1.2  First time the failure probability reaches a fixed level b\n"
        f"n = {cfg.prob_time_paths} paths per state. Paths that never reach b stay uncounted "
        "(the curve is a cumulative fraction of ALL paths, not of successes).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_switching_detection(cfg: Stage1Config, sw: dict, out: Path, alpha: float | None = None) -> Path:
    """Pre-failure cost, and post-failure detection at a MATCHED false-alarm cost.

    The nominal-budget comparison is not cost-matched: the rolling detectors buy a
    flatter post-failure curve with many more pre-failure alarms.  The middle
    panel therefore uses the fixed continuation false-alarm level, which is the
    only setting comparable across T.
    """
    alpha = cfg.far_targets[-1] if alpha is None else alpha
    m = sw["metrics"]
    lvl = cfg.switch_fixed_continuation_fa
    keys = list(COLOURS)[:4]

    def _grp(prefix, far=None):
        d = m[m.group.str.startswith(prefix)].copy()
        if far is not None:
            d = d[d.far_target == far]
        d["T"] = d.group.str.extract(r"=(\d+)").astype(int)
        return d.sort_values("T")

    nominal = _grp("fixed_T=", alpha)
    matched = _grp(f"contFA{lvl:g}_T=")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))

    ax = axes[0]
    for key in keys:
        s_ = nominal[nominal.detector == key]
        ax.plot(s_["T"], s_.pre_failure_false_alarm_rate, "o-", color=COLOURS[key], lw=1.5, ms=5,
                label=key.replace("_", " "))
    ax.set_title(f"(a) Alarms raised BEFORE the failure\nnominal budget $\\alpha$={alpha:g}, "
                 "$P(\\tau \\leq T)$ — all false", fontsize=9.5)
    ax.set_ylabel("pre-failure false-alarm rate")
    ax.legend(fontsize=7.4, loc="upper left")

    ax = axes[1]
    for key in keys:
        s_ = matched[matched.detector == key]
        ax.plot(s_["T"], s_.cond_detect_h504, "o-", color=COLOURS[key], lw=1.8, ms=5,
                label=key.replace("_", " "))
    ax.set_title(f"(b) Detection after failure at MATCHED cost\ncontinuation FA fixed at {lvl:g} "
                 "for every $T$, $h$=504d", fontsize=9.5)
    ax.set_ylabel("conditional post-failure detection rate")
    ax.legend(fontsize=7.4, loc="upper right")

    ax = axes[2]
    for key in keys:
        s_ = matched[matched.detector == key]
        ax.plot(s_["T"], s_.pre_failure_false_alarm_rate, "o-", color=COLOURS[key], lw=1.5, ms=5,
                label=key.replace("_", " "))
    ax.set_title(f"(c) What that matched cost is bought with\npre-failure alarms at continuation "
                 f"FA = {lvl:g}", fontsize=9.5)
    ax.set_ylabel("pre-failure false-alarm rate")
    ax.legend(fontsize=7.4, loc="upper left")

    for ax in axes:
        ax.set_xlabel("$T$ = valid history before failure (trading days)")
        ax.set_xticks(list(cfg.switch_fixed_T))
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Fig 1.3  A longer valid history makes the same failure harder to catch — for BOTH families\n"
        f"n = {cfg.switch_fixed_paths} paths per $T$, common random numbers across $T$. Panel (b) is the "
        "cost-matched comparison; panel (c) shows the price each family pays for it.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_evidence_recovery(cfg: Stage1Config, sw: dict, out: Path,
                             detector: str = "binary_gaussian") -> Path:
    """Log-evidence around the switch, on ALL paths (no survivor conditioning).

    The right panel aggregates the PER-PATH increment U_t - U_T and only then
    takes the mean and the 10-90% band: a difference of two time-point medians is
    a different quantity and would misstate the recovery.
    """
    traces = [t for t in sw["traces"] if t[1] == detector]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    post = cfg.switch_post_window
    lead = 252

    ax = axes[0]
    for i, (group, _, qU, _qd, _m, _n) in enumerate(traces):
        T = int(group.split("=")[1])
        lo, med, hi = qU
        rel = np.arange(1, med.size + 1) - T
        m = (rel >= -lead) & (rel <= post)
        col = T_COLOURS[i % len(T_COLOURS)]
        ax.plot(rel[m], med[m], color=col, lw=1.7, label=f"T = {T}d")
        ax.fill_between(rel[m], lo[m], hi[m], color=col, alpha=0.13, lw=0)
    ax.axhline(-sw["threshold_for_trace"], color="k", ls="--", lw=1.1,
               label="alarm level in $U$ (= $-$Stage 1 threshold)")
    ax.axvline(0, color="0.4", lw=1.1)
    ax.set_xlabel("trading days relative to failure ($t - T$)")
    ax.set_ylabel("failure log-odds  $U_t = \\mathrm{logit}\\, q_t = -L_t$")
    ax.set_title("Level: evidence accumulated before the switch\nmedian and 10-90% of paths",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for i, (group, _, _qU, qd, mean_d, _n) in enumerate(traces):
        T = int(group.split("=")[1])
        lo, med, hi = qd
        rel = np.arange(1, med.size + 1) - T
        m = (rel >= 0) & (rel <= post)
        col = T_COLOURS[i % len(T_COLOURS)]
        ax.plot(rel[m], mean_d[m], color=col, lw=1.7, label=f"T = {T}d (mean)")
        ax.fill_between(rel[m], lo[m], hi[m], color=col, alpha=0.10, lw=0)
    h = np.arange(0, post + 1)
    ax.plot(h, h / (2.0 * cfg.D), color="k", ls=":", lw=1.4,
            label="theory: $h\\,s^2/(2D)$, $S_{true}=0$")
    ax.axvline(0, color="0.4", lw=1.1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("trading days after failure ($h$)")
    ax.set_ylabel("$U_{T+h} - U_T$  per path, then averaged")
    ax.set_title("Increment: per-path $U_{T+h}-U_T$\nmean and 10-90% of paths",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    for a in axes:
        a.grid(alpha=0.3)

    n = traces[0][5] if traces else 0
    fig.suptitle(
        f"Fig 1.4  {detector.replace('_', ' ')} log-evidence around the switch — ALL paths, "
        f"n = {n} per $T$\n"
        "Groups share calendar-time noise across $T$, which does NOT make their noise identical in "
        "time-since-failure; bands are 10-90% ACROSS PATHS, not confidence intervals.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out)
    plt.close(fig)
    return out
