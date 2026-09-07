"""Stage 1 figures.

Four PNGs, all with explicit units, legends, sample sizes and the ground-truth
state written on the axes.  Anything drawn from a single realisation is labelled
"example path" so it cannot be mistaken for an aggregate result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display required
import matplotlib.pyplot as plt
import numpy as np

from .config import Stage1Config
from .detectors import DETECTORS_BY_KEY, influence_curve, probability_from_log_odds
from .evaluate import random_closure_reference

DPI = 150

COLOURS = {
    "binary_gaussian": "#1f77b4",
    "binary_student_t": "#d62728",
    "trailing_sharpe_252": "#2ca02c",
    "known_vol_rolling_252": "#9467bd",
    "random_closure": "#7f7f7f",
}
STYLES = {
    "binary_gaussian": "-",
    "binary_student_t": "-",
    "trailing_sharpe_252": "-",
    "known_vol_rolling_252": "--",
    "random_closure": ":",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # PingFang SC first so the Chinese labels get real glyphs; matplotlib
            # falls through to the Latin faces for everything else.
            "font.sans-serif": ["PingFang SC", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _mark_eval_days(ax, cfg: Stage1Config, label_rolling: bool = True) -> None:
    for d in cfg.eval_horizons:
        ax.axvline(d, color="0.75", lw=0.7, zorder=0)
    if label_rolling:
        ax.axvline(
            cfg.rolling_window,
            color="0.45",
            lw=1.0,
            ls=(0, (4, 3)),
            zorder=1,
        )


# --------------------------------------------------------------------------- #
# Figure 1 -- example paths
# --------------------------------------------------------------------------- #


def figure_example_paths(
    cfg: Stage1Config,
    pathsets: dict,
    stats: dict,
    calibrations: dict,
    out_path: Path,
    path_index: int = 0,
) -> Path:
    days = np.arange(1, cfg.horizon_days + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)

    panels = [
        ("test_valid", "真实有效 true state: VALID (S = 1)", 0),
        ("test_invalid", "真实无效 true state: INVALID (S = 0)", 1),
    ]
    for set_name, title, col in panels:
        r = pathsets[set_name].returns[path_index]
        ax = axes[0, col]
        ax.plot(days, np.cumsum(r), color="#333333", lw=1.2)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(f"{title}\n示例路径 example path #{path_index}", fontsize=10.5)
        ax.set_ylabel("累计超额收益 cumulative excess return\n(sum of daily $r_t$)")
        _mark_eval_days(ax, cfg)

        ax = axes[1, col]
        loosest = max(cfg.far_targets)
        for key in ("binary_gaussian", "binary_student_t"):
            lo = stats[key][set_name][path_index]
            ax.plot(
                days,
                probability_from_log_odds(lo),
                color=COLOURS[key],
                lw=1.3,
                label=DETECTORS_BY_KEY[key].label,
            )
            # first strict down-crossing of the loosest threshold, if any
            thr = calibrations[(key, loosest)].threshold
            hit = np.nonzero(lo < thr)[0]
            if hit.size:
                t0 = int(hit[0]) + 1
                y0 = 1.0 / (1.0 + np.exp(-lo[t0 - 1]))
                ax.plot([t0], [y0], marker="v", ms=9, color=COLOURS[key],
                        mec="k", mew=0.6, ls="none", zorder=5)
                ax.annotate(
                    f"首次报警 first alarm ($\\alpha$={loosest:g}): day {t0}",
                    xy=(t0, y0),
                    xytext=(10, 30 if key == "binary_gaussian" else -30),
                    textcoords="offset points", fontsize=7.5,
                    color=COLOURS[key], zorder=6,
                    arrowprops=dict(arrowstyle="-", color=COLOURS[key], lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=COLOURS[key], lw=0.6, alpha=0.92),
                )
        # Each detector alarms against its OWN calibrated threshold, so each one is
        # drawn in its own colour -- plotting a single model's line beside both
        # models' alarm markers would make the other marker look misplaced.
        for key in ("binary_gaussian", "binary_student_t"):
            short = "Gaussian" if key == "binary_gaussian" else "Student-t"
            for alpha, ls in zip(cfg.far_targets, ["--", ":"]):
                thr = calibrations[(key, alpha)].threshold
                ax.axhline(
                    1.0 / (1.0 + np.exp(-thr)),
                    color=COLOURS[key],
                    ls=ls,
                    lw=0.9,
                    alpha=0.75,
                    label=f"{short} threshold, $\\alpha$={alpha:g}",
                )
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(valid | data up to $t$)")
        ax.set_xlabel("交易日 trading day $t$  (1 = first live day)")
        _mark_eval_days(ax, cfg)
        ax.legend(loc="upper right", framealpha=0.94, fontsize=8.0, ncol=2)

    fig.suptitle(
        "图 1 / Fig 1  固定示例路径：累计收益与两个贝叶斯检测器的有效概率\n"
        "Single example paths — NOT aggregates. "
        f"$D$={cfg.D}, $H$={cfg.horizon_days}d, $\\sigma_{{ann}}$={cfg.sigma_annual:g}, "
        f"prior P(valid)={cfg.prior_valid:g}, Student-t $\\nu$={cfg.student_t_df:g}.\n"
        f"Vertical grey lines: {' / '.join(str(d) for d in cfg.eval_horizons)}d;   "
        f"dashed line: rolling-model start ({cfg.rolling_window}d).",
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.835, bottom=0.085, left=0.085, right=0.985,
                        hspace=0.20, wspace=0.21)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 2 -- FAR and detection curves on the independent test sets
# --------------------------------------------------------------------------- #


def figure_operating_curves(
    cfg: Stage1Config, passages: dict, out_path: Path
) -> Path:
    days = np.arange(1, cfg.horizon_days + 1)
    n_alpha = len(cfg.far_targets)
    fig, axes = plt.subplots(n_alpha, 2, figsize=(13.0, 4.3 * n_alpha), sharex=True)
    axes = np.atleast_2d(axes)

    for i, alpha in enumerate(cfg.far_targets):
        ref = random_closure_reference(alpha, cfg.horizon_days)
        for j, (set_name, what, ylab) in enumerate(
            [
                (
                    "test_valid",
                    "累计误杀率 cumulative FALSE-ALARM rate | true state VALID (S=1)",
                    "$\\Pr(\\tau \\leq t \\mid S=1)$",
                ),
                (
                    "test_invalid",
                    "累计检出率 cumulative DETECTION rate | true state INVALID (S=0)",
                    "$\\Pr(\\tau \\leq t \\mid S=0)$",
                ),
            ]
        ):
            ax = axes[i, j]
            for key in DETECTORS_BY_KEY:
                fp = passages[(key, alpha, set_name)]
                ax.plot(
                    days,
                    fp.cumulative_rate,
                    color=COLOURS[key],
                    ls=STYLES[key],
                    lw=1.5,
                    label=DETECTORS_BY_KEY[key].label,
                )
            ax.plot(
                days,
                ref["cumulative_rate"],
                color=COLOURS["random_closure"],
                ls=STYLES["random_closure"],
                lw=1.5,
                label="Random closure (analytic reference)",
            )
            if j == 0:
                ax.axhline(
                    alpha,
                    color="k",
                    lw=0.9,
                    ls="-.",
                    label=f"FAR budget $\\alpha$ = {alpha:g}",
                )
                ax.set_ylim(0, max(alpha * 2.2, 0.06))
            else:
                ax.set_ylim(0, 1.0)
            _mark_eval_days(ax, cfg)
            n = passages[("binary_gaussian", alpha, set_name)].n_paths
            ax.set_title(f"{what}\nFAR target $\\alpha$ = {alpha:g},  n = {n} test paths", fontsize=10)
            ax.set_ylabel(ylab)
            if i == n_alpha - 1:
                ax.set_xlabel("交易日 trading day $t$")
            ax.legend(loc="upper left", framealpha=0.92, ncol=1)

    fig.suptitle(
        "图 2 / Fig 2  独立测试集上的累计误杀率与累计检出率（阈值仅在校准集上选定）\n"
        f"Thresholds calibrated on {cfg.n_calibration:,} held-out VALID paths, then frozen. "
        f"Dashed vertical line = first day the {cfg.rolling_window}d rolling detectors may alarm; "
        f"grey lines = {' / '.join(str(d) for d in cfg.eval_horizons)} trading days.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 3 -- truncated mean detection time and miss rate
# --------------------------------------------------------------------------- #


def figure_detection_time(
    cfg: Stage1Config, rows: list[dict], out_path: Path
) -> Path:
    keys = list(DETECTORS_BY_KEY) + ["random_closure"]
    labels = [
        DETECTORS_BY_KEY[k].label if k in DETECTORS_BY_KEY else "Random closure\n(analytic)"
        for k in keys
    ]
    by = {(r["detector"], r["far_target"]): r for r in rows}
    x = np.arange(len(keys), dtype=float)
    width = 0.8 / len(cfg.far_targets)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    ax = axes[0]
    for i, alpha in enumerate(cfg.far_targets):
        vals, errs = [], []
        for k in keys:
            r = by[(k, alpha)]
            vals.append(float(r["trunc_mean_detect_days"]))
            se = r["trunc_mean_detect_days_se"]
            errs.append(float(se) if se != "" else 0.0)
        off = (i - (len(cfg.far_targets) - 1) / 2) * width
        bars = ax.bar(
            x + off, vals, width, yerr=errs, capsize=3,
            label=f"$\\alpha$ = {alpha:g}", alpha=0.88,
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.axhline(cfg.horizon_days, color="k", ls="--", lw=1.0, label=f"H = {cfg.horizon_days}d (ceiling)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("截断平均检测时间 $E[\\min(\\tau, H)]$  (trading days)")
    ax.set_title(
        "无效策略的截断平均检测时间\n"
        f"Truncated mean detection time | S = 0, n = {cfg.n_test_invalid} test paths\n"
        "error bars = Monte-Carlo SE of the mean; undetected paths counted at H",
        fontsize=10,
    )
    ax.set_ylim(0, cfg.horizon_days * 1.32)
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    sec = ax.secondary_yaxis("right", functions=(lambda d: d / cfg.D, lambda y: y * cfg.D))
    sec.set_ylabel("years")

    ax = axes[1]
    for i, alpha in enumerate(cfg.far_targets):
        vals, lo, hi = [], [], []
        for k in keys:
            r = by[(k, alpha)]
            v = float(r["undetected_at_H"])
            vals.append(v)
            lo.append(v - float(r["undetected_at_H_lo"]) if r["undetected_at_H_lo"] != "" else 0.0)
            hi.append(float(r["undetected_at_H_hi"]) - v if r["undetected_at_H_hi"] != "" else 0.0)
        off = (i - (len(cfg.far_targets) - 1) / 2) * width
        bars = ax.bar(
            x + off, vals, width, yerr=[lo, hi], capsize=3,
            label=f"$\\alpha$ = {alpha:g}", alpha=0.88,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("两年仍未检出的比例  $\\Pr(\\tau > H \\mid S=0)$")
    ax.set_title(
        "两年监控期结束仍未检出的比例\n"
        f"Still-undetected fraction at H = {cfg.horizon_days}d | S = 0\n"
        "error bars = pointwise Wilson 95% (Monte-Carlo noise only)",
        fontsize=10,
    )
    ax.set_ylim(0, 1.28)
    ax.legend(loc="upper center", ncol=2, fontsize=8)

    fig.suptitle(
        "图 3 / Fig 3  检测速度与漏检：截断平均检测时间和两年未检出比例\n"
        "Random-closure bars are exact analytic values, so they carry no Monte-Carlo interval.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 4 -- single-shock diagnostic
# --------------------------------------------------------------------------- #


def figure_shock_diagnostic(
    cfg: Stage1Config, diag: dict, calibrations: dict, out_path: Path
) -> Path:
    days = np.arange(1, cfg.horizon_days + 1)
    variants = [
        ("base", "原始 base", "#333333", "-"),
        ("shock_plus", f"+{cfg.shock_in_daily_sigma:g}$\\sigma_d$ on day {cfg.shock_day}", "#1a9850", "-"),
        ("shock_minus", f"-{cfg.shock_in_daily_sigma:g}$\\sigma_d$ on day {cfg.shock_day}", "#d73027", "-"),
    ]
    models = [("binary_gaussian", "Binary Gaussian"), ("binary_student_t", f"Binary Student-t ($\\nu$={cfg.student_t_df:g})")]

    fig = plt.figure(figsize=(13.0, 13.4))
    # extra hspace: the bottom panel carries a 3-line title that must clear the
    # zoomed increment panels' x-axis labels above it.
    gs = fig.add_gridspec(
        4, 2, height_ratios=[1.0, 1.0, 1.0, 1.15], hspace=0.78, wspace=0.22
    )
    axes = np.empty((3, 2), dtype=object)
    for rr in range(3):
        for cc in range(2):
            axes[rr, cc] = fig.add_subplot(gs[rr, cc])
    ax_inf = fig.add_subplot(gs[3, :])
    lo = cfg.shock_day - 6
    hi = cfg.shock_day + 6

    for j, (mkey, mlabel) in enumerate(models):
        ax = axes[0, j]
        for vkey, vlabel, col, ls in variants:
            ax.plot(days, diag[mkey][vkey]["log_odds"], color=col, ls=ls, lw=1.3, label=vlabel)
        for alpha, tls in zip(cfg.far_targets, ["--", ":"]):
            ax.axhline(
                calibrations[(mkey, alpha)].threshold, color="0.35", ls=tls, lw=0.9,
                label=f"alarm threshold, $\\alpha$={alpha:g}",
            )
        ax.axvline(cfg.shock_day, color="0.6", lw=0.9)
        ax.set_title(f"{mlabel} — log odds $L_t$", fontsize=10.5)
        ax.set_ylabel("$L_t$ = log P(valid)/P(invalid)")
        ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92)

        ax = axes[1, j]
        for vkey, vlabel, col, ls in variants:
            ax.plot(days, diag[mkey][vkey]["prob"], color=col, ls=ls, lw=1.3, label=vlabel)
        ax.axvline(cfg.shock_day, color="0.6", lw=0.9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{mlabel} — P(valid)", fontsize=10.5)
        ax.set_ylabel("P(valid | data up to $t$)")

        ax = axes[2, j]
        w = 0.26
        for i, (vkey, vlabel, col, ls) in enumerate(variants):
            inc = diag[mkey][vkey]["increment"][lo - 1 : hi]
            ax.bar(np.arange(lo, hi + 1) + (i - 1) * w, inc, w, color=col, label=vlabel, alpha=0.9)
        ax.axvline(cfg.shock_day, color="0.6", lw=0.9)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(f"{mlabel} — one-step log-likelihood increment", fontsize=10.5)
        ax.set_ylabel("$\\log p(z_t|S{=}1) - \\log p(z_t|S{=}0)$")
        ax.set_xlabel(f"交易日 trading day (zoom {lo}–{hi})")
        if j == 0:
            ax.legend(loc="lower left", fontsize=8)

    for j in range(2):
        axes[0, j].set_xlabel("")
        axes[1, j].set_xlabel("交易日 trading day $t$")

    # ---- why the two models react differently: the influence functions ----- #
    curve = influence_curve(cfg, -25.0, 25.0, 6001)
    ax_inf.plot(curve["z"], curve["gaussian"], color=COLOURS["binary_gaussian"], lw=1.6,
                label="Binary Gaussian: $z/\\sqrt{D} - 1/(2D)$ — affine, unbounded")
    ax_inf.plot(curve["z"], curve["student_t"], color=COLOURS["binary_student_t"], lw=1.6,
                label=f"Binary Student-t ($\\nu$={cfg.student_t_df:g}) — bounded and redescending")
    ax_inf.axhline(0.0, color="0.4", lw=0.8)
    ax_inf.set_ylim(-0.75, 0.75)
    short = {"base": "原始 base", "shock_plus": f"+{cfg.shock_in_daily_sigma:g}$\\sigma_d$",
             "shock_minus": f"-{cfg.shock_in_daily_sigma:g}$\\sigma_d$"}
    for vkey, _vlabel, col, _ls in variants:
        z_shock = float(diag["z_on_shock_day"][vkey])
        ax_inf.axvline(z_shock, color=col, lw=1.0, ls="--", alpha=0.85)
        ax_inf.annotate(
            f"{short[vkey]}\n$z$={z_shock:+.2f}", xy=(z_shock, 0.66), fontsize=7.5,
            color=col, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col, lw=0.7, alpha=0.92),
        )
    ax_inf.set_xlabel("标准化收益 standardised return  $z_t = r_t / \\sigma_d$")
    ax_inf.set_ylabel("one-step increment\n$\\log p(z|S{=}1) - \\log p(z|S{=}0)$")
    ax_inf.set_title(
        "为什么两条曲线反应不同：一步增量作为 $z$ 的函数（影响函数）\n"
        "Why the two react differently: the Gaussian update grows without bound in $z$,\n"
        "while the Student-t update peaks near $|z|\\approx2$ and then decays back to 0 —\n"
        "an extreme observation is nearly equally implausible under BOTH states, so it "
        "carries almost no evidence.",
        fontsize=9.5,
    )
    ax_inf.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "图 4 / Fig 4  单次冲击诊断（固定种子、固定路径，非 benchmark 结果）\n"
        f"One fixed VALID path (stream 'diagnostic', path #{cfg.diagnostic_path_index}); the two copies\n"
        f"differ from it ONLY by $\\pm{cfg.shock_in_daily_sigma:g}\\sigma_d$ added on day {cfg.shock_day}.  "
        "Curves are shown past the first crossing for\nillustration; "
        "$\\tau$ is still recorded at the first crossing.",
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.885, bottom=0.045, left=0.075, right=0.985)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
