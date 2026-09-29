"""market_recovery: one module per research stage.

- Model and computation: Calibration-recovery experiment: is the existing procedure sound when the true
- Figures: The two recovery figures.
- Report: Report for the calibration-recovery experiment.
- Command-line entry: Calibration-recovery experiment.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import hashlib
import math

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage3 as m3
from . import market_stage4 as m4

# ---------------------------------------------------------------- protocol --
SCENARIOS: dict[str, tuple[float, float]] = {
    "A1.0_rho0.98": (1.0, 0.98),
    "A1.4_rho0.96": (1.4, 0.96),
}
TRUE_SIGMA_ANNUAL = 0.192          # a DESIGN value for this simulation study
N_OUTER = 20                       # independent outer replicates per scenario
N_DAYS = 1258                      # pseudo-training and pseudo-evaluation length
BASELINE = m4.BASELINE             # the fixed shape, A = 1.0, rho = 0.98
GRID_A, GRID_RHO = m4.GRID_A, m4.GRID_RHO
SCREEN_PATHS, FINAL_PATHS, N_FINALISTS = 400, 2000, 3
EVAL_PATHS = 2000

# every stream is separately addressed; nothing is shared by accident
E_TRAIN_DATA = 20260941
E_EVAL_DATA = 20260942
E_CAL_REF = 20260943
E_CAL_SCREEN = 20260944
E_CAL_FINAL = 20260945
E_EVAL_REF = 20260946
E_EVAL_SIM = 20260947

VERSIONS = ("fixed_baseline", "selected", "true_shape", "true_everything")
HELD_OUT = ("tail_below_m3_count", "tail_above_p3_count",
            "acf_sqret_lag1", "acf_sqret_lag5", "acf_sqret_lag21")


def tag(*parts) -> int:
    """Stable across processes; never Python's per-process string hash."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") & 0x7FFFFFFF


def pure_sv(cfg, A: float, rho: float, sigma_annual: float):
    """kappa = 0 makes the sv_jump generator reproduce pure SV bit for bit."""
    return m3.branch_cfg(cfg, "sv_only", A, rho, sigma_annual)


def draw(cfg, A: float, rho: float, sigma_annual: float, n_paths: int, n_days: int,
         entropy: int, *parts) -> np.ndarray:
    """Stationary initialisation, Sharpe 0, no post-hoc rescaling of any path."""
    return m3.simulate(pure_sv(cfg, A, rho, sigma_annual), n_paths, n_days,
                       np.random.SeedSequence([entropy, tag(*parts)]))


def reference_scales(cfg, sigma_annual: float, n_days: int, entropy: int, *parts,
                     n_paths: int = FINAL_PATHS):
    """Across-path spread under the FIXED BASELINE, using only information the
    fitter is allowed to have: the baseline parameters and the estimated scale."""
    st = m2.path_statistics_chunked(
        draw(cfg, BASELINE[0], BASELINE[1], sigma_annual, n_paths, n_days,
             entropy, *parts))
    return m3.standardising_scales({k: st[k] for k in
                                    m3.RV_TARGETS + m3.ACF_TARGETS})


def calibrate_array(cfg, x: np.ndarray, sigma_annual: float, scales: dict,
                    *parts, screen_paths: int = SCREEN_PATHS,
                    final_paths: int = FINAL_PATHS,
                    n_finalists: int = N_FINALISTS) -> dict:
    """The stage-3/4 calibration, applied to a bare return array.

    Every step is the one the earlier stages use and the test-suite asserts the
    equivalence: 40 cells screened on ONE common stream, the best `n_finalists` plus
    the fixed baseline re-run on an independent stream, the lowest re-check loss
    selected, and anything within one Monte-Carlo standard error reported with it.
    The truth is never passed in.
    """
    n_days = len(x)
    real_t = m3.real_targets({k: float(v[0]) for k, v in
                              m2.path_statistics(x[None, :]).items()})
    screen_seed = np.random.SeedSequence([E_CAL_SCREEN, tag(*parts)])
    rows = []
    for A in GRID_A:
        for rho in GRID_RHO:
            st = m2.path_statistics_chunked(
                m3.simulate(pure_sv(cfg, A, rho, sigma_annual), screen_paths, n_days,
                            screen_seed))
            L = m3.loss(m3.target_values(st), real_t, scales)
            rows.append({"A": A, "rho": rho, "loss": L["loss"],
                         "rv21_component": L["rv21_component"],
                         "acf_abs_component": L["acf_abs_component"]})
    grid = pd.DataFrame(rows).sort_values("loss", ignore_index=True)
    grid["screen_rank"] = np.arange(1, len(grid) + 1)

    cand = grid.head(n_finalists)[["A", "rho"]].values.tolist()
    if [BASELINE[0], BASELINE[1]] not in cand:
        cand.append([BASELINE[0], BASELINE[1]])
    fin = []
    for i, (A, rho) in enumerate(cand):
        st = m2.path_statistics_chunked(
            m3.simulate(pure_sv(cfg, A, rho, sigma_annual), final_paths, n_days,
                        np.random.SeedSequence([E_CAL_FINAL, tag(*parts), i])))
        L = m3.loss(m3.target_values(st), real_t, scales)
        fin.append({"A": A, "rho": rho, "loss": L["loss"],
                    "loss_mc_se": m3.loss_mc_se(st, real_t, scales),
                    "is_baseline": A == BASELINE[0] and rho == BASELINE[1]})
    fin = pd.DataFrame(fin).sort_values("loss", ignore_index=True)
    best = fin.iloc[0]
    return {"A": float(best.A), "rho": float(best.rho),
            "training_loss": float(best.loss),
            "training_loss_mc_se": float(best.loss_mc_se),
            "n_within_one_mc_se": int((fin.loss <= best.loss + best.loss_mc_se).sum()),
            "on_grid_boundary": bool(best.A in (GRID_A[0], GRID_A[-1]) or
                                     best.rho in (GRID_RHO[0], GRID_RHO[-1])),
            "grid": grid, "finalists": fin}


# ------------------------------------------------------- secondary scoring --


def energy_score(sim_a: np.ndarray, sim_b: np.ndarray, obs: np.ndarray) -> float:
    """ES = mean||X - y|| - 0.5 mean||X - X'||, on the scaled 8-vector.

    `sim_a` and `sim_b` are two INDEPENDENT halves of the simulated statistic
    vectors, so no pairwise distance matrix is built. Evaluation only: it never
    selects a candidate and it does not replace the loss.
    """
    n = min(len(sim_a), len(sim_b))
    a, b = sim_a[:n], sim_b[:n]
    return float(np.linalg.norm(a - obs, axis=1).mean()
                 - 0.5 * np.linalg.norm(a - b, axis=1).mean())


def stat_vectors(stats: dict, scales: dict) -> np.ndarray:
    """The 8 fitted statistics per path, in the loss's own transform and scale."""
    cols = []
    for k in m3.RV_TARGETS:
        cols.append(np.log(np.asarray(stats[k], float)) / scales[k])
    for k in m3.ACF_TARGETS:
        cols.append(np.asarray(stats[k], float) / scales[k])
    return np.column_stack(cols)


def obs_vector(real: dict, scales: dict) -> np.ndarray:
    v = []
    for k in m3.RV_TARGETS:
        v.append(math.log(real[k]) / scales[k])
    for k in m3.ACF_TARGETS:
        v.append(real[k] / scales[k])
    return np.asarray(v)

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"A1.0_rho0.98": "#1f77b4", "A1.4_rho0.96": "#d62728"}


def parameter_selection(res: pd.DataFrame, grid_a, grid_rho, out_path: Path) -> Path:
    sel = res[res.version == "selected"]
    scens = list(dict.fromkeys(sel.scenario))
    fig, axes = plt.subplots(1, len(scens) + 1, figsize=(4.4 * (len(scens) + 1), 4.6))
    fig.suptitle("What the existing calibration chose, on pseudo-histories whose truth "
                 "we know\nPure SV; the fitter never saw the true parameters, the true "
                 "scale or the latent path.", fontsize=12, y=0.97)
    for j, scen in enumerate(scens):
        ax = axes[j]
        s = sel[sel.scenario == scen]
        tA, trho = float(s.true_A.iloc[0]), float(s.true_rho.iloc[0])
        counts = s.groupby(["A", "rho"]).size()
        for (A, rho), n in counts.items():
            ax.scatter(rho, A, s=40 + 90 * n, color=COL.get(scen, "0.4"), alpha=0.55,
                       edgecolor="k", lw=0.6, zorder=3)
            ax.annotate(str(n), (rho, A), ha="center", va="center", fontsize=8.5,
                        zorder=4)
        ax.scatter([trho], [tA], marker="*", s=420, color="gold", edgecolor="k",
                   lw=1.1, zorder=5, label="the truth")
        ax.set_xticks(list(grid_rho)); ax.set_yticks(list(grid_a))
        ax.set_xlabel("rho selected"); ax.set_ylabel("A selected")
        hit = int(s.selected_is_true.sum())
        ax.set_title(f"{scen.replace('_', ', ')}\nchose the truth in {hit} of {len(s)} "
                     f"replicates", fontsize=10.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.92,
                  facecolor="white", edgecolor="0.7")
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[-1]
    for scen in scens:
        s = sel[sel.scenario == scen]
        ax.scatter(s.sigma_hat_error * 100, s.A, color=COL.get(scen, "0.4"), s=42,
                   alpha=0.75, label=scen.replace("_", ", "))
    ax.axvline(0, color="0.4", ls="--", lw=1.1)
    ax.set_xlabel("estimated annualised scale − true scale\n(percentage points)",
                  fontsize=9.5)
    ax.set_ylabel("A selected")
    ax.set_title("the scale estimate, and what was chosen with it", fontsize=10.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015,
             "Spread across replicates is outer pseudo-history variation. It is not "
             "real-market uncertainty, and 20 replicates support a preliminary "
             "diagnosis for these two fixed scenarios only.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.9))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def evaluation_difference(res: pd.DataFrame, out_path: Path,
                          finalist_fields: pd.DataFrame) -> Path:
    """`finalist_fields` carries the FINAL RE-CHECK training losses of the fixed
    baseline and of the selected configuration.  They cannot be taken from `res`:
    its `training_loss` column is a per-replicate field repeated on all four
    version rows, so differencing it across versions yields zero everywhere."""
    piv = res.pivot_table(index=["scenario", "replicate"], columns="version",
                          values="eval_loss")
    scens = list(dict.fromkeys(res.scenario))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax = axes[0]
    data, labels, colours = [], [], []
    for scen in scens:
        p = piv.loc[scen]
        for other, name in (("selected", "calibrated"), ("true_shape", "true shape"),
                            ("true_everything", "true everything")):
            data.append((p[other] - p["fixed_baseline"]).to_numpy())
            labels.append(f"{scen.split('_')[0]}\n{name}")
            colours.append(COL.get(scen, "0.4"))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "lw": 1.6})
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.13, 0.13, len(d)), d, s=13,
                   color="0.25", alpha=0.65, zorder=3)
    ax.axhline(0, color="black", lw=1.2)
    for i, lab in enumerate(labels, start=1):
        if "true shape" in lab and "A1.0" in lab:
            ax.annotate("identical to the baseline\nby construction", (i, 0),
                        xytext=(0, 26), textcoords="offset points", ha="center",
                        fontsize=8, color="0.35",
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    ax.set_ylabel("independent-replication loss\nminus the fixed baseline")
    ax.set_title("(a) paired difference on data the fit never saw\n"
                 "below zero = better than leaving the shape fixed", fontsize=11)
    ax.tick_params(axis="x", labelsize=8.5)
    ax = axes[1]
    for scen in scens:
        s = res[(res.scenario == scen) & (res.version == "selected")]
        b = res[(res.scenario == scen) & (res.version == "fixed_baseline")]
        m = s.merge(b, on=["scenario", "replicate"], suffixes=("_sel", "_base")).merge(
            finalist_fields[["scenario", "replicate", "training_improvement"]],
            on=["scenario", "replicate"], validate="one_to_one")
        # x = training improvement from the final re-check, where the fixed baseline and
        # the selected configuration were re-scored on ONE stream at the full finalist
        # budget.  Both terms carry selection error -- the winner was chosen on that
        # stream -- and the finite-simulation error of that budget.
        ax.scatter(m.training_improvement,
                   m.eval_loss_sel - m.eval_loss_base,
                   color=COL.get(scen, "0.4"), s=44, alpha=0.8,
                   label=scen.replace("_", ", "))
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("training improvement over the fixed baseline\n"
                  "(baseline − selected, final re-check)")
    ax.set_ylabel("evaluation loss, calibrated − fixed")
    ax.set_title("(b) training improvement against what it cost or bought\n"
                 "on an independent replication", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    for a in axes:
        a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
    # the note is wrapped by hand: at this figure width one line overruns the canvas
    # and matplotlib clips it silently at both ends.
    fig.text(0.5, 0.035,
             "The four versions share one evaluation stream inside a replicate, which reduces the "
             "noise in a paired difference but does not remove the finite-simulation\n"
             "error each loss still carries. Panel (b)'s training improvement is measured on the "
             "stream the winner was selected on, so it carries selection error as well.\n"
             "'True everything' is an oracle reference, not a lower bound of this loss.",
             ha="center", va="bottom", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0.01, 0.135, 0.99, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main_plots(argv: list[str] | None = None) -> int:
    import argparse
    pass

    ap = argparse.ArgumentParser(description="Redraw the recovery figures from the "
                                             "saved artefacts; no simulation is re-run.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    a = ap.parse_args(argv)
    res = pd.read_csv(a.out / "market_recovery_results.csv")
    ff = pd.read_csv(a.out / "market_recovery_finalist_fields.csv")
    for q in (parameter_selection(res, GRID_A, GRID_RHO,
                                  a.out / "market_recovery_parameter_selection.png"),
              evaluation_difference(res,
                                    a.out / "market_recovery_evaluation_difference.png",
                                    ff)):
        print(q)
    return 0

# ============================================================================
# Report
# ============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


from . import market_stage3 as m3


def build(out: Path) -> str:
    s = json.loads((out / "market_recovery_summary.json").read_text(encoding="utf-8"))
    r = pd.read_csv(out / "market_recovery_results.csv")
    p = s["protocol"]
    piv = r.pivot_table(index=["scenario", "replicate"], columns="version",
                        values="eval_loss")
    epiv = r.pivot_table(index=["scenario", "replicate"], columns="version",
                         values="energy_score")
    sel = r[r.version == "selected"]

    L: list[str] = []
    A = L.append
    A("# Calibration recovery — is the procedure sound when the truth is known?\n")
    A(f"{p['scope']} {p['budget']['total_fits']} fits: "
      f"{p['budget']['outer_replicates_per_scenario']} independent pseudo-histories "
      f"for each of two true parameter settings, each "
      f"{p['budget']['days_per_series']:,} days, with an independent "
      f"{p['budget']['days_per_series']:,}-day replication from the same DGP for "
      f"scoring. True annualised scale {p['true_scale']['sigma_annual']} — "
      f"{p['true_scale']['status']}. Wall time {s['wall_seconds']:.0f}s.\n")
    A(f"**{p['replication_semantics']}**\n")
    A(f"The protocol was written before the run and the existing procedure was checked "
      f"against its description first: {p['unchanged_from_the_earlier_stages']}\n")

    A("\n## 1. Parameter recovery\n")
    A("![what was chosen](market_recovery_parameter_selection.png)\n")
    ff = pd.read_csv(out / "market_recovery_finalist_fields.csv")
    A("Two different things were conflated in the first write-up and are now separate. "
      "A cell can be **screened into the top three**, or it can be **actually scored "
      "in the final re-check** — and the fixed baseline is appended to the candidate "
      "set whenever the screen has not already produced it, so in the scenario where "
      "the truth *is* the baseline it was scored every time whatever its screening "
      "rank.\n")
    A("| scenario | truth | chose the truth | median A | median rho | truth's median screening rank | truth in the screened top 3 | truth ACTUALLY scored | on a grid edge |")
    A("|---|---|---|---|---|---|---|---|---|")
    for scen in SCENARIOS:
        g = sel[sel.scenario == scen]
        f = ff[ff.scenario == scen]
        A(f"| {scen} | A={g.true_A.iloc[0]:g}, rho={g.true_rho.iloc[0]:g} | "
          f"**{int(g.selected_is_true.sum())} of {len(g)}** | {g.A.median():g} | "
          f"{g.rho.median():g} | {int(g.true_screen_rank.median())} of 40 | "
          f"{int(f.truth_in_screen_top3.sum())} of {len(f)} | "
          f"**{int(f.truth_actually_scored_in_finalists.sum())} of {len(f)}** | "
          f"{int(g.on_grid_boundary.sum())} of {len(g)} |")
    A("")
    pe = pd.read_csv(out / "market_recovery_parameter_errors.csv")
    A("Selection error against the truth, over the 20 pseudo-histories:\n")
    A("| scenario | quantity | mean error | its Monte-Carlo s.e. | median error | RMSE |")
    A("|---|---|---|---|---|---|")
    for _, q in pe[pe.parameter != "grid_boundary_selection_rate"].iterrows():
        A(f"| {q.scenario} | {q.parameter} | {q.mean_error:+.4f} | "
          f"{q['mean_error_mc_se']:.4f} | {q.median_error:+.4f} | {q.rmse:.4f} |")
    A("")
    sg = pe[(pe.parameter == "sigma_annual")]
    A(f"On the scale estimate specifically: the mean error is "
      f"{sg.mean_error.iloc[0]:+.4f} and {sg.mean_error.iloc[1]:+.4f} in annualised "
      f"decimal terms ({100*sg.mean_error.iloc[0]:+.2f} and "
      f"{100*sg.mean_error.iloc[1]:+.2f} percentage points), each with a "
      f"Monte-Carlo standard error of about {sg['mean_error_mc_se'].mean():.4f}, and "
      f"an RMSE of {sg.rmse.iloc[0]:.4f} and {sg.rmse.iloc[1]:.4f}. That is the "
      f"measurement; whether a scale error of this size matters for the loss is not "
      f"settled by this experiment and no claim is made either way.\n")
    A("Every 2.5–97.5% figure here is an **empirical range across 20 simulated "
      "pseudo-histories under a known DGP**. It is not a confidence interval for any "
      "real-market parameter.\n")

    A("\n## 2. Distribution recovery on the independent replication\n")
    A("![evaluation difference](market_recovery_evaluation_difference.png)\n")
    ps = pd.read_csv(out / "market_recovery_paired_statistics.csv")
    A("All differences below are **paired**, computed replicate by replicate and then "
      "summarised. The median of the paired differences is not the difference of the "
      "two medians, and both are shown so the gap is visible.\n")
    for metric, title in (("J_eval_loss", "J, the fitted objective on the "
                                          "independent replication"),
                          ("energy_score", "the secondary energy score")):
        A(f"**{title}**\n")
        A("| scenario | comparison | paired mean | its MC s.e. | paired median | difference of the two medians | better | tied | worse |")
        A("|---|---|---|---|---|---|---|---|---|")
        for _, q in ps[ps.metric == metric].iterrows():
            A(f"| {q.scenario} | {q.comparison} | {q.paired_mean:+.4f} | "
              f"{q.paired_mean_mc_se:.4f} | {q.paired_median:+.4f} | "
              f"{q['difference_of_the_two_medians']:+.4f} | {int(q.n_better)} | "
              f"{int(q.n_tied)} | {int(q.n_worse)} |")
        A("")
    A("The four versions share one evaluation stream inside a replicate. Common random "
      "numbers **reduce the noise of the paired comparison**; they do not remove the "
      "finite simulation error, which is why every paired mean above carries its own "
      "Monte-Carlo standard error. "
      f"**{p['versions_compared']['true_everything']}**\n")
    A("About the energy score: it uses **the same transform and the same "
      "standardisation as J** — log for the three realised-volatility quantiles, raw "
      "for the five autocorrelations, each divided by the same reference scale — but "
      "**not J's two-group weighting**. J gives each volatility target 1/6 and each "
      "autocorrelation target 1/10; the energy score weights all eight equally at 1/8. "
      "That was its pre-set definition and it has not been changed after seeing the "
      "results. It evaluates the joint distribution of the eight-statistic summary "
      "only; it does not verify the full return process. Where it disagrees with J the "
      "disagreement is left standing.\n")

    A("\n## 3. The unfitted diagnostics\n")
    hd = pd.read_csv(out / "market_recovery_heldout_diagnostics.csv")
    A("Five statistics that took no part in the objective, reported per item rather "
      "than summarised into a verdict. 'Inside' counts how often the replication's own "
      "value fell in the simulated 2.5–97.5% range; the raw error is the simulated "
      "median minus that value, in the statistic's own units.\n")
    for scen in SCENARIOS:
        A(f"**{scen}**\n")
        A("| statistic | " + " | ".join(f"{v} inside / mean raw err" for v in VERSIONS) + " |")
        A("|---|" + "---|" * len(VERSIONS))
        for k in HELD_OUT:
            cells = []
            for v in VERSIONS:
                q = hd[(hd.scenario == scen) & (hd.statistic == k) &
                       (hd.version == v)].iloc[0]
                cells.append(f"{int(q.n_inside_sim_range)}/{int(q.n)}, "
                             f"{q.mean_raw_error:+.3f}")
            A(f"| `{k}` | " + " | ".join(cells) + " |")
        A("")

    A("\n## 4. The four questions\n")
    ps = pd.read_csv(out / "market_recovery_paired_statistics.csv")

    def pick(scen, comp, metric="J_eval_loss"):
        return ps[(ps.scenario == scen) & (ps.comparison == comp) &
                  (ps.metric == metric)].iloc[0]

    q1 = pick("A1.0_rho0.98", "selected - fixed_baseline")
    A(f"**Does calibrating hurt an independent sample when the baseline is already "
      f"right?** In the scenario where the truth *is* the fixed baseline, the paired "
      f"difference has mean **{q1.paired_mean:+.3f}** with a Monte-Carlo standard "
      f"error of {q1.paired_mean_mc_se:.3f} and median {q1.paired_median:+.3f}: "
      f"**{int(q1.n_better)} better, {int(q1.n_tied)} tied, {int(q1.n_worse)} worse** "
      f"out of {int(q1.n)}. The ties are the replicates in which the calibration "
      f"selected the baseline itself. The direction is towards a cost, and the mean is "
      f"about {abs(q1.paired_mean / q1.paired_mean_mc_se):.1f} standard errors from "
      f"zero — suggestive at this budget, not established.\n")
    q2 = pick("A1.4_rho0.96", "selected - fixed_baseline")
    t2 = pick("A1.4_rho0.96", "true_shape - fixed_baseline")
    A(f"**Does calibrating help when the baseline is biased?** Paired mean "
      f"**{q2.paired_mean:+.3f}** (s.e. {q2.paired_mean_mc_se:.3f}), median "
      f"{q2.paired_median:+.3f}, better in {int(q2.n_better)} of {int(q2.n)}. Knowing "
      f"the true shape instead gives a paired mean of **{t2.paired_mean:+.3f}** "
      f"(s.e. {t2.paired_mean_mc_se:.3f}). Both are negative and the calibrated figure "
      f"is the smaller of the two in magnitude; the ratio between them is **not** "
      f"quoted, because dividing two means with standard errors of this size does not "
      f"produce a supportable number.\n")
    e2 = pick("A1.4_rho0.96", "selected - fixed_baseline", "energy_score")
    A(f"The energy score agrees in sign and is somewhat more favourable to calibrating "
      f"in this scenario ({int(e2.n_better)} of {int(e2.n)} better against "
      f"{int(q2.n_better)} on J). It carries different weights and evaluates only the "
      f"eight-statistic summary, so the two are not interchangeable.\n")
    sel2 = sel[sel.scenario == "A1.4_rho0.96"]
    A(f"**Does unstable parameter selection mean the generated distribution is "
      f"unstable?** This experiment does not answer that. What it can report is that "
      f"the selection moves across pseudo-histories — the exact truth was chosen in "
      f"{int(sel[sel.scenario=='A1.0_rho0.98'].selected_is_true.sum())} and "
      f"{int(sel2.selected_is_true.sum())} of 20 — while the paired evaluation "
      f"differences above stay modest in the first scenario. Reading that as evidence "
      f"that the distribution is stable, or that the parameters are unidentifiable, "
      f"would require a common yardstick for the two spreads, which this design does "
      f"not provide. The fixed-history check in the companion run isolates one "
      f"component and is reported in `market_sensitivity_report.md`: at four "
      f"fixed pseudo-histories, reseeding only the screening and finalist "
      f"streams moved the selection in none of twenty fits, so at those "
      f"histories the internal simulation draw is not what moved the choice. "
      f"What did move it is not identified.\n")
    A("**What could explain the real-market pattern — better training, worse "
      "afterwards?** This design produces a cost from calibrating **with the mechanism "
      "held completely fixed**: in the first scenario the paired difference is "
      f"{q1.paired_mean:+.3f} on average with the truth unchanged between the training "
      f"and the evaluation series. So selection under a correct baseline is one "
      f"candidate explanation. Whether it accounts for the size of the stage-3 and "
      f"stage-4 reversals is **not** something this experiment can say: those came "
      f"from a different design, on real data whose own statistics moved between the "
      f"periods, and the two were never put on a common footing. The comparison of "
      f"magnitudes made in the earlier write-up is withdrawn.\n")

    A("\n## 4b. What kind of problem is this?\n")
    A("| candidate | what the evidence supports |")
    A("|---|---|")
    A("| an implementation error | **not found**: 13 of 13 protocol descriptions match "
      "the code, the streams are separately addressed, the fitter has no parameter "
      "through which the truth could arrive, and the baseline and the fitted shape "
      "share one estimated scale |")
    ff2 = pd.read_csv(out / "market_recovery_finalist_fields.csv")
    A(f"| the screen not carrying the truth forward | **observed**: the truth reached "
      f"the screened top three in "
      f"{int(ff2[ff2.scenario=='A1.0_rho0.98'].truth_in_screen_top3.sum())} and "
      f"{int(ff2[ff2.scenario=='A1.4_rho0.96'].truth_in_screen_top3.sum())} of 20. "
      f"Whether that is screening noise specifically, as opposed to the objective not "
      f"ranking the truth first, is **not separated by this design** |")
    A("| limited parameter identification | **consistent with** the selection spread, "
      "but not established: no reference scale for 'how much movement is too much' was "
      "fixed in advance |")
    A("| a scoring-criterion problem | **cannot be ruled out**: the objective matches "
      "simulated medians to one sample, so the truth is not guaranteed to minimise it. "
      "The oracle rows show how far the truth itself sits from the baseline |")
    A("| currently indistinguishable | the last three are not separated by this design |")
    A("")

    A("\n## 5. Scope\n")
    for k, v in p["three_uncertainties"].items():
        A(f"- **{k.replace('_', ' ')}** — {v}")
    A(f"- {p['what_20_replicates_support']}")
    A("")
    A("\n## 6. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.market_recovery --timing")
    A(".venv/bin/python -m strategy_survivorship.market_recovery")
    A(".venv/bin/python -m strategy_survivorship.market_recovery report")
    A(".venv/bin/python -m pytest tests/test_market_recovery.py -q")
    A("```")
    return "\n".join(L) + "\n"


def main_report(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recovery-experiment report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    p = args.out / "market_recovery_report.md"
    p.write_text(build(args.out), encoding="utf-8")
    print(p)
    return 0

# ============================================================================
# Command-line entry
# ============================================================================

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg

from . import market_stage2 as m2
from . import market_stage3 as m3
from . import market_stage4 as m4
from .config import DEFAULT

OUT = Path("outputs/market")


def protocol() -> dict:
    return {
        "question": "when the true data-generating mechanism is known, is the existing "
                    "calibration procedure sound?",
        "scope": "pure SV only. No jumps, no leverage term, no monitor, no market data.",
        "scenarios": {k: {"A": v[0], "rho": v[1]} for k, v in SCENARIOS.items()},
        "true_scale": {"sigma_annual": TRUE_SIGMA_ANNUAL,
                       "status": "a DESIGN value for this simulation study, not an "
                                 "estimate from any market",
                       "drift": 0.0},
        "budget": {"outer_replicates_per_scenario": N_OUTER,
                   "total_fits": N_OUTER * len(SCENARIOS),
                   "days_per_series": N_DAYS,
                   "screen_paths": SCREEN_PATHS, "final_paths": FINAL_PATHS,
                   "eval_paths": EVAL_PATHS, "n_finalists": N_FINALISTS,
                   "fixed_in_advance": "the budget is not extended and the truth, the "
                                       "loss and the grid are not changed on the "
                                       "strength of a result"},
        "replication_semantics": "the evaluation series is an INDEPENDENT REPLICATION "
                                 "from the same DGP, drawn on its own stream. It is "
                                 "not a continuation conditioned on the training "
                                 "period's end state, and nothing here is a "
                                 "conditional forecast.",
        "information_isolation": {
            "the_fitter_never_sees": "the true A, the true rho, the true scale or the "
                                     "latent volatility path",
            "scale": "estimated from the pseudo-TRAINING returns only, and the fixed "
                     "baseline and the fitted shape share that one estimate",
            "streams": {"pseudo_training_data": E_TRAIN_DATA,
                        "pseudo_evaluation_data": E_EVAL_DATA,
                        "calibration_reference": E_CAL_REF,
                        "calibration_screening": E_CAL_SCREEN,
                        "calibration_finalists": E_CAL_FINAL,
                        "evaluation_reference": E_EVAL_REF,
                        "evaluation_simulation": E_EVAL_SIM},
            "seed_derivation": "blake2b(coordinates); never Python's per-process "
                               "string hash",
            "common_random_numbers": "the four versions share one evaluation stream "
                                     "inside a replicate, so their differences are not "
                                     "simulation noise between them",
        },
        "unchanged_from_the_earlier_stages": "the 8 targets, their transforms, the two "
                                             "equal-weight groups, the "
                                             "baseline-reference scale rule, the 40-cell "
                                             "grid, 400-path screening and the top-3 "
                                             "plus fixed baseline 2000-path re-check. "
                                             "Verified against the code, 13 of 13 "
                                             "descriptions matched.",
        "true_parameters_in_the_search": "both true pairs happen to be grid points, so "
                                         "their screening rank is well defined. The "
                                         "truth is NOT inserted into the finalists if "
                                         "the original rule does not select it; it is "
                                         "carried only as an evaluation reference.",
        "versions_compared": {
            "fixed_baseline": f"A={BASELINE[0]}, rho={BASELINE[1]}, estimated scale",
            "selected": "the calibrated A and rho, estimated scale",
            "true_shape": "the true A and rho, estimated scale",
            "true_everything": "the true A, rho AND the true scale -- an ORACLE "
                               "REFERENCE, not a lower bound of the loss: the "
                               "objective matches simulated medians to one sample and "
                               "nothing makes the truth minimise it at n = 1258",
        },
        "evaluation_scales": "one reference per replicate, built at the BASELINE with "
                             "the estimated scale, on its own stream, BEFORE the "
                             "evaluation data is used. Shared by all four versions.",
        "secondary_energy_score": "a pre-specified energy score over the JOINT "
                                  "distribution of the 8 statistics, in the same "
                                  "transform and scale, estimated from two independent "
                                  "halves of the simulated paths. Evaluation only: it "
                                  "selects nothing and does not replace the loss.",
        "held_out_diagnostics": list(HELD_OUT),
        "three_uncertainties": {
            "inner_simulation_error": "how much a reported loss moves if the same "
                                      "configuration is simulated again; quantified "
                                      "as loss_mc_se",
            "outer_pseudo_history_variation": "how much the answer moves across the 20 "
                                              "pseudo-histories; this is what the "
                                              "outer spread shows",
            "real_market_uncertainty": "NOT addressed here at all. These are simulated "
                                       "histories from a known DGP.",
        },
        "what_20_replicates_support": "a preliminary diagnosis for these two fixed "
                                      "scenarios. It is not a confidence statement "
                                      "about the S&P 500 or about any future period.",
    }


def one_replicate(cfg, scen: str, rep: int, screen_paths: int, final_paths: int,
                  eval_paths: int) -> dict:
    A_t, rho_t = SCENARIOS[scen]
    x_tr = draw(cfg, A_t, rho_t, TRUE_SIGMA_ANNUAL, 1, N_DAYS,
                   E_TRAIN_DATA, scen, rep)[0]
    x_ev = draw(cfg, A_t, rho_t, TRUE_SIGMA_ANNUAL, 1, N_DAYS,
                   E_EVAL_DATA, scen, rep)[0]
    sigma_hat = float(mdg.moment_summary(x_tr)["sd_ddof1"]) * math.sqrt(mdg.DAYS_PER_YEAR)

    cal_scales, _ = reference_scales(cfg, sigma_hat, N_DAYS, E_CAL_REF,
                                        scen, rep, n_paths=final_paths)
    cal = calibrate_array(cfg, x_tr, sigma_hat, cal_scales, scen, rep,
                             screen_paths=screen_paths, final_paths=final_paths)
    g = cal["grid"]
    hit = g[(g.A == A_t) & (g.rho == rho_t)]
    true_rank = int(hit.screen_rank.iloc[0]) if len(hit) else -1

    eval_scales, _ = reference_scales(cfg, sigma_hat, N_DAYS, E_EVAL_REF,
                                         scen, rep, n_paths=final_paths)
    real_ev = m2.path_statistics(x_ev[None, :])
    real_flat = {k: float(v[0]) for k, v in real_ev.items()}
    real_t = m3.real_targets(real_flat)
    obs = obs_vector(real_flat, eval_scales)

    seed = np.random.SeedSequence([E_EVAL_SIM, tag(scen, rep)])
    rows = []
    for version in VERSIONS:
        A, rho, sig = {"fixed_baseline": (BASELINE[0], BASELINE[1], sigma_hat),
                       "selected": (cal["A"], cal["rho"], sigma_hat),
                       "true_shape": (A_t, rho_t, sigma_hat),
                       "true_everything": (A_t, rho_t, TRUE_SIGMA_ANNUAL)}[version]
        st = m2.path_statistics_chunked(
            m3.simulate(pure_sv(cfg, A, rho, sig), eval_paths, N_DAYS, seed))
        L = m3.loss(m3.target_values(st), real_t, eval_scales)
        vec = stat_vectors(st, eval_scales)
        half = len(vec) // 2
        row = {"scenario": scen, "replicate": rep, "version": version,
               "A": A, "rho": rho, "sigma_used": sig,
               "true_A": A_t, "true_rho": rho_t,
               "sigma_true": TRUE_SIGMA_ANNUAL, "sigma_hat": sigma_hat,
               "sigma_hat_error": sigma_hat - TRUE_SIGMA_ANNUAL,
               "eval_loss": L["loss"],
               "eval_rv21_component": L["rv21_component"],
               "eval_acf_component": L["acf_abs_component"],
               "eval_loss_mc_se": m3.loss_mc_se(st, real_t, eval_scales),
               "energy_score": energy_score(vec[:half], vec[half:2 * half], obs),
               "selected_A": cal["A"], "selected_rho": cal["rho"],
               "training_loss": cal["training_loss"],
               "training_loss_mc_se": cal["training_loss_mc_se"],
               "n_within_one_mc_se": cal["n_within_one_mc_se"],
               "on_grid_boundary": cal["on_grid_boundary"],
               "true_screen_rank": true_rank,
               "true_would_be_finalist": bool(0 < true_rank <= N_FINALISTS),
               "selected_is_true": cal["A"] == A_t and cal["rho"] == rho_t}
        for k in m3.RV_TARGETS + m3.ACF_TARGETS:          # raw-unit gaps
            row[f"gap_{k}"] = float(np.median(st[k])) - real_flat[k]
            row[f"real_{k}"] = real_flat[k]
        for k in HELD_OUT:                              # not fitted
            v = np.asarray(st[k], float)
            row[f"heldout_{k}_sim_median"] = float(np.median(v))
            row[f"heldout_{k}_real"] = real_flat[k]
            row[f"heldout_{k}_inside"] = bool(np.quantile(v, .025) <= real_flat[k]
                                              <= np.quantile(v, .975))
        rows.append(row)
    return {"rows": rows, "grid": cal["grid"].assign(scenario=scen, replicate=rep),
            "finalists": cal["finalists"].assign(scenario=scen, replicate=rep)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibration-recovery experiment.")
    ap.add_argument("--outer", type=int, default=N_OUTER)
    ap.add_argument("--timing", action="store_true",
                    help="one cheap replicate to measure cost; writes nothing")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    cfg = DEFAULT

    if args.timing:
        t0 = time.time()
        one_replicate(cfg, next(iter(SCENARIOS)), 0, 60, 120, 120)
        cheap = time.time() - t0
        full = cheap * (SCREEN_PATHS / 60 * 0.62 + FINAL_PATHS / 120 * 0.38)
        print(f"timing check only, nothing written: one cheap replicate {cheap:.1f}s; "
              f"a full replicate is roughly {full:.0f}s, so "
              f"{N_OUTER * len(SCENARIOS)} fits is roughly "
              f"{full * N_OUTER * len(SCENARIOS) / 60:.0f} min")
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    proto = protocol()
    proto["budget"]["outer_replicates_per_scenario"] = args.outer
    proto["budget"]["total_fits"] = args.outer * len(SCENARIOS)
    (out / "market_recovery_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    print("protocol written before the run")

    rows, grids, fins = [], [], []
    t0 = time.time()
    for scen in SCENARIOS:
        for rep in range(args.outer):
            r = one_replicate(cfg, scen, rep, SCREEN_PATHS, FINAL_PATHS,
                              EVAL_PATHS)
            rows += r["rows"]; grids.append(r["grid"]); fins.append(r["finalists"])
            if (rep + 1) % 5 == 0:
                print(f"  {scen}  {rep + 1}/{args.outer}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(out / "market_recovery_results.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(
        out / "market_recovery_grids.csv.gz", index=False, compression="gzip")
    pd.concat(fins, ignore_index=True).to_csv(
        out / "market_recovery_finalists.csv", index=False)

    (out / "market_recovery_summary.json").write_text(json.dumps({
        "protocol": proto, "code_identity": m4.code_identity(),
        "written_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": time.time() - t0,
        "n_rows": int(len(res)),
    }, indent=2, default=float), encoding="utf-8")

    piv = res.pivot_table(index=["scenario", "replicate"], columns="version",
                          values="eval_loss")
    print()
    for scen in SCENARIOS:
        s = res[(res.scenario == scen) & (res.version == "selected")]
        p = piv.loc[scen]
        d = p["selected"] - p["fixed_baseline"]
        print(f"  {scen}: selected A median {s.A.median():.1f}, rho median "
              f"{s.rho.median():.2f}; picked the truth in "
              f"{int(s.selected_is_true.sum())}/{len(s)}; "
              f"fitted − fixed eval loss median {d.median():+.3f}, "
              f"better in {int((d < 0).sum())}/{len(d)}")
    print(f"artefacts -> {out}/market_recovery_*")
    return 0


def cli() -> int:
    """Run the stage, or regenerate its plots / report from saved outputs.

    python -m strategy_survivorship.market_recovery [OPTIONS]            run the stage
    python -m strategy_survivorship.market_recovery plots | report [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"plots": main_plots, "report": main_report}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
