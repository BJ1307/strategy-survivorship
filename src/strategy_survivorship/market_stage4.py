"""market_stage4: one module per research stage.

- Model and computation: Market stage 4: a retrospective walk-forward replay of the stage-3 calibration.
- Figures: The two stage-4 figures: the year-by-year difference, and the parameter path.
- Report: Market stage 4 report: the corrections, the replay, and what it does not settle.
- Command-line entry: Market stage 4: retrospective walk-forward replay on 2017-2023.
"""
from __future__ import annotations


# ============================================================================
# Model and computation
# ============================================================================

import datetime as dt
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_diagnostics as mdg
from . import market_stage2 as m2
from . import market_stage3 as m3

# ------------------------------------------------------------------ protocol --
DEV_START = dt.date(2017, 1, 1)
DEV_END = dt.date(2023, 12, 31)          # the development scope; nothing later is read
EVAL_YEARS = (2020, 2021, 2022, 2023)
RULES = ("rolling3y", "expanding")
BRANCHES = m3.BRANCHES                   # sv_only, sv_jump
METHODS = ("recalibrated", "fixed_baseline")
GRID_A, GRID_RHO = m3.GRID_A, m3.GRID_RHO      # the stage-3 grid, unchanged
BASELINE = m3.BASELINE                          # A = 1.0, rho = 0.98

SCREEN_PATHS = 400
FINAL_PATHS = 2000
EVAL_PATHS = 2000
N_FINALISTS = 3

CAL_SCREEN_ENTROPY = 20260917
CAL_FINAL_ENTROPY = 20260918
CAL_REFERENCE_ENTROPY = 20260919
EVAL_ENTROPY = 20260920
EVAL_REFERENCE_ENTROPY = 20260921

# auxiliary diagnostics: reported every year, never added to the objective
AUXILIARY = ("skewness", "excess_kurtosis",
             "tail_below_m2_count", "tail_above_p2_count",
             "tail_below_m3_count", "tail_above_p3_count",
             "acf_ret_lag1", "acf_ret_lag5",
             "acf_sqret_lag1", "acf_sqret_lag5",
             "leadlag_ret_vs_future_sq_lag1", "leadlag_ret_vs_future_sq_lag5",
             "concentration_max_exceed_63d")


def _tag(*parts) -> int:
    """A stable integer from the protocol coordinates, never from the data.

    DEFECT FIXED HERE.  The first version used Python's built-in ``hash(str(p))``.
    CPython randomises string hashing per process unless ``PYTHONHASHSEED`` is set,
    and it is not set anywhere in this repository or in the run environment, so that
    version drew a DIFFERENT random stream on every run.  Results were internally
    consistent inside one process -- which is why the time-boundary check still
    passed -- but they did not reproduce across runs.

    blake2b is deterministic across processes, versions and platforms.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") & 0x7FFFFFFF


# --------------------------------------------------------------- provenance --

SOURCE_FILES = ("market_data.py", "market_diagnostics.py", "market_stage2.py",
                "market_stage3.py", "market_stage4.py", "noise.py", "config.py")


def code_identity() -> dict:
    """What actually ran: HEAD plus a fingerprint of every source file used.

    The working tree carries uncommitted work, so HEAD alone does not identify the
    version that produced a result.  Each file is hashed individually.
    """
    here = Path(__file__).resolve().parent
    repo = here.parent.parent

    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    files = {}
    for name in SOURCE_FILES:
        f = here / name
        files[name] = (hashlib.sha256(f.read_bytes()).hexdigest()
                       if f.exists() else "MISSING")
    dirty = _git("status", "--porcelain")
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_is_dirty": bool(dirty),
        "uncommitted_paths": sorted(l[3:] for l in dirty.splitlines()) if dirty else [],
        "source_sha256": files,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pythonhashseed_env": "not set (and no longer relied on)",
    }


def stream_description() -> dict:
    """Exactly how every random stream in this stage is addressed."""
    return {
        "mechanism": "numpy SeedSequence([entropy, tag, ...]); the tag is "
                     "blake2b(coordinates) truncated to 31 bits, deterministic across "
                     "processes",
        "calibration_reference": f"SeedSequence([{CAL_REFERENCE_ENTROPY}, "
                                 f"tag(rule, branch, eval_year)])",
        "calibration_screening": f"SeedSequence([{CAL_SCREEN_ENTROPY}, "
                                 f"tag(rule, branch, eval_year)]) -- ONE stream reused "
                                 f"for every grid cell, which is the common-random-"
                                 f"numbers design",
        "calibration_finalists": f"SeedSequence([{CAL_FINAL_ENTROPY}, "
                                 f"tag(rule, branch, eval_year), candidate_index]) -- "
                                 f"independent of the screening stream",
        "evaluation_reference": f"SeedSequence([{EVAL_REFERENCE_ENTROPY}, "
                                f"tag('evalref', branch, year)])",
        "evaluation": f"SeedSequence([{EVAL_ENTROPY}, tag(label, year, branch)])",
        "never_derived_from": "the return data, the wall clock, or the process",
    }


def window_for(rule: str, eval_year: int) -> tuple[dt.date, dt.date]:
    """The calibration window that ends the day before `eval_year` starts."""
    if rule not in RULES:
        raise ValueError(f"unknown window rule {rule!r}")
    end = dt.date(eval_year - 1, 12, 31)
    start = (dt.date(eval_year - 3, 1, 1) if rule == "rolling3y" else DEV_START)
    if start < DEV_START:
        raise ValueError(f"{rule} at {eval_year} would need data before {DEV_START}")
    return start, end


def year_bounds(year: int) -> tuple[dt.date, dt.date]:
    if year > DEV_END.year:
        raise ValueError(f"{year} is outside the development scope; it is not read")
    return dt.date(year, 1, 1), dt.date(year, 12, 31)


def data_up_to(returns: pd.DataFrame, origin: dt.date) -> pd.DataFrame:
    """Everything on or before `origin`, and nothing after it.

    The calibration path only ever receives the output of this function, which is
    what makes the time boundary testable: perturbing a return after `origin`
    cannot reach anything downstream.
    """
    out = returns[returns["date"] <= pd.Timestamp(origin)]
    return out.sort_values("date", ignore_index=True)


def scale_from(window_returns: np.ndarray) -> float:
    """sigma_annual estimated from the window, never from the evaluation year."""
    return float(mdg.moment_summary(window_returns)["sd_ddof1"]) * \
        math.sqrt(mdg.DAYS_PER_YEAR)


# ------------------------------------------------------- standardising scales --


def reference_scales(cfg, branch: str, sigma_annual: float, n_days: int,
                     entropy: int, tag: int, n_paths: int = FINAL_PATHS
                     ) -> tuple[dict, list[str]]:
    """Across-path spread of each target under the BASELINE at this origin.

    Built from information available at the origin only: the baseline parameters
    are fixed by protocol, the scale comes from the window, and the length is a
    calendar fact. The stage-2 full-training reference is deliberately NOT reused
    here -- it contains information from after the early origins.
    """
    c = m3.branch_cfg(cfg, branch, BASELINE[0], BASELINE[1], sigma_annual)
    r = m3.simulate(c, n_paths, n_days, np.random.SeedSequence([entropy, tag]))
    st = m2.path_statistics_chunked(r)
    return m3.standardising_scales({k: st[k] for k in
                                    m3.RV_TARGETS + m3.ACF_TARGETS})


# ---------------------------------------------------------------- calibration --


def calibrate_at_origin(cfg, returns: pd.DataFrame, rule: str, branch: str,
                        eval_year: int, screen_paths: int = SCREEN_PATHS,
                        final_paths: int = FINAL_PATHS,
                        n_finalists: int = N_FINALISTS) -> dict:
    """One calibration, using only data on or before the window end.

    Everything that could leak is derived here and nowhere else: the window, the
    scale, the standardising reference and the seeds. The seeds come from the
    protocol coordinates (rule, branch, year), never from the data, so the result
    is reproducible and the time-boundary check is meaningful.
    """
    start, end = window_for(rule, eval_year)
    hist = data_up_to(returns, end)
    win = mdg.period_slice(hist, start, end)
    if win.empty:
        raise ValueError(f"no data in the {rule} window for {eval_year}")
    x = win["ret"].to_numpy()
    n_days = len(x)
    sigma_annual = scale_from(x)
    real_t = m3.real_targets({k: float(v[0]) for k, v in
                              m2.path_statistics(x[None, :]).items()})
    tag = _tag(rule, branch, eval_year)
    scales, floored = reference_scales(cfg, branch, sigma_annual, n_days,
                                       CAL_REFERENCE_ENTROPY, tag)

    # screening: common random numbers across every cell in this calibration
    screen_seed = np.random.SeedSequence([CAL_SCREEN_ENTROPY, tag])
    rows = []
    for A in GRID_A:
        for rho in GRID_RHO:
            c = m3.branch_cfg(cfg, branch, A, rho, sigma_annual)
            st = m2.path_statistics_chunked(
                m3.simulate(c, screen_paths, n_days, screen_seed))
            L = m3.loss(m3.target_values(st), real_t, scales)
            rows.append({"A": A, "rho": rho, "loss": L["loss"],
                         "rv21_component": L["rv21_component"],
                         "acf_abs_component": L["acf_abs_component"]})
    grid = pd.DataFrame(rows)

    # finalists on fresh, independent streams
    finals = []
    cand = grid.sort_values("loss").head(n_finalists)[["A", "rho"]].values.tolist()
    if [BASELINE[0], BASELINE[1]] not in cand:
        cand.append([BASELINE[0], BASELINE[1]])
    for i, (A, rho) in enumerate(cand):
        c = m3.branch_cfg(cfg, branch, A, rho, sigma_annual)
        st = m2.path_statistics_chunked(
            m3.simulate(c, final_paths, n_days,
                        np.random.SeedSequence([CAL_FINAL_ENTROPY, tag, i])))
        L = m3.loss(m3.target_values(st), real_t, scales)
        finals.append({"A": A, "rho": rho, "loss": L["loss"],
                       "loss_mc_se": m3.loss_mc_se(st, real_t, scales),
                       "rv21_component": L["rv21_component"],
                       "acf_abs_component": L["acf_abs_component"],
                       "is_baseline": A == BASELINE[0] and rho == BASELINE[1]})
    fin = pd.DataFrame(finals).sort_values("loss", ignore_index=True)
    best = fin.iloc[0]
    near = fin[fin["loss"] <= best["loss"] + best["loss_mc_se"]]
    return {
        "rule": rule, "branch": branch, "eval_year": eval_year,
        "window_start": str(start), "window_end": str(end),
        "window_n_returns": n_days, "sigma_annual": sigma_annual,
        "A": float(best.A), "rho": float(best.rho),
        "calibration_loss": float(best.loss),
        "calibration_loss_mc_se": float(best.loss_mc_se),
        "n_within_one_mc_se": int(len(near)),
        "on_grid_boundary": bool(best.A in (GRID_A[0], GRID_A[-1]) or
                                 best.rho in (GRID_RHO[0], GRID_RHO[-1])),
        "standardising_scales": scales, "scales_floored": floored,
        "grid": grid, "finalists": fin, "seed_tag": tag,
    }


# ----------------------------------------------------------------- evaluation --


def evaluate_year(cfg, returns: pd.DataFrame, year: int, branch: str,
                  A: float, rho: float, sigma_annual: float, eval_scales: dict,
                  label: str, n_paths: int = EVAL_PATHS) -> dict:
    """Score one frozen configuration on one evaluation year.

    Every statistic is computed INSIDE the year, on both sides. RV_21 therefore
    starts on the 21st trading day of the year and the market gets exactly the
    same treatment as the simulation -- no statistic reaches across the boundary,
    and no embargo is imposed because none is needed.
    """
    lo, hi = year_bounds(year)
    ys = mdg.period_slice(returns, lo, hi)
    xv = ys["ret"].to_numpy()
    n_days = len(xv)
    real = m2.path_statistics(xv[None, :])
    real_t = m3.real_targets({k: float(v[0]) for k, v in real.items()})
    c = m3.branch_cfg(cfg, branch, A, rho, sigma_annual)
    st = m2.path_statistics_chunked(
        m3.simulate(c, n_paths, n_days,
                    np.random.SeedSequence([EVAL_ENTROPY, _tag(label, year, branch)])))
    L = m3.loss(m3.target_values(st), real_t, eval_scales)
    return {"year": year, "branch": branch, "label": label, "A": A, "rho": rho,
            "sigma_annual": sigma_annual, "eval_n_days": n_days,
            "loss": L["loss"], "rv21_component": L["rv21_component"],
            "acf_abs_component": L["acf_abs_component"],
            "loss_mc_se": m3.loss_mc_se(st, real_t, eval_scales),
            "terms": L["terms"], "sim_stats": st, "real_stats": real}


def evaluation_scales(cfg, returns: pd.DataFrame, year: int, branch: str,
                      n_paths: int = FINAL_PATHS) -> tuple[dict, list[str], float, int]:
    """One scoring reference per (year, branch), shared by EVERY method that year.

    Fixed by protocol before any evaluation number exists: the baseline parameters,
    the scale from the EXPANDING window that ends before the year, and a length
    equal to the year's trading-day count -- a calendar fact, not a return. Using
    one reference for the whole year is what makes the methods comparable; using
    the expanding scale is what makes it constructible under either window rule.
    """
    start, end = window_for("expanding", year)
    win = mdg.period_slice(data_up_to(returns, end), start, end)
    sigma_annual = scale_from(win["ret"].to_numpy())
    lo, hi = year_bounds(year)
    n_days = int(len(mdg.period_slice(returns, lo, hi)))
    tag = _tag("evalref", branch, year)
    scales, floored = reference_scales(cfg, branch, sigma_annual, n_days,
                                       EVAL_REFERENCE_ENTROPY, tag, n_paths)
    return scales, floored, sigma_annual, n_days

# ============================================================================
# Figures
# ============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELL = {("rolling3y", "sv_only"): ("#1f77b4", "o", "rolling 3y, pure SV"),
        ("rolling3y", "sv_jump"): ("#ff7f0e", "s", "rolling 3y, SV + jumps"),
        ("expanding", "sv_only"): ("#2ca02c", "^", "expanding, pure SV"),
        ("expanding", "sv_jump"): ("#d62728", "v", "expanding, SV + jumps")}


def year_differences(res: pd.DataFrame, out_path: Path) -> Path:
    res = res.copy()
    res["se"] = np.sqrt(res.loss_mc_se_recalibrated ** 2 +
                        res.loss_mc_se_fixed_baseline ** 2)
    years = sorted(res.eval_year.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    fig.suptitle("Re-estimating A and rho before each year, minus the matched fixed "
                 "baseline\nNegative means the re-estimate described that year "
                 "better. Retrospective replay on 2017-2023.", fontsize=12.5, y=0.99)

    ax = axes[0]
    w = 0.19
    for i, (key, (col, mk, lab)) in enumerate(CELL.items()):
        sub = res[(res.rule == key[0]) & (res.branch == key[1])].set_index("eval_year")
        x = np.arange(len(years)) + (i - 1.5) * w
        y = [sub.loc[t, "loss_difference"] for t in years]
        e = [sub.loc[t, "se"] for t in years]
        ax.bar(x, y, width=w * 0.92, color=col, alpha=0.85, label=lab)
        ax.errorbar(x, y, yerr=np.array(e) * 2, fmt="none", ecolor="0.25", lw=1.1,
                    capsize=2.5)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(t) for t in years])
    ax.set_xlabel("evaluation year")
    ax.set_ylabel("loss difference\n(re-estimated − fixed baseline)")
    ax.set_title("(a) year by year; bars below zero favour re-estimating", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper left")

    ax = axes[1]
    allm = res.loss_difference.mean()
    drops = [("all 4 years", allm)] + [(f"without {t}",
                                        res[res.eval_year != t].loss_difference.mean())
                                       for t in years]
    cols = ["0.35"] + ["#d62728" if abs(v) < 0.1 else "0.6" for _, v in drops[1:]]
    ax.barh(range(len(drops)), [v for _, v in drops], color=cols, height=0.62)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(range(len(drops)))
    ax.set_yticklabels([k for k, _ in drops], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean loss difference over the remaining cells")
    ax.set_title("(b) does the average rest on one year?", fontsize=11.5)
    for i, (_, v) in enumerate(drops):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)

    for a in axes:
        a.grid(alpha=0.3, axis="x" if a is axes[1] else "y")
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.040,
             "Error bars are +-2 combined Monte-Carlo standard errors: simulation "
             "precision only, not sampling error for the market.",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.008,
             "Four years and two overlapping window rules are not independent "
             "replicates, and no significance statement is made from them.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.925))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parameter_path(res: pd.DataFrame, out_path: Path, baseline=(1.0, 0.98)) -> Path:
    years = sorted(res.eval_year.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    fig.suptitle("What the re-estimate chose, before each year "
                 "(the grid is unchanged from stage 3)", fontsize=12.5, y=0.98)
    for ax, param, base, lab in ((axes[0], "A", baseline[0],
                                  "A  (volatility swing amplitude)"),
                                 (axes[1], "rho", baseline[1],
                                  "rho  (latent persistence)")):
        for key, (col, mk, name) in CELL.items():
            sub = res[(res.rule == key[0]) &
                      (res.branch == key[1])].set_index("eval_year")
            ax.plot(years, [sub.loc[t, param] for t in years], marker=mk, ms=7,
                    lw=1.6, color=col, label=name, alpha=0.9)
        ax.axhline(base, color="black", ls="--", lw=1.3)
        ax.annotate(f"fixed baseline {base:g}", (years[0], base), xytext=(4, 5),
                    textcoords="offset points", fontsize=9)
        ax.set_xticks(years)
        ax.set_xlabel("evaluation year")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    fig.text(0.5, 0.012,
             "Inside this development scope the rolling window never actually drops "
             "2020: the 2023 window is 2020-2022. The two rules are therefore far "
             "more alike here than their definitions suggest.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.925))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

# ============================================================================
# Report
# ============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NICE = {"sv_only": "pure SV", "sv_jump": "SV + jumps",
        "rolling3y": "rolling 3y", "expanding": "expanding"}


def build(out: Path) -> str:
    s = json.loads((out / "market_stage4_summary.json").read_text(encoding="utf-8"))
    res = pd.read_csv(out / "market_stage4_results.csv")
    ev = pd.read_csv(out / "market_stage4_evaluations.csv")
    aux = pd.read_csv(out / "market_stage4_auxiliary.csv")
    corr, proto = s["stage3_corrections"], s["protocol"]
    res["se"] = np.sqrt(res.loss_mc_se_recalibrated ** 2 +
                        res.loss_mc_se_fixed_baseline ** 2)
    res["beyond_mc_noise"] = res.loss_difference.abs() > 2 * res.se
    res["d_rv"] = res.rv21_component_recalibrated - res.rv21_component_fixed_baseline
    res["d_acf"] = res.acf_abs_component_recalibrated - res.acf_abs_component_fixed_baseline
    years = sorted(res.eval_year.unique())

    L: list[str] = []
    A = L.append
    A("# Market stage 4 — a retrospective walk-forward replay\n")
    A(f"Does re-estimating `A` and `rho` before each year describe the following year "
      f"better than leaving them fixed? Development scope "
      f"{proto['development_scope'][0]} to {proto['development_scope'][1]}, "
      f"{s['development_scope_returns']:,} returns. {proto['holdout']}\n")
    A(f"**{proto['status']}**\n")

    # ------------------------------------------------- the reproducibility fix ---
    fx_path = out / "market_stage4_seedfix_record.json"
    if fx_path.exists():
        fx = json.loads(fx_path.read_text())
        A("\n## 0. A reproducibility defect in this stage, found and fixed\n")
        A(f"**The defect.** {fx['defect']}\n")
        A(f"**The fix.** {fx['fix']} {fx['verification']}\n")
        A(f"**The re-run.** {fx['rerun']}\n")
        mv, nm = fx["what_moved"], fx["what_did_not_move"]
        A(f"Of {mv['cells_total']} cells, **{mv['cells_whose_selected_parameters_moved']}** "
          f"changed their selected parameters and **{mv['cells_whose_sign_flipped']}** "
          f"changed sign:\n")
        A("| year | rule | branch | A before | A after |")
        A("|---|---|---|---|---|")
        for r in mv["which"]:
            A(f"| {r['eval_year']} | {r['rule']} | {r['branch']} | "
              f"{r['A_prefix']:g} | {r['A_postfix']:g} |")
        A("")
        A("| year | rule | branch | difference before | after |")
        A("|---|---|---|---|---|")
        for r in mv["sign_flips"]:
            A(f"| {r['eval_year']} | {r['rule']} | {r['branch']} | "
              f"{r['loss_difference_prefix']:+.3f} | {r['loss_difference_postfix']:+.3f} |")
        A("")
        A("| headline quantity | before the fix | after |")
        A("|---|---|---|")
        A(f"| cells won by re-estimating | {nm['cells_won_overall'][0]} of 16 | "
          f"{nm['cells_won_overall'][1]} of 16 |")
        A(f"| mean loss difference | {nm['mean_loss_difference'][0]:+.3f} | "
          f"{nm['mean_loss_difference'][1]:+.3f} |")
        A(f"| mean without 2022 | {nm['mean_without_2022'][0]:+.3f} | "
          f"{nm['mean_without_2022'][1]:+.3f} |")
        A(f"| 2022 mean | {nm['2022_mean'][0]:+.3f} | {nm['2022_mean'][1]:+.3f} |")
        A(f"| 2020 cells won | {nm['2020_cells_won_by_recalibration'][0]} of 4 | "
          f"{nm['2020_cells_won_by_recalibration'][1]} of 4 |")
        A("")
        A(f"{nm['reading']}\n")
        ci = s.get("code_identity", {})
        if ci:
            A(f"**What ran.** HEAD `{ci['git_head'][:12]}` on `{ci['git_branch']}`, "
              f"working tree dirty: `{ci['working_tree_is_dirty']}` with "
              f"{len(ci['uncommitted_paths'])} uncommitted paths — so HEAD alone does "
              f"not identify this run. Every source file used is fingerprinted in "
              f"`market_stage4_summary.json` (`market_stage4.py` "
              f"`{ci['source_sha256']['market_stage4.py'][:16]}…`), together with the "
              f"full random-stream description, the scoring weights actually used "
              f"(`market_stage4_calibration_weights.csv`, "
              f"`market_stage4_evaluation_weights.csv`), every screened cell "
              f"(`market_stage4_grids.csv`) and every finalist "
              f"(`market_stage4_finalists.csv`).\n")

    # ------------------------------------------------------- corrections ----
    A("\n## 1. Corrections to the stage-3 record\n")
    A(f"{corr['provenance']}\n")
    o = corr["1_order_of_operations"]
    A(f"**(a) Order of operations.** Stage 3 implied {o['what_stage_3_implied']}. "
      f"In fact {o['what_actually_happened']} {o['status_now']} "
      f"{o['not_affected']}\n")
    a2 = corr["2_attribution"]
    A(f"**(b) Attribution withdrawn.** \"{a2['withdrawn']}\" is withdrawn. "
      f"{a2['replacement']}\n")
    a3 = corr["3_fitted_versus_held_out"]
    A(f"**(c) Fitted versus held out.** The objective is exactly "
      f"`{'`, `'.join(a3['fitted_targets'])}`. {a3['note']}. "
      f"{a3['boundary_crossing_is_not_enough']}\n")
    A("Re-checking the centre, scaled by the baseline's own 2.5–97.5% width:\n")
    A("| branch | held-out statistics checked | centre error worse | better | essentially unchanged |")
    A("|---|---|---|---|---|")
    for b, d in a3["centre_error_check"].items():
        A(f"| {NICE[b]} | {d['n_held_out_checked']} | "
          f"**{len(d['centre_error_worse'])}** | {d['centre_error_better']} | "
          f"{d['essentially_unchanged']} |")
    A("")
    A("The statistics whose centre moved away from the sample:\n")
    A("| branch | statistic | baseline centre error | calibrated | change |")
    A("|---|---|---|---|---|")
    for b, d in a3["centre_error_check"].items():
        for r in d["centre_error_worse"]:
            A(f"| {NICE[b]} | `{r['statistic']}` | {r['baseline_centre_error']:.3f} | "
              f"{r['selected_centre_error']:.3f} | {r['change']:+.3f} |")
    A("")
    A("So stage 3's \"nothing broke\" was too generous: two statistics for pure SV and "
      "four for SV+jumps had their centre drift away while staying inside a band that "
      "had itself widened. `tail_above_p3_count` is the clearest case — the baseline "
      "matched it exactly and the calibrated version does not.\n")
    a4 = corr["4_uncertainty_and_scope"]
    A(f"**(d) Scope and uncertainty.** {a4['structural_claims_restricted']}\n")
    A("Three different uncertainties, only one of which is quantified anywhere in "
      "stages 3 and 4:\n")
    for k, v in a4["three_uncertainties_are_different"].items():
        A(f"- **{k.replace('_', ' ')}** — {v}")
    A("")

    # ---------------------------------------------------------- protocol ----
    A("\n## 2. The replay protocol, frozen before the run\n")
    A(f"- Evaluation years {proto['evaluation_years']}; "
      f"{proto['cadence']}")
    A(f"- Window rules: **rolling 3y** — {proto['window_rules']['rolling3y']}; "
      f"**expanding** — {proto['window_rules']['expanding']}. "
      f"{proto['window_rules']['note']}")
    A(f"- Grid: {proto['grid']['note']}")
    A(f"- Comparison: {proto['comparison']['recalibrated']} against "
      f"{proto['comparison']['fixed_baseline']}. {proto['comparison']['shared']}")
    A(f"- Objective: {proto['objective']}")
    A(f"- Calibration reference: {proto['standardising_references']['calibration']}")
    A(f"- Evaluation reference: {proto['standardising_references']['evaluation']}")
    A(f"- {proto['statistics_inside_the_year']}")
    A(f"- {proto['initialisation']}")
    A(f"- Auxiliary diagnostics ({len(proto['auxiliary']['list'])} of them) are "
      f"{proto['auxiliary']['rule']}")
    A("")
    tb = s["time_boundary_checks"][0]
    A(f"**Time-boundary check.** Multiplying every return after the "
      f"{tb['rule']} origin for {tb['eval_year']} by −3 and adding 0.05 — "
      f"{tb['n_returns_perturbed']} returns — left the chosen `A`, `rho`, the scale, "
      f"the window size, the calibration loss, the seed and the standardising scales "
      f"**identical**: `{tb['identical']}`. The calibration only ever sees "
      f"`data_up_to(origin)`.\n")

    # ------------------------------------------------------------ results ---
    A("\n## 3. Results\n")
    A("![year differences](market_stage4_year_differences.png)\n")
    A("Full table — every year, both window rules, both branches. A negative "
      "difference means the re-estimate described that year better.\n")
    A("| year | rule | branch | window | A | rho | re-estimated loss | fixed baseline | difference | ±2 MC s.e. | beyond MC noise |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res.sort_values(["eval_year", "branch", "rule"]).iterrows():
        A(f"| {r.eval_year} | {NICE[r.rule]} | {NICE[r.branch]} | "
          f"{r.window_start[:4]}–{r.window_end[:4]} | {r.A:g} | {r.rho:g} | "
          f"{r.loss_recalibrated:.3f} | {r.loss_fixed_baseline:.3f} | "
          f"**{r.loss_difference:+.3f}** | {2*r.se:.3f} | {r.beyond_mc_noise} |")
    A("")
    helped = int(res.recalibration_helped.sum())
    A(f"Re-estimating beat its matched baseline in **{helped} of {len(res)}** "
      f"year-rule-branch cells. Mean difference **{res.loss_difference.mean():+.3f}** "
      f"(positive = re-estimating worse).\n")

    A("### Per year\n")
    A("| year | mean difference | best cell | worst cell | cells where re-estimating won |")
    A("|---|---|---|---|---|")
    for y in years:
        sub = res[res.eval_year == y]
        A(f"| {y} | {sub.loss_difference.mean():+.3f} | "
          f"{sub.loss_difference.min():+.3f} | {sub.loss_difference.max():+.3f} | "
          f"{int(sub.recalibration_helped.sum())} of {len(sub)} |")
    A("")
    A("### Does the average rest on one year?\n")
    A("**Yes — almost entirely on 2022.**\n")
    A("| cells included | mean difference |")
    A("|---|---|")
    A(f"| all four years | {res.loss_difference.mean():+.3f} |")
    for y in years:
        A(f"| without {y} | {res[res.eval_year != y].loss_difference.mean():+.3f} |")
    A("")
    w22 = res[res.eval_year == 2022]
    A(f"Drop 2022 and the average is {res[res.eval_year != 2022].loss_difference.mean():+.3f} "
      f"— a wash. 2022 alone contributes a mean of {w22.loss_difference.mean():+.3f}, "
      f"and it is the only year in which every cell loses by more than "
      f"{w22.loss_difference.min():.1f}. Any summary that quotes the four-year average "
      f"without saying this is misleading.\n")
    A("### Which component moves\n")
    A("| year | volatility-distribution component | clustering component |")
    A("|---|---|---|")
    for y in years:
        sub = res[res.eval_year == y]
        A(f"| {y} | {sub.d_rv.mean():+.3f} | {sub.d_acf.mean():+.3f} |")
    A("")
    A("In 2020 the gain comes from the clustering group; the volatility distribution "
      "gets slightly worse. In 2022 both deteriorate and the volatility distribution "
      "dominates.\n")

    A("### Parameters chosen\n")
    A("![parameter path](market_stage4_parameter_path.png)\n")
    A("| year | rolling 3y, pure SV | rolling 3y, SV+jumps | expanding, pure SV | expanding, SV+jumps |")
    A("|---|---|---|---|---|")
    for y in years:
        cells = []
        for rule in ("rolling3y", "expanding"):
            for b in ("sv_only", "sv_jump"):
                r = res[(res.eval_year == y) & (res.rule == rule) &
                        (res.branch == b)].iloc[0]
                cells.append(f"A={r.A:g}, rho={r.rho:g}")
        A(f"| {y} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    A("")
    A(f"Every selected cell lies inside the grid "
      f"({'no' if not res.on_grid_boundary.any() else 'some'} boundary solutions). "
      f"`A` rises from 1.2 before 2020 to 1.6 before 2021 and settles at 1.4; `rho` "
      f"rises from 0.94–0.96 to 0.98 and stays. Every re-estimate sits above the "
      f"baseline `A = 1.0`.\n")

    A("### Rolling against expanding\n")
    A("| rule | branch | mean difference | cells won |")
    A("|---|---|---|---|")
    for (rule, b), g in res.groupby(["rule", "branch"]):
        A(f"| {NICE[rule]} | {NICE[b]} | {g.loss_difference.mean():+.3f} | "
          f"{int(g.recalibration_helped.sum())} of {len(g)} |")
    A("")
    A("**The two rules barely differ, and this replay cannot separate them.** Inside "
      "2017-2023 the rolling window never actually drops 2020: the window before 2023 "
      "is 2020-2022. The first rolling window that excludes 2020 would be the one "
      "before 2024, which is in the holdout and is not read. The near-identical "
      "parameter paths are a consequence of the scope, not evidence that window length "
      "does not matter.\n")

    A("### Auxiliary diagnostics\n")
    A("Thirteen statistics were tracked every year and **none was added to the "
      "objective**. Counting how often the sample value falls inside the simulated "
      "2.5–97.5% range, across all years and cells:\n")
    aux["inside"] = (aux["real"] >= aux["sim_p2.5"]) & (aux["real"] <= aux["sim_p97.5"])
    tab = aux.pivot_table(index="statistic", columns="method", values="inside",
                          aggfunc="mean")
    A("| statistic | re-estimated inside | fixed baseline inside |")
    A("|---|---|---|")
    for k in AUXILIARY:
        if k in tab.index:
            A(f"| `{k}` | {tab.loc[k, 'recalibrated']:.0%} | "
              f"{tab.loc[k, 'fixed_baseline']:.0%} |")
    A("")
    A("These are coverage counts over overlapping cells, not a score and not a test.\n")

    # ------------------------------------------------------- what it means --
    A("\n## 4. What this does and does not settle\n")
    A("**What the replay shows.** On this objective, over these four years, "
      "re-estimating `A` and `rho` before each year did **not** describe the following "
      "year better on average. It helped in 2020, was roughly neutral to mildly worse "
      "in 2021 and 2023, and was clearly worse in 2022. Fifteen of the sixteen "
      "differences are larger than twice the combined simulation error, so they are "
      "not Monte-Carlo artefacts.\n")
    A("**What it does not show.**\n")
    A("- The shared scoring reference is shared by the two METHODS inside one "
      "(year, branch) cell. It is **not** shared across branches, so a loss LEVEL for "
      "pure SV and one for SV+jumps are on different scales and must not be ranked "
      "against each other; only the within-cell difference is meaningful.")
    A("- A Monte-Carlo standard error here is the precision of the evaluation "
      "computation **given** the parameters and the reference weights. It contains no "
      "market sampling uncertainty and no parameter-estimation uncertainty.")
    for x in proto["reporting_limits"]:
        A(f"- {x}")
    A("- 2020 is the one year where the calibration window contained no extreme "
      "episode and the evaluation year did; 2022 is the reverse. The replay cannot "
      "tell whether that ordering is the mechanism or a coincidence of four draws.")
    A("- No failure here is attributed to a regime change. That would be a claim about "
      "the data-generating process, and this design cannot separate it from "
      "finite-sample variation, parameter uncertainty or model misspecification.")
    A("- The main window rule is **not** being swapped on the strength of this replay, "
      "and the grid was not widened when results were poor.")
    A("")
    A("**If annual re-estimation is ever adopted.** Freeze the *algorithm* and its "
      "hyper-parameters — grid, objective, window rule, budget, selection rule — and "
      "let the *parameters* update each year under that frozen rule. \"Re-estimate "
      "annually\" and \"do not re-tune between years\" are not in conflict: the first "
      "is the parameters moving under a fixed procedure, the second is the procedure "
      "itself staying untouched once evaluation begins.\n")
    A("**A point that matters for the wider project.** A generator that re-estimates "
      "well would still say nothing about whether the monitor's two-year 15% "
      "cumulative false-alarm control survives. If a periodic parameter update were "
      "ever used inside a live monitor, the whole rule *including the update action* "
      "would have to be recalibrated as one procedure. A monitor does not get to draw "
      "a fresh 15% budget each time its generator is updated.\n")

    A("\n## 5. Reproducing\n")
    A("```bash")
    A(".venv/bin/python -m strategy_survivorship.market_stage4")
    A(".venv/bin/python -m strategy_survivorship.market_stage4 report")
    A(".venv/bin/python -m pytest tests/test_market_stage4.py -q")
    A("```")
    A("")
    A("Stage 3's outputs are unchanged; the corrections to its write-up live in "
      "`market_stage4_stage3_corrections.json` and in section 1 above. The monitors, "
      "the EWMA code, `mixed_noise` and every earlier simulation result are untouched, "
      "no new model family was added, and nothing was committed or pushed.")
    return "\n".join(L) + "\n"


def main_report(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 4 report.")
    ap.add_argument("--out", type=Path, default=Path("outputs/market"))
    args = ap.parse_args(argv)
    p = args.out / "market_stage4_report.md"
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

from .config import DEFAULT

IN = OUT = Path("outputs/market")


# --------------------------------------------------------------------------- #
# step 1: corrections to the stage-3 record
# --------------------------------------------------------------------------- #


def stage3_corrections(out: Path) -> dict:
    """Four corrections, recomputed from stage 3's own artefacts, which stay as they are."""
    tr = pd.read_csv(out / "market_stage3_training_comparison.csv")
    targets = list(m3.RV_TARGETS) + list(m3.ACF_TARGETS)
    centre = {}
    for branch in m3.BRANCHES:
        b = tr[tr.config.str.startswith("baseline") &
               tr.config.str.contains(branch)].set_index("statistic")
        c = tr[tr.config.str.startswith("selected") &
               tr.config.str.contains(branch)].set_index("statistic")
        rows = []
        for k in [k for k in b.index if k in c.index and k not in targets]:
            real = b.loc[k, "real"]
            width = b.loc[k, "sim_p97.5"] - b.loc[k, "sim_p2.5"]
            if not (np.isfinite(real) and np.isfinite(width) and width > 0):
                continue
            eb = abs(b.loc[k, "sim_median"] - real) / width
            ec = abs(c.loc[k, "sim_median"] - real) / width
            rows.append({"statistic": k, "baseline_centre_error": eb,
                         "selected_centre_error": ec, "change": ec - eb})
        d = pd.DataFrame(rows)
        centre[branch] = {
            "n_held_out_checked": int(len(d)),
            "centre_error_worse": d[d.change > 0.02].sort_values(
                "change", ascending=False).to_dict("records"),
            "centre_error_better": int((d.change < -0.02).sum()),
            "essentially_unchanged": int(((d.change >= -0.02) &
                                          (d.change <= 0.02)).sum()),
        }
    return {
        "provenance": "recomputed from outputs/market/market_stage3_*, which are "
                      "unchanged. Stage 3's numbers all stand; what is corrected is "
                      "how they were described.",
        "1_order_of_operations": {
            "what_stage_3_implied": "that both validation loss definitions were fixed "
                                    "before any validation number existed",
            "what_actually_happened": "the smoke run of run_market_stage3 executed the "
                                      "WHOLE pipeline, including the validation period, "
                                      "and printed validation losses. The "
                                      "validation-length standardisation was added "
                                      "AFTER that output had been seen, and the code "
                                      "comment claiming otherwise was wrong.",
            "status_now": "loss_frozen_training_scales remains the pre-specified "
                          "primary result. loss_validation_length_scales is a "
                          "POST-HOC SUPPLEMENTARY analysis. Its agreement with the "
                          "primary ordering is therefore weaker evidence than stage 3 "
                          "presented it as, and it selects nothing.",
            "not_affected": "the grid, the objective, the selection rule and the "
                            "selected parameters were all fixed before the search and "
                            "are untouched by this correction.",
        },
        "2_attribution": {
            "withdrawn": "the cause lies in the market itself, not in the fitting "
                         "process",
            "replacement": "a transfer failure across periods was observed. This round "
                           "cannot separate the contributions of a change in the "
                           "data-generating mechanism, finite-sample variation in "
                           "either period, uncertainty in the estimated parameters, "
                           "and model misspecification. The market statistics do "
                           "differ sharply between the two periods; that is an "
                           "observation, not an attribution.",
        },
        "3_fitted_versus_held_out": {
            "fitted_targets": targets,
            "note": "the RV_21 quantiles and the absolute-return ACF are the objective "
                    "itself and must never be cited as held-out evidence",
            "boundary_crossing_is_not_enough": "stage 3 reported that no held-out "
                                               "statistic moved from inside the "
                                               "simulated range to outside. That test "
                                               "cannot see a centre that drifted while "
                                               "staying inside a wide band, so the "
                                               "centre error is checked here as well, "
                                               "scaled by the baseline's own band.",
            "centre_error_check": centre,
        },
        "4_uncertainty_and_scope": {
            "structural_claims_restricted": "stage 3's statement that the two target "
                                            "groups cannot be satisfied together holds "
                                            "FOR THIS GRID AND THIS OBJECTIVE. A "
                                            "different target set, a different "
                                            "weighting or a wider grid could change it. "
                                            "It is not a proof about the model family.",
            "three_uncertainties_are_different": {
                "simulation_standard_error": "how much a reported loss or median moves "
                                             "if the same model is simulated again. "
                                             "Reported as loss_mc_se, and it is the "
                                             "only one stage 3 quantified.",
                "market_sample_uncertainty": "how much the SAMPLE statistic would move "
                                             "on another draw of the same length from "
                                             "the same market. NOT quantified: one "
                                             "history, no replicate.",
                "parameter_uncertainty": "how much the selected (A, rho) would move on "
                                         "another sample. NOT quantified. Every "
                                         "simulated range in stages 3 and 4 is "
                                         "conditional on the selected parameters and "
                                         "contains none of it.",
            },
        },
    }


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #


def protocol(screen_paths: int, final_paths: int, eval_paths: int) -> dict:
    return {
        "question": "does re-estimating A and rho before each year describe the "
                    "following year's statistics better than leaving them fixed?",
        "status": "RETROSPECTIVE replay on history that has already been examined. "
                  "Re-running a calibration on data we have already studied is not "
                  "the same as having run it live, and nothing here is evidence about "
                  "a future year.",
        "development_scope": [str(DEV_START), str(DEV_END)],
        "holdout": "2024-2025 takes no part in any statistic, figure or choice. To be "
                   "exact about what happens: the CSV is read whole, then truncated at "
                   "the development end by data_up_to() BEFORE anything is computed, so "
                   "those rows are loaded into memory and discarded unexamined rather "
                   "than never touched by the file reader. A functional check corrupts "
                   "every holdout return and confirms the development frame and a full "
                   "calibration are unchanged.",
        "evaluation_years": list(EVAL_YEARS),
        "window_rules": {
            "rolling3y": "the three calendar years before the evaluation year",
            "expanding": "2017 through the year before the evaluation year",
            "note": "for 2020 the two rules coincide (2017-2019), so that year "
                    "carries no information about the choice between them",
        },
        "cadence": "calibrate once before each year, frozen inside the year. Neither "
                   "the window length nor the update frequency is searched.",
        "branches": {b: ("pure SV, kappa = 0" if b == "sv_only"
                         else "SV + jumps at the recorded kappa and lambda")
                     for b in BRANCHES},
        "grid": {"A": list(GRID_A), "rho": list(GRID_RHO),
                 "note": "the stage-3 grid, unchanged. It is NOT widened if a result "
                         "is poor."},
        "comparison": {
            "recalibrated": "A and rho re-estimated on the window",
            "fixed_baseline": f"A = {BASELINE[0]}, rho = {BASELINE[1]}",
            "shared": "both use the SAME overall scale, estimated from that window. "
                      "The scale is never derived from the evaluation year.",
        },
        "objective": "the stage-3 objective, unchanged: RV_21 quantiles compared in "
                     "logs and the centred-absolute-return ACF at lags "
                     f"{list(m3.LAGS) if hasattr(m3, 'LAGS') else [1, 5, 10, 21, 63]}, "
                     "two groups of equal total weight. A finite-sample "
                     "simulated-moment distance, not a likelihood and not a test.",
        "lengths": {
            "calibration": "the simulated length matches the history window",
            "evaluation": "the simulated length matches the evaluation year",
        },
        "standardising_references": {
            "calibration": "built at EACH origin from then-available information: the "
                           "baseline parameters, the window's own scale estimate and "
                           "the window's length. Fixed across candidates inside that "
                           "calibration. The stage-2 full-training reference is NOT "
                           "carried into early windows -- it contains information from "
                           "after them.",
            "evaluation": "ONE reference per (year, branch): baseline parameters, the "
                          "scale from the EXPANDING window ending before the year, and "
                          "a length equal to the year's trading-day count, which is a "
                          "calendar fact.",
            "what_the_shared_reference_does_and_does_not_cover":
                "it is shared by the two METHODS (re-estimated and fixed baseline) "
                "inside one (year, branch) cell, which is what makes the difference "
                "between them meaningful. It is NOT shared across branches: pure SV "
                "and SV+jumps each get their own weights, so a loss LEVEL from one "
                "branch is not on the same scale as a loss level from the other and "
                "the two must not be ranked against each other.",
        },
        "statistics_inside_the_year": "every evaluation statistic is computed inside "
                                      "the evaluation year, identically for the market "
                                      "and for every simulated path. RV_21 therefore "
                                      "begins on the 21st trading day of the year. No "
                                      "statistic crosses the boundary, so no embargo "
                                      "is imposed.",
        "initialisation": "the log-variance AR(1) starts from its stationary law. This "
                          "is a check on whether the distribution SHAPE transfers after "
                          "a rolling re-fit; it is not a conditional forecast and the "
                          "latent state is never filtered on the real path.",
        "budgets_and_seeds": {
            "screen_paths": screen_paths, "final_paths": final_paths,
            "eval_paths": eval_paths, "n_finalists": N_FINALISTS,
            "screening": "common random numbers across every cell of one calibration",
            "final": "fresh streams, independent of the screen",
            "seeds": "derived from the protocol coordinates (rule, branch, year), "
                     "never from the data",
        },
        "auxiliary": {
            "list": list(AUXILIARY),
            "rule": "reported every year, NEVER added to the objective whatever they "
                    "show",
        },
        "reporting_limits": [
            "this is a RETROSPECTIVE walk-forward replay over development history "
            "that has already been examined; it is not a live track record",
            "the sixteen year-rule-branch cells are not sixteen independent market "
            "experiments: they share one index, one history, overlapping calibration "
            "windows and two branches of the same model family",
            "four years and two overlapping window rules are not independent "
            "replicates; no significance statement is made from them",
            "a Monte-Carlo standard error here is the precision of the EVALUATION "
            "computation given the parameters and the reference weights. It excludes "
            "market sampling uncertainty and parameter-estimation uncertainty "
            "entirely",
            "a failure is not attributed to a regime change by default",
            "the main window rule is not swapped on the strength of the replay",
            "every simulated range is conditional on the model AND the parameters, and "
            "carries no parameter-estimation uncertainty",
        ],
        "if_annual_re_estimation_is_ever_adopted":
            "freeze the ALGORITHM and its hyper-parameters -- the grid, the objective, "
            "the window rule, the budget, the selection rule -- and let the PARAMETERS "
            "update every year by that frozen rule. 'Re-estimate annually' and 'do not "
            "re-tune between years' are not in conflict: the first is the parameters "
            "moving under a fixed procedure, the second is the procedure itself not "
            "being touched once the evaluation has begun.",
        "relation_to_the_monitoring_project": "a generator that re-estimates well does "
                                              "NOT imply the monitor's two-year 15% "
                                              "cumulative false-alarm control still "
                                              "holds. Using such updates in a real "
                                              "monitor would require recalibrating the "
                                              "whole rule INCLUDING the update action; "
                                              "a budget is not re-issued at each update.",
    }


# --------------------------------------------------------------------------- #


def time_boundary_check(cfg, returns: pd.DataFrame, rule: str, branch: str,
                        year: int, paths: int = 40) -> dict:
    """Perturbing returns AFTER the origin must change nothing at the origin."""
    _, end = window_for(rule, year)
    mutated = returns.copy()
    after = mutated["date"] > pd.Timestamp(end)
    mutated.loc[after, "ret"] = mutated.loc[after, "ret"] * -3.0 + 0.05
    a = calibrate_at_origin(cfg, returns, rule, branch, year, paths, paths, 2)
    b = calibrate_at_origin(cfg, mutated, rule, branch, year, paths, paths, 2)
    same = {k: (a[k] == b[k]) for k in ("A", "rho", "sigma_annual", "window_n_returns",
                                        "calibration_loss", "seed_tag")}
    same["standardising_scales"] = a["standardising_scales"] == b["standardising_scales"]
    return {"rule": rule, "branch": branch, "eval_year": year,
            "n_returns_perturbed": int(after.sum()),
            "identical": all(same.values()), "fields": same}


def two_process_reproducibility_check() -> dict:
    """Run the same computation in two FRESH interpreters and compare.

    This is the check the old hash-based tag would have failed: CPython randomises
    string hashing per process, so a seed derived from `hash(str(...))` differed on
    every run while looking stable inside one.
    """
    import subprocess
    import sys
    snippet = (
        "import json, numpy as np;"
        "from strategy_survivorship import market_stage4 as m4, market_stage3 as m3;"
        "from strategy_survivorship.config import DEFAULT as c;"
        "t=m4._tag('rolling3y','sv_only',2021);"
        "r=m3.simulate(m3.branch_cfg(c,'sv_only',1.2,0.96,0.19),"
        "  8, 200, np.random.SeedSequence([m4.CAL_SCREEN_ENTROPY, t]));"
        "print(json.dumps({'tag': int(t), 'mean': float(r.mean()),"
        "  'sd': float(r.std(ddof=1)), 'first': float(r[0,0])}))")
    outs = []
    for _ in range(2):
        res = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                             text=True, timeout=180)
        if res.returncode != 0:
            return {"ran": False, "error": res.stderr[-400:]}
        outs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    return {"ran": True, "process_1": outs[0], "process_2": outs[1],
            "identical": outs[0] == outs[1],
            "note": "two fresh interpreters, no PYTHONHASHSEED set"}


def holdout_perturbation_check(cfg, raw: pd.DataFrame) -> dict:
    """Corrupting every holdout return must leave every development result alone.

    The full CSV is loaded and then truncated at the development end, so nothing
    after it can reach a calculation. This exercises that claim rather than
    asserting it.
    """
    bad = raw.copy()
    after = bad["date"] > pd.Timestamp(DEV_END)
    bad.loc[after, "ret"] = bad.loc[after, "ret"] * -7.0 + 0.25
    a = data_up_to(raw, DEV_END)
    b = data_up_to(bad, DEV_END)
    frames_equal = a.equals(b)
    ca = calibrate_at_origin(cfg, a, "expanding", "sv_only", 2023, 30, 30, 2)
    cb = calibrate_at_origin(cfg, b, "expanding", "sv_only", 2023, 30, 30, 2)
    keys = ("A", "rho", "sigma_annual", "calibration_loss", "seed_tag",
            "window_n_returns")
    return {"holdout_returns_corrupted": int(after.sum()),
            "development_frame_identical": bool(frames_equal),
            "calibration_identical": all(ca[k] == cb[k] for k in keys),
            "standardising_scales_identical":
                ca["standardising_scales"] == cb["standardising_scales"],
            "note": "the holdout was loaded with the CSV and excluded before any "
                    "analysis; it was never read as data"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Market stage 4: walk-forward replay.")
    ap.add_argument("--screen-paths", type=int, default=SCREEN_PATHS)
    ap.add_argument("--final-paths", type=int, default=FINAL_PATHS)
    ap.add_argument("--eval-paths", type=int, default=EVAL_PATHS)
    ap.add_argument("--smoke", action="store_true",
                    help="a small run, still confined to 2017-2023")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    sp = 40 if args.smoke else args.screen_paths
    fp = 60 if args.smoke else args.final_paths
    ep = 60 if args.smoke else args.eval_paths
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULT

    raw = pd.read_csv(IN / "market_sp500_returns.csv",
                      parse_dates=["date", "prev_date"])
    returns = data_up_to(raw, DEV_END)
    assert returns["date"].max() <= pd.Timestamp(DEV_END), "holdout leaked in"
    print(f"development scope only: {returns['date'].min().date()} .. "
          f"{returns['date'].max().date()}, {len(returns)} returns "
          f"({len(raw) - len(returns)} later returns discarded unread)")

    corr = stage3_corrections(out)
    proto = protocol(sp, fp, ep)
    (out / "market_stage4_protocol.json").write_text(
        json.dumps(proto, indent=2, default=float), encoding="utf-8")
    (out / "market_stage4_stage3_corrections.json").write_text(
        json.dumps(corr, indent=2, default=float), encoding="utf-8")
    print("protocol and stage-3 corrections written before the replay")

    repro = two_process_reproducibility_check()
    print(f"two-process reproducibility: identical={repro.get('identical')}  "
          f"tag={repro.get('process_1', {}).get('tag')}")
    if not repro.get("identical"):
        raise AssertionError(f"random stream is not reproducible: {repro}")
    hop = holdout_perturbation_check(cfg, raw)
    print(f"holdout perturbation ({hop['holdout_returns_corrupted']} returns "
          f"corrupted): development frame identical="
          f"{hop['development_frame_identical']}, calibration identical="
          f"{hop['calibration_identical']}")
    if not (hop["development_frame_identical"] and hop["calibration_identical"]
            and hop["standardising_scales_identical"]):
        raise AssertionError(f"holdout leaked: {hop}")

    tb = [time_boundary_check(cfg, returns, "rolling3y", "sv_only", 2021)]
    print(f"time-boundary check: perturbing {tb[0]['n_returns_perturbed']} returns "
          f"after the origin changed nothing: {tb[0]['identical']}")
    if not tb[0]["identical"]:
        raise AssertionError(f"time boundary violated: {tb[0]['fields']}")

    # one shared scoring reference per (year, branch)
    eval_ref = {}
    for year in EVAL_YEARS:
        for branch in BRANCHES:
            s, fl, sig, nd = evaluation_scales(cfg, returns, year, branch, fp)
            eval_ref[(year, branch)] = {"scales": s, "floored": fl,
                                        "expanding_sigma_annual": sig, "n_days": nd}

    cal_rows, eval_rows, aux_rows, grids, fin_rows = [], [], [], [], []
    cal_weights, eval_weights = [], []
    for (y, b), v in eval_ref.items():
        for tgt, w in v["scales"].items():
            eval_weights.append({"eval_year": y, "branch": b, "target": tgt,
                                 "standardising_scale": w,
                                 "shared_by": "every method scored in this "
                                              "(year, branch) cell"})
    t0 = time.time()
    for year in EVAL_YEARS:
        for rule in RULES:
            for branch in BRANCHES:
                cal = calibrate_at_origin(cfg, returns, rule, branch, year, sp, fp)
                g = cal.pop("grid"); fin = cal.pop("finalists")
                for frame in (g, fin):
                    frame.insert(0, "eval_year", year); frame.insert(1, "rule", rule)
                    frame.insert(2, "branch", branch)
                grids.append(g); fin_rows.append(fin)
                for tgt, w in cal["standardising_scales"].items():
                    cal_weights.append({"eval_year": year, "rule": rule,
                                        "branch": branch, "target": tgt,
                                        "standardising_scale": w,
                                        "floored": tgt in cal["scales_floored"]})
                cal_rows.append({k: v for k, v in cal.items()
                                 if k != "standardising_scales"})
                ref = eval_ref[(year, branch)]
                for method, (A, rho) in (("recalibrated", (cal["A"], cal["rho"])),
                                         ("fixed_baseline", BASELINE)):
                    label = f"{rule}|{branch}|{method}"
                    e = evaluate_year(cfg, returns, year, branch, A, rho,
                                         cal["sigma_annual"], ref["scales"], label, ep)
                    eval_rows.append({
                        "eval_year": year, "rule": rule, "branch": branch,
                        "method": method, "A": A, "rho": rho,
                        "sigma_annual": cal["sigma_annual"],
                        "eval_n_days": e["eval_n_days"], "loss": e["loss"],
                        "loss_mc_se": e["loss_mc_se"],
                        "rv21_component": e["rv21_component"],
                        "acf_abs_component": e["acf_abs_component"],
                        "calibration_window": f"{cal['window_start']}..{cal['window_end']}",
                        "calibration_n_returns": cal["window_n_returns"],
                        "on_grid_boundary": cal["on_grid_boundary"],
                        "n_within_one_mc_se": cal["n_within_one_mc_se"]})
                    for k in AUXILIARY:
                        v = np.asarray(e["sim_stats"][k], float)
                        v = v[np.isfinite(v)]
                        aux_rows.append({
                            "eval_year": year, "rule": rule, "branch": branch,
                            "method": method, "statistic": k,
                            "real": float(e["real_stats"][k][0]),
                            "sim_median": float(np.median(v)),
                            "sim_p2.5": float(np.quantile(v, .025)),
                            "sim_p97.5": float(np.quantile(v, .975))})
        print(f"  {year} done ({time.time() - t0:.0f}s elapsed)")

    cal_df = pd.DataFrame(cal_rows)
    ev = pd.DataFrame(eval_rows)
    pd.concat(grids, ignore_index=True).to_csv(out / "market_stage4_grids.csv",
                                               index=False)
    pd.concat(fin_rows, ignore_index=True).to_csv(
        out / "market_stage4_finalists.csv", index=False)
    pd.DataFrame(cal_weights).to_csv(out / "market_stage4_calibration_weights.csv",
                                     index=False)
    pd.DataFrame(eval_weights).to_csv(out / "market_stage4_evaluation_weights.csv",
                                      index=False)
    cal_df.to_csv(out / "market_stage4_calibrations.csv", index=False)
    ev.to_csv(out / "market_stage4_evaluations.csv", index=False)
    pd.DataFrame(aux_rows).to_csv(out / "market_stage4_auxiliary.csv", index=False)

    # the headline table: recalibrated minus its MATCHED baseline, year by year
    piv = ev.pivot_table(index=["eval_year", "rule", "branch"], columns="method",
                         values=["loss", "rv21_component", "acf_abs_component",
                                 "loss_mc_se"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    piv["loss_difference"] = piv["loss_recalibrated"] - piv["loss_fixed_baseline"]
    piv["recalibration_helped"] = piv["loss_difference"] < 0
    piv = piv.merge(cal_df[["eval_year", "rule", "branch", "A", "rho",
                            "window_start", "window_end", "window_n_returns",
                            "sigma_annual", "on_grid_boundary"]],
                    on=["eval_year", "rule", "branch"], how="left")
    piv.to_csv(out / "market_stage4_results.csv", index=False)

    (out / "market_stage4_summary.json").write_text(json.dumps({
        "protocol": proto, "stage3_corrections": corr,
        "code_identity": code_identity(),
        "random_streams": stream_description(),
        "simulation_budget": {"screen_paths": sp, "final_paths": fp,
                              "eval_paths": ep, "n_finalists": N_FINALISTS,
                              "grid_cells_per_calibration":
                                  len(GRID_A) * len(GRID_RHO),
                              "calibrations": len(EVAL_YEARS) * len(RULES)
                                              * len(BRANCHES),
                              "evaluations": len(eval_rows)},
        "reproducibility": repro,
        "time_boundary_checks": tb,
        "holdout_perturbation_check": hop,
        "development_scope_returns": int(len(returns)),
        "evaluation_reference": {f"{y}|{b}": {
            "expanding_sigma_annual": v["expanding_sigma_annual"],
            "n_days": v["n_days"], "scales_floored": v["floored"]}
            for (y, b), v in eval_ref.items()},
    }, indent=2, default=float), encoding="utf-8")

    print()
    for _, r in piv.sort_values(["branch", "rule", "eval_year"]).iterrows():
        print(f"  {r.eval_year} {r.rule:10s} {r.branch:8s} A={r.A:.1f} rho={r.rho:.2f} "
              f"| recal {r.loss_recalibrated:7.3f}  base {r.loss_fixed_baseline:7.3f}  "
              f"diff {r.loss_difference:+7.3f} {'better' if r.loss_difference < 0 else 'WORSE'}")
    pass
    year_differences(piv, out / "market_stage4_year_differences.png")
    parameter_path(piv, out / "market_stage4_parameter_path.png")
    print(f"\nrecalibration beat its matched baseline in "
          f"{int(piv.recalibration_helped.sum())} of {len(piv)} year-rule-branch cells")
    print(f"artefacts -> {out}/market_stage4_*")
    return 0


def cli() -> int:
    """Run the stage, or regenerate its report from saved outputs.

    python -m strategy_survivorship.market_stage4 [OPTIONS]            run the stage
    python -m strategy_survivorship.market_stage4 report [OPTIONS]    read saved outputs only
    """
    import sys

    sub = {"report": main_report}
    if len(sys.argv) > 1 and sys.argv[1] in sub:
        return sub[sys.argv.pop(1)]() or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
