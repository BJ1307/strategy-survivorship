"""Market stage 4: a retrospective walk-forward replay of the stage-3 calibration.

The question: does re-estimating `A` and `rho` before each year describe the
following year's statistics better than leaving the shape parameters fixed?

This is a **retrospective** exercise on history that has already been looked at.
2017-2023 is the development scope; 2024-2025 is never read, plotted or measured.
Re-running a calibration on data we have already studied is not the same as having
run it live, and no result here is evidence about a future year.

What this is NOT
----------------
It is a check on whether the *distribution shape* transfers after a rolling
re-fit. The latent volatility state is always started from its stationary law and
is never filtered on the real path, so nothing here is a conditional forecast of
the coming year, and nothing here uses the evaluation year to choose anything.
"""

from __future__ import annotations

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
                "market_stage3.py", "market_stage4.py", "run_market_stage4.py",
                "noise.py", "config.py")


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
