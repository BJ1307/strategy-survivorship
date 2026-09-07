"""Stage 1 end-to-end pipeline.

    python -m strategy_survivorship.run_stage1 [--smoke] [--out DIR]

Simulate -> calibrate on held-out valid paths -> evaluate on independent test
paths -> single-shock diagnostic -> tables, figures and report.  No notebook and
no manual cell ordering is involved anywhere in this path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import DEFAULT, Stage1Config
from .detectors import (
    DETECTORS,
    DETECTORS_BY_KEY,
    binary_gaussian_increments,
    binary_student_t_increments,
    influence_curve,
    probability_from_log_odds,
    standardise,
)
from .evaluate import (
    NO_ALARM,
    build_metric_row,
    calibrate_threshold,
    first_passage,
    random_closure_reference,
    random_reference_row,
)
from .report import write_report
from .robustness import calibration_robustness
from .simulate import build_pathsets, make_streams, simulate_diagnostic_paths, stream_fingerprint

TEST_SETS = ("test_valid", "test_invalid")


# --------------------------------------------------------------------------- #


def environment_info() -> dict:
    import matplotlib
    import scipy

    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def run_diagnostic(cfg: Stage1Config) -> tuple[dict, pd.DataFrame]:
    """Single-shock diagnostic traces for the two Bayesian detectors."""
    paths = simulate_diagnostic_paths(cfg)
    out: dict = {"binary_gaussian": {}, "binary_student_t": {}, "z_on_shock_day": {}}
    frames = []
    for vkey, r in paths.items():
        r2 = r[None, :]
        for mkey, inc_fn in (
            ("binary_gaussian", binary_gaussian_increments),
            ("binary_student_t", binary_student_t_increments),
        ):
            inc = inc_fn(r2, cfg)[0]
            lo = cfg.prior_log_odds + np.cumsum(inc)
            out[mkey][vkey] = {
                "increment": inc,
                "log_odds": lo,
                "prob": probability_from_log_odds(lo),
            }
        out["z_on_shock_day"][vkey] = float(standardise(r, cfg)[cfg.shock_day - 1])
        frames.append(
            pd.DataFrame(
                {
                    "day": np.arange(1, cfg.horizon_days + 1),
                    "variant": vkey,
                    "return": r,
                    "z_standardised": standardise(r, cfg),
                    "gaussian_increment": out["binary_gaussian"][vkey]["increment"],
                    "gaussian_log_odds": out["binary_gaussian"][vkey]["log_odds"],
                    "gaussian_prob_valid": out["binary_gaussian"][vkey]["prob"],
                    "student_t_increment": out["binary_student_t"][vkey]["increment"],
                    "student_t_log_odds": out["binary_student_t"][vkey]["log_odds"],
                    "student_t_prob_valid": out["binary_student_t"][vkey]["prob"],
                }
            )
        )
    return out, pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Stage 1 benchmark end to end.")
    ap.add_argument("--smoke", action="store_true", help="tiny sample sizes, for a pipeline check")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--seed", type=int, default=None, help="override the root seed")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--replications",
        type=int,
        default=None,
        help="independent re-runs used to size the calibration uncertainty "
        "(default: cfg.n_calibration_replications; 0 disables the stage)",
    )
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    if args.seed is not None:
        cfg = replace(cfg, root_seed=args.seed)
    cfg.validate()

    out_dir = args.out or (Path("outputs/smoke") if args.smoke else Path("outputs"))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    # ---- simulate --------------------------------------------------------- #
    t0 = time.perf_counter()
    pathsets = build_pathsets(cfg)
    streams = make_streams(cfg)
    timings["simulate_s"] = time.perf_counter() - t0
    print(
        f"[simulate] calibration={pathsets['calibration_valid'].n_paths} "
        f"test_valid={pathsets['test_valid'].n_paths} "
        f"test_invalid={pathsets['test_invalid'].n_paths} "
        f"x {cfg.horizon_days}d  ({timings['simulate_s']:.2f}s)"
    )

    # ---- detector statistics ---------------------------------------------- #
    t0 = time.perf_counter()
    stats: dict[str, dict[str, np.ndarray]] = {}
    for det in DETECTORS:
        stats[det.key] = {s: det.compute(pathsets[s].returns, cfg) for s in TEST_SETS}
    timings["test_statistics_s"] = time.perf_counter() - t0

    # ---- calibration (valid paths only) ----------------------------------- #
    t0 = time.perf_counter()
    calibrations = {}
    for det in DETECTORS:
        cal_stats = det.compute(pathsets["calibration_valid"].returns, cfg)
        eligible = det.first_eligible_day(cfg)
        for alpha in cfg.far_targets:
            c = calibrate_threshold(
                cal_stats, alpha, eligible, cfg.horizon_days, detector_key=det.key
            )
            if c.achieved_far > alpha + 1e-12:
                raise AssertionError(
                    f"calibration FAR {c.achieved_far} exceeds target {alpha} for {det.key}"
                )
            calibrations[(det.key, alpha)] = c
            print(
                f"[calibrate] {det.key:24s} alpha={alpha:.2f}  "
                f"threshold={c.threshold:+.6f}  achieved_cal_FAR={c.achieved_far:.4f}"
            )
        del cal_stats
    timings["calibration_s"] = time.perf_counter() - t0

    # ---- frozen-threshold evaluation on the independent test sets --------- #
    t0 = time.perf_counter()
    passages, rows, fp_records, curve_records = {}, [], [], []
    for det in DETECTORS:
        eligible = det.first_eligible_day(cfg)
        for alpha in cfg.far_targets:
            thr = calibrations[(det.key, alpha)].threshold
            for set_name in TEST_SETS:
                fp = first_passage(stats[det.key][set_name], thr, eligible, cfg.horizon_days)
                passages[(det.key, alpha, set_name)] = fp
                fp_records.append(
                    pd.DataFrame(
                        {
                            "path_set": set_name,
                            "path_id": np.arange(fp.n_paths, dtype=np.int32),
                            "true_state": "valid" if pathsets[set_name].is_valid else "invalid",
                            "true_sharpe_annual": pathsets[set_name].sharpe_true,
                            "detector": det.key,
                            "far_target": alpha,
                            "threshold": thr,
                            "first_eligible_day": eligible,
                            "alarmed": fp.alarmed,
                            "first_alarm_day": fp.first_day,  # -1 = never alarmed
                            "truncated_days": fp.truncated_days,
                        }
                    )
                )
                curve_records.append(
                    pd.DataFrame(
                        {
                            "day": np.arange(1, cfg.horizon_days + 1, dtype=np.int32),
                            "detector": det.key,
                            "far_target": alpha,
                            "path_set": set_name,
                            "true_state": "valid" if pathsets[set_name].is_valid else "invalid",
                            "cumulative_alarm_rate": fp.cumulative_rate,
                        }
                    )
                )
            rows.append(
                build_metric_row(
                    detector=det,
                    cfg=cfg,
                    calib=calibrations[(det.key, alpha)],
                    fp_valid=passages[(det.key, alpha, "test_valid")],
                    fp_invalid=passages[(det.key, alpha, "test_invalid")],
                )
            )
    for alpha in cfg.far_targets:
        rows.append(random_reference_row(cfg, alpha))
        ref = random_closure_reference(alpha, cfg.horizon_days)
        for set_name in TEST_SETS:
            curve_records.append(
                pd.DataFrame(
                    {
                        "day": np.arange(1, cfg.horizon_days + 1, dtype=np.int32),
                        "detector": "random_closure",
                        "far_target": alpha,
                        "path_set": set_name,
                        "true_state": "valid" if set_name == "test_valid" else "invalid",
                        "cumulative_alarm_rate": ref["cumulative_rate"],
                    }
                )
            )
    timings["evaluation_s"] = time.perf_counter() - t0

    # ---- diagnostic -------------------------------------------------------- #
    t0 = time.perf_counter()
    diag, diag_df = run_diagnostic(cfg)
    timings["diagnostic_s"] = time.perf_counter() - t0

    # ---- how much does the calibration draw itself move the FAR? ---------- #
    n_rep = cfg.n_calibration_replications if args.replications is None else args.replications
    robustness = None
    if n_rep and n_rep >= 2:
        t0 = time.perf_counter()
        robustness = calibration_robustness(cfg, n_rep)
        robustness.to_csv(out_dir / "stage1_calibration_robustness.csv", index=False)
        timings["calibration_robustness_s"] = time.perf_counter() - t0
        print(
            f"[robustness] {n_rep} independent calibrate-then-measure replications  "
            f"({timings['calibration_robustness_s']:.1f}s)"
        )

    # ---- tables ------------------------------------------------------------ #
    t0 = time.perf_counter()
    order = {d.key: i for i, d in enumerate(DETECTORS)}
    order["random_closure"] = len(order)
    rows.sort(key=lambda r: (r["far_target"], order[r["detector"]]))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "stage1_metrics.csv", index=False)
    pd.concat(fp_records, ignore_index=True).to_csv(
        out_dir / "stage1_first_passages.csv", index=False
    )
    pd.concat(curve_records, ignore_index=True).to_csv(
        out_dir / "stage1_curves.csv", index=False
    )
    diag_df.to_csv(out_dir / "stage1_diagnostic_traces.csv", index=False)
    timings["tables_s"] = time.perf_counter() - t0

    # ---- figures ----------------------------------------------------------- #
    figures: list[str] = []
    if not args.no_figures:
        t0 = time.perf_counter()
        plots.setup_style()
        figures = [
            str(
                plots.figure_example_paths(
                    cfg, pathsets, stats, calibrations, fig_dir / "fig1_example_paths.png"
                ).relative_to(out_dir)
            ),
            str(
                plots.figure_operating_curves(
                    cfg, passages, fig_dir / "fig2_operating_curves.png"
                ).relative_to(out_dir)
            ),
            str(
                plots.figure_detection_time(
                    cfg, rows, fig_dir / "fig3_detection_time.png"
                ).relative_to(out_dir)
            ),
            str(
                plots.figure_shock_diagnostic(
                    cfg, diag, calibrations, fig_dir / "fig4_shock_diagnostic.png"
                ).relative_to(out_dir)
            ),
        ]
        timings["figures_s"] = time.perf_counter() - t0

    # ---- summary / metadata ------------------------------------------------ #
    timings["total_s"] = time.perf_counter() - t_start
    shock_idx = cfg.shock_day - 1
    _curve = influence_curve(cfg, -60.0, 60.0, 24001)
    _neg = _curve["z"] < 0
    _peak = int(np.argmax(np.abs(_curve["student_t"][_neg])))
    _at = lambda z0: int(np.argmin(np.abs(_curve["z"] - z0)))
    influence_facts = {
        "note": (
            "one-step log-likelihood increment as a function of z; the Student-t "
            "influence function is bounded and redescending, the Gaussian one is affine"
        ),
        "student_t_peak_abs_increment": float(np.abs(_curve["student_t"][_neg])[_peak]),
        "student_t_peak_at_z": float(_curve["z"][_neg][_peak]),
        "increment_at_z": {
            f"{z0:g}": {
                "gaussian": float(_curve["gaussian"][_at(z0)]),
                "student_t": float(_curve["student_t"][_at(z0)]),
            }
            for z0 in (-1.0, -2.0, -8.0, -20.0, -50.0)
        },
    }
    summary = {
        "label": cfg.label,
        "stage": 1,
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(streams),
        "sample_sizes": {
            "calibration_valid_paths": pathsets["calibration_valid"].n_paths,
            "test_valid_paths": pathsets["test_valid"].n_paths,
            "test_invalid_paths": pathsets["test_invalid"].n_paths,
            "days_per_path": cfg.horizon_days,
        },
        "thresholds": {
            f"{k[0]}|alpha={k[1]:g}": {
                "threshold": v.threshold,
                "achieved_calibration_far": v.achieved_far,
                "first_eligible_day": v.first_eligible_day,
                "rule": v.rule,
            }
            for k, v in calibrations.items()
        },
        "metrics": json.loads(metrics.to_json(orient="records")),
        "diagnostic": {
            "path_stream": "diagnostic",
            "path_index": cfg.diagnostic_path_index,
            "shock_day": cfg.shock_day,
            "shock_in_daily_sigma": cfg.shock_in_daily_sigma,
            "one_step_increment_at_shock": {
                m: {v: float(diag[m][v]["increment"][shock_idx]) for v in ("base", "shock_plus", "shock_minus")}
                for m in ("binary_gaussian", "binary_student_t")
            },
            "log_odds_at_horizon": {
                m: {v: float(diag[m][v]["log_odds"][-1]) for v in ("base", "shock_plus", "shock_minus")}
                for m in ("binary_gaussian", "binary_student_t")
            },
            "z_on_shock_day": {k: float(v) for k, v in diag["z_on_shock_day"].items()},
            "log_odds_shift_vs_base_at_horizon": {
                m: {
                    v: float(diag[m][v]["log_odds"][-1] - diag[m]["base"]["log_odds"][-1])
                    for v in ("shock_plus", "shock_minus")
                }
                for m in ("binary_gaussian", "binary_student_t")
            },
            "student_t_influence": influence_facts,
        },
        "calibration_robustness": (
            json.loads(robustness.to_json(orient="records"))
            if robustness is not None
            else None
        ),
        "timings_seconds": timings,
        "figures": figures,
        "outputs": sorted(p.name for p in out_dir.glob("stage1_*")),
    }
    (out_dir / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "environment": environment_info(),
                "random_streams": stream_fingerprint(streams),
                "timings_seconds": timings,
                "command": "python -m strategy_survivorship.run_stage1"
                + (" --smoke" if args.smoke else ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_report(cfg, metrics, summary, out_dir / "stage1_report.md")

    print(f"[done] {timings['total_s']:.1f}s -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
