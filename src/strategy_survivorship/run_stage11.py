"""Stage 1.1 diagnostics.

    python -m strategy_survivorship.run_stage11 [--smoke] [--out DIR]

Three blocks, all auxiliary to the Stage 1 benchmark and none of them changing it:

1. analytic calibration uncertainty (exact Beta / Beta-binomial law),
2. probability-time correspondence (how long belief takes to move),
3. random failure time (valid first, invalid later).

Stage 1's frozen thresholds are reused so the decision rule stays fixed while the
data-generating process changes.  They were calibrated on a 504-day horizon and
carry no cumulative false-alarm guarantee on the longer windows used here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import plots_stage11 as p11
from .analytic_calibration import analytic_calibration_table, compare_with_replications
from .config import DEFAULT, Stage1Config
from .detectors import DETECTORS, DETECTORS_BY_KEY, binary_gaussian_log_odds, binary_student_t_log_odds
from .evaluate import calibrate_threshold
from .probability_time import (
    analytic_mean_probability,
    analytic_threshold_time,
    brier_and_reliability,
    failure_probability_summary,
    first_threshold_hit,
    gaussian_failure_log_odds_law,
    threshold_table,
)
from .report_stage11 import write_stage11_report
from .run_stage1 import environment_info
from .paired import paired_table
from .simulate import build_pathsets, make_streams, simulate_returns, stream_fingerprint
from .switching import (
    NO_ALARM,
    matched_pre_failure_thresholds,
    belief_at_failure,
    bin_random_T,
    first_alarm_days,
    simulate_switching_returns,
    switching_metrics,
)

BAYES = {"binary_gaussian": binary_gaussian_log_odds, "binary_student_t": binary_student_t_log_odds}


class Status:
    """Tiny append-only status file so a long run stays pollable (CLAUDE.md)."""

    def __init__(self, path: Path):
        self.path, self.t0 = path, time.perf_counter()

    def __call__(self, stage: str, **counts):
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.perf_counter() - self.t0, 1),
            "stage": stage,
            **counts,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{payload['elapsed_s']:6.1f}s] {stage}" + (f"  {counts}" if counts else ""), flush=True)


def stage1_thresholds(cfg: Stage1Config) -> dict:
    """Recompute the Stage 1 frozen thresholds (same seed, same calibration set)."""
    paths = build_pathsets(cfg)
    out = {}
    for det in DETECTORS:
        stat = det.compute(paths["calibration_valid"].returns, cfg)
        for alpha in cfg.far_targets:
            out[(det.key, alpha)] = calibrate_threshold(
                stat, alpha, det.first_eligible_day(cfg), cfg.horizon_days, det.key
            ).threshold
    return out


# --------------------------------------------------------------------------- #


def block_probability_time(cfg: Stage1Config, status: Status) -> dict:
    """q_n distributions, probability-threshold timing, Brier / reliability."""
    streams = make_streams(cfg)
    n_days = max(cfg.prob_time_long_days)
    states = {
        "valid": (cfg.sharpe_valid, "prob_time_valid"),
        "invalid": (cfg.sharpe_invalid, "prob_time_invalid"),
    }
    returns = {
        name: simulate_returns(streams[stream], cfg.prob_time_paths, n_days, sharpe, cfg)
        for name, (sharpe, stream) in states.items()
    }
    status("prob-time: simulated", paths=cfg.prob_time_paths, days=n_days)

    report_days = tuple(cfg.prob_time_report_days) + tuple(cfg.prob_time_long_days)
    summaries, thresholds, briers, reliab, curves = [], [], [], [], []
    q_store: dict[tuple[str, str], np.ndarray] = {}

    for key, fn in BAYES.items():
        for state, r in returns.items():
            q = expit(-fn(r, cfg))  # q_n = expit(U_n) = expit(-L_n)
            q_store[(key, state)] = q
            summaries.append(
                failure_probability_summary(q, report_days, "stage1.1 diagnostic", key, state)
            )
            thresholds.append(
                threshold_table(q, cfg, cfg.prob_thresholds, key, state, cfg.prob_coverage_target)
            )
            for b in cfg.prob_thresholds:
                hit = first_threshold_hit(q, b)
                curves.append(
                    pd.DataFrame(
                        {
                            "day": np.arange(1, n_days + 1, dtype=np.int32),
                            "detector": key,
                            "true_state": state,
                            "threshold_b": b,
                            "cumulative_reached": hit.cumulative,
                        }
                    )
                )
        for day in cfg.brier_days:
            s, tab = brier_and_reliability(q_store[(key, "invalid")], q_store[(key, "valid")], day, key)
            briers.append(s)
            reliab.append(tab)
        status(f"prob-time: {key} done", detectors_done=len(briers) // len(cfg.brier_days))

    # analytic cross-check of the Gaussian law (mean/sd of U and E[q])
    checks = []
    for state, true_s in (("valid", cfg.sharpe_valid), ("invalid", cfg.sharpe_invalid)):
        U = -binary_gaussian_log_odds(returns[state], cfg)
        for d in report_days:
            m, sd = gaussian_failure_log_odds_law(d, cfg, true_s)
            emp = U[:, d - 1]
            checks.append(
                {
                    "true_state": state,
                    "day": d,
                    "U_mean_empirical": float(emp.mean()),
                    "U_mean_analytic": float(m),
                    "U_mean_z": float((emp.mean() - m) / (sd / np.sqrt(emp.size))),
                    "U_sd_empirical": float(emp.std(ddof=1)),
                    "U_sd_analytic": float(sd),
                    "mean_q_empirical": float(expit(emp).mean()),
                    "mean_q_analytic": float(analytic_mean_probability(d, cfg, true_s)[0]),
                    "sigmoid_of_mean_U": float(expit(emp.mean())),
                }
            )

    sens = [
        analytic_threshold_time(b, cfg, alt_sharpe=s)
        for s in cfg.sensitivity_sharpe
        for b in cfg.prob_thresholds
    ]
    return {
        "summary": pd.concat(summaries, ignore_index=True),
        "thresholds": pd.concat(thresholds, ignore_index=True),
        "brier": pd.DataFrame(briers),
        "reliability": pd.concat(reliab, ignore_index=True),
        "curves": pd.concat(curves, ignore_index=True),
        "analytic_checks": pd.DataFrame(checks),
        "sensitivity": pd.DataFrame(sens),
        "q_curves": {k: np.quantile(v, [0.1, 0.5, 0.9], axis=0) for k, v in q_store.items()},
        "n_days": n_days,
    }


def block_switching(cfg: Stage1Config, thresholds: dict, status: Status) -> dict:
    """Valid-then-failing strategies, fixed and random switch times."""
    streams = make_streams(cfg)
    n_days = cfg.switch_max_days
    post = cfg.switch_post_window
    rows, beliefs, traces = [], [], []

    def evaluate(returns, T, group, keep_traces=False):
        stats_cache = {}
        for det in DETECTORS:
            stat = det.compute(returns, cfg)
            tau = {
                a: first_alarm_days(stat, thresholds[(det.key, a)], det.first_eligible_day(cfg))
                for a in cfg.far_targets
            }
            for a in cfg.far_targets:
                rows.append(
                    switching_metrics(
                        tau[a], T, cfg, cfg.switch_post_horizons, post,
                        detector=det.key, far_target=a, group=group,
                    )
                )
            if det.key in BAYES:
                stats_cache[det.key] = stat
            else:
                del stat
        for key, stat in stats_cache.items():
            a0 = cfg.far_targets[0]
            tau0 = first_alarm_days(stat, thresholds[(key, a0)], DETECTORS_BY_KEY[key].first_eligible_day(cfg))
            survived = ~((tau0 != NO_ALARM) & (tau0 <= T))
            beliefs.append(belief_at_failure(-stat, T, survived, key, group))
            if keep_traces:
                traces.append((group, key, np.quantile(-stat, [0.1, 0.5, 0.9], axis=0)))
        return

    # --- fixed T: common random numbers across T, so differences are the drift --
    for T in cfg.switch_fixed_T:
        r, Tarr = simulate_switching_returns(
            streams["switch_fixed"], cfg.switch_fixed_paths, n_days, T, cfg
        )
        keep = min(T + post, n_days)
        evaluate(r[:, :keep], Tarr, f"fixed_T={T}", keep_traces=True)
        status(f"switching: fixed T={T}", paths=cfg.switch_fixed_paths, days=keep)

    # --- matched pre-failure false-alarm rate --------------------------------
    # A common nominal budget is NOT a common realised pre-failure alarm rate, and
    # a detector that alarms more before the failure will also look better after
    # it.  Re-calibrate every detector on an INDEPENDENT block of valid-only paths
    # so the post-failure comparison starts from the same pre-failure cost.
    matched_rows = []
    cal_valid, _ = simulate_switching_returns(
        streams["switch_matched_cal"], cfg.switch_matched_cal_paths, n_days, np.inf, cfg
    )
    for T in cfg.switch_fixed_T:
        if T == 0:
            continue  # nothing can alarm before day 1; pre-failure rate is 0 by construction
        r, Tarr = simulate_switching_returns(
            streams["switch_fixed"], cfg.switch_fixed_paths, n_days, T, cfg
        )
        keep = min(T + post, n_days)
        for det in DETECTORS:
            thr_m = matched_pre_failure_thresholds(
                cal_valid, det, cfg, T, cfg.switch_matched_pre_fa
            )
            stat = det.compute(r[:, :keep], cfg)
            tau = first_alarm_days(stat, thr_m, det.first_eligible_day(cfg))
            row = switching_metrics(
                tau, Tarr, cfg, cfg.switch_post_horizons, post,
                detector=det.key, far_target=cfg.switch_matched_pre_fa,
                group=f"matched_preFA_T={T}",
            )
            row["matched_threshold"] = thr_m
            row["matched_target_pre_fa"] = cfg.switch_matched_pre_fa
            matched_rows.append(row)
            del stat
        status(f"switching: matched pre-FA T={T}", target=cfg.switch_matched_pre_fa)
    rows.extend(matched_rows)

    # --- random T ------------------------------------------------------------
    rng = np.random.default_rng(streams["switch_random"])
    T_rand = rng.integers(0, cfg.switch_random_T_max + 1, size=cfg.switch_random_paths).astype(float)
    r, _ = simulate_switching_returns(
        streams["switch_random"], cfg.switch_random_paths, n_days, T_rand, cfg
    )
    evaluate(r, T_rand, "random_T")
    status("switching: random T", paths=cfg.switch_random_paths)

    # binned by pre-declared intervals
    idx, labels = bin_random_T(T_rand, cfg.switch_T_bin_edges)
    for b, label in enumerate(labels):
        m = idx == b
        if not m.any():
            continue
        sub = r[m]
        for det in DETECTORS:
            stat = det.compute(sub, cfg)
            for a in cfg.far_targets:
                tau = first_alarm_days(stat, thresholds[(det.key, a)], det.first_eligible_day(cfg))
                rows.append(
                    switching_metrics(
                        tau, T_rand[m], cfg, cfg.switch_post_horizons, post,
                        detector=det.key, far_target=a, group=f"random_T[{label}]",
                    )
                )
            del stat
    status("switching: random T binned", bins=len(labels))
    return {
        "metrics": pd.DataFrame(rows),
        "beliefs": pd.DataFrame(beliefs),
        "traces": traces,
        "T_random": T_rand,
        # alarm level used by the trace figure, in U space
        "threshold_for_trace": thresholds[("binary_gaussian", cfg.far_targets[-1])],
        "trace_alpha": cfg.far_targets[-1],
    }


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1.1 diagnostics.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--stage1-first-passages",
        type=Path,
        default=Path("outputs/stage1_first_passages.csv"),
        help="Stage 1 per-path table used for the paired detector comparison",
    )
    ap.add_argument(
        "--replication-csv",
        type=Path,
        default=Path("outputs/stage1_calibration_robustness.csv"),
        help="optional Stage 1 replication study to cross-check the analytic law",
    )
    args = ap.parse_args(argv)

    cfg = DEFAULT.smoke() if args.smoke else DEFAULT
    cfg.validate()
    out_dir = args.out or (Path("outputs/smoke11") if args.smoke else Path("outputs"))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    # 1 ---- analytic calibration -------------------------------------------- #
    analytic = analytic_calibration_table(cfg)
    reps = pd.read_csv(args.replication_csv) if args.replication_csv.exists() else None
    comparison = compare_with_replications(analytic, reps)
    analytic.to_csv(out_dir / "stage11_analytic_calibration.csv", index=False)
    if comparison is not None:
        comparison.to_csv(out_dir / "stage11_calibration_analytic_vs_replication.csv", index=False)
    status("analytic calibration done", budgets=len(analytic))

    # 2 ---- probability and time -------------------------------------------- #
    pt = block_probability_time(cfg, status)
    pt["summary"].to_csv(out_dir / "stage11_failure_probability.csv", index=False)
    pt["thresholds"].to_csv(out_dir / "stage11_probability_thresholds.csv", index=False)
    pt["brier"].to_csv(out_dir / "stage11_brier.csv", index=False)
    pt["reliability"].to_csv(out_dir / "stage11_reliability.csv", index=False)
    pt["analytic_checks"].to_csv(out_dir / "stage11_analytic_checks.csv", index=False)
    pt["sensitivity"].to_csv(out_dir / "stage11_threshold_time_sensitivity.csv", index=False)

    # 2b --- paired detector comparison on the Stage 1 test paths ------------ #
    fp_csv = args.stage1_first_passages
    paired = None
    if fp_csv.exists():
        paired = paired_table(pd.read_csv(fp_csv), cfg.horizon_days)
        paired.to_csv(out_dir / "stage11_paired_comparison.csv", index=False)
        status("paired comparison done", rows=len(paired))

    # 3 ---- random failure time --------------------------------------------- #
    thr = stage1_thresholds(cfg)
    sw = block_switching(cfg, thr, status)
    sw["metrics"].to_csv(out_dir / "stage11_switching_metrics.csv", index=False)
    sw["beliefs"].to_csv(out_dir / "stage11_switching_belief_at_T.csv", index=False)

    # 4 ---- figures ---------------------------------------------------------- #
    figures = []
    if not args.no_figures:
        p11.setup_style()
        figures = [
            str(p11.figure_failure_probability(cfg, pt, fig_dir / "fig11_failure_probability.png").relative_to(out_dir)),
            str(p11.figure_threshold_hits(cfg, pt, fig_dir / "fig12_threshold_hits.png").relative_to(out_dir)),
            str(p11.figure_switching_detection(cfg, sw, fig_dir / "fig13_switching_detection.png").relative_to(out_dir)),
            str(p11.figure_evidence_recovery(cfg, sw, fig_dir / "fig14_evidence_recovery.png").relative_to(out_dir)),
        ]
        status("figures done", n=len(figures))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "1.1",
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "stage1_thresholds_reused": {f"{k[0]}|alpha={k[1]:g}": v for k, v in thr.items()},
        "threshold_caveat": (
            "Stage 1 thresholds were calibrated for a 504-day horizon. On the "
            "longer switching windows they carry no cumulative false-alarm "
            "guarantee; they are reused only to keep the decision rule fixed."
        ),
        "analytic_calibration": json.loads(analytic.to_json(orient="records")),
        "paired_comparison": (
            json.loads(paired.to_json(orient="records")) if paired is not None else None
        ),
        "analytic_vs_replication": (
            json.loads(comparison.to_json(orient="records")) if comparison is not None else None
        ),
        "probability_time": {
            "diagnostic_horizon_days": pt["n_days"],
            "n_paths_per_state": cfg.prob_time_paths,
            "summary": json.loads(pt["summary"].to_json(orient="records")),
            "thresholds": json.loads(pt["thresholds"].to_json(orient="records")),
            "brier": json.loads(pt["brier"].to_json(orient="records")),
            "analytic_checks": json.loads(pt["analytic_checks"].to_json(orient="records")),
            "sensitivity": json.loads(pt["sensitivity"].to_json(orient="records")),
        },
        "switching": {
            "fixed_T": list(cfg.switch_fixed_T),
            "n_paths_fixed": cfg.switch_fixed_paths,
            "n_paths_random": cfg.switch_random_paths,
            "post_window_days": cfg.switch_post_window,
            "T_bin_edges": list(cfg.switch_T_bin_edges),
            "metrics": json.loads(sw["metrics"].to_json(orient="records")),
            "belief_at_T": json.loads(sw["beliefs"].to_json(orient="records")),
        },
        "elapsed_s": elapsed,
        "figures": figures,
    }
    (out_dir / "stage11_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_stage11_report(
        cfg, summary, pt, sw, analytic, comparison, paired, out_dir / "stage11_report.md"
    )
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
