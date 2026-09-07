"""Stage 2E pipeline: does truncating the variance update help?

    python -m strategy_survivorship.run_stage2e [--smoke] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import plots_stage2e as p2e
from .config import DEFAULT
from .evaluate import excludes_zero
from .probability_time import brier_and_reliability
from .report_stage2e import write_stage2e_report
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint
from .stage2b_followup import paired_brier_difference
from .stage2d import COMBINED, load_frozen_thresholds, specs
from .stage2d_uncertainty import paired_frozen, paired_truncated_time
from .stage2e import CORE_2x2, LABEL_2E, METHODS_2E, calibrate, evaluate, statistic

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
        for tag, v in (("plain", vp), ("trunc", vt)):
            gi = ewma_gaussian_increments(r2, cfg, v)[0]
            ti = ewma_student_t_increments(r2, cfg, v)[0]
            cols[f"inc_{tag}_gaussian"] = gi
            cols[f"inc_{tag}_student_t"] = ti
            cols[f"cum_{tag}_gaussian"] = cfg.prior_log_odds + np.cumsum(gi)
            cols[f"cum_{tag}_student_t"] = cfg.prior_log_odds + np.cumsum(ti)
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
        p2e.setup_style()
        print("figures redrawn from disk:", p2e.draw_all(cfg, out_dir))
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
        p2e.setup_style()
        figures = p2e.draw_all(cfg, out_dir)
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
