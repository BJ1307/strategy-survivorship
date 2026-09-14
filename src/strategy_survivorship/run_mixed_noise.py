"""mixed_noise: every simulated strategy draws its own (A, kappa).

    python -m strategy_survivorship.run_mixed_noise [--smoke]

Writes only files named mixed_noise_*; no earlier stage's CSV is touched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT
from .mixed_noise import (BATCHES, MAIN_PAIRS, METHODS_MIX, REPORT_DAYS, bootstrap,
                          calibrate, evaluate, experiment_cfg, ranks, streams_for,
                          subgroups)
from .run_stage1 import environment_info
from .run_stage11 import Status
from .simulate import make_streams, stream_fingerprint


def protocol(cfg) -> dict:
    """Written before the run; the report quotes it verbatim."""
    ec = experiment_cfg(cfg)
    return {
        "question": "Do the monitors still work when every strategy has different "
                    "noise characteristics, instead of one fixed parameter point?",
        "states": {"valid_sharpe": ec.sharpe_valid, "invalid_sharpe": ec.sharpe_invalid,
                   "prior": ec.prior_valid, "fixed_from_day_one": True},
        "horizon_days": ec.horizon_days, "days_per_year": ec.D,
        "annual_vol": ec.sigma_annual, "far_budget": ec.far_targets[-1],
        "per_strategy_parameters": {
            "A": f"Uniform(0, {cfg.mixed_A_max:g})   volatility swing amplitude",
            "kappa": f"Uniform(0, {cfg.mixed_kappa_max:g})   jump scale",
            "drawn": "once per strategy, held for the whole window, independent of "
                     "the state and of every noise stream",
            "status": "a simulation choice fixed before the run; not estimated from "
                      "any market and not specified by the supervisor",
        },
        "held_fixed": {"rho": ec.noise_sv_rho,
                       "jump_intensity_per_year": ec.noise_jump_lambda_annual,
                       "latent_AR1_init": "stationary"},
        "parameter_roles": {
            "A": "how far the volatility level swings",
            "rho": "how long a volatility level persists",
            "kappa": "how large a jump is",
            "K": "how many jumps land on a given day",
        },
        "normalisation": "divide by the THEORETICAL constant sqrt(1 + kappa_i^2 "
                         "lambda/D) for that path's own kappa; never by a path's "
                         "sample mean or sample variance",
        "batches": {b: (cfg.mixed_calibration_paths if b == BATCHES[0]
                        else cfg.mixed_test_paths) for b in BATCHES},
        "batch_independence": "parameters and noise are drawn independently per batch; "
                              "the two test batches are shared across methods so the "
                              "comparisons are paired",
        "thresholds": "one per method, calibrated on the whole mixed calibration "
                      "population, then frozen; never chosen using true A, kappa, "
                      "post-hoc groups or test results",
        "detector_inputs": "returns only; no latent volatility, no jump labels, no "
                           "true parameters",
        "methods": list(METHODS_MIX),
        "pre_specified_comparisons": [f"{a} - {b}" for a, b in MAIN_PAIRS],
        "out_of_sample_scope": "the test batches are genuinely out of sample, but they "
                               "come from the SAME generating family as calibration; "
                               "this is not evidence about real strategy returns",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mixed_noise: strategies that differ.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from dataclasses import replace as _r

    cfg = DEFAULT
    if args.smoke:
        cfg = _r(cfg, mixed_calibration_paths=800, mixed_test_paths=800,
                 mixed_bootstrap_reps=40)
    cfg.validate()
    out_dir = args.out or (Path("outputs/smokemix") if args.smoke else Path("outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    t0 = time.perf_counter()

    proto = protocol(cfg)
    (out_dir / "mixed_noise_protocol.json").write_text(
        json.dumps(proto, indent=2, ensure_ascii=False), encoding="utf-8")
    status("protocol written", methods=len(METHODS_MIX),
           A_max=cfg.mixed_A_max, kappa_max=cfg.mixed_kappa_max)

    seeds = streams_for(cfg)
    cal = calibrate(cfg, seeds)
    cal["table"].to_csv(out_dir / "mixed_noise_thresholds.csv", index=False)
    rk = cal["ranks"]
    status("thresholds frozen", n_rules=rk["n_rules"], rank=rk["rank"],
           delta_per_rule=rk["delta_per_rule"])

    rows, minima, taus, bv, bi = evaluate(cfg, seeds, cal["thresholds"])
    metrics = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS_MIX)}
    metrics = metrics.sort_values(by="method", key=lambda s: s.map(order))
    metrics.to_csv(out_dir / "mixed_noise_metrics.csv", index=False)
    status("test scored", rows=len(metrics))

    sg = subgroups(cfg, taus, bv, bi)
    pd.DataFrame(sg).to_csv(out_dir / "mixed_noise_subgroups.csv", index=False)
    status("subgroups done", rows=len(sg))

    # the drawn parameters themselves, so the mixture can be checked
    pd.DataFrame({"batch": "test_invalid", "A": bi["A"], "kappa": bi["kappa"]}
                 ).describe().to_csv(out_dir / "mixed_noise_parameter_summary.csv")

    boots = bootstrap(cfg, cal["minima"], minima, rk,
                      make_streams(cfg)["mixed_noise_bootstrap"])
    pd.DataFrame(boots).to_csv(out_dir / "mixed_noise_bootstrap.csv", index=False)
    status("bootstrap done", rows=len(boots), reps=cfg.mixed_bootstrap_reps)

    elapsed = time.perf_counter() - t0
    summary = {
        "experiment": "mixed_noise", "protocol": proto,
        "config": cfg.to_dict(), "environment": environment_info(),
        "random_streams": stream_fingerprint(make_streams(cfg)),
        "ranks": rk, "report_days": list(REPORT_DAYS),
        "metrics": json.loads(metrics.to_json(orient="records")),
        "subgroups": sg, "bootstrap": boots,
        "parameter_check": {
            "A_mean": float(bi["A"].mean()), "A_min": float(bi["A"].min()),
            "A_max": float(bi["A"].max()),
            "kappa_mean": float(bi["kappa"].mean()), "kappa_min": float(bi["kappa"].min()),
            "kappa_max": float(bi["kappa"].max()),
        },
        "elapsed_s": elapsed,
    }
    (out_dir / "mixed_noise_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    status("done", elapsed_s=round(elapsed, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
