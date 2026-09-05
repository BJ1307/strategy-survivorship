"""How much does the *calibration sample* itself move the realised false-alarm rate?

The Wilson intervals reported for a single run cover only the Monte-Carlo noise of
one rate measured on one test set.  They say nothing about the fact that the
threshold was itself estimated from a finite calibration sample, and that a
different calibration draw would have produced a different threshold and hence a
different realised FAR.

This module answers that empirically: repeat the whole
    simulate -> calibrate on 5,000 valid paths -> freeze -> measure FAR on 5,000
    independent valid paths
loop under R independent root seeds, and compare the spread of the realised FAR
against the binomial standard error of a single measurement.  The ratio is the
factor by which a pointwise Wilson interval understates the real uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import Stage1Config
from .detectors import DETECTORS
from .evaluate import calibrate_threshold, first_passage
from .simulate import build_pathsets


def calibration_robustness(cfg: Stage1Config, n_replications: int) -> pd.DataFrame:
    """Re-run calibrate-then-measure under ``n_replications`` independent seeds.

    Seeds are ``cfg.root_seed + 1 ... + n_replications`` so the headline run
    (``cfg.root_seed``) is not one of them -- the replications are an independent
    check of the *procedure*, not a re-use of the reported experiment.
    """
    if n_replications < 2:
        raise ValueError("need at least 2 replications to estimate a spread")

    records: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for i in range(1, n_replications + 1):
        rep_cfg = replace(cfg, root_seed=cfg.root_seed + i)
        paths = build_pathsets(rep_cfg)
        for det in DETECTORS:
            cal = det.compute(paths["calibration_valid"].returns, rep_cfg)
            test = det.compute(paths["test_valid"].returns, rep_cfg)
            eligible = det.first_eligible_day(rep_cfg)
            for alpha in rep_cfg.far_targets:
                c = calibrate_threshold(
                    cal, alpha, eligible, rep_cfg.horizon_days, det.key
                )
                fp = first_passage(test, c.threshold, eligible, rep_cfg.horizon_days)
                records.setdefault((det.key, alpha), []).append(
                    (c.threshold, float(fp.alarmed.mean()))
                )
            del cal, test

    rows = []
    n_test = cfg.n_test_valid
    for (key, alpha), vals in records.items():
        thresholds = np.array([v[0] for v in vals])
        fars = np.array([v[1] for v in vals])
        binomial_se = math.sqrt(alpha * (1.0 - alpha) / n_test)
        observed_sd = float(fars.std(ddof=1))
        # relative standard error of an sd estimated from R replications
        sd_of_sd = observed_sd / math.sqrt(2.0 * (n_replications - 1))
        # Variance decomposition.  Redrawing everything each replication means
        #   Var(realised FAR) = Var(threshold effect) + E[binomial variance],
        # so the calibration-only component is the excess over the binomial term.
        # It is clipped at zero: a negative estimate means the excess is smaller
        # than this study can resolve, not that calibration removes variance.
        excess_var = observed_sd**2 - binomial_se**2
        calib_sd = math.sqrt(max(excess_var, 0.0))
        # One-sided test of H0: total sd equals the binomial sd (no calibration
        # contribution), using the chi-square sampling law of a sample variance.
        # Approximate: the replication FARs are proportions, not exactly normal.
        dof = n_replications - 1
        p_value = float(chi2.sf(dof * observed_sd**2 / binomial_se**2, dof))
        rows.append(
            {
                "detector": key,
                "far_target": alpha,
                "n_replications": n_replications,
                "mean_test_far": float(fars.mean()),
                "sd_test_far": observed_sd,
                "sd_test_far_se": sd_of_sd,
                "min_test_far": float(fars.min()),
                "max_test_far": float(fars.max()),
                "frac_replications_above_target": float((fars > alpha).mean()),
                "binomial_se_single_run": binomial_se,
                "sd_inflation_vs_binomial": observed_sd / binomial_se,
                "calibration_only_sd": calib_sd,
                "calibration_excess_var_is_negative": bool(excess_var < 0),
                "p_value_no_calibration_excess": p_value,
                "sd_threshold": float(thresholds.std(ddof=1)),
                "mean_threshold": float(thresholds.mean()),
            }
        )
    order = {d.key: i for i, d in enumerate(DETECTORS)}
    return pd.DataFrame(rows).sort_values(
        ["far_target", "detector"], key=lambda s: s.map(order) if s.name == "detector" else s
    ).reset_index(drop=True)
