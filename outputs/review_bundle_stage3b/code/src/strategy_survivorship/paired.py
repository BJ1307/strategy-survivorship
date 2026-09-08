"""Paired comparisons between detectors on the Stage 1 test paths.

All detectors were scored on the *same* simulated paths, so their differences are
paired and the pairing removes the shared path-to-path noise.  Every interval
here is conditional on the Stage 1 thresholds actually frozen in that run: it
answers "given these thresholds, how different are the two rules on fresh data",
not "how much would the gap move if the whole pipeline were recalibrated".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

NO_ALARM = -1


def paired_table(
    first_passages: pd.DataFrame, horizon: int, reference: str = "binary_gaussian", z: float = 1.959963984540054
) -> pd.DataFrame:
    """Paired differences in 2-year detection rate and truncated detection time."""
    inv = first_passages[first_passages.true_state == "invalid"]
    rows = []
    for alpha, block in inv.groupby("far_target"):
        wide_det, wide_time = {}, {}
        for key, sub in block.groupby("detector"):
            sub = sub.sort_values("path_id")
            wide_det[key] = ((sub.first_alarm_day != NO_ALARM) & (sub.first_alarm_day <= horizon)).to_numpy()
            wide_time[key] = sub.truncated_days.to_numpy(dtype=float)
        if reference not in wide_det:
            continue
        for key in wide_det:
            if key == reference:
                continue
            n = wide_det[key].size
            d = wide_det[reference].astype(float) - wide_det[key].astype(float)
            # discordant pairs carry all the information for a paired proportion
            b = int(((d > 0)).sum())  # reference detects, other does not
            c = int(((d < 0)).sum())  # other detects, reference does not
            se_d = d.std(ddof=1) / math.sqrt(n)
            dt = wide_time[reference] - wide_time[key]
            se_t = dt.std(ddof=1) / math.sqrt(n)
            rows.append(
                {
                    "far_target": alpha,
                    "reference": reference,
                    "other": key,
                    "n_paired_paths": n,
                    "detect_rate_reference": float(wide_det[reference].mean()),
                    "detect_rate_other": float(wide_det[key].mean()),
                    "detect_diff": float(d.mean()),
                    "detect_diff_lo": float(d.mean() - z * se_d),
                    "detect_diff_hi": float(d.mean() + z * se_d),
                    "discordant_ref_only": b,
                    "discordant_other_only": c,
                    "trunc_time_reference": float(wide_time[reference].mean()),
                    "trunc_time_other": float(wide_time[key].mean()),
                    "trunc_time_diff_days": float(dt.mean()),
                    "trunc_time_diff_lo": float(dt.mean() - z * se_t),
                    "trunc_time_diff_hi": float(dt.mean() + z * se_t),
                    "paired_se_ratio_vs_unpaired": float(
                        se_d
                        / math.sqrt(
                            wide_det[reference].var(ddof=1) / n + wide_det[key].var(ddof=1) / n
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)
