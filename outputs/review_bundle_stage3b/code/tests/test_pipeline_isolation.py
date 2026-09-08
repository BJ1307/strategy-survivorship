"""End-to-end guarantees: data isolation, frozen thresholds, self-consistent output."""

import json

import numpy as np
import pandas as pd
import pytest

from strategy_survivorship import run_stage1, simulate
from strategy_survivorship.detectors import DETECTORS_BY_KEY


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("stage1")
    assert run_stage1.main(["--smoke", "--out", str(out)]) == 0
    return {
        "dir": out,
        "metrics": pd.read_csv(out / "stage1_metrics.csv"),
        "passages": pd.read_csv(out / "stage1_first_passages.csv"),
        "curves": pd.read_csv(out / "stage1_curves.csv"),
        "summary": json.loads((out / "stage1_summary.json").read_text()),
    }


def _count_dgp_draws(tmp_path, monkeypatch, extra_args):
    calls = []
    original = simulate.simulate_returns

    def counting(seed_seq, n_paths, n_days, sharpe, cfg):
        calls.append((n_paths, n_days, sharpe))
        return original(seed_seq, n_paths, n_days, sharpe, cfg)

    monkeypatch.setattr(simulate, "simulate_returns", counting)
    args = ["--smoke", "--out", str(tmp_path), "--no-figures"] + extra_args
    assert run_stage1.main(args) == 0
    return calls


def test_every_detector_scores_the_same_simulated_paths(tmp_path, monkeypatch):
    """The benchmark must sample the DGP exactly 4 times: 3 blocks + 1 diagnostic path.

    A per-detector redraw would show up here immediately as extra calls.  The
    replication study is switched off so this counts the benchmark alone.
    """
    calls = _count_dgp_draws(tmp_path, monkeypatch, ["--replications", "0"])
    assert len(calls) == 4, calls
    assert sum(1 for c in calls if c[0] > 1) == 3  # the three benchmark blocks
    assert sum(1 for c in calls if c[0] == 1) == 1  # the single diagnostic path


def test_replication_study_draws_whole_blocks_not_per_detector(tmp_path, monkeypatch):
    """Each replication redraws the three blocks once -- not once per detector.

    The replication study deliberately re-samples the DGP (that is the point), so
    the guard here is that it costs exactly 3 draws per replication regardless of
    how many detectors are being calibrated.
    """
    n_rep = 3
    calls = _count_dgp_draws(tmp_path, monkeypatch, ["--replications", str(n_rep)])
    assert len(calls) == 4 + 3 * n_rep, calls
    assert sum(1 for c in calls if c[0] == 1) == 1  # still one diagnostic path


def test_calibration_paths_never_appear_in_the_evaluation(run):
    assert set(run["passages"]["path_set"]) == {"test_valid", "test_invalid"}
    assert set(run["curves"]["path_set"]) == {"test_valid", "test_invalid"}
    n_cal = run["summary"]["sample_sizes"]["calibration_valid_paths"]
    assert run["summary"]["thresholds"]["binary_gaussian|alpha=0.05"]["threshold"] < 0
    assert n_cal > 0


def test_true_state_labels_are_consistent(run):
    p = run["passages"]
    assert set(p.loc[p.path_set == "test_valid", "true_state"]) == {"valid"}
    assert set(p.loc[p.path_set == "test_invalid", "true_state"]) == {"invalid"}
    assert set(p.loc[p.true_state == "valid", "true_sharpe_annual"]) == {1.0}
    assert set(p.loc[p.true_state == "invalid", "true_sharpe_annual"]) == {0.0}


def test_one_threshold_per_detector_and_target_across_both_test_sets(run):
    """A threshold picked on calibration data must not vary by test set or state."""
    g = run["passages"].groupby(["detector", "far_target"])["threshold"].nunique()
    assert (g == 1).all()
    for (det, alpha), thr in run["passages"].groupby(["detector", "far_target"])["threshold"].first().items():
        key = f"{det}|alpha={alpha:g}"
        assert thr == pytest.approx(run["summary"]["thresholds"][key]["threshold"])
        row = run["metrics"].query("detector == @det and far_target == @alpha").iloc[0]
        assert row["threshold"] == pytest.approx(thr)


def test_no_detector_alarms_before_it_is_eligible(run):
    p, curves = run["passages"], run["curves"]
    fired = p[p["alarmed"]]
    assert (fired["first_alarm_day"] >= fired["first_eligible_day"]).all()
    for key in DETECTORS_BY_KEY:
        eligible = int(p.query("detector == @key")["first_eligible_day"].iloc[0])
        early = curves.query("detector == @key and day < @eligible")
        assert len(early) == (eligible - 1) * 4  # 2 targets x 2 test sets
        assert (early["cumulative_alarm_rate"] == 0.0).all()
    # the 252-day rolling detectors really are the silent ones
    silent = {k for k in DETECTORS_BY_KEY
              if int(p.query("detector == @k")["first_eligible_day"].iloc[0]) > 1}
    assert silent == {"trailing_sharpe_252", "known_vol_rolling_252"}


def test_censored_paths_are_distinguishable_from_last_day_alarms(run):
    p = run["passages"]
    H = run["summary"]["config"]["horizon_days"]
    censored = p[~p["alarmed"]]
    assert (censored["first_alarm_day"] == -1).all()
    assert (censored["truncated_days"] == H).all()
    last_day = p[p["alarmed"] & (p["first_alarm_day"] == H)]
    assert (last_day["truncated_days"] == H).all()
    if len(last_day):  # both categories share min(tau,H)=H but differ in `alarmed`
        assert set(p[p["truncated_days"] == H]["alarmed"]) == {True, False}


def test_metrics_agree_with_the_per_path_table(run):
    """Every headline number must be recomputable from stage1_first_passages.csv."""
    p, m = run["passages"], run["metrics"]
    H = run["summary"]["config"]["horizon_days"]
    for _, row in m.iterrows():
        if row["detector"] == "random_closure":
            continue
        sel = p.query("detector == @row.detector and far_target == @row.far_target")
        valid = sel.query("path_set == 'test_valid'")
        invalid = sel.query("path_set == 'test_invalid'")
        for day in (126, 252, 504):
            hit_v = ((valid["first_alarm_day"] > 0) & (valid["first_alarm_day"] <= day)).mean()
            hit_i = ((invalid["first_alarm_day"] > 0) & (invalid["first_alarm_day"] <= day)).mean()
            assert row[f"far_d{day}"] == pytest.approx(hit_v)
            assert row[f"detect_d{day}"] == pytest.approx(hit_i)
        assert row["undetected_at_H"] == pytest.approx(1.0 - row["detect_d504"])
        assert row["trunc_mean_detect_days"] == pytest.approx(invalid["truncated_days"].mean())
        assert row["trunc_mean_detect_years"] == pytest.approx(row["trunc_mean_detect_days"] / 252)
        assert 0 < row["trunc_mean_detect_days"] <= H


def test_curves_agree_with_the_metrics(run):
    c, m = run["curves"], run["metrics"]
    for _, row in m.iterrows():
        sub = c.query(
            "detector == @row.detector and far_target == @row.far_target "
            "and path_set == 'test_invalid'"
        ).sort_values("day")
        assert np.all(np.diff(sub["cumulative_alarm_rate"]) >= -1e-12)  # monotone
        assert sub.iloc[-1]["cumulative_alarm_rate"] == pytest.approx(row["detect_d504"])


def test_calibration_far_respects_every_budget(run):
    m = run["metrics"].dropna(subset=["calibration_far"])
    assert (m["calibration_far"] <= m["far_target"] + 1e-12).all()


def test_median_is_reported_as_unreached_when_it_is(run):
    for _, row in run["metrics"].iterrows():
        if pd.isna(row["median_detect_days"]) or row["median_detect_days"] == "":
            assert row["detect_d504"] < 0.5
            assert "not reached" in str(row["median_detect_note"])
        else:
            assert row["detect_d504"] >= 0.5


def test_all_declared_outputs_exist_and_are_non_trivial(run):
    d = run["dir"]
    expected = [
        "stage1_report.md",
        "stage1_metrics.csv",
        "stage1_summary.json",
        "stage1_first_passages.csv",
        "stage1_curves.csv",
        "stage1_diagnostic_traces.csv",
        "run_metadata.json",
    ]
    for name in expected:
        assert (d / name).stat().st_size > 200, name
    for name in (
        "fig1_example_paths.png",
        "fig2_operating_curves.png",
        "fig3_detection_time.png",
        "fig4_shock_diagnostic.png",
    ):
        assert (d / "figures" / name).stat().st_size > 20_000, name


def test_pipeline_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        assert run_stage1.main(["--smoke", "--out", str(out), "--no-figures"]) == 0
    left = pd.read_csv(a / "stage1_metrics.csv")
    right = pd.read_csv(b / "stage1_metrics.csv")
    pd.testing.assert_frame_equal(left, right)


def test_diagnostic_traces_isolate_the_shock(run):
    d = pd.read_csv(run["dir"] / "stage1_diagnostic_traces.csv")
    cfg = run["summary"]["config"]
    wide = d.pivot(index="day", columns="variant", values="return")
    diff = (wide["shock_plus"] - wide["base"]).abs()
    assert (diff > 0).sum() == 1
    assert diff.idxmax() == cfg["shock_day"]
    assert diff.max() == pytest.approx(cfg["shock_in_daily_sigma"] * cfg["derived"]["sigma_daily"])
    # The Gaussian increment is affine in z, so a +k*sigma_d shock displaces the
    # terminal log-odds by exactly k/sqrt(D) and that displacement is permanent.
    end = d[d["day"] == cfg["horizon_days"]].set_index("variant")
    k, sqrt_D = cfg["shock_in_daily_sigma"], np.sqrt(cfg["trading_days_per_year"])
    for variant, sign in (("shock_plus", +1.0), ("shock_minus", -1.0)):
        g_shift = end.loc[variant, "gaussian_log_odds"] - end.loc["base", "gaussian_log_odds"]
        t_shift = end.loc[variant, "student_t_log_odds"] - end.loc["base", "student_t_log_odds"]
        assert g_shift == pytest.approx(sign * k / sqrt_D, rel=1e-10)
        # The Student-t influence function redescends, so the shocked day carries
        # LESS evidence than the unshocked one. The displacement is therefore
        # heavily damped, and its sign is not even guaranteed to follow the shock.
        assert abs(t_shift) < 0.25 * abs(g_shift)
