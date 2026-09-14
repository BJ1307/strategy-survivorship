"""Report for the fixed-pseudo-history internal-simulation sensitivity check.

    python -m strategy_survivorship.report_market_sensitivity

Reads only the saved `market_sensitivity_*` artefacts; nothing is re-simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import market_recovery as mr
from . import market_sensitivity as ms

OUT = Path("outputs/market")


def build(out: Path) -> Path:
    res = pd.read_csv(out / "market_sensitivity_results.csv")
    proto = json.loads((out / "market_sensitivity_protocol.json").read_text())
    summ = json.loads((out / "market_sensitivity_summary.json").read_text())
    L: list[str] = []
    A = L.append

    A("# Internal-simulation sensitivity at a fixed pseudo-history\n")
    A("Generated from the saved artefacts of the sensitivity run. No simulation is "
      "re-run to produce this file.\n")

    A("## What was varied and what was held\n")
    A(f"- **Question.** {proto['question']}")
    A(f"- **Held fixed inside a history.** {', '.join(proto['held_fixed_per_history'])}.")
    A(f"- **Varied.** {proto['varied']}.")
    A(f"- **Histories.** {proto['history_choice']}.")
    A(f"- **Budget.** {len(ms.SEED_SETS)} seed sets on each of "
      f"{len(ms.HISTORIES)} histories = {ms.N_FITS} fits, fixed before the run. "
      f"{proto['budget']['stop_rule']}")
    A(f"- **Not claimed.** {proto['what_this_is_not']}\n")

    A("![seed sets](market_sensitivity_seed_sets.png)\n")
    A("## Result, per history\n")
    A("| history | truth | selections across the 5 seed sets | selected | "
      "true screen rank | eval J, calibrated − fixed | eval ES, calibrated − fixed |")
    A("|---|---|---|---|---|---|---|")
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        picks = sorted({(a, r) for a, r in zip(g.selected_A, g.selected_rho)})
        ranks = sorted(set(g.true_screen_rank))
        rk = str(ranks[0]) if len(ranks) == 1 else f"{min(ranks)}–{max(ranks)}"
        dj, de = g.delta_J_selected_minus_baseline, g.delta_energy_selected_minus_baseline
        hit = "yes" if picks[0] == (g.true_A.iloc[0], g.true_rho.iloc[0]) else "no"
        A(f"| {scen} rep {rep} | A={g.true_A.iloc[0]}, ρ={g.true_rho.iloc[0]} | "
          f"**{len(picks)}** distinct | A={picks[0][0]}, ρ={picks[0][1]} ({hit}) | "
          f"{rk} | {dj.min():+.3f} | {de.min():+.3f} |")
    A("")
    A("The two evaluation columns show a single value per history rather than a range "
      "because the selection did not move and the evaluation stream was held fixed; "
      "the identical numbers are a consequence of that, not a separate finding.\n")

    A("## Evidence that the seeds really did move\n")
    A("A check that returns 'nothing changed' is worth nothing unless the varied "
      "streams demonstrably varied. Two fields confirm they did.\n")
    A("| history | distinct final re-check training losses (of 5) | spread | "
      "true screen rank |")
    A("|---|---|---|---|")
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        t = g.training_loss_selected
        ranks = sorted(set(g.true_screen_rank))
        A(f"| {scen} rep {rep} | {t.nunique()} | "
          f"{t.min():.4f}–{t.max():.4f} (sd {t.std(ddof=1):.2e}) | "
          f"{'stable at ' + str(ranks[0]) if len(ranks) == 1 else 'moved: ' + '/'.join(map(str, ranks))} |")
    A("")
    A("Every history produced five distinct training losses, and in two of the four the "
      "truth's position in the screen moved between seed sets. The streams are wired "
      "through; the selection simply did not follow them.\n")

    A("## What the four histories say\n")
    n_hist, n_fits = len(ms.HISTORIES), int(len(res))
    n_moved = sum(len({(a, r) for a, r in zip(g.selected_A, g.selected_rho)}) > 1
                  for _, g in res.groupby(["scenario", "replicate"]))
    A(f"The selection changed in **{n_moved} of {n_hist}** histories when only the "
      f"screening and finalist streams were reseeded. Across all {n_fits} fits the "
      f"chosen cell was constant within its history.\n")
    A("Two of the four histories selected something other than the truth, and did so "
      "**identically under all five seed sets**:\n")
    for (scen, rep), g in res.groupby(["scenario", "replicate"]):
        if g.selected_is_true.iloc[0]:
            continue
        A(f"- `{scen}` rep {rep}: truth A={g.true_A.iloc[0]}, ρ={g.true_rho.iloc[0]}; "
          f"chosen A={g.selected_A.iloc[0]}, ρ={g.selected_rho.iloc[0]} five times out "
          f"of five, with the truth ranked {g.true_screen_rank.min()}"
          f"{'' if g.true_screen_rank.nunique() == 1 else '–' + str(g.true_screen_rank.max())}"
          f" in the screen.")
    A("")
    A("For those histories the miss is a property of the drawn pseudo-training sample "
      "under this objective, not of the internal simulation draw. That is the single "
      "thing this check establishes.\n")

    A("## The limits of this check\n")
    ub = 1.0 - 0.05 ** (1.0 / n_fits)
    A(f"- Zero changes in {n_fits} fits does not mean the probability of a change is "
      f"zero. Treating the five seed sets within a history as independent given that "
      f"history, the one-sided 95% upper bound on the per-fit probability of a "
      f"differing selection is **{ub:.2f}**. A reseeding effect smaller than that is "
      f"entirely compatible with what was observed.")
    A(f"- The bound is conditional on **these four histories**. They were picked by "
      f"index before the run, but four is four; a history sitting between two grid "
      f"cells could behave differently.")
    A("- The check varies the screening and finalist streams. It does **not** vary the "
      "evaluation stream, so it says nothing about how much the evaluation losses "
      "themselves would move.")
    A("- Twenty fits on four histories are not twenty market histories, and this design "
      "does **not** decompose the total selection error into sources. It bounds one "
      "component at a fixed history and leaves the rest unmeasured.\n")

    A("## How this bears on the recovery experiment\n")
    A("The recovery experiment's selections moved across its twenty pseudo-histories. "
      "This check shows that at four of those histories the internal simulation draw "
      "was not what moved them, within the resolution stated above. It does **not** "
      "identify what did: the drawn training sample, the objective's ranking of the "
      "grid, and the screen-then-rescore rule are still not separated from one "
      "another.\n")

    A("---\n")
    A(f"Run: {summ['n_fits']} fits in {summ['wall_seconds']:.0f} s. "
      f"Code identity `{summ['code_identity'].get('head', '?')[:12]}`"
      f"{', working tree dirty' if summ['code_identity'].get('dirty') else ''}. "
      f"Written {summ['written_at_utc']}.")
    A("Artefacts: `market_sensitivity_protocol.json`, `market_sensitivity_results.csv`, "
      "`market_sensitivity_summary.json`, `market_sensitivity_seed_sets.png`.")

    p = out / "market_sensitivity_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the sensitivity-check report.")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    print(build(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
