"""Generate ``stage3a1_report.md``. Every number is read from the result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .gaussian_bound import max_detection
from .stage3a1 import (CAL_HORIZON_A, MAIN, MAIN_METHOD, METHODS_3A1, SCENARIOS_3A1,
                       SECOND_PAIRS)

CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "ewma_student_t": "普通 EWMA Student-t", "ewma_trunc_student_t": "截断 EWMA Student-t"}
SC = {"gaussian_ctrl": "独立同分布高斯对照", "sv_jump": "SV+跳跃 (A=1, ρ=0.98, κ=5, λ=2/年)"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v, note):
    return f"{int(v)}d" if v == v else (note or "本窗口内未达到")


def write_stage3a1_report(cfg, S, metrics, boot, ptime, outside, ref, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    Hs = list(cfg.stage3a1_horizons)
    sharpes = sorted(S["sharpes"], reverse=True)
    s_hi, s_lo = max(sharpes), min(sharpes)
    a_main, a_alt = MAIN["alpha"], min(cfg.far_targets)
    H0, sc0, s0 = MAIN["horizon"], MAIN["scenario"], MAIN["sharpe"]
    M = metrics

    def g(sc, s, a, arm, H, m, col):
        r = M[(M.scenario == sc) & (M.sharpe_valid == s) & (M.far_target == a)
              & (M.arm == arm) & (M.cutoff_H == H) & (M.method == m)][col]
        if not len(r):
            return float("nan")
        try:
            return float(r.iloc[0])
        except (TypeError, ValueError):
            return float("nan")

    def bo(kind, sc, a, s, m, H):
        r = boot[(boot.kind == kind) & (boot.scenario == sc) & (boot.far_target == a)
                 & (boot.sharpe_valid == s) & (boot.method == m) & (boot.cutoff_H == H)]
        return r.iloc[0] if len(r) else None

    A("# Stage 3A.1 报告：监测期限诊断")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage3a1` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")
    A("**本轮只改变用于校准误杀约束的监测期限。**检测器、DGP 参数、EWMA 系数、"
      "Student-t 自由度、截断参数、候选均值传入、因果方差更新与首次越界规则全部不变；"
      "不搜索模型参数，不优化动态边界，不做随机 T 或 BOCPD。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 两种安排回答的是不同的问题")
    A("")
    A(f"- **安排 A**：按 {CAL_HORIZON_A} 天校准**一个**门槛，再读它在第 "
      f"{'、'.join(str(h) for h in Hs)} 天的累计表现。")
    A(f"- **安排 B**：分别按 H = {'、'.join(str(h) for h in Hs)} 校准门槛，"
      f"并在各自截止日评价。")
    A("")
    A("**B 的四个期限是四个独立设计的监测方案。**每一个都在自己的窗口里花掉自己的 α，"
      "不能把它们依次拼起来还声称共享同一个两年预算。因此，"
      "**B 相对 A 的早期检出提升是提前动用误杀预算的结果，"
      "不是模型从同一批数据里提取到了更多信息**——两种安排看到的收益、"
      "统计量与似然增量逐日完全相同。")
    A("")
    A(f"主设定提前写定：**s={s0:g}、{SC[sc0]}、α={a_main:g}、H={H0}**。"
      f"首要比较是 {CN[MAIN_METHOD]} 在 B 与 A 下的半年检出率差及**同期误杀率差**。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 数据、期限前缀与校准协议")
    A("")
    sp = S["sampling"]
    A(f"- 两个噪声情境（{SC['gaussian_ctrl']}、{SC['sv_jump']}），"
      f"四个方法，两个 Sharpe，四个期限，两个预算。")
    A(f"- 每个情境三块数据：{cfg.stage3a1_calibration_paths} 条校准有效路径、"
      f"{cfg.stage3a1_test_paths} 条测试有效路径、{cfg.stage3a1_test_paths} 条测试无效路径，"
      f"每条均生成 {CAL_HORIZON_A} 天。")
    A(f"- 共 **{sp['base_noise_paths']:,} 条基础噪声路径**；两个 Sharpe 与全部期限"
      f"共享各自数据块中的基础噪声，因此 {sp['return_path_evaluations']:,} 次评估"
      "**不是**同样数量的独立路径。校准与测试数据块独立。")
    A(f"- 使用独立于 Stage 3A 的新随机流（`stage3a1_calibration` / `_test` / `_bootstrap`）。")
    A("")
    A("**不同期限只截取同一条完整统计量路径的前缀。**统计量在 504 天上算一次，"
      "各期限取 `[起始日, H]` 的最小值；改变 H 不会改变此前任何一天的 EWMA 状态、"
      "收益或似然增量。测试套件对此逐点核验（见 `tests/test_stage3a1.py`）。")
    A("")
    A(f"本轮实际有 **{S['n_rules_this_round']} 条规则**"
      f"（4 方法 × 2 情境 × 2 s × 2 α × 4 H）。为保持与 Stage 3A 相同的校准缓冲强度，"
      f"仍采用每条规则 δ = 0.05/{S['delta_denominator_used']} = "
      f"{S['delta_per_cell']:.4e}，比按 128 条规则拆分更保守。")
    A("")
    for a in sorted(cfg.far_targets):
        v = S["ranks"][str(a)] if str(a) in S["ranks"] else S["ranks"][a]
        A(f"- α={a:g}：秩 **{v['buffered']}**（二项式与 Beta 两路一致：{v['beta_route']}）。")
    A("")
    A(f"**范围声明**：{S['calibration_scope']}。这不是跨全部历史阶段的统一保证。"
      "全部门槛在正式测试前冻结并写入 `stage3a1_thresholds.csv`。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 主结果：专门为半年校准之后")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[sc0]}，α={a_main:g}")
        A("")
        rows = [(m, arm, H) for m in METHODS_3A1 for H in Hs for arm in ("A", "B")]
        L.extend(_t(rows, ["方法", "安排", "截止日 H", "检出率", "95% 区间", "累计误杀率",
                           "95% 区间", "E[min(τ,H)]", "中位首次报警", "未检出比例",
                           "门槛", "工作概率门槛"],
                    lambda r: [
                        CN[r[0]], r[1], str(r[2]),
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'detect'):.4f}",
                        f"[{g(sc0, s, a_main, r[1], r[2], r[0], 'detect_lo'):.4f}, "
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'detect_hi'):.4f}]",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'far'):.4f}",
                        f"[{g(sc0, s, a_main, r[1], r[2], r[0], 'far_lo'):.4f}, "
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'far_hi'):.4f}]",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'expected_min_tau_H'):.1f}",
                        _med(g(sc0, s, a_main, r[1], r[2], r[0], 'median_first_alarm'),
                             f"H={r[2]} 内未达到"),
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'undetected_at_H'):.4f}",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'threshold'):.4f}",
                        f"{g(sc0, s, a_main, r[1], r[2], r[0], 'working_prob_threshold'):.4f}"]))
        A("")
    A(f"样本量：每格有效路径 {cfg.stage3a1_test_paths}、无效路径 {cfg.stage3a1_test_paths}。"
      "区间为 Wilson 区间。")
    A("")
    A("**`E[min(τ,H)]` 的上限随 H 变化，不同 H 的这个数不可直接比较**："
      "H=63 的截断均值最大只能是 63 天。跨期限比较“更快”是没有意义的；"
      "本报告的所有时间比较都在**同一截止日**上进行。")
    A("")
    A("工作概率门槛指的是同一条规则写成 `q > q*` 时的 `q*`（`q = expit(−L)`），"
      "只对贝叶斯统计量有这个读法。")
    A("")

    # ---------------- 4 ----------------
    A("## 4. 首要比较：B − A")
    A("")
    A(f"{cfg.stage3a1_bootstrap_reps} 次配对 bootstrap，同时重抽校准与测试路径；"
      "同一次重抽在两个安排、各期限与相关方法之间共用路径索引，各自重算门槛。"
      "**区间是逐项区间**，校准缓冲不覆盖这些差。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[sc0]}，α={a_main:g}")
        A("")
        rows = [(m, H) for m in METHODS_3A1 for H in Hs]
        def rd(r):
            m, H = r
            d = bo("detect_B_minus_A", sc0, a_main, s, m, H)
            f = bo("far_B_minus_A", sc0, a_main, s, m, H)
            pt = bo("detect_B_minus_A", sc0, a_main, s, m, H)
            tm = ptime[(ptime.scenario == sc0) & (ptime.sharpe_valid == s)
                       & (ptime.far_target == a_main) & (ptime.method == m)
                       & (ptime.cutoff_H == H)]
            return [CN[m], str(H),
                    f"{float(d.point) * 100:+.2f} pp" if d is not None else "-",
                    f"[{float(d.lo) * 100:+.2f}, {float(d.hi) * 100:+.2f}]" if d is not None else "-",
                    f"{float(f.point) * 100:+.2f} pp" if f is not None else "-",
                    f"[{float(f.lo) * 100:+.2f}, {float(f.hi) * 100:+.2f}]" if f is not None else "-",
                    f"{float(tm.trunc_time_diff_days.iloc[0]):+.2f}" if len(tm) else "-",
                    f"[{float(tm.lo.iloc[0]):+.2f}, {float(tm.hi.iloc[0]):+.2f}]" if len(tm) else "-"]
        L.extend(_t(rows, ["方法", "H", "检出率差", "95% 区间", "同期误杀率差", "95% 区间",
                           "配对 E[min(τ,H)] 差（天）", "95% 区间"], rd))
        A("")
    d0 = bo("detect_B_minus_A", sc0, a_main, s0, MAIN_METHOD, H0)
    f0 = bo("far_B_minus_A", sc0, a_main, s0, MAIN_METHOD, H0)
    A(f"**首要比较（{CN[MAIN_METHOD]}，s={s0:g}，H={H0}，α={a_main:g}）**："
      f"检出率差 {float(d0.point) * 100:+.2f} pp "
      f"[{float(d0.lo) * 100:+.2f}, {float(d0.hi) * 100:+.2f}]，"
      f"同期误杀率差 {float(f0.point) * 100:+.2f} pp "
      f"[{float(f0.lo) * 100:+.2f}, {float(f0.hi) * 100:+.2f}]。"
      "两个数必须一起读。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 第二项比较：各期限自校准后，方差自适应的增益是否仍在")
    A("")
    rows = []
    for s in sharpes:
        for ma, mb in SECOND_PAIRS:
            for H in Hs:
                r = bo(f"gain_B_{ma}_minus_{mb}", sc0, a_main, s, f"{ma}|{mb}", H)
                if r is not None:
                    rows.append((s, ma, mb, H, r))
    L.extend(_t(rows, ["s", "比较（均按该期限自校准）", "H", "检出率差", "95% 区间", "排除 0"],
                lambda r: [f"{r[0]:g}", f"{CN[r[1]]} − {CN[r[2]]}", str(r[3]),
                           f"{float(r[4].point) * 100:+.2f} pp",
                           f"[{float(r[4].lo) * 100:+.2f}, {float(r[4].hi) * 100:+.2f}]",
                           "是" if bool(r[4].excludes_zero) else "否"]))
    A("")
    A("截断的额外增益一并报告；**本轮不据此搜索截断参数**。")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 期限外诊断：短期门槛继续用下去会怎样")
    A("")
    A("把按 H=63、126、252 校准的门槛继续用到第 504 天，"
      "看实际累计误杀率变成多少。**短期门槛不自动继承两年保证**："
      "它得到的预算是针对自己那个窗口的。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[sc0]}，各自窗口的预算 α={a_main:g}")
        A("")
        d = outside[(outside.scenario == sc0) & (outside.sharpe_valid == s)
                    & (outside.far_target == a_main)]
        rows = [r for _, r in d.sort_values(["threshold_calibrated_over", "method"]).iterrows()]
        L.extend(_t(rows, ["门槛校准期限", "方法", "自身窗口内误杀"]
                    + [f"延用到 d{h} 的累计误杀" for h in Hs],
                    lambda r: [str(int(r["threshold_calibrated_over"])), CN[r["method"]],
                               f"{r['far_at_own_H']:.4f}"]
                    + [f"{r[f'far_extended_to_d{h}']:.4f}" for h in Hs]))
        A("")

    # ---------------- 7 ----------------
    A("## 7. 高斯参照")
    A("")
    A("```")
    A("    D_max(h, α) = Φ( s·√h − Φ⁻¹(1−α) ),   h 以年为单位")
    A("```")
    A("")
    A("**只适用于已知方差、独立同分布高斯、两个简单假设的情境。**"
      "它在每个截止时间上分别计算，每次都把全部 α 用在那一个截止日；"
      "因此它不是一条可以同时达到的监测曲线。"
      "组合情境的剩余差距**不能**叫作信息上限——本项目没有为组合情境推导过任何上界。")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h（年）", "天数", "该截止日的最强检验检出率"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:.4g}", f"{int(r['days'])}",
                           f"{r['max_detection']:.4f}"]))
    A("")
    gc = [(g("gaussian_ctrl", s0, a_main, "B", H0, m, "detect"), m) for m in METHODS_3A1]
    best = max(gc)
    A(f"高斯对照、s={s0:g}、H={H0}、α={a_main:g}：安排 B 下最好的方法检出 "
      f"{best[0]:.2%}（{CN[best[1]]}），该截止日的参照为 "
      f"{max_detection(s0, H0 / cfg.D, a_main):.2%}。"
      "代入经验误杀率所得的比较仍称**估计参照**；"
      "“没有越界”只作一致性检查，不构成对实现的数学验证。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 直接回答")
    A("")
    dB = g(sc0, s0, a_main, "B", H0, MAIN_METHOD, "detect")
    dA = g(sc0, s0, a_main, "A", H0, MAIN_METHOD, "detect")
    fB = g(sc0, s0, a_main, "B", H0, MAIN_METHOD, "far")
    fA = g(sc0, s0, a_main, "A", H0, MAIN_METHOD, "far")
    bestB = max((g(sc0, s0, a_main, "B", H0, m, "detect"), m) for m in METHODS_3A1)
    A(f"**（一）专门为半年监测校准后，弱信号能检出多少？** "
      f"s={s0:g}、{SC[sc0]}、α={a_main:g}、H={H0} 天，安排 B 下 {CN[MAIN_METHOD]} 检出 "
      f"{dB:.2%}，最好的方法是 {CN[bestB[1]]} 的 {bestB[0]:.2%}。")
    A("")
    A(f"**（二）相对当前两年规则的半年表现，增加了多少检出、多少同期误杀？** "
      f"同一方法在安排 A 下的半年检出为 {dA:.2%}，误杀 {fA:.2%}；"
      f"安排 B 为 {dB:.2%} / {fB:.2%}。"
      f"即多检出 {(dB - dA) * 100:+.2f} pp，同期多误杀 {(fB - fA) * 100:+.2f} pp。")
    A("")
    A("**这一提升来自监测期限与早期误杀安排的改变，不是模型获得了新的收益数据。**"
      "两种安排的输入逐日完全相同；B 把原本留给后一年半的误杀预算提前花在了前半年。")
    A("")
    dA504 = g(sc0, s0, a_main, "A", CAL_HORIZON_A, MAIN_METHOD, "detect")
    fA504 = g(sc0, s0, a_main, "A", CAL_HORIZON_A, MAIN_METHOD, "far")
    A("**把“花掉的预算”对齐之后再看一次。**同一个方法：")
    A("")
    L.extend(_t([("B", H0, dB, fB), ("A", CAL_HORIZON_A, dA504, fA504)],
                ["安排", "窗口（天）", "检出率", "已花掉的累计误杀"],
                lambda r: [r[0], str(r[1]), f"{r[2]:.2%}", f"{r[3]:.2%}"]))
    A("")
    A(f"两者花掉的误杀几乎相同（{fB:.2%} 对 {fA504:.2%}），"
      f"但两年窗口换来的检出是半年窗口的 {dA504 / dB:.2f} 倍。"
      "**在同样的误杀代价下，把窗口拉长比把预算提前花掉更有效。**"
      "这一点单看 B−A 的正差是看不出来的。")
    A("")
    rows2 = [r for r in rows if False]
    sec = [bo(f"gain_B_{ma}_minus_{mb}", sc0, a_main, s0, f"{ma}|{mb}", H0)
           for ma, mb in SECOND_PAIRS]
    if sec[0] is not None:
        A(f"**（三）方差自适应的收益是否仍在？** 在按 H={H0} 自校准之后，"
          f"{CN[SECOND_PAIRS[0][0]]} − {CN[SECOND_PAIRS[0][1]]} 为 "
          f"{float(sec[0].point) * 100:+.2f} pp "
          f"[{float(sec[0].lo) * 100:+.2f}, {float(sec[0].hi) * 100:+.2f}]，"
          f"区间{'排除' if bool(sec[0].excludes_zero) else '覆盖'} 0。"
          + (f"截断的额外增益为 {float(sec[1].point) * 100:+.2f} pp "
             f"[{float(sec[1].lo) * 100:+.2f}, {float(sec[1].hi) * 100:+.2f}]。"
             if sec[1] is not None else ""))
        A("")
    A(f"**（四）能否支持“半年内发现大多数无效策略”？** 在本轮测到的设定里不能："
      f"s={s0:g}、α={a_main:g}、H={H0} 下最好的方法只有 {bestB[0]:.1%}，"
      "远不到多数。要接近“多数”，需要同时满足更强的信号、更宽的误杀预算、"
      "或者更长的窗口；本轮只在两个 s、两个 α、四个 H 上有测量，不外推。")
    A("")
    A(f"**（五）对两年结果的理解有什么改变？** 两年检出率偏低**不能**归因于"
      "“门槛是按两年校准的”这一件事："
      f"即便专门为半年重新校准，半年检出也只从 {dA:.2%} 升到 {dB:.2%}，"
      f"而这已经动用了 {fB:.2%} 的误杀（安排 A 在同一天只用了 {fA:.2%}）。"
      "期限的重新设定改变的是**预算的时间分配**，不改变数据里的信息量。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 限制")
    A("")
    A("- B 的四个期限是四个独立方案，不能拼接成一个共享两年预算的计划。")
    A("- 所有区间都是逐项区间，未做多重比较校正；校准缓冲只约束门槛的误杀，"
      "不覆盖检出率差。")
    A(f"- 只测了 s ∈ {{{s_hi:g}, {s_lo:g}}}、α ∈ {{{a_alt:g}, {a_main:g}}}、"
      f"H ∈ {{{', '.join(str(h) for h in Hs)}}}，不外推到其它取值。")
    A("- 不同 H 的 `E[min(τ,H)]` 上限不同，跨期限不可比。")
    A("- 高斯参照只适用于 iid 高斯对照；组合情境没有已知上界。")
    A("- 本轮不含 252 日 rolling 方法：把它改成短窗口会改变模型，超出本轮范围。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.run_stage3a1")
    A("uv run python -m strategy_survivorship.run_stage3a1 --figures-only")
    A("uv run python -m strategy_survivorship.run_stage3a1 --report-only")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。`--figures-only` 与 `--report-only` "
      "只读取已生成的 CSV，不重跑模拟。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
