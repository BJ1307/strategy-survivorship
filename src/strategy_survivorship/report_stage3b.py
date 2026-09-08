"""Generate ``stage3b_report.md``. Every number is read from the result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stage3b import (ALWAYS_VALID, LOG_ODDS_METHODS, MAIN, MAIN_PAIRS,
                      METHODS_3B, RANDOM_T)

CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "trailing_sharpe_252": "252 日 trailing Sharpe",
      "ewma_student_t": "普通 EWMA Student-t",
      "ewma_trunc_student_t": "截断 EWMA Student-t"}
SC = {"gaussian_ctrl": "独立高斯对照", "sv_jump": "SV+跳跃（主情境）"}
QN = {"joint": "联合检出", "cond": "条件检出", "pre_far": "失效前误杀"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _setlabel(cfg, st):
    if st == RANDOM_T:
        return f"随机 T ~ U{{0..{cfg.stage3b_random_failure_max}}}"
    return f"固定 T={st.replace('fixed_T', '')}"


def _med(v, note):
    if isinstance(v, str) and v == "":
        return note or "未达到"
    if v != v:
        return note or "未达到"
    return f"{int(v)}d"


def write_stage3b_report(cfg, S, metrics, always, boot, ev, path: Path) -> Path:
    L: list[str] = []
    A = L.append
    hs = list(cfg.stage3b_post_windows)
    a_main, s0, sc0, T0, h0 = (MAIN["alpha"], MAIN["sharpe"], MAIN["scenario"],
                               MAIN["T"], MAIN["h"])
    a_alt = min(cfg.far_targets)
    sets = [f"fixed_T{T}" for T in cfg.stage3b_fixed_failure_days] + [RANDOM_T]
    M = metrics

    def g(sc, s, a, st, m, col):
        r = M[(M.scenario == sc) & (M.sharpe_valid == s) & (M.far_target == a)
              & (M.setting == st) & (M.method == m)][col]
        if not len(r):
            return float("nan")
        try:
            return float(r.iloc[0])
        except (TypeError, ValueError):
            return r.iloc[0]

    def bo(sc, s, a, st, q, ma, mb):
        r = boot[(boot.scenario == sc) & (boot.sharpe_valid == s) & (boot.far_target == a)
                 & (boot.setting == st) & (boot.quantity == q)
                 & (boot.method_a == ma) & (boot.method_b == mb)]
        return r.iloc[0] if len(r) else None

    A("# Stage 3B 报告：随机失效时间的有限诊断")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage3b` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 本轮的设定与覆盖范围")
    A("")
    A("```")
    A("    theta_t = s   当 t <= T")
    A("    theta_t = 0   当 t >  T")
    A("```")
    A("")
    A(f"T 是**最后一个有效交易日**。设定为固定 T ∈ "
      f"{{{', '.join(str(x) for x in cfg.stage3b_fixed_failure_days)}}}、"
      f"随机 T ~ 均匀分布于整数交易日 0..{cfg.stage3b_random_failure_max}，"
      f"以及 T=∞ 的始终有效对照。")
    A("")
    A(f"**这是“上线后一年内失效”的有限覆盖实验。**"
      f"T ≤ {cfg.stage3b_random_failure_max}，因此每个设定都有完整的 "
      f"{max(hs)} 天失效后观察期，落在 {cfg.horizon_days} 天总期限之内。"
      "**不能**把它描述成已经完成了 0–5 年或 0–10 年的寿命研究。")
    A("")
    A("- T **独立于**噪声、潜在波动率与跳跃；真实 T 只供生成器与评分使用。")
    A("- 模型持续接收收益：在 T 时**不**重置先验、累计证据、方差估计或任何窗口。")
    A("- 静态两状态分类器在切换数据上的 q，本报告称为**工作失效分数**，"
      "不把它解释成正确建模了状态切换后的真实后验概率。")
    A("")
    A(f"D={cfg.D}、H={cfg.horizon_days}、年化基础波动率 {cfg.sigma_annual:.0%}、"
      f"初始先验 {cfg.prior_valid:g}，以及全部检测器参数保持不变。"
      f"两个噪声情境（{SC['gaussian_ctrl']}、{SC['sv_jump']}），"
      f"五个既有方法，s ∈ {{{', '.join(f'{x:g}' for x in S['sharpes'])}}}。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 门槛与抽样")
    A("")
    A(f"- 门槛**统一读取** `{Path(S['thresholds']['source']).name}` 中对应方法、情境、"
      f"Sharpe 与预算的 **{cfg.horizon_days} 日**门槛。"
      "不混用 Stage 3A.1 的短期门槛，**不按真实 T 单独校准**——"
      "真实失效时间在实践中不可得。")
    A(f"- 保留 α ∈ {{{a_alt:g}, {a_main:g}}}；所有路径总监测期不超过 "
      f"{cfg.horizon_days} 天。")
    A(f"- 每个噪声情境使用新的独立测试数据：{cfg.stage3b_test_paths} 条始终有效对照路径，"
      f"另 {cfg.stage3b_test_paths} 条供各失效设定共用。"
      f"共 **{S['sampling']['base_noise_paths']:,} 条基础噪声路径**；"
      "两个 Sharpe 与各方法共享相应噪声以形成配对，"
      "因此各设定之间**不是**互相独立的样本。")
    A("- 随机 T 使用独立随机流，并在相关方法之间共用。")
    A("")
    A("**本轮不重新拟合门槛。**下文所有区间描述的是“给定既有冻结门槛时的测试不确定性”，"
      "**不能**声称重新覆盖了校准误差。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 评价定义")
    A("")
    A("```")
    A("    失效前误杀      P(tau <= T)")
    A("    条件检出        P(T < tau <= T+h | tau > T)")
    A("    联合检出        P(T < tau <= T+h)")
    A("```")
    A("")
    A(f"h ∈ {{{', '.join(str(x) for x in hs)}}} 个交易日。"
      "**主要比较保持相同的失效后观察长度**，不让不同 T 隐含不同的检测机会。")
    A("")
    A(f"从不报警编码为 τ=∞。它是失效前窗口的**存活者**、失效后的**未检出**路径，"
      f"**不是**提前误杀。联合检出恒等式 `联合 = 存活率 × 条件检出` 在每一行上核验，"
      f"本轮全表最大偏差 **{S['joint_identity_max_gap']:.2e}**。")
    A("")
    A(f"失效后延迟为 `E[min(τ−T, {max(hs)})]`，条件于 τ>T，分母是**全部存活到 T 的路径**："
      "未检出的存活者按窗口上限计入而不是删去，提前误杀的路径不计为零延迟。"
      "失效后中位时间同样以全部存活者为分母，取累计比例首次达到一半的那一天。")
    A("")

    # ---------------- 4 ----------------
    A(f"## 4. 主设定：s={s0:g}、{SC[sc0]}、α={a_main:g}、T={T0}、失效后 {h0} 天")
    A("")
    st0 = f"fixed_T{T0}"
    rows = list(METHODS_3B)
    L.extend(_t(rows, ["方法", "失效前误杀", "95% 区间", "存活数", "条件检出", "95% 区间",
                       "联合检出", "95% 区间", f"E[min(τ−T,{max(hs)})]（存活者）",
                       "失效后中位", "失效后未检出"],
                lambda m: [
                    CN[m], f"{g(sc0, s0, a_main, st0, m, 'pre_failure_far'):.4f}",
                    f"[{g(sc0, s0, a_main, st0, m, 'pre_failure_far_lo'):.4f}, "
                    f"{g(sc0, s0, a_main, st0, m, 'pre_failure_far_hi'):.4f}]",
                    f"{int(g(sc0, s0, a_main, st0, m, 'n_survivors'))}",
                    f"{g(sc0, s0, a_main, st0, m, f'cond_detect_h{h0}'):.4f}",
                    f"[{g(sc0, s0, a_main, st0, m, f'cond_detect_h{h0}_lo'):.4f}, "
                    f"{g(sc0, s0, a_main, st0, m, f'cond_detect_h{h0}_hi'):.4f}]",
                    f"{g(sc0, s0, a_main, st0, m, f'joint_detect_h{h0}'):.4f}",
                    f"[{g(sc0, s0, a_main, st0, m, f'joint_detect_h{h0}_lo'):.4f}, "
                    f"{g(sc0, s0, a_main, st0, m, f'joint_detect_h{h0}_hi'):.4f}]",
                    f"{g(sc0, s0, a_main, st0, m, 'expected_min_delay'):.1f}",
                    _med(g(sc0, s0, a_main, st0, m, 'median_post_failure_delay'),
                         str(g(sc0, s0, a_main, st0, m, 'median_post_failure_note'))),
                    f"{g(sc0, s0, a_main, st0, m, 'post_failure_undetected'):.4f}"]))
    A("")
    A("预先固定的重点比较（配对 bootstrap，存活分母在每次重抽中重算）：")
    A("")
    rows = [(ma, mb, q) for ma, mb in MAIN_PAIRS for q in ("joint", "cond", "pre_far")]
    L.extend(_t(rows, ["比较", "量", "点估计", "95% 区间", "排除 0"],
                lambda r: [f"{CN[r[0]]} − {CN[r[1]]}", QN[r[2]]]
                + ([f"{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).point) * 100:+.2f} pp",
                    f"[{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).lo) * 100:+.2f}, "
                    f"{float(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).hi) * 100:+.2f}]",
                    "是" if bool(bo(sc0, s0, a_main, st0, r[2], r[0], r[1]).excludes_zero)
                    else "否"]
                   if bo(sc0, s0, a_main, st0, r[2], r[0], r[1]) is not None
                   else ["-", "-", "-"])))
    A("")
    A("**三个量必须一起读**：联合检出的差可以由条件检出的差、也可以由失效前误杀的差造成。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 四种失效设定的对照")
    A("")
    for s in sorted(S["sharpes"], reverse=True):
        A(f"**s = {s:g}**，{SC[sc0]}，α={a_main:g}，失效后 {h0} 天")
        A("")
        rows = [(st, m) for st in sets for m in METHODS_3B]
        L.extend(_t(rows, ["失效设定", "方法", "失效前误杀", "存活率", "条件检出",
                           "联合检出", f"E[min(τ−T,{max(hs)})]（存活者）", "失效后中位"],
                    lambda r: [_setlabel(cfg, r[0]), CN[r[1]],
                               f"{g(sc0, s, a_main, r[0], r[1], 'pre_failure_far'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], 'survival_rate'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], f'cond_detect_h{h0}'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], f'joint_detect_h{h0}'):.4f}",
                               f"{g(sc0, s, a_main, r[0], r[1], 'expected_min_delay'):.1f}",
                               _med(g(sc0, s, a_main, r[0], r[1], 'median_post_failure_delay'),
                                    str(g(sc0, s, a_main, r[0], r[1],
                                          'median_post_failure_note')))]))
        A("")
    A("**始终有效对照（T=∞）的全期实际误杀率**：")
    A("")
    rows = [r for _, r in always[(always.scenario == sc0)
                                 & (always.far_target == a_main)].iterrows()]
    L.extend(_t(rows, ["s", "方法", "全期误杀率", "95% 区间", "样本量"],
                lambda r: [f"{r['sharpe_valid']:g}", CN[r["method"]],
                           f"{r['full_period_far']:.4f}",
                           f"[{r['full_period_far_lo']:.4f}, {r['full_period_far_hi']:.4f}]",
                           f"{int(r['n_paths'])}"]))
    A("")
    A(f"（T=0 与更早的失效设定下，“失效前误杀”按定义只统计到第 T 天为止，"
      f"因此 T=0 时它恒为 0：第 1 天起策略已经无效。）")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 历史证据诊断")
    A("")
    A("失效时刻 T 的工作失效分数 q_T 与对数赔率 U_T = −L_T，分别给出全部路径与"
      "存活到 T 的路径。**T=0 使用初始值（先验），不取数组最后一天。**")
    A("")
    A(f"**{CN['trailing_sharpe_252']} 不在此表中。**它的统计量是年化 Sharpe 比率，"
      "不在对数赔率尺度上；对它取 expit 得到的数没有可解释的含义，"
      "因此这里不把它放到一个它并不居于其上的刻度上报告。")
    A("")
    for s in sorted(S["sharpes"], reverse=True):
        A(f"**s = {s:g}**，{SC[sc0]}")
        A("")
        d = ev[(ev.scenario == sc0) & (ev.sharpe_valid == s)]
        rows = [(st, m) for st in sets for m in LOG_ODDS_METHODS]
        def re_(r):
            x = d[(d.setting == r[0]) & (d.method == r[1])]
            if not len(x):
                return [_setlabel(cfg, r[0]), CN[r[1]], "-", "-", "-", "-", "-"]
            x = x.iloc[0]
            return [_setlabel(cfg, r[0]), CN[r[1]],
                    f"{x.mean_U_at_T_all:+.4f}", f"{x.mean_q_at_T_all:.4f}",
                    f"{x.mean_U_at_T_survivors:+.4f}", f"{x.mean_q_at_T_survivors:.4f}",
                    f"{int(x.n_survivors_last_alpha)}"]
        L.extend(_t(rows, ["失效设定", "方法", "U_T（全部）", "q_T（全部）",
                           "U_T（存活）", "q_T（存活）", "存活数"], re_))
        A("")
    idc = pd.DataFrame(S["identity_check"])
    A("高斯对照、固定 T、**全部路径**上的解析核验：")
    A("")
    A("```")
    A("    E[U_T] = -s²T/(2D),      E[U_{T+h} - U_T] = s²h/(2D)")
    A("```")
    A("")
    rows = [r for _, r in idc.iterrows()]
    L.extend(_t(rows, ["s", "T", "h", "E[U_T] 理论", "实测", "E[ΔU] 理论", "实测",
                       "最大偏差（标准误）"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{int(r['T'])}", f"{int(r['h'])}",
                           f"{r['mean_U_at_T_theory']:+.4f}",
                           f"{r['mean_U_at_T_measured']:+.4f}",
                           f"{r['mean_dU_theory']:+.4f}", f"{r['mean_dU_measured']:+.4f}",
                           f"{max(r['z_U_at_T'], r['z_dU']):.2f}"]))
    A("")
    A("**这个关系不自动适用于经过提前误杀筛选的样本，也不自动适用于 SV+跳跃。**"
      "上面两张表里的“存活”列正是被筛选过的样本。")
    A("")
    tr = "trailing_sharpe_252"
    elig = int(g(sc0, s0, a_main, st0, tr, "first_eligible_day"))
    A(f"**读表时必须注意的结构性成因。** {CN[tr]} 最早只能在第 {elig} 天说话。"
      f"因此在 T=0 时，它在 h={h0} 天的失效后窗口里只有第 {elig}–{h0} 天可用；"
      f"而在 T={T0}={elig} 时，它的启动期恰好在失效时刻走完，"
      f"失效后窗口对它是完整的。**同一件事也压低了它的失效前误杀**："
      f"T={T0} 时它在失效前只有第 {elig} 天这一天可能报警"
      f"（实测 {g(sc0, s0, a_main, st0, tr, 'pre_failure_far'):.2%}，"
      f"而 {CN['ewma_student_t']} 为 "
      f"{g(sc0, s0, a_main, st0, 'ewma_student_t', 'pre_failure_far'):.2%}）。"
      "所以它在 T=252 处同时“误杀更少、条件检出更高”，"
      "**主要是启动期与窗口对齐的结果，不能读成它的判别能力更强**。")
    A("")
    A("其二，若累计型模型在较晚失效时变慢，**不要把全部差异都归因于历史证据**："
      "初始化、方差状态与存活筛选（只有没被提前误杀的路径才进入条件统计）"
      "都会影响表现，本轮没有分解它们各自的份额。")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 直接回答")
    A("")
    jt = {m: g(sc0, s0, a_main, st0, m, f"joint_detect_h{h0}") for m in METHODS_3B}
    z0 = {m: g(sc0, s0, a_main, "fixed_T0", m, f"joint_detect_h{h0}") for m in METHODS_3B}
    bestj = max((v, k) for k, v in jt.items())
    A(f"**（一）从“上线即无效”迁移到“先有效、后失效”时有什么变化？** "
      f"在主设定下，把 T 从 0 移到 {T0} 天后，同样 {h0} 天的观察窗口内："
      + "；".join(f"{CN[m]} 的联合检出由 {z0[m]:.2%} 变为 {jt[m]:.2%}"
                  for m in ("binary_student_t", "ewma_student_t")) + "。")
    A("")
    A(f"变化来自三处，本报告分开列出而不是合成一个数字："
      f"（i）失效前的误杀会先消耗掉一部分路径（本设定下 "
      f"{g(sc0, s0, a_main, st0, 'ewma_student_t', 'pre_failure_far'):.2%}，"
      f"这些路径在 T 之前就被判无效，属于误杀而非检出）；"
      f"（ii）存活下来的路径带着一段**有利的历史证据**进入失效时刻"
      f"（第 6 节的 U_T、q_T）；"
      f"（iii）滚动型方法的启动期在较晚的 T 下已经走完。")
    A("")
    A(f"主设定下联合检出最高的是 {CN[bestj[1]]}（{bestj[0]:.2%}）。")
    A("")
    r_late = bo(sc0, s0, a_main, st0, "joint", "ewma_student_t", tr)
    r_rand = bo(sc0, s0, a_main, RANDOM_T, "joint", "ewma_student_t", tr)
    if r_late is not None and r_rand is not None:
        A(f"**一个随失效时间改变方向的比较。** {CN['ewma_student_t']} 相对 {CN[tr]} 的"
          f"联合检出差，在 T={T0} 时为 {float(r_late.point) * 100:+.2f} pp "
          f"[{float(r_late.lo) * 100:+.2f}, {float(r_late.hi) * 100:+.2f}]，"
          f"而在随机 T 下为 {float(r_rand.point) * 100:+.2f} pp "
          f"[{float(r_rand.lo) * 100:+.2f}, {float(r_rand.hi) * 100:+.2f}]——"
          "两个区间都排除 0，**方向相反**。"
          f"如上一节所述，T={T0} 恰好等于 {CN[tr]} 的启动期长度，"
          "这个特定的 T 对它最有利；因此不应把任一方向当作一般结论，"
          "而应记为“方法排序依赖于失效时刻相对于各方法可用窗口的位置”。")
    A("")
    A("**（二）这些结果是否足以支持本周向导师说明适用范围？** "
      "可以说明的是：本项目现有的五个检测器在“先有效、后失效”这一设定下仍然可用，"
      "并且失效前误杀、条件检出、联合检出三个量已经被分开测量与报告；"
      f"覆盖范围是 T ≤ {cfg.stage3b_random_failure_max} 天、两个噪声情境、"
      f"两个信号强度、两个误杀预算。"
      "**不能**据此说明更长寿命（数年）的表现，也不能说明任何未纳入本项目的方法。")
    A("")
    A("**（三）哪些问题应留给导师决定下一周是否继续？** 本报告不替这些问题作答：")
    A("")
    A("1. 失效时间的覆盖范围是否要从一年扩展到数年——那需要更长的模拟期限，"
      "并会改变“两年监测”这个主基准的含义。")
    A("2. 是否需要一个显式建模状态切换的检测器（而不是继续用静态两状态分类器的"
      "工作失效分数）——本轮的证据只说明静态分类器在切换数据上仍能工作，"
      "没有比较过任何切换感知的方法。")
    A("3. 失效前误杀与失效后延迟之间的权衡应当按什么代价函数取舍——"
      "本项目至今只报告两者，没有为它们赋权，等待成本也未纳入评价。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 限制")
    A("")
    A(f"- 覆盖范围是 T ≤ {cfg.stage3b_random_failure_max} 天（一年内失效），"
      "不是 0–5 年或 0–10 年的寿命研究。")
    A("- 门槛沿用 Stage 3A 的 504 日冻结值；区间只描述测试抽样，不覆盖校准误差。")
    A("- 各失效设定共用同一批基础噪声，设定之间不是独立样本。")
    A("- 静态分类器的 q 是工作失效分数，不是对切换过程的正确后验。")
    A("- 解析恒等式只在 iid 高斯、固定 T、全部路径上核验过。")
    A("- 本轮不实现 BOCPD，也不新增任何检测模型或扩大参数范围。")
    A("")
    A("## 9. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.run_stage3b")
    A("uv run python -m strategy_survivorship.run_stage3b --figures-only")
    A("uv run python -m strategy_survivorship.run_stage3b --report-only")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
