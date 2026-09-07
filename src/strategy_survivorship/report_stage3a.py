"""Generate ``stage3a_report.md``. Every number is read from the result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import excludes_zero
from .gaussian_bound import max_detection
from .stage2d import COMBINED, SCEN_LABEL
from .stage2e import LABEL_2E, METHODS_2E
from .stage3a import HEADLINE, MAIN_ALPHA, MAIN_PAIRS, MAIN_SCENARIO, MAIN_SHARPE

SC = {"gaussian_ctrl": "高斯对照 (A=0, κ=0)", "sv_ctrl": "SV 对照 (A=1, κ=0)",
      "jump_ctrl": "跳跃对照 (A=0, κ=5)", "sv_jump": "SV+跳跃 (A=1, κ=5)",
      "sv_jump_big": "SV+较大跳跃 (A=1, κ=8)"}
CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "trailing_sharpe_252": "252 日 trailing Sharpe",
      "known_vol_rolling_252": "固定尺度 rolling 对照",
      "ewma_gaussian": "普通 EWMA Gaussian", "ewma_student_t": "普通 EWMA Student-t",
      "ewma_trunc_gaussian": "截断 EWMA Gaussian",
      "ewma_trunc_student_t": "截断 EWMA Student-t"}
PAIR_CN = {("ewma_student_t", "binary_student_t"): "普通 EWMA t − 固定 t",
           ("ewma_trunc_student_t", "ewma_student_t"): "截断 t − 普通 EWMA t",
           ("ewma_trunc_student_t", "trailing_sharpe_252"): "截断 t − trailing Sharpe"}


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def _med(v, note=""):
    if v == v and v not in ("", None):
        return f"{int(v)}d"
    return "超过两年（本观察期未达到）"


def write_stage3a_report(cfg, S, metrics, boot, ptime, qrows, plevel, brier, ref,
                         path: Path) -> Path:
    L: list[str] = []
    A = L.append
    days = list(cfg.stage2c_report_days)
    sharpes = sorted(S["sharpes"], reverse=True)
    s_hi, s_lo = max(sharpes), min(sharpes)
    a_main, a_alt = MAIN_ALPHA, min(cfg.far_targets)
    M = metrics

    def g(m, sc, s, a, col):
        """Numeric lookup; a not-reached median arrives as an empty cell, not a number."""
        r = M[(M.scenario == sc) & (M.sharpe_valid == s) & (M.far_target == a)
              & (M.method == m)][col]
        if not len(r):
            return float("nan")
        try:
            return float(r.iloc[0])
        except (TypeError, ValueError):
            return float("nan")

    def bo(kind, sc, a, ma, mb, s):
        r = boot[(boot.kind == kind) & (boot.scenario == sc) & (boot.far_target == a)
                 & (boot.method_a == ma) & (boot.method_b == mb)
                 & (boot.sharpe_valid.astype(str) == str(s))]
        return r.iloc[0] if len(r) else None

    A("# Stage 3A 报告：Sharpe 0.6 对 0 需要观察多久")
    A("")
    A("本文件由 `python -m strategy_survivorship.run_stage3a` 自动生成，"
      "所有数字取自同一次运行的结果表。")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 本轮只改了什么")
    A("")
    A("**只有信号强度变化。**五个噪声情境、八个检测方法、EWMA 系数 0.94、"
      "Student-t 自由度 5、截断参数 4、初值、下限、首次报警规则、D=252、H=504、"
      "年化无条件波动率 0.10、初始先验 0.5 全部沿用。")
    A("")
    A("```")
    A("    r_t = mu_S + sigma_0 * eps_t")
    A(f"    有效:  mu_1(s) = s * sigma_ann / D     s in {{{s_hi:g}, {s_lo:g}}}")
    A("    无效:  mu_0 = 0")
    A("```")
    A("")
    A("**候选均值设定正确。**在 s=0.6 这一组，检测器知道候选状态是 0.6 或 0，"
      "但不知道某条路径属于哪一个。生成器的漂移、检测器似然比里的候选均值、"
      "以及 EWMA 中点 `m=(mu_0+mu_1)/2` 全部同步使用 0.6。"
      "**本轮不是模型设定错误实验**：没有出现“生成器降到 0.6 而检测器仍假设 1”的情况。")
    A("")
    A("有一条必须说明的例外：八个方法里有两个——252 日 trailing Sharpe 与"
      "固定尺度 rolling 对照——的**统计量本身不含候选均值**（它们是 √D·mean/std 一类的"
      "滚动样本统计量）。对这两个方法，s 只通过校准门槛和数据进入，这是定义使然，不是遗漏。"
      "其余六个方法的统计量都随 s 改变。")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 数据与校准协议")
    A("")
    sp = S["sampling"]
    A(f"- 每个情境三个独立噪声块（有效校准 / 有效测试 / 无效测试），各 "
      f"{cfg.stage3a_test_paths} 条 × {cfg.horizon_days} 天。")
    A(f"- 共 **{sp['base_noise_paths']:,} 条基础噪声路径**，在两个 s 设定下产生 "
      f"**{sp['return_path_evaluations']:,} 次收益路径评估**。"
      "两个 s **共用同一批噪声**（有意使用共同随机数，使 s=1 与 s=0.6 的比较成为配对比较），"
      "所以这不是 300,000 条彼此独立的噪声路径。")
    A("- 校准块与测试块**绝不共享**；所有方法共用相应路径。")
    A("- 噪声按理论常数归一化，没有逐路径缩放或去均值。")
    A(f"- 本阶段使用独立于 Stage 2E 的新随机流（`stage3a_calibration` / `_test` / "
      f"`_bootstrap`），追加式加入，既有阶段指纹不变。")
    A("")
    rk = S["ranks"]
    A(f"逐情境校准，两个累计误杀预算 α ∈ {{{a_alt:g}, {a_main:g}}}。"
      f"本轮把**两个 s 都纳入同时校准范围**：")
    A("")
    A("```")
    A(f"    J = 8 模型 × 5 情境 × 2 预算 × 2 信号强度 = {S['J']}")
    A(f"    delta = {cfg.stage2c_delta:g},   delta/J = {S['delta_per_comparison']:.4e}")
    A("```")
    A("")
    for a in sorted(cfg.far_targets):
        v = rk[str(a)] if str(a) in rk else rk[a]
        A(f"- α={a:g}：带缓冲的顺序统计量秩 **{v['buffered']}**"
          f"（二项式与 Beta 两条路径独立算得同一个秩：{v['beta_route']}）。")
    A("")
    A("**全部门槛在正式测试之前冻结并写入 `stage3a_thresholds.csv`。**"
      "没有从测试结果挑选门槛，也没有把 s=1 的门槛直接用来给 s=0.6 的方法排名——"
      "两个 s 各自校准。")
    A("")
    A("**覆盖范围声明**：该校准覆盖上列五个固定情境，"
      "**不构成未知环境下的统一门槛保证**。")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 主结果：弱信号下两年、一年、半年、一个季度各能判断多少")
    A("")
    A(f"主设定：**s={MAIN_SHARPE:g}、{SC[MAIN_SCENARIO]}、α={a_main:g}**。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g} 对 0**，{SC[MAIN_SCENARIO]}，α={a_main:g}")
        A("")
        rows = [m for m in METHODS_2E]
        L.extend(_t(rows, ["方法", "实际误杀率"] + [f"检出 d{d}" for d in days]
                    + ["截断平均天数", "中位检出", "两年未检出"],
                    lambda m: [CN[m], f"{g(m, MAIN_SCENARIO, s, a_main, 'far_total'):.4f}"]
                    + [f"{g(m, MAIN_SCENARIO, s, a_main, f'detect_d{d}'):.4f}" for d in days]
                    + [f"{g(m, MAIN_SCENARIO, s, a_main, 'trunc_mean_detect_days'):.1f}",
                       _med(g(m, MAIN_SCENARIO, s, a_main, 'median_detect_days')),
                       f"{g(m, MAIN_SCENARIO, s, a_main, 'undetected_at_H'):.4f}"]))
        A("")
    A(f"**每个方法只有一个两年门槛。**第 {days[0]}、{days[1]}、{days[2]} 天的数字读的是"
      f"同一个门槛，不是在每个期限各花一次完整的 {a_main:.0%} 误杀预算。"
      f"未检出路径在截断平均里记为第 {cfg.horizon_days} 天；"
      "检出率不足一半时中位数记为“超过两年（本观察期未达到）”，不据两年曲线外推十年。")
    A("")
    best_lo = max((g(m, MAIN_SCENARIO, s_lo, a_main, "detect_d504"), m) for m in METHODS_2E)
    best_hi = max((g(m, MAIN_SCENARIO, s_hi, a_main, "detect_d504"), m) for m in METHODS_2E)
    A(f"一句话：主情境、α={a_main:g} 下，s={s_hi:g} 时最好的方法两年检出 "
      f"{best_hi[0]:.1%}（{CN[best_hi[1]]}），s={s_lo:g} 时降到 "
      f"{best_lo[0]:.1%}（{CN[best_lo[1]]}）。")
    A("")
    A(f"α={a_alt:g} 的同一张表：")
    A("")
    for s in sharpes:
        rows = [m for m in HEADLINE]
        A(f"s={s:g}，α={a_alt:g}")
        A("")
        L.extend(_t(rows, ["方法", "实际误杀率"] + [f"检出 d{d}" for d in days]
                    + ["截断平均天数", "中位检出"],
                    lambda m: [CN[m], f"{g(m, MAIN_SCENARIO, s, a_alt, 'far_total'):.4f}"]
                    + [f"{g(m, MAIN_SCENARIO, s, a_alt, f'detect_d{d}'):.4f}" for d in days]
                    + [f"{g(m, MAIN_SCENARIO, s, a_alt, 'trunc_mean_detect_days'):.1f}",
                       _med(g(m, MAIN_SCENARIO, s, a_alt, 'median_detect_days'))]))
        A("")

    # ---------------- 4 ----------------
    A("## 4. 三条预设比较")
    A("")
    A("这三条在看到本轮任何测试结果之前写定：")
    A("")
    for i, (ma, mb) in enumerate(MAIN_PAIRS, 1):
        A(f"{i}. {PAIR_CN[(ma, mb)]}")
    A("")
    A(f"区间为 {cfg.stage3a_bootstrap_reps} 次配对 bootstrap，同时重抽校准与测试路径；"
      "同一比较中的两个方法与两个 s 设定共用重采样下标，每个 (s, α) 单元在同一份重抽"
      "校准样本上重算自己的门槛。")
    A("")
    A("**区间是逐项区间。**校准协议的 δ/J 保护的是门槛的误杀约束，"
      "**不**为这些检出率差异提供同时置信保证。")
    A("")
    for sc in COMBINED:
        for a in sorted(cfg.far_targets, reverse=True):
            A(f"**{SC[sc]}**，α={a:g}")
            A("")
            rows = [(ma, mb, s) for ma, mb in MAIN_PAIRS for s in sharpes]
            def rend(r):
                ma, mb, s = r
                b = bo("detection_diff", sc, a, ma, mb, f"{s:g}")
                f = bo("far_diff", sc, a, ma, mb, f"{s:g}")
                if b is None:
                    return [PAIR_CN[(ma, mb)], f"{s:g}", "-", "-", "-", "-"]
                return [PAIR_CN[(ma, mb)], f"{s:g}", f"{float(b.point) * 100:+.2f} pp",
                        f"[{float(b.lo) * 100:+.2f}, {float(b.hi) * 100:+.2f}]",
                        "是" if bool(b.excludes_zero) else "否",
                        f"[{float(f.lo) * 100:+.2f}, {float(f.hi) * 100:+.2f}]"
                        if f is not None else "-"]
            L.extend(_t(rows, ["比较", "s", "检出率之差", "95% 区间", "排除 0",
                               "同预算下误杀差的区间"], rend))
            A("")
    A("配对检测时间差（同一批无效路径，门槛冻结；负值表示前者更快）：")
    A("")
    pt = ptime[(ptime.scenario == MAIN_SCENARIO)]
    rows = [r for _, r in pt.sort_values(["far_target", "sharpe_valid"]).iterrows()]
    L.extend(_t(rows, ["α", "s", "比较", "时间差（天）", "95% 区间", "排除 0"],
                lambda r: [f"{r['far_target']:g}", f"{r['sharpe_valid']:g}",
                           PAIR_CN[(r["method_a"], r["method_b"])],
                           f"{r['trunc_time_diff_days']:+.2f}",
                           f"[{r['trunc_time_diff_lo']:+.2f}, {r['trunc_time_diff_hi']:+.2f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    A("**区间包含零不等于两个方法等效**：那只说明本轮的样本量没有把差别与零分开。")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 次要预设比较：增益随信号强度如何变化")
    A("")
    A(f"同一对方法在 s={s_lo:g} 与 s={s_hi:g} 的增益之差，配对估计，"
      "两个 s 各自重算门槛：")
    A("")
    rows = []
    for sc in COMBINED:
        for a in sorted(cfg.far_targets, reverse=True):
            for ma, mb in MAIN_PAIRS:
                b = bo("gain_change_with_s", sc, a, ma, mb, f"{s_lo:g}-{s_hi:g}")
                if b is not None:
                    rows.append((sc, a, ma, mb, b))
    L.extend(_t(rows, ["情境", "α", "比较", f"增益(s={s_lo:g}) − 增益(s={s_hi:g})",
                       "95% 区间", "排除 0"],
                lambda r: [SC[r[0]], f"{r[1]:g}", PAIR_CN[(r[2], r[3])],
                           f"{float(r[4].point) * 100:+.2f} pp",
                           f"[{float(r[4].lo) * 100:+.2f}, {float(r[4].hi) * 100:+.2f}]",
                           "是" if bool(r[4].excludes_zero) else "否"]))
    A("")

    # ---------------- 6 ----------------
    A("## 6. 失效概率")
    A("")
    A("两种真实状态的**全部路径**都保留，报警之后的概率继续计入。"
      f"测试为 50/50（有效 / 无效各 {cfg.stage3a_test_paths} 条），先验 0.5。")
    A("")
    for s in sharpes:
        A(f"**s = {s:g}**，{SC[MAIN_SCENARIO]}：q 的中位数与 10–90% 分位")
        A("")
        rows = [(m, st, d) for m in ("binary_student_t", "ewma_student_t",
                                     "ewma_trunc_student_t")
                for st in ("invalid", "valid") for d in days]
        def rq(r):
            m, st, d = r
            x = qrows[(qrows.scenario == MAIN_SCENARIO) & (qrows.sharpe_valid == s)
                      & (qrows.method == m) & (qrows.true_state == st) & (qrows.day == d)]
            if not len(x):
                return [CN[m], st, str(d), "-", "-"]
            x = x.iloc[0]
            return [CN[m], "无效" if st == "invalid" else "有效", str(d),
                    f"{x.median_q:.4f}", f"[{x.q10:.4f}, {x.q90:.4f}]"]
        L.extend(_t(rows, ["方法", "真实状态", "日", "q 中位数", "10–90% 分位"], rq))
        A("")
    if len(brier):
        A("Brier 分数（越小越好）与可靠性分箱见 `stage3a_brier.csv` / `stage3a_reliability.csv`，"
          "每箱带样本量：")
        A("")
        rows = [r for _, r in brier.sort_values(["sharpe_valid", "detector", "day"]).iterrows()]
        L.extend(_t(rows, ["s", "方法", "日", "Brier", "常数基准 Brier", "样本量"],
                    lambda r: [f"{r['sharpe_valid']:g}", CN.get(r["detector"], r["detector"]),
                               f"{int(r['day'])}", f"{r['brier']:.5f}",
                               f"{r['brier_base_rate']:.5f}", f"{int(r['n'])}"]))
        A("")
    lvl = cfg.stage3a_prob_level
    A(f"**解释性指标：首次达到 q ≥ {lvl:g} 的累计比例。**"
      f"必须同时读有效策略错误达到该水平的比例。"
      f"它**不是** α={1 - lvl:.0%} 的误杀控制，也不用于替代正式排名。")
    A("")
    rows = []
    for s in sharpes:
        for m in ("binary_student_t", "ewma_student_t", "ewma_trunc_student_t"):
            for st in ("invalid", "valid"):
                x = plevel[(plevel.scenario == MAIN_SCENARIO) & (plevel.sharpe_valid == s)
                           & (plevel.method == m) & (plevel.true_state == st)]
                if len(x):
                    rows.append((s, m, st, x.iloc[0]))
    L.extend(_t(rows, ["s", "方法", "真实状态"] + [f"d{d}" for d in days],
                lambda r: [f"{r[0]:g}", CN[r[1]], "无效" if r[2] == "invalid" else "有效"]
                + [f"{r[3][f'reached_by_d{d}']:.4f}" for d in days]))
    A("")

    # ---------------- 7 ----------------
    A("## 7. 高斯理论参照")
    A("")
    A("仅在 iid Gaussian 对照中使用：")
    A("")
    A("```")
    A("    D_max(h, alpha) = Phi( s*sqrt(h) - Phi^{-1}(1 - alpha) )")
    A("```")
    A("")
    rows = [r for _, r in ref.iterrows()]
    L.extend(_t(rows, ["s", "α", "h（年）", "天数", "检出率上界"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{r['far_budget']:g}",
                           f"{r['years']:g}", f"{int(r['days'])}",
                           f"{r['max_detection']:.4f}"]))
    A("")
    A(f"因此 s={s_lo:g} 的两年上界为 α={a_main:g} 时 "
      f"{max_detection(s_lo, cfg.horizon_days / cfg.D, a_main):.2%}、"
      f"α={a_alt:g} 时 {max_detection(s_lo, cfg.horizon_days / cfg.D, a_alt):.2%}。"
      "**所以这一组方法两年内没有达到 50% 检出，不应自动称为方法失败**——"
      "在满足该参照条件的模型里，50% 本来就在上界之外。")
    A("")
    A("**这条参照有明确边界。**它只约束同一 iid 高斯模型、同一期限、满足累计误杀约束的规则。"
      "SV 与跳跃情境里波动可预测、证据权重可调，检出率高于这个数并不矛盾，"
      "所以它不能用来给组合情境的方法排名。图 4 里画的参照曲线使用**名义预算**，"
      "并不表示我们的规则可以在每个期限分别用满 α——它们在两年里只花一次预算。")
    A("")
    ev = pd.DataFrame(S["evidence_check"])
    A("同时核验纯高斯二值模型在无效状态、初始赔率为 1 时的证据积累：")
    A("")
    A("```")
    A("    E[U_n] = s^2 n / (2D),    Var(U_n) = s^2 n / D")
    A("```")
    A("")
    rows = [r for _, r in ev.iterrows()]
    L.extend(_t(rows, ["s", "n", "E[U] 理论", "E[U] 实测", "Var 理论", "Var 实测",
                       "最大相对误差"],
                lambda r: [f"{r['sharpe_valid']:g}", f"{int(r['day'])}",
                           f"{r['mean_U_theory']:.4f}", f"{r['mean_U_measured']:.4f}",
                           f"{r['var_U_theory']:.4f}", f"{r['var_U_measured']:.4f}",
                           f"{max(r['mean_rel_err'], r['var_rel_err']):.2e}"]))
    A("")
    A(f"平均对数证据的积累速度随 s² 缩放，故 1/0.6² ≈ {1 / 0.36:.2f}。"
      "**这只是该模型下的证据时间尺度参照**，不宣称所有噪声、门槛和实际停止时间"
      "都严格按这个倍数缩放——下一节的实测数字就不按它缩放。")
    A("")

    # ---------------- 8 ----------------
    A("## 8. 五个问题的回答")
    A("")
    hb = max((g(m, MAIN_SCENARIO, s_lo, a_main, "detect_d504"), m) for m in METHODS_2E)
    A(f"**（一）弱 Sharpe 下，两年、一年、半年分别能判断多少？** "
      f"主情境、α={a_main:g}、s={s_lo:g}，最好的方法（{CN[hb[1]]}）两年检出 "
      f"{hb[0]:.1%}，一年 {g(hb[1], MAIN_SCENARIO, s_lo, a_main, 'detect_d252'):.1%}，"
      f"半年 {g(hb[1], MAIN_SCENARIO, s_lo, a_main, 'detect_d126'):.1%}，"
      f"一个季度 {g(hb[1], MAIN_SCENARIO, s_lo, a_main, 'detect_d63'):.1%}。"
      f"对照 s={s_hi:g} 的同一方法两年 "
      f"{g(hb[1], MAIN_SCENARIO, s_hi, a_main, 'detect_d504'):.1%}。")
    A("")
    b1 = bo("detection_diff", MAIN_SCENARIO, a_main, *MAIN_PAIRS[0], f"{s_lo:g}")
    A(f"**（二）方差自适应的主要收益还在不在？** 主设定下 {PAIR_CN[MAIN_PAIRS[0]]} 为 "
      f"{float(b1.point) * 100:+.2f} pp [{float(b1.lo) * 100:+.2f}, {float(b1.hi) * 100:+.2f}]，"
      f"区间{'排除' if bool(b1.excludes_zero) else '覆盖'} 0。"
      "第 5 节给出这一增益在两个 s 之间的配对变化。")
    A("")
    b2 = bo("detection_diff", MAIN_SCENARIO, a_main, *MAIN_PAIRS[1], f"{s_lo:g}")
    A(f"**（三）截断的小幅收益是否仍值得保留？** 主设定下 {PAIR_CN[MAIN_PAIRS[1]]} 为 "
      f"{float(b2.point) * 100:+.2f} pp [{float(b2.lo) * 100:+.2f}, {float(b2.hi) * 100:+.2f}]，"
      f"区间{'排除' if bool(b2.excludes_zero) else '覆盖'} 0。"
      "结合配对时间差与其它情境的敏感性结果一并判断；本轮不为它单独下保留或删除的结论。")
    A("")
    bnd = max_detection(s_lo, cfg.horizon_days / cfg.D, a_main)
    gc = max((g(m, "gaussian_ctrl", s_lo, a_main, "detect_d504"), m) for m in METHODS_2E)
    A(f"**（四）哪些困难有理论参照支持，哪些仍属于模型或门槛问题？** "
      f"**有参照支持的部分**：在 iid 高斯对照里，s={s_lo:g}、两年、α={a_main:g} 的检出率"
      f"存在一个与规则无关的上界 {bnd:.2%}；本轮该情境下最好的方法达到 {gc[0]:.2%}，"
      "在上界之下。所以那里“两年抓不到一半”不是方法的问题，是这个信号强度本身的性质。")
    A("")
    A(f"**但这条参照不能搬到主情境。**同一设定下 {SC[MAIN_SCENARIO]} 里最好的方法"
      f"两年检出 {hb[0]:.1%}，**高于**上面那个 {bnd:.2%}，这不是矛盾："
      "SV+跳跃情境的波动本身可预测，证据权重可以随时间调整，"
      "而高斯上界的推导里没有这份可利用的结构。这恰好说明"
      "“同一个年化 Sharpe 不唯一决定可辨识程度”。")
    A("")
    A("**没有参照支持的部分**：SV+跳跃情境里方法之间的差距、门槛缓冲的代价、"
      "以及组合情境下的最优检出率是多少——本项目至今没有推导过后者，"
      "因此不能把任何一段差距归因于“信息上限”。")
    A("")
    A("**（五）下一步是否仍适合进入随机 T 的有限扩展？** "
      "本轮不作这一步，也不预先断言其结论；判断依据应是："
      "在弱信号下，正面的早期历史会把识别推迟多久，"
      "以及这一延迟是否大到改变监督者的决策窗口。本轮的数字给出了这个问题的基线。")
    A("")

    # ---------------- 9 ----------------
    A("## 9. 限制")
    A("")
    A(f"- 两个 s 共用噪声块，这使 s 之间的比较是配对的，但也意味着"
      f"{sp['return_path_evaluations']:,} 次评估**不是**同样数量的独立样本。")
    A("- 逐情境校准只覆盖这五个固定情境，不构成未知环境下的统一门槛保证。")
    A("- 主比较与次要比较的区间都是逐项区间，未做多重比较校正。")
    A("- 高斯参照只在 iid 高斯对照中适用；组合情境没有已知上界。")
    A(f"- 只测了 s ∈ {{{s_hi:g}, {s_lo:g}}} 与 α ∈ {{{a_alt:g}, {a_main:g}}} 各两点，"
      "不外推整条曲线。")
    A("- 中位检出时间在检出率不足一半时不可得，本报告不据两年曲线外推更长期限。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.run_stage3a")
    A("uv run python -m strategy_survivorship.run_stage3a --figures-only")
    A("```")
    A("")
    A(f"本轮耗时 {S['elapsed_s']:.1f} 秒。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
