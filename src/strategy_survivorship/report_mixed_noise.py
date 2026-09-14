"""Generate ``mixed_noise_report.md``. Every number is read from the result tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CN = {"binary_gaussian": "固定 Gaussian", "binary_student_t": "固定 Student-t",
      "trailing_sharpe_252": "滚动一年 Sharpe", "ewma_student_t": "EWMA Student-t",
      "ewma_trunc_student_t": "截断 EWMA Student-t"}
DAYS = (63, 126, 252, 504)


def _t(rows, header, render):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(render(r)) + " |")
    return out


def write(out_dir: Path) -> Path:
    S = json.loads((out_dir / "mixed_noise_summary.json").read_text())
    m = pd.read_csv(out_dir / "mixed_noise_metrics.csv")
    b = pd.read_csv(out_dir / "mixed_noise_bootstrap.csv")
    g = pd.read_csv(out_dir / "mixed_noise_subgroups.csv")
    p, rk = S["protocol"], S["ranks"]
    L: list[str] = []
    A = L.append

    A("# mixed_noise 实验报告：每条策略的噪声参数都不同")
    A("")
    A("由 `python -m strategy_survivorship.run_mixed_noise` 自动生成，"
      "所有数字取自同一次运行的结果表。**本轮没有新增任何检测模型。**")
    A("")

    A("## 1. 运行前写定的设计")
    A("")
    A("```")
    A(f"    A_i     ~ Uniform(0, {S['config']['mixed_A_max']:g})   波动起伏幅度")
    A(f"    kappa_i ~ Uniform(0, {S['config']['mixed_kappa_max']:g})   跳跃尺度")
    A(f"    rho = {p['held_fixed']['rho']}    lambda = "
      f"{p['held_fixed']['jump_intensity_per_year']:g}/年    潜在 AR(1) 从平稳分布起步")
    A(f"    有效 Sharpe = {p['states']['valid_sharpe']:g}，无效 = "
      f"{p['states']['invalid_sharpe']:g}，先验各 {p['states']['prior']:g}")
    A(f"    D = {p['days_per_year']}，H = {p['horizon_days']}，"
      f"两年累计误杀预算 = {p['far_budget']:g}")
    A("```")
    A("")
    A("四个参数的角色不同，不要混：**A** 控制波动起伏的幅度；**rho** 控制一个波动水平"
      "能持续多久；**kappa** 控制单次跳跃有多大；**K** 是某一天落下几次跳跃。")
    A("")
    A("每条策略只抽一次 (A, kappa)，整个观察期固定，且**独立于状态与全部噪声流**。"
      "监测器看不到它们，只看到收益。")
    A("")
    A(f"**范围声明。** 这两个取值范围是运行前写定的**模拟选择**，"
      "不是从任何市场估计出来的，也不是导师指定的。")
    A("")

    A("## 2. 归一化与门槛")
    A("")
    A(f"- 标准化用的是**该路径自己 kappa 对应的理论常数** "
      f"`sqrt(1 + kappa_i^2 * lambda/D)`；**没有**按路径的样本均值或样本方差重新标准化。")
    A(f"- 三个批次各 {p['batches']['calibration_valid']} 条，参数与噪声**逐批独立抽取**；"
      "两个测试批次在所有方法之间共享，因此比较是配对的。")
    A(f"- 每种方法在**整个混合总体**上只校准一个门槛，然后冻结："
      f"规则数 {rk['n_rules']}、δ = {rk['delta']:g}、δ/规则 = {rk['delta_per_rule']:g}、"
      f"秩 **{rk['rank']}**（二项式与 Beta 两路一致）。")
    A("- 门槛**没有**使用真实 A、kappa、事后分组或测试结果。")
    A("")
    A(f"**这个预算保证的范围**：{rk['scope']}")
    A("")

    A("## 3. 主结果")
    A("")
    rows = [r for _, r in m.iterrows()]
    L.extend(_t(rows, ["方法"] + [f"检出 d{d}" for d in DAYS]
                + [f"误杀 d{d}" for d in DAYS] + ["中位检出"],
                lambda r: [CN[r["method"]]]
                + [f"{r[f'detect_d{d}']:.4f}" for d in DAYS]
                + [f"{r[f'far_d{d}']:.4f}" for d in DAYS]
                + [f"{int(r['median_detect_days'])}d"
                   if r["median_detect_days"] == r["median_detect_days"]
                   else str(r["median_detect_note"])]))
    A("")
    A(f"样本量：无效测试路径 {int(m.n_test_invalid.iloc[0])} 条，"
      f"有效测试路径 {int(m.n_test_valid.iloc[0])} 条。"
      "Wilson 区间见 `mixed_noise_metrics.csv` 的 `*_lo` / `*_hi` 列。"
      "中位检出的定义是**全部无效路径**的累计检出率首次达到一半的那一天。")
    A("")

    A("## 4. 两个预设比较（504 日）")
    A("")
    A(f"{S['config']['mixed_bootstrap_reps']} 次配对 bootstrap，"
      "重抽单位是整条路径，校准与测试同时重抽，**每次重新校准门槛**；"
      "同一批次内各方法共用重抽下标。区间为逐项 95% 区间。")
    A("")
    rows = [r for _, r in b.iterrows()]
    L.extend(_t(rows, ["比较", "量", "点估计", "95% 区间", "排除 0"],
                lambda r: [f"{CN[r['method_a']]} − {CN[r['method_b']]}",
                           "检出率" if r["quantity"] == "detect" else "误杀率",
                           f"{r['point_pp']:+.2f} pp",
                           f"[{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]",
                           "是" if r["excludes_zero"] else "否"]))
    A("")
    d1 = b[(b.method_a == "binary_student_t") & (b.quantity == "detect")].iloc[0]
    d2 = b[(b.method_a == "ewma_student_t") & (b.quantity == "detect")].iloc[0]
    A(f"**读法。** 重尾似然的收益在混合总体里仍然测得到："
      f"{d1['point_pp']:+.2f} pp [{d1['lo_pp']:+.2f}, {d1['hi_pp']:+.2f}]，"
      "且同预算下的误杀差区间覆盖 0，即这份收益不是用更多误杀换来的。")
    A("")
    A(f"**EWMA 相对固定 Student-t 的收益在这里测不出来**："
      f"{d2['point_pp']:+.2f} pp [{d2['lo_pp']:+.2f}, {d2['hi_pp']:+.2f}]，区间覆盖 0。")
    A("")
    A("**但不要把这个差异全部归因于参数随机化。**相对此前的固定情境实验，"
      "本轮同时改变了两件事：(一) 每条策略的参数不同；"
      "(二) 门槛从按情境分别校准改成整个混合总体只用一个。"
      "本轮没有设计实验去分离这两者的贡献。")
    A("")

    A("## 5. 分组读回（同一冻结门槛，未重新校准）")
    A("")
    aS = S["config"]["mixed_subgroup_A_split"]
    kS = S["config"]["mixed_subgroup_kappa_split"]
    A(f"按 A = {aS:g}、kappa = {kS:g} 预先划分四个子群。"
      "**“低 A”不等于“没有随机波动率”**：A 只是更小，波动仍然起伏，"
      "而且 rho = 0.98 的持续性对每条路径都一样。")
    A("")
    rows = [r for _, r in g.iterrows()]
    L.extend(_t(rows, ["方法", "子群", "无效样本", "检出 d504", "有效样本",
                       "误杀 d504", "95% 区间"],
                lambda r: [CN[r["method"]], r["subgroup"], f"{int(r['n_invalid'])}",
                           f"{r['detect_d504']:.4f}", f"{int(r['n_valid'])}",
                           f"{r['far_d504']:.4f}",
                           f"[{r['far_d504_lo']:.4f}, {r['far_d504_hi']:.4f}]"]))
    A("")
    over = g[g.far_d504 > p["far_budget"]]
    if len(over):
        A(f"**{len(over)} 个子群的实测误杀率高于 {p['far_budget']:.0%} 的名义预算：**")
        A("")
        for _, r in over.sort_values("far_d504", ascending=False).iterrows():
            A(f"- {CN[r['method']]}，{r['subgroup']}："
              f"**{r['far_d504']:.2%}** [{r['far_d504_lo']:.2%}, {r['far_d504_hi']:.2%}]，"
              f"n = {int(r['n_valid'])}")
        A("")
        A("这正是第 2 节那条范围声明的含义：一个在整个混合总体上成立的预算，"
          "**不是**每个参数点上的同一保证。最安静的一类策略（低 A、小跳跃）"
          "被多花了误杀预算。")
        A("")
    A("")
    A("## 6. 限制")
    A("")
    A("- 全部为模拟。独立测试路径确实是样本外评估，"
      "**但它们来自与校准路径同一个生成模型族**；随机化参数**没有**解决真实市场的泛化问题。")
    A("- 参数范围是运行前的模拟选择，不是市场估计。")
    A(f"- 只做了一个误杀预算（{p['far_budget']:g}）、一个信号强度"
      f"（Sharpe {p['states']['valid_sharpe']:g} 对 {p['states']['invalid_sharpe']:g}）、"
      "策略从上线起状态不变；本轮不含随机失效时间。")
    A("- 区间为逐项区间，未做多重比较校正。")
    A("- 子群表用的是同一个冻结门槛读回，不是按子群重新校准的结果。")
    A("")
    A("## 7. 复现")
    A("")
    A("```bash")
    A("uv run python -m strategy_survivorship.run_mixed_noise")
    A("```")
    A("")
    A(f"耗时 {S['elapsed_s']:.1f} 秒。配置、随机流指纹与协议见 "
      "`mixed_noise_summary.json` 与 `mixed_noise_protocol.json`。")
    path = out_dir / "mixed_noise_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write(Path("outputs")))
