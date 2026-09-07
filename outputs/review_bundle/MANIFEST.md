# Stage 2E / 2E.1 review bundle

由 `python -m strategy_survivorship.run_stage2e` 与 `run_stage2e1` 生成。
本轮（2E.1）**没有新增模型、没有搜索截断参数、没有改动任何门槛，也没有开始弱 Sharpe 实验**。

## 先读这三份

| 文件 | 内容 |
|---|---|
| `reports/stage2e1_report.md` | **本轮核查说明**：撤回/改写了哪些结论、跨预算配对统计、理论巧合基准、高斯参照 |
| `reports/stage2e_report.md` | Stage 2E 主报告（措辞已按本轮核查修正） |
| `reports/stage2d_report.md` | Stage 2D 报告（含方差定义修正与追加证据） |

## 图（Stage 2E，不是 2D）

| 文件 | 内容 |
|---|---|
| `figures/fig2e1_mechanism.png` | 单条 SV+跳跃路径上，截断对方差预测与累计对数几率的作用 |
| `figures/fig2e2_2x2.png` | 2×2 消融的两年检出率，五情境 × 两预算，各用自己的缓冲门槛 |
| `figures/fig2e3_differences.png` | 三条预设配对差与 2×2 交互项，含校准重抽的 95% 区间 |
| `figures/fig2e4_far_timing.png` | 首次误杀的三类互斥时点分解，带类 1 的理论巧合水平标记 |

## Stage 2E / 2E.1 数据

| 文件 | 内容 |
|---|---|
| `stage2e/stage2e_metrics.csv` | 5 情境 × 8 模型 × 2 预算 × 2 臂的全部指标 |
| `stage2e/stage2e_thresholds.csv` | 冻结门槛（J=80，秩 431 / 1386） |
| `stage2e/stage2e_bootstrap.csv` | 三条预设配对差与交互项，含校准重抽 |
| `stage2e/stage2e_paired.csv` / `_paired_time.csv` | 冻结门槛下的配对检出差 / 检测时间差 |
| `stage2e/stage2e_brier.csv` / `_reliability.csv` | 配对 Brier 差 / 分箱可靠性（带每箱样本量） |
| `stage2e/stage2e_far_timing.csv` | 首次误杀三类分解（观测） |
| `stage2e/stage2e1_cross_budget.csv` | 新增：`I(0.05) − I(0.15)` 等跨预算配对差（探索性） |
| `stage2e/stage2e1_far_timing.csv` | 新增：三类分解，含理论巧合基准与两种分母 |
| `stage2e/stage2e1_summary_table.csv` | 新增：两组合情境 × 两预算的检出/误杀/时间汇总 |
| `stage2e/stage2e1_paired_time.csv` | 新增：四组配对的检测时间差 |
| `stage2e/stage2e1_gaussian_reference.csv` | 新增：D_max(h,α)，s∈{1,0.6} |
| `stage2e/stage2e1_gaussian_headroom.csv` | 新增：高斯对照下实测 vs 同模型上界 |
| `stage2e/stage2e1_threshold_check.csv` | 新增：重建门槛与冻结门槛的逐位比对 |

## Stage 2D 补充

`stage2d/stage2d_paired_time.csv`、`_brier.csv`、`_reliability.csv` 为本系列追加的证据；
`_metrics.csv` 与 `_shock.csv` 供对照。

## 代码

`code/` 下为本轮涉及的关键实现：`ewma.py`（截断递推）、`noise.py`（两个方差定义）、
`stage2e.py`（八模型与时点分类）、`stage2e1.py`（跨预算配对、理论基准）、
`gaussian_bound.py`（NP 参照）、`config.py`（全部参数与随机流顺序）。
`theory.md` 为完整推导：§16 为 Stage 2E，§17 为高斯参照。

## 复现

```bash
uv run python -m pytest                                 # 241 项
uv run python -m strategy_survivorship.run_stage2e      # 约 70 秒
uv run python -m strategy_survivorship.run_stage2e1     # 约 40 秒
```

同种子重跑得到相同 CSV，这是**确定性复现检查**，不是用新独立样本做的重复验证。
