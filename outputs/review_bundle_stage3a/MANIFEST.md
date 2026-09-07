# Stage 3A review bundle — Sharpe 0.6 vs 0，保留 1 vs 0 参照

由 `python -m strategy_survivorship.run_stage3a` 生成。
**本轮没有新增检测模型、没有搜索参数、没有做随机 T 或 BOCPD。**只有信号强度改变。

Stage 2E / 2E.1 的审阅包保留在 `outputs/review_bundle.zip`，可一并上传核对。

## 报告

`stage3a_report.md` —— 主设定 s=0.6 / SV+跳跃 / α=0.15，含五个问题的回答与限制。
`theory.md` §18 为本阶段推导，§17 为高斯参照。

## 图

| 文件 | 内容 |
|---|---|
| `figures/fig3a1_detection_curves.png` | 两个信号强度下的检出率—时间曲线，附同期实际误杀 |
| `figures/fig3a2_differences.png` | 两个组合情境里三条**预设**比较的差异及区间 |
| `figures/fig3a3_failure_probability.png` | 两种真实状态下失效概率随季度/半年/一年/两年变化 |
| `figures/fig3a4_gaussian_reference.png` | 纯高斯实际结果与 D_max 参照（名义预算） |

## 数据

| 文件 | 内容 |
|---|---|
| `data/stage3a_metrics.csv` | 5 情境 × 8 方法 × 2 信号强度 × 2 预算的全部指标 |
| `data/stage3a_thresholds.csv` | 160 个冻结门槛（J=160，秩 427 / 1379，含覆盖范围声明）|
| `data/stage3a_bootstrap.csv` | 三条预设比较、增益随 s 的配对变化、同预算误杀差 |
| `data/stage3a_paired_time.csv` | 冻结门槛下的配对检测时间差 |
| `data/stage3a_failure_probability.csv` | q 的中位数与 10–90% 分位，两种真实状态、全部路径 |
| `data/stage3a_probability_level.csv` | 首次达到 q≥0.9 的累计比例（含有效策略误达比例）|
| `data/stage3a_brier.csv` / `_reliability.csv` | Brier 分数 / 分箱可靠性（带每箱样本量）|
| `data/stage3a_gaussian_reference.csv` | D_max(h,α)，s ∈ {1, 0.6} |
| `data/stage3a_evidence_check.csv` | E[U_n]=s²n/(2D)、Var(U_n)=s²n/D 的实测核验 |

## 代码

`code/` 含 `stage3a.py`（信号强度、共同随机数、校准、配对 bootstrap）、
`run_stage3a.py`、`gaussian_bound.py`、`config.py`、`ewma.py`、`noise.py`、`detectors.py`。

## 复现

```bash
uv run python -m pytest                                # 255 项
uv run python -m strategy_survivorship.run_stage3a     # 约 120 秒
```
