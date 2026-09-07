# Stage 3A.1 review bundle — 监测期限诊断

由 `python -m strategy_survivorship.run_stage3a1` 生成。
**本轮只改变用于校准误杀约束的监测期限**：不搜索模型参数，不优化动态边界，
不做随机 T 或 BOCPD。检测器、DGP、EWMA 系数、候选均值传入与首次越界规则全部不变。

Stage 3A 与 Stage 2E.1 的审阅包保留在 `outputs/review_bundle_stage3a.zip`
与 `outputs/review_bundle.zip`，可一并转交。

## 两种安排

- **A**：按 504 天校准一个门槛，读它在第 63/126/252/504 天的累计表现。
- **B**：分别按 H=63/126/252/504 校准门槛，在各自截止日评价。
  B 的四个期限是**四个独立设计的监测方案**，各自在自己的窗口花掉自己的 α，
  不能拼接成共享同一个两年预算的计划。

## 报告与图

| 文件 | 内容 |
|---|---|
| `stage3a1_report.md` | 主设定 s=0.6 / SV+跳跃 / α=0.15 / H=126，含五个问题的回答 |
| `theory.md` | §16 Stage 2E、§17 高斯参照、§18 Stage 3A、§19 Stage 3A.1 |
| `figures/fig3a1_1_arms.png` | A 与 B 在相同截止日的检出率，以及配套实际误杀率 |
| `figures/fig3a1_2_differences.png` | B−A 的检出差与误杀差，两张并排，须一起读 |
| `figures/fig3a1_3_out_of_horizon.png` | 短期门槛延用后的误杀变化；右侧单独给高斯理论参照 |

## 数据

| 文件 | 内容 |
|---|---|
| `data/stage3a1_metrics.csv` | 2 情境 × 4 方法 × 2 s × 2 α × 2 安排 × 4 截止日 |
| `data/stage3a1_thresholds.csv` | 128 个冻结门槛，含工作概率门槛与覆盖范围声明 |
| `data/stage3a1_bootstrap.csv` | B−A 的检出差与误杀差、各期限自校准后的方法间增益 |
| `data/stage3a1_paired_time.csv` | 同一截止日上的配对 E[min(τ,H)] 差 |
| `data/stage3a1_out_of_horizon.csv` | 短期门槛延用到各天的实际累计误杀与检出 |
| `data/stage3a1_gaussian_reference.csv` | D_max(h,α)，逐截止日分别计算 |
| `data/stage3a1_summary.json` | 参数、随机流指纹、环境版本、抽样说明 |

## 代码与环境

`code/` 含**完整的 `strategy_survivorship` 包**（本阶段入口的传递依赖有 27 个模块，
因此不做子集裁剪）、`pyproject.toml`、`requirements.txt`、`requirements-lock.txt`，
以及与本阶段相关的测试。

```bash
cd code
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m pytest tests
PYTHONPATH=src .venv/bin/python -m strategy_survivorship.run_stage3a1
```

`--figures-only` 与 `--report-only` 只读取已生成的 CSV，不重跑模拟。
