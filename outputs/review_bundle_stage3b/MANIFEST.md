# Stage 3B review bundle — 随机失效时间的有限诊断

由 `python -m strategy_survivorship.run_stage3b` 生成。
**本轮不新增检测模型、不扩大参数范围、不实现 BOCPD。**
先修复了 Stage 3A.1 的中位数定义与一个无效测试，详见 `FIXES.md`。

之前的审阅包保留在 `outputs/review_bundle{,_stage3a,_stage3a1}.zip`，可一并转交。

## 设定

θ_t = s（t ≤ T）、0（t > T）。T 是最后一个有效交易日，固定为 0/126/252，
或均匀抽自 0..252，外加 T=∞ 的始终有效对照。
**覆盖范围是「一年内失效」**，不是 0–5 年或 0–10 年的寿命研究。
门槛读取 Stage 3A 冻结的 504 日值，不按真实 T 单独校准。

## 报告与图

| 文件 | 内容 |
|---|---|
| `FIXES.md` | 本轮修复记录与影响范围核对 |
| `stage3b_report.md` | 主设定 s=0.6 / SV+跳跃 / α=0.15 / T=252 / h=252 |
| `theory.md` | §20 为本阶段推导；§17 高斯参照、§18 Stage 3A、§19 Stage 3A.1 |
| `figures/fig3b1_rates.png` | 失效前误杀、条件检出、联合检出三个量并列 |
| `figures/fig3b2_delay.png` | 共同失效后窗口上的延迟与条件检出 |
| `figures/fig3b3_evidence.png` | 失效时刻的 U_T 与 q_T，全部路径对存活路径 |

## 数据

| 文件 | 内容 |
|---|---|
| `data/stage3b_metrics.csv` | 2 情境 × 5 方法 × 2 s × 2 α × 4 失效设定的全部指标 |
| `data/stage3b_always_valid.csv` | T=∞ 对照的全期实际误杀率 |
| `data/stage3b_bootstrap.csv` | 预设比较的配对区间（存活分母每次重抽重算）|
| `data/stage3b_evidence.csv` | 失效时刻的证据诊断（仅对数赔率方法）|
| `data/stage3b_identity_check.csv` | E[U_T]、E[ΔU] 的解析核验 |
| `data/stage3a_thresholds.csv` | 本轮读取的冻结门槛来源 |
| `data/stage3a1_metrics_median_fixed.csv` | 中位数修复后的 Stage 3A.1 指标 |

## 代码与环境

`code/` 含完整的 `strategy_survivorship` 包、全部测试、`pyproject.toml` 与两个
requirements 文件。

```bash
cd code
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
PYTHONPATH=src .venv/bin/python -m pytest tests          # 307 项
PYTHONPATH=src .venv/bin/python -m strategy_survivorship.run_stage3b
```

`--figures-only` 与 `--report-only` 只读取已生成的 CSV，不重跑模拟。

## 下一步（收口阶段的待办，本轮不做）

- 全仓精简与最终 GitHub 交付整理
- 公司电脑上的长交接文档
- 三个留给导师决定的问题，见报告第 7 节
