# strategy-survivorship — Stage 1

一个新策略开始样本外运行之后，它究竟真的有收益优势，还是从第一天起就没有优势？
本仓库研究**在可接受的误杀率下，尽快识别无效策略**。

**第一阶段**只建立最小但完整的可复现流程：

```
模拟 → 检测器逐日更新 → 阈值校准 → 独立数据评估 → 图表与报告
```

正式 benchmark 的数据生成过程**只有**固定波动率、独立高斯收益。厚尾、时变波动率、
跳跃、以及“先有效后失效”留给后续阶段。另有一个很小的单次冲击诊断，只用于理解
Student-t 的更新行为，**不混入** benchmark。

公式、符号、单位与推导见 [`theory.md`](theory.md)。
结果报告见 [`outputs/stage1_report.md`](outputs/stage1_report.md)。

---

## 环境配置

需要 Python ≥ 3.11。记录运行使用的是 **Python 3.13.13**，依赖只有
NumPy / SciPy / pandas / Matplotlib（测试另需 pytest）。无 GPU、无付费服务、无云端计算。

```bash
# 用 uv（推荐，可精确重建同一个解释器版本）
uv venv --python 3.13.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python -e .

# 或用任意 Python >= 3.11
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

`requirements-lock.txt` 是实际运行环境的精确版本；`requirements.txt` 只给下界。
实际解释器与库版本也会写进每次运行的 `outputs/run_metadata.json`。

---

## 完整重跑命令

```bash
.venv/bin/python -m pytest                                # 验证
.venv/bin/python -m strategy_survivorship.run_stage1      # 主 benchmark（约 3 秒）
.venv/bin/python -m strategy_survivorship.run_stage11     # Stage 1.1 诊断（约 35 秒）
.venv/bin/python -m strategy_survivorship.run_stage2a     # Stage 2A 现实噪声（约 10 秒）
.venv/bin/python -m strategy_survivorship.run_stage2b     # Stage 2B EWMA 波动率预测（约 30 秒）
.venv/bin/python -m strategy_survivorship.run_stage2c     # Stage 2C 统一门槛与误杀控制（约 70 秒）
.venv/bin/python -m strategy_survivorship.run_stage2d     # Stage 2D SV+跳跃组合（约 70 秒）
.venv/bin/python -m strategy_survivorship.run_stage2e     # Stage 2E 截断方差更新 2x2 消融（约 70 秒）
.venv/bin/python -m strategy_survivorship.run_stage2e1    # Stage 2E.1 收尾、跨预算统计、高斯参照（约 40 秒）
.venv/bin/python -m strategy_survivorship.run_stage3a     # Stage 3A Sharpe 0.6 vs 0，保留 1 vs 0（约 120 秒）
.venv/bin/python -m strategy_survivorship.run_stage2e --figures-only  # 仅从 CSV 重绘
```

这两条命令重建 `outputs/` 下的**全部**结果：表格、JSON、四张图和研究报告。
不依赖任何 notebook，也不需要手动按顺序执行单元格。
在一台普通笔记本上主 benchmark 约 3 秒、Stage 1.1 约 20 秒。
校准不确定性已改用解析解，不再需要昂贵的重复模拟。

其它选项：

| 选项 | 作用 |
|---|---|
| `--smoke` | 400 条路径的小规模冒烟运行，约 1 秒，输出到 `outputs/smoke/` |
| `--out DIR` | 指定输出目录 |
| `--seed N` | 覆盖根随机种子（用于稳定性检查，**不用于**挑选好看的结果） |
| `--replications N` | 校准不确定性的重复模拟次数（默认 `0`，已被解析解取代；设 N≥2 可作交叉验证）|
| `--no-figures` | 跳过绘图 |

---

## 目录结构

```
src/strategy_survivorship/
  config.py       本阶段全部实验参数（唯一的集中配置）
  simulate.py     数据生成过程与随机流（SeedSequence）
  detectors.py    四个检测器 + 影响函数
  evaluate.py     阈值校准、首次穿越、指标与区间
  plots.py        四张图
  robustness.py   多种子复算（可选交叉验证，默认关闭）
  analytic_calibration.py  门槛真实误杀概率的 Beta / Beta-binomial 精确律
  probability_time.py      失效概率 q_n、概率门槛首达时间、Brier 与可靠性
  switching.py             随机失效时间 T 的 DGP 与指标（含匹配失效前误杀对照）
  paired.py                检测器之间的配对差异与区间
  contfa_validation.py     冻结门槛的独立样本验证
  noise.py                 Stage 2A 四种噪声：高斯 / 厚尾 / 随机波动率 / 跳跃
  stage2a.py               情境构建、oracle 检测器、两类比较
  plots_stage2a.py / report_stage2a.py / run_stage2a.py   Stage 2A 图、报告、流程
  ewma.py                  Stage 2B 因果 EWMA 方差预测 + 两个检测器 + QLIKE + N_info
  stage2b.py               两臂比较、持续性对照、信息隔离的检测器分派
  stage2b_uncertainty.py   同时重采样校准集与测试集的 bootstrap
  stage2b_vol_diagnostics.py  波动率预测诊断
  plots_stage2b.py / report_stage2b.py / run_stage2b.py   Stage 2B 图、报告、流程
  stage2b_followup.py      Stage 2C 追加的限定性分析（同一批 2B 数据）
  stage2c.py               次序统计量门槛 + 校准误差缓冲 + 跨情境统一
  plots_stage2c.py / report_stage2c.py / run_stage2c.py   Stage 2C 图、报告、流程
  stage2d.py               SV+跳跃组合 DGP、两个比较条件、单次冲击诊断
  stage2d_uncertainty.py   冻结门槛配对区间 + 含校准重采样的 bootstrap
  plots_stage2d.py / report_stage2d.py / run_stage2d.py   Stage 2D 图、报告、流程
  plots_stage11.py / report_stage11.py / run_stage11.py   Stage 1.1 图、报告、流程
  report.py       生成 stage1_report.md
  notes.py        写入报告的“已修正问题”与“限制”
  run_stage1.py   端到端流程（一条命令）
tests/            有实际统计意义的验证测试
outputs/          全部生成结果（下表）
theory.md         公式、符号、单位、推导、参考链接
```

## 输出

| 文件 | 内容 |
|---|---|
| `outputs/stage1_report.md` | 中文研究报告：做了什么、参数、核心结果、修正的问题、限制 |
| `outputs/stage1_metrics.csv` | 每个检测器 × 每个误杀率目标的主要指标与区间 |
| `outputs/stage1_summary.json` | 参数、样本规模、阈值、指标、诊断、耗时的紧凑汇总 |
| `outputs/stage1_first_passages.csv` | 每条测试路径：真实状态、是否报警、首次报警日、截断时间 |
| `outputs/stage1_curves.csv` | 每日累计报警率曲线（逐检测器 / α / 状态） |
| `outputs/stage1_calibration_robustness.csv` | 100 次复算（一次性交叉验证，非默认步骤）|
| `outputs/stage11_report.md` | **Stage 1.1 报告**：解释修正、概率-时间、随机失效时间 |
| `outputs/stage11_*.csv` | Stage 1.1 各项结果（解析校准、配对比较、概率门槛、失效时间指标等）|
| `outputs/figures/fig1{1,2,3,4}_*.png` | Stage 1.1 四张图 |
| `outputs/stage2a_report.md` | **Stage 2A 报告**：现实噪声下的基线压力测试 |
| `outputs/stage2a_metrics.csv` | 情境 × 臂 × α × 检测器的全部指标 |
| `outputs/stage2a_noise_diagnostics.csv` | 四种噪声的矩、尾频率与结构诊断 |
| `outputs/stage2a_paired.csv` / `stage2a_probability_calibration.csv` | 配对区间 / Brier |
| `outputs/stage11_frozen_threshold_validation.csv` | 冻结门槛的独立样本验证 |
| `outputs/figures/fig2a{1,2,3,4}_*.png` | Stage 2A 四张图 |
| `outputs/stage2b_report.md` | **Stage 2B 报告**：EWMA 波动率预测能否改善验证 |
| `outputs/stage2b_metrics.csv` | 情境 × 臂 × α × 检测器的全部指标 |
| `outputs/stage2b_bootstrap.csv` | 含校准不确定性的 bootstrap 区间（2000 次）|
| `outputs/stage2b_vol_diagnostics.csv` / `_qlike_benchmarks.csv` / `_information_days.csv` / `_vol_trace.csv` | 波动率诊断 |
| `outputs/stage2b_probability_calibration.csv` / `_reliability.csv` | 概率校准 |
| `outputs/figures/fig2b{1,2,3,4}_*.png` | Stage 2B 四张图 |
| `outputs/stage2c_report.md` | **Stage 2C 报告**：未知情境下的统一门槛与误杀控制 |
| `outputs/stage2c_thresholds.csv` / `_unified_thresholds.csv` | 每情境门槛、统一门槛与约束情境 |
| `outputs/stage2c_metrics.csv` | 8 情境 × 3 臂 × 2 预算 × 6 方法的全部指标 |
| `outputs/stage2c_paired.csv` / `_bootstrap.csv` | 配对区间 / 含校准重采样的敏感性区间 |
| `outputs/stage2c_failure_probability.csv` | 全部路径的无效概率分布（63/126/252/504 天）|
| `outputs/figures/fig2c{1,2,3,4}_*.png` | Stage 2C 四张图 |
| `outputs/stage2d_report.md` | **Stage 2D 报告**：SV 与跳跃同时存在时的检测 |
| `outputs/stage2d_metrics.csv` / `_thresholds.csv` | 5 情境 × 2 条件 × 2 预算 × 6 方法指标、诊断门槛 |
| `outputs/stage2d_paired.csv` / `_bootstrap.csv` / `_brier.csv` | 配对区间 / 敏感性区间 / 配对 Brier |
| `outputs/stage2d_failure_probability.csv` / `_shock.csv` | 全路径 q 分布 / 单次冲击诊断 |
| `outputs/stage2d_paired_time.csv` / `_reliability.csv` | 配对截断时间之差 / 分箱可靠性 |
| `outputs/figures/fig2d{1,2,3,4}_*.png` | Stage 2D 四张图 |
| `outputs/stage2e_report.md` | **Stage 2E 报告**：截断方差更新的 2×2 消融 |
| `outputs/stage2e_metrics.csv` / `_thresholds.csv` | 5 情境 × 8 模型 × 2 预算指标（J=80 缓冲门槛）|
| `outputs/stage2e_bootstrap.csv` | 三条预设配对差 **与 2×2 交互项**，含校准重采样的区间 |
| `outputs/stage2e_paired.csv` / `_paired_time.csv` / `_brier.csv` / `_reliability.csv` | 冻结门槛下的配对区间、时间差、概率诊断 |
| `outputs/stage2e_far_timing.csv` | 首次误杀的三类互斥时点分解（三类之和 = 总 FAR）|
| `outputs/stage2e_failure_probability.csv` / `_shock.csv` | 全路径 q 分布 / 四路冲击诊断 |
| `outputs/figures/fig2e{1,2,3,4}_*.png` | Stage 2E 四张图 |
| `outputs/stage2e1_report.md` | **Stage 2E.1 核查说明**：撤回/改写的结论、跨预算统计、理论基准 |
| `outputs/stage2e1_cross_budget.csv` | `I(0.05) − I(0.15)` 等跨预算配对差（探索性）|
| `outputs/stage2e1_far_timing.csv` | 三类时点分解，含 Poisson 理论巧合基准与两种分母 |
| `outputs/stage2e1_summary_table.csv` / `_paired_time.csv` | 导师口径的检出/误杀/时间汇总与配对时间差 |
| `outputs/stage2e1_gaussian_reference.csv` / `_headroom.csv` | D_max(h,α) 参照表与高斯对照下的余量 |
| `outputs/stage2e1_threshold_check.csv` | 重建门槛与冻结门槛的逐位比对 |
| `outputs/review_bundle.zip` | Stage 2E/2E.1 审阅包（保留） |
| `outputs/stage3a_report.md` | **Stage 3A 报告**：弱信号下需要多少观测 |
| `outputs/stage3a_metrics.csv` / `_thresholds.csv` | 5 情境 × 8 方法 × 2 信号强度 × 2 预算（J=160，秩 427/1379）|
| `outputs/stage3a_bootstrap.csv` / `_paired_time.csv` | 三条预设比较、增益随 s 的配对变化、配对时间差 |
| `outputs/stage3a_failure_probability.csv` / `_probability_level.csv` | q 的分位数 / 首达 q≥0.9 的比例（两种真实状态）|
| `outputs/stage3a_brier.csv` / `_reliability.csv` | Brier 分数 / 分箱可靠性 |
| `outputs/stage3a_gaussian_reference.csv` / `_evidence_check.csv` | D_max 参照表 / 证据积累恒等式核验 |
| `outputs/figures/fig3a{1,2,3,4}_*.png` | Stage 3A 四张图 |
| `outputs/review_bundle_stage3a.zip` | Stage 3A 审阅包 |
| `outputs/stage1_diagnostic_traces.csv` | 冲击诊断逐日收益、增量、log odds、概率 |
| `outputs/run_metadata.json` | 运行配置、随机流指纹、环境版本、分阶段耗时 |
| `outputs/figures/fig1_example_paths.png` | 固定示例路径的累计收益与两个贝叶斯概率 |
| `outputs/figures/fig2_operating_curves.png` | 测试集累计误杀率与累计检出率随时间变化 |
| `outputs/figures/fig3_detection_time.png` | 截断平均检测时间与两年未检出比例 |
| `outputs/figures/fig4_shock_diagnostic.png` | 原始 / ±冲击下两个模型的更新差异与影响函数 |

完整模拟数组不入库；给定配置、种子与环境即可完全重新生成。

---

## 本阶段的四个检测器

| 检测器 | 角色 | 统计量（越大越支持有效） | 最早可报警日 |
|---|---|---|---|
| Binary Gaussian | 基线 | 后验 log odds | 第 1 天 |
| Binary Student-t (ν=5) | 基线 | 后验 log odds | 第 1 天 |
| Trailing 12m Sharpe | 基线 | 年化滚动 Sharpe（ddof=1） | 第 252 天 |
| Known-vol rolling | 诊断对照 | 滚动均值 / 已知 σ_d | 第 252 天 |
| Random closure | 解析参考 | 无 | 第 1 天 |

**三点必须一起读**：(1) 贝叶斯模型从第 1 天更新，滚动模型要等满一年窗口——性能差距里
包含这一运行方式差异；(2) 主实验中日波动率对所有检测器都是已知的理想条件；
(3) 本轮 DGP 下两个贝叶斯模型的似然**恰好正确**，且真实备择假设 S=1 就是它们的候选假设之一，
而滚动 Sharpe 两者都不假设——“设定恰好正确”这部分优势在真实场景中不会免费获得。

## 本阶段不做的事

不接市场数据、外部账户或数据库；不用深度学习、强化学习、GAN、HMM、复杂最优停止
或资本配置；不引入交易收益奖励或任意错误成本——导师未指定完整标量 reward，
本阶段只控制整个监控期内的累计误杀率，并报告检出率曲线与截断平均检测时间。
