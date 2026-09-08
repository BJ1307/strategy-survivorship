# 交接文档：给公司电脑上的 Claude

这份文档的目的是：**你没有看过之前的对话，也能接手这个项目。**
读完它你应该知道做过什么、结论是什么、每个数字从哪来、哪些操作不需要重跑实验。

---

## 0. 第一件事

**先读导师的反馈，确认优先级，再动手。**
不要自动增加检测模型、不要做参数扫描、不要扩大情境范围。
本周研究已经收口，等待导师反馈。

---

## 1. 仓库与版本

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/BJ1307/strategy-survivorship`（私有） |
| 分支 | `main` |
| 生成全部实验结果的版本 | `17595a5b885b5e9d896fe416000865c6cea708a3`，tag **`pre-consolidation`** |
| 本次整理版本 | `eb8620dc4b0b48056013e1a874f515e2f876c003`（提交信息以 `Consolidation:` 开头） |
| 语言 | Python 3.13，依赖见 `requirements-lock.txt` |

### 结果来源清单

| 产物 | 由哪个版本生成 | 收口时是否改动 |
|---|---|---|
| `outputs/stage1*` … `outputs/stage3a1*` 的全部 CSV/JSON | `17595a5`（`pre-consolidation`） | 否 |
| `outputs/stage3b_*.csv` / `.json` | `17595a5` 的代码，收口时确定性重跑一次 | 数值不变；新增 `working_prob_threshold_note` 列，滚动方法的 `working_prob_threshold` 由数值改为留空 |
| `outputs/stage3a1_metrics.csv` 的 `median_*` 两列 | Stage 3B 轮次修复中位数定义时重建 | 是（只这两列；门槛文件逐字节相同） |
| `outputs/figures/fig3b*` | 收口时按 Sharpe 分别重画 | 是（文件名加 `_s1` / `_s06` 后缀） |
| `docs/figures/fig1..4` | 收口时由 `figures_brief` 从上述 CSV 新画 | 新增 |
| 各阶段 `*_report.md` | 收口时用 `--report-only` 从既有 CSV 重生成 | 文字与新增章节，数值来自同一批 CSV |

**收口过程没有重新校准任何门槛，也没有改变任何随机流。**
干净 checkout 验证：330 项测试通过，四张精选图从已提交 CSV 重建后与仓库中的文件逐字节一致。

### 阅读顺序

1. `docs/BRIEF.md` —— 两页导师简报，四张主图，先看这个。
2. `README.md` —— 目录结构、复现命令、输出清单。
3. 本文件 —— 完整方法、协议、指标定义与数据出处。
4. `theory.md` —— 全部推导（§1–15 早期阶段，§16 截断消融，§17 高斯参照，
   §18 弱信号，§19 监测期限，§20 随机失效时间）。
5. `outputs/stage*_report.md` —— 每个阶段自己的完整报告。

---

## 2. 环境与运行

```bash
git clone https://github.com/BJ1307/strategy-survivorship
cd strategy-survivorship
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .

.venv/bin/python -m pytest -q          # 全部测试
```

### 哪些操作不需要重跑实验

| 目的 | 命令 | 是否跑模拟 |
|---|---|---|
| 重画四张主图 | `make brief` | 否，只读 CSV |
| 重画所有阶段图 | `make figures` | 否 |
| 重生成阶段报告 | `make report-only` | 否 |
| 重跑某个阶段 | `python -m strategy_survivorship.run_stage3b` | 是 |

**改措辞、改图例、改报告文字，一律用 `--report-only` / `--figures-only`，不要重跑模拟。**

### 完整复现（全部阶段，约 10 分钟）

```bash
.venv/bin/python -m strategy_survivorship.run_stage1     # 主 benchmark，约 3 秒
.venv/bin/python -m strategy_survivorship.run_stage11    # 概率-时间对应，约 35 秒
.venv/bin/python -m strategy_survivorship.run_stage2a    # 现实噪声，约 10 秒
.venv/bin/python -m strategy_survivorship.run_stage2b    # EWMA 方差预测，约 30 秒
.venv/bin/python -m strategy_survivorship.run_stage2c    # 统一门槛，约 70 秒
.venv/bin/python -m strategy_survivorship.run_stage2d    # SV+跳跃组合，约 70 秒
.venv/bin/python -m strategy_survivorship.run_stage2e    # 截断消融，约 70 秒
.venv/bin/python -m strategy_survivorship.run_stage2e1   # 收尾与跨预算，约 40 秒
.venv/bin/python -m strategy_survivorship.run_stage3a    # Sharpe 0.6 vs 1，约 120 秒
.venv/bin/python -m strategy_survivorship.run_stage3a1   # 监测期限，约 60 秒
.venv/bin/python -m strategy_survivorship.run_stage3b    # 随机失效时间，约 60 秒
.venv/bin/python -m strategy_survivorship.figures_brief  # 四张主图
```

阶段之间有依赖：Stage 3B 读 `outputs/stage3a_thresholds.csv`，
Stage 2D 读 `outputs/stage2c_unified_thresholds.csv`。按上面顺序跑即可。

---

## 3. 导师的问题与优先级

导师关心的原始问题：新策略上线后，如何区分有效与无效；能否更快识别同时控制误杀；
时变波动率与跳跃如何影响判断；几个月/几年后失效概率变化多少；不知道环境时门槛是否可靠。

**优先级（本周确认）**：

1. **Sharpe = 1 对 0 是主线**，这是导师明确指定的设定。
2. **「上线即无效」是主要用例**（Stage 1–3A）。
3. **Sharpe = 0.6 是弱信号压力测试**，是补充，不取代主线。
4. **随机失效时间 T 是补充诊断**（Stage 3B），覆盖范围仅第一年内失效。

历史记录不改写：Stage 3B 事先写定的主比较用的是 s=0.6；
按导师要求调整的是**展示重点**，不是追溯把 s=1 说成当时的预设主分析。

---

## 4. 数据生成过程（DGP）

```
    r_t = mu_S + sigma_0 * eps_t

    sigma_0   = sigma_ann / sqrt(D) = 0.10 / sqrt(252)
    mu_1(s)   = s * sigma_ann / D            有效
    mu_0      = 0                            无效
    D = 252,  H = 504,  先验 P(valid) = 0.5
```

`eps_t` 由五个生成器之一产生，**全部按理论常数归一化到均值 0、方差 1**，
不做逐路径缩放或去均值（那会把实现样本矩泄漏给检测器）：

| 情境 | eps 定义 | 参数 |
|---|---|---|
| `gaussian` | `z_t` | — |
| `student_t` | `t_ν / a_ν`，`a_ν = sqrt(ν/(ν−2))` | ν=5 |
| `stoch_vol` | `sqrt(v_t) z_t`，`log v_t = A a_t − A²/2`，`a_t = ρa_{t−1} + sqrt(1−ρ²)ξ_t` | ρ=0.98, A=1 |
| `jump` | `(z_t + κ sqrt(K_t) w_t)/c`，`K_t ~ Poisson(λ/D)` | λ=2/年, κ=5 |
| `sv_jump` | `(sqrt(v_t) z_t + κ sqrt(K_t) w_t)/c`，`c = sqrt(1 + κ²λ/D)` | 上两行合并 |

**归一化常数 `c` 的后果**：κ 增大时为保持总方差为 1，非跳跃日反而更安静
（κ=8 时缩小到 0.814 倍）。所以「跳跃更大」不等于「情境更难」。

**Stage 2D 的两个潜在方差量**（只供评分，任何检测器都读不到）：

```
    扩散方差            V_diff = sigma_0^2 * v_t / c^2          条件于 v_t 且当日无跳跃
    给定潜在波动的总方差 V_tot  = sigma_0^2 * (v_t + k^2 λ/D)/c^2  条件于 v_t，不条件于当日跳跃数
```

**跳跃计数与潜在波动状态相互独立**（实测 corr = −0.0024，n=200,000，SE≈0.0022）；
与两者都相关的是**观测到的**大幅收益。

**Stage 3B 的失效时间**：`theta_t = s (t ≤ T)`、`0 (t > T)`，T 是**最后一个有效交易日**。
T 独立于噪声、潜在波动率与跳跃，只对生成器和评分可见。
模型在 T 处**不重置**先验、累计证据、方差估计或任何窗口。

---

## 5. 八个检测方法

统计量一律是 `L_t = log P(valid | r_{1:t}) / P(invalid | r_{1:t})`，**从先验对数赔率出发累加**；
报警规则是 `L_t < 门槛`（严格小于）的**首次穿越**，吸收型（报警后不再改变）。
报告里的 `U = −L`、`q = expit(U)` 是失效方向的对数赔率与概率。

| 方法 | 统计量 | 最早可报警日 |
|---|---|---|
| `binary_gaussian` | `L += z_t·s/√D − s²/(2D)` | 1 |
| `binary_student_t` | Student-t(ν=5) 对数似然比，尺度 `a_ν=sqrt((ν−2)/ν)` | 1 |
| `trailing_sharpe_252` | `√D · mean/std`（252 日滚动，ddof=1） | 252 |
| `known_vol_rolling_252` | `√D · mean/σ₀`（252 日滚动） | 252 |
| `ewma_gaussian` | `L += (μ₁−μ₀)(r_t−m)/V_t` | 1 |
| `ewma_student_t` | Student-t 似然比，尺度 `sqrt((ν−2)/ν · V_t)` | 1 |
| `ewma_trunc_gaussian` | 同上，方差用截断递推 | 1 |
| `ewma_trunc_student_t` | 同上，方差用截断递推 | 1 |

**因果更新顺序（关键，不要改动）**：

```
    V_1 = sigma_daily^2
    第 t 天:  用 V_t（只由 r_1..r_{t-1} 决定）算第 t 天的似然增量
              然后 V_{t+1} = 0.94 V_t + 0.06 e_t^2,   e_t = r_t - m
    截断版本: V_{t+1} = 0.94 V_t + 0.06 min(e_t^2, 16 V_t)     c = 4，未搜索
    m = (mu_0 + mu_1)/2  固定中点，随 s 改变
```

**候选均值必须与生成器同步.** 在 s=0.6 的实验里，生成器漂移、似然比里的 `μ₁`、
以及 EWMA 中点全部用 0.6。只降低生成器而让检测器仍假设 s=1 是**模型设定错误实验**，
本项目没有做。两个滚动方法的统计量不含候选均值，s 只通过门槛与数据进入——这是定义使然。

**概率尺度的限制**：只有对数赔率型统计量才能取 `expit`。
`trailing_sharpe_252` 与 `known_vol_rolling_252` 的统计量是年化 Sharpe 比率，
对它们取 `expit` 无意义，所有概率相关字段对它们留空
（见 `src/strategy_survivorship/stage3b_scales.py`）。

---

## 6. 各阶段的目的、结果与修正记录

| 阶段 | 目的 | 主要结果 | 修正过的错误 |
|---|---|---|---|
| **1** | 最小可复现流程：模拟→检测→校准→评估 | α=15% 下两年只抓到约 59% | 贝叶斯检测器把备择 Sharpe 硬编码为 1；中位检出时间只在已检出路径上算；截断路径与第 504 天报警混淆 |
| **1.1** | 概率-时间对应、随机失效时间初探 | 达到 q=0.9 约需 4.4 年（连续时间参照） | 「Wilson 区间通常是下界」的错误说法；未经验证的因果归因；复制研究功效不足 |
| **2A** | 四种现实噪声下的压力测试 | 冻结门槛迁移尚可；Student-t 在非高斯噪声下占优 | 逐路径自相关估计有偏；Student-t 尺度与标准差混淆 |
| **2B** | EWMA 方差预测是否有用 | **只在 SV 情境有用**（+15.6 pp），别处略有损害 | 连续误杀混淆；T=0 处滚动方法可用天数不足 |
| **2C** | 未知情境下的统一门槛 | 加缓冲后 EWMA 优势仍在（+17.3 pp） | 「全部比较显著」当作停止规则（无效，已撤回） |
| **2D** | SV 与跳跃同时存在 | EWMA Gaussian 的 FAR 在门槛迁移下超标（0.194 / 0.271 对 0.15） | 方差定义漏了归一化常数 `c²`，混淆扩散方差与总方差；`paired_truncated_time` 是死代码 |
| **2E** | 截断方差更新的 2×2 消融 | 截断有用但是小效应（主情境 +0.65 pp） | 报告结论在看到数字前被写死为「负结果」；交互项只报了一个预算 |
| **2E.1** | 收尾与解释修正 | `I(0.05)−I(0.15)` 直接估计：+0.0282 / +0.0392，都排除 0 | 「逐位相同」实为浮点重结合；「长期方差低估 0.0121%」缺乏推导；「跳跃与 SV 本来就相关」与 DGP 不符；「已达信噪比上界」无依据 |
| **3A** | Sharpe 0.6 对 0，保留 1 对 0 | s=1 两年 76.2%，s=0.6 两年 52.3% | 图 1 的 y 轴标签把 L 写成失效对数赔率（只影响显示）；`far_diff` 点估计写成 NaN |
| **3A.1** | 监测期限诊断 | 专为半年校准只是把误杀预算提前花掉 | 中位数取已报警子集；A/B 一致性测试是自比较（恒真） |
| **3B** | 随机失效时间的有限诊断 | 方法排序随失效时刻改变 | 对所有方法算 `working_prob_threshold`；对滚动 Sharpe 优势的过度归因；表头里的 `\|` 打断表格 |

**这张表里的「失败案例」是有意保留的。**负面结果（高斯情境下稳健处理的代价、
门槛迁移失败、弱信号低检出、基线方法在某些设定下胜出）都留在报告里，没有被筛掉。

---

## 7. 校准协议、冻结门槛与随机流

### 门槛怎么来的

每个「规则」= (方法, 情境, Sharpe, 预算)，有时还加 (期限)。门槛是**校准路径上统计量最小值的
第 k 个顺序统计量**，k 由带缓冲的秩给出：找最大的 k 使 `P{Bin(n, α) ≤ k−1} ≤ δ/J`。
两条独立路径（二项式尾、Beta 上尾）互相印证。

| 阶段 | J（同时比较数） | δ/J | n_cal | 秩（α=5% / 15%） |
|---|---|---|---|---|
| 2C | 60 | 8.33e-4 | 10,000 | 433 / 1389 |
| 2D | 60 | 8.33e-4 | 10,000 | 433 / 1389 |
| 2E | 80 | 6.25e-4 | 10,000 | 431 / 1386 |
| 3A | 160 | 3.125e-4 | 10,000 | **427 / 1379** |
| 3A.1 | 128 条规则，仍用 δ/160 | 3.125e-4 | 10,000 | 427 / 1379 |
| 3B | 不校准，读 3A 的 504 日门槛 | — | — | — |

**范围声明**：这些保证覆盖所列的固定情境，**不是**未知环境下的统一保证，
也不是跨全部历史阶段的联合保证。

**门槛在正式测试前冻结**，写入 `outputs/stage*_thresholds.csv`。
测试数据从未参与门槛选择。Stage 3B 特别注意：**不按真实 T 校准**——
真实失效时间在实践中不可得。

读冻结门槛时用 Python 自带的 `float()`，不要用 `pandas` 的快速解析器：
后者在部分十进制串上差 1 ULP，会造成与流程无关的假失败。

### 随机流

`config.py` 的 `STREAM_ORDER` 是**只追加**的列表，用 NumPy `SeedSequence` 派生。
新增流必须加在末尾，这样早期阶段的指纹逐位不变（每次改动后都核验过）。

| 阶段 | 校准流 | 测试流 | bootstrap 流 |
|---|---|---|---|
| 3A | `stage3a_calibration` | `stage3a_test` | `stage3a_bootstrap` |
| 3A.1 | `stage3a1_calibration` | `stage3a1_test` | `stage3a1_bootstrap` |
| 3B | 不需要 | `stage3b_test` | `stage3b_bootstrap`，失效时间用 `stage3b_failure_time` |

### 共享样本（重要）

- **Stage 3A**：两个 Sharpe **共用**每个 (情境, 角色) 的噪声块，故 s=1 与 s=0.6 的比较是**配对**的。
  150,000 条基础噪声路径产生 300,000 次收益路径评估——**不是** 300,000 条独立路径。
- **Stage 3A.1**：四个期限是**同一条统计量路径的前缀**，不是独立样本。
- **Stage 3B**：每情境 10,000 条始终有效对照 + 10,000 条供全部失效设定共用；
  共 40,000 条基础噪声路径，各设定之间不独立。

### 区间的含义

| 阶段 | 区间覆盖什么 |
|---|---|
| 2C–3A.1 的主 bootstrap | 校准 + 测试抽样（每次重抽重算门槛） |
| **3B 的 bootstrap** | **只有测试抽样**；门槛在 3A 已冻结，**不重新涵盖校准误差** |
| Wilson 区间 | 单个比率的二项区间 |

**所有区间都是逐项区间**，未做多重比较校正。校准协议的 δ/J 保护的是**门槛的误杀约束**，
不为检出率之差提供同时置信保证。

---

## 8. 指标定义（逐条）

| 指标 | 定义 | 陷阱 |
|---|---|---|
| 累计误杀率 FAR | `P_{有效}(τ ≤ d)` | 一个门槛对应一次预算，不是每个期限各花一次 |
| 累计检出率 | `P_{无效}(τ ≤ d)` | — |
| **中位检测时间** | **全部路径**的累计报警比例**首次达到 1/2** 的那一天；恰好 1/2 算达到；未达到则标记未达到 | **不是已报警子集的中位数**。从不报警的路径不是缺失数据，是大于期限的检测时间，必须留在分母。旧实现在这里错过两次 |
| 截断平均时间 | `E[min(τ, H)]`，未检出记为 H | 上限随 H 变化，**跨期限不可比** |
| 未检出比例 | `1 − P(τ ≤ H)` | — |
| **失效前误杀** | `P(τ ≤ T)` | `τ=∞` 编码为 −1，**绝不能**当作小数值算进来 |
| **条件检出** | `P(T < τ ≤ T+h \| τ > T)` | 分母是存活者，每次 bootstrap 重抽都要**重算** |
| **联合检出** | `P(T < τ ≤ T+h)` = 存活率 × 条件检出 | 恒等式在每行核验，最大偏差 2.8e-17 |
| 失效后延迟 | `E[min(τ−T, 252) \| τ > T]` | 未检出的存活者按上限计入**不删去**；提前误杀**不是**零延迟，它根本不在存活集合里 |
| 工作概率门槛 | `q* = expit(−门槛)`，规则 `L < 门槛` 等价于 `q > q*` | **只对对数赔率统计量有定义** |
| 工作失效分数 | 静态两状态分类器在切换数据上的 `q` | **不是**对切换过程的正确后验 |

---

## 9. 结论 → 数据出处

| 结论 | CSV | 筛选条件 | 图 | 生成命令 |
|---|---|---|---|---|
| 稳健处理的三步贡献 | `outputs/stage2e_metrics.csv` | `arm=per_scenario, far_target=0.15` | `docs/figures/fig1_*.png` | `make brief` |
| s=1 两年 76.2%，一年 48.7% | `outputs/stage3a_metrics.csv` | `scenario=sv_jump, sharpe_valid=1.0, far_target=0.15` | `docs/figures/fig2_*.png` | `make brief` |
| s=0.6 两年 52.3% | 同上 | `sharpe_valid=0.6` | 同上 | 同上 |
| 高斯上界 64.7% / 42.6% | `outputs/stage3a_gaussian_reference.csv` | `years=2.0` | — | `run_stage3a` |
| 期限重设只是提前花预算 | `outputs/stage3a1_bootstrap.csv` | `kind∈{detect,far}_B_minus_A, sharpe_valid=1.0, cutoff_H=126` | `docs/figures/fig3_*.png` | `make brief` |
| 短门槛延用到两年误杀 40.3% | `outputs/stage3a1_out_of_horizon.csv` | `threshold_calibrated_over=63, sharpe_valid=1.0` | `outputs/figures/fig3a1_3_*.png` | `run_stage3a1 --figures-only` |
| 失效时刻改变方法排序 | `outputs/stage3b_bootstrap.csv` | `sharpe_valid=1.0, quantity=joint, method_b=trailing_sharpe_252` | `docs/figures/fig4_*.png` | `make brief` |
| 截断的额外贡献 +0.2~1.2 pp | `outputs/stage3b_metrics.csv` | `sharpe_valid=1.0, far_target=0.15` | — | `run_stage3b --report-only` |
| 跨预算交互项 | `outputs/stage2e1_cross_budget.csv` | `contrast="I(0.05) - I(0.15)"` | — | `run_stage2e1` |

α=5% 的完整结果在同样的 CSV 里（把 `far_target` 换成 0.05），
以及各阶段报告的补充章节。

---

## 10. 已知限制

1. **全部为模拟。**没有任何真实市场数据，也没有部署验证。
2. **候选 Sharpe 只有 {1, 0.6}**，检测器知道候选值。真实情况下候选未知。
3. **噪声情境有限**：五个情境都出自同一族 SV+跳跃 DGP。
4. **Stage 3B 只覆盖第一年内失效**（T ≤ 252），不是多年寿命研究。
5. **校准只覆盖固定情境**，不是未知环境保证。
6. **等待成本未纳入评价**：只有检出率与误杀率两个指标。
7. **区间是逐项的**，未做多重比较校正；Stage 3B 的区间还不覆盖校准误差。
8. **多处机制假设未被检验**：滚动 Sharpe 在 T=252 的优势成因、
   紧预算下 Gaussian 版本的瓶颈，都只是与数据相容的读法。

---

## 11. 参考文献与用途

| 文献 | 在本项目里的用途 |
|---|---|
| Neyman–Pearson 引理 | `theory.md` §17 高斯截止时间参照的推导依据 |
| 序贯变点检测综述 | 理解累计证据、变点检测与误杀—延迟权衡；未来方向 |
| BOCPD 原论文（Adams & MacKay 2007） | 随机 T 之后的候选方向，**本项目未实现** |
| Gelman & Stern，「显著与不显著之差」 | 修正交互项与跨预算的比较方式（`theory.md` §16.5） |
| 中位生存时间（含删失）与截断平均时间 | 统一检测时间的统计口径（本文件 §8） |
| Lo (2002)，Sharpe 比率的统计 | 样本 Sharpe 的误差与时间尺度换算 |
| Lucas & Zhang，Score-Driven EWMA | 稳健波动率更新的背景，本项目未采用 GAS |
| NumPy `SeedSequence` 文档 | 可复现的独立随机流设计 |

---

## 12. 交接检查清单

接手时确认：

- [ ] `git log -1` 与本文件 §1 记录的整理版本一致
- [ ] `.venv/bin/python -m pytest -q` 全绿
- [ ] `make brief` 能重画四张主图且与 `docs/figures/` 现有文件一致
- [ ] `docs/BRIEF.md` 里的数字能在 §9 指出的 CSV 里查到
- [ ] 已读导师反馈，并据此确认下一步优先级

**然后停下来问，不要直接开始新实验。**
