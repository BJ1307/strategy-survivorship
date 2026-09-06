# 第一阶段报告：策略有效性监控基线

本文件由 `python -m strategy_survivorship.run_stage1` 自动生成，所有数字与同一次运行产生的 CSV、JSON 和图片完全一致。

## 1. 本阶段做了什么

导师的问题是：一个新策略开始样本外运行之后，它究竟真的有收益优势，还是从第一天起就没有优势。本阶段只建立**最小但完整**的评估流程，不解决厚尾、时变波动率、跳跃或“先有效后失效”这些后续问题。

流程固定为五步，全部由一条命令驱动：

1. **模拟**：固定波动率、独立高斯日收益，真实年化 Sharpe 为 1（有效）或 0（无效）。
2. **逐日更新**：四个检测器每天输出一个“越大越支持有效”的统计量。
3. **阈值校准**：只用 5000 条**有效**校准路径选阈值，使整个 504 日监控期内的累计误杀率不超过目标。
4. **独立评估**：阈值冻结后，在从未参与校准的测试集上测量误杀率、检出率和检测时间。
5. **图表与报告**：四张图、若干表格和本报告。

另外做了一个**单次冲击诊断**：一条预先固定的有效路径，在第 252 天分别加上 ±8σ_d 的冲击，只用来解释 Gaussian 与 Student-t 的更新行为差异。它**不进入**正式 benchmark 的任何误杀率或检出率数字。

## 2. 实验参数

| 参数 | 符号 | 取值 |
|---|---|---|
| 每年交易日数 | D | 252 |
| 监控期限 | H | 504 交易日（2 年） |
| 年化波动率 | σ_ann | 0.1 |
| 日波动率 | σ_d = σ_ann/√D | 0.00629941 |
| 有效状态真实年化 Sharpe | S | 1 |
| 无效状态真实年化 Sharpe | S | 0 |
| 贝叶斯初始有效概率 | P(valid) | 0.5（log odds = 0） |
| Student-t 自由度 | ν | 5 |
| Student-t 尺度 | a_ν = √((ν−2)/ν) | 0.77459667 |
| 滚动窗口 | W | 252 交易日，样本标准差 ddof=1 |
| 累计误杀率目标 | α | 0.05, 0.15 |
| 评估时点 | — | 第 126, 252, 504 交易日（半年 / 一年 / 两年） |
| 根随机种子 | — | 20260905 |
| 校准集 | — | 5000 条有效路径 |
| 测试集 | — | 5000 条有效 + 5000 条无效路径 |

日收益生成过程：

```
r_t = S * σ_ann / D  +  (σ_ann / √D) * ε_t ,   ε_t ~ iid N(0, 1)
```

收益已理解为**扣除成本后**的策略超额收益，本阶段不再额外扣一次成本。真实 Sharpe 是生成过程的参数：路径生成后**没有**任何事后平移或重新缩放，每条路径的样本 Sharpe 保持其自然的随机波动。

随机流：以 `SeedSequence(20260905)` 为根，按固定顺序 `calibration_valid / test_valid / test_invalid / diagnostic / prob_time_valid / prob_time_invalid / switch_fixed / switch_random / switch_matched_cal` spawn 出四条互相独立的子流。所有检测器评价的是**同一份**测试路径，没有任何模型使用自己单独生成的收益。

## 3. 四个检测器

| 检测器 | 角色 | 统计量 | 最早可报警日 |
|---|---|---|---|
| Binary Gaussian | baseline | posterior log-odds L_t | 第 1 天 |
| Binary Student-t (nu=5) | baseline | posterior log-odds L_t | 第 1 天 |
| Trailing 12m Sharpe | baseline | annualised trailing Sharpe (ddof=1) | 第 252 天 |
| Known-vol rolling (control) | diagnostic control | annualised trailing mean / known sigma | 第 252 天 |
| Random closure (analytic) | 解析参考 | 无（每日独立抛硬币） | 第 1 天 |

统一决策协议：**统计量严格小于阈值即报警**；首次报警后决策关闭，τ 是首次穿越时间，不可重开。
尚未定义的日子（滚动模型的前 251 天）记为 NaN，既不参与校准也不能报警。

> **运行方式差异（重要）**：两个贝叶斯模型从第 1 天就开始更新，两个滚动模型要等满 252 天窗口才发声。本阶段测到的性能差距里包含了这一运行方式差异，**不能**全部解释成贝叶斯公式本身更优。此外主实验中日波动率对所有检测器都是**已知**的理想条件；known-vol rolling 控制组的存在正是为了分离“估计波动率”这一项的代价。

## 4. 阈值校准

累计误杀率定义为 `F_H = Pr_{S=1}(τ ≤ H)`。一条路径在监控期内报警，当且仅当它在可报警日上的**统计量最小值**严格低于阈值，因此阈值直接取校准集上这些最小值的第 `floor(α·N)` 个顺序统计量（0 基）。配合严格小于的报警规则，经验累计误杀率 `#{min < c}/N ≤ floor(α·N)/N ≤ α`，并列值只会把实际误杀率往下压，不会超过目标。缺失值不参与最小值计算。

| 检测器 | α | 阈值 | 最早可报警日 | 校准集实际误杀率 |
|---|---|---|---|---|
| Binary Gaussian | 0.05 | -1.958043 | 1 | 0.0500 |
| Binary Student-t (nu=5) | 0.05 | -2.411874 | 1 | 0.0500 |
| Trailing 12m Sharpe | 0.05 | -1.486472 | 252 | 0.0500 |
| Known-vol rolling (control) | 0.05 | -1.470119 | 252 | 0.0500 |
| Binary Gaussian | 0.15 | -1.332639 | 1 | 0.1500 |
| Binary Student-t (nu=5) | 0.15 | -1.629865 | 1 | 0.1500 |
| Trailing 12m Sharpe | 0.15 | -0.956320 | 252 | 0.1500 |
| Known-vol rolling (control) | 0.15 | -0.952674 | 252 | 0.1500 |

> 注意最后一列：`floor(α·N)/N` 在 N = 5000、α = 0.05 和 0.15 时正好等于 α，所以“校准集实际误杀率”是这条规则的**机械结果**，不是对方法有效性的证据。真正有信息量的是下一节测试集上的数字。

阈值一经确定即冻结，测试集结果**没有**被用来回调阈值，也没有挑选种子。

## 5. 核心结果（独立测试集）

### 5.1 误杀率目标 α = 0.05

**有效策略上的累计误杀率**（n = 5000，括号内为逐点 Wilson 95% 区间）

| 检测器 | 半年 (126d) | 一年 (252d) | 两年 (504d) |
|---|---|---|---|
| Binary Gaussian | 0.0032 [0.0020, 0.0052] | 0.0174 [0.0141, 0.0214] | 0.0516 [0.0458, 0.0581] |
| Binary Student-t (nu=5) | 0.0024 [0.0014, 0.0042] | 0.0178 [0.0145, 0.0219] | 0.0538 [0.0479, 0.0604] |
| Trailing 12m Sharpe | 0.0000 [0.0000, 0.0008] | 0.0070 [0.0050, 0.0097] | 0.0526 [0.0467, 0.0591] |
| Known-vol rolling (control) | 0.0000 [0.0000, 0.0008] | 0.0076 [0.0055, 0.0104] | 0.0532 [0.0473, 0.0598] |
| Random closure (analytic) | 0.0127 — | 0.0253 — | 0.0500 — |

**无效策略上的累计检出率**（n = 5000）

| 检测器 | 半年 (126d) | 一年 (252d) | 两年 (504d) | 两年仍未检出 |
|---|---|---|---|---|
| Binary Gaussian | 0.0156 [0.0125, 0.0194] | 0.1140 [0.1055, 0.1231] | 0.3682 [0.3549, 0.3817] | 0.6318 [0.6183, 0.6451] |
| Binary Student-t (nu=5) | 0.0118 [0.0092, 0.0152] | 0.0998 [0.0918, 0.1084] | 0.3386 [0.3256, 0.3518] | 0.6614 [0.6482, 0.6744] |
| Trailing 12m Sharpe | 0.0000 [0.0000, 0.0008] | 0.0692 [0.0625, 0.0766] | 0.3086 [0.2959, 0.3215] | 0.6914 [0.6785, 0.7041] |
| Known-vol rolling (control) | 0.0000 [0.0000, 0.0008] | 0.0702 [0.0634, 0.0776] | 0.3124 [0.2997, 0.3254] | 0.6876 [0.6746, 0.7003] |
| Random closure (analytic) | 0.0127 — | 0.0253 — | 0.0500 — | 0.9500 — |

**检测时间**（无效策略；未检出按 H = 504 天计入截断等待时间）

| 检测器 | 截断平均检测时间 (交易日) | (年) | 总体中位检测时间 |
|---|---|---|---|
| Binary Gaussian | 435.0 ± 1.6 | 1.726 | 在观察期内未达到（>504 天） |
| Binary Student-t (nu=5) | 441.7 ± 1.5 | 1.753 | 在观察期内未达到（>504 天） |
| Trailing 12m Sharpe | 452.7 ± 1.3 | 1.797 | 在观察期内未达到（>504 天） |
| Known-vol rolling (control) | 452.1 ± 1.3 | 1.794 | 在观察期内未达到（>504 天） |
| Random closure (analytic) | 491.3 | 1.950 | 在观察期内未达到（>504 天） |

### 5.2 误杀率目标 α = 0.15

**有效策略上的累计误杀率**（n = 5000，括号内为逐点 Wilson 95% 区间）

| 检测器 | 半年 (126d) | 一年 (252d) | 两年 (504d) |
|---|---|---|---|
| Binary Gaussian | 0.0240 [0.0201, 0.0286] | 0.0788 [0.0717, 0.0866] | 0.1410 [0.1316, 0.1509] |
| Binary Student-t (nu=5) | 0.0264 [0.0223, 0.0312] | 0.0848 [0.0774, 0.0928] | 0.1520 [0.1423, 0.1622] |
| Trailing 12m Sharpe | 0.0000 [0.0000, 0.0008] | 0.0270 [0.0229, 0.0319] | 0.1576 [0.1478, 0.1680] |
| Known-vol rolling (control) | 0.0000 [0.0000, 0.0008] | 0.0266 [0.0225, 0.0314] | 0.1562 [0.1464, 0.1665] |
| Random closure (analytic) | 0.0398 — | 0.0780 — | 0.1500 — |

**无效策略上的累计检出率**（n = 5000）

| 检测器 | 半年 (126d) | 一年 (252d) | 两年 (504d) | 两年仍未检出 |
|---|---|---|---|---|
| Binary Gaussian | 0.1022 [0.0941, 0.1109] | 0.3100 [0.2973, 0.3230] | 0.5860 [0.5723, 0.5996] | 0.4140 [0.4004, 0.4277] |
| Binary Student-t (nu=5) | 0.0930 [0.0853, 0.1014] | 0.2998 [0.2873, 0.3126] | 0.5642 [0.5504, 0.5779] | 0.4358 [0.4221, 0.4496] |
| Trailing 12m Sharpe | 0.0000 [0.0000, 0.0008] | 0.1642 [0.1542, 0.1747] | 0.5564 [0.5426, 0.5701] | 0.4436 [0.4299, 0.4574] |
| Known-vol rolling (control) | 0.0000 [0.0000, 0.0008] | 0.1656 [0.1556, 0.1762] | 0.5572 [0.5434, 0.5709] | 0.4428 [0.4291, 0.4566] |
| Random closure (analytic) | 0.0398 — | 0.0780 — | 0.1500 — | 0.8500 — |

**检测时间**（无效策略；未检出按 H = 504 天计入截断等待时间）

| 检测器 | 截断平均检测时间 (交易日) | (年) | 总体中位检测时间 |
|---|---|---|---|
| Binary Gaussian | 358.0 ± 2.2 | 1.421 | 409 天 |
| Binary Student-t (nu=5) | 363.1 ± 2.2 | 1.441 | 425 天 |
| Trailing 12m Sharpe | 402.9 ± 1.5 | 1.599 | 451 天 |
| Known-vol rolling (control) | 402.6 ± 1.5 | 1.597 | 450 天 |
| Random closure (analytic) | 465.3 | 1.846 | 在观察期内未达到（>504 天） |

### 5.3 怎么读这些数字

比较应当围绕**实际误杀率、检出曲线和等待时间**三者一起看。本节任何排序都只在**当前匹配的高斯 DGP** 与**当前已实现的决策规则**下成立，不构成普遍最优的结论；Stage 1.1 的随机失效时间诊断显示，一旦策略先有效后失效，这个排序会反转：

- 校准把每个检测器**在校准集上**压到同一个误杀率预算，因此检出率之间的比较是在大致相同的误杀成本下进行的。但测试集上的实际误杀率仍会围绕目标波动，读检出率时必须同时看同一行的实际误杀率。
- 两个滚动模型在第 252 天之前既不会误杀也不可能检出，它们的曲线在前一年恒为 0。这是运行方式，不是性能。
- **两个贝叶斯检测器在本轮享有一个结构性优势**：它们的似然恰好就是真实生成过程（Gaussian 版本连噪声分布都完全正确），而且真实的备择假设 S = 1 正是它们两个候选假设之一。滚动 Sharpe 是一个不知道备择假设、也不假设噪声分布的通用估计量。因此这里的差距同时包含了“递推方式”和“模型设定恰好正确”两件事，后者在真实场景中不会免费获得。
- Random closure 是解析参考线，对有效和无效策略给出**完全相同**的累计报警概率；任何一个真正在利用数据的检测器，其检出曲线都应当明显高于它。
- **known-vol 控制组回答了它被造出来的那个问题**：把滚动 Sharpe 的分母换成已知日波动率之后，检出率与截断平均检测时间几乎没有变化（见上表两行的差异）。也就是说在 252 天窗口、n = 5000 的规模下，**估计波动率本身几乎不构成代价**。至于滚动模型落后的**原因**是什么，本阶段没有做能够分离的实验：要归因于启动延迟，必须另做一组让所有检测器都从第 252 天才允许报警、并各自独立校准的对照。在那之前不对原因下结论。
- 本轮数据是高斯的，Student-t 检测器属于**似然失配**模型，它在这里表现不如 Gaussian 是预期之中的结果，不是需要修掉的 bug；我们也没有为了让它好看而调整数据或参数。

### 5.4 两种不同的不确定性，不要互相当作对方的界

正文表格里的 Wilson 区间回答的是：**给定这次已经冻结的门槛**，在 5,000 条独立测试路径上测到的误杀率有多少蒙特卡洛噪声。对这个问题它是合适的区间估计，**不是**下界。

另一个问题是：**整条流程重新走一遍**（重新抽校准集、重新选门槛、重新抽测试集），实测误杀率会有多大波动。这两个问题不同，答案也不同，任何一个都不应被当作另一个的界。

第二个问题有**精确解**，不需要昂贵的重复模拟：门槛是校准集路径最小值的第 `j = ⌊αN⌋+1` 个顺序统计量，在最小值分布连续时其真实误杀概率服从 `Beta(j, N+1−j)`，测试集报警数服从 `BetaBinomial(n_test, j, N+1−j)`。推导、数值与同 100 次重复模拟的一致性检验见 Stage 1.1 报告第 3 节（`outputs/stage11_report.md`）与 `stage11_analytic_calibration.csv`。

## 6. 单次冲击诊断

取 `diagnostic` 随机流的第 0 条**有效**路径（种子与路径编号都预先固定），复制成两份，分别在第 252 天加上 ±8σ_d。除该天外三条序列完全相同。

冲击当天的一步对数似然增量：

| 模型 | 原始 | +8σ_d | −8σ_d |
|---|---|---|---|
| Binary Gaussian | -0.039691 | +0.464261 | -0.543644 |
| Binary Student-t (ν=5) | -0.070088 | +0.048601 | -0.042101 |

第 504 天的 log odds：

| 模型 | 原始 | +8σ_d | −8σ_d |
|---|---|---|---|
| Binary Gaussian | +0.5205 | +1.0245 | +0.0166 |
| Binary Student-t (ν=5) | +0.7729 | +0.8916 | +0.8009 |

第 504 天 log odds 相对原始路径的位移（冲击的永久影响）：

| 模型 | +8σ_d 位移 | −8σ_d 位移 |
|---|---|---|
| Binary Gaussian | +0.50395 | -0.50395 |
| Binary Student-t (ν=5) | +0.11869 | +0.02799 |

**Gaussian**：一步增量为 `z_t/√D − 1/(2D)`，对 z 完全线性且无界。冲击把当天的 z 从 -0.5986 推到 +7.4014 / -8.5986，于是 log odds 恰好位移 ±8/√D = ±0.50395，并且因为统计量是增量的累计和，这个位移**永久保留**到监控期结束。

**Student-t**：增量不是单调放大的，而是**先升后降（redescending）**的影响函数。|增量| 在 z ≈ -1.70 处达到峰值 0.1091，此后随 |z| 增大反而衰减回 0：

| z | Gaussian 增量 | Student-t 增量 |
|---|---|---|
| -1 | -0.06498 | -0.09592 |
| -2 | -0.12797 | -0.10773 |
| -8 | -0.50594 | -0.04497 |
| -20 | -1.26187 | -0.01873 |
| -50 | -3.15169 | -0.00755 |

原因是两个假设的位置参数只相差 1/√D ≈ 0.0630，在 t 密度的尾部这点差异被巨大的 |z| 淹没，两条对数密度之差趋于 0：**足够极端的观测在两个状态下几乎同样不可能，因此几乎不携带证据**。

这带来一个值得记住的后果：本例中 −8σ_d 冲击让 Student-t 的终点 log odds **上升** +0.02799，方向与冲击相反。原始那天的 z = -0.5986 落在影响函数斜率最陡的区域，被推到 z = -8.5986 的极端尾部之后，它提供的反面证据反而**变少**了。这不是实现错误，而是重尾似然降权的直接推论；图 4 最下方的影响函数面板画出了整条曲线。

> 这只是**一次观测**的降权。它不等于解决了持续性随机波动率问题：如果波动率本身在一段时间内变化，Student-t 似然并不会因此校正对 Sharpe 的推断。该问题留给后续阶段。

图中在首次报警之后仍继续画出统计量，但 τ 始终按第一次穿越记录。

## 7. 验证

验证针对的是**会改变结论**的风险：单位换算、递推公式、数据隔离、窗口边界、首次穿越与截断等待时间。运行 `pytest` 即可复现。

主要检查项：

- 年化↔日频换算；生成过程的均值与方差同设定一致（Monte-Carlo 容差按标准误设定）。
- Binary Gaussian 的逐步递推与直接累加两个正态 logpdf 之差在 1e-12 以内。
- Gaussian log odds 的解析矩：`E[L_n|S=1]=n/(2D)`、`E[L_n|S=0]=−n/(2D)`、`Var(L_n|S)=n/D`。
- Student-t 的尺度对应**单位方差**；ν 增大时其更新收敛到 Gaussian 更新。
- 在线（逐日）输出与批量计算一致；修改未来收益不改变过去的统计量和首次报警时间。
- 滚动窗口起始日与边界正确，前 251 天为 NaN，无未来数据填充，非 expanding window。
- 校准集与测试集真正分离；四个检测器确实使用同一份测试路径数组。
- 首次穿越、从未穿越、最后一天穿越三种情形，以及截断等待时间的处理。
- 阈值规则在有并列值时仍满足经验误杀率 ≤ α。

## 8. 发现并修正的问题

1. **Student-t scale confused with standard deviation** — A t(nu) with scale a has variance a^2 * nu/(nu-2). Using scale = 1 would have given the Student-t detector observation noise of variance nu/(nu-2) = 5/3 in z-space, i.e. a different (easier) problem from the Gaussian detector's. Fixed by a_nu = sqrt((nu-2)/nu), and pinned by a unit-variance test.
2. **Rolling detectors were credited with days they cannot see** — A 252-day trailing window is undefined for the first 251 days. Those days are held as NaN, excluded from the calibration minimum and unable to alarm, instead of being back-filled or silently treated as an expanding window.
3. **Threshold quantile could overshoot the FAR budget** — Taking the empirical alpha-quantile of the per-path minima with a non-strict alarm rule lets the achieved calibration FAR land above alpha when there are ties. Fixed by pairing a strict '<' alarm rule with the floor(alpha*N)-th order statistic, which is conservative by construction, and asserting achieved <= target in the run.
4. **Censored paths were indistinguishable from day-504 alarms** — Both give min(tau, H) = 504. The first-passage table now stores the alarm flag and a -1 sentinel separately from the truncated time, so a genuine day-504 crossing and a never-crossing path stay distinguishable.
5. **Median detection time conditional on detection** — Reporting the median only over detected paths flatters detectors that rarely fire. The median is computed over all invalid paths, and reported as 'not reached within the horizon' when fewer than half alarm by H.
6. **Wilson intervals were the only stated uncertainty** — The pointwise Wilson interval answers 'given THIS frozen threshold, how noisy is the rate measured on 5,000 fresh test paths'. That was the only uncertainty reported, so the report said nothing about how much the whole pipeline moves when the calibration sample is redrawn. Stage 1.1 answers the second question exactly -- the threshold is an order statistic, so its true false-alarm probability is Beta(j, N+1-j) with j = floor(alpha*N)+1 -- and the report now keeps the two questions separate instead of treating one as a bound on the other.
7. **The stopping rule for the replication study was invalid** — The replication count was raised from 20 to 100 with the stated reason that at 20 'two of the eight combinations returned an inflation factor below 1' and at 100 'all eight reject'. Deciding when to stop sampling by looking at significance biases the result towards significance, and an empirical sd landing below a reference value in a small sample is ordinary sampling noise, not evidence that the theory is wrong. Both the rule and the reasoning are withdrawn. Stage 1.1 replaces the study with the exact Beta / Beta-binomial law and keeps the 100 replications only as a one-off cross-check (it agrees, max |z| = 1.57); the replication stage is no longer part of the default run.
8. **An unverified causal attribution about the rolling detectors** — The report stated that the rolling detectors lag 'because of the one-year start-up delay and the equal weighting inside the window'. No experiment isolated that: establishing it needs a control in which every detector may only alarm from day 252 and each is calibrated independently. The claim is deleted; only what the known-vol control actually shows is kept.
9. **The Gaussian ranking was stated without its scope** — Gaussian leads only under the matched Gaussian DGP and the decision rules as implemented. Stage 1.1's random-failure-time diagnostic shows the ranking reverses once a strategy is valid first and fails later, so the Stage 1 comparison is now explicitly scoped and cross-references that result.

## 9. 限制

- The benchmark DGP is fixed-volatility iid Gaussian. Fat tails, time-varying volatility, jumps and valid-then-decaying strategies are out of scope for this stage, so no conclusion here transfers to them.
- The daily volatility is *known* to every detector. This is an idealised condition that favours all four detectors, and the known-vol rolling control exists precisely to show how much of the rolling detectors' handicap is the volatility estimate.
- The two Bayesian detectors update from day 1 while the two rolling detectors are silent until day 252. Part of the measured gap is this difference in operating regime, but no experiment here isolates how much, so the cause is left open rather than attributed.
- Every Stage 1 result assumes the state is CONSTANT over the whole horizon. Stage 1.1 shows the detector ranking reverses once the strategy is valid first and fails later, so the Stage 1 ranking should not be carried into any decaying-strategy setting.
- Both Bayesian detectors assume the true Sharpe is exactly 0 or exactly 1, and under this DGP that assumption is exactly right -- the true alternative is literally one of the two hypotheses, and the Gaussian detector's noise law is correct as well. The trailing Sharpe assumes neither. A meaningful part of the measured gap is therefore correct specification, which is not free in practice; a continuous or three-state prior is deferred to a later stage.
- Thresholds are calibrated on a finite (5,000-path) calibration sample, so the frozen threshold's true false-alarm probability is a random variable. Its exact law is Beta(j, N+1-j) with j = floor(alpha*N)+1; see Stage 1.1 section 3. Note E[p_FA] = j/(N+1) sits slightly ABOVE the nominal alpha.
- The reported Wilson intervals are pointwise and conditional on the frozen threshold: they cover Monte-Carlo noise in one rate at one day, not the whole time curve simultaneously, and not the spread induced by redrawing the calibration sample. That second spread is a different question with its own exact answer (Stage 1.1 section 3); neither bounds the other.
- The Student-t detector is evaluated only under a Gaussian DGP here, where it is mis-specified by construction. Down-weighting one outlier is not a solution to persistent stochastic volatility, and nothing in this stage tests that claim.
- The single-shock diagnostic uses one pre-registered path. It explains update behaviour and nothing else; it contributes no detection or false-alarm estimate.

## 10. 复现

```bash
uv venv --python 3.13.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest
.venv/bin/python -m strategy_survivorship.run_stage1
```

| 项 | 值 |
|---|---|
| Python | 3.13.13 (CPython) |
| 平台 | macOS-14.2-arm64-arm-64bit-Mach-O / arm64 |
| NumPy / SciPy / pandas / Matplotlib | 2.5.2 / 1.18.1 / 3.0.5 / 3.11.1 |
| 总运行耗时 | 2.4 s |
| 分阶段耗时 | simulate 0.1s，testtatistics 1.0s，calibration 0.6s，evaluation 0.3s，diagnostic 0.0s，tables 0.4s |

**输出文件**

| 文件 | 内容 |
|---|---|
| `stage1_report.md` | 本报告 |
| `stage1_metrics.csv` | 每个检测器 × 每个 α 的主要指标与区间 |
| `stage1_summary.json` | 参数、样本规模、阈值、指标、诊断、耗时、限制的紧凑汇总 |
| `stage1_first_passages.csv` | 每条测试路径的真实状态、是否报警、首次报警日、截断时间 |
| `stage1_curves.csv` | 每日累计报警率曲线（逐检测器 / α / 状态） |
| `stage1_calibration_robustness.csv` | 多种子复算下阈值与实际误杀率的分布 |
| `stage1_diagnostic_traces.csv` | 冲击诊断的逐日收益、增量、log odds 和概率 |
| `run_metadata.json` | 运行配置、随机流指纹、环境版本、耗时 |
| `figures/fig1_example_paths.png` | 固定示例路径的累计收益与两个贝叶斯概率 |
| `figures/fig2_operating_curves.png` | 测试集累计误杀率与累计检出率随时间变化 |
| `figures/fig3_detection_time.png` | 截断平均检测时间与两年未检出比例 |
| `figures/fig4_shock_diagnostic.png` | 原始 / ±冲击下 Gaussian 与 Student-t 的更新差异 |

公式、符号、单位与推导见仓库根目录的 `theory.md`。
