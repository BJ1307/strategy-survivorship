# 最终六页演示稿规格（可编辑，交给任意一台机器上的 Claude Code）

状态：**已完成并交付**（2026-09-11）。本文件记录这份稿子的规格与约束，
供跨电脑接手时直接参照。修改前请先读 §0。

---

## 0. 接手前必须知道的三件事

1. **交付物只有一份六页正文。** 没有备用页、没有附录、没有独立备用 PDF。
   `backup.tex` 已从交付包中移除；历史研究文件仍留在仓库，不必删除。
2. **不新增实验、不新增模型、不重跑全部阶段。** EWMA 保留为已有对照，
   固定 Student-t 是解释主线。不要重新引入有限记忆、变点模型、LSTM、
   参数优化或新的模拟方案。
3. **Downloads 里有多个同名相近的文件。** 要上传给审阅者的是
   `strategy-survivorship-overleaf-eb7b73d.zip`（当前版）与
   `strategy_survivorship_FINAL_6pages_2026-09-11.pdf`。
   名字里带 `BACKUP_before_mixed` 或 `ORIGINAL-BACKUP` 的都是**旧版**，
   不要拿它们覆盖新版。

---

## 1. 汇报主线（固定）

> 我们逐日监测策略收益；先在几类模拟环境中比较方法，再让不同策略具有不同噪声参数。
> 新混合实验中，固定 Student-t 比 Gaussian 的检出率高约五个百分点；
> EWMA 的额外改善较小且区间包含零。统一门槛下，总体误杀接近预算，
> 但某些子群仍有更高误杀。下一步保持现有方法，检查真实策略收益。

导师的要求：模型复杂度可以接受，但**解释要简单**，**不要把人工生成环境中的
优势讲成普遍优势**，并开始转向真实数据。

---

## 2. 六页结构

| 页 | 标题 | 内容 | 图/表 |
|---|---|---|---|
| 1 | Can we tell whether a strategy is effective? | 观察每日超额收益；模拟中有效 Sharpe 1、无效 0；监测器只看到收益；状态从上线起不变；单条路径仅建立直觉 | `n01_paths.png` |
| 2 | How do we simulate different strategies? | 尾部频率 + 绝对收益自相关；**这两张图是五个固定情境的诊断**；本轮新增 `A_i~U(0,1)`、`kappa_i~U(0,8)` | `n03_dgp_ab.png` |
| 3 | How does the monitor decide? | 自然语言等式；Student-t 降低对极端收益的敏感度；越过门槛即报警；detection / false alarms 定义；两年 15% 预算；先在独立数据选门槛再冻结 | 无 |
| 4 | Performance depends on the simulated environment | Stage 2E 五个固定情境，三方法（固定 G / 固定 t / EWMA t），两年检出与实测误杀 | `n04_scenarios.png` |
| 5 | What happens when strategies have different noise? | mixed_noise 四行两列表 + 两个预设比较 + 一行时间摘要 | 表格 |
| 6 | What have we learned, and what comes next? | Completed / Limit / Next + 两个英文问题 | 无 |

每页一个核心问题。讲稿写在 `\note{}` 里，围绕「问题、如何读图、一个结果、一个限制」。

---

## 3. 数值（全部已对 CSV 核验）

### mixed_noise（第 5 页）— `outputs/mixed_noise_*.csv`

| 方法 | 两年检出 | 两年误杀 |
|---|---|---|
| 固定 Gaussian | 56.70% | 14.10% |
| 固定 Student-t | 61.64% | 14.34% |
| 滚动一年 Sharpe | 55.99% | 14.81% |
| EWMA Student-t | 62.64% | 14.20% |

预设差异（1,000 次配对 bootstrap，整条路径重抽，校准与测试同时重抽，
每次重新校准门槛，逐项 95% 区间）：

- 固定 t − 固定 G：**+4.94 pp**，[3.96, 6.09]
- EWMA t − 固定 t：**+1.00 pp**，[−0.24, 1.88]

固定 Student-t 的累计检出：126 日 11.8%，252 日 34.2%，504 日 61.6%。

子群（按 `A=0.5`、`kappa=4` 预先划分，同一冻结门槛读回）：
固定 t 在**低 A、小跳跃**子群误杀 **17.46%**，[16.02, 19.00]，n=2486；
同方法在高 A、大跳跃子群为 11.23%。

### Stage 2E（第 4 页）— `outputs/stage2e_metrics.csv`

筛选 `arm=per_scenario`、`far_target=0.15`。纯高斯情境两年检出：
固定 Gaussian 56.57%、固定 Student-t 53.37%、EWMA Student-t 52.58%
（稳健处理在那里是**代价**，必须保留）。

**两个实验不可互换**：第 4 页来自 Stage 2E（逐情境校准），
第 5 页来自 mixed_noise（随机参数 + 单一门槛）。
不要把新旧差异全部归因于参数随机化——门槛校准方式也变了。

---

## 4. 措辞红线

**禁止**：

- 「误杀没有任何增加」
- 「收益不是用误杀换的」
- 「EWMA 没有效果」
- 「随机参数已经验证真实市场泛化」
- 「one extreme day cannot dominate the verdict」
- 「最安静的一组」（应称「低 A、小跳跃子群」——所有路径的无条件总方差已统一）
- 「a strategy is one thing or the other」这类含糊表达

**使用**：

- "Detection improved, with similar overall false-alarm rates."
- "The additional gain from EWMA was small and uncertain in this experiment."
- "Student-t reduces sensitivity to extreme returns."
- "The benefit depends on the noise we simulate."
- "Each method uses one threshold across the whole mixed population."
  （注意：不是所有方法共用同一个数值门槛）

第 5 页讲稿必须说明：`rho` 仍固定 0.98，参数随机化**没有**移除持续波动假设；
`A`、`kappa` 是**策略之间**不同，不是每天重抽；每日跳跃次数 `K` 原本就随机；
各路径按理论常数保持相同无条件总方差。

---

## 5. 排版与打包

- 白底、深色文字、固定方法颜色；无封面、无目录、无页眉重复标题。
- 图尽量占主要空间；拥挤时**删重复文字、删图内长标题、裁留白**，
  不要靠缩小字号解决。第 5 页用表格，不必转成图。
- 正文数字保留一位小数，精确值留在 `SOURCES.md`。
- ZIP 根目录：`main.tex`、`README.txt`、`SOURCES.md`、`figures/`（正斜线相对路径）。
  不含备用页、旧 PDF、TeX 中间文件、未引用图片。
- 编译：XeLaTeX + ctexbeamer + fandol（Overleaf 自带，无需下载字体）。

---

## 6. 交付前检查清单

- [ ] 从最终 ZIP 解压到**全新目录**再编译，确认 **6 页**
- [ ] `overfull` = 0，`Missing character` = 0，`No file` = 0
- [ ] 逐页渲染检查：图可读、文字不溢出、无乱码
- [ ] 讲稿版（取消注释 `\setbeameroption`）同样 0 溢出，讲稿无截断
- [ ] 无未引用图片、无 `backup`／`appendix` 失效引用（投影内容中为 0）
- [ ] 所有数字与 CSV 一致，两个实验的来源分别标注
- [ ] 覆盖 `~/Downloads/strategy-survivorship-overleaf-eb7b73d.zip`，
      并给出新 PDF；旧版另存带 `BACKUP` 的文件名

## 7. 复现命令

```bash
python -m strategy_survivorship.run_mixed_noise      # 混合实验，约 10 秒
python -m strategy_survivorship.report_mixed_noise   # 实验报告
python -m pytest                                     # 345 项测试
xelatex main.tex                                     # 编译两遍
```

本轮不 commit、不 push GitHub；不修改任何既有 CSV。
