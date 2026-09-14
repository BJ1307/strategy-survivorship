# 六页稿最后一轮微调规格（可编辑，跨电脑交接用）

状态：**已完成并交付**（2026-09-11）。
本轮只做文字与版式微调：**不新增实验、不重跑模拟、不修改 CSV、不加备用页或附录**。

前一份规格见 `DECK_FINAL_6PAGE_SPEC.md`（六页整体结构与措辞红线），本文件只记录
这一轮改了什么。

---

## 0. 先看这条：不要回退

Downloads 里同时存在多个版本。**要交付的是**：

| 用途 | 路径 |
|---|---|
| 上传 Overleaf | `~/Downloads/final.zip` |
| 发给审阅者 | `~/Downloads/final.pdf` |

`~/Downloads/strategy-survivorship-overleaf-eb7b73d.zip` 已同步为同一份内容，
以免误传旧版。名字里带 `BACKUP` 的一律是旧版：
`BACKUP-before-final6`（七页）、`BACKUP-before-mixed`、`ORIGINAL-BACKUP`。
审阅者上一次收到的 `BACKUP_before_final6` 是**七页旧稿**，不要再用。

---

## 1. 第五页：先回答「一年、两年能检出多少」

**标题**：`How much can we detect in one or two years?`

**副标题**：
`Different noise parameters across strategies; one frozen threshold per method; a 15% two-year false-alarm budget.`

**表格**（四行，全部读自 `outputs/mixed_noise_metrics.csv`，一位小数，
固定 Student-t 行加粗）：

| Method | Detected by 1 year | Detected by 2 years | False alarms by 2 years |
|---|---|---|---|
| fixed Gaussian | 29.7% | 56.7% | 14.1% |
| **fixed Student-t** | **34.2%** | **61.6%** | **14.3%** |
| trailing 1-year Sharpe | 17.4% | 56.0% | 14.8% |
| EWMA Student-t | 34.8% | 62.6% | 14.2% |

一年列取 `detect_d252`，两年列取 `detect_d504`，误杀列取 `far_d504`。
**不得推算**。精确值（29.66 / 34.18 / 17.41 / 34.81 等）记在 `SOURCES.md`。

**主结论**：
`Fixed Student-t detects about one in three ineffective strategies within one year, and three in five within two.`

**比较句（正文只保留这一句）**：
`Student-t gains 4.9 percentage points over Gaussian; the extra 1.0-point gain from EWMA is uncertain.`

**限制句（正文只保留这一句）**：
`Overall false-alarm control does not ensure the same control for every subgroup.`

已删除原来独立的「半年／一年／两年」时间摘要，避免与表格重复；
半年 11.8% 移入讲稿。完整 bootstrap 区间不上投影，留在讲稿与
`outputs/mixed_noise_report.md`。

**讲稿必须点明**：

- 前两列分母是**无效**策略，最后一列分母是**有效**策略，两者不可相加或互换。
- 固定 t 在**低 A、小跳跃**子群误杀约 17.5%，区间 [16.0, 19.0]（供追问，不加表格）。
- **一次冻结的两年门槛用于全部观察时点**；一年那一列读的是同一个门槛，
  不是一年又重新拿到一份 15% 预算。
- 不要说「EWMA 没有效果」，不要说「误杀完全相同」，不要叫「最安静的一组」。

---

## 2. 第六页：只留两句话

**标题**：`Next: real strategy returns`

**正文**：

1. `Test the fixed Student-t monitor on real strategy returns, with Gaussian and EWMA as comparisons.`
2. `Which histories should we use, and what would count as a useful warning?`

已删除 Completed / Limit / Next 三块、重复结果，以及此前那两组较长的问题。
不新增模型或研究分支。真实数据缺少可靠标签、不能把 15% 预算直接搬到真实市场、
不能用同一段未来样本既挑门槛又验证门槛——这些留在讲稿。

---

## 3. 其余页面：只删减与核查

第 1–4 页顺序与内容不变，只删重复标题与重复解释，不缩小字体、不重新设计。

仍需保持：

- `A_i ~ Uniform(0,1)`、`kappa_i ~ Uniform(0,8)`，每条路径抽一次并保持不变；
  检测器**看不到**真实参数。
- 第 2 页那两张诊断图画的是**五个固定情境**，不是混合样本诊断图。
- 来源分开：跨情境性能来自 **Stage 2E**；一年／两年结果来自 **mixed_noise**。

---

## 4. 交付检查

- [ ] 从最终 ZIP 解压到**全新目录**编译，确认 **6 页**
- [ ] `overfull` = 0、`Missing character` = 0、`No file` = 0、未引用图 = 0
- [ ] 目视检查：新增一年列后表格仍足够大、可读
- [ ] 讲稿版（取消注释 `\setbeameroption`）同样 0 溢出，六段讲稿均完整收尾
- [ ] 投影内容中无 `backup`／`appendix`／`Completed`／`Limit` 等失效引用
- [ ] 表格四行数字与 `mixed_noise_metrics.csv` 逐格一致

本轮不 commit、不 push；未重跑任何模拟或测试，也不把未运行的检查写成已运行。

---

## 5. 追加的两处措辞（同一轮口头修改，已落地）

这两句是在上面六页版本基础上补的**纯文字**修改：不新增实验、不重跑模拟、
不改 CSV、不加备用页或附录。

**5.1 第 2 页——解释两个新参数的含义**

在 `A_i ~ Uniform(0,1)`、`kappa_i ~ Uniform(0,8)` 那条之后，加一条灰色说明：

> A controls how strongly volatility rises and falls. κ controls jump size,
> while jump frequency stays fixed. Each strategy draws its own pair once, and
> keeps it throughout the simulation.

要点：**跳跃的频率不随 κ 变**（λ = 2/年固定），变的只是跳跃幅度；参数每条路径
只抽一次、全程不变；检测器看不到这对参数。

**5.2 第 5 页——把抽象的局限句改直白**

原句（过于抽象，删除）：

> Overall false-alarm control does not ensure the same control for every subgroup.

改为：

> False alarms are below 15% overall, but higher for some types of strategies.

两半都已对着 CSV 核过，改写前后语义一致：

- “below 15% overall”：`mixed_noise_metrics.csv` 中五种方法的两年误报率为
  14.1%、14.3%、14.8%、14.2%、14.2%，全部 < 15%。
- “higher for some types”：`mixed_noise_subgroups.csv` 中 20 个分组里有 2 个
  超过 15%——`binary_student_t` 在 `A < 0.5, kappa < 4` 为 17.5%，
  `trailing_sharpe_252` 在同一格为 16.0%。

若日后数字变化，必须重新核对这两句，不要保留旧措辞。

**5.3 本轮交付核查（已实际执行）**

- 从 `final.zip` 解压到全新目录编译：**6 页**，`overfull` = 0、
  `Missing character` = 0、`No file` = 0、无 LaTeX Warning。
- 讲稿版（取消注释 `\setbeameroption`）同样 6 页、`overfull` = 0。
- 目视检查第 2 页与第 5 页渲染结果。
- 交付：`~/Downloads/final.zip`、
  `~/Downloads/strategy-survivorship-overleaf-eb7b73d.zip`（两者 md5 相同）、
  `~/Downloads/final.pdf`。
- 未 commit、未 push。

---

## 6. 恢复前三页模型逻辑（本轮，交付为 `final 2`）

不新增研究、不改 CSV、不重跑模拟、不加页数或附录。仍为六页正文。

**6.1 第 1 页**：标题 `Detect ineffective new strategies early`；目标句
`Identify ineffective strategies as early as possible, while limiting false
alarms on effective ones.`（**概括**导师要求，不是逐字引用）；场景句说明
out of sample = 策略开发完成后新观察到的收益，并声明本实验\*\*没有\*\*模拟策略挖掘、
回测挑选或选择偏差校正；底部灰底结论句 34% / 62% / 14%（取自
`mixed_noise_metrics.csv` 的 34.18 / 61.64 / 14.34）。
「两年累计误杀 15%」写成\*\*本实验的操作化定义\*\*，不写成导师逐字指定。

**6.2 第 2 页**：标题 `How do we generate the returns?`；开头
`Daily return = expected return + ordinary fluctuations + occasional jumps.`；
单行生成式（与 `noise.returns_from_noise` ＋ `mixed_noise.draw_noise` 一致）：
`r = s·σa/D + (σa/√D)·(√v·z + κ√K·w)/√(1+κ²λ/D)`；三条理由（真实优势是我们设的、
波动会持续也会跳、**统一无条件总方差**）；A 与 κ 的作用两句照写；
`A_i ~ U(0,1)`、`κ_i ~ U(0,8)`，每条策略抽一次、监测器看不到。
图注只写 `Diagnostics for five fixed parameter settings.`

**6.3 第 3 页**：标题 `From daily returns to an alarm`，左右两栏（左略宽）。
左「1. Build evidence」：先验 `P(s=1)=P(s=0)=1/2`；
`U_n = U_{n-1} + log[f_ineffective(r_n)/f_effective(r_n)]`，`U_0=0`；
似然选择与 EWMA，并紧跟一句「模拟器本来就含持续波动，这不是普遍市场优势的证据」。
右「2. Calibrate the alarm」：`P(max_{1≤n≤504} U_n > b | effective) ≤ 0.15`
（代码里报警是\*\*严格\*\*下穿，所以用 `>` 不用 `≥`；仓库存的是 `L`，`U = −L`）；
门槛在独立有效样本上带缓冲选取、冻结后用独立测试样本评价。

**6.4 全文**：删除 `slide 4`／`Slide 5`／`上一页`／`下一页` 等页码指引，改用
「固定情境实验」「混合参数实验」。第 4 页结论改为
`In the tested settings the gains depend on the noise assumptions: fixed
Gaussian is the best of the three in the Gaussian control, at 56.6% against
53.4% and 52.6%.`

**6.5 图片**：`n03_dgp_ab.png` 重绘，**只改版式**——画布变小但字号不变（因此在幻灯片
上缩放后字更大）、去掉两个长面板标题、图例把 `k` 改成 `κ`。同一 `SeedSequence(20260906)`、
同样 2000 条路径；新旧脚本各导出十条曲线逐点比对，**完全相同**。
`n01_paths.png` 未改动，其面板仍写旧措辞 `valid` / `invalid`，正文用
effective / ineffective 并标出对应关系。

**6.6 修掉两个旧缺陷（此前交付的 ZIP 里就有，只影响讲稿版）**

1. 讲稿版下 pgfpages 会重新装箱，嵌套盒子（`tabular`、`columns`、`minipage`）里的
   文字丢掉默认颜色、被画成白色。**第 5 页的结果表在以前每一版讲稿里都是看不见的。**
   现在把第 3 页两栏和第 5 页表格包在 `\color{black}{...}` 里。**不要删掉这层包裹。**
2. 第 5 页讲稿超出讲稿面板、尾部被静默截断。已压缩该页讲稿，事实一条没丢。
   现在六页讲稿都用程序核对：在渲染出的讲稿面板里查找源文的末尾字符。

**6.7 本轮实际执行的核查**

- 从 `final 2.zip` 解压到全新目录编译：**6 页**，`overfull` = 0、`underfull` = 0、
  `Missing character` = 0、`No file` = 0、`LaTeX Warning` = 0。
- 同一份解压再编译讲稿版：6 页、0 溢出，六段讲稿**全部完整**。
- 幻灯片六页逐页目视检查。
- 第 1／4／5 页每个数字与 `mixed_noise_metrics.csv`、`stage2e_metrics.csv`、
  `mixed_noise_bootstrap.csv` 逐格核对一致。
- 交付：`~/Downloads/final 2.zip`、`~/Downloads/final 2.pdf`、
  `~/Downloads/strategy-survivorship-overleaf-eb7b73d.zip`（与前者 md5 相同）。
- 未 commit、未 push。
