# strategy-survivorship

一个新策略上线后要么有效、要么无效。**在可接受的累计误杀约束下，多快能识别出无效的那一个？**
本仓库用模拟数据回答这个问题，并把结果解释成交易日、年数、检出比例与失效概率。

全部为模拟研究，**不是真实市场部署的验证**。

- **先读**：[`docs/BRIEF.md`](docs/BRIEF.md) —— 两页导师简报，四张主图。
- **接手项目**：[`docs/HANDOFF_COMPANY_CLAUDE.md`](docs/HANDOFF_COMPANY_CLAUDE.md) —— 完整方法、协议、指标定义与数据出处。
- **推导**：[`theory.md`](theory.md)。
- **各阶段报告**：`outputs/stage*_report.md`。

---

## 核心结果（Sharpe 1 对 0 为主线）

主情境「随机波动率 + 跳跃」，两年累计误杀预算 15%，最好的方法是截断 EWMA Student-t：

| 观察时长 | 一个季度 | 半年 | 一年 | 两年 |
|---|---|---|---|---|
| **Sharpe 1（主线）** | 4.1% | 21.1% | 48.7% | **76.2%** |
| Sharpe 0.6（弱信号压力测试） | 0.6% | 8.4% | 25.8% | **52.3%** |

中位检出时间：s=1 约 260 天，s=0.6 约 475 天。

三步稳健处理的贡献量级依次递减：重尾似然 +13.4 pp、方差自适应 +6.4 pp、截断更新 +0.7 pp；
**在纯高斯情境里这套处理反而是负收益（−3.2 pp）**。

![四张主图之一](docs/figures/fig2_how_long_to_observe.png)

---

## 环境与运行

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q
```

### 不需要重跑实验的操作

```bash
make brief          # 重画四张主图（只读 CSV）
make figures        # 重画全部阶段图（只读 CSV）
make report-only    # 重生成阶段报告（只读 CSV）
```

### 完整复现（约 10 分钟，按顺序）

```bash
for s in stage1 stage11 stage2a stage2b stage2c stage2d stage2e stage2e1 \
         stage3a stage3a1 stage3b; do
  .venv/bin/python -m strategy_survivorship.run_$s
done
.venv/bin/python -m strategy_survivorship.figures_brief
```

阶段之间有依赖：Stage 2D 读 Stage 2C 的统一门槛，Stage 3B 读 Stage 3A 的冻结门槛。

---

## 研究路线

| 阶段 | 问题 | 报告 |
|---|---|---|
| 1 | 最小可复现流程：模拟→检测→校准→评估 | [`stage1_report.md`](outputs/stage1_report.md) |
| 1.1 | 失效概率与时间的对应 | [`stage11_report.md`](outputs/stage11_report.md) |
| 2A | 厚尾、随机波动率、跳跃下的压力测试 | [`stage2a_report.md`](outputs/stage2a_report.md) |
| 2B | 因果 EWMA 方差预测是否有用 | [`stage2b_report.md`](outputs/stage2b_report.md) |
| 2C | 环境未知时的统一门槛与误杀控制 | [`stage2c_report.md`](outputs/stage2c_report.md) |
| 2D | 随机波动率与跳跃同时存在 | [`stage2d_report.md`](outputs/stage2d_report.md) |
| 2E | 截断方差更新的 2×2 消融 | [`stage2e_report.md`](outputs/stage2e_report.md) |
| 2E.1 | 收尾、跨预算统计、高斯理论参照 | [`stage2e1_report.md`](outputs/stage2e1_report.md) |
| 3A | 弱信号 Sharpe 0.6，保留 1 作参照 | [`stage3a_report.md`](outputs/stage3a_report.md) |
| 3A.1 | 监测期限本身是不是原因 | [`stage3a1_report.md`](outputs/stage3a1_report.md) |
| 3B | 先有效、后失效（第一年内） | [`stage3b_report.md`](outputs/stage3b_report.md) |

| 真实数据 1 | 真实指数噪声的可复现参照与训练期诊断（**不含**模拟对比） | 本地生成：`outputs/market/market_stage1_report.md` |
| 真实数据 2 | 冻结生成器与 S&P 500 训练样本的诊断对照（**不搜参数、不碰监测器**） | 本地生成：`outputs/market/market_stage2_report.md` |
| 真实数据 3 | 只校准 A、ρ，冻结后在 2022–2023 验证（**改善未能延续**） | 本地生成：`outputs/market/market_stage3_report.md` |
| 真实数据 4 | 2017–2023 回顾性 walk-forward 回放（**定期重估平均未胜过固定参数**） | 本地生成：`outputs/market/market_stage4_report.md` |
| 跨市场 | S&P／Nasdaq-100／Nikkei 225／动量因子 × 三个**固定**模型的基线对照 | 本地生成：`outputs/market/market_cross_report.md` |
| 跨资产 | 八个收益类对象 + 七个市场状态序列，按三类口径分开 | [`docs/CROSS_ASSET_FINDINGS.md`](docs/CROSS_ASSET_FINDINGS.md) |
| 恢复实验 | 真实生成机制已知时，现有校准程序能否找回参数（**参数恢复差、筛选不稳**） | 本地生成：`outputs/market/market_recovery_report.md` |
| 敏感性 | 固定伪历史、只换内部模拟种子，选择是否移动（**20 次拟合全部不动**） | 本地生成：`outputs/market/market_sensitivity_report.md` |

真实数据阶段的口径写在 [`docs/MARKET_STAGE1_PROTOCOL.md`](docs/MARKET_STAGE1_PROTOCOL.md)，
字段说明在 [`docs/MARKET_DATA_DICTIONARY.md`](docs/MARKET_DATA_DICTIONARY.md)。
**市场数据与由它派生的每日序列、图和报告都不进 Git**（FRED 载明 S&P 数据不得再分发）。
`data/` 整个目录被忽略，**provenance 记录也在其中**，因此同样不在 Git 里，按需另行提供。
克隆后用下面的命令在本地重建：

```bash
.venv/bin/python -m strategy_survivorship.run_market_stage1
.venv/bin/python -m strategy_survivorship.report_market_stage1
.venv/bin/python -m strategy_survivorship.run_market_stage2 --paths 2000
.venv/bin/python -m strategy_survivorship.report_market_stage2
```

```bash
.venv/bin/python -m strategy_survivorship.run_market_stage3
.venv/bin/python -m strategy_survivorship.report_market_stage3
.venv/bin/python -m strategy_survivorship.run_market_stage4      # 约 2 分钟
.venv/bin/python -m strategy_survivorship.report_market_stage4
```

```bash
.venv/bin/python -m strategy_survivorship.run_market_cross --paths 2000
.venv/bin/python -m strategy_survivorship.report_market_cross
```

```bash
.venv/bin/python -m strategy_survivorship.run_market_assets --paths 1500
.venv/bin/python -m strategy_survivorship.plots_market_assets
```

```bash
.venv/bin/python -m strategy_survivorship.run_market_recovery      # 约 9 分钟
.venv/bin/python -m strategy_survivorship.report_market_recovery
.venv/bin/python -m strategy_survivorship.plots_market_recovery
.venv/bin/python -m strategy_survivorship.run_market_sensitivity   # 约 4 分钟
.venv/bin/python -m strategy_survivorship.report_market_sensitivity
.venv/bin/python -m strategy_survivorship.plots_market_sensitivity
```

收益取样的窗口边界口径（**按结束日期归属、首日可回取前一个收盘价、缺失交易日不静默跨过**）
写在 [`docs/RETURN_BOUNDARY_CONVENTION.md`](docs/RETURN_BOUNDARY_CONVENTION.md)，
由 `market_returns.py` 统一实现；两个旧入口在共同口径下逐位一致，可用下面一条命令复核：

```bash
.venv/bin/python -m strategy_survivorship.run_market_returns_check
```

下一轮的模型基准比较（iid Student-t、GARCH(1,1)-t 对照现有 SV 核）**只有设计、尚未实现**，
写在 [`docs/BENCHMARK_PROTOCOL_NEXT_ROUND.md`](docs/BENCHMARK_PROTOCOL_NEXT_ROUND.md)。

参数与配置变更记入 [`docs/PARAMETER_CHANGE_LOG.md`](docs/PARAMETER_CHANGE_LOG.md)，
给导师的数据请求在 [`docs/DATA_REQUEST_FOR_SUPERVISOR.md`](docs/DATA_REQUEST_FOR_SUPERVISOR.md)。
导师数据来源清单在 [`docs/DATA_SOURCE_INVENTORY.md`](docs/DATA_SOURCE_INVENTORY.md)，
下一轮跨市场扫描规范在 [`docs/CROSS_MARKET_SCAN_SPEC.md`](docs/CROSS_MARKET_SCAN_SPEC.md)。

**负面结果保留在报告里**：高斯情境下稳健处理的代价、门槛迁移失败、弱信号低检出、
以及基线方法在某些设定下胜出，都没有被筛掉。

---

## 目录

```
src/strategy_survivorship/   模型、DGP、校准、评估、绘图与报告生成
tests/                       307 项测试：定义、因果性、协议、评分口径
outputs/                     全部结果 CSV/JSON、各阶段报告与图
docs/                        导师简报、交接文档、四张主图
theory.md                    完整推导
```

`outputs/` 里的结果由 tag `pre-consolidation` 生成；审阅包是构建产物，
用 `make bundle` 重建或从该 tag 取回。

---

## 限制

模拟研究；候选 Sharpe 只有 {1, 0.6} 且检测器知道候选值；五个噪声情境同出一族；
Stage 3B 只覆盖第一年内失效；校准只覆盖所列固定情境；全部区间为逐项区间；
评价只含检出率与误杀率，**等待成本未纳入**。
