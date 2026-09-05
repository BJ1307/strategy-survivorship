# theory.md — 第一阶段的符号、单位、公式与推导

本文件只覆盖第一阶段实际实现的内容。文末列出参考资料；**下面的阈值协议、
观察期限和默认参数都是本项目的实现选择，不是这些参考文献直接规定的结论。**

---

## 1. 符号与单位

| 符号 | 含义 | 单位 |
|---|---|---|
| $D$ | 每年交易日数（本阶段 252） | 天/年 |
| $H$ | 监控期限（本阶段 504） | 交易日 |
| $t$ | 交易日序号，$t=1,\dots,H$ | 交易日 |
| $\sigma_{\mathrm{ann}}$ | 年化波动率 | 年$^{-1/2}$ |
| $\sigma_d=\sigma_{\mathrm{ann}}/\sqrt D$ | 日波动率 | 日$^{-1/2}$ |
| $S$ | **年化** Sharpe，生成过程的参数 | 年$^{-1/2}$ |
| $r_t$ | 第 $t$ 日的策略超额收益（已扣成本） | 日收益 |
| $z_t=r_t/\sigma_d$ | 标准化收益 | 无量纲 |
| $L_t$ | 后验 log odds | nat |
| $p_t=\operatorname{expit}(L_t)$ | 有效概率 | 无量纲 |
| $\tau$ | 首次报警时间（first passage） | 交易日 |
| $\alpha$ | 整个 $H$ 期内的累计误杀率预算 | 无量纲 |
| $\nu$ | Student-t 自由度（本阶段 5） | 无量纲 |
| $a_\nu$ | Student-t 尺度参数 | 无量纲 |

**单位约定。** 年化量与日频量之间只用两条换算：波动率按 $\sqrt{\text{时间}}$ 缩放，
均值按时间线性缩放。因此年化 Sharpe $S$ 对应的日漂移是

$$\mu_d=\frac{S\,\sigma_{\mathrm{ann}}}{D},\qquad
\frac{\mu_d}{\sigma_d}=\frac{S\sigma_{\mathrm{ann}}/D}{\sigma_{\mathrm{ann}}/\sqrt D}
=\frac{S}{\sqrt D},$$

即**日 Sharpe 等于 $S/\sqrt D$**，再乘 $\sqrt D$ 年化回 $S$。

---

## 2. 数据生成过程

$$r_t=\frac{S\sigma_{\mathrm{ann}}}{D}
+\frac{\sigma_{\mathrm{ann}}}{\sqrt D}\,\varepsilon_t,
\qquad \varepsilon_t\overset{\mathrm{iid}}{\sim}N(0,1).$$

有效状态 $S=1$，无效状态 $S=0$。标准化后

$$z_t\mid S\;\sim\;N\!\left(\frac{S}{\sqrt D},\,1\right).$$

$S$ 是**生成过程的参数**，不是任何一条路径的样本 Sharpe。路径生成之后没有做任何
事后平移或缩放，所以每条路径的年化样本 Sharpe 仍然围绕 $S$ 波动，标准差约为
$\sqrt{D/H}=\sqrt{252/504}\approx0.707$——这正是本问题困难的来源。

---

## 3. 检测器 A：Binary Gaussian

两个简单假设 $S\in\{0,1\}$，日波动率已知。一步对数似然比：

$$
\begin{aligned}
\Delta_t
&=\log\varphi\!\left(z_t-\tfrac{1}{\sqrt D}\right)-\log\varphi(z_t)\\
&=-\tfrac12\left(z_t-\tfrac{1}{\sqrt D}\right)^2+\tfrac12 z_t^2\\
&=-\tfrac12\left(z_t^2-\tfrac{2z_t}{\sqrt D}+\tfrac1D\right)+\tfrac12 z_t^2\\
&=\boxed{\;\frac{z_t}{\sqrt D}-\frac{1}{2D}\;}
\end{aligned}
$$

二次项与归一化常数完全抵消，所以增量对 $z_t$ 是**仿射**的。由贝叶斯定理，
log odds 可加：

$$L_t=L_{t-1}+\Delta_t,\qquad L_0=\log\frac{P(\text{valid})}{1-P(\text{valid})}=0
\;\;(\text{先验 }0.5).$$

### 解析矩（用于校验实现）

在 $S=1$ 下 $E[z_t]=1/\sqrt D$，$\operatorname{Var}(z_t)=1$：

$$E[\Delta_t\mid S{=}1]=\frac{1}{\sqrt D}\cdot\frac{1}{\sqrt D}-\frac{1}{2D}
=\frac{1}{D}-\frac{1}{2D}=\frac{1}{2D}.$$

在 $S=0$ 下 $E[z_t]=0$，于是 $E[\Delta_t\mid S{=}0]=-\frac{1}{2D}$。两种状态下

$$\operatorname{Var}(\Delta_t\mid S)=\frac{1}{D}\operatorname{Var}(z_t)=\frac1D.$$

$\Delta_t$ 独立，故对 $L_0=0$：

$$\boxed{\;E[L_n\mid S{=}1]=\frac{n}{2D},\quad
E[L_n\mid S{=}0]=-\frac{n}{2D},\quad
\operatorname{Var}(L_n\mid S)=\frac{n}{D}.\;}$$

代入 $n=H=504$、$D=252$：漂移仅 $\pm1$ nat，而标准差 $\sqrt2\approx1.41$ nat。
**两年的证据量比噪声还小**，这解释了后面为什么检出率无法很高。

数值上报警判定在 log odds 空间完成；$p_t=\operatorname{expit}(L_t)$ 只用于展示，
避免概率接近 0 或 1 时的精度损失。

---

## 4. 检测器 B：Binary Student-t

状态、先验和已知日波动率都与 A 相同，只替换观测似然。在 $z$ 空间中两个状态的位置
参数分别是 $0$ 与 $1/\sqrt D$，自由度 $\nu=5$。

**尺度不是标准差。** 若 $X\sim t_\nu(\text{loc},\text{scale}=a)$，则

$$\operatorname{Var}(X)=a^2\frac{\nu}{\nu-2}.$$

要让两种似然的观测噪声方差都等于 1，必须取

$$a_\nu=\sqrt{\frac{\nu-2}{\nu}},\qquad \nu=5\Rightarrow a_5=\sqrt{3/5}\approx0.774597.$$

（若误用 $a=1$，Student-t 检测器面对的噪声方差会是 $\nu/(\nu-2)=5/3$，
等于换了一个不同的、更容易的问题。）

递推：

$$L_t=L_{t-1}
+\log t_\nu\!\left(z_t;\ \text{loc}=\tfrac{1}{\sqrt D},\ \text{scale}=a_\nu\right)
-\log t_\nu\!\left(z_t;\ \text{loc}=0,\ \text{scale}=a_\nu\right).$$

实现使用 `scipy.stats.t.logpdf`，不对密度取对数。

### 影响函数：有界且 redescending

$t_\nu$ 的对数密度为 $-\frac{\nu+1}{2}\log\!\left(1+\frac{(x-\text{loc})^2}{\nu a^2}\right)+\text{const}$，
于是

$$\Delta^{t}(z)=-\frac{\nu+1}{2}
\left[\log\!\left(1+\frac{(z-1/\sqrt D)^2}{\nu a_\nu^2}\right)
-\log\!\left(1+\frac{z^2}{\nu a_\nu^2}\right)\right].$$

当 $|z|\to\infty$ 时括号内两项之比趋于 1，对数之差趋于 0，故

$$\Delta^{t}(z)\longrightarrow 0 .$$

也就是说 $|\Delta^t|$ 在中等 $|z|$（本参数下 $z\approx-1.7$，峰值约 0.109）取极大，
之后**衰减回零**。直观解释：两个假设的位置只差 $1/\sqrt D\approx0.063$，一个足够极端
的观测在两个状态下都同样不可能，因此几乎不携带区分信息。

对比之下 Gaussian 的 $\Delta(z)=z/\sqrt D-1/(2D)$ 无界。这就是为什么单次巨大冲击
把 Gaussian 的 log odds 永久平移 $\pm k/\sqrt D$，却几乎不动 Student-t——甚至可能
把它推向相反方向（见报告第 6 节）。

> 这是对**单次观测**的降权，不是对持续性随机波动率的解决方案。

**本阶段的数据是高斯的，因此 Student-t 检测器是似然失配模型。** 这是有意安排，
用来量化“为稳健性买保险”的代价，不通过调整数据或参数来让它胜出。

---

## 5. 检测器 C：Trailing 12-month Sharpe

$t\ge W$（$W=252$）时

$$\widehat S_t=\sqrt D\;\frac{\bar r_{t-W+1:t}}{s_{t-W+1:t}},\qquad
s^2=\frac{1}{W-1}\sum_{i=t-W+1}^{t}(r_i-\bar r)^2\quad(\texttt{ddof}=1).$$

$t<W$ 时不定义（记 `NaN`），既不参与阈值校准也不能报警。不使用未来数据填窗口，
不退化为 expanding window。

关于样本 Sharpe 的统计误差：即使在 iid 正态、无自相关的理想条件下，年化样本 Sharpe
的标准误约为 $\sqrt{(1+S^2/2)/n_{\text{years}}}$（Lo 2002）。$n_\text{years}=1$、
$S\approx1$ 时约 0.7，与信号本身同量级。

## 6. 检测器 D：Known-vol rolling（诊断对照）

窗口与 C 完全相同，只把分母换成本实验中已知的日波动率：

$$S_t^{\mathrm{known\ vol}}=\sqrt D\;\frac{\bar r_{t-W+1:t}}{\sigma_d}.$$

C 与 D 之差**只**来自“估计波动率 vs 已知波动率”，其余设定完全一致。

---

## 7. 决策协议、阈值与指标

**报警规则。** 统计量在可报警日上**严格小于**阈值即报警；首次报警关闭决策，
$\tau$ 是首次穿越时间，不可重开。`NaN` 永不触发报警。

**累计误杀率。**

$$F_H=\Pr_{S=1}(\tau\le H).$$

**阈值选择。** 一条路径在监控期内报警，当且仅当其可报警日上的最小统计量低于阈值：

$$\{\tau\le H\}=\Big\{\min_{t\in[t_0,H]}X_t<c\Big\},\qquad t_0=\text{最早可报警日}.$$

设校准集（**只用有效路径**）的 $N$ 个路径最小值排序后为 $m_{(1)}\le\dots\le m_{(N)}$，
取 $k=\lfloor\alpha N\rfloor$，阈值 $c=m_{(k+1)}$（0 基索引 $k$）。则

$$\widehat F_H=\frac{\#\{m_i<c\}}{N}\le\frac{k}{N}\le\alpha,$$

并列值只会让左边更小。这是**构造性保守**的：无需分布假设。缺失值不参与最小值。

**截断等待时间。** $\min(\tau,H)$；未报警路径按 $H$ 计入，但 `alarmed` 标志与
`first_alarm_day=-1` 哨兵使它与“恰好第 $H$ 天报警”可区分。

**总体中位检测时间。** 满足 $\Pr(\tau\le t)\ge0.5$ 的最小 $t$；若两年内检出率不足
一半则报告“在观察期内未达到”，**不**退化为“已检出样本的中位数”。

**Wilson 区间。** 对比例 $\hat p=k/n$：

$$\frac{\hat p+\frac{z^2}{2n}\pm\frac{z}{1}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z=1.95996.$$

它只覆盖**单个时点、单个比例**的 Monte-Carlo 噪声；不是整条时间曲线的同时置信带，
也不包含阈值校准本身的不确定性。

**随机关闭解析参考。** 每日独立以概率 $q$ 关闭，令

$$q=1-(1-\alpha)^{1/H}
\;\Longrightarrow\;
\Pr(\tau\le t)=1-(1-q)^t=1-(1-\alpha)^{t/H},$$

在 $t=H$ 时恰好等于 $\alpha$，且对有效与无效策略**完全相同**。其截断期望为

$$E[\min(\tau,H)]=\sum_{t=0}^{H-1}\Pr(\tau>t)=\sum_{t=0}^{H-1}(1-q)^t
=\frac{1-(1-q)^H}{q}=\frac{\alpha}{q}.$$

这些是精确解析值，因此**不**为其编造 Monte-Carlo 区间。由于 $\alpha<0.5$，该参考线
的中位检测时间在 $H$ 内永远达不到。

---

## 7.5 校准阈值本身的抽样不确定性

阈值 `c` 是从一份有限的校准样本估出来的统计量，不是常数。把整个流程
（模拟 → 校准 → 冻结 → 在独立测试集上测量）视为一次实验，实际误杀率
`F̂_H` 的方差可以分解为两项：

```
Var(F̂_H) = Var_calib( E[F̂_H | c] )  +  E_calib[ Var(F̂_H | c) ]
                 阈值抽样贡献              测试集二项抽样贡献
```

第二项就是通常报告的 `α(1-α)/n_test`，也是 Wilson 区间覆盖的那一项；
第一项完全不在 Wilson 区间里。`robustness.py` 在 R 个独立根种子下重复整个
流程，用复算间的样本方差估计左边，用 `α(1-α)/n_test` 作为右边第二项，
两者相减（截断在 0）得到阈值分量：

```
sd_calib ≈ sqrt( max(0, s²_R − α(1-α)/n_test) )
```

并用样本方差的卡方抽样律做单边检验 `H0: Var(F̂_H) = α(1-α)/n_test`：

```
(R-1) · s²_R / σ0²  ~  χ²_{R-1}      （近似：复算间的比率视为近正态）
```

**R 需要足够大。** 一个由 R 次复算估出的标准差，其自身相对标准误约
`1/sqrt(2(R-1))`：R = 20 时约 16%，不足以把 1.0 和 1.3 区分开，实测中甚至
会出现放大倍数小于 1 的估计（理论上不可能，因为总方差不低于二项项）。
R = 100 时降到约 7%，足以分辨。本阶段默认取 R = 100。

---

## 8. 参考资料

- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52.
  <https://alo.mit.edu/wp-content/uploads/2017/06/The-Statistics-of-Sharpe-Ratios.pdf>
  ——样本 Sharpe 的统计误差与时间尺度换算。
- SciPy, `scipy.stats.t`：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html>
  ——`logpdf`、`loc`/`scale` 的含义（scale 不是标准差）。
- Stan Functions Reference / User's Guide，Student-t 的尺度混合表示：
  <https://mc-stan.org/docs/stan-users-guide/robust-noise-models.html>
  ——重尾似然等价于对每个观测赋予随机方差，是理解“降权”的另一视角。
- NumPy random `SeedSequence` / `Generator`：
  <https://numpy.org/doc/stable/reference/random/parallel.html>
  ——校准、测试与诊断数据的可复现、互相独立的随机流。

**范围声明。** 上述文献支撑的是公式与 API 用法。本阶段实现的两个二值基线不是对任何
尚未确认题名的 “EKV” 或 “Stefan” 论文的复现，也不应被这样引用。
