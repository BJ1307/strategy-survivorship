# theory.md — Notation, units, formulas and derivations for the first stage

This file covers only what was actually implemented in the first stage. References are listed at the
end; **the threshold protocol, observation horizons and default parameters below are implementation
choices of this project, not conclusions prescribed directly by those references.**

---

## 1. Notation and units

| Symbol | Meaning | Unit |
|---|---|---|
| $D$ | Trading days per year (252 in this stage) | days/year |
| $H$ | Monitoring horizon (504 in this stage) | trading days |
| $t$ | Trading-day index, $t=1,\dots,H$ | trading days |
| $\sigma_{\mathrm{ann}}$ | Annualised volatility | year$^{-1/2}$ |
| $\sigma_d=\sigma_{\mathrm{ann}}/\sqrt D$ | Daily volatility | day$^{-1/2}$ |
| $S$ | **Annualised** Sharpe, a parameter of the generating process | year$^{-1/2}$ |
| $r_t$ | Strategy excess return on day $t$ (net of costs) | daily return |
| $z_t=r_t/\sigma_d$ | Standardised return | dimensionless |
| $L_t$ | Posterior log odds | nat |
| $p_t=\operatorname{expit}(L_t)$ | Probability of being valid | dimensionless |
| $\tau$ | First alarm time (first passage) | trading days |
| $\alpha$ | Cumulative false-alarm budget over the whole horizon $H$ | dimensionless |
| $\nu$ | Student-t degrees of freedom (5 in this stage) | dimensionless |
| $a_\nu$ | Student-t scale parameter | dimensionless |

**Unit convention.** Only two conversions are used between annualised and daily quantities:
volatility scales with $\sqrt{\text{time}}$, and the mean scales linearly with time. The daily drift
corresponding to an annualised Sharpe $S$ is therefore

$$\mu_d=\frac{S\,\sigma_{\mathrm{ann}}}{D},\qquad
\frac{\mu_d}{\sigma_d}=\frac{S\sigma_{\mathrm{ann}}/D}{\sigma_{\mathrm{ann}}/\sqrt D}
=\frac{S}{\sqrt D},$$

i.e. **the daily Sharpe equals $S/\sqrt D$**, and multiplying by $\sqrt D$ annualises it back to $S$.

---

## 2. Data-generating process

$$r_t=\frac{S\sigma_{\mathrm{ann}}}{D}
+\frac{\sigma_{\mathrm{ann}}}{\sqrt D}\,\varepsilon_t,
\qquad \varepsilon_t\overset{\mathrm{iid}}{\sim}N(0,1).$$

The valid state is $S=1$ and the invalid state is $S=0$. After standardisation

$$z_t\mid S\;\sim\;N\!\left(\frac{S}{\sqrt D},\,1\right).$$

$S$ is a **parameter of the generating process**, not the sample Sharpe of any single path. No
ex-post shifting or rescaling is applied after the paths are generated, so the annualised sample
Sharpe of each path still fluctuates around $S$, with a standard deviation of about
$\sqrt{D/H}=\sqrt{252/504}\approx0.707$ — this is precisely where the difficulty of the problem
comes from.

---

## 3. Detector A: Binary Gaussian

Two simple hypotheses $S\in\{0,1\}$, with known daily volatility. The one-step log-likelihood ratio is

$$
\begin{aligned}
\Delta_t
&=\log\varphi\!\left(z_t-\tfrac{1}{\sqrt D}\right)-\log\varphi(z_t)\\
&=-\tfrac12\left(z_t-\tfrac{1}{\sqrt D}\right)^2+\tfrac12 z_t^2\\
&=-\tfrac12\left(z_t^2-\tfrac{2z_t}{\sqrt D}+\tfrac1D\right)+\tfrac12 z_t^2\\
&=\boxed{\;\frac{z_t}{\sqrt D}-\frac{1}{2D}\;}
\end{aligned}
$$

The quadratic terms and the normalising constants cancel exactly, so the increment is **affine** in
$z_t$. By Bayes' theorem the log odds are additive:

$$L_t=L_{t-1}+\Delta_t,\qquad L_0=\log\frac{P(\text{valid})}{1-P(\text{valid})}=0
\;\;(\text{prior }0.5).$$

### Analytic moments (used to verify the implementation)

Under $S=1$, $E[z_t]=1/\sqrt D$ and $\operatorname{Var}(z_t)=1$:

$$E[\Delta_t\mid S{=}1]=\frac{1}{\sqrt D}\cdot\frac{1}{\sqrt D}-\frac{1}{2D}
=\frac{1}{D}-\frac{1}{2D}=\frac{1}{2D}.$$

Under $S=0$, $E[z_t]=0$, so $E[\Delta_t\mid S{=}0]=-\frac{1}{2D}$. In both states

$$\operatorname{Var}(\Delta_t\mid S)=\frac{1}{D}\operatorname{Var}(z_t)=\frac1D.$$

The $\Delta_t$ are independent, so with $L_0=0$:

$$\boxed{\;E[L_n\mid S{=}1]=\frac{n}{2D},\quad
E[L_n\mid S{=}0]=-\frac{n}{2D},\quad
\operatorname{Var}(L_n\mid S)=\frac{n}{D}.\;}$$

Substituting $n=H=504$ and $D=252$: the drift is only $\pm1$ nat, while the standard deviation is
$\sqrt2\approx1.41$ nat. **Two years of evidence amount to less than the noise**, which explains why
the detection rates later on cannot be high.

Numerically, the alarm decision is made in log-odds space; $p_t=\operatorname{expit}(L_t)$ is used
only for display, to avoid loss of precision when the probability is close to 0 or 1.

---

## 4. Detector B: Binary Student-t

The states, the prior and the known daily volatility are the same as in A; only the observation
likelihood is replaced. In $z$-space the location parameters of the two states are $0$ and
$1/\sqrt D$, with degrees of freedom $\nu=5$.

**Scale is not the standard deviation.** If $X\sim t_\nu(\text{loc},\text{scale}=a)$, then

$$\operatorname{Var}(X)=a^2\frac{\nu}{\nu-2}.$$

For the observation-noise variance to equal 1 under both likelihoods, one must take

$$a_\nu=\sqrt{\frac{\nu-2}{\nu}},\qquad \nu=5\Rightarrow a_5=\sqrt{3/5}\approx0.774597.$$

(If $a=1$ were used by mistake, the noise variance faced by the Student-t detector would be
$\nu/(\nu-2)=5/3$, which amounts to switching to a different, easier problem.)

Recursion:

$$L_t=L_{t-1}
+\log t_\nu\!\left(z_t;\ \text{loc}=\tfrac{1}{\sqrt D},\ \text{scale}=a_\nu\right)
-\log t_\nu\!\left(z_t;\ \text{loc}=0,\ \text{scale}=a_\nu\right).$$

The implementation uses `scipy.stats.t.logpdf` rather than taking the logarithm of the density.

### Influence function: bounded and redescending

The log density of $t_\nu$ is $-\frac{\nu+1}{2}\log\!\left(1+\frac{(x-\text{loc})^2}{\nu a^2}\right)+\text{const}$,
so

$$\Delta^{t}(z)=-\frac{\nu+1}{2}
\left[\log\!\left(1+\frac{(z-1/\sqrt D)^2}{\nu a_\nu^2}\right)
-\log\!\left(1+\frac{z^2}{\nu a_\nu^2}\right)\right].$$

As $|z|\to\infty$ the ratio of the two terms inside the brackets tends to 1 and the difference of
the logarithms tends to 0, hence

$$\Delta^{t}(z)\longrightarrow 0 .$$

In other words, $|\Delta^t|$ reaches its maximum at moderate $|z|$ (for these parameters at
$z\approx-1.7$, with a peak of about 0.109) and then **decays back to zero**. Intuition: the two
hypotheses differ in location by only $1/\sqrt D\approx0.063$; a sufficiently extreme observation is
equally improbable under both states and therefore carries almost no discriminating information.

By contrast, the Gaussian $\Delta(z)=z/\sqrt D-1/(2D)$ is unbounded. This is why a single huge shock
permanently shifts the Gaussian log odds by $\pm k/\sqrt D$ but barely moves the Student-t — and may
even push it in the opposite direction (see Section 6 of the report, generated locally).

> This is a down-weighting of **single observations**, not a solution to persistent stochastic
> volatility.

**The data in this stage are Gaussian, so the Student-t detector is a misspecified-likelihood
model.** This is deliberate: it quantifies the cost of "buying insurance for robustness", and
neither the data nor the parameters are adjusted to make it win.

---

## 5. Detector C: Trailing 12-month Sharpe

For $t\ge W$ ($W=252$)

$$\widehat S_t=\sqrt D\;\frac{\bar r_{t-W+1:t}}{s_{t-W+1:t}},\qquad
s^2=\frac{1}{W-1}\sum_{i=t-W+1}^{t}(r_i-\bar r)^2\quad(\texttt{ddof}=1).$$

For $t<W$ it is undefined (recorded as `NaN`) and takes part neither in threshold calibration nor in
alarms. Future data are not used to fill the window, and it does not fall back to an expanding
window.

On the statistical error of the sample Sharpe: even under ideal iid normal conditions with no
autocorrelation, the standard error of the annualised sample Sharpe is about
$\sqrt{(1+S^2/2)/n_{\text{years}}}$ (Lo 2002). With $n_\text{years}=1$ and $S\approx1$ it is about
0.7, the same order of magnitude as the signal itself.

## 6. Detector D: Known-vol rolling (diagnostic control)

The window is identical to C; only the denominator is replaced by the daily volatility that is known
in this experiment:

$$S_t^{\mathrm{known\ vol}}=\sqrt D\;\frac{\bar r_{t-W+1:t}}{\sigma_d}.$$

The difference between C and D comes **only** from "estimated vs known volatility"; every other
setting is identical.

---

## 7. Decision protocol, thresholds and metrics

**Alarm rule.** An alarm is raised when the statistic is **strictly less than** the threshold on an
alarm-eligible day; the first alarm closes the decision, $\tau$ is the first-passage time, and the
decision cannot be reopened. `NaN` never triggers an alarm.

**Cumulative false-alarm rate.**

$$F_H=\Pr_{S=1}(\tau\le H).$$

**Threshold selection.** A path raises an alarm within the monitoring horizon if and only if the
minimum of its statistic over alarm-eligible days is below the threshold:

$$\{\tau\le H\}=\Big\{\min_{t\in[t_0,H]}X_t<c\Big\},\qquad t_0=\text{earliest alarm-eligible day}.$$

Let the $N$ path minima of the calibration set (**valid paths only**), sorted, be
$m_{(1)}\le\dots\le m_{(N)}$; take $k=\lfloor\alpha N\rfloor$ and threshold $c=m_{(k+1)}$
(0-based index $k$). Then

$$\widehat F_H=\frac{\#\{m_i<c\}}{N}\le\frac{k}{N}\le\alpha,$$

and ties only make the left-hand side smaller. This is **conservative by construction**: no
distributional assumption is needed. Missing values do not enter the minimum.

**Truncated waiting time.** $\min(\tau,H)$; paths without an alarm are counted as $H$, but the
`alarmed` flag and the `first_alarm_day=-1` sentinel distinguish them from "alarm on exactly day
$H$".

**Population median detection time.** The smallest $t$ with $\Pr(\tau\le t)\ge0.5$; if less than half
of the paths are detected within two years, report "not reached within the observation window",
and do **not** fall back to "the median of the detected samples".

**Wilson interval.** For a proportion $\hat p=k/n$:

$$\frac{\hat p+\frac{z^2}{2n}\pm\frac{z}{1}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z=1.95996.$$

It covers only the Monte-Carlo noise of **a single proportion at a single time point**; it is not a
simultaneous confidence band for the whole time curve, and it does not include the uncertainty of
the threshold calibration itself.

**Random-closure analytic reference.** Each day the decision is closed independently with
probability $q$; let

$$q=1-(1-\alpha)^{1/H}
\;\Longrightarrow\;
\Pr(\tau\le t)=1-(1-q)^t=1-(1-\alpha)^{t/H},$$

which equals $\alpha$ exactly at $t=H$ and is **identical** for valid and invalid strategies. Its
truncated expectation is

$$E[\min(\tau,H)]=\sum_{t=0}^{H-1}\Pr(\tau>t)=\sum_{t=0}^{H-1}(1-q)^t
=\frac{1-(1-q)^H}{q}=\frac{\alpha}{q}.$$

These are exact analytic values, so **no** Monte-Carlo intervals are fabricated for them. Since
$\alpha<0.5$, the median detection time of this reference line is never reached within $H$.

---

## 7.5 Sampling uncertainty of the calibrated threshold itself

The threshold `c` is a statistic estimated from a finite calibration sample, not a constant.
Treating the whole pipeline (simulate → calibrate → freeze → measure on an independent test set) as
one experiment, the variance of the realised false-alarm rate `F̂_H` decomposes into two terms:

```
Var(F̂_H) = Var_calib( E[F̂_H | c] )  +  E_calib[ Var(F̂_H | c) ]
             threshold-sampling part      test-set binomial-sampling part
```

The second term is the usual `α(1-α)/n_test`, which is also the term covered by the Wilson
interval; the first term is entirely absent from the Wilson interval. `robustness.py` repeats the
whole pipeline under R independent root seeds, estimates the left-hand side by the sample variance
across replications, uses `α(1-α)/n_test` as the second term on the right, and subtracts the two
(truncating at 0) to obtain the threshold component:

```
sd_calib ≈ sqrt( max(0, s²_R − α(1-α)/n_test) )
```

It also performs a one-sided test of `H0: Var(F̂_H) = α(1-α)/n_test` using the chi-square sampling
law of the sample variance:

```
(R-1) · s²_R / σ0²  ~  χ²_{R-1}      (approximation: the rates across replications are treated as near-normal)
```

**This simulation has been superseded by the exact solution in Section 8.** The sampling law of order
statistics gives a closed-form answer that requires no repeated simulation, so the question "how
large should R be" does not arise. The earlier practice of choosing R according to "whether all
comparisons are significant" was an invalid stopping rule (deciding when to stop sampling on the
basis of significance biases the conclusions towards significance) and has been withdrawn; the 100
replications are kept as a one-off cross-check (in agreement with the analytic values,
max |z| = 1.57) and have been removed from the default run.

---

## 8. Stage 1.1: Exact sampling law of the threshold

The threshold is the `j`-th order statistic (1-based) of the calibration-set path minima:

```
j = floor(alpha * N) + 1          alarm rule: statistic < c
```

Let `M_i` be the minimum of the statistic over the alarm-eligible days of path i, with CDF `F`,
continuous (no ties). Then the **true** false-alarm probability of the threshold `c = m_(j)` is

```
p_FA(c) = P(M_test < c) = F(c) = F(m_(j)) ~ Beta(j, N+1-j)
```

This is the standard probability-integral-transform result for order statistics: the `F(M_i)` are
iid U(0,1), and their j-th order statistic follows `Beta(j, N+1-j)`. Note that it is
**detector-independent** — it only requires the distribution of the minimum to be continuous,
regardless of the specific form of the statistic.

Given `c`, the number of alarms in the test set is `Binomial(n_test, p_FA)`; mixing over the Beta
gives

```
alarms ~ BetaBinomial(n_test, j, N+1-j)
```

Its variance divided by `n_test²` is

```
Var(FAR) = p(1-p) * (N+1+n_test) / (n_test * (N+2)),      p = j/(N+1)
```

When `n_test = N` this tends to `2 p(1-p)/n_test`, i.e. **the total standard deviation is exactly √2
times the binomial standard deviation**.

`E[p_FA] = j/(N+1)` is slightly **above** the nominal α (because of the rounding direction). This is
a different matter from "the empirical false-alarm rate on the calibration set equals α exactly":
the latter is a mechanical consequence of the rule, while the former is the true coverage level of
this threshold.

> The Beta here is **the sampling distribution of a frequentist coverage probability**, not a
> Bayesian prior on strategy validity.

---

## 9. Stage 1.1: Failure probability and time

The notation is unified as

```
q_n = P(theta = 0 | r_1:n)        failure probability
U_n = logit(q_n) = -L_n           failure log odds
```

The Gaussian detector takes S=0 versus S=s as its two hypotheses, with one-step increment
`z_t * s/sqrt(D) - s^2/(2D)`. When the true Sharpe is `S_true`,

```
E[increment] = s (s - 2 S_true) / (2D),     Var(increment) = s^2 / D
```

Hence

```
U_n ~ N( U_0 + n * s(s - 2 S_true)/(2D),  n * s^2 / D )
```

When `S_true = 0` this reduces to `N(U_0 + n s^2/(2D), n s^2/D)` as given in the problem statement.

**The mean probability is not the sigmoid of the mean.** `E[q_n] = E[expit(U_n)]` must be integrated
over the distribution of `U_n` (this implementation uses Gauss–Hermite quadrature);
`expit(E[U_n])` is a different number, and the two already differ by about 0.05 at two years.

**Time scale of a fixed probability threshold.** `U` is a Brownian motion with drift
`mu = s^2/(2D)` and variance `sigma^2 = s^2/D`; its first time to reach `a = logit(b) - U_0` follows
an inverse Gaussian distribution,

```
E[tau] = a / mu = 2 D logit(b) / s^2
```

For b = 0.9, s = 1, D = 252: `E[tau] ≈ 1107` trading days ≈ 4.4 years. The time scale varies as
`1/s^2`; lowering s from 1 to 0.6 multiplies it by about 2.8.

> **Continuous vs daily observation**: the formula above is the result for continuous observation.
> The real detector looks only once per day, can stop only on integer days and cannot catch
> intraday crossings, so the actual first-passage time is **no shorter** than the analytic value.
> Treat it as a time-scale reference and a lower bound.

---

## 10. Stage 1.1: Random failure time

`T` is the **last valid trading day**:

```
t <= T  ->  Sharpe = 1        T = 0        invalid from the first day
t >  T  ->  Sharpe = 0        T = infinity never fails
```

Each path is observed for a further `post = 504` trading days after failure, so the monitoring
window of path i is `1 .. T_i + post`, and the post-failure window has the same length for every
`T`.

The implementation draws the noise first and then adds the drift, so two runs with the same seed and
the same shape have **bit-identical** noise: changing `T` changes only the drift after `T` and does
not touch any earlier return. This makes `T=0` bit-identical to the fixed invalid DGP and `T=inf`
bit-identical to the fixed valid DGP, which can be used directly as regression tests.

Denominator convention for the metrics: an alarm with `tau <= T` is a **pre-failure false alarm**;
that path is **permanently removed** from the post-failure denominator and can no longer be counted
as a detection. The truncated post-failure delay `min(tau - T, post)` is computed only on the
surviving paths, with censored paths counted as `post`; the median delay is reported only when **the
detected fraction is ≥ 50%**, and is otherwise marked as not reached — otherwise the censoring cap
would be misread as a delay.

---

## 11. Stage 2A: Four noise types

Returns are always `r_t = mu_S + sigma_0 * eps_t`, with `mu_S = S*sigma_ann/D` and
`sigma_0 = sigma_ann/sqrt(D)`. All four kinds of `eps` have **zero mean and unit unconditional
variance**, and the normalisation uses theoretical constants only.

**Heavy-tailed**: `eps = sqrt((nu-2)/nu) * X`, `X ~ t_nu`. `Var(X) = nu/(nu-2)`, so `Var(eps) = 1`.
With `nu = 5` the second moment exists and the fourth moment exists but its sampling variance does
not, so **the sample kurtosis cannot be used as an acceptance statistic**; tail frequencies are used
instead.

**Stochastic volatility**: `a_t = rho a_{t-1} + sqrt(1-rho^2) xi_t`, `a_0 ~ N(0,1)`, so
`a_t ~ N(0,1)` holds for every `t` (no burn-in). Taking `v_t = exp(a_t - 1/2)`,

```
E[v_t] = exp(-1/2) E[exp(a_t)] = exp(-1/2) exp(1/2) = 1
Var(eps_t) = E[v_t] E[z_t^2] = 1        (unconditional)
```

The true daily volatility is `sigma_0 sqrt(v_t)`. **The drift does not scale with v_t** — otherwise
the conditional Sharpe would be forcibly fixed, which is a different problem. The valid state only
requires the **long-run** Sharpe to be 1.

**Jumps**: `K_t ~ Poisson(lambda/D)`, `w_t ~ N(0,1)`,

```
eps_t = (z_t + kappa sqrt(K_t) w_t) / sqrt(1 + kappa^2 lambda/D)
```

Given `K_t`, `kappa sqrt(K_t) w_t ~ N(0, kappa^2 K_t)`, which has the same distribution as the sum of
`K_t` independent `N(0, kappa^2)` jumps. By the law of total variance,
`Var(kappa sqrt(K) w) = E[kappa^2 K] = kappa^2 lambda/D`, and the mean is 0. The numerator variance is
`1 + kappa^2 lambda/D`, so `Var(eps) = 1`.
**Units**: `lambda` is the **annual** jump intensity (expected number of jumps per year), with daily
intensity `lambda/D`; `kappa` is the standard deviation of a **single jump**, in units of the daily
Gaussian shock.

**Degenerate cases**: a stochastic-volatility amplitude of 0, `kappa = 0` or `lambda = 0` must each
reproduce the Gaussian model bit for bit.

**Error assessment**: with `rho = 0.98` the half-life of the log variance is about
`log(0.5)/log(0.98) ~ 34` days, so trading days within a path are strongly correlated. All standard
errors are computed at the **path level**; path-days are not treated as independent samples.

---

## 12. Stage 2A: True-volatility oracle

Added only in the stochastic-volatility scenario; it uses the true current variance but keeps the two
**fixed-mean** hypotheses:

```
dL_t = (mu1 - mu0) [ r_t - (mu1+mu0)/2 ] / sigma_t^2,    mu0 = 0, mu1 = sigma_ann/D
```

Derivation: `log N(r; mu1, sig^2) - log N(r; mu0, sig^2) = [(r-mu0)^2 - (r-mu1)^2]/(2 sig^2)`
`= (mu1-mu0)[r - (mu0+mu1)/2]/sig^2`.

Note that this is **not** "dividing returns by the current volatility and then reusing the update for
a fixed conditional Sharpe" — the latter tests a different pair of hypotheses. This oracle uses
information that no deployable rule can obtain, and serves only to calibrate an upper benchmark.

---

## 13. Stage 2B: Causal EWMA variance forecast and two detectors

The forecast uses past returns only:

```
m = (mu_0 + mu_1)/2,   v_1 = sigma_0^2
v_{t+1} = lambda v_t + (1 - lambda) (r_t - m)^2,   lambda = 0.94
```

`v_t` is the forecast of that day's variance made **before seeing day t's return**, and depends only on
`r_1..r_{t-1}`. No full-sample backcast, full-path sample variance or true-state initialisation is
used.

**The cost of the fixed midpoint m.** Under both true-mean hypotheses the squared residual picks up the
same extra term: `E[(r - m)^2 | mu_i] = v + (mu_1 - mu_0)^2/4`. This is therefore not a strictly
unbiased variance recursion. In this round `(mu_1-mu_0)^2/4` is about 0.0992% of `sigma_0^2`.
It was chosen so as not to read the true state and not to introduce a second mean estimator.

**The Gaussian update** is the exact log-likelihood ratio of `N(mu_1, v_t)` against `N(mu_0, v_t)`;
the quadratic terms cancel:

```
dL_t = (mu_1 - mu_0)(r_t - m) / v_t
```

**The Student-t update** shares the same `v_t`, with scale

```
b_t = sqrt( (nu-2)/nu * v_t )
```

so that the **variance** of the t distribution equals `v_t` (`b_t` is a scale parameter, not a
standard deviation; the two differ by a factor of `sqrt(nu/(nu-2))`).

**Degeneracy relation** (a strong constraint on implementation correctness): with `lambda = 1`, `v_t`
stays at `sigma_0^2`, and the two EWMA detectors must reduce exactly to their respective fixed-scale
baselines.

**Nature of the output**: this is **the model posterior under each detector's own EWMA
conditional-distribution assumption**, not the exact posterior obtained by integrating over the true
latent SV state.

**QLIKE** `v/vhat - log(v/vhat) - 1` is a mechanism diagnostic; under model misspecification the
ranking by different forecast losses may differ (Patton), and the main conclusions are still
determined by the false-alarm and detection metrics.

**Information-equivalent observation days** `N_info(n) = sum_{t<=n} sigma_0^2 / v_t`.
When `v_t = exp(a_t - amp^2/2)` and `a_t ~ N(0, amp^2)`,
`E[sigma_0^2/v_t] = E[exp(amp^2/2 - a_t)] = exp(amp^2)`, so `E[N_info(n)] = n e^{amp^2}`,
which is `n·e` for `amp = 1`. It uses the true variance and is an interpretive tool for the oracle,
not data available to a deployable detector.

---

## 14. Stage 2C: Order-statistic thresholds with a calibration-error buffer

The alarm rule is "statistic strictly less than threshold", so a path raises an alarm within the
monitoring horizon if and only if its **minimum statistic** `M = min_t L_t` falls below the threshold
`c`. The **true** FAR of threshold `c` is simply the value `F(c)` of the CDF of `M`.

Take `c = M_(k)` (the k-th smallest minimum among N calibration paths). When `F` is continuous and
the **paths** are iid (days within a path need not be independent), the probability integral
transform gives

```
F(M_(k)) ~ Beta(k, N+1-k)
```

Using the Beta–binomial duality

```
P{Beta(k, N+1-k) > alpha} = P{Binomial(N, alpha) <= k-1}
```

we require `P{Bin(N, alpha) <= k-1} <= delta/J` and take the **largest** feasible k: `M_(k)` is
non-decreasing in k, and a higher threshold produces more alarms, so the largest feasible k is the
least conservative choice that still meets the requirement.

**Direction of the unified threshold.** `F_g` is non-decreasing, so `c_unified = min_g c_g <= c_g`
implies `F_g(c_unified) <= F_g(c_g) <= alpha` (on the event where the guarantee for that scenario
holds). A Bonferroni union bound over the J items gives a simultaneous statement at level
`1 - delta`.

**Three probabilities that are different things**: `delta` is the probability that the calibration
procedure fails; `alpha` is the strategy's false-alarm budget; the Wilson intervals in the test
tables are sampling intervals for a single rate on a finite test set.

**Scope of the guarantee**: it holds only under the listed complete path-generating laws and their
assumptions. It does not cover arbitrary market distributions, arbitrary parameters or spliced
combinations of laws, nor does it guarantee that the FAR observed in any single finite test is below
the target. This is an order-statistic construction for monotone path-wise false-alarm risk, **not** a
full implementation of the Learn-then-Test framework.

---

## 15. Stage 2D: Combined SV and jump DGP

```
r_t   = mu_S + sigma_0 eps_t
a_t   = rho a_{t-1} + sqrt(1-rho^2) xi_t,   a_1 ~ N(0,1)
v_t   = exp(A a_t - A^2/2)
K_t   ~ Poisson(lambda/D)
eps_t = ( sqrt(v_t) z_t + kappa sqrt(K_t) w_t ) / sqrt(1 + kappa^2 lambda/D)
```

The underlying random sources `xi, z, w, K` are mutually independent. **Jumps are an independent
additive component**: they are not multiplied by `sqrt(v_t)`, so jumps landing on calm days are
relatively larger.

**Zero mean**: each of the two terms is the product of a zero-mean random variable and an independent
factor with zero mean.

**Unit unconditional variance**: `a_t` has marginal `N(0,1)`, so `A a_t ~ N(0, A^2)` and
`E[v_t] = exp(A^2/2 - A^2/2) = 1` (for any `A`). The two numerator terms are independent,

```
Var(sqrt(v) z) = E[v] E[z^2] = 1
Var(kappa sqrt(K) w) = kappa^2 E[K] E[w^2] = kappa^2 lambda/D
```

The numerator variance is `1 + kappa^2 lambda/D`, so `Var(eps) = 1` holds for **any** `(A, kappa)`.

**Degeneracy relations**: `kappa = 0` gives `eps = sqrt(v) z`, i.e. the Stage 2A SV model;
`A = 0` gives `v = 1` and `eps = (z + kappa sqrt(K) w)/sqrt(1+kappa^2 lambda/D)`,
i.e. the Stage 2A jump model; with both switched off, `eps = z`.

**kappa is not a monotone difficulty parameter.** The unit-variance constant simultaneously shrinks the
diffusion scale on jump-free days:

```
sd on jump-free days = 1 / sqrt(1 + kappa^2 lambda/D)
```

With `lambda = 2` and `D = 252`, `kappa = 5` gives 0.9135 and `kappa = 8` gives 0.8143.
That is, the larger `kappa` is, the **quieter** ordinary trading days become, and the single-day
signal-to-noise ratio is actually higher. The report does not presuppose a monotone effect of
`kappa`.

**Effect of a single shock on the EWMA variance forecast.** Adding a shock of `c sigma_0` on day t
raises the next day's forecast variance by about `(1-lambda_ewma) c^2 sigma_0^2` (ignoring the
midpoint term), which then decays by a factor `lambda_ewma` per day, with half-life
`ln(0.5)/ln(lambda_ewma)`. For `lambda_ewma = 0.94` the half-life is about 11.2 trading days.
The Student-t likelihood can suppress the **same-day** evidence increment, but **on the following
days** it shares the same inflated variance forecast with the Gaussian and is therefore affected in
the same way.

---

## 16. Stage 2E: Truncated variance update and 2×2 ablation

### 16.1 Definition

Only one step of the variance recursion is changed; the likelihood itself is untouched:

```
    plain:      V_{t+1} = lambda V_t + (1-lambda) e_t^2
    truncated:  V_{t+1} = lambda V_t + (1-lambda) min(e_t^2, c^2 V_t)
    e_t = r_t - m,  lambda = 0.94,  c = 4 (fixed, not searched)
```

`V_1 = sigma_daily^2`, the same as in the plain version. **The same-day likelihood still reads the full
`r_t`**: truncation acts only on the term written into the next day's variance state.

**This is a new model, not a robustified rewrite.** A detection rule is defined jointly by its
statistic and its threshold; changing the recursion for `V_t` changes the entire path of the
statistic, so the two truncated versions inherit no FAR guarantee from the plain-EWMA thresholds and
must be recalibrated.

### 16.2 Two proven properties

**Property 1 (day-by-day dominance):** For the same return sequence, `Vt_trunc <= V_plain` holds for
all t.

Induction. They are equal at t=1. Suppose `Vt <= V`. Then
`min(e^2, c^2 Vt) <= e^2` and `lambda Vt <= lambda V`; adding the two gives `Vt_{t+1} <= V_{t+1}`. ∎

**Property 2 (the direction of the FAR is not determined automatically):** The Gaussian increment is
`(mu1-mu0)(r_t-m)/v_t`, with the variance in the denominator. A smaller `v_t` makes `|dL_t|`
**larger**, i.e. the statistic of the truncated version fluctuates more, not less. The inference
"truncation makes the variance more stable ⇒ fewer false alarms" is therefore invalid; the direction
of the FAR can only be settled by calibration.

### 16.3 Effect of truncation on the variance level: what can and cannot be said

**What can be said (one exact reference quantity).** If the standardised residual
`u = e_t/sqrt(V_t)` is treated as standard normal, the second moment of the input to a one-step
update is

```
    E[min(u^2, c^2)] = (2 Phi(c) - 1) - 2 c phi(c) + 2 c^2 (1 - Phi(c))
```

For `c = 4` this equals `0.99987946`, which is `0.01205%` less than the untruncated `E[u^2] = 1`.

**What cannot be said.** This number is **not** the long-run variance bias of the recursion. Deriving
a long-run bias from it would require at least the following four conditions to hold simultaneously,
and not one of them holds in this project:

1. **The truncation point is not a constant.** The cap `c^2 V_t` depends on the model's own state
   `V_t`, which is random and correlated with the history of `e_t`;
   `E[min(e_t^2, c^2 V_t)] != E[V_t] E[min(u^2, c^2)]`.
2. **`u` is not standard normal.** The true DGP contains stochastic volatility and jumps, so the
   distribution of `e_t/sqrt(V_t)` is heavy-tailed and time-varying; the fraction truncated by the
   `min` is therefore not constant.
3. **The midpoint is fixed.** `e_t = r_t - m` is computed around a **fixed** `m`, not around the
   conditional mean, so `E[e_t^2]` itself contains a term `(mu - m)^2`.
4. **The recursion is a nonlinear feedback.** The weights of plain EWMA sum to 1, so the linear
   fixed-point argument `E[V_inf] = E[e^2]` holds; in the truncated version the update term
   `min(e_t^2, c^2 Vt_t)` appears on both sides of the equation, and `E[Vt_inf]` cannot be obtained by
   substituting the mean of the input into the same fixed point. Variance pushed down → cap
   `c^2 Vt_t` lower → truncation more frequent → variance pushed down further: this is a
   self-reinforcing feedback, and its steady state is not the result of a linear extrapolation.

This project therefore records only the first statement, and does **not** interpret the truncated
`Vt_t` as an unbiased estimator of any of the working scale, the diffusion variance or the total
variance. It is an internal state of a detection rule; its justification comes from its calibrated
detection performance, not from whether it equals some true value.

**Precise statement for the same pre-shock state.** Truncation acts only on the term written into the
**next day's** variance, so, given the same pre-shock state, it does not change the **same-day**
variance forecast that has already been computed, nor does it remove the shock from the same-day
likelihood in advance. The two models' same-day forecasts nevertheless usually differ — because their
past update histories differ, `Vt_t` and `V_t` have long since diverged. These two points must be
stated separately.

### 16.4 The interaction term: what it answers and what it does not

The four cells of the 2×2 ablation give two main effects and one interaction:

```
    main effect (truncation, t side):  D(trunc-t) - D(EWMA t)
    main effect (truncation, G side):  D(trunc-G) - D(EWMA G)
    interaction I(alpha):              [D(trunc-t) - D(EWMA t)] - [D(trunc-G) - D(EWMA G)]
```

**`I > 0` says only one thing: the gain from truncation on the Student-t side is larger than the gain
on the Gaussian side.** It does not prove that the gain on the Gaussian side is zero (for example,
+3pp on the t side and +1pp on the G side also give a positive interaction), it does not prove that
the Student-t likelihood is a necessary condition, and it does not prove that the two mechanisms are
complements rather than substitutes. Likewise, `I < 0` does not prove that they are substitutes.

**An interval covering 0 does not mean that additivity holds.** It only means that the resolution of
this round did not separate the two. The interval half-width must be reported alongside: if the
half-width is of the same order as the main effects, then "no interaction" and "the sample size
cannot reveal an interaction" are indistinguishable in this batch of data.

"Counting on which side more scenarios are significant" is not a test of the interaction: both sides
can be significant and of the same size.

### 16.5 Comparing two budgets: a difference in significance is not the significance of a difference

If the interval for `I(0.05)` excludes 0 while the interval for `I(0.15)` covers 0, one **cannot**
conclude from this that the interaction differs between the two budgets. This is the classic error
pointed out by Gelman–Stern. The correct estimand is the difference itself:

```
    I(0.05) - I(0.15)
```

Its interval must be obtained by **paired** resampling; it cannot be obtained by subtracting two
independently generated intervals. In each resample:

```
    draw a set of calibration indices ic and a set of test indices ii     <- shared by both budgets and all four models
    for each budget alpha:
        k_alpha = buffered rank for this budget (0.05 -> 431, 0.15 -> 1386)
        for each model m:
            threshold = k_alpha-th smallest of cal_minima[m][ic]    <- recomputed on the same resampled calibration sample
            D[m, alpha] = mean( test_invalid_minima[m][ii] < threshold )
    I[alpha] = (D[tt,a]-D[et,a]) - (D[tg,a]-D[eg,a])
    record I[0.05] - I[0.15]
```

**Limits of the multiplicity control.** The Bonferroni split `delta/J` in the calibration protocol
protects the **false-alarm constraint of the thresholds**: it ensures, with high probability, that
each method's realised false-alarm rate does not exceed its own budget. It does **not** provide a
simultaneous confidence guarantee for bootstrap intervals of detection-rate differences — those are a
different family of statistics and would require a joint correction applied to the differences
themselves.

### 16.6 Three-way classification of false-alarm timing and the theoretical coincidence baseline

For the **first** false alarm `tau` on a valid-strategy path, the latent jump count `k_t` (visible
only to the evaluator) is used to split it into three classes:

```
    class 1: k_tau >= 1                                          the alarm day itself carries a jump marker
    class 2: k_tau = 0 and there exists s, max(1, tau-W) <= s <= tau-1, k_s >= 1
    class 3: all others
```

`W = 20`. By construction the three classes are mutually exclusive and their union is
`{tau != -1}`, so the three always sum to the total FAR.

**Theoretical coincidence baseline.** `K_t ~ Poisson(lambda/D)`, with `lambda = 2` per year and
`D = 252`. Note that `lambda/D = 0.00793651` is the **mean count**, not a probability; the probability
that a given day carries a marker is

```
    p_J = 1 - exp(-lambda/D) = 0.00790510
```

The two differ from the fourth significant digit onwards. In the `kappa = 0` scenario the jump size is
0, so the marker affects neither the returns nor the stopping rule and is therefore independent of
`tau`; the conditional probabilities of the three classes (conditional on the path actually having a
false alarm) are

```
    P(class 1 | alarm) = p_J
    P(class 2 | alarm) = E_tau[ (1 - p_J) * (1 - (1 - p_J)^min(W, tau-1)) ]
    P(class 3 | alarm) = 1 - P(class 1) - P(class 2)
```

**Class 2 must be aggregated over alarm days.** An alarm on day `tau` has only `min(W, tau-1)` trading
days to look back on, and the window of an early alarm is shorter than `W` days, so the probability
depends on `tau` and must be averaged over the **observed distribution of alarm days**; the
`tau >> W` limiting constant `(1-p_J)(1-(1-p_J)^W) = 0.14561318` cannot simply be used.
In this project's data the two happen to coincide to machine precision, because the earliest first
false alarm occurs on day 42 (the statistic starts from `log-odds = 0` and needs time to drift below a
fairly deep threshold), so no path falls in the boundary region. This is a property of this batch of
data, not a reason to omit the formula.

The corresponding **unconditional** probabilities (with all valid paths as the denominator) are the
three numbers above each multiplied by the total FAR `F`, and the three sum to `F`.

**Both denominators must be reported.** Looking only at "share of false alarms" can be distorted by
changes in the other classes: a falling share may also be caused by rising false alarms elsewhere
rather than by fewer false alarms in that class.

**This is a timing association, not a causal decomposition.** The observed share minus the coincidence
baseline cannot be called the causal contribution of jumps to false alarms. Note also the independence
structure of this DGP: the jump counts `k_t` are drawn independently by `rng.poisson(lambda/D)` and are
**independent** of the latent stochastic-volatility state `v_t` (observed `corr(v_t, k_t) = -0.0024`,
n = 200,000, standard error 0.0022). What is correlated with both is the **observed** large return
(`corr(v_t, |r_t|) = 0.43`, `corr(k_t, |r_t|) = 0.37`), not the latent volatility state itself.

---

## 17. A Gaussian theoretical reference with explicit conditions

The sole purpose of this section is to prevent "the gains from changes are getting smaller" from being
read as "we are already close to the information upper bound". What it provides is a reference that
is **exact where its assumptions hold and inapplicable elsewhere**, not an upper bound for the
scenarios of this project.

### 17.1 Setup and derivation

Let the daily returns `r_t` be iid `N(mu, sigma^2)` with `sigma` known, `D = 252`, observed for `h`
years, i.e. `n = hD` days. Two simple hypotheses: valid `mu1 = s sigma / sqrt(D)` (annualised Sharpe
`s`) and invalid `mu0 = 0`. "Alarm" = judged invalid. The constraint is that the probability of an
alarm within the horizon **when valid** does not exceed `alpha`, so in Neyman–Pearson terms **valid**
is the null hypothesis, **invalid** is the alternative, and the detection rate is the power.

The sufficient statistic is the sample mean `bar r`; the likelihood ratio is monotone in `bar r`, so
the most powerful test raises an alarm when `bar r` is small enough. Under the null,
`(bar r - mu1) sqrt(n) / sigma_d ~ N(0,1)` (`sigma_d` is the daily standard deviation), and level
`alpha` requires the critical point `mu1 - z_{1-alpha} sigma_d / sqrt(n)`.
Standardising it under the alternative `mu = 0` and substituting `mu1 sqrt(n)/sigma_d = s sqrt(h)`
gives

```
    D_max(h, alpha) = Phi( s sqrt(h) - Phi^{-1}(1 - alpha) )
```

`sigma` and `D` cancel completely, leaving only `s sqrt(h)`.

### 17.2 Why sequential rules are also bound by it

Suppose any sequential rule raises an alarm on some day before time `n`. The event "an alarm has been
raised by day `n`" is measurable with respect to the information set `F_n` at time `n`, i.e. it is the
rejection region of a (possibly randomised) test on the same data. Its level is the cumulative
false-alarm rate `<= alpha`. The Neyman–Pearson lemma gives an upper bound on the power of **all**
such tests, sequential or not, because stopping early can only discard information, never add it.

**When this argument fails:** if the rule can use information outside `F_n` (for example future data
or exogenous cross-sectional evidence), or if the two hypotheses are not the pair of simple
hypotheses set out in this section, the upper bound does not apply.

### 17.3 Reference table

`s in {1, 0.6}`, `h in {0.25, 0.5, 1, 2}`, `alpha in {0.05, 0.15}`,
generated by `gaussian_bound.reference_table()` and checked against a second, independent numerical
route (constructing the test directly on the daily scale, without any analytic reduction); the two
agree to 4.4e-16.

### 17.4 Scope

- **Not applicable to the SV+jump scenarios.** There, volatility itself is predictable and the weight
  given to evidence can be adjusted over time, so detection rates above this number are not a
  contradiction; the performance of the EWMA-type methods from Stage 2B onwards is exactly a
  manifestation of this.
- **It gives no lower bound.**
- **It cannot be used to say "existing methods are already close to the information upper bound
  determined by the Sharpe".** The same annualised Sharpe does not uniquely determine how identifiable
  a return series is: predictable volatility, the shape of the centre of the distribution and the jump
  structure all change how much evidence can be extracted.
- **Denser sampling does not automatically bring more information about the drift.** In a model with
  constant drift and iid noise, the information about the drift within a given calendar window is
  determined by `s sqrt(h)` and is independent of the sampling frequency; higher frequency helps to
  identify the volatility and jump structure, which is a different matter.

**The only place where it can legitimately be applied** is the Gaussian control scenario
(`A = 0, kappa = 0`), where returns really are iid Gaussian. Plugging in each method's **realised**
false-alarm rate (rather than the budget), all eight methods of this project fall below the upper
bound; the closest is the exact likelihood-ratio statistic `binary_gaussian`, which reaches 87.7% of
the bound (`alpha = 0.15`). The remaining gap is the cost of sequential monitoring: the false-alarm
budget is spread over every day of the two years, whereas the upper bound corresponds to spending the
entire budget at once at the end of the horizon.

---

---

## 18. Stage 3A: A weaker signal

### 18.1 Change only one thing

```
    r_t = mu_S + sigma_0 eps_t
    valid:    mu_1(s) = s sigma_ann / D,   s in {1, 0.6}
    invalid:  mu_0 = 0
```

D = 252, H = 504, sigma_ann = 0.10, prior 0.5, five noise scenarios, eight detection methods,
EWMA lambda = 0.94, Student-t nu = 5 and truncation c = 4 are all carried over unchanged.

**The candidate means must be kept in sync with the generator.** In the s = 0.6 group the detector
knows that the candidate states are 0.6 or 0, so both `mu_1` in the likelihood ratio and the EWMA
midpoint `m = (mu_0 + mu_1)/2` use 0.6.
Lowering only the generator while the detector continues to assume s = 1 is a model-misspecification
experiment, which is a different question.

Two exceptions follow from the definitions: the statistics of the 252-day trailing Sharpe and of the
fixed-scale rolling control are rolling sample quantities of the `sqrt(D) * mean / scale` type and
contain no candidate mean; for them, `s` enters only through the calibrated threshold and the data.

### 18.2 Common random numbers

The two `s` settings **share** the noise block of each (scenario, role): the same `eps`, with only the
drift changed. This makes every comparison between s = 1 and s = 0.6 a paired comparison, and also
means that

```
    base noise paths        = 5 scenarios x 3 roles x 10,000 = 150,000
    return-path evaluations = 150,000 x 2 values of s        = 300,000
```

The latter are **not** 300,000 mutually independent noise paths. Calibration blocks and test blocks
are never shared.

### 18.3 Calibration scope

This round includes both values of `s` in the simultaneous calibration:

```
    J = 8 methods x 5 scenarios x 2 budgets x 2 signal strengths = 160
    delta = 0.05,  delta/J = 3.125e-4
    n_cal = 10,000  ->  rank 427 (alpha = 0.05),  1379 (alpha = 0.15)
```

Three independent routes (binomial tail, Beta upper tail, brute-force search) give the same pair of
ranks. Widening J from 80 to 160 lowers the ranks (1386 -> 1379, 431 -> 427), i.e. the thresholds
become deeper and more conservative; widening the set of simultaneous comparisons should not buy
looser thresholds.

Coverage: these five fixed scenarios; this does **not** constitute a unified-threshold guarantee in an
unknown environment.

### 18.4 One horizon, one threshold

Each method has only **one** two-year threshold per (scenario, s, alpha). The cumulative detection and
false-alarm rates on days 63, 126 and 252 in the report are read from the same threshold, not obtained
by spending a full alpha at each horizon. The Gaussian reference curve in the figures is computed
horizon by horizon with the **nominal budget**, so it lies above what our rule can achieve; this is a
difference in design, not a defect.

### 18.5 Time scale of evidence accumulation

In the pure Gaussian binary model, in the invalid state and with initial odds of 1, `U_n = -L_n`
satisfies

```
    E[U_n] = s^2 n / (2D),    Var(U_n) = s^2 n / D
```

because the daily increment of `L` is `z_t s/sqrt(D) - s^2/(2D)`, which under `mu = 0` has mean
`-s^2/(2D)` and variance `s^2/D`. The rate at which mean log evidence accumulates scales with `s^2`,
hence `1/0.6^2 = 2.78`.

**This is only an evidence time-scale reference under that model.** It does not claim that all noise
scenarios, thresholds and actual stopping times scale by a factor of 2.78 — the measured numbers in
this stage do not scale that way, because the thresholds are calibrated by quantile, the noise is not
iid Gaussian, and the stopping time is a first-passage time rather than a fixed-horizon statistic.

### 18.6 The interpretive metric q >= 0.9

"The cumulative fraction that first reaches a failure probability of 0.9" is a convenient scale for
communication, but it is **not** a 10% false-alarm control: `q >= 0.9` is a boundary on the posterior
probability and has no identity relationship with "the probability that a valid strategy crosses it".
The fraction of valid strategies that wrongly reach this level must therefore be reported alongside
it, and it is not used in place of the formal ranking.

---

---

## 19. Stage 3A.1: Monitoring-horizon diagnostic

### 19.1 The two arrangements answer different questions

```
    A   calibrate one threshold for H = 504 and read its cumulative performance at days 63/126/252/504
    B   calibrate a separate threshold for each H in {63,126,252,504} and evaluate each at its own cutoff
```

**The four horizons in B are four independently designed monitoring schemes.** Each spends its own
`alpha` within its own window. Chaining them together and claiming that they still share one two-year
budget counts four independent false-alarm budgets as one.

The early-detection gain of B over A therefore **must be read together with the false-alarm rate over
the same period**: the day-by-day inputs of the two arrangements are identical (the same returns, the
same statistic path); B merely spends its false-alarm budget earlier. Spending more false-alarm budget
earlier is not an improvement in the model's ability to use information.

**Comparing at a matched spent budget** is the more convincing reading: if B spends `f` of false
alarms in a half-year window to obtain `d_B`, and A spends the same `f` in a two-year window to obtain
`d_A`, then `d_A / d_B` directly answers "at the same false-alarm cost, which is more effective:
lengthening the window or spending earlier".

### 19.2 Horizons must be prefixes

The statistic is computed **once** over the full 504 days, and each horizon takes the minimum over
`[start day, H]`:

```
    prefix_minima(stat, elig, H) = min over days [elig, H] of stat
```

Changing `H` does not recompute the statistic, so it cannot change the EWMA state, returns or
likelihood increments on any earlier day. The tests also verify this from the other side: truncating
the **data** at `H` and then computing the statistic, and slicing the **statistic** at `H`, give
pointwise equal results (`atol = 1e-12`).

Two structural properties follow: because the horizons are nested, the prefix minimum can only
decrease as `H` increases; and under the same budget a longer horizon has more opportunities to
cross, so its calibrated threshold is necessarily no higher than that of a shorter horizon.

### 19.3 Consistency check

`B` at `H = 504` is by definition `A`: the same threshold and the same cutoff.
Their detection rates, false-alarm rates and `E[min(tau,H)]` must be bit-identical, and `B - A` must be
identically 0. This is the internal consistency check of the whole comparison of arrangements.

### 19.4 Truncated times are not comparable across horizons

The upper limit of `E[min(tau, H)]` is `H` itself: the truncated mean for `H = 63` can be at most
63 days. "The short-horizon scheme has a smaller truncated mean time" therefore **cannot** be read as
"it is faster". All time comparisons in this stage are paired at **the same cutoff**.

### 19.5 Beyond-horizon diagnostic

The thresholds calibrated for `H = 63/126/252` are applied all the way to day 504, and the realised
cumulative false-alarm rate is recorded. The budget obtained by a short-horizon threshold applies to
its own window and **does not automatically inherit the two-year guarantee**; this diagnostic
quantifies the size of the overshoot.

### 19.6 Calibration scope

The number of rules in this round is `4 x 2 x 2 x 2 x 4 = 128`, but the per-cell buffer `delta/160` of
Stage 3A is kept (rather than the looser `delta/128`), so the calibration is no weaker than in the
previous round. With `n_cal = 10,000` the ranks are still 427 (`alpha = 0.05`) and 1379
(`alpha = 0.15`). This guarantee covers the two scenarios, four methods, two values of `s`, two
budgets and four horizons listed for this round; it is **not** a unified guarantee across all
historical stages.

---

---

## 20. Stage 3B: A limited diagnostic of random failure times

### 20.1 Setup

```
    theta_t = s   when t <= T          T is the last valid trading day
    theta_t = 0   when t >  T
```

`T` is fixed at 0, 126 or 252, or drawn uniformly from the integer trading days `0..252`, plus an
always-valid control with `T = infinity`. `T` is independent of the noise, the latent volatility and
the jumps, and is visible only to the generator and the scoring; the model does not reset its prior,
accumulated evidence, variance state or any window at `T`.

**The coverage is "failure within one year".** `T <= 252` ensures that every setting has a complete
252-day post-failure window inside the 504-day horizon. This is not a lifetime study over 0–5 or
0–10 years.

The thresholds are the frozen 504-day calibrated values from Stage 3A and are **not calibrated
separately for the true T** — the true failure time is not available in practice, and calibrating on
it would produce a rule that cannot be deployed.

### 20.2 Three quantities that must be read together

```
    pre-failure false alarm   P(tau <= T)
    conditional detection     P(T < tau <= T+h | tau > T)
    joint detection           P(T < tau <= T+h) = P(tau > T) * P(T < tau <= T+h | tau > T)
```

A difference in joint detection can come either from a difference in conditional detection or from a
difference in pre-failure false alarms; reporting only one of them conflates two completely different
behaviours. The identity is verified on every row.

**Encoding of `tau = infinity`.** Never alarming is recorded as `-1` in the implementation. Such a path
is a **survivor** of the pre-failure window and an **undetected** path after failure, never an early
false alarm. Treating `-1` as a small number would simultaneously overstate pre-failure false alarms
and understate the survival rate, getting the direction completely backwards.

**The post-failure delay** `E[min(tau - T, h_max)]` (conditional on `tau > T`) has all survivors as its
denominator: undetected survivors are counted at the window cap rather than dropped, and paths with an
early false alarm are not counted as zero delay — they are not in the survivor set at all. The
post-failure median likewise uses all survivors as its denominator.

**Common window.** Different `T` must use the same `h`; otherwise earlier failures would implicitly get
longer observation opportunities, and the comparison would be of opportunity rather than of method.

### 20.3 Evidence at the failure time

`U_T = -L_T` and `q_T = expit(U_T)` describe how much favourable history the model carries at the
instant of failure. **For `T = 0` the prior value is used**, not the last day of the array — the latter
would misreport the state at the end of the whole horizon as the initial state.

This diagnostic is valid only for **log-odds-type** statistics. The statistic of the 252-day trailing
Sharpe is an annualised Sharpe ratio, not on the log-odds scale; applying `expit` to it yields a number
with no interpretable meaning, so it is not listed.

Under iid Gaussian noise, a fixed `T` and over **all paths**, one can verify

```
    E[U_T] = -s^2 T / (2D),      E[U_{T+h} - U_T] = s^2 h / (2D)
```

**These two do not automatically apply to samples filtered by early false alarms** (conditional
statistics are exactly such samples), nor do they automatically apply to SV+jumps.

### 20.4 Two sources of confounding when reading the results

At `T = 252` the 252-day trailing Sharpe simultaneously has fewer false alarms and higher conditional
and joint detection; this is an observed fact. **This stage does not decompose the source of that
advantage**; at least four mechanisms may be acting at the same time:

1. **Warm-up period**: it can speak at the earliest on day 252, so for `T = 0` only half of the
   post-failure window is usable, while for `T = 252` the warm-up ends exactly at the moment of
   failure; the same fact also limits its opportunities to alarm before failure.
2. **The rolling window progressively removes pre-failure data**: after failure, the favourable history
   slides out of the window, whereas cumulative models do not automatically discard it.
3. **Variance state**: EWMA-type methods carry into the moment of failure a variance estimate formed
   from past data.
4. **Survivor selection**: conditional statistics are computed only on paths that were not stopped by
   an early false alarm, and the survivor sets differ across methods.

This advantage therefore cannot be attributed to any single one of these mechanisms, nor can it be
asserted that it is "just window alignment" or "stronger discriminating power". This stage tested only
four failure settings and did not scan `T`, so it also cannot claim that `T = 252` is the globally most
favourable failure time.

### 20.5 A single definition of median detection time

The whole project uses one definition: **the first day on which the cumulative alarm fraction over all
paths reaches 1/2**, where reaching exactly 1/2 counts as reached; if it is not reached, this is marked
as such, without extrapolation.

A path that never alarms is not missing data but a detection time greater than the horizon, so it must
stay in the denominator. Taking "the median of the alarmed subset" answers a different question and is
undefined when the detection rate is below one half — which is precisely the norm in weak-signal
settings.

---

## 21. References

- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52.
  <https://alo.mit.edu/wp-content/uploads/2017/06/The-Statistics-of-Sharpe-Ratios.pdf>
  — statistical error of the sample Sharpe and time-scale conversion.
- SciPy, `scipy.stats.t`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html>
  — meaning of `logpdf` and of `loc`/`scale` (scale is not the standard deviation).
- Stan Functions Reference / User's Guide, scale-mixture representation of the Student-t:
  <https://mc-stan.org/docs/stan-users-guide/robust-noise-models.html>
  — a heavy-tailed likelihood is equivalent to giving each observation a random variance, which is
  another way of understanding "down-weighting".
- NumPy random `SeedSequence` / `Generator`:
  <https://numpy.org/doc/stable/reference/random/parallel.html>
  — reproducible, mutually independent random streams for the calibration, test and diagnostic data.

**Scope statement.** The references above support the formulas and the API usage. The two binary
baselines implemented in this stage are not reproductions of any "EKV" or "Stefan" paper whose title
has not been confirmed, and they should not be cited as such.
