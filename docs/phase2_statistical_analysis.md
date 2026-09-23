# Phase 2: Statistical Foundation & Covariate Balance Report

## 1. Executive Summary

This document presents the Phase 1 & 2 statistical validation for the **IncrementIQ / Criteo Uplift Modeling** pipeline. The objectives of this phase are:
1. **Data Ingestion Integrity**: Verify the raw 13.98M-row gzip dataset against published benchmarks with an explicit schema and post-treatment leak suppression.
2. **Covariate Balance (SMD)**: Validate whether the treatment and control groups are balanced across all 12 pre-treatment features ($f_0$–$f_{11}$) to confirm true randomization.
3. **Classical A/B Hypothesis Testing**: Measure the average treatment effect (ATE), relative lift, confidence intervals, and two-proportion $z$-test statistics for both dense (`visit`) and sparse (`conversion`) outcomes.
4. **Retrospective Power Analysis**: Compute the Minimum Detectable Effect (MDE) under the actual observed sample sizes and 85/15 imbalance ratio.

---

## 2. Ingestion & Baseline Data Validation (Phase 1)

The dataset was ingested using PySpark with an explicit `StructType` schema. The post-treatment leak column `exposure` was pruned immediately upon ingestion.

| Metric | Expected Benchmark | Observed (Full Parquet) | Observed (10% Dev Sample) | Validation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Rows** | ~13,979,592 | **13,979,592** | **1,399,999** | Passed |
| **Treatment Ratio** | $0.84600$ | $0.85000$ | $0.84990$ | Passed ($\le 0.02$ tol) |
| **Visit Rate** | $0.04699$ | $0.04699$ | $0.04713$ | Passed ($\le 0.02$ tol) |
| **Conversion Rate** | $0.00292$ | $0.00292$ | $0.00301$ | Passed ($\le 0.02$ tol) |
| **Missing Values (Nulls)** | 0 across all cols | 0 | 0 | Passed (0 nulls) |
| **Schema Integrity** | 12 features + 3 targets | Exact Match | Exact Match | Passed |

---

## 3. Covariate Balance Table (Standardized Mean Difference)

To verify that the randomized controlled trial (RCT) protocol was properly executed without selection bias, we compute the Standardized Mean Difference (SMD) for every feature:

$$\text{SMD} = \frac{\bar{X}_{\text{treat}} - \bar{X}_{\text{ctrl}}}{\sqrt{\frac{s_{\text{treat}}^2 + s_{\text{ctrl}}^2}{2}}}$$

The conventional threshold for covariate balance is $|\text{SMD}| < 0.10$. Computation is executed as a single distributed Spark aggregation without bringing row-level records to the driver.

### Balance Results (10% Dev Sample, $N = 1,399,999$)

| Feature | Treatment Mean ($\bar{X}_1$) | Control Mean ($\bar{X}_0$) | Treatment Var ($s_1^2$) | Control Var ($s_0^2$) | SMD | Balance Status ($|\text{SMD}| < 0.1$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **f0** | 19.614674 | 19.656744 | 28.905622 | 29.052891 | **-0.0078** | Balanced |
| **f1** | 10.070336 | 10.068067 | 0.011455 | 0.009220 | **+0.0223** | Balanced |
| **f2** | 8.446304 | 8.447678 | 0.089338 | 0.090256 | **-0.0046** | Balanced |
| **f3** | 4.170755 | 4.233904 | 1.823802 | 1.538554 | **-0.0487** | Balanced |
| **f4** | 10.339425 | 10.335177 | 0.118758 | 0.111972 | **+0.0125** | Balanced |
| **f5** | 4.026641 | 4.039800 | 0.190935 | 0.157276 | **-0.0315** | Balanced |
| **f6** | -4.178113 | -3.991857 | 21.144351 | 19.650448 | **-0.0412** | Balanced |
| **f7** | 5.105622 | 5.077515 | 1.470136 | 1.338192 | **+0.0237** | Balanced |
| **f8** | 3.933333 | 3.934674 | 0.003241 | 0.003071 | **-0.0239** | Balanced |
| **f9** | 16.063363 | 15.894540 | 49.883021 | 46.377543 | **+0.0243** | Balanced |
| **f10** | 5.333736 | 5.330968 | 0.028524 | 0.026215 | **+0.0167** | Balanced |
| **f11** | -0.171010 | -0.170873 | 0.000529 | 0.000503 | **-0.0061** | Balanced |

> **Conclusion**: **0 / 12 features unbalanced**. The largest observed discrepancy is feature $f_3$ at $|\text{SMD}| = 0.0487$, well below the $0.10$ threshold. This confirms exchangeability between groups: $(Y(1), Y(0)) \perp T$.

---

## 4. Classical A/B Hypothesis Testing

We conduct two-sided two-proportion $z$-tests on both target outcomes. Group aggregations are performed in Spark, and statistical inference is computed on the driver.

### Raw Counts ($N = 1,399,999$)
- **Treatment ($T=1$)**: $n_1 = 1,189,865$
- **Control ($T=0$)**: $n_0 = 210,134$
- **Allocation Ratio**: $84.99\% / 15.01\%$

### Test Results

| Parameter | Visit Outcome (Dense) | Conversion Outcome (Sparse) |
| :--- | :--- | :--- |
| **Control Successes ($x_0$)** | 8,002 | 440 |
| **Treatment Successes ($x_1$)** | 57,979 | 3,768 |
| **Control Proportion ($p_0$)** | **3.808%** ($0.038080$) | **0.209%** ($0.002094$) |
| **Treatment Proportion ($p_1$)** | **4.873%** ($0.048727$) | **0.317%** ($0.003167$) |
| **Absolute Lift ($\Delta = p_1 - p_0$)** | **+1.065%** ($+0.010647$) | **+0.107%** ($+0.001073$) |
| **Relative Lift ($\frac{\Delta}{p_0}$)** | **+27.96%** | **+51.24%** |
| **95% Confidence Interval (Abs Lift)** | `[+0.009742, +0.011552]` | `[+0.000853, +0.001293]` |
| **Z-Statistic** | **21.2322** | **8.2823** |
| **p-value** | $4.82 \times 10^{-100}$ | $1.21 \times 10^{-16}$ |
| **Statistically Significant ($\alpha=0.05$)** | **True** | **True** |

---

## 5. Retrospective Power Analysis & MDE

Given the actual sample sizes ($n_1 = 1,189,865$, $n_0 = 210,134$, ratio $= 5.66$), we calculate the retrospective Minimum Detectable Effect (MDE) at $\alpha = 0.05$ and $\text{Power} = 0.80$:

$$\text{Effect Size } h = 2 \cdot \left(\arcsin\sqrt{p_1} - \arcsin\sqrt{p_0}\right)$$

| Metric | Visit Outcome | Conversion Outcome |
| :--- | :--- | :--- |
| **Baseline Rate ($p_0$)** | $3.808\%$ | $0.209\%$ |
| **Detectable Cohen's $h$** | $0.006629$ | $0.006629$ |
| **Absolute MDE** | **0.128%** ($0.001279$) | **0.031%** ($0.000314$) |
| **Relative MDE (%)** | **3.36%** | **15.00%** |
| **Observed Relative Lift** | **27.96%** | **51.24%** |
| **Adequately Powered?** | **Yes** (Lift $\gg$ MDE) | **Yes** (Lift $\gg$ MDE) |

### Analytical Discussion: The 85/15 Imbalance and Conversion Sparsity
- **Visit Rate**: With a base rate of $3.81\%$ in control, an effect of just $+0.13\%$ absolute ($3.36\%$ relative) was detectable. The observed lift ($+27.96\%$) is over $8\times$ the detectable threshold, yielding $z = 21.23$ ($p \approx 10^{-100}$).
- **Conversion Rate**: Conversion is extremely sparse ($0.21\%$ control rate). Due to the small number of events ($440$ in control) and 85/15 treatment imbalance, the relative MDE required is $15.00\%$. Because the true campaign relative lift is large ($+51.24\%$), the test is statistically significant ($p = 1.21 \times 10^{-16}$). However, the 95% confidence interval spans $[+0.085\%, +0.129\%]$, demonstrating the higher variance and wider uncertainty characteristic of sparse binary conversion metrics.
