# Phase 2: Statistical Foundation & Covariate Balance Report

## 1. Executive Summary

This document presents the Phase 1 & 2 statistical validation for the **IncrementIQ / Criteo Uplift Modeling** pipeline across both the **10% Stratified Dev Sample** ($N = 1,399,999$) and the **Full Population Dataset** ($N = 13,979,592$).

Key Findings:
1. **Data Ingestion Fidelity**: All 13,979,592 rows loaded with 0 nulls, exact schema matching, and precise replication of published Criteo benchmark rates.
2. **Covariate Balance**: Standardized Mean Difference (SMD) across all 12 pre-treatment features ($f_0$–$f_{11}$) remains $|\text{SMD}| \le 0.0488 \ll 0.10$, confirming unconfounded experimental randomization.
3. **Hypothesis Testing**:
   - **Visit**: $+27.07\%$ relative lift ($+1.034\%$ absolute lift, $z = 65.25$, $p \approx 0.0$).
   - **Conversion**: $+59.45\%$ relative lift ($+0.115\%$ absolute lift, $z = 28.52$, $p = 7.31 \times 10^{-179}$).
4. **Power & MDE**: On the full population, the 85/15 imbalance easily supports detecting an absolute effect of $\ge 0.040\%$ on visits and $\ge 0.0093\%$ on conversions at $80\%$ statistical power.

---

## 2. Ingestion & Population Benchmark Validation (Phase 1)

Ingestion was executed in PySpark with an explicit `StructType` schema (avoiding the two-pass overhead of `inferSchema=True`). The post-treatment leak column `exposure` was pruned immediately at initial load.

| Metric | Documented Benchmark | Observed (Full Parquet) | Observed (10% Dev Sample) | Validation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Rows** | ~13,979,592 | **13,979,592** | **1,399,999** | Passed |
| **Treatment Ratio** | $0.84600$ | **0.85000** | **0.84990** | Passed ($\le 0.02$ tol) |
| **Visit Rate** | $0.04699$ | **0.04699** | **0.04713** | Passed ($\le 0.02$ tol) |
| **Conversion Rate** | $0.00292$ | **0.00292** | **0.00301** | Passed ($\le 0.02$ tol) |
| **Missing Values (Nulls)** | 0 across all cols | **0** | **0** | Passed (0 nulls) |
| **Schema Integrity** | 12 features + 3 targets | Exact Match | Exact Match | Passed |

---

## 3. Covariate Balance Table (Standardized Mean Difference)

Standardized Mean Difference (SMD) measures balance between treatment ($T=1$) and control ($T=0$):

$$\text{SMD} = \frac{\bar{X}_{\text{treat}} - \bar{X}_{\text{ctrl}}}{\sqrt{\frac{s_{\text{treat}}^2 + s_{\text{ctrl}}^2}{2}}}$$

Values with $|\text{SMD}| < 0.10$ are considered balanced. Spark computes this in a single distributed `groupBy(treatment).agg(...)` pass returning only 2 summary rows.

| Feature | Full Treatment Mean ($\bar{X}_1$) | Full Control Mean ($\bar{X}_0$) | Full SMD | Sample SMD | Balance Status ($|\text{SMD}| < 0.1$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **f0** | 19.614755 | 19.651705 | **-0.0069** | -0.0078 | Balanced |
| **f1** | 10.070337 | 10.067935 | **+0.0240** | +0.0223 | Balanced |
| **f2** | 8.446302 | 8.448173 | **-0.0062** | -0.0046 | Balanced |
| **f3** | 4.169412 | 4.232821 | **-0.0488** | -0.0487 | Balanced |
| **f4** | 10.339245 | 10.336526 | **+0.0080** | +0.0125 | Balanced |
| **f5** | 4.026602 | 4.039339 | **-0.0306** | -0.0315 | Balanced |
| **f6** | -4.177242 | -3.997230 | **-0.0404** | -0.0412 | Balanced |
| **f7** | 5.105436 | 5.078490 | **+0.0213** | +0.0237 | Balanced |
| **f8** | 3.933333 | 3.934579 | **-0.0224** | -0.0239 | Balanced |
| **f9** | 16.062256 | 15.897444 | **+0.0240** | +0.0243 | Balanced |
| **f10** | 5.333830 | 5.332098 | **+0.0106** | +0.0167 | Balanced |
| **f11** | -0.170997 | -0.170877 | **-0.0052** | -0.0061 | Balanced |

> **Conclusion**: **0 / 12 features unbalanced**. The largest observed SMD across the entire 14M population is feature $f_3$ at $|\text{SMD}| = 0.0488$. Randomization is verified.

---

## 4. Classical A/B Hypothesis Testing & Lift Analysis

Counts aggregated in Spark; two-sided proportions $z$-tests and non-pooled 95% confidence intervals computed via `statsmodels`.

### Sample Sizes & Counts
- **Full Population ($N = 13,979,592$)**:
  - Treatment: $n_1 = 11,882,655$ ($85.00\%$)
  - Control: $n_0 = 2,096,937$ ($15.00\%$)
- **10% Dev Sample ($N = 1,399,999$)**:
  - Treatment: $n_1 = 1,189,865$ ($84.99\%$)
  - Control: $n_0 = 210,134$ ($15.01\%$)

### Hypothesis Testing Results

| Metric | Visit (Full Dataset) | Visit (10% Sample) | Conversion (Full Dataset) | Conversion (10% Sample) |
| :--- | :--- | :--- | :--- | :--- |
| **Control Successes ($x_0$)** | 80,105 | 8,002 | 4,063 | 440 |
| **Treatment Successes ($x_1$)** | 576,824 | 57,979 | 36,711 | 3,768 |
| **Control Rate ($p_0$)** | **3.820%** | 3.808% | **0.194%** | 0.209% |
| **Treatment Rate ($p_1$)** | **4.854%** | 4.873% | **0.309%** | 0.317% |
| **Absolute Lift ($\Delta$)** | **+1.034%** | +1.065% | **+0.115%** | +0.107% |
| **Relative Lift ($\%$)** | **+27.07%** | +27.96% | **+59.45%** | +51.24% |
| **95% Confidence Interval** | `[+1.006%, +1.063%]` | `[+0.974%, +1.155%]` | `[+0.109%, +0.122%]` | `[+0.085%, +0.129%]` |
| **Z-Statistic** | **65.2474** | 21.2322 | **28.5165** | 8.2823 |
| **p-value** | **0.00** ($< 10^{-300}$) | $4.82 \times 10^{-100}$ | **$7.31 \times 10^{-179}$** | $1.21 \times 10^{-16}$ |
| **Significant at $\alpha=0.05$** | **True** | True | **True** | True |

---

## 5. Retrospective Power Analysis & Minimum Detectable Effect (MDE)

Given the observed sample sizes at $\alpha = 0.05$ and $\text{Power} = 0.80$:

| Metric | Visit (Full) | Visit (Sample) | Conversion (Full) | Conversion (Sample) |
| :--- | :--- | :--- | :--- | :--- |
| **Detectable Cohen's $h$** | $0.002098$ | $0.006629$ | $0.002098$ | $0.006629$ |
| **Absolute MDE** | **0.040%** ($0.000403$) | 0.128% ($0.001279$) | **0.0093%** ($0.000093$) | 0.0314% ($0.000314$) |
| **Relative MDE (%)** | **1.06%** | 3.36% | **4.82%** | 15.00% |
| **Observed Lift** | **27.07%** | 27.96% | **59.45%** | 51.24% |
| **Statistical Power** | **$> 99.9\%$** | $> 99.9\%$ | **$> 99.9\%$** | $> 99.9\%$ |

### Analytical Interpretation:
1. **The 85/15 Asymmetry**: Due to the severe allocation imbalance ($85\%$ treatment vs. $15\%$ control), the effective sample size is driven by the smaller control group ($n_0 \approx 2.10\text{M}$ full, $210\text{k}$ sample).
2. **Dense vs. Sparse Outcome Dynamics**:
   - **Visit** is a dense event ($\sim 3.8\%$). Even on the 10% sample, the experiment had enough statistical power to detect a tiny $3.36\%$ relative lift. The observed $+27.07\%$ lift yields a massive $z = 65.25$.
   - **Conversion** is an extreme needle-in-a-haystack outcome ($\sim 0.19\%$). On the 10% sample, the MDE was $15.00\%$ relative. On the full dataset, the MDE drops to $4.82\%$, allowing precise bounds on the true campaign lift ($[+0.109\%, +0.122\%]$ absolute).
