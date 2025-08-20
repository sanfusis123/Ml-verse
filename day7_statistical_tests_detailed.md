# 🧪 Day 7: Statistical Tests - Complete Guide with Mathematical Examples

## 📊 **Hypothesis Testing Framework**

### **Core Concepts**

#### **Step-by-Step Process:**
1. **State Hypotheses**: H₀ (null) and H₁ (alternative)
2. **Choose Significance Level**: α (typically 0.05)
3. **Select Test Statistic**: Based on data type and assumptions
4. **Calculate Test Statistic**: From sample data
5. **Find p-value**: Probability of observing data given H₀ is true
6. **Make Decision**: Compare p-value with α

#### **Decision Rules:**
- **p-value ≤ α**: Reject H₀ (statistically significant)
- **p-value > α**: Fail to reject H₀ (not statistically significant)

#### **Error Types:**
- **Type I Error (α)**: Reject true H₀ (False Positive)
- **Type II Error (β)**: Accept false H₀ (False Negative)
- **Power**: 1 - β (probability of correctly rejecting false H₀)

---

## 📈 **T-Tests**

### **1. One-Sample T-Test**

#### **Purpose:** Test if sample mean differs from known population mean

#### **Formula:**
```
t = (x̄ - μ₀) / (s / √n)

where:
x̄ = sample mean
μ₀ = hypothesized population mean
s = sample standard deviation
n = sample size
df = n - 1
```

#### **Example 1: Coffee Shop Wait Time**

**Problem:** A coffee shop claims average wait time is 5 minutes. You sample 12 customers and get these wait times (in minutes):
`[4.2, 5.1, 6.3, 4.8, 5.5, 4.9, 5.8, 4.6, 5.2, 4.7, 5.4, 5.1]`

Test if actual mean differs from claimed 5 minutes (α = 0.05).

**Step 1: State Hypotheses**
- H₀: μ = 5 (wait time is 5 minutes)
- H₁: μ ≠ 5 (wait time is not 5 minutes) [two-tailed test]

**Step 2: Calculate Sample Statistics**
```
n = 12
x̄ = (4.2 + 5.1 + 6.3 + 4.8 + 5.5 + 4.9 + 5.8 + 4.6 + 5.2 + 4.7 + 5.4 + 5.1) / 12
x̄ = 61.6 / 12 = 5.133

s² = Σ(xᵢ - x̄)² / (n-1)
```

**Calculating each deviation:**
```
(4.2 - 5.133)² = (-0.933)² = 0.870
(5.1 - 5.133)² = (-0.033)² = 0.001
(6.3 - 5.133)² = (1.167)² = 1.361
(4.8 - 5.133)² = (-0.333)² = 0.111
(5.5 - 5.133)² = (0.367)² = 0.135
(4.9 - 5.133)² = (-0.233)² = 0.054
(5.8 - 5.133)² = (0.667)² = 0.445
(4.6 - 5.133)² = (-0.533)² = 0.284
(5.2 - 5.133)² = (0.067)² = 0.004
(4.7 - 5.133)² = (-0.433)² = 0.187
(5.4 - 5.133)² = (0.267)² = 0.071
(5.1 - 5.133)² = (-0.033)² = 0.001

Sum = 3.524
s² = 3.524 / 11 = 0.320
s = √0.320 = 0.566
```

**Step 3: Calculate t-statistic**
```
t = (5.133 - 5.0) / (0.566 / √12)
t = 0.133 / (0.566 / 3.464)
t = 0.133 / 0.163
t = 0.816
```

**Step 4: Find Critical Value and p-value**
```
df = n - 1 = 11
For two-tailed test with α = 0.05: t₀.₀₂₅,₁₁ = ±2.201

|t| = 0.816 < 2.201
p-value ≈ 0.432 > 0.05
```

**Step 5: Conclusion**
Fail to reject H₀. There's insufficient evidence that the mean wait time differs from 5 minutes.

### **2. Two-Sample T-Test (Independent Samples)**

#### **Purpose:** Compare means of two independent groups

#### **Formula (Equal Variances):**
```
t = (x̄₁ - x̄₂) / √[s²ₚ(1/n₁ + 1/n₂)]

where:
s²ₚ = [(n₁-1)s₁² + (n₂-1)s₂²] / (n₁ + n₂ - 2)  [pooled variance]
df = n₁ + n₂ - 2
```

#### **Example 2: Treatment Effectiveness**

**Problem:** Compare effectiveness of two weight loss programs. 
- Group A (n₁ = 8): [3.2, 4.1, 2.8, 3.7, 4.0, 3.5, 3.1, 3.9] kg lost
- Group B (n₂ = 10): [2.1, 2.8, 1.9, 2.5, 2.3, 2.7, 2.0, 2.4, 2.6, 2.2] kg lost

Test if Group A is more effective (α = 0.05).

**Step 1: State Hypotheses**
- H₀: μ₁ = μ₂ (no difference in effectiveness)
- H₁: μ₁ > μ₂ (Group A is more effective) [one-tailed test]

**Step 2: Calculate Sample Statistics**

**Group A:**
```
n₁ = 8
x̄₁ = (3.2 + 4.1 + 2.8 + 3.7 + 4.0 + 3.5 + 3.1 + 3.9) / 8 = 28.3 / 8 = 3.538

s₁² = Σ(x₁ᵢ - x̄₁)² / (n₁-1)
Deviations squared: 0.114, 0.316, 0.544, 0.026, 0.213, 0.001, 0.192, 0.131
Sum = 1.537
s₁² = 1.537 / 7 = 0.220
```

**Group B:**
```
n₂ = 10
x̄₂ = (2.1 + 2.8 + 1.9 + 2.5 + 2.3 + 2.7 + 2.0 + 2.4 + 2.6 + 2.2) / 10 = 23.5 / 10 = 2.35

s₂² = Σ(x₂ᵢ - x̄₂)² / (n₂-1)
Deviations squared: 0.063, 0.203, 0.203, 0.023, 0.003, 0.123, 0.123, 0.003, 0.063, 0.023
Sum = 0.830
s₂² = 0.830 / 9 = 0.092
```

**Step 3: Calculate Pooled Variance**
```
s²ₚ = [(8-1)(0.220) + (10-1)(0.092)] / (8 + 10 - 2)
s²ₚ = [7(0.220) + 9(0.092)] / 16
s²ₚ = [1.540 + 0.828] / 16 = 2.368 / 16 = 0.148
```

**Step 4: Calculate t-statistic**
```
t = (3.538 - 2.35) / √[0.148(1/8 + 1/10)]
t = 1.188 / √[0.148(0.125 + 0.10)]
t = 1.188 / √[0.148(0.225)]
t = 1.188 / √0.0333
t = 1.188 / 0.182 = 6.527
```

**Step 5: Find Critical Value**
```
df = 8 + 10 - 2 = 16
For one-tailed test with α = 0.05: t₀.₀₅,₁₆ = 1.746

t = 6.527 > 1.746
p-value < 0.001
```

**Step 6: Conclusion**
Reject H₀. Group A is significantly more effective than Group B.

### **3. Paired T-Test**

#### **Purpose:** Compare two related measurements (before/after, matched pairs)

#### **Formula:**
```
t = d̄ / (sₐ / √n)

where:
d̄ = mean of differences
sₐ = standard deviation of differences
df = n - 1
```

#### **Example 3: Before/After Training**

**Problem:** Test if training improves performance scores:

| Subject | Before | After | Difference (d) |
|---------|--------|-------|----------------|
| 1       | 72     | 78    | 6              |
| 2       | 68     | 73    | 5              |
| 3       | 75     | 80    | 5              |
| 4       | 70     | 75    | 5              |
| 5       | 73     | 79    | 6              |
| 6       | 69     | 71    | 2              |
| 7       | 71     | 77    | 6              |
| 8       | 74     | 78    | 4              |

**Step 1: State Hypotheses**
- H₀: μₐ = 0 (no improvement)
- H₁: μₐ > 0 (training improves scores) [one-tailed test]

**Step 2: Calculate Difference Statistics**
```
Differences: [6, 5, 5, 5, 6, 2, 6, 4]
n = 8
d̄ = (6 + 5 + 5 + 5 + 6 + 2 + 6 + 4) / 8 = 39 / 8 = 4.875

sₐ² = Σ(dᵢ - d̄)² / (n-1)
Deviations from d̄ = 4.875: [1.125, 0.125, 0.125, 0.125, 1.125, -2.875, 1.125, -0.875]
Squared deviations: [1.266, 0.016, 0.016, 0.016, 1.266, 8.266, 1.266, 0.766]
Sum = 12.878
sₐ² = 12.878 / 7 = 1.840
sₐ = √1.840 = 1.356
```

**Step 3: Calculate t-statistic**
```
t = 4.875 / (1.356 / √8)
t = 4.875 / (1.356 / 2.828)
t = 4.875 / 0.479
t = 10.177
```

**Step 4: Find Critical Value**
```
df = 8 - 1 = 7
For one-tailed test with α = 0.05: t₀.₀₅,₇ = 1.895

t = 10.177 > 1.895
p-value < 0.001
```

**Step 5: Conclusion**
Reject H₀. Training significantly improves performance scores.

---

## 📊 **Z-Tests**

### **One-Sample Z-Test**

#### **Purpose:** Test sample mean when population σ is known or n > 30

#### **Formula:**
```
z = (x̄ - μ₀) / (σ / √n)

where σ is known population standard deviation
```

#### **Example 4: Quality Control**

**Problem:** A factory produces bolts with known σ = 0.05 mm. Target diameter is 10.0 mm. A sample of 36 bolts has x̄ = 10.02 mm. Is the process off-target?

**Step 1: State Hypotheses**
- H₀: μ = 10.0 (process is on target)
- H₁: μ ≠ 10.0 (process is off target) [two-tailed test]

**Step 2: Calculate z-statistic**
```
z = (10.02 - 10.0) / (0.05 / √36)
z = 0.02 / (0.05 / 6)
z = 0.02 / 0.00833
z = 2.40
```

**Step 3: Find Critical Value**
```
For two-tailed test with α = 0.05: z₀.₀₂₅ = ±1.96

|z| = 2.40 > 1.96
p-value = 2 × P(Z > 2.40) = 2 × 0.0082 = 0.0164
```

**Step 4: Conclusion**
Reject H₀. The process is significantly off-target.

---

## 🔍 **Chi-Square Tests**

### **1. Chi-Square Goodness of Fit Test**

#### **Purpose:** Test if sample follows expected distribution

#### **Formula:**
```
χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ

where:
Oᵢ = observed frequency
Eᵢ = expected frequency
df = k - 1 - number of estimated parameters
```

#### **Example 5: Die Fairness**

**Problem:** Test if a die is fair. 600 rolls resulted in:

| Face | 1  | 2  | 3  | 4  | 5  | 6  |
|------|----|----|----|----|----|----|
| Observed | 95 | 103| 98 | 107| 92 | 105|

**Step 1: State Hypotheses**
- H₀: Die is fair (equal probabilities)
- H₁: Die is not fair

**Step 2: Calculate Expected Frequencies**
```
For fair die: Expected frequency for each face = 600/6 = 100
```

**Step 3: Calculate Chi-Square Statistic**
```
χ² = (95-100)²/100 + (103-100)²/100 + (98-100)²/100 + (107-100)²/100 + (92-100)²/100 + (105-100)²/100

χ² = 25/100 + 9/100 + 4/100 + 49/100 + 64/100 + 25/100
χ² = 0.25 + 0.09 + 0.04 + 0.49 + 0.64 + 0.25 = 1.76
```

**Step 4: Find Critical Value**
```
df = 6 - 1 = 5
For α = 0.05: χ²₀.₀₅,₅ = 11.07

χ² = 1.76 < 11.07
p-value > 0.05
```

**Step 5: Conclusion**
Fail to reject H₀. The die appears to be fair.

### **2. Chi-Square Test of Independence**

#### **Purpose:** Test if two categorical variables are independent

#### **Formula:**
```
χ² = Σ Σ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ

where:
Eᵢⱼ = (Row total × Column total) / Grand total
df = (rows - 1) × (columns - 1)
```

#### **Example 6: Education and Job Satisfaction**

**Problem:** Test if education level and job satisfaction are independent:

|           | Low Satisfaction | High Satisfaction | Total |
|-----------|------------------|-------------------|-------|
| High School | 20              | 30                | 50    |
| College     | 15              | 45                | 60    |
| Graduate    | 10              | 30                | 40    |
| **Total**   | **45**          | **105**           | **150**|

**Step 1: State Hypotheses**
- H₀: Education and job satisfaction are independent
- H₁: Education and job satisfaction are not independent

**Step 2: Calculate Expected Frequencies**
```
E₁₁ = (50 × 45) / 150 = 15.0
E₁₂ = (50 × 105) / 150 = 35.0
E₂₁ = (60 × 45) / 150 = 18.0
E₂₂ = (60 × 105) / 150 = 42.0
E₃₁ = (40 × 45) / 150 = 12.0
E₃₂ = (40 × 105) / 150 = 28.0
```

**Step 3: Calculate Chi-Square Statistic**
```
χ² = (20-15)²/15 + (30-35)²/35 + (15-18)²/18 + (45-42)²/42 + (10-12)²/12 + (30-28)²/28

χ² = 25/15 + 25/35 + 9/18 + 9/42 + 4/12 + 4/28
χ² = 1.67 + 0.71 + 0.50 + 0.21 + 0.33 + 0.14 = 3.56
```

**Step 4: Find Critical Value**
```
df = (3-1) × (2-1) = 2
For α = 0.05: χ²₀.₀₅,₂ = 5.99

χ² = 3.56 < 5.99
p-value ≈ 0.169 > 0.05
```

**Step 5: Conclusion**
Fail to reject H₀. Education level and job satisfaction appear to be independent.

---

## 📈 **ANOVA (Analysis of Variance)**

### **One-Way ANOVA**

#### **Purpose:** Compare means of three or more groups

#### **Assumptions:**
1. Normal distribution within groups
2. Equal variances (homoscedasticity)
3. Independent observations

#### **Formula:**
```
F = MSB / MSW = (SSB/(k-1)) / (SSW/(N-k))

where:
SST = Σ(xᵢ - x̄)² (Total Sum of Squares)
SSB = Σnⱼ(x̄ⱼ - x̄)² (Between-group Sum of Squares)
SSW = SST - SSB (Within-group Sum of Squares)
```

#### **Example 7: Teaching Methods**

**Problem:** Compare effectiveness of three teaching methods. Test scores:

- **Method A** (n₁ = 5): [85, 87, 90, 88, 85]
- **Method B** (n₂ = 4): [78, 82, 80, 84]
- **Method C** (n₃ = 6): [92, 95, 90, 93, 91, 89]

**Step 1: State Hypotheses**
- H₀: μ₁ = μ₂ = μ₃ (all methods equally effective)
- H₁: At least one method differs

**Step 2: Calculate Group Means**
```
Method A: x̄₁ = (85+87+90+88+85)/5 = 435/5 = 87.0
Method B: x̄₂ = (78+82+80+84)/4 = 324/4 = 81.0
Method C: x̄₃ = (92+95+90+93+91+89)/6 = 550/6 = 91.67

Overall mean: x̄ = (435+324+550)/15 = 1309/15 = 87.27
```

**Step 3: Calculate Sum of Squares**

**SSB (Between-group):**
```
SSB = n₁(x̄₁ - x̄)² + n₂(x̄₂ - x̄)² + n₃(x̄₃ - x̄)²
SSB = 5(87.0 - 87.27)² + 4(81.0 - 87.27)² + 6(91.67 - 87.27)²
SSB = 5(0.0729) + 4(39.31) + 6(19.36)
SSB = 0.36 + 157.24 + 116.16 = 273.76
```

**SSW (Within-group):**
```
Method A: (85-87)² + (87-87)² + (90-87)² + (88-87)² + (85-87)² = 4+0+9+1+4 = 18
Method B: (78-81)² + (82-81)² + (80-81)² + (84-81)² = 9+1+1+9 = 20
Method C: (92-91.67)² + (95-91.67)² + (90-91.67)² + (93-91.67)² + (91-91.67)² + (89-91.67)²
         = 0.11 + 11.09 + 2.79 + 1.77 + 0.45 + 7.13 = 23.34

SSW = 18 + 20 + 23.34 = 61.34
```

**Step 4: Calculate Mean Squares and F-statistic**
```
k = 3 (number of groups)
N = 15 (total sample size)

MSB = SSB / (k-1) = 273.76 / 2 = 136.88
MSW = SSW / (N-k) = 61.34 / 12 = 5.11

F = MSB / MSW = 136.88 / 5.11 = 26.78
```

**Step 5: Find Critical Value**
```
df₁ = k - 1 = 2
df₂ = N - k = 12
For α = 0.05: F₀.₀₅,₂,₁₂ = 3.89

F = 26.78 > 3.89
p-value < 0.001
```

**Step 6: Conclusion**
Reject H₀. At least one teaching method differs significantly from the others.

**ANOVA Table:**
| Source | SS     | df | MS     | F     | p-value |
|--------|--------|----|--------|-------|---------|
| Between| 273.76 | 2  | 136.88 | 26.78 | <0.001  |
| Within | 61.34  | 12 | 5.11   |       |         |
| Total  | 335.10 | 14 |        |       |         |

---

## 🎯 **Test Selection Guide**

### **Decision Tree for Test Selection:**

```
Is the data continuous?
├── YES
│   ├── How many groups?
│   │   ├── 1 group
│   │   │   ├── σ known or n≥30? → Z-test
│   │   │   └── σ unknown and n<30? → t-test
│   │   ├── 2 groups
│   │   │   ├── Independent samples? → Two-sample t-test
│   │   │   └── Paired samples? → Paired t-test
│   │   └── 3+ groups → ANOVA
└── NO (Categorical)
    ├── One variable → Chi-square goodness of fit
    └── Two variables → Chi-square independence
```

### **Key Assumptions Check:**

| Test | Normality | Equal Variance | Independence |
|------|-----------|----------------|--------------|
| t-test | Required | Required (for 2-sample) | Required |
| Z-test | Required | Not critical | Required |
| Chi-square | Not required | Not applicable | Required |
| ANOVA | Required | Required | Required |

### **Effect Size Measures:**

**For t-tests:**
```
Cohen's d = (x̄₁ - x̄₂) / spooled

Small effect: d = 0.2
Medium effect: d = 0.5
Large effect: d = 0.8
```

**For ANOVA:**
```
η² (eta squared) = SSB / SST

Small effect: η² = 0.01
Medium effect: η² = 0.06
Large effect: η² = 0.14
```

**For Chi-square:**
```
Cramér's V = √(χ² / (N × min(rows-1, cols-1)))

Small effect: V = 0.1
Medium effect: V = 0.3
Large effect: V = 0.5
```

---

## 💡 **Common Mistakes to Avoid**

1. **Multiple Comparisons**: When doing multiple tests, adjust α (Bonferroni correction: α/number of tests)

2. **Assumption Violations**: Always check normality (Shapiro-Wilk test) and equal variances (Levene's test)

3. **Sample Size**: Ensure adequate power (typically 80%) for meaningful results

4. **One-tailed vs Two-tailed**: Be clear about directionality of hypothesis

5. **Statistical vs Practical Significance**: Large samples can show statistical significance for trivial differences

## 🔑 **Key Formulas Summary**

| Test | Formula | df |
|------|---------|-----|
| One-sample t | t = (x̄ - μ₀)/(s/√n) | n-1 |
| Two-sample t | t = (x̄₁ - x̄₂)/√[s²ₚ(1/n₁ + 1/n₂)] | n₁+n₂-2 |
| Paired t | t = d̄/(sₐ/√n) | n-1 |
| Z-test | z = (x̄ - μ₀)/(σ/√n) | ∞ |
| Chi-square GoF | χ² = Σ(O-E)²/E | k-1 |
| Chi-square Independence | χ² = ΣΣ(Oᵢⱼ-Eᵢⱼ)²/Eᵢⱼ | (r-1)(c-1) |
| ANOVA | F = MSB/MSW | k-1, N-k |

These comprehensive examples and calculations will help you understand the mathematical foundations of statistical testing for your ML interviews!