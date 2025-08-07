# Day 7: Statistical Tests - T-test, Z-test, Chi-square, ANOVA

## 📚 Topics
- Hypothesis testing fundamentals
- Parametric tests (T-test, Z-test, ANOVA)
- Non-parametric alternatives
- Chi-square tests
- Multiple testing corrections

---

## 1. Hypothesis Testing Fundamentals

### 📖 Core Concepts

#### Components of Hypothesis Testing
1. **Null Hypothesis (H₀)**: Default assumption (no effect/difference)
2. **Alternative Hypothesis (H₁)**: Research hypothesis
3. **Test Statistic**: Measure calculated from data
4. **p-value**: Probability of observing data given H₀ is true
5. **Significance Level (α)**: Threshold for rejection (typically 0.05)

#### Types of Errors
- **Type I Error**: Rejecting true H₀ (False Positive), probability = α
- **Type II Error**: Failing to reject false H₀ (False Negative), probability = β
- **Power**: 1 - β (probability of detecting true effect)

### 🔢 Mathematical Foundation

#### Central Limit Theorem
For large n, sample mean distribution:
```
X̄ ~ N(μ, σ²/n)
```

#### Standard Error
```
SE = σ/√n (population known)
SE = s/√n (sample estimate)
```

### 💻 Implementation Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestPower, FTestAnovaPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# 1. Hypothesis Testing Visualization
print("=== Hypothesis Testing Fundamentals ===")

# Visualize Type I and Type II errors
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Type I Error
ax = axes[0]
x = np.linspace(-4, 4, 1000)
null_dist = stats.norm.pdf(x, 0, 1)
ax.plot(x, null_dist, 'b-', label='H₀ Distribution')
ax.fill_between(x[x > 1.96], null_dist[x > 1.96], alpha=0.3, color='red', label='Type I Error (α)')
ax.fill_between(x[x < -1.96], null_dist[x < -1.96], alpha=0.3, color='red')
ax.axvline(1.96, color='red', linestyle='--', alpha=0.7)
ax.axvline(-1.96, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Test Statistic')
ax.set_ylabel('Probability Density')
ax.set_title('Type I Error (α = 0.05)')
ax.legend()

# Type II Error
ax = axes[1]
null_dist = stats.norm.pdf(x, 0, 1)
alt_dist = stats.norm.pdf(x, 2, 1)
ax.plot(x, null_dist, 'b-', label='H₀ Distribution')
ax.plot(x, alt_dist, 'g-', label='H₁ Distribution')
ax.fill_between(x[x < 1.96], alt_dist[x < 1.96], alpha=0.3, color='orange', label='Type II Error (β)')
ax.fill_between(x[x > 1.96], alt_dist[x > 1.96], alpha=0.3, color='green', label='Power (1-β)')
ax.axvline(1.96, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Test Statistic')
ax.set_ylabel('Probability Density')
ax.set_title('Type II Error and Power')
ax.legend()

plt.tight_layout()
plt.show()

# 2. Power Analysis
print("\n=== Power Analysis ===")

# Calculate power for different effect sizes
effect_sizes = np.linspace(0, 2, 50)
sample_sizes = [10, 30, 50, 100]
alpha = 0.05

power_analysis = TTestPower()

plt.figure(figsize=(10, 6))
for n in sample_sizes:
    power_values = [power_analysis.solve_power(effect_size=es, nobs=n, alpha=alpha) 
                    for es in effect_sizes]
    plt.plot(effect_sizes, power_values, label=f'n={n}')

plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.8')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Statistical Power')
plt.title('Power Analysis for Two-Sample T-Test')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Sample size calculation for desired power
desired_power = 0.8
effect_size = 0.5
required_n = power_analysis.solve_power(effect_size=effect_size, 
                                       power=desired_power, 
                                       alpha=alpha)
print(f"Required sample size for power={desired_power}, effect size={effect_size}: {required_n:.0f}")
```

---

## 2. T-Tests

### 📖 Types of T-Tests

#### One-Sample T-Test
Tests if sample mean differs from hypothesized value:
```
t = (x̄ - μ₀) / (s/√n)
```

#### Two-Sample T-Test
Tests if two sample means differ:
```
t = (x̄₁ - x̄₂) / √(s²p/n₁ + s²p/n₂)
```
where s²p is pooled variance

#### Paired T-Test
Tests difference in paired observations:
```
t = d̄ / (sd/√n)
```

### 🔢 Assumptions
1. Normal distribution (or large sample)
2. Independent observations
3. Equal variances (for standard two-sample)

### 💻 T-Test Implementation Code

```python
# 3. T-Tests Implementation
print("\n=== T-Tests ===")

# Generate sample data
np.random.seed(42)

# One-sample t-test data
population_mean = 100
sample_data = np.random.normal(102, 15, 50)  # Sample with different mean

# Two-sample t-test data
group1 = np.random.normal(100, 15, 60)
group2 = np.random.normal(105, 15, 55)  # Different mean

# Paired t-test data (before/after treatment)
before = np.random.normal(100, 10, 40)
after = before + np.random.normal(5, 5, 40)  # Treatment effect

# Perform tests
results = {}

# One-sample t-test
t_stat_1s, p_val_1s = stats.ttest_1samp(sample_data, population_mean)
results['One-sample'] = {'t-statistic': t_stat_1s, 'p-value': p_val_1s}

# Two-sample t-test (assuming equal variances)
t_stat_2s, p_val_2s = stats.ttest_ind(group1, group2, equal_var=True)
results['Two-sample (equal var)'] = {'t-statistic': t_stat_2s, 'p-value': p_val_2s}

# Welch's t-test (unequal variances)
t_stat_welch, p_val_welch = stats.ttest_ind(group1, group2, equal_var=False)
results['Welch\'s t-test'] = {'t-statistic': t_stat_welch, 'p-value': p_val_welch}

# Paired t-test
t_stat_paired, p_val_paired = stats.ttest_rel(before, after)
results['Paired t-test'] = {'t-statistic': t_stat_paired, 'p-value': p_val_paired}

# Display results
results_df = pd.DataFrame(results).T
print(results_df.round(4))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# One-sample t-test
ax = axes[0, 0]
ax.hist(sample_data, bins=20, alpha=0.7, density=True)
ax.axvline(population_mean, color='red', linestyle='--', label=f'H₀: μ={population_mean}')
ax.axvline(np.mean(sample_data), color='green', linestyle='--', 
           label=f'Sample mean={np.mean(sample_data):.1f}')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title(f'One-Sample T-Test (p={p_val_1s:.4f})')
ax.legend()

# Two-sample t-test
ax = axes[0, 1]
ax.hist(group1, bins=20, alpha=0.5, label='Group 1', density=True)
ax.hist(group2, bins=20, alpha=0.5, label='Group 2', density=True)
ax.axvline(np.mean(group1), color='blue', linestyle='--')
ax.axvline(np.mean(group2), color='orange', linestyle='--')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title(f'Two-Sample T-Test (p={p_val_2s:.4f})')
ax.legend()

# Paired t-test
ax = axes[1, 0]
differences = after - before
ax.hist(differences, bins=20, alpha=0.7, density=True)
ax.axvline(0, color='red', linestyle='--', label='H₀: μd=0')
ax.axvline(np.mean(differences), color='green', linestyle='--', 
           label=f'Mean diff={np.mean(differences):.1f}')
ax.set_xlabel('Difference (After - Before)')
ax.set_ylabel('Density')
ax.set_title(f'Paired T-Test (p={p_val_paired:.4f})')
ax.legend()

# Effect sizes
ax = axes[1, 1]
# Cohen's d calculation
def cohens_d(x, y):
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (np.mean(x) - np.mean(y)) / pooled_std

effect_sizes = {
    'Two-sample': cohens_d(group2, group1),
    'Paired': np.mean(differences) / np.std(differences, ddof=1)
}

bars = ax.bar(effect_sizes.keys(), effect_sizes.values())
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
ax.set_ylabel('Cohen\'s d')
ax.set_title('Effect Sizes')
ax.legend()

plt.tight_layout()
plt.show()

# 4. Checking T-Test Assumptions
print("\n=== T-Test Assumptions ===")

# Normality test
_, p_norm_g1 = stats.shapiro(group1)
_, p_norm_g2 = stats.shapiro(group2)

# Variance equality test (Levene's test)
_, p_var = stats.levene(group1, group2)

print(f"Normality test (Shapiro-Wilk):")
print(f"  Group 1: p = {p_norm_g1:.4f} {'(Normal)' if p_norm_g1 > 0.05 else '(Not normal)'}")
print(f"  Group 2: p = {p_norm_g2:.4f} {'(Normal)' if p_norm_g2 > 0.05 else '(Not normal)'}")
print(f"\nVariance equality (Levene's test):")
print(f"  p = {p_var:.4f} {'(Equal variances)' if p_var > 0.05 else '(Unequal variances)'}")

# QQ plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

stats.probplot(group1, dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot: Group 1')

stats.probplot(group2, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Group 2')

plt.tight_layout()
plt.show()
```

---

## 3. Z-Test

### 📖 Core Concepts

Z-test used when:
- Population variance is known
- Large sample size (n > 30)
- Testing proportions

### 🔢 Test Statistics

#### One-sample Z-test
```
z = (x̄ - μ₀) / (σ/√n)
```

#### Two-proportion Z-test
```
z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))
```

### 💻 Z-Test Implementation Code

```python
# 5. Z-Test Implementation
print("\n=== Z-Tests ===")

# One-sample Z-test (known population variance)
population_mean = 100
population_std = 15
sample_size = 100
sample_mean = 103

z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"One-sample Z-test:")
print(f"  Z-statistic: {z_stat:.4f}")
print(f"  p-value: {p_value_z:.4f}")

# Two-proportion Z-test
# Example: Conversion rates
n1, x1 = 1000, 120  # Group 1: 1000 trials, 120 successes
n2, x2 = 1200, 180  # Group 2: 1200 trials, 180 successes

p1 = x1 / n1
p2 = x2 / n2
p_pooled = (x1 + x2) / (n1 + n2)

z_prop = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
p_value_prop = 2 * (1 - stats.norm.cdf(abs(z_prop)))

print(f"\nTwo-proportion Z-test:")
print(f"  Proportion 1: {p1:.3f}")
print(f"  Proportion 2: {p2:.3f}")
print(f"  Z-statistic: {z_prop:.4f}")
print(f"  p-value: {p_value_prop:.4f}")

# Confidence intervals for proportions
def proportion_ci(p, n, confidence=0.95):
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    margin = z_crit * np.sqrt(p * (1 - p) / n)
    return p - margin, p + margin

ci1 = proportion_ci(p1, n1)
ci2 = proportion_ci(p2, n2)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Z-test distribution
ax = axes[0]
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
ax.plot(x, y, 'b-', label='Standard Normal')
ax.fill_between(x[x > 1.96], y[x > 1.96], alpha=0.3, color='red')
ax.fill_between(x[x < -1.96], y[x < -1.96], alpha=0.3, color='red')
ax.axvline(z_stat, color='green', linestyle='--', 
           label=f'Test statistic: {z_stat:.2f}')
ax.set_xlabel('Z-score')
ax.set_ylabel('Density')
ax.set_title('One-Sample Z-Test')
ax.legend()

# Proportion comparison
ax = axes[1]
groups = ['Group 1', 'Group 2']
proportions = [p1, p2]
errors = [(p1 - ci1[0]), (p2 - ci2[0])]

bars = ax.bar(groups, proportions, yerr=errors, capsize=10, alpha=0.7)
ax.set_ylabel('Proportion')
ax.set_title(f'Two-Proportion Z-Test (p={p_value_prop:.4f})')
ax.set_ylim(0, max(proportions) * 1.3)

# Add CI values
for i, (prop, ci) in enumerate([(p1, ci1), (p2, ci2)]):
    ax.text(i, prop + errors[i] + 0.01, 
            f'{prop:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]', 
            ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

---

## 4. Chi-Square Tests

### 📖 Types of Chi-Square Tests

#### Goodness of Fit
Tests if observed frequencies match expected distribution:
```
χ² = Σ (Observed - Expected)² / Expected
```

#### Test of Independence
Tests if two categorical variables are independent

#### Test of Homogeneity
Tests if different populations have same distribution

### 🔢 Degrees of Freedom
- Goodness of fit: df = k - 1 - p
- Independence: df = (r - 1)(c - 1)

### 💻 Chi-Square Test Implementation Code

```python
# 6. Chi-Square Tests
print("\n=== Chi-Square Tests ===")

# Chi-square goodness of fit
# Example: Die fairness test
observed_freq = np.array([8, 12, 10, 15, 13, 12])  # 70 rolls
expected_freq = np.array([70/6] * 6)  # Fair die expectation

chi2_gof, p_gof = stats.chisquare(observed_freq, expected_freq)
print(f"Chi-square Goodness of Fit:")
print(f"  χ² = {chi2_gof:.4f}")
print(f"  p-value = {p_gof:.4f}")

# Chi-square test of independence
# Example: Treatment response by gender
data = pd.DataFrame({
    'Treatment': ['A']*60 + ['B']*60,
    'Gender': ['M']*30 + ['F']*30 + ['M']*35 + ['F']*25,
    'Response': (['Success']*20 + ['Failure']*10 + ['Success']*15 + ['Failure']*15 +
                 ['Success']*25 + ['Failure']*10 + ['Success']*10 + ['Failure']*15)
})

# Create contingency table
contingency = pd.crosstab(data['Treatment'], data['Response'])
print(f"\nContingency Table:")
print(contingency)

# Perform test
chi2_ind, p_ind, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square Test of Independence:")
print(f"  χ² = {chi2_ind:.4f}")
print(f"  p-value = {p_ind:.4f}")
print(f"  Degrees of freedom = {dof}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Goodness of fit
ax = axes[0, 0]
x = np.arange(1, 7)
width = 0.35
bars1 = ax.bar(x - width/2, observed_freq, width, label='Observed', alpha=0.7)
bars2 = ax.bar(x + width/2, expected_freq, width, label='Expected', alpha=0.7)
ax.set_xlabel('Die Face')
ax.set_ylabel('Frequency')
ax.set_title(f'Goodness of Fit Test (p={p_gof:.4f})')
ax.set_xticks(x)
ax.legend()

# Independence test - observed vs expected
ax = axes[0, 1]
expected_df = pd.DataFrame(expected, 
                          index=contingency.index, 
                          columns=contingency.columns)
x = np.arange(len(contingency.columns))
width = 0.35

for i, treatment in enumerate(contingency.index):
    offset = width * (i - 0.5)
    ax.bar(x + offset, contingency.loc[treatment], 
           width, label=f'{treatment} (Observed)', alpha=0.7)

ax.set_xlabel('Response')
ax.set_ylabel('Count')
ax.set_title(f'Independence Test (p={p_ind:.4f})')
ax.set_xticks(x)
ax.set_xticklabels(contingency.columns)
ax.legend()

# Residuals heatmap
ax = axes[1, 0]
residuals = (contingency - expected) / np.sqrt(expected)
sns.heatmap(residuals, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=ax, cbar_kws={'label': 'Standardized Residuals'})
ax.set_title('Standardized Residuals')

# Chi-square distribution
ax = axes[1, 1]
x = np.linspace(0, 20, 1000)
for df in [1, 3, 5, 10]:
    y = stats.chi2.pdf(x, df)
    ax.plot(x, y, label=f'df = {df}')

ax.axvline(chi2_ind, color='red', linestyle='--', 
           label=f'Test statistic = {chi2_ind:.2f}')
ax.set_xlabel('χ²')
ax.set_ylabel('Density')
ax.set_title('Chi-Square Distribution')
ax.legend()
ax.set_xlim(0, 20)

plt.tight_layout()
plt.show()

# 7. Effect Size for Chi-Square
print("\n=== Effect Sizes for Chi-Square ===")

# Cramér's V
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2_ind / (n * (min(contingency.shape) - 1)))
print(f"Cramér's V: {cramers_v:.3f}")

# Interpretation
if cramers_v < 0.1:
    interpretation = "Negligible"
elif cramers_v < 0.3:
    interpretation = "Small"
elif cramers_v < 0.5:
    interpretation = "Medium"
else:
    interpretation = "Large"
print(f"Effect size interpretation: {interpretation}")
```

---

## 5. ANOVA (Analysis of Variance)

### 📖 Core Concepts

#### One-Way ANOVA
Tests if means differ across multiple groups:
```
F = Between-group variance / Within-group variance
```

#### Assumptions
1. Independence
2. Normality within groups
3. Homogeneity of variances

### 🔢 Mathematical Foundation

#### Sum of Squares
- **SST** (Total): Σ(x_ij - x̄)²
- **SSB** (Between): Σn_i(x̄_i - x̄)²
- **SSW** (Within): Σ Σ(x_ij - x̄_i)²

### 💻 ANOVA Implementation Code

```python
# 8. ANOVA Implementation
print("\n=== ANOVA ===")

# Generate data for multiple groups
np.random.seed(42)
group_a = np.random.normal(100, 10, 30)
group_b = np.random.normal(105, 10, 30)
group_c = np.random.normal(110, 10, 30)
group_d = np.random.normal(103, 10, 30)

# Perform one-way ANOVA
f_stat, p_anova = stats.f_oneway(group_a, group_b, group_c, group_d)
print(f"One-way ANOVA:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_anova:.4f}")

# Create DataFrame for detailed analysis
anova_data = pd.DataFrame({
    'Value': np.concatenate([group_a, group_b, group_c, group_d]),
    'Group': ['A']*30 + ['B']*30 + ['C']*30 + ['D']*30
})

# Manual ANOVA calculation
groups = anova_data.groupby('Group')['Value']
grand_mean = anova_data['Value'].mean()

# Sum of squares
sst = np.sum((anova_data['Value'] - grand_mean) ** 2)
ssb = np.sum([len(group) * (group.mean() - grand_mean) ** 2 
              for name, group in groups])
ssw = np.sum([np.sum((group - group.mean()) ** 2) 
              for name, group in groups])

# Degrees of freedom
df_between = len(groups) - 1
df_within = len(anova_data) - len(groups)
df_total = len(anova_data) - 1

# Mean squares
msb = ssb / df_between
msw = ssw / df_within

# F-statistic
f_manual = msb / msw

# ANOVA table
anova_table = pd.DataFrame({
    'Source': ['Between Groups', 'Within Groups', 'Total'],
    'SS': [ssb, ssw, sst],
    'df': [df_between, df_within, df_total],
    'MS': [msb, msw, np.nan],
    'F': [f_manual, np.nan, np.nan],
    'p-value': [p_anova, np.nan, np.nan]
})

print("\nANOVA Table:")
print(anova_table.round(4))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plots
ax = axes[0, 0]
anova_data.boxplot(column='Value', by='Group', ax=ax)
ax.set_title(f'Group Comparisons (p={p_anova:.4f})')
ax.set_xlabel('Group')
ax.set_ylabel('Value')
plt.sca(ax)
plt.xticks(rotation=0)

# Violin plots
ax = axes[0, 1]
groups_list = [group_a, group_b, group_c, group_d]
parts = ax.violinplot(groups_list, positions=range(1, 5), showmeans=True)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(['A', 'B', 'C', 'D'])
ax.set_xlabel('Group')
ax.set_ylabel('Value')
ax.set_title('Distribution by Group')

# Mean plot with confidence intervals
ax = axes[1, 0]
means = [np.mean(g) for g in groups_list]
sems = [stats.sem(g) for g in groups_list]
ci_95 = [1.96 * sem for sem in sems]

x_pos = range(len(means))
ax.errorbar(x_pos, means, yerr=ci_95, fmt='o-', capsize=5, capthick=2)
ax.axhline(grand_mean, color='red', linestyle='--', alpha=0.7, 
           label=f'Grand mean: {grand_mean:.1f}')
ax.set_xticks(x_pos)
ax.set_xticklabels(['A', 'B', 'C', 'D'])
ax.set_xlabel('Group')
ax.set_ylabel('Mean ± 95% CI')
ax.set_title('Group Means with Confidence Intervals')
ax.legend()

# Post-hoc analysis (Tukey HSD)
ax = axes[1, 1]
tukey = pairwise_tukeyhsd(anova_data['Value'], anova_data['Group'], alpha=0.05)
tukey_df = pd.DataFrame(data=tukey.summary().data[1:], 
                        columns=tukey.summary().data[0])

# Create matrix for visualization
groups_unique = ['A', 'B', 'C', 'D']
p_matrix = np.ones((4, 4))
for _, row in tukey_df.iterrows():
    i = groups_unique.index(row['group1'])
    j = groups_unique.index(row['group2'])
    p_matrix[i, j] = p_matrix[j, i] = float(row['p-adj'])

mask = np.triu(np.ones_like(p_matrix), k=1)
sns.heatmap(p_matrix, mask=mask, annot=True, fmt='.3f', 
            xticklabels=groups_unique, yticklabels=groups_unique,
            cmap='RdYlGn_r', center=0.05, vmin=0, vmax=0.1, ax=ax)
ax.set_title('Tukey HSD Post-hoc Test (p-values)')

plt.tight_layout()
plt.show()

# 9. Assumptions Testing for ANOVA
print("\n=== ANOVA Assumptions ===")

# Normality test for each group
print("Normality Tests (Shapiro-Wilk):")
for name, group in zip(['A', 'B', 'C', 'D'], groups_list):
    _, p_norm = stats.shapiro(group)
    print(f"  Group {name}: p = {p_norm:.4f} "
          f"{'(Normal)' if p_norm > 0.05 else '(Not normal)'}")

# Homogeneity of variances (Levene's test)
_, p_levene = stats.levene(group_a, group_b, group_c, group_d)
print(f"\nLevene's Test for Homogeneity of Variances:")
print(f"  p = {p_levene:.4f} "
      f"{'(Equal variances)' if p_levene > 0.05 else '(Unequal variances)'}")

# 10. Non-parametric Alternative (Kruskal-Wallis)
print("\n=== Non-parametric Alternative ===")

h_stat, p_kruskal = stats.kruskal(group_a, group_b, group_c, group_d)
print(f"Kruskal-Wallis Test:")
print(f"  H-statistic: {h_stat:.4f}")
print(f"  p-value: {p_kruskal:.4f}")

# Effect size for ANOVA (eta-squared)
eta_squared = ssb / sst
print(f"\nEffect Size (η²): {eta_squared:.3f}")
print(f"Interpretation: ", end="")
if eta_squared < 0.01:
    print("Negligible")
elif eta_squared < 0.06:
    print("Small")
elif eta_squared < 0.14:
    print("Medium")
else:
    print("Large")

# 11. Power Analysis for ANOVA
print("\n=== Power Analysis for ANOVA ===")

power_anova = FTestAnovaPower()
effect_size_anova = np.sqrt(eta_squared / (1 - eta_squared))
power = power_anova.solve_power(effect_size=effect_size_anova, 
                                nobs=30*4, alpha=0.05, k_groups=4)
print(f"Statistical Power: {power:.3f}")

# Sample size for desired power
required_n_per_group = power_anova.solve_power(
    effect_size=0.4,  # Medium effect
    power=0.8,
    alpha=0.05,
    k_groups=4
) / 4
print(f"Required n per group for power=0.8: {required_n_per_group:.0f}")
```

## 🎯 Interview Questions

1. **Q: When do you use a t-test vs z-test?**
   - A: T-test when population variance unknown or small sample; z-test when variance known or large sample (n>30).

2. **Q: What's the difference between Type I and Type II errors?**
   - A: Type I = false positive (reject true H₀); Type II = false negative (fail to reject false H₀).

3. **Q: Why use ANOVA instead of multiple t-tests?**
   - A: Multiple t-tests inflate Type I error rate; ANOVA controls overall error rate.

4. **Q: What are the assumptions of parametric tests?**
   - A: Independence, normality, homogeneity of variance (for group comparisons).

5. **Q: When should you use non-parametric tests?**
   - A: When assumptions violated, ordinal data, small samples, or robust analysis needed.

## 📝 Practice Exercises

1. Implement a function to perform power analysis for different tests
2. Create a statistical test selection flowchart
3. Build a function to check all assumptions for a given test
4. Implement bootstrap hypothesis testing

## 🔗 Key Takeaways
- Choose appropriate test based on data type and assumptions
- Always check assumptions before applying parametric tests
- Consider effect size, not just p-values
- Power analysis crucial for experiment design
- Multiple testing requires correction methods