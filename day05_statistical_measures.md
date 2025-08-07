# Day 5: Statistical Measures - Mean, Median, Mode, Variance, Covariance

## 📚 Topics
- Measures of central tendency
- Measures of dispersion
- Covariance and covariance matrices
- Practical applications in ML
- Robust statistics

---

## 1. Measures of Central Tendency

### 📖 Core Concepts

#### Mean (Average)
- **Arithmetic Mean**: μ = (1/n) Σx_i
- **Weighted Mean**: μ_w = Σ(w_i × x_i) / Σw_i
- **Geometric Mean**: μ_g = (∏x_i)^(1/n)
- **Harmonic Mean**: μ_h = n / Σ(1/x_i)

#### Median
- Middle value when data is sorted
- Robust to outliers
- For even n: average of two middle values

#### Mode
- Most frequent value(s)
- Can be multiple (multimodal)
- Useful for categorical data

### 🔢 Mathematical Properties

#### Mean Properties
1. Linear transformation: E[aX + b] = aE[X] + b
2. Sum property: E[X + Y] = E[X] + E[Y]
3. Minimizes sum of squared deviations

#### Median Properties
1. Minimizes sum of absolute deviations
2. 50th percentile
3. Robust estimator (breakdown point = 50%)

### 💻 Implementation Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# 1. Measures of Central Tendency
print("=== Measures of Central Tendency ===")

# Generate sample data with outliers
normal_data = np.random.normal(100, 15, 980)
outliers = np.random.uniform(200, 300, 20)
data = np.concatenate([normal_data, outliers])
np.random.shuffle(data)

# Calculate different means
arithmetic_mean = np.mean(data)
median = np.median(data)
mode_result = stats.mode(data, keepdims=True)
mode = mode_result.mode[0] if mode_result.count[0] > 1 else None
geometric_mean = stats.gmean(data[data > 0])  # Only positive values
harmonic_mean = stats.hmean(data[data > 0])
trimmed_mean = stats.trim_mean(data, 0.1)  # 10% trimmed mean

print(f"Arithmetic Mean: {arithmetic_mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {mode:.2f}" if mode else "Mode: No unique mode")
print(f"Geometric Mean: {geometric_mean:.2f}")
print(f"Harmonic Mean: {harmonic_mean:.2f}")
print(f"10% Trimmed Mean: {trimmed_mean:.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram with measures
ax = axes[0]
n, bins, patches = ax.hist(data, bins=50, alpha=0.7, density=True, edgecolor='black')
ax.axvline(arithmetic_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {arithmetic_mean:.1f}')
ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
ax.axvline(trimmed_mean, color='orange', linestyle='--', linewidth=2, label=f'Trimmed Mean: {trimmed_mean:.1f}')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Distribution with Outliers')
ax.legend()

# Box plot
ax = axes[1]
box_data = pd.DataFrame({
    'Original': data,
    'Without Outliers': normal_data
})
box_data.boxplot(ax=ax)
ax.set_ylabel('Value')
ax.set_title('Box Plot Comparison')

plt.tight_layout()
plt.show()

# 2. Weighted Statistics
print("\n=== Weighted Statistics ===")

# Example: Student grades with credit weights
grades = np.array([85, 92, 78, 95, 88])
credits = np.array([3, 4, 3, 2, 3])

# Weighted mean (GPA)
weighted_mean = np.average(grades, weights=credits)
unweighted_mean = np.mean(grades)

print(f"Grades: {grades}")
print(f"Credits: {credits}")
print(f"Weighted Mean (GPA): {weighted_mean:.2f}")
print(f"Unweighted Mean: {unweighted_mean:.2f}")

# 3. Different Types of Means Comparison
print("\n=== Comparison of Different Means ===")

# Generate positive skewed data
skewed_data = np.random.lognormal(3, 1, 1000)

means_dict = {
    'Arithmetic': np.mean(skewed_data),
    'Geometric': stats.gmean(skewed_data),
    'Harmonic': stats.hmean(skewed_data),
    'Median': np.median(skewed_data),
    'Trimmed (10%)': stats.trim_mean(skewed_data, 0.1)
}

print("For log-normal distributed data:")
for name, value in means_dict.items():
    print(f"{name:>15}: {value:>8.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(skewed_data, bins=50, alpha=0.7, density=True, edgecolor='black')
colors = ['red', 'green', 'blue', 'orange', 'purple']
for (name, value), color in zip(means_dict.items(), colors):
    plt.axvline(value, color=color, linestyle='--', linewidth=2, label=f'{name}: {value:.1f}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Different Measures of Central Tendency on Skewed Data')
plt.legend()
plt.xlim(0, 100)
plt.show()

# 4. Mode for Different Data Types
print("\n=== Mode Analysis ===")

# Continuous data (binned)
continuous_data = np.random.normal(50, 10, 1000)
hist, bin_edges = np.histogram(continuous_data, bins=20)
mode_bin_index = np.argmax(hist)
mode_value = (bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2
print(f"Mode of continuous data (binned): {mode_value:.2f}")

# Categorical data
categories = np.random.choice(['A', 'B', 'C', 'D'], size=1000, p=[0.4, 0.3, 0.2, 0.1])
mode_cat = stats.mode(categories, keepdims=True)
print(f"Mode of categorical data: {mode_cat.mode[0]} (count: {mode_cat.count[0]})")

# Multimodal data
multimodal = np.concatenate([
    np.random.normal(20, 3, 400),
    np.random.normal(50, 3, 400),
    np.random.normal(80, 3, 200)
])

plt.figure(figsize=(10, 6))
plt.hist(multimodal, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Multimodal Distribution')
plt.show()
```

---

## 2. Measures of Dispersion

### 📖 Core Concepts

#### Variance
- Population variance: σ² = (1/N) Σ(x_i - μ)²
- Sample variance: s² = (1/(n-1)) Σ(x_i - x̄)²
- Properties: Var(aX + b) = a²Var(X)

#### Standard Deviation
- Square root of variance: σ = √(σ²)
- Same units as original data
- 68-95-99.7 rule for normal distribution

#### Other Measures
- **Range**: max - min
- **Interquartile Range (IQR)**: Q3 - Q1
- **Mean Absolute Deviation (MAD)**: (1/n) Σ|x_i - μ|
- **Coefficient of Variation (CV)**: σ/μ

### 🔢 Mathematical Foundation

#### Variance Decomposition
```
Total Variance = Explained Variance + Unexplained Variance
```

#### Bessel's Correction
Sample variance uses (n-1) instead of n to obtain unbiased estimator:
```
E[s²] = σ²
```

### 💻 Measures of Dispersion Code

```python
# 5. Measures of Dispersion
print("\n=== Measures of Dispersion ===")

# Generate datasets with different dispersions
low_variance = np.random.normal(50, 2, 1000)
high_variance = np.random.normal(50, 10, 1000)
uniform_data = np.random.uniform(30, 70, 1000)

datasets = {
    'Low Variance': low_variance,
    'High Variance': high_variance,
    'Uniform': uniform_data
}

# Calculate dispersion measures
dispersion_stats = pd.DataFrame()

for name, data in datasets.items():
    stats_dict = {
        'Mean': np.mean(data),
        'Variance': np.var(data, ddof=1),  # Sample variance
        'Std Dev': np.std(data, ddof=1),
        'Range': np.ptp(data),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25),
        'MAD': np.mean(np.abs(data - np.mean(data))),
        'CV': np.std(data, ddof=1) / np.mean(data)
    }
    dispersion_stats[name] = stats_dict

print(dispersion_stats.round(3))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograms
ax = axes[0, 0]
for name, data in datasets.items():
    ax.hist(data, bins=30, alpha=0.5, label=name, density=True)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Distribution Comparison')
ax.legend()

# Box plots
ax = axes[0, 1]
pd.DataFrame(datasets).boxplot(ax=ax)
ax.set_ylabel('Value')
ax.set_title('Box Plot Comparison')

# Variance visualization
ax = axes[1, 0]
x = np.linspace(30, 70, 100)
for name, data in datasets.items():
    mean = np.mean(data)
    std = np.std(data)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, label=f'{name} (σ={std:.1f})')
    ax.fill_between(x, 0, y, alpha=0.2)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Normal Approximations')
ax.legend()

# Dispersion metrics bar plot
ax = axes[1, 1]
metrics = ['Variance', 'Std Dev', 'IQR', 'MAD']
x_pos = np.arange(len(metrics))
width = 0.25

for i, (name, data) in enumerate(datasets.items()):
    values = [dispersion_stats.loc[metric, name] for metric in metrics]
    ax.bar(x_pos + i*width, values, width, label=name)

ax.set_xlabel('Metric')
ax.set_ylabel('Value')
ax.set_title('Dispersion Metrics Comparison')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()

# 6. Robust Measures of Dispersion
print("\n=== Robust Dispersion Measures ===")

# Data with outliers
clean_data = np.random.normal(50, 5, 950)
outlier_data = np.concatenate([clean_data, np.random.uniform(100, 200, 50)])

# Calculate robust measures
mad = stats.median_abs_deviation(outlier_data)
iqr = np.percentile(outlier_data, 75) - np.percentile(outlier_data, 25)
std = np.std(outlier_data)

print(f"Standard Deviation: {std:.2f}")
print(f"MAD (Median Absolute Deviation): {mad:.2f}")
print(f"IQR (Interquartile Range): {iqr:.2f}")
print(f"Robust Std Estimate (1.4826 * MAD): {1.4826 * mad:.2f}")

# 7. Standardization and Normalization
print("\n=== Standardization Example ===")

# Original data
original = np.random.normal(100, 15, 1000)

# Standardization (z-score)
z_scores = (original - np.mean(original)) / np.std(original)

# Min-Max normalization
min_max = (original - np.min(original)) / (np.max(original) - np.min(original))

# Robust scaling (using median and IQR)
median = np.median(original)
q1, q3 = np.percentile(original, [25, 75])
robust_scaled = (original - median) / (q3 - q1)

# Compare statistics
comparison = pd.DataFrame({
    'Original': [np.mean(original), np.std(original), np.min(original), np.max(original)],
    'Z-Score': [np.mean(z_scores), np.std(z_scores), np.min(z_scores), np.max(z_scores)],
    'Min-Max': [np.mean(min_max), np.std(min_max), np.min(min_max), np.max(min_max)],
    'Robust': [np.mean(robust_scaled), np.std(robust_scaled), np.min(robust_scaled), np.max(robust_scaled)]
}, index=['Mean', 'Std', 'Min', 'Max'])

print(comparison.round(3))
```

---

## 3. Covariance and Covariance Matrix

### 📖 Core Concepts

#### Covariance
Measures linear relationship between two variables:
```
Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)] = E[XY] - E[X]E[Y]
```

#### Covariance Matrix
For multivariate data X = [X₁, X₂, ..., Xₚ]:
```
Σ_ij = Cov(X_i, X_j)
```

#### Properties
1. Symmetric: Cov(X, Y) = Cov(Y, X)
2. Variance on diagonal: Cov(X, X) = Var(X)
3. Positive semi-definite
4. Scale dependent

### 🔢 Mathematical Foundation

#### Sample Covariance
```
s_XY = (1/(n-1)) Σ(x_i - x̄)(y_i - ȳ)
```

#### Covariance Matrix Properties
- Eigenvalues ≥ 0 (positive semi-definite)
- Determinant = product of eigenvalues
- Trace = sum of variances

### 💻 Covariance Implementation Code

```python
# 8. Covariance Analysis
print("\n=== Covariance Analysis ===")

# Generate correlated data
n_samples = 1000
mean = [0, 0]
cov_matrix = [[1, 0.8], [0.8, 2]]
X, Y = np.random.multivariate_normal(mean, cov_matrix, n_samples).T

# Calculate covariance
cov_manual = np.mean((X - np.mean(X)) * (Y - np.mean(Y)))
cov_numpy = np.cov(X, Y)[0, 1]

print(f"Manual Covariance: {cov_manual:.4f}")
print(f"NumPy Covariance: {cov_numpy:.4f}")
print(f"True Covariance: {cov_matrix[0][1]}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot
ax = axes[0]
ax.scatter(X, Y, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Scatter Plot (Cov={cov_numpy:.2f})')
ax.grid(True, alpha=0.3)

# Positive, zero, and negative covariance
ax = axes[1]
correlations = [0.9, 0, -0.9]
colors = ['red', 'green', 'blue']

for corr, color in zip(correlations, colors):
    cov_temp = [[1, corr], [corr, 1]]
    X_temp, Y_temp = np.random.multivariate_normal([0, 0], cov_temp, 200).T
    ax.scatter(X_temp, Y_temp, alpha=0.5, color=color, label=f'ρ={corr}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Different Covariances')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Covariance vs correlation
ax = axes[2]
# Different scales
X_scaled = X * 10
Y_scaled = Y * 0.1
cov_scaled = np.cov(X_scaled, Y_scaled)[0, 1]
corr_scaled = np.corrcoef(X_scaled, Y_scaled)[0, 1]
corr_original = np.corrcoef(X, Y)[0, 1]

ax.text(0.1, 0.8, f'Original:', transform=ax.transAxes, fontsize=12, weight='bold')
ax.text(0.1, 0.7, f'Cov(X,Y) = {cov_numpy:.3f}', transform=ax.transAxes)
ax.text(0.1, 0.6, f'Corr(X,Y) = {corr_original:.3f}', transform=ax.transAxes)

ax.text(0.1, 0.4, f'After scaling:', transform=ax.transAxes, fontsize=12, weight='bold')
ax.text(0.1, 0.3, f'Cov(10X, 0.1Y) = {cov_scaled:.3f}', transform=ax.transAxes)
ax.text(0.1, 0.2, f'Corr(10X, 0.1Y) = {corr_scaled:.3f}', transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Scale Dependency')

plt.tight_layout()
plt.show()

# 9. Covariance Matrix
print("\n=== Covariance Matrix ===")

# Generate multivariate data
n_features = 4
n_samples = 1000

# Create correlation structure
true_cov = np.array([
    [1.0, 0.8, 0.2, 0.0],
    [0.8, 1.0, 0.3, 0.1],
    [0.2, 0.3, 1.0, 0.7],
    [0.0, 0.1, 0.7, 1.0]
])

# Scale to create covariance
scales = np.array([1, 2, 1.5, 0.5])
true_cov = true_cov * np.outer(scales, scales)

# Generate data
data = np.random.multivariate_normal(np.zeros(n_features), true_cov, n_samples)

# Calculate sample covariance matrix
sample_cov = np.cov(data.T)

print("True Covariance Matrix:")
print(true_cov)
print("\nSample Covariance Matrix:")
print(sample_cov.round(3))

# Visualize covariance matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# True covariance
ax = axes[0]
im = ax.imshow(true_cov, cmap='RdBu', vmin=-2, vmax=2)
ax.set_title('True Covariance')
for i in range(n_features):
    for j in range(n_features):
        ax.text(j, i, f'{true_cov[i,j]:.2f}', ha='center', va='center')
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xticklabels([f'X{i+1}' for i in range(n_features)])
ax.set_yticklabels([f'X{i+1}' for i in range(n_features)])

# Sample covariance
ax = axes[1]
im = ax.imshow(sample_cov, cmap='RdBu', vmin=-2, vmax=2)
ax.set_title('Sample Covariance')
for i in range(n_features):
    for j in range(n_features):
        ax.text(j, i, f'{sample_cov[i,j]:.2f}', ha='center', va='center')
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xticklabels([f'X{i+1}' for i in range(n_features)])
ax.set_yticklabels([f'X{i+1}' for i in range(n_features)])

# Correlation matrix
corr_matrix = np.corrcoef(data.T)
ax = axes[2]
im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
ax.set_title('Correlation Matrix')
for i in range(n_features):
    for j in range(n_features):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center')
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xticklabels([f'X{i+1}' for i in range(n_features)])
ax.set_yticklabels([f'X{i+1}' for i in range(n_features)])

plt.colorbar(im, ax=axes.ravel().tolist())
plt.tight_layout()
plt.show()

# 10. Eigendecomposition of Covariance Matrix
print("\n=== Covariance Matrix Eigenanalysis ===")

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(sample_cov)

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues.round(3))
print("\nVariance explained ratio:", (eigenvalues / np.sum(eigenvalues)).round(3))
print("\nFirst principal component:", eigenvectors[:, 0].round(3))

# Visualize principal components
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
ax = axes[0]
ax.bar(range(1, n_features+1), eigenvalues)
ax.set_xlabel('Component')
ax.set_ylabel('Eigenvalue')
ax.set_title('Scree Plot')
ax.set_xticks(range(1, n_features+1))

# Cumulative variance explained
ax = axes[1]
cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
ax.plot(range(1, n_features+1), cumvar, 'bo-')
ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance Explained')
ax.set_title('Variance Explained')
ax.set_xticks(range(1, n_features+1))
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 11. Robust Covariance Estimation
print("\n=== Robust Covariance Estimation ===")

# Generate data with outliers
n_samples = 1000
n_outliers = 50
n_features = 2

# Clean data
clean_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples - n_outliers)

# Outliers
outliers = np.random.uniform(-10, 10, (n_outliers, n_features))

# Combined data
contaminated_data = np.vstack([clean_data, outliers])

# Estimate covariance
empirical_cov = EmpiricalCovariance().fit(contaminated_data)
robust_cov = MinCovDet().fit(contaminated_data)

print("Empirical Covariance:")
print(empirical_cov.covariance_.round(3))
print("\nRobust Covariance (MCD):")
print(robust_cov.covariance_.round(3))
print("\nTrue Covariance:")
print(np.array([[1, 0.5], [0.5, 1]]))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (title, cov_est) in zip(axes, 
    [('Empirical Covariance', empirical_cov), 
     ('Robust Covariance', robust_cov)]):
    
    # Plot data
    ax.scatter(contaminated_data[:n_samples-n_outliers, 0], 
              contaminated_data[:n_samples-n_outliers, 1], 
              alpha=0.5, label='Clean data')
    ax.scatter(contaminated_data[n_samples-n_outliers:, 0], 
              contaminated_data[n_samples-n_outliers:, 1], 
              alpha=0.5, color='red', label='Outliers')
    
    # Plot confidence ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform circle to ellipse
    center = cov_est.location_
    transform = np.linalg.cholesky(cov_est.covariance_)
    ellipse = center[:, np.newaxis] + 2 * transform @ circle
    
    ax.plot(ellipse[0], ellipse[1], 'g-', linewidth=2, label='95% confidence')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🎯 Interview Questions

1. **Q: When should you use median instead of mean?**
   - A: When data has outliers or is skewed; median is more robust (50% breakdown point).

2. **Q: What's the difference between population and sample variance?**
   - A: Sample variance uses (n-1) divisor (Bessel's correction) to get unbiased estimate of population variance.

3. **Q: How does covariance relate to correlation?**
   - A: Correlation = Covariance / (σ_X × σ_Y). Correlation is standardized covariance, scale-independent.

4. **Q: What does a positive definite covariance matrix mean?**
   - A: All eigenvalues > 0; represents valid covariance structure; ensures variance > 0 for any linear combination.

5. **Q: Why use robust statistics?**
   - A: Traditional measures sensitive to outliers; robust measures (median, MAD, trimmed mean) maintain reliability with contaminated data.

## 📝 Practice Exercises

1. Implement a function to detect outliers using IQR method
2. Calculate and visualize a covariance matrix for a dataset
3. Compare different central tendency measures on various distributions
4. Implement robust standardization using median and MAD

## 🔗 Key Takeaways
- Mean is optimal for symmetric distributions, median for skewed/outliers
- Variance measures spread but is scale-dependent
- Covariance captures linear relationships between variables
- Robust statistics essential for real-world noisy data
- These measures form the foundation for many ML algorithms