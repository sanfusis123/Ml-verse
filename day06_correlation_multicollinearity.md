# Day 6: Correlation and Multicollinearity

## 📚 Topics
- Correlation coefficients (Pearson, Spearman, Kendall)
- Partial and semi-partial correlation
- Multicollinearity detection and remediation
- Variance Inflation Factor (VIF)
- Applications in feature selection

---

## 1. Correlation Fundamentals

### 📖 Core Concepts

#### Pearson Correlation Coefficient
Linear correlation between two continuous variables:
```
ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
```
- Range: [-1, 1]
- Assumes linear relationship
- Sensitive to outliers

#### Spearman's Rank Correlation
Correlation between ranks, captures monotonic relationships:
```
ρ_s = 1 - (6Σd_i²) / (n(n²-1))
```
where d_i = difference in ranks

#### Kendall's Tau
Based on concordant and discordant pairs:
```
τ = (concordant - discordant) / (n(n-1)/2)
```

### 🔢 Mathematical Properties

#### Correlation Properties
1. Symmetry: ρ(X,Y) = ρ(Y,X)
2. Scale invariant: ρ(aX+b, cY+d) = sign(ac)ρ(X,Y)
3. |ρ| ≤ 1 (Cauchy-Schwarz inequality)
4. Independence ⟹ ρ = 0 (but not vice versa)

### 💻 Correlation Implementation Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# 1. Different Types of Correlations
print("=== Types of Correlations ===")

# Generate different relationships
n = 1000
x = np.random.uniform(-3, 3, n)

# Linear relationship
y_linear = 2 * x + np.random.normal(0, 0.5, n)

# Non-linear monotonic
y_monotonic = x**3 + np.random.normal(0, 2, n)

# Non-monotonic
y_quadratic = -x**2 + 5 + np.random.normal(0, 1, n)

# No relationship
y_random = np.random.normal(0, 1, n)

# Calculate correlations
relationships = {
    'Linear': (x, y_linear),
    'Monotonic (Cubic)': (x, y_monotonic),
    'Non-monotonic (Quadratic)': (x, y_quadratic),
    'No Relationship': (x, y_random)
}

correlation_results = pd.DataFrame(columns=['Pearson', 'Spearman', 'Kendall'])

for name, (x_data, y_data) in relationships.items():
    pearson_r, pearson_p = stats.pearsonr(x_data, y_data)
    spearman_r, spearman_p = stats.spearmanr(x_data, y_data)
    kendall_tau, kendall_p = stats.kendalltau(x_data, y_data)
    
    correlation_results.loc[name] = [pearson_r, spearman_r, kendall_tau]

print(correlation_results.round(3))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, (x_data, y_data)) in zip(axes, relationships.items()):
    ax.scatter(x_data, y_data, alpha=0.5, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{name}\nPearson: {correlation_results.loc[name, "Pearson"]:.3f}, '
                 f'Spearman: {correlation_results.loc[name, "Spearman"]:.3f}')
    
    # Add regression line for linear relationship
    if name == 'Linear':
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x_data), p(sorted(x_data)), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# 2. Correlation Matrix and Heatmap
print("\n=== Correlation Matrix Analysis ===")

# Generate correlated features
n_samples = 500
n_features = 8

# Create correlation structure
np.random.seed(42)
base_features = np.random.randn(n_samples, 3)

# Create correlated features
data = pd.DataFrame()
data['X1'] = base_features[:, 0]
data['X2'] = base_features[:, 1]
data['X3'] = base_features[:, 2]
data['X4'] = 0.8 * data['X1'] + 0.2 * np.random.randn(n_samples)  # Highly correlated with X1
data['X5'] = 0.6 * data['X2'] + 0.4 * data['X3'] + 0.2 * np.random.randn(n_samples)
data['X6'] = -0.7 * data['X1'] + 0.3 * np.random.randn(n_samples)  # Negatively correlated
data['X7'] = np.random.randn(n_samples)  # Independent
data['X8'] = data['X3']**2 + np.random.randn(n_samples) * 0.5  # Non-linear relationship

# Calculate correlation matrices
pearson_corr = data.corr(method='pearson')
spearman_corr = data.corr(method='spearman')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Pearson correlation
ax = axes[0]
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Pearson Correlation Matrix')

# Spearman correlation
ax = axes[1]
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Spearman Correlation Matrix')

plt.tight_layout()
plt.show()

# 3. Partial Correlation
print("\n=== Partial Correlation ===")

def partial_correlation(data, x, y, z):
    """
    Calculate partial correlation between x and y controlling for z
    """
    # Standardize variables
    data_std = (data - data.mean()) / data.std()
    
    # Regress x on z
    z_data = data_std[z].values.reshape(-1, 1)
    x_resid = data_std[x] - np.dot(z_data, np.linalg.lstsq(z_data, data_std[x], rcond=None)[0])
    
    # Regress y on z
    y_resid = data_std[y] - np.dot(z_data, np.linalg.lstsq(z_data, data_std[y], rcond=None)[0])
    
    # Correlation of residuals
    return np.corrcoef(x_resid, y_resid)[0, 1]

# Example: X4 is correlated with X1, X5 is correlated with X2 and X3
print("Zero-order correlations:")
print(f"Corr(X4, X5) = {data[['X4', 'X5']].corr().iloc[0, 1]:.3f}")
print(f"Corr(X1, X5) = {data[['X1', 'X5']].corr().iloc[0, 1]:.3f}")

print("\nPartial correlations:")
partial_X4_X5_given_X1 = partial_correlation(data, 'X4', 'X5', ['X1'])
print(f"Partial Corr(X4, X5 | X1) = {partial_X4_X5_given_X1:.3f}")

partial_X1_X5_given_X2 = partial_correlation(data, 'X1', 'X5', ['X2'])
print(f"Partial Corr(X1, X5 | X2) = {partial_X1_X5_given_X2:.3f}")

# 4. Correlation Confidence Intervals
print("\n=== Correlation Confidence Intervals ===")

def correlation_confidence_interval(r, n, confidence=0.95):
    """
    Calculate confidence interval for correlation coefficient using Fisher's z-transformation
    """
    # Fisher's z-transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    
    # Standard error
    se = 1 / np.sqrt(n - 3)
    
    # Critical value
    alpha = 1 - confidence
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Confidence interval in z-space
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # Transform back to r-space
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    return r_lower, r_upper

# Example
sample_sizes = [30, 100, 500, 1000]
true_corr = 0.6

fig, ax = plt.subplots(figsize=(10, 6))

for i, n in enumerate(sample_sizes):
    # Generate correlated data
    mean = [0, 0]
    cov = [[1, true_corr], [true_corr, 1]]
    sample_data = np.random.multivariate_normal(mean, cov, n)
    
    # Calculate correlation
    r = np.corrcoef(sample_data[:, 0], sample_data[:, 1])[0, 1]
    
    # Confidence interval
    ci_lower, ci_upper = correlation_confidence_interval(r, n)
    
    # Plot
    ax.errorbar(i, r, yerr=[[r - ci_lower], [ci_upper - r]], 
                fmt='o', capsize=5, capthick=2, label=f'n={n}')

ax.axhline(y=true_corr, color='red', linestyle='--', label='True correlation')
ax.set_xticks(range(len(sample_sizes)))
ax.set_xticklabels(sample_sizes)
ax.set_xlabel('Sample Size')
ax.set_ylabel('Correlation Coefficient')
ax.set_title('Correlation Estimates with 95% Confidence Intervals')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Robust Correlation
print("\n=== Robust Correlation (with outliers) ===")

# Generate data with outliers
n = 200
x_clean = np.random.normal(0, 1, n-20)
y_clean = 0.8 * x_clean + np.random.normal(0, 0.5, n-20)

# Add outliers
x_outliers = np.random.uniform(3, 5, 20)
y_outliers = np.random.uniform(-5, -3, 20)

x_all = np.concatenate([x_clean, x_outliers])
y_all = np.concatenate([y_clean, y_outliers])

# Calculate correlations
pearson_clean = stats.pearsonr(x_clean, y_clean)[0]
spearman_clean = stats.spearmanr(x_clean, y_clean)[0]

pearson_contaminated = stats.pearsonr(x_all, y_all)[0]
spearman_contaminated = stats.spearmanr(x_all, y_all)[0]

# Percentage bend correlation (more robust)
from scipy.stats import trim_mean

def percentage_bend_correlation(x, y, beta=0.2):
    """Simple robust correlation measure"""
    # Remove top and bottom beta percent
    x_trimmed = stats.trimboth(x, beta)
    y_trimmed = stats.trimboth(y, beta)
    return np.corrcoef(x_trimmed, y_trimmed)[0, 1]

robust_corr = percentage_bend_correlation(x_all, y_all)

print(f"Clean data - Pearson: {pearson_clean:.3f}, Spearman: {spearman_clean:.3f}")
print(f"With outliers - Pearson: {pearson_contaminated:.3f}, Spearman: {spearman_contaminated:.3f}")
print(f"Robust correlation: {robust_corr:.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Clean data
ax = axes[0]
ax.scatter(x_clean, y_clean, alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Clean Data\nPearson: {pearson_clean:.3f}')
z = np.polyfit(x_clean, y_clean, 1)
p = np.poly1d(z)
ax.plot(sorted(x_clean), p(sorted(x_clean)), "r--", alpha=0.8)

# Contaminated data
ax = axes[1]
ax.scatter(x_clean, y_clean, alpha=0.6, label='Clean')
ax.scatter(x_outliers, y_outliers, alpha=0.6, color='red', label='Outliers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Contaminated Data\nPearson: {pearson_contaminated:.3f}, Robust: {robust_corr:.3f}')
ax.legend()

plt.tight_layout()
plt.show()
```

---

## 2. Multicollinearity

### 📖 Core Concepts

#### What is Multicollinearity?
High correlation among predictor variables in regression models, causing:
- Unstable coefficient estimates
- Large standard errors
- Difficulty interpreting individual effects
- Overfitting

#### Types
1. **Perfect Multicollinearity**: Exact linear relationship
2. **High Multicollinearity**: Strong but not perfect correlation

### 🔢 Detection Methods

#### Variance Inflation Factor (VIF)
```
VIF_j = 1 / (1 - R²_j)
```
where R²_j is R-squared from regressing X_j on all other predictors

#### Interpretation
- VIF = 1: No correlation
- VIF < 5: Moderate correlation
- VIF > 10: High multicollinearity
- VIF > 20: Severe multicollinearity

### 💻 Multicollinearity Detection and Remediation Code

```python
# 6. Multicollinearity in Regression
print("\n=== Multicollinearity Detection ===")

# Generate dataset with multicollinearity
n_samples = 200
np.random.seed(42)

# Independent variables
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
X3 = 0.5 * X1 + 0.5 * X2 + 0.1 * np.random.normal(0, 1, n_samples)  # Multicollinear
X4 = 0.9 * X1 + 0.1 * np.random.normal(0, 1, n_samples)  # Highly multicollinear
X5 = np.random.normal(0, 1, n_samples)  # Independent

# Create DataFrame
X = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4,
    'X5': X5
})

# True relationship: Y = 2*X1 + 3*X2 + noise
y = 2 * X1 + 3 * X2 + np.random.normal(0, 0.5, n_samples)

# Calculate VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(df.shape[1])]
    return vif_data

vif_results = calculate_vif(X)
print("Variance Inflation Factors:")
print(vif_results)

# Condition Number
X_matrix = X.values
condition_number = np.linalg.cond(X_matrix)
print(f"\nCondition Number: {condition_number:.2f}")
print("(Values > 30 indicate multicollinearity)")

# 7. Effects of Multicollinearity on Regression
print("\n=== Effects on Regression Coefficients ===")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fit models with different feature sets
models = {
    'All features': X.columns.tolist(),
    'Remove X4 (high VIF)': ['X1', 'X2', 'X3', 'X5'],
    'Remove X3 & X4': ['X1', 'X2', 'X5'],
    'Only independent': ['X1', 'X2', 'X5']
}

results = []

for model_name, features in models.items():
    X_subset = X[features]
    
    # Fit model
    lr = LinearRegression()
    lr.fit(X_subset, y)
    
    # Predictions
    y_pred = lr.predict(X_subset)
    r2 = r2_score(y, y_pred)
    
    # Store results
    coef_dict = {'Model': model_name, 'R²': r2}
    for feat, coef in zip(features, lr.coef_):
        coef_dict[feat] = coef
    results.append(coef_dict)

results_df = pd.DataFrame(results).fillna('-')
print(results_df.round(3))
print("\nTrue coefficients: X1=2.0, X2=3.0")

# 8. Visualizing Multicollinearity
print("\n=== Multicollinearity Visualization ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Correlation heatmap
ax = axes[0, 0]
corr_matrix = X.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, ax=ax, square=True)
ax.set_title('Correlation Heatmap')

# VIF bar plot
ax = axes[0, 1]
bars = ax.bar(vif_results['Variable'], vif_results['VIF'])
ax.axhline(y=5, color='orange', linestyle='--', label='VIF = 5')
ax.axhline(y=10, color='red', linestyle='--', label='VIF = 10')
ax.set_xlabel('Variable')
ax.set_ylabel('VIF')
ax.set_title('Variance Inflation Factors')
ax.legend()

# Color bars based on VIF value
for bar, vif in zip(bars, vif_results['VIF']):
    if vif > 10:
        bar.set_color('red')
    elif vif > 5:
        bar.set_color('orange')
    else:
        bar.set_color('green')

# Scatter plot matrix for highly correlated pairs
ax = axes[1, 0]
ax.scatter(X['X1'], X['X4'], alpha=0.6)
ax.set_xlabel('X1')
ax.set_ylabel('X4')
ax.set_title(f'Highly Correlated Pair\nCorr = {X["X1"].corr(X["X4"]):.3f}')
z = np.polyfit(X['X1'], X['X4'], 1)
p = np.poly1d(z)
ax.plot(sorted(X['X1']), p(sorted(X['X1'])), "r--", alpha=0.8)

# Coefficient stability
ax = axes[1, 1]
n_bootstrap = 100
coef_samples = {feat: [] for feat in ['X1', 'X2', 'X3', 'X4', 'X5']}

for _ in range(n_bootstrap):
    # Bootstrap sample
    idx = np.random.choice(n_samples, n_samples, replace=True)
    X_boot = X.iloc[idx]
    y_boot = y[idx]
    
    # Fit model
    lr = LinearRegression()
    lr.fit(X_boot, y_boot)
    
    # Store coefficients
    for feat, coef in zip(X.columns, lr.coef_):
        coef_samples[feat].append(coef)

# Plot coefficient distributions
positions = range(len(coef_samples))
ax.boxplot(coef_samples.values(), positions=positions, labels=coef_samples.keys())
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Variable')
ax.set_ylabel('Coefficient Value')
ax.set_title('Coefficient Stability (Bootstrap)')

plt.tight_layout()
plt.show()

# 9. Remediation Strategies
print("\n=== Multicollinearity Remediation ===")

# Method 1: Remove highly correlated features
def remove_collinear_features(X, threshold=0.9):
    """Remove features with correlation above threshold"""
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold)]
    
    return X.drop(columns=to_drop), to_drop

X_reduced, dropped = remove_collinear_features(X, threshold=0.9)
print(f"Dropped features (correlation > 0.9): {dropped}")
print(f"\nVIF after removing correlated features:")
print(calculate_vif(X_reduced))

# Method 2: Principal Component Regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum().round(3)}")

# Method 3: Ridge Regression (handles multicollinearity)
from sklearn.linear_model import Ridge, RidgeCV

# Find optimal alpha
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X, y)

print(f"\nOptimal Ridge alpha: {ridge_cv.alpha_:.4f}")

# Compare OLS vs Ridge coefficients
lr_ols = LinearRegression().fit(X, y)
ridge_opt = Ridge(alpha=ridge_cv.alpha_).fit(X, y)

coef_comparison = pd.DataFrame({
    'Feature': X.columns,
    'OLS': lr_ols.coef_,
    'Ridge': ridge_opt.coef_,
    'Difference': lr_ols.coef_ - ridge_opt.coef_
})
print("\nCoefficient Comparison (OLS vs Ridge):")
print(coef_comparison.round(3))

# 10. Feature Clustering to Handle Multicollinearity
print("\n=== Feature Clustering ===")

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Correlation distance matrix
corr = X.corr().abs()
distance_matrix = 1 - corr

# Hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='average')

# Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=X.columns, leaf_rotation=0)
plt.title('Feature Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Distance (1 - |correlation|)')
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Form clusters
clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
cluster_df = pd.DataFrame({'Feature': X.columns, 'Cluster': clusters})
print("\nFeature Clusters:")
print(cluster_df.groupby('Cluster')['Feature'].apply(list))

# 11. Practical Example: Real Estate Price Prediction
print("\n=== Practical Example: House Price Prediction ===")

# Create realistic correlated features
n_houses = 500
np.random.seed(42)

# Base features
house_size = np.random.normal(1500, 500, n_houses)  # Square feet
bedrooms = np.round(house_size / 500 + np.random.normal(0, 0.5, n_houses))
bathrooms = bedrooms * 0.7 + np.random.normal(0, 0.3, n_houses)
lot_size = house_size * 3 + np.random.normal(0, 1000, n_houses)
age = np.random.uniform(0, 50, n_houses)
garage_size = bedrooms * 200 + np.random.normal(0, 100, n_houses)

# Create DataFrame
house_data = pd.DataFrame({
    'HouseSize': house_size,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'LotSize': lot_size,
    'Age': age,
    'GarageSize': garage_size
})

# Price (correlated with features)
price = (
    100 * house_size + 
    10000 * bedrooms + 
    15000 * bathrooms - 
    500 * age + 
    np.random.normal(0, 20000, n_houses)
)

# Check multicollinearity
vif_house = calculate_vif(house_data)
print("House Features VIF:")
print(vif_house)

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(house_data.corr(), annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, square=True)
plt.title('House Features Correlation Matrix')
plt.tight_layout()
plt.show()

# Model comparison
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    house_data, price, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=10),
    'PCA + OLS': 'pca'
}

results = []

for name, model in models.items():
    if name == 'PCA + OLS':
        # PCA transformation
        pca = PCA(n_components=4)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        train_score = model.score(X_train_pca, y_train)
        test_score = model.score(X_test_pca, y_test)
    else:
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
    
    results.append({
        'Model': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'Generalization': test_score - train_score
    })

results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df.round(3))
```

## 🎯 Interview Questions

1. **Q: What's the difference between correlation and causation?**
   - A: Correlation measures association; causation implies one variable directly affects another. Correlation doesn't imply causation.

2. **Q: When would you use Spearman over Pearson correlation?**
   - A: When relationship is monotonic but not linear, data has outliers, or variables are ordinal.

3. **Q: How do you detect multicollinearity?**
   - A: VIF > 10, condition number > 30, high pairwise correlations, unstable coefficients.

4. **Q: What are the consequences of multicollinearity?**
   - A: Unstable coefficients, large standard errors, difficulty interpreting individual effects, overfitting.

5. **Q: How do you handle multicollinearity?**
   - A: Remove correlated features, use PCA, ridge regression, combine correlated features, collect more data.

## 📝 Practice Exercises

1. Implement a function to calculate partial correlation matrix
2. Build a multicollinearity diagnostic tool
3. Compare correlation measures on different data distributions
4. Implement stepwise VIF-based feature selection

## 🔗 Key Takeaways
- Different correlation measures capture different relationships
- Pearson for linear, Spearman for monotonic, Kendall for ordinal
- Multicollinearity affects regression stability and interpretation
- VIF and condition number are key diagnostic tools
- Ridge regression and PCA are effective remediation strategies