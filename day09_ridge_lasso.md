# Day 9: Ridge and Lasso Regression - Regularization Techniques

## 📚 Table of Contents
1. [Introduction to Regularization](#introduction)
2. [Ridge Regression (L2)](#ridge-regression)
3. [Lasso Regression (L1)](#lasso-regression)
4. [Elastic Net](#elastic-net)
5. [Implementation and Comparisons](#implementation)
6. [Advanced Topics](#advanced-topics)
7. [Comprehensive Interview Q&A](#interview-qa)

---

## 1. Introduction to Regularization {#introduction}

### Why Regularization?

Linear regression can suffer from:
1. **Overfitting**: Model learns noise in training data
2. **Multicollinearity**: Unstable coefficients when features are correlated
3. **High variance**: Small changes in data cause large changes in coefficients
4. **Poor generalization**: Performs well on training but poorly on test data

### What is Regularization?

Regularization adds a penalty term to the cost function to:
- Shrink coefficients toward zero
- Reduce model complexity
- Improve generalization
- Handle multicollinearity

### Mathematical Framework

Standard linear regression minimizes:
```
J(β) = RSS = Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

Regularized regression minimizes:
```
J(β) = RSS + λ × Penalty(β)
```

Where:
- λ (lambda): Regularization parameter (controls strength)
- Penalty(β): Function of coefficients

### Types of Regularization

1. **L2 (Ridge)**: Penalty = Σβⱼ²
2. **L1 (Lasso)**: Penalty = Σ|βⱼ|
3. **Elastic Net**: Combination of L1 and L2

---

## 2. Ridge Regression (L2 Regularization) {#ridge-regression}

### Mathematical Formulation

Ridge regression minimizes:
```
J(β) = Σᵢ₌₁ⁿ (yᵢ - β₀ - Σⱼ₌₁ᵖ βⱼxᵢⱼ)² + λΣⱼ₌₁ᵖ βⱼ²
```

Note: Typically, intercept β₀ is not penalized.

### Closed-Form Solution

```
β_ridge = (X^T X + λI)^(-1) X^T y
```

Where I is the identity matrix (with 0 in position [0,0] to not penalize intercept).

### Key Properties

1. **Shrinkage**: Coefficients shrink toward zero but never reach exactly zero
2. **Handles multicollinearity**: Stabilizes coefficients when features are correlated
3. **Bias-variance tradeoff**: Increases bias to reduce variance
4. **All features retained**: No automatic feature selection

### Geometric Interpretation

Ridge regression can be viewed as constrained optimization:
```
minimize: RSS
subject to: Σβⱼ² ≤ t
```

The constraint region is a hypersphere (circle in 2D).

### When to Use Ridge

- When you believe most/all features are relevant
- High multicollinearity among features
- When p > n (more features than samples)
- When you want stable coefficient estimates

---

## 3. Lasso Regression (L1 Regularization) {#lasso-regression}

### Mathematical Formulation

Lasso (Least Absolute Shrinkage and Selection Operator) minimizes:
```
J(β) = Σᵢ₌₁ⁿ (yᵢ - β₀ - Σⱼ₌₁ᵖ βⱼxᵢⱼ)² + λΣⱼ₌₁ᵖ |βⱼ|
```

### Key Properties

1. **Sparse solutions**: Can drive coefficients to exactly zero
2. **Automatic feature selection**: Zero coefficients effectively remove features
3. **Non-differentiable**: Requires special optimization algorithms
4. **Instability with correlated features**: Tends to pick one from group

### Geometric Interpretation

Lasso constraint region:
```
minimize: RSS
subject to: Σ|βⱼ| ≤ t
```

The constraint region is a diamond (L1 ball) with corners on axes.

### Optimization Algorithms

Since |β| is not differentiable at 0, special algorithms are needed:
1. **Coordinate Descent**: Update one coefficient at a time
2. **LARS (Least Angle Regression)**: Efficient path algorithm
3. **Proximal Gradient Methods**: Handle non-smooth penalty

### When to Use Lasso

- When you suspect many features are irrelevant
- Need automatic feature selection
- Want interpretable model with few features
- Have high-dimensional data (p >> n)

---

## 4. Elastic Net {#elastic-net}

### Mathematical Formulation

Elastic Net combines L1 and L2 penalties:
```
J(β) = RSS + λ₁Σⱼ₌₁ᵖ |βⱼ| + λ₂Σⱼ₌₁ᵖ βⱼ²
```

Alternative parameterization:
```
J(β) = RSS + λ[(1-α)Σⱼ₌₁ᵖ βⱼ² + αΣⱼ₌₁ᵖ |βⱼ|]
```

Where:
- α ∈ [0,1]: Mixing parameter
- α = 0: Pure Ridge
- α = 1: Pure Lasso

### Advantages

1. **Group selection**: Selects groups of correlated features
2. **Stability**: More stable than Lasso with correlated features
3. **Flexibility**: Can tune between Ridge and Lasso behavior

### When to Use Elastic Net

- Correlated features in groups
- Want feature selection but have multicollinearity
- p >> n with grouped variables
- Ridge too dense, Lasso too unstable

---

## 5. Implementation and Comparisons {#implementation}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# Custom Implementation
class RegularizedRegression:
    """Implementation of Ridge, Lasso, and Elastic Net from scratch"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, 
                 reg_type='ridge'):
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # For elastic net
        self.max_iter = max_iter
        self.tol = tol
        self.reg_type = reg_type
        self.coef_ = None
        self.intercept_ = None
        
    def _soft_thresholding(self, x, lambda_):
        """Soft thresholding operator for Lasso"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    def _coordinate_descent(self, X, y):
        """Coordinate descent for Lasso and Elastic Net"""
        n, p = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(p)
        self.intercept_ = np.mean(y)
        
        # Standardize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_scaled = (X - X_mean) / X_std
        
        # Center y
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        # Precompute X'X and X'y for efficiency
        XtX = X_scaled.T @ X_scaled
        Xty = X_scaled.T @ y_centered
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Update each coefficient
            for j in range(p):
                # Compute residual without j-th feature
                r_j = Xty[j] - XtX[j, :].dot(self.coef_) + XtX[j, j] * self.coef_[j]
                
                if self.reg_type == 'lasso':
                    # Lasso update
                    self.coef_[j] = self._soft_thresholding(r_j, self.alpha) / XtX[j, j]
                    
                elif self.reg_type == 'elastic_net':
                    # Elastic Net update
                    l1_penalty = self.alpha * self.l1_ratio
                    l2_penalty = self.alpha * (1 - self.l1_ratio)
                    self.coef_[j] = self._soft_thresholding(r_j, l1_penalty) / (XtX[j, j] + l2_penalty)
            
            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        # Rescale coefficients
        self.coef_ = self.coef_ / X_std
        self.intercept_ = y_mean - np.sum(X_mean * self.coef_)
    
    def fit(self, X, y):
        """Fit the model"""
        if self.reg_type == 'ridge':
            # Closed-form solution for Ridge
            n, p = X.shape
            
            # Add intercept column
            X_with_intercept = np.column_stack([np.ones(n), X])
            
            # Ridge solution
            lambda_matrix = self.alpha * np.eye(p + 1)
            lambda_matrix[0, 0] = 0  # Don't penalize intercept
            
            # Solve normal equation with regularization
            theta = np.linalg.solve(
                X_with_intercept.T @ X_with_intercept + lambda_matrix,
                X_with_intercept.T @ y
            )
            
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
            
        else:
            # Use coordinate descent for Lasso and Elastic Net
            self._coordinate_descent(X, y)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.coef_ + self.intercept_

# Demonstration 1: Ridge vs Lasso on Correlated Features
print("=== Ridge vs Lasso with Correlated Features ===")

# Create dataset with correlated features
n_samples = 100
n_features = 20
n_informative = 5

# Generate base features
X = np.random.randn(n_samples, n_informative)

# Add correlated features
for i in range(n_informative):
    for j in range(3):  # Add 3 correlated versions of each informative feature
        noise = np.random.randn(n_samples) * 0.1
        X = np.column_stack([X, X[:, i] + noise])

# Trim to desired number of features
X = X[:, :n_features]

# Generate target with sparse coefficients
true_coef = np.zeros(n_features)
true_coef[:n_informative] = np.array([1.5, -2.0, 0.5, 1.0, -1.5])
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    model.fit(X_scaled, y)
    results[name] = {
        'coefficients': model.coef_,
        'n_nonzero': np.sum(np.abs(model.coef_) > 1e-4),
        'train_r2': model.score(X_scaled, y)
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    
    # Plot coefficients
    positions = np.arange(n_features)
    colors = ['red' if i < n_informative else 'blue' for i in range(n_features)]
    
    ax.bar(positions, result['coefficients'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{name}\nNon-zero: {result["n_nonzero"]}, R²: {result["train_r2"]:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Add legend for first plot
    if idx == 0:
        ax.bar([], [], color='red', alpha=0.7, label='True informative')
        ax.bar([], [], color='blue', alpha=0.7, label='Correlated noise')
        ax.legend()

plt.tight_layout()
plt.show()

# Demonstration 2: Regularization Path
print("\n=== Regularization Path ===")

# Range of alpha values
alphas = np.logspace(-4, 2, 100)

# Store coefficients for each alpha
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    ridge_coefs.append(ridge.coef_)
    
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_scaled, y)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ridge path
ax = axes[0]
for i in range(n_features):
    color = 'red' if i < n_informative else 'blue'
    ax.plot(alphas, ridge_coefs[:, i], color=color, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Alpha (λ)')
ax.set_ylabel('Coefficient Value')
ax.set_title('Ridge Regularization Path')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Lasso path
ax = axes[1]
for i in range(n_features):
    color = 'red' if i < n_informative else 'blue'
    ax.plot(alphas, lasso_coefs[:, i], color=color, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Alpha (λ)')
ax.set_ylabel('Coefficient Value')
ax.set_title('Lasso Regularization Path')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add custom legend
ax.plot([], [], color='red', alpha=0.7, label='True informative')
ax.plot([], [], color='blue', alpha=0.7, label='Correlated noise')
ax.legend()

plt.tight_layout()
plt.show()

# Demonstration 3: Cross-Validation for Alpha Selection
print("\n=== Cross-Validation for Optimal Alpha ===")

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Load real dataset
diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define alpha ranges
alphas = np.logspace(-3, 3, 100)

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Ridge CV
ridge_cv = RidgeCV(alphas=alphas, cv=kfold)
ridge_cv.fit(X_train_scaled, y_train)

# Lasso CV
lasso_cv = LassoCV(alphas=alphas, cv=kfold, max_iter=1000)
lasso_cv.fit(X_train_scaled, y_train)

# Elastic Net CV
elastic_cv = ElasticNetCV(alphas=alphas, cv=kfold, l1_ratio=0.5, max_iter=1000)
elastic_cv.fit(X_train_scaled, y_train)

print(f"Optimal Ridge alpha: {ridge_cv.alpha_:.4f}")
print(f"Optimal Lasso alpha: {lasso_cv.alpha_:.4f}")
print(f"Optimal Elastic Net alpha: {elastic_cv.alpha_:.4f}")

# Manual cross-validation to show process
def manual_cv(model_class, alphas, X, y, cv):
    """Perform manual cross-validation"""
    mean_scores = []
    std_scores = []
    
    for alpha in alphas:
        if model_class == Ridge:
            model = model_class(alpha=alpha)
        else:
            model = model_class(alpha=alpha, max_iter=1000)
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean())
        std_scores.append(scores.std())
    
    return np.array(mean_scores), np.array(std_scores)

# Calculate CV scores
ridge_means, ridge_stds = manual_cv(Ridge, alphas, X_train_scaled, y_train, kfold)
lasso_means, lasso_stds = manual_cv(Lasso, alphas, X_train_scaled, y_train, kfold)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ridge CV curve
ax = axes[0]
ax.plot(alphas, ridge_means, 'b-', label='Mean CV Score')
ax.fill_between(alphas, ridge_means - ridge_stds, ridge_means + ridge_stds, 
                alpha=0.2, color='blue')
ax.axvline(ridge_cv.alpha_, color='red', linestyle='--', 
           label=f'Optimal α = {ridge_cv.alpha_:.4f}')
ax.set_xscale('log')
ax.set_xlabel('Alpha (λ)')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Ridge: Cross-Validation Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Lasso CV curve
ax = axes[1]
ax.plot(alphas, lasso_means, 'g-', label='Mean CV Score')
ax.fill_between(alphas, lasso_means - lasso_stds, lasso_means + lasso_stds, 
                alpha=0.2, color='green')
ax.axvline(lasso_cv.alpha_, color='red', linestyle='--', 
           label=f'Optimal α = {lasso_cv.alpha_:.4f}')
ax.set_xscale('log')
ax.set_xlabel('Alpha (λ)')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Lasso: Cross-Validation Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model comparison on test set
models_cv = {
    'OLS': LinearRegression(),
    'Ridge (CV)': ridge_cv,
    'Lasso (CV)': lasso_cv,
    'Elastic Net (CV)': elastic_cv
}

print("\n=== Test Set Performance ===")
for name, model in models_cv.items():
    if name == 'OLS':
        model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    n_features = np.sum(np.abs(model.coef_) > 1e-4)
    
    print(f"{name:15} | MSE: {mse:7.2f} | R²: {r2:.4f} | Features: {n_features}")

# Demonstration 4: Feature Importance and Selection
print("\n=== Feature Importance Analysis ===")

# Get feature names
feature_names = diabetes.feature_names

# Create coefficient comparison
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'OLS': LinearRegression().fit(X_train_scaled, y_train).coef_,
    'Ridge': ridge_cv.coef_,
    'Lasso': lasso_cv.coef_,
    'ElasticNet': elastic_cv.coef_
})

# Sort by absolute OLS coefficient
coef_df['OLS_abs'] = np.abs(coef_df['OLS'])
coef_df = coef_df.sort_values('OLS_abs', ascending=False).drop('OLS_abs', axis=1)

print("\nCoefficient Comparison:")
print(coef_df.round(3))

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for grouped bar chart
x = np.arange(len(feature_names))
width = 0.2

methods = ['OLS', 'Ridge', 'Lasso', 'ElasticNet']
colors = ['blue', 'orange', 'green', 'red']

for i, (method, color) in enumerate(zip(methods, colors)):
    values = coef_df.set_index('Feature').loc[feature_names, method].values
    ax.bar(x + i*width, values, width, label=method, color=color, alpha=0.7)

ax.set_xlabel('Features')
ax.set_ylabel('Coefficient Value')
ax.set_title('Feature Coefficients: Comparison Across Methods')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

# Demonstration 5: Geometric Interpretation
print("\n=== Geometric Interpretation ===")

# Create 2D example for visualization
np.random.seed(42)
n_points = 50
X_2d = np.random.randn(n_points, 2)
true_beta = np.array([1, 2])
y_2d = X_2d @ true_beta + np.random.randn(n_points) * 0.5

# Create grid for contour plots
beta0_range = np.linspace(-1, 3, 100)
beta1_range = np.linspace(0, 4, 100)
B0, B1 = np.meshgrid(beta0_range, beta1_range)

# Calculate RSS for each point
RSS = np.zeros_like(B0)
for i in range(len(beta0_range)):
    for j in range(len(beta1_range)):
        beta = np.array([beta0_range[i], beta1_range[j]])
        residuals = y_2d - X_2d @ beta
        RSS[j, i] = np.sum(residuals**2)

# Fit models
ols_2d = LinearRegression(fit_intercept=False).fit(X_2d, y_2d)
ridge_2d = Ridge(alpha=10, fit_intercept=False).fit(X_2d, y_2d)
lasso_2d = Lasso(alpha=1, fit_intercept=False).fit(X_2d, y_2d)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# OLS (unconstrained)
ax = axes[0]
contour = ax.contour(B0, B1, RSS, levels=20, colors='gray', alpha=0.5)
ax.clabel(contour, inline=True, fontsize=8)
ax.plot(ols_2d.coef_[0], ols_2d.coef_[1], 'ro', markersize=10, label='OLS')
ax.plot(true_beta[0], true_beta[1], 'g*', markersize=15, label='True')
ax.set_xlabel('β₀')
ax.set_ylabel('β₁')
ax.set_title('OLS (Unconstrained)')
ax.legend()
ax.grid(True, alpha=0.3)

# Ridge (L2 constraint)
ax = axes[1]
ax.contour(B0, B1, RSS, levels=20, colors='gray', alpha=0.5)

# Add L2 constraint circle
circle = plt.Circle((0, 0), radius=np.sqrt(ridge_2d.coef_[0]**2 + ridge_2d.coef_[1]**2), 
                   fill=False, color='blue', linewidth=2)
ax.add_patch(circle)

ax.plot(ridge_2d.coef_[0], ridge_2d.coef_[1], 'ro', markersize=10, label='Ridge')
ax.plot(true_beta[0], true_beta[1], 'g*', markersize=15, label='True')
ax.set_xlabel('β₀')
ax.set_ylabel('β₁')
ax.set_title('Ridge (L2 Constraint)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(0, 4)

# Lasso (L1 constraint)
ax = axes[2]
ax.contour(B0, B1, RSS, levels=20, colors='gray', alpha=0.5)

# Add L1 constraint diamond
l1_radius = np.abs(lasso_2d.coef_[0]) + np.abs(lasso_2d.coef_[1])
diamond = plt.Polygon([(l1_radius, 0), (0, l1_radius), 
                      (-l1_radius, 0), (0, -l1_radius)], 
                     fill=False, edgecolor='green', linewidth=2)
ax.add_patch(diamond)

ax.plot(lasso_2d.coef_[0], lasso_2d.coef_[1], 'ro', markersize=10, label='Lasso')
ax.plot(true_beta[0], true_beta[1], 'g*', markersize=15, label='True')
ax.set_xlabel('β₀')
ax.set_ylabel('β₁')
ax.set_title('Lasso (L1 Constraint)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3)
ax.set_ylim(0, 4)

plt.tight_layout()
plt.show()

# Demonstration 6: Stability Analysis
print("\n=== Stability Analysis with Bootstrap ===")

# Bootstrap analysis
n_bootstrap = 100
n_samples_bootstrap = len(X_train)

# Store coefficients from each bootstrap sample
bootstrap_coefs = {
    'OLS': [],
    'Ridge': [],
    'Lasso': [],
    'ElasticNet': []
}

for i in range(n_bootstrap):
    # Create bootstrap sample
    idx = np.random.choice(n_samples_bootstrap, size=n_samples_bootstrap, replace=True)
    X_boot = X_train_scaled[idx]
    y_boot = y_train[idx]
    
    # Fit models
    models_boot = {
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=ridge_cv.alpha_),
        'Lasso': Lasso(alpha=lasso_cv.alpha_),
        'ElasticNet': ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=0.5)
    }
    
    for name, model in models_boot.items():
        model.fit(X_boot, y_boot)
        bootstrap_coefs[name].append(model.coef_)

# Convert to arrays
for name in bootstrap_coefs:
    bootstrap_coefs[name] = np.array(bootstrap_coefs[name])

# Calculate coefficient stability (standard deviation)
stability_df = pd.DataFrame({
    'Feature': feature_names,
    'OLS_std': np.std(bootstrap_coefs['OLS'], axis=0),
    'Ridge_std': np.std(bootstrap_coefs['Ridge'], axis=0),
    'Lasso_std': np.std(bootstrap_coefs['Lasso'], axis=0),
    'ElasticNet_std': np.std(bootstrap_coefs['ElasticNet'], axis=0)
})

print("\nCoefficient Stability (Bootstrap Standard Deviation):")
print(stability_df.round(4))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (name, coefs) in enumerate(bootstrap_coefs.items()):
    ax = axes[idx]
    
    # Box plot of bootstrap coefficients
    ax.boxplot(coefs, labels=feature_names)
    ax.set_xlabel('Features')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{name}: Bootstrap Distribution')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

---

## 6. Advanced Topics {#advanced-topics}

### Adaptive Lasso

Adaptive Lasso uses weighted penalties:
```
J(β) = RSS + λΣⱼ wⱼ|βⱼ|
```

Where weights wⱼ = 1/|β̂ⱼ|^γ from initial estimates.

**Advantages:**
- Oracle property: Correct variable selection asymptotically
- Less biased for large coefficients

### Group Lasso

For grouped features:
```
J(β) = RSS + λΣₖ √(pₖ)||β_group_k||₂
```

**Applications:**
- Categorical variables (one-hot encoded)
- Polynomial features from same variable
- Multi-task learning

### Bayesian Interpretation

**Ridge**: Gaussian prior on coefficients
```
β ~ N(0, τ²I)
```

**Lasso**: Laplace (double exponential) prior
```
β ~ Laplace(0, b)
```

The regularization parameter λ relates to the prior variance.

### Computational Considerations

1. **Standardization**: Always standardize features before regularization
2. **Warm starts**: Use solution from previous λ as initialization
3. **Active set**: Only update non-zero coefficients (Lasso)
4. **Screening rules**: Eliminate features that will be zero

---

## 7. Comprehensive Interview Questions & Answers {#interview-qa}

### Conceptual Understanding

**Q1: Explain the fundamental difference between Ridge and Lasso regression.**

**A:** The key differences are:

1. **Penalty type**:
   - Ridge: L2 penalty (squared coefficients)
   - Lasso: L1 penalty (absolute values)

2. **Solution behavior**:
   - Ridge: Shrinks all coefficients proportionally, never to exactly zero
   - Lasso: Can shrink coefficients to exactly zero, performing feature selection

3. **Geometric interpretation**:
   - Ridge: Circular constraint region (hypersphere)
   - Lasso: Diamond-shaped constraint region with corners on axes

4. **Use cases**:
   - Ridge: When all features are somewhat relevant
   - Lasso: When many features are irrelevant

5. **Computational**:
   - Ridge: Closed-form solution exists
   - Lasso: Requires iterative optimization

**Q2: Why does Lasso perform feature selection while Ridge doesn't?**

**A:** This is due to the geometry of the constraint regions:

- **Lasso**: The L1 constraint creates a diamond shape with corners on the axes. When the elliptical contours of the RSS intersect this diamond, they're likely to touch at a corner, where one or more coefficients are exactly zero.

- **Ridge**: The L2 constraint creates a circle (or hypersphere). The smooth, round shape means RSS contours typically intersect at points where no coefficients are zero.

Mathematically, the L1 penalty has a discontinuous derivative at zero, creating a "threshold" effect that can push coefficients to exactly zero.

**Q3: What is the elastic net and when should you use it?**

**A:** Elastic Net combines L1 and L2 penalties:
```
Penalty = α × L1 + (1-α) × L2
```

**Use Elastic Net when:**
1. **Grouped variables**: You have correlated features in groups (Lasso arbitrarily selects one)
2. **p >> n**: Many more features than samples
3. **Want selection + grouping**: Need feature selection but also want to keep correlated features together
4. **Lasso too unstable**: Lasso's selection varies too much with small data changes

**Advantages:**
- Encourages grouped selection
- More stable than pure Lasso
- Can select more than n features when p > n

### Mathematical Deep Dive

**Q4: Derive the Ridge regression solution analytically.**

**A:** Starting with the cost function:
```
J(β) = ||y - Xβ||² + λ||β||²
```

Expanding:
```
J(β) = (y - Xβ)ᵀ(y - Xβ) + λβᵀβ
J(β) = yᵀy - 2βᵀXᵀy + βᵀXᵀXβ + λβᵀβ
```

Taking derivative with respect to β:
```
∂J/∂β = -2Xᵀy + 2XᵀXβ + 2λβ
      = -2Xᵀy + 2(XᵀX + λI)β
```

Setting to zero:
```
(XᵀX + λI)β = Xᵀy
β = (XᵀX + λI)⁻¹Xᵀy
```

**Q5: Why can't we derive a closed-form solution for Lasso?**

**A:** The Lasso objective function:
```
J(β) = ||y - Xβ||² + λΣ|βⱼ|
```

The absolute value function |β| is not differentiable at β = 0. Specifically:
- For β > 0: ∂|β|/∂β = 1
- For β < 0: ∂|β|/∂β = -1
- At β = 0: Undefined

This non-differentiability means we can't simply set the derivative to zero and solve. Instead, we need subdifferential calculus and iterative optimization methods like coordinate descent.

### Practical Applications

**Q6: How do you choose the regularization parameter λ?**

**A:** Several methods:

1. **Cross-validation** (most common):
   - Split data into k folds
   - Try different λ values
   - Choose λ that minimizes CV error
   - Use nested CV for final evaluation

2. **Information criteria**:
   - AIC: -2log(L) + 2k
   - BIC: -2log(L) + k×log(n)
   - Where k = number of non-zero coefficients

3. **One-standard-error rule**:
   - Choose simplest model within 1 SE of minimum CV error
   - More conservative, prevents overfitting

4. **Analytical approaches**:
   - LARS provides entire regularization path efficiently
   - Can examine how coefficients change with λ

**Q7: Should you standardize features before applying Ridge/Lasso?**

**A:** **Yes, always standardize!** Here's why:

1. **Scale dependency**: Regularization penalizes large coefficients. Without standardization, features with larger scales will be penalized more heavily.

2. **Fair comparison**: Standardization ensures all features are on equal footing for selection/shrinkage.

3. **Interpretation**: With standardized features, coefficient magnitudes directly indicate importance.

**Process:**
```python
# Standardize
X_scaled = (X - X.mean()) / X.std()

# Fit model
model.fit(X_scaled, y)

# Transform back if needed
beta_original = beta_scaled / X.std()
intercept_original = y.mean() - sum(beta_original * X.mean())
```

### Algorithm Understanding

**Q8: Explain coordinate descent for Lasso optimization.**

**A:** Coordinate descent optimizes one parameter at a time while keeping others fixed:

1. **Initialize**: Start with β = 0 or warm start

2. **Iterate**: For each coefficient βⱼ:
   - Compute partial residual: rⱼ = y - Σₖ≠ⱼ xₖβₖ
   - Apply soft thresholding: βⱼ = soft_threshold(xⱼᵀrⱼ, λ) / ||xⱼ||²

3. **Soft thresholding**:
   ```
   soft_threshold(z, λ) = sign(z) × max(|z| - λ, 0)
   ```

4. **Convergence**: Repeat until coefficients stabilize

**Advantages:**
- Simple to implement
- Efficient for sparse solutions
- Can use warm starts

**Q9: What are the assumptions of regularized regression?**

**A:** Regularized regression inherits linear regression assumptions with modifications:

1. **Linearity**: Still assumes linear relationship
2. **Independence**: Observations independent
3. **Homoscedasticity**: Less critical due to regularization
4. **Normality**: Less important for prediction, still matters for inference
5. **No perfect multicollinearity**: Regularization actually helps with multicollinearity!

**Key differences:**
- More robust to multicollinearity
- Biased estimators (trade bias for variance)
- Standard inference procedures don't apply directly

### Comparison Questions

**Q10: Ridge vs Lasso vs Elastic Net - How to choose?**

**A:** Decision framework:

**Choose Ridge when:**
- You believe most/all features are relevant
- Features are highly correlated
- Want stable predictions
- Need all features retained

**Choose Lasso when:**
- Many features likely irrelevant
- Want automatic feature selection
- Need interpretable model
- Have p >> n

**Choose Elastic Net when:**
- Have correlated features in groups
- Want selection but Lasso too unstable
- p >> n with grouped variables
- Need balance between Ridge and Lasso

**Q11: What happens to Ridge and Lasso as λ → 0 and λ → ∞?**

**A:**

**As λ → 0:**
- Both approach OLS solution
- No regularization effect
- May overfit if p > n or multicollinearity

**As λ → ∞:**
- **Ridge**: All coefficients → 0 (but never exactly 0)
- **Lasso**: All coefficients = 0 (sparse solution)
- Model becomes just intercept (mean of y)

This behavior shows Ridge as "soft" shrinkage and Lasso as "hard" thresholding.

### Advanced Topics

**Q12: Explain the "bet on sparsity" principle.**

**A:** Coined by Hastie, Tibshirani, and Friedman:

"Use a procedure that does well in sparse problems, since no procedure does well in dense problems."

**Meaning:**
- If truth is sparse (few relevant features), Lasso performs very well
- If truth is dense (many relevant features), all methods struggle, but Lasso isn't much worse
- Therefore, when in doubt, choose methods that assume sparsity

**Implications:**
- Prefer L1 methods in high dimensions
- Start with Lasso, move to Elastic Net if needed
- Ridge only when confident most features matter

**Q13: How does regularization relate to the bias-variance tradeoff?**

**A:**

**Without regularization (OLS):**
- Low bias (unbiased estimator)
- High variance (sensitive to training data)
- May overfit

**With regularization:**
- Increases bias (shrinks coefficients)
- Decreases variance (more stable)
- Better generalization

**The tradeoff:**
- Small λ: Low bias, high variance
- Large λ: High bias, low variance
- Optimal λ: Balances bias² + variance

Total Error = Bias² + Variance + Irreducible Error

**Q14: What are some variations of Lasso?**

**A:**

1. **Adaptive Lasso**: Weights penalties by initial estimates
2. **Group Lasso**: Selects groups of features together
3. **Fused Lasso**: Encourages sparsity in coefficients and their differences
4. **Graphical Lasso**: Estimates sparse precision matrices
5. **Square-root Lasso**: Scale-invariant version

Each addresses specific limitations of standard Lasso.

### Implementation Considerations

**Q15: What are common pitfalls when implementing Ridge/Lasso?**

**A:**

1. **Not standardizing features**: Leads to unfair penalization
2. **Including intercept in penalty**: Usually shouldn't penalize intercept
3. **Wrong λ scale**: Ridge and Lasso may need different λ ranges
4. **Ignoring convergence**: Lasso may need more iterations
5. **Not using warm starts**: Inefficient for regularization paths
6. **Incorrect CV**: Using same data for λ selection and evaluation
7. **Interpreting coefficients**: Forgetting about standardization when interpreting

### Real-world Scenarios

**Q16: A data scientist says Lasso selected different features on different runs. Why?**

**A:** Several possible reasons:

1. **Correlated features**: Lasso arbitrarily picks one from correlated groups
2. **Borderline λ**: Features near threshold may flip in/out
3. **Optimization**: Different initializations or convergence criteria
4. **Data variations**: Small data changes affect selection
5. **Cross-validation**: Different CV splits give different optimal λ

**Solutions:**
- Use Elastic Net for more stable selection
- Bootstrap to assess selection stability
- Consider adaptive Lasso
- Use larger λ for more aggressive selection

**Q17: How would you explain Ridge regression to a business stakeholder?**

**A:** "Imagine you're fitting a line through data points. Regular linear regression finds the line that's closest to all points. However, if you have many variables or they're related to each other, the model might fit too closely to your training data and work poorly on new data.

Ridge regression prevents this by adding a 'penalty' for large coefficients. It's like telling the model: 'Find a good fit, but keep the coefficients reasonable.' This makes predictions more stable and reliable, especially when variables are correlated or you have limited data.

Think of it as preferring a smooth, gentle curve over a wildly fluctuating one – it might not fit the training data perfectly, but it generalizes better."

### Debugging and Diagnostics

**Q18: How do you diagnose if regularization is helping?**

**A:** Check these indicators:

1. **Learning curves**: Gap between train/test error should decrease
2. **Coefficient stability**: Bootstrap or CV variance should decrease
3. **Test performance**: Should improve compared to OLS
4. **Condition number**: Should decrease for Ridge with multicollinearity
5. **Feature selection**: Lasso should identify relevant features

**Diagnostic code:**
```python
# Compare train/test performance
# Plot regularization paths
# Bootstrap coefficient stability
# Cross-validation curves
```

**Q19: What if Lasso selects no features (all coefficients are zero)?**

**A:** This means λ is too large. Solutions:

1. **Reduce λ**: Try smaller values
2. **Check scale**: Ensure features are standardized
3. **Examine data**: Maybe no linear relationships exist
4. **Try Elastic Net**: May be more forgiving
5. **Consider transformations**: Log, polynomial features
6. **Check implementation**: Verify code is correct

### Extensions and Research

**Q20: What are current research directions in regularization?**

**A:**

1. **Deep learning**: L2 as weight decay, dropout as regularization
2. **Structured sparsity**: Beyond individual features (graphs, trees)
3. **Adaptive methods**: Learning penalty weights
4. **Non-convex penalties**: SCAD, MCP for less biased estimation
5. **Multiple penalties**: Different regularization for different parameter groups
6. **Robust versions**: Handling outliers with regularization
7. **Causal inference**: Regularization for causal effect estimation

---

## Practice Problems

1. Implement coordinate descent for Lasso from scratch
2. Create a function to compute optimal λ using CV
3. Visualize regularization paths for real data
4. Compare feature selection stability across methods
5. Implement adaptive Lasso
6. Build group Lasso for categorical variables

## Key Takeaways

1. **Regularization trades bias for variance** to improve generalization
2. **Ridge shrinks, Lasso selects** - choose based on your needs
3. **Always standardize** features before regularization
4. **Cross-validation** is essential for choosing λ
5. **Elastic Net** combines benefits when you're unsure
6. **Coefficient paths** reveal feature importance
7. **Bootstrap** assesses selection stability
8. **No free lunch** - regularization assumes sparsity/smoothness