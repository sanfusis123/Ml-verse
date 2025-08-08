# Day 8: Linear Regression and Gradient Descent

## 📚 Table of Contents
1. [Introduction and Concepts](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Gradient Descent Deep Dive](#gradient-descent)
4. [Implementation from Scratch](#implementation)
5. [Advanced Topics](#advanced-topics)
6. [Interview Questions & Answers](#interview-qa)

---

## 1. Introduction and Concepts {#introduction}

### What is Linear Regression?

Linear regression is a fundamental supervised learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data.

### Why is it Important?

1. **Foundation of ML**: Many complex algorithms build upon linear regression concepts
2. **Interpretability**: Coefficients directly show feature importance
3. **Baseline Model**: Often used as a benchmark for more complex models
4. **Real-world Applications**: Price prediction, trend analysis, risk assessment

### Types of Linear Regression

1. **Simple Linear Regression**: One independent variable
   - Example: Predicting house price based on size alone

2. **Multiple Linear Regression**: Multiple independent variables
   - Example: Predicting house price based on size, location, age, etc.

3. **Polynomial Regression**: Non-linear relationships using polynomial features
   - Example: Modeling curved relationships

---

## 2. Mathematical Foundation {#mathematical-foundation}

### The Linear Model

For simple linear regression:
```
y = β₀ + β₁x + ε
```

For multiple linear regression:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

Where:
- y: Dependent variable (target)
- x₁, x₂, ..., xₚ: Independent variables (features)
- β₀: Intercept (bias term)
- β₁, β₂, ..., βₚ: Coefficients (weights)
- ε: Error term (residual)

### Matrix Notation

For n samples and p features:
```
Y = Xβ + ε
```

Where:
- Y: (n × 1) target vector
- X: (n × (p+1)) design matrix (including column of 1s for intercept)
- β: ((p+1) × 1) coefficient vector
- ε: (n × 1) error vector

### Assumptions of Linear Regression

1. **Linearity**: Relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

### Cost Function

**Ordinary Least Squares (OLS)**: Minimize sum of squared residuals
```
J(β) = (1/2n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)² = (1/2n) ||Y - Xβ||²
```

### Closed-Form Solution (Normal Equation)

The optimal coefficients that minimize the cost function:
```
β* = (X^T X)^(-1) X^T Y
```

**Derivation**:
1. Take derivative of J(β) with respect to β
2. Set derivative to zero
3. Solve for β

**When to use**:
- Small to medium datasets (n < 10,000)
- When X^T X is invertible
- When you need exact solution

**Limitations**:
- Computational complexity: O(p³) for matrix inversion
- Memory intensive for large p
- Singular matrix issues

---

## 3. Gradient Descent Deep Dive {#gradient-descent}

### What is Gradient Descent?

An iterative optimization algorithm that finds the minimum of a function by moving in the direction of steepest descent (negative gradient).

### Types of Gradient Descent

1. **Batch Gradient Descent**
   - Uses entire dataset for each update
   - Stable convergence
   - Slow for large datasets

2. **Stochastic Gradient Descent (SGD)**
   - Uses one sample at a time
   - Faster but noisy updates
   - Can escape local minima

3. **Mini-batch Gradient Descent**
   - Uses small batches (typically 32-512 samples)
   - Balance between batch and SGD
   - Most commonly used in practice

### Update Rule

```
β := β - α × ∇J(β)
```

Where:
- α: Learning rate
- ∇J(β): Gradient of cost function

For linear regression:
```
∂J/∂βⱼ = -(1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)xᵢⱼ
```

### Learning Rate Selection

- **Too small**: Slow convergence
- **Too large**: May overshoot minimum or diverge
- **Adaptive methods**: Adam, RMSprop, AdaGrad

### Convergence Criteria

1. Maximum iterations reached
2. Cost change below threshold: |J(t) - J(t-1)| < ε
3. Gradient magnitude below threshold: ||∇J|| < ε

---

## 4. Implementation from Scratch {#implementation}

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

class LinearRegression:
    """
    Linear Regression implementation with multiple optimization methods
    """
    def __init__(self, method='gradient_descent', learning_rate=0.01, 
                 n_iterations=1000, regularization=None, lambda_reg=0.01):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.costs_history = []
        
    def _add_intercept(self, X):
        """Add intercept term to feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _cost_function(self, X, y, theta):
        """Calculate cost (MSE)"""
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        
        # Add regularization term if specified
        if self.regularization == 'l2':
            cost += (self.lambda_reg/(2*m)) * np.sum(theta[1:]**2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg/m) * np.sum(np.abs(theta[1:]))
            
        return cost
    
    def _gradient(self, X, y, theta):
        """Calculate gradient"""
        m = len(y)
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        
        # Add regularization gradient if specified
        if self.regularization == 'l2':
            reg_term = (self.lambda_reg/m) * theta
            reg_term[0] = 0  # Don't regularize intercept
            gradient += reg_term
        elif self.regularization == 'l1':
            reg_term = (self.lambda_reg/m) * np.sign(theta)
            reg_term[0] = 0
            gradient += reg_term
            
        return gradient
    
    def _gradient_descent(self, X, y):
        """Batch gradient descent"""
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.n_iterations):
            gradient = self._gradient(X, y, self.theta)
            self.theta -= self.learning_rate * gradient
            
            # Store cost history
            cost = self._cost_function(X, y, self.theta)
            self.costs_history.append(cost)
            
            # Early stopping if converged
            if i > 0 and abs(self.costs_history[-1] - self.costs_history[-2]) < 1e-7:
                print(f"Converged at iteration {i}")
                break
    
    def _stochastic_gradient_descent(self, X, y):
        """Stochastic gradient descent"""
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Update weights for each sample
            for j in range(m):
                xi = X_shuffled[j:j+1]
                yi = y_shuffled[j:j+1]
                gradient = self._gradient(xi, yi, self.theta)
                self.theta -= self.learning_rate * gradient
            
            # Store cost for entire dataset
            cost = self._cost_function(X, y, self.theta)
            self.costs_history.append(cost)
    
    def _normal_equation(self, X, y):
        """Closed-form solution"""
        # Normal equation: θ = (X^T X)^(-1) X^T y
        
        if self.regularization == 'l2':
            # Ridge regression: θ = (X^T X + λI)^(-1) X^T y
            lambda_identity = self.lambda_reg * np.eye(X.shape[1])
            lambda_identity[0, 0] = 0  # Don't regularize intercept
            self.theta = np.linalg.inv(X.T.dot(X) + lambda_identity).dot(X.T).dot(y)
        else:
            self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    def fit(self, X, y):
        """Train the model"""
        # Add intercept
        X = self._add_intercept(X)
        
        # Choose optimization method
        if self.method == 'gradient_descent':
            self._gradient_descent(X, y)
        elif self.method == 'sgd':
            self._stochastic_gradient_descent(X, y)
        elif self.method == 'normal_equation':
            self._normal_equation(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store coefficients separately
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        
    def predict(self, X):
        """Make predictions"""
        X = self._add_intercept(X)
        return X.dot(self.theta)
    
    def score(self, X, y):
        """Calculate R-squared score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

# Demonstration 1: Simple Linear Regression
print("=== Simple Linear Regression Demo ===")

# Generate synthetic data
X_simple = 2 * np.random.rand(100, 1)
y_simple = 4 + 3 * X_simple[:, 0] + np.random.randn(100)

# Train model
lr_simple = LinearRegression(method='gradient_descent', learning_rate=0.1, n_iterations=1000)
lr_simple.fit(X_simple, y_simple)

print(f"True coefficients: intercept=4, slope=3")
print(f"Learned coefficients: intercept={lr_simple.intercept_:.3f}, slope={lr_simple.coef_[0]:.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot with regression line
ax = axes[0]
ax.scatter(X_simple, y_simple, alpha=0.5)
X_plot = np.linspace(0, 2, 100).reshape(-1, 1)
y_plot = lr_simple.predict(X_plot)
ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='Fitted line')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Simple Linear Regression')
ax.legend()

# Cost function over iterations
ax = axes[1]
ax.plot(lr_simple.costs_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Cost Function During Training')
ax.set_yscale('log')

plt.tight_layout()
plt.show()

# Demonstration 2: Multiple Linear Regression
print("\n=== Multiple Linear Regression Demo ===")

# Generate data with multiple features
X_multi, y_multi = make_regression(n_samples=1000, n_features=5, n_informative=3, 
                                   noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different methods
methods = ['normal_equation', 'gradient_descent', 'sgd']
results = {}

for method in methods:
    lr = LinearRegression(method=method, learning_rate=0.01, n_iterations=500)
    lr.fit(X_train_scaled, y_train)
    
    train_score = lr.score(X_train_scaled, y_train)
    test_score = lr.score(X_test_scaled, y_test)
    
    results[method] = {
        'train_r2': train_score,
        'test_r2': test_score,
        'coefficients': lr.coef_
    }
    
    print(f"\n{method.upper()}:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")

# Demonstration 3: Gradient Descent Visualization
print("\n=== Gradient Descent Visualization ===")

# Create a simple 2D cost function for visualization
def create_cost_surface(X, y, theta0_range, theta1_range):
    """Create cost surface for visualization"""
    theta0_vals = np.linspace(theta0_range[0], theta0_range[1], 100)
    theta1_vals = np.linspace(theta1_range[0], theta1_range[1], 100)
    
    cost_surface = np.zeros((len(theta0_vals), len(theta1_vals)))
    
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta = np.array([theta0, theta1])
            predictions = X.dot(theta)
            cost_surface[i, j] = (1/(2*len(y))) * np.sum((predictions - y)**2)
    
    return theta0_vals, theta1_vals, cost_surface

# Use simple linear regression data
X_vis = lr_simple._add_intercept(X_simple)
theta0_range = [2, 6]
theta1_range = [1, 5]

theta0_vals, theta1_vals, cost_surface = create_cost_surface(X_vis, y_simple, theta0_range, theta1_range)

# Track gradient descent path
lr_vis = LinearRegression(method='gradient_descent', learning_rate=0.1, n_iterations=50)
lr_vis.theta = np.array([2.0, 1.0])  # Start from a specific point
theta_history = [lr_vis.theta.copy()]

# Manual gradient descent to track path
for i in range(50):
    gradient = lr_vis._gradient(X_vis, y_simple, lr_vis.theta)
    lr_vis.theta -= lr_vis.learning_rate * gradient
    theta_history.append(lr_vis.theta.copy())

theta_history = np.array(theta_history)

# Visualization
fig = plt.figure(figsize=(12, 5))

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')
THETA0, THETA1 = np.meshgrid(theta0_vals, theta1_vals)
ax1.plot_surface(THETA0, THETA1, cost_surface.T, cmap='viridis', alpha=0.6)
ax1.plot(theta_history[:, 0], theta_history[:, 1], 
         [create_cost_surface(X_vis, y_simple, [t[0], t[0]], [t[1], t[1]])[2][0, 0] 
          for t in theta_history], 
         'r.-', markersize=8, linewidth=2)
ax1.set_xlabel('θ₀ (intercept)')
ax1.set_ylabel('θ₁ (slope)')
ax1.set_zlabel('Cost')
ax1.set_title('Gradient Descent on Cost Surface')

# Contour plot
ax2 = fig.add_subplot(122)
contours = ax2.contour(THETA0, THETA1, cost_surface.T, levels=20)
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(theta_history[:, 0], theta_history[:, 1], 'r.-', markersize=8, linewidth=2)
ax2.set_xlabel('θ₀ (intercept)')
ax2.set_ylabel('θ₁ (slope)')
ax2.set_title('Gradient Descent Path (Contour View)')

plt.tight_layout()
plt.show()

# Demonstration 4: Effect of Learning Rate
print("\n=== Effect of Learning Rate ===")

learning_rates = [0.001, 0.01, 0.1, 0.5]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    model = LinearRegression(method='gradient_descent', learning_rate=lr, n_iterations=100)
    model.fit(X_simple, y_simple)
    
    ax = axes[idx]
    ax.plot(model.costs_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(f'Learning Rate = {lr}')
    ax.set_yscale('log')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Demonstration 5: Regularization Effects
print("\n=== Regularization Effects ===")

# Create dataset with multicollinearity
n_samples = 100
n_features = 20
X_reg = np.random.randn(n_samples, n_features)
# Create correlated features
for i in range(5, 10):
    X_reg[:, i] = X_reg[:, i-5] + 0.5 * np.random.randn(n_samples)

# True coefficients (sparse)
true_coef = np.zeros(n_features)
true_coef[:5] = [1.5, -2.0, 0.5, 1.0, -0.5]
y_reg = X_reg.dot(true_coef) + 0.1 * np.random.randn(n_samples)

# Train models with different regularization
lambda_values = [0, 0.01, 0.1, 1.0, 10.0]
regularization_types = [None, 'l2']

results_reg = {}
for reg_type in regularization_types:
    results_reg[reg_type] = []
    for lambda_val in lambda_values:
        lr = LinearRegression(method='normal_equation', regularization=reg_type, lambda_reg=lambda_val)
        lr.fit(X_reg, y_reg)
        results_reg[reg_type].append(lr.coef_)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, reg_type in enumerate(regularization_types):
    ax = axes[idx]
    coefficients = np.array(results_reg[reg_type])
    
    for i in range(n_features):
        ax.plot(lambda_values, coefficients[:, i], marker='o', label=f'β_{i+1}' if i < 5 else None)
    
    ax.set_xlabel('Regularization Parameter (λ)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{"No Regularization" if reg_type is None else "Ridge Regularization"}')
    ax.set_xscale('log')
    ax.grid(True)
    if idx == 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

---

## 5. Advanced Topics {#advanced-topics}

### Feature Engineering for Linear Regression

1. **Polynomial Features**
   - Transform x → [1, x, x², x³, ...]
   - Captures non-linear relationships
   - Risk of overfitting with high degrees

2. **Interaction Terms**
   - Include products of features: x₁x₂
   - Captures feature dependencies
   - Domain knowledge helps identify relevant interactions

3. **Log Transformations**
   - Log(y) for exponential relationships
   - Log(x) for logarithmic relationships
   - Useful for right-skewed distributions

### Handling Violations of Assumptions

1. **Non-linearity**
   - Add polynomial features
   - Use splines or piecewise regression
   - Consider non-linear models

2. **Heteroscedasticity**
   - Weighted least squares
   - Transform variables
   - Use robust standard errors

3. **Multicollinearity**
   - Remove correlated features
   - Use regularization
   - Principal Component Regression

4. **Non-normal residuals**
   - Transform target variable
   - Use robust regression methods
   - Consider GLM for appropriate distribution

### Robust Regression Methods

1. **Huber Regression**
   - Less sensitive to outliers
   - Combines squared loss and absolute loss

2. **RANSAC (Random Sample Consensus)**
   - Iteratively fits on inlier subsets
   - Excellent for data with many outliers

3. **Theil-Sen Estimator**
   - Median of slopes between all point pairs
   - Robust to up to 29% outliers

### Model Diagnostics

```python
# Residual Analysis Implementation
def diagnose_linear_regression(X, y, model):
    """Comprehensive diagnostic plots for linear regression"""
    predictions = model.predict(X)
    residuals = y - predictions
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(predictions, residuals, alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    
    # 2. Q-Q Plot
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    
    # 3. Scale-Location
    ax = axes[1, 0]
    standardized_residuals = residuals / np.std(residuals)
    ax.scatter(predictions, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.set_title('Scale-Location')
    
    # 4. Residuals vs Leverage
    ax = axes[1, 1]
    # Calculate leverage (hat values)
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(hat_matrix)
    
    ax.scatter(leverage, standardized_residuals, alpha=0.5)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    ax.set_title('Residuals vs Leverage')
    
    # Add Cook's distance contours
    p = X.shape[1] + 1  # number of parameters
    x_range = np.linspace(0, max(leverage), 100)
    for cook_d in [0.5, 1.0]:
        y_range = np.sqrt(cook_d * p * (1 - x_range) / x_range)
        ax.plot(x_range, y_range, 'r--', alpha=0.5)
        ax.plot(x_range, -y_range, 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# Example usage
lr_diagnostic = LinearRegression(method='normal_equation')
lr_diagnostic.fit(X_simple, y_simple)
diagnose_linear_regression(X_simple, y_simple, lr_diagnostic)
```

---

## 6. Comprehensive Interview Questions & Answers {#interview-qa}

### Basic Concepts

**Q1: What is linear regression and when should you use it?**

**A:** Linear regression is a supervised learning algorithm that models the linear relationship between a dependent variable and one or more independent variables. Use it when:
- The relationship between variables is approximately linear
- You need interpretable results (coefficients show feature impact)
- You're establishing a baseline model
- The problem involves continuous target prediction
- You have limited data (works well with small datasets)

**Q2: Explain the difference between simple and multiple linear regression.**

**A:** 
- **Simple Linear Regression**: Uses one independent variable to predict the target. Equation: y = β₀ + β₁x + ε
- **Multiple Linear Regression**: Uses multiple independent variables. Equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

Key differences:
- Complexity: Multiple regression can capture more complex relationships
- Interpretation: Simple regression shows direct relationship; multiple regression shows partial effects
- Collinearity: Not an issue in simple regression, can be problematic in multiple regression

**Q3: What are the assumptions of linear regression?**

**A:** The five key assumptions (LINE-N):
1. **Linearity**: The relationship between X and Y is linear
   - Check: Residual plots should show no patterns
   
2. **Independence**: Observations are independent of each other
   - Check: Durbin-Watson test for autocorrelation
   
3. **Normality**: Residuals are normally distributed
   - Check: Q-Q plots, Shapiro-Wilk test
   
4. **Equal variance (Homoscedasticity)**: Constant variance of residuals
   - Check: Scale-location plot, Breusch-Pagan test
   
5. **No multicollinearity**: Independent variables are not highly correlated
   - Check: VIF (Variance Inflation Factor), correlation matrix

### Mathematical Understanding

**Q4: Derive the normal equation for linear regression.**

**A:** Starting with the cost function:
```
J(β) = (1/2n) ||Y - Xβ||²
```

Step 1: Expand the cost function
```
J(β) = (1/2n)(Y - Xβ)ᵀ(Y - Xβ)
J(β) = (1/2n)(Yᵀ Y - Yᵀ Xβ - βᵀ Xᵀ Y + βᵀ Xᵀ Xβ)
```

Step 2: Take derivative with respect to β
```
∂J/∂β = (1/n)(-Xᵀ Y + Xᵀ Xβ)
```

Step 3: Set derivative to zero
```
-Xᵀ Y + Xᵀ Xβ = 0
Xᵀ Xβ = Xᵀ Y
```

Step 4: Solve for β
```
β = (Xᵀ X)⁻¹ Xᵀ Y
```

**Q5: Why do we use squared error instead of absolute error?**

**A:** Several reasons:
1. **Mathematical convenience**: Squared error is differentiable everywhere, absolute error is not differentiable at zero
2. **Unique solution**: Squared error gives a unique optimal solution
3. **Penalizes large errors more**: Squared error gives more weight to outliers
4. **Statistical properties**: Under normal distribution assumption, least squares gives maximum likelihood estimate
5. **Computational efficiency**: Closed-form solution exists for squared error

### Gradient Descent

**Q6: Explain gradient descent and its variants.**

**A:** Gradient descent is an iterative optimization algorithm that finds the minimum of a function by moving in the direction of steepest descent.

**Variants:**
1. **Batch Gradient Descent**
   - Uses entire dataset for each update
   - Pros: Stable convergence, guaranteed to find global minimum for convex functions
   - Cons: Slow for large datasets, memory intensive

2. **Stochastic Gradient Descent (SGD)**
   - Uses one sample at a time
   - Pros: Fast, can escape local minima, online learning capable
   - Cons: Noisy updates, may not converge to exact minimum

3. **Mini-batch Gradient Descent**
   - Uses small batches (32-512 samples)
   - Pros: Balance of speed and stability, vectorization benefits
   - Cons: Requires batch size tuning

**Q7: How do you choose the learning rate?**

**A:** Several approaches:
1. **Grid search**: Try values like [0.001, 0.01, 0.1, 1.0]
2. **Learning rate schedules**: Decay over time (e.g., α = α₀/(1 + decay_rate × epoch))
3. **Adaptive methods**: Adam, RMSprop automatically adjust learning rate
4. **Line search**: Find optimal step size at each iteration
5. **Monitor convergence**: Plot cost vs iterations, adjust if too slow or diverging

Rule of thumb: Start with 0.01, increase if too slow, decrease if diverging.

### Regularization

**Q8: Explain L1 and L2 regularization in the context of linear regression.**

**A:** 
**L2 Regularization (Ridge Regression):**
- Adds penalty: λ Σβᵢ²
- Effect: Shrinks coefficients toward zero
- Handles multicollinearity well
- Keeps all features (no feature selection)
- Computationally efficient (closed-form solution exists)

**L1 Regularization (Lasso Regression):**
- Adds penalty: λ Σ|βᵢ|
- Effect: Can drive coefficients to exactly zero
- Performs automatic feature selection
- Useful when you suspect many features are irrelevant
- No closed-form solution (requires iterative methods)

**Q9: When would you use Ridge vs Lasso regression?**

**A:**
**Use Ridge when:**
- You believe most features are relevant
- Features are highly correlated (multicollinearity)
- You want to keep all features but reduce their impact
- You need computational efficiency

**Use Lasso when:**
- You suspect many features are irrelevant
- You want automatic feature selection
- Model interpretability is important
- You have high-dimensional data (p > n)

**Use Elastic Net when:**
- You want benefits of both
- You have correlated features and want feature selection
- You have p >> n with grouped variables

### Practical Considerations

**Q10: How do you handle categorical variables in linear regression?**

**A:** Several encoding methods:
1. **One-Hot Encoding**: Create binary columns for each category
   - Drop one category to avoid dummy variable trap
   - Good for nominal variables
   
2. **Ordinal Encoding**: Map to integers
   - Only for ordinal variables with natural ordering
   
3. **Target Encoding**: Replace with mean of target
   - Risk of overfitting, use with cross-validation
   
4. **Binary Encoding**: Encode as binary numbers
   - Efficient for high-cardinality features

Example:
```python
# One-hot encoding
pd.get_dummies(df['category'], drop_first=True)
```

**Q11: How do you diagnose and fix multicollinearity?**

**A:** 
**Detection:**
1. Correlation matrix: |r| > 0.8 indicates high correlation
2. VIF: VIF > 10 indicates multicollinearity
3. Condition number: > 30 indicates multicollinearity
4. Eigenvalues: Near-zero eigenvalues indicate multicollinearity

**Solutions:**
1. Remove correlated features
2. Combine correlated features (e.g., average)
3. Use regularization (Ridge regression)
4. Principal Component Regression
5. Collect more data

**Q12: What's the difference between correlation and regression?**

**A:**
**Correlation:**
- Measures strength and direction of linear relationship
- Symmetric: Corr(X,Y) = Corr(Y,X)
- Scale-free: Range [-1, 1]
- No causation implied
- Single number summary

**Regression:**
- Models relationship and predicts values
- Asymmetric: Y ~ X is different from X ~ Y
- Scale-dependent coefficients
- Can imply predictive relationship
- Provides equation and predictions

### Advanced Topics

**Q13: How do you handle non-linear relationships in linear regression?**

**A:** Several approaches:
1. **Polynomial features**: Add x², x³, etc.
2. **Interaction terms**: Add x₁×x₂
3. **Transformations**: Log, sqrt, reciprocal
4. **Splines**: Piecewise polynomials
5. **Binning**: Convert continuous to categorical
6. **Switch models**: Use polynomial regression, GAM, or non-linear models

**Q14: Explain the bias-variance tradeoff in linear regression.**

**A:**
- **Bias**: Error from incorrect assumptions (underfitting)
  - Linear regression has high bias if true relationship is non-linear
  
- **Variance**: Error from sensitivity to training data (overfitting)
  - Increases with model complexity (more features, polynomial terms)

**In linear regression:**
- Simple linear regression: High bias, low variance
- Multiple regression: Lower bias, higher variance
- Polynomial regression: Even lower bias, even higher variance
- Regularized regression: Increases bias to reduce variance

**Q15: How do you evaluate a linear regression model?**

**A:** Multiple metrics and methods:

**Metrics:**
1. **R-squared**: Proportion of variance explained (0-1)
2. **Adjusted R-squared**: Penalizes for number of features
3. **MSE/RMSE**: Average squared/root squared error
4. **MAE**: Average absolute error
5. **MAPE**: Mean absolute percentage error

**Diagnostic plots:**
1. Residuals vs Fitted: Check linearity and homoscedasticity
2. Q-Q plot: Check normality
3. Scale-Location: Check homoscedasticity
4. Residuals vs Leverage: Identify influential points

**Validation:**
1. Train-test split
2. Cross-validation
3. Learning curves
4. Prediction intervals

### Comparison Questions

**Q16: Linear Regression vs Logistic Regression?**

**A:**
|Aspect|Linear Regression|Logistic Regression|
|------|----------------|-------------------|
|Output|Continuous values|Probabilities (0-1)|
|Use case|Regression|Classification|
|Function|y = β₀ + β₁x|p = 1/(1 + e^(-z))|
|Cost function|Squared error|Log loss|
|Assumptions|Linear relationship|Log-odds linear relationship|

**Q17: When would linear regression fail?**

**A:** Linear regression fails when:
1. **Non-linear relationships**: True relationship is curved
2. **Discrete outputs**: Predicting categories
3. **Bounded outputs**: Target has natural limits (0-1, 0-100)
4. **Heavy outliers**: Sensitive to extreme values
5. **Heteroscedasticity**: Non-constant variance
6. **Multicollinearity**: Highly correlated features
7. **Non-normal residuals**: Violations affect inference

### Implementation Questions

**Q18: What's the computational complexity of linear regression?**

**A:**
1. **Normal Equation**: 
   - Time: O(p³) for matrix inversion + O(np²) for multiplication
   - Space: O(p²) for storing X^T X
   
2. **Gradient Descent**:
   - Time: O(npt) where t = iterations
   - Space: O(p) for parameters
   
3. **SGD**:
   - Time: O(pt) per epoch
   - Space: O(p)

Choose based on: n (samples), p (features), memory constraints

**Q19: How do you implement linear regression for streaming data?**

**A:** Use online learning approaches:
1. **Stochastic Gradient Descent**: Update on each new sample
2. **Recursive Least Squares**: Update normal equation incrementally
3. **Sliding window**: Maintain fixed-size recent data
4. **Exponential weighting**: Give more weight to recent data
5. **Mini-batch updates**: Process small batches as they arrive

```python
class OnlineLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.zeros(n_features + 1)
        self.learning_rate = learning_rate
    
    def partial_fit(self, X, y):
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        # SGD update
        predictions = X.dot(self.weights)
        errors = predictions - y
        gradient = X.T.dot(errors) / len(y)
        self.weights -= self.learning_rate * gradient
```

**Q20: What are common pitfalls in implementing linear regression?**

**A:**
1. **Forgetting to scale features**: Different scales affect gradient descent
2. **Not checking for multicollinearity**: Unstable coefficients
3. **Including dependent features**: Date and year, area and square feet
4. **Ignoring outliers**: Can severely affect results
5. **Not validating assumptions**: May lead to invalid inference
6. **Overfitting with polynomial features**: Too many features
7. **Data leakage**: Including future information
8. **Not handling missing values**: Can't work with NaN
9. **Extrapolation**: Predicting outside training range
10. **Interpreting correlation as causation**: Coefficients don't imply causation

### Real-world Scenarios

**Q21: A client says their linear regression model has R² = 0.99 on training data but performs poorly on new data. What's likely wrong?**

**A:** Classic overfitting. Possible causes and solutions:
1. **Too many features**: Use regularization or feature selection
2. **Polynomial features too high degree**: Reduce polynomial degree
3. **Small training set**: Collect more data
4. **Data leakage**: Check if target information leaked into features
5. **Not representative training data**: Ensure random split

Diagnostic steps:
- Check train vs test performance
- Plot learning curves
- Examine feature importance
- Cross-validation results

**Q22: How would you explain linear regression to a non-technical stakeholder?**

**A:** "Linear regression is like finding the best-fit line through a cloud of points. Imagine plotting employee experience (x-axis) against salary (y-axis). Linear regression finds the straight line that best represents this relationship, allowing us to predict salary based on experience.

The model gives us:
- **Intercept**: Starting salary (0 experience)
- **Slope**: Salary increase per year of experience
- **R-squared**: How well the line fits (0% = no fit, 100% = perfect fit)

Limitations: Assumes straight-line relationship and can be affected by outliers."

---

## Practice Problems

1. **Implement gradient descent with momentum**
2. **Build a regularization path visualization**
3. **Create online learning linear regression**
4. **Implement weighted least squares**
5. **Build diagnostic plots package**

## Key Takeaways

1. Linear regression is the foundation of many ML algorithms
2. Understanding the math helps in debugging and improvement
3. Assumptions must be validated for valid inference
4. Regularization is crucial for handling real-world data
5. Gradient descent concepts extend to neural networks
6. Diagnostic plots are essential for model validation
7. Choose between normal equation and gradient descent based on data size
8. Feature engineering can capture non-linear relationships