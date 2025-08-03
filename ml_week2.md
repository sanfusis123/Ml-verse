# Machine Learning Interview Preparation - Week 2: Regression & Core ML Concepts

## Day 8: Linear Regression and Gradient Descent

### Linear Regression

Linear regression models the relationship between a dependent variable y and one or more independent variables X.

**Mathematical Formulation:**

For simple linear regression:
```
y = β₀ + β₁x + ε
```

For multiple linear regression:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

In matrix form:
```
Y = Xβ + ε
```

**Normal Equation (Closed-form solution):**
```
β = (XᵀX)⁻¹XᵀY
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic data
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X = X.flatten()

# Add intercept term
X_with_intercept = np.column_stack([np.ones(len(X)), X])

# Closed-form solution using Normal Equation
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
print(f"Coefficients (Normal Equation):")
print(f"β₀ (intercept): {beta[0]:.4f}")
print(f"β₁ (slope): {beta[1]:.4f}")

# Predictions
y_pred = X_with_intercept @ beta

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data points')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Multiple Linear Regression
X_multi, y_multi = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add intercept
X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
X_test_with_intercept = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

# Fit model
beta_multi = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept) @ X_train_with_intercept.T @ y_train

print(f"\nMultiple Linear Regression Coefficients:")
for i, coef in enumerate(beta_multi):
    if i == 0:
        print(f"β₀ (intercept): {coef:.4f}")
    else:
        print(f"β{i} (feature {i}): {coef:.4f}")
```

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function.

**Cost Function (Mean Squared Error):**
```
J(β) = (1/2m) Σᵢ(h(xᵢ) - yᵢ)²
```

**Gradient:**
```
∂J/∂βⱼ = (1/m) Σᵢ(h(xᵢ) - yᵢ)xᵢⱼ
```

**Update Rule:**
```
β := β - α∇J(β)
```

```python
class LinearRegressionGD:
    """Linear Regression with Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.costs = []
        
    def _add_intercept(self, X):
        """Add intercept term to X"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _cost_function(self, X, y, theta):
        """Calculate MSE cost"""
        m = len(y)
        predictions = X @ theta
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def _gradient(self, X, y, theta):
        """Calculate gradient of cost function"""
        m = len(y)
        predictions = X @ theta
        gradient = (1/m) * X.T @ (predictions - y)
        return gradient
    
    def fit(self, X, y):
        """Fit the model using gradient descent"""
        # Add intercept
        X = self._add_intercept(X)
        
        # Initialize parameters
        self.theta = np.zeros(X.shape[1])
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Calculate cost
            cost = self._cost_function(X, y, self.theta)
            self.costs.append(cost)
            
            # Calculate gradient
            gradient = self._gradient(X, y, self.theta)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = self._add_intercept(X)
        return X @ self.theta

# Compare different learning rates
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

learning_rates = [0.001, 0.01, 0.1, 0.5]

for idx, lr in enumerate(learning_rates):
    # Generate data
    X_gd = X.reshape(-1, 1)
    
    # Fit model
    model = LinearRegressionGD(learning_rate=lr, n_iterations=100, verbose=False)
    model.fit(X_gd, y)
    
    # Plot cost history
    ax = axes[idx // 2, idx % 2]
    ax.plot(model.costs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(f'Learning Rate: {lr}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Learning Rate on Convergence')
plt.tight_layout()
plt.show()

# Visualize gradient descent process
def visualize_gradient_descent(X, y, learning_rate=0.01, n_iterations=50):
    """Visualize the gradient descent process"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Parameter space for visualization
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-2, 4, 100)
    
    # Calculate cost for each combination
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta = np.array([theta0, theta1])
            predictions = X_with_intercept @ theta
            J_vals[i, j] = (1/(2*len(y))) * np.sum((predictions - y)**2)
    
    # Run gradient descent and track parameters
    theta = np.array([0., 0.])  # Initial parameters
    theta_history = [theta.copy()]
    
    for _ in range(n_iterations):
        predictions = X_with_intercept @ theta
        gradient = (1/len(y)) * X_with_intercept.T @ (predictions - y)
        theta -= learning_rate * gradient
        theta_history.append(theta.copy())
    
    theta_history = np.array(theta_history)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = plt.subplot(121, projection='3d')
    THETA0, THETA1 = np.meshgrid(theta0_vals, theta1_vals)
    ax1.plot_surface(THETA0, THETA1, J_vals.T, cmap='viridis', alpha=0.6)
    ax1.plot(theta_history[:, 0], theta_history[:, 1], 
             [J_vals[np.argmin(np.abs(theta0_vals - t0)), 
                     np.argmin(np.abs(theta1_vals - t1))] 
              for t0, t1 in theta_history],
             'r.-', markersize=8, linewidth=2)
    ax1.set_xlabel('θ₀ (intercept)')
    ax1.set_ylabel('θ₁ (slope)')
    ax1.set_zlabel('Cost J(θ)')
    ax1.set_title('3D Cost Function')
    
    # Contour plot
    ax2 = plt.subplot(122)
    contour = ax2.contour(THETA0, THETA1, J_vals.T, levels=50)
    ax2.plot(theta_history[:, 0], theta_history[:, 1], 'r.-', 
             markersize=8, linewidth=2, label='Gradient descent path')
    ax2.set_xlabel('θ₀ (intercept)')
    ax2.set_ylabel('θ₁ (slope)')
    ax2.set_title('Contour Plot of Cost Function')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

visualize_gradient_descent(X, y)
```

### Variants of Gradient Descent

```python
class GradientDescentVariants:
    """Implementation of different gradient descent variants"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def batch_gradient_descent(self, X, y, theta_init):
        """Standard (Batch) Gradient Descent"""
        theta = theta_init.copy()
        costs = []
        m = len(y)
        
        for _ in range(self.n_iterations):
            predictions = X @ theta
            gradient = (1/m) * X.T @ (predictions - y)
            theta -= self.learning_rate * gradient
            
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            costs.append(cost)
            
        return theta, costs
    
    def stochastic_gradient_descent(self, X, y, theta_init):
        """Stochastic Gradient Descent (SGD)"""
        theta = theta_init.copy()
        costs = []
        m = len(y)
        
        for _ in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Update for each sample
            for i in range(m):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                prediction = xi @ theta
                gradient = xi.T @ (prediction - yi)
                theta -= self.learning_rate * gradient
            
            # Calculate cost for entire dataset
            predictions = X @ theta
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            costs.append(cost)
            
        return theta, costs
    
    def mini_batch_gradient_descent(self, X, y, theta_init, batch_size=32):
        """Mini-batch Gradient Descent"""
        theta = theta_init.copy()
        costs = []
        m = len(y)
        
        for _ in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                predictions = X_batch @ theta
                gradient = (1/len(y_batch)) * X_batch.T @ (predictions - y_batch)
                theta -= self.learning_rate * gradient
            
            # Calculate cost for entire dataset
            predictions = X @ theta
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            costs.append(cost)
            
        return theta, costs

# Compare gradient descent variants
X_gd, y_gd = make_regression(n_samples=1000, n_features=2, noise=20, random_state=42)
X_gd_with_intercept = np.column_stack([np.ones(len(X_gd)), X_gd])

# Initialize parameters
theta_init = np.zeros(X_gd_with_intercept.shape[1])

# Create optimizer
optimizer = GradientDescentVariants(learning_rate=0.01, n_iterations=50)

# Run different variants
theta_batch, costs_batch = optimizer.batch_gradient_descent(X_gd_with_intercept, y_gd, theta_init)
theta_sgd, costs_sgd = optimizer.stochastic_gradient_descent(X_gd_with_intercept, y_gd, theta_init)
theta_mini, costs_mini = optimizer.mini_batch_gradient_descent(X_gd_with_intercept, y_gd, theta_init)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(costs_batch, label='Batch GD', linewidth=2)
plt.plot(costs_sgd, label='Stochastic GD', alpha=0.7)
plt.plot(costs_mini, label='Mini-batch GD', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence Comparison of Gradient Descent Variants')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

### Advanced Optimization Techniques

```python
class AdvancedOptimizers:
    """Advanced optimization algorithms"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def gradient_descent_with_momentum(self, X, y, theta_init, beta=0.9):
        """Gradient Descent with Momentum"""
        theta = theta_init.copy()
        velocity = np.zeros_like(theta)
        costs = []
        m = len(y)
        
        for _ in range(self.n_iterations):
            predictions = X @ theta
            gradient = (1/m) * X.T @ (predictions - y)
            
            # Update velocity
            velocity = beta * velocity - self.learning_rate * gradient
            
            # Update parameters
            theta += velocity
            
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            costs.append(cost)
            
        return theta, costs
    
    def adam(self, X, y, theta_init, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer"""
        theta = theta_init.copy()
        m_t = np.zeros_like(theta)  # First moment
        v_t = np.zeros_like(theta)  # Second moment
        costs = []
        m = len(y)
        
        for t in range(1, self.n_iterations + 1):
            predictions = X @ theta
            gradient = (1/m) * X.T @ (predictions - y)
            
            # Update biased first moment estimate
            m_t = beta1 * m_t + (1 - beta1) * gradient
            
            # Update biased second raw moment estimate
            v_t = beta2 * v_t + (1 - beta2) * gradient**2
            
            # Compute bias-corrected first moment estimate
            m_hat = m_t / (1 - beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v_t / (1 - beta2**t)
            
            # Update parameters
            theta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            costs.append(cost)
            
        return theta, costs
    
    def adagrad(self, X, y, theta_init, epsilon=1e-8):
        """AdaGrad optimizer"""
        theta = theta_init.copy()
        accumulated_grad = np.zeros_like(theta)
        costs = []
        m = len(y)
        
        for _ in range(self.n_iterations):
            predictions = X @ theta
            gradient = (1/m) * X.T @ (predictions - y)
            
            # Accumulate squared gradients
            accumulated_grad += gradient**2
            
            # Update parameters with adaptive learning rate
            theta -= self.learning_rate * gradient / (np.sqrt(accumulated_grad) + epsilon)
            
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            costs.append(cost)
            
        return theta, costs

# Compare advanced optimizers
advanced_opt = AdvancedOptimizers(learning_rate=0.1, n_iterations=100)

theta_momentum, costs_momentum = advanced_opt.gradient_descent_with_momentum(X_gd_with_intercept, y_gd, theta_init)
theta_adam, costs_adam = advanced_opt.adam(X_gd_with_intercept, y_gd, theta_init)
theta_adagrad, costs_adagrad = advanced_opt.adagrad(X_gd_with_intercept, y_gd, theta_init)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Convergence comparison
ax1.plot(costs_batch[:100], label='Vanilla GD', linewidth=2)
ax1.plot(costs_momentum, label='Momentum', linewidth=2)
ax1.plot(costs_adam, label='Adam', linewidth=2)
ax1.plot(costs_adagrad, label='AdaGrad', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')
ax1.set_title('Advanced Optimizers Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Learning rate schedules
iterations = np.arange(1, 101)
lr_constant = np.ones_like(iterations) * 0.1
lr_exponential = 0.1 * 0.95**iterations
lr_inverse = 0.1 / (1 + 0.01 * iterations)
lr_step = np.where(iterations % 30 == 0, lr_constant * 0.5, lr_constant)

ax2.plot(iterations, lr_constant, label='Constant', linewidth=2)
ax2.plot(iterations, lr_exponential, label='Exponential decay', linewidth=2)
ax2.plot(iterations, lr_inverse, label='Inverse time decay', linewidth=2)
ax2.plot(iterations, lr_step, label='Step decay', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedules')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Day 9: Ridge and Lasso Regression

### Ridge Regression (L2 Regularization)

Ridge regression adds a penalty term to the cost function to prevent overfitting:

**Cost Function:**
```
J(β) = (1/2m) Σᵢ(yᵢ - βᵀxᵢ)² + λΣⱼβⱼ²
```

**Closed-form solution:**
```
β = (XᵀX + λI)⁻¹XᵀY
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Generate data with polynomial features (prone to overfitting)
np.random.seed(42)
n_samples = 100
X_poly = np.sort(np.random.uniform(-3, 3, n_samples))
y_true = 0.5 * X_poly**3 - X_poly**2 + 2 * X_poly + 1
y_poly = y_true + np.random.normal(0, 3, n_samples)

# Create polynomial features
degrees = [1, 5, 15]
alphas = [0, 0.01, 0.1, 1.0]

fig, axes = plt.subplots(len(degrees), len(alphas), figsize=(15, 12))

for i, degree in enumerate(degrees):
    for j, alpha in enumerate(alphas):
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly_features = poly_features.fit_transform(X_poly.reshape(-1, 1))
        
        # Fit Ridge regression
        if alpha == 0:
            # Ordinary least squares
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:
            model = Ridge(alpha=alpha)
        
        model.fit(X_poly_features, y_poly)
        
        # Predictions
        X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
        X_plot_poly = poly_features.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        
        # Plot
        ax = axes[i, j]
        ax.scatter(X_poly, y_poly, alpha=0.6, s=20)
        ax.plot(X_plot, y_plot, 'r-', linewidth=2)
        ax.set_ylim(-20, 20)
        ax.set_title(f'Degree={degree}, α={alpha}')
        
        if i == len(degrees) - 1:
            ax.set_xlabel('X')
        if j == 0:
            ax.set_ylabel('y')

plt.suptitle('Ridge Regression: Effect of Polynomial Degree and Regularization', fontsize=16)
plt.tight_layout()
plt.show()

# Ridge coefficient paths
alphas = np.logspace(-3, 3, 100)
coefs = []

# Generate more complex data
X_ridge, y_ridge = make_regression(n_samples=100, n_features=20, 
                                  n_informative=10, noise=2, random_state=42)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_ridge, y_ridge)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

plt.figure(figsize=(10, 6))
for i in range(X_ridge.shape[1]):
    plt.plot(alphas, coefs[:, i])

plt.xscale('log')
plt.xlabel('Regularization parameter (α)')
plt.ylabel('Coefficient value')
plt.title('Ridge Coefficient Paths')
plt.grid(True, alpha=0.3)
plt.show()
```

### Lasso Regression (L1 Regularization)

Lasso regression uses L1 penalty which can lead to sparse solutions:

**Cost Function:**
```
J(β) = (1/2m) Σᵢ(yᵢ - βᵀxᵢ)² + λΣⱼ|βⱼ|
```

```python
# Lasso coefficient paths
from sklearn.linear_model import lasso_path

# Use the same data as Ridge
alphas_lasso, coefs_lasso, _ = lasso_path(X_ridge, y_ridge, alphas=alphas)

plt.figure(figsize=(10, 6))
for i in range(X_ridge.shape[1]):
    plt.plot(alphas_lasso, coefs_lasso[i, :])

plt.xscale('log')
plt.xlabel('Regularization parameter (α)')
plt.ylabel('Coefficient value')
plt.title('Lasso Coefficient Paths (Note: coefficients shrink to exactly zero)')
plt.grid(True, alpha=0.3)
plt.show()

# Compare Ridge vs Lasso
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ridge)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_ridge, 
                                                    test_size=0.3, random_state=42)

# Compare models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1),
    'Lasso (α=1.0)': Lasso(alpha=1.0),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Count non-zero coefficients
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
    else:
        n_nonzero = X_train.shape[1]
    
    results.append({
        'Model': name,
        'MSE': mse,
        'R²': r2,
        'Non-zero coefs': n_nonzero
    })

results_df = pd.DataFrame(results)
print("Model Comparison:")
print(results_df.to_string(index=False))

# Visualize coefficient comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models_to_plot = ['Linear Regression', 'Ridge (α=1.0)', 'Lasso (α=1.0)']

for idx, model_name in enumerate(models_to_plot):
    model = models[model_name]
    coefs = model.coef_
    
    ax = axes[idx]
    ax.bar(range(len(coefs)), coefs)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Coefficient value')
    ax.set_title(model_name)
    ax.grid(True, alpha=0.3)

plt.suptitle('Coefficient Comparison: Linear vs Ridge vs Lasso')
plt.tight_layout()
plt.show()
```

### Elastic Net (Combining L1 and L2)

Elastic Net combines Ridge and Lasso penalties:

**Cost Function:**
```
J(β) = (1/2m) Σᵢ(yᵢ - βᵀxᵢ)² + λ₁Σⱼ|βⱼ| + λ₂Σⱼβⱼ²
```

```python
# Elastic Net demonstration
from sklearn.linear_model import ElasticNetCV

# Generate correlated features
n_samples, n_features = 100, 30
X_corr = np.random.randn(n_samples, n_features)
# Make features correlated
for i in range(5):
    X_corr[:, i+5] = X_corr[:, i] + np.random.normal(0, 0.1, n_samples)
    X_corr[:, i+10] = X_corr[:, i] + np.random.normal(0, 0.2, n_samples)

# True coefficients (sparse with grouped features)
true_coef = np.zeros(n_features)
true_coef[0:5] = [3, -2, 1.5, 0, 0]
true_coef[15:20] = [-1, 2, 0, -1.5, 0]

y_corr = X_corr @ true_coef + np.random.normal(0, 0.5, n_samples)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X_corr, y_corr, 
                                                    test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare regularization methods
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]  # 0=Ridge, 1=Lasso
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, l1_ratio in enumerate(l1_ratios):
    if l1_ratio == 0:
        model = Ridge(alpha=0.1)
        title = 'Ridge (L1 ratio = 0)'
    elif l1_ratio == 1:
        model = Lasso(alpha=0.1)
        title = 'Lasso (L1 ratio = 1)'
    else:
        model = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10000)
        title = f'Elastic Net (L1 ratio = {l1_ratio})'
    
    model.fit(X_train_scaled, y_train)
    
    # Plot coefficients
    ax = axes[idx]
    x_pos = np.arange(len(model.coef_))
    ax.bar(x_pos[true_coef != 0], model.coef_[true_coef != 0], 
           color='green', label='True non-zero', alpha=0.7)
    ax.bar(x_pos[true_coef == 0], model.coef_[true_coef == 0], 
           color='red', label='True zero', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Coefficient value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    ax.text(0.02, 0.98, f'MSE: {mse:.3f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

# Add true coefficients for reference
ax = axes[5]
ax.bar(range(len(true_coef)), true_coef, color='blue', alpha=0.7)
ax.set_xlabel('Feature index')
ax.set_ylabel('Coefficient value')
ax.set_title('True Coefficients')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Cross-Validation for Hyperparameter Selection

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Ridge CV
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)

# Lasso CV
lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

# Elastic Net CV
elastic_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], 
                         cv=5, max_iter=10000)
elastic_cv.fit(X_train_scaled, y_train)

print("Optimal hyperparameters from CV:")
print(f"Ridge - optimal α: {ridge_cv.alpha_:.4f}")
print(f"Lasso - optimal α: {lasso_cv.alpha_:.4f}")
print(f"Elastic Net - optimal α: {elastic_cv.alpha_:.4f}, l1_ratio: {elastic_cv.l1_ratio_:.2f}")

# Visualize CV scores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ridge CV scores
axes[0].plot(np.log10(ridge_cv.alphas_), ridge_cv.cv_values_.mean(axis=0), 'b-')
axes[0].fill_between(np.log10(ridge_cv.alphas_), 
                     ridge_cv.cv_values_.mean(axis=0) - ridge_cv.cv_values_.std(axis=0),
                     ridge_cv.cv_values_.mean(axis=0) + ridge_cv.cv_values_.std(axis=0),
                     alpha=0.3)
axes[0].axvline(np.log10(ridge_cv.alpha_), color='red', linestyle='--', 
                label=f'Optimal α = {ridge_cv.alpha_:.4f}')
axes[0].set_xlabel('log10(α)')
axes[0].set_ylabel('Mean CV Score')
axes[0].set_title('Ridge Cross-Validation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Lasso path with CV
axes[1].plot(np.log10(lasso_cv.alphas_), lasso_cv.mse_path_.mean(axis=1), 'g-')
axes[1].fill_between(np.log10(lasso_cv.alphas_),
                     lasso_cv.mse_path_.mean(axis=1) - lasso_cv.mse_path_.std(axis=1),
                     lasso_cv.mse_path_.mean(axis=1) + lasso_cv.mse_path_.std(axis=1),
                     alpha=0.3)
axes[1].axvline(np.log10(lasso_cv.alpha_), color='red', linestyle='--',
                label=f'Optimal α = {lasso_cv.alpha_:.4f}')
axes[1].set_xlabel('log10(α)')
axes[1].set_ylabel('Mean MSE')
axes[1].set_title('Lasso Cross-Validation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Day 10: Overfitting vs Underfitting, Bias-Variance Tradeoff

### Understanding Overfitting and Underfitting

```python
# Generate synthetic data
np.random.seed(42)
n_samples = 100
X_true = np.sort(np.random.uniform(0, 4, n_samples))
y_true = np.sin(2 * X_true) + 0.5 * X_true
y_noisy = y_true + np.random.normal(0, 0.3, n_samples)

# Fit models with different complexities
degrees = [1, 3, 5, 10, 20]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

X_plot = np.linspace(0, 4, 300)
train_errors = []
test_errors = []

# Split data
train_idx = np.random.choice(n_samples, size=70, replace=False)
test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

X_train, y_train = X_true[train_idx], y_noisy[train_idx]
X_test, y_test = X_true[test_idx], y_noisy[test_idx]

for idx, degree in enumerate(degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))
    X_plot_poly = poly.transform(X_plot.reshape(-1, 1))
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    y_plot_pred = model.predict(X_plot_poly)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    # Plot
    ax = axes[idx]
    ax.scatter(X_train, y_train, alpha=0.6, label='Train data', s=30)
    ax.scatter(X_test, y_test, alpha=0.6, label='Test data', s=30, marker='s')
    ax.plot(X_plot, y_plot_pred, 'r-', linewidth=2, label='Model')
    ax.plot(X_true, y_true, 'g--', linewidth=2, alpha=0.7, label='True function')
    
    ax.set_title(f'Degree {degree}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 4)
    
# Plot learning curves
ax = axes[5]
ax.plot(degrees, train_errors, 'o-', label='Training error', linewidth=2, markersize=8)
ax.plot(degrees, test_errors, 's-', label='Test error', linewidth=2, markersize=8)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_title('Model Complexity vs Error')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Underfitting (low degree) → Good fit → Overfitting (high degree)', fontsize=16)
plt.tight_layout()
plt.show()
```

### Bias-Variance Decomposition

The expected test error can be decomposed as:
```
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²
```

Where:
- **Bias²**: Error from erroneous assumptions in the learning algorithm
- **Variance**: Error from sensitivity to small fluctuations in the training set
- **σ²**: Irreducible error (noise)

```python
def bias_variance_decomposition(X, y_true, noise_level, model_class, 
                               model_params, n_simulations=100):
    """
    Estimate bias and variance through simulation
    """
    n_samples = len(X)
    predictions = []
    
    for _ in range(n_simulations):
        # Generate new noisy data
        y_noisy = y_true + np.random.normal(0, noise_level, n_samples)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X, y_noisy)
        
        # Store predictions
        y_pred = model.predict(X)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # Calculate bias and variance
    mean_predictions = np.mean(predictions, axis=0)
    
    bias_squared = np.mean((mean_predictions - y_true)**2)
    variance = np.mean(np.var(predictions, axis=0))
    noise = noise_level**2
    
    total_error = bias_squared + variance + noise
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'noise': noise,
        'total_error': total_error,
        'predictions': predictions,
        'mean_predictions': mean_predictions
    }

# Analyze bias-variance for different model complexities
noise_level = 0.3
complexities = [1, 2, 3, 5, 10, 15]
results = []

# True function
X_bv = np.linspace(0, 1, 50).reshape(-1, 1)
y_true_bv = np.sin(4 * np.pi * X_bv).ravel()

for degree in complexities:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_bv)
    
    # Analyze
    result = bias_variance_decomposition(
        X_poly, y_true_bv, noise_level,
        LinearRegression, {}, n_simulations=100
    )
    result['degree'] = degree
    results.append(result)

# Visualize bias-variance tradeoff
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Bias² and Variance vs Complexity
ax = axes[0, 0]
degrees_list = [r['degree'] for r in results]
bias_list = [r['bias_squared'] for r in results]
var_list = [r['variance'] for r in results]
total_list = [r['total_error'] for r in results]

ax.plot(degrees_list, bias_list, 'b-o', label='Bias²', linewidth=2, markersize=8)
ax.plot(degrees_list, var_list, 'r-s', label='Variance', linewidth=2, markersize=8)
ax.plot(degrees_list, total_list, 'g-^', label='Total Error', linewidth=2, markersize=8)
ax.set_xlabel('Model Complexity (Polynomial Degree)')
ax.set_ylabel('Error')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2-4: Visualize predictions for different complexities
example_degrees = [1, 5, 15]
for idx, degree in enumerate(example_degrees):
    ax = axes.ravel()[idx + 1]
    
    # Get results for this degree
    result = next(r for r in results if r['degree'] == degree)
    predictions = result['predictions']
    mean_pred = result['mean_predictions']
    
    # Plot sample predictions
    for i in range(min(20, len(predictions))):
        ax.plot(X_bv, predictions[i], 'b-', alpha=0.1, linewidth=0.5)
    
    # Plot mean prediction and true function
    ax.plot(X_bv, mean_pred, 'r-', linewidth=3, label='Mean prediction')
    ax.plot(X_bv, y_true_bv, 'g--', linewidth=2, label='True function')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Degree {degree}: Bias²={result["bias_squared"]:.3f}, '
                f'Var={result["variance"]:.3f}')
    ax.legend()
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show()
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, cv=5, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """Plot learning curves for an estimator"""
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )
    
    # Convert to positive MSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score", linewidth=2, markersize=8)
    plt.plot(train_sizes, val_scores_mean, 's-', color="g",
             label="Cross-validation score", linewidth=2, markersize=8)
    
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    return train_scores_mean, val_scores_mean

# Generate dataset
X_lc, y_lc = make_regression(n_samples=1000, n_features=20, 
                            n_informative=15, noise=10, random_state=42)

# Compare learning curves for different models
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Model 1: Linear Regression (might underfit)
ax = plt.subplot(2, 2, 1)
plot_learning_curves(LinearRegression(), X_lc, y_lc)
plt.title('Linear Regression Learning Curves')

# Model 2: Polynomial features (might overfit)
ax = plt.subplot(2, 2, 2)
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=5)),
    ('linear', LinearRegression())
])
plot_learning_curves(poly_pipeline, X_lc, y_lc)
plt.title('Polynomial Regression (degree=5) Learning Curves')

# Model 3: Ridge regression
ax = plt.subplot(2, 2, 3)
plot_learning_curves(Ridge(alpha=1.0), X_lc, y_lc)
plt.title('Ridge Regression Learning Curves')

# Model 4: Lasso regression
ax = plt.subplot(2, 2, 4)
plot_learning_curves(Lasso(alpha=0.1, max_iter=2000), X_lc, y_lc)
plt.title('Lasso Regression Learning Curves')

plt.tight_layout()
plt.show()
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5):
    """Plot validation curve for hyperparameter tuning"""
    
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    # Convert to positive MSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_scores_mean, 'o-', label="Training score",
                 color="darkorange", linewidth=2, markersize=8)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange")
    plt.semilogx(param_range, val_scores_mean, 's-', label="Cross-validation score",
                 color="navy", linewidth=2, markersize=8)
    plt.fill_between(param_range, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.2,
                     color="navy")
    
    plt.xlabel(param_name)
    plt.ylabel("MSE")
    plt.title("Validation Curve")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Find optimal value
    optimal_idx = np.argmin(val_scores_mean)
    optimal_value = param_range[optimal_idx]
    plt.axvline(optimal_value, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal {param_name}={optimal_value:.4f}')

# Validation curves for Ridge regression
param_range = np.logspace(-3, 3, 50)
plot_validation_curve(Ridge(), X_lc, y_lc, "alpha", param_range)
plt.show()

# Strategies to address overfitting and underfitting
strategies = """
Strategies to Address Overfitting:
1. Regularization (L1, L2, Elastic Net)
2. Cross-validation
3. Early stopping
4. Dropout (for neural networks)
5. Reduce model complexity
6. Increase training data
7. Feature selection
8. Ensemble methods (bagging)

Strategies to Address Underfitting:
1. Increase model complexity
2. Add polynomial features
3. Reduce regularization
4. Feature engineering
5. Try non-linear models
6. Ensemble methods (boosting)
"""

print(strategies)
```

## Day 11: Regression Metrics

### Key Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error

def regression_metrics_demo():
    """Demonstrate various regression metrics"""
    
    # Generate sample predictions
    np.random.seed(42)
    n_samples = 100
    
    # True values
    y_true = np.random.uniform(10, 100, n_samples)
    
    # Different prediction scenarios
    # Good predictions
    y_pred_good = y_true + np.random.normal(0, 5, n_samples)
    
    # Biased predictions (systematic overestimation)
    y_pred_biased = y_true * 1.2 + np.random.normal(0, 3, n_samples)
    
    # High variance predictions
    y_pred_high_var = y_true + np.random.normal(0, 20, n_samples)
    
    # Calculate metrics for each scenario
    scenarios = [
        ('Good Model', y_true, y_pred_good),
        ('Biased Model', y_true, y_pred_biased),
        ('High Variance Model', y_true, y_pred_high_var)
    ]
    
    metrics_results = []
    
    for name, y_t, y_p in scenarios:
        metrics = {
            'Model': name,
            'MSE': mean_squared_error(y_t, y_p),
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            'MAE': mean_absolute_error(y_t, y_p),
            'R²': r2_score(y_t, y_p),
            'Explained Var': explained_variance_score(y_t, y_p),
            'MAPE': mean_absolute_percentage_error(y_t, y_p)
        }
        metrics_results.append(metrics)
    
    # Display results
    metrics_df = pd.DataFrame(metrics_results)
    print("Regression Metrics Comparison:")
    print(metrics_df.round(3))
    
    # Visualize predictions vs actual
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, y_t, y_p) in enumerate(scenarios):
        # Scatter plot
        ax = axes[0, idx]
        ax.scatter(y_t, y_p, alpha=0.6)
        ax.plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'r--', linewidth=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{name}\nR² = {r2_score(y_t, y_p):.3f}')
        ax.grid(True, alpha=0.3)
        
        # Residual plot
        ax = axes[1, idx]
        residuals = y_t - y_p
        ax.scatter(y_p, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residual Plot')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

metrics_comparison = regression_metrics_demo()
```

### Understanding R² and Adjusted R²

```python
def r_squared_demo():
    """Demonstrate R² behavior and limitations"""
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    
    # Simple linear relationship
    X_simple = np.random.uniform(0, 10, n_samples)
    y_simple = 2 * X_simple + 3 + np.random.normal(0, 2, n_samples)
    
    # Create models with increasing complexity
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Model 1: Constant prediction (baseline)
    y_pred_constant = np.full_like(y_simple, np.mean(y_simple))
    r2_constant = r2_score(y_simple, y_pred_constant)
    
    ax = axes[0, 0]
    ax.scatter(X_simple, y_simple, alpha=0.6)
    ax.axhline(y=np.mean(y_simple), color='r', linewidth=2, label='Mean prediction')
    ax.set_title(f'Baseline Model\nR² = {r2_constant:.3f} (always 0)')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    
    # Model 2: Linear regression
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_simple.reshape(-1, 1), y_simple)
    y_pred_linear = lr.predict(X_simple.reshape(-1, 1))
    r2_linear = r2_score(y_simple, y_pred_linear)
    
    ax = axes[0, 1]
    ax.scatter(X_simple, y_simple, alpha=0.6)
    ax.plot(X_simple, y_pred_linear, 'r-', linewidth=2, label='Linear fit')
    ax.set_title(f'Linear Model\nR² = {r2_linear:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    
    # Demonstrate R² with multiple features
    # Adding random features
    n_features_list = [1, 5, 10, 20, 30, 40, 50]
    r2_scores_train = []
    r2_scores_test = []
    adjusted_r2_train = []
    
    X_base = X_simple.reshape(-1, 1)
    
    for n_features in n_features_list:
        # Add random features
        if n_features > 1:
            X_random = np.random.randn(n_samples, n_features - 1)
            X_multi = np.hstack([X_base, X_random])
        else:
            X_multi = X_base
        
        # Split data
        split_idx = int(0.7 * n_samples)
        X_train, X_test = X_multi[:split_idx], X_multi[split_idx:]
        y_train, y_test = y_simple[:split_idx], y_simple[split_idx:]
        
        # Fit model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Calculate R²
        r2_train = r2_score(y_train, lr.predict(X_train))
        r2_test = r2_score(y_test, lr.predict(X_test))
        
        # Adjusted R²
        n, p = X_train.shape
        adj_r2 = 1 - (1 - r2_train) * (n - 1) / (n - p - 1)
        
        r2_scores_train.append(r2_train)
        r2_scores_test.append(r2_test)
        adjusted_r2_train.append(adj_r2)
    
    # Plot R² vs number of features
    ax = axes[1, 0]
    ax.plot(n_features_list, r2_scores_train, 'o-', label='R² (train)', linewidth=2, markersize=8)
    ax.plot(n_features_list, r2_scores_test, 's-', label='R² (test)', linewidth=2, markersize=8)
    ax.plot(n_features_list, adjusted_r2_train, '^-', label='Adjusted R² (train)', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('R² Score')
    ax.set_title('R² vs Model Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Interpretation guide
    ax = axes[1, 1]
    ax.text(0.1, 0.9, "R² Interpretation:", fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.8, "• R² = 1: Perfect prediction", transform=ax.transAxes)
    ax.text(0.1, 0.7, "• R² = 0: No better than mean", transform=ax.transAxes)
    ax.text(0.1, 0.6, "• R² < 0: Worse than mean", transform=ax.transAxes)
    ax.text(0.1, 0.4, "Adjusted R² Formula:", fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.3, "Adj R² = 1 - (1-R²)(n-1)/(n-p-1)", transform=ax.transAxes, fontfamily='monospace')
    ax.text(0.1, 0.2, "where n = samples, p = features", transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

r_squared_demo()
```

### Custom Metrics and Loss Functions

```python
def custom_metrics_demo():
    """Demonstrate custom metrics for specific use cases"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    # Scenario 1: Asymmetric loss (overestimation is worse)
    y_true = np.random.exponential(50, n_samples)
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    
    def asymmetric_loss(y_true, y_pred, alpha=0.7):
        """
        Custom asymmetric loss where overestimation is penalized more
        alpha > 0.5 penalizes overestimation more
        """
        residuals = y_true - y_pred
        return np.mean(np.where(residuals < 0, 
                               alpha * residuals**2, 
                               (1-alpha) * residuals**2))
    
    # Scenario 2: Log-cosh loss (robust to outliers)
    def log_cosh_loss(y_true, y_pred):
        """Log-cosh loss: smoother than MAE, more robust than MSE"""
        return np.mean(np.log(np.cosh(y_pred - y_true)))
    
    # Scenario 3: Quantile loss
    def quantile_loss(y_true, y_pred, quantile=0.5):
        """Quantile loss for quantile regression"""
        residuals = y_true - y_pred
        return np.mean(np.where(residuals >= 0,
                               quantile * residuals,
                               (quantile - 1) * residuals))
    
    # Compare losses
    losses = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Asymmetric (α=0.7)': asymmetric_loss(y_true, y_pred, 0.7),
        'Log-Cosh': log_cosh_loss(y_true, y_pred),
        'Quantile (0.5)': quantile_loss(y_true, y_pred, 0.5),
        'Quantile (0.9)': quantile_loss(y_true, y_pred, 0.9)
    }
    
    print("Custom Loss Functions Comparison:")
    for name, value in losses.items():
        print(f"{name}: {value:.3f}")
    
    # Visualize loss functions behavior
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss as function of residual
    ax = axes[0, 0]
    residuals = np.linspace(-10, 10, 100)
    
    ax.plot(residuals, residuals**2, label='Squared (MSE)', linewidth=2)
    ax.plot(residuals, np.abs(residuals), label='Absolute (MAE)', linewidth=2)
    ax.plot(residuals, np.log(np.cosh(residuals)), label='Log-Cosh', linewidth=2)
    
    # Asymmetric loss
    asym_loss = np.where(residuals < 0, 0.7 * residuals**2, 0.3 * residuals**2)
    ax.plot(residuals, asym_loss, label='Asymmetric (α=0.7)', linewidth=2)
    
    ax.set_xlabel('Residual (y_true - y_pred)')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Functions Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted with different metrics
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([0, y_true.max()], [0, y_true.max()], 'r--', linewidth=2)
    
    # Add metric annotations
    metrics_text = f"MSE: {mean_squared_error(y_true, y_pred):.2f}\n"
    metrics_text += f"MAE: {mean_absolute_error(y_true, y_pred):.2f}\n"
    metrics_text += f"R²: {r2_score(y_true, y_pred):.3f}"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions vs Actual')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax = axes[1, 0]
    errors = y_pred - y_true
    ax.hist(errors, bins=30, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(errors), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(errors):.2f}')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Metrics interpretation
    ax = axes[1, 1]
    metrics_info = """
    Metric Selection Guidelines:
    
    • MSE: Penalizes large errors heavily
      - Use when large errors are particularly bad
      
    • MAE: Treats all errors equally
      - Use for robust estimation
      - Less sensitive to outliers
      
    • MAPE: Percentage-based error
      - Use when errors should be relative to magnitude
      - Problematic when true values are near zero
      
    • R²: Proportion of variance explained
      - Use for model comparison
      - Can be misleading with nonlinear relationships
      
    • Custom losses: Domain-specific requirements
      - Asymmetric: Different costs for over/under estimation
      - Quantile: For prediction intervals
    """
    
    ax.text(0.05, 0.95, metrics_info, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

custom_metrics_demo()
```

## Day 12: Logistic Regression

### Binary Classification with Logistic Regression

Logistic regression models the probability of a binary outcome:

**Logistic (Sigmoid) Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Model:**
```
P(y=1|x) = σ(βᵀx) = 1 / (1 + e^(-βᵀx))
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

# Visualize sigmoid function
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid(z), 'b-', linewidth=3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
plt.xlabel('z = βᵀx')
plt.ylabel('σ(z) = P(y=1|x)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.show()

# Generate binary classification data
from sklearn.datasets import make_classification

X_log, y_log = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                  n_informative=2, random_state=42,
                                  n_clusters_per_class=1, flip_y=0.1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, 
                                                    test_size=0.3, random_state=42)

# Fit logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Get predictions and probabilities
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    """Plot decision boundary for 2D classification"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=20)
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                         edgecolor='black', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='P(y=1|x)')
    
    # Add coefficient visualization
    if hasattr(model, 'coef_'):
        w = model.coef_[0]
        b = model.intercept_[0]
        # Decision boundary: w1*x1 + w2*x2 + b = 0
        # x2 = -(w1*x1 + b) / w2
        x_boundary = np.array([x_min, x_max])
        y_boundary = -(w[0] * x_boundary + b) / w[1]
        plt.plot(x_boundary, y_boundary, 'k--', linewidth=3, 
                label=f'Decision boundary: {w[0]:.2f}x₁ + {w[1]:.2f}x₂ + {b:.2f} = 0')
        plt.legend()

plot_decision_boundary(X_test, y_test, log_reg, 'Logistic Regression Decision Boundary')
plt.show()
```

### ROC Curve and AUC

```python
def plot_roc_and_pr_curves(y_true, y_prob, model_name='Model'):
    """Plot ROC and Precision-Recall curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add threshold annotations
    thresholds_to_annotate = [0.2, 0.5, 0.8]
    for threshold in thresholds_to_annotate:
        idx = np.argmin(np.abs(thresholds - threshold))
        ax1.annotate(f'θ={threshold:.1f}', 
                    xy=(fpr[idx], tpr[idx]), 
                    xytext=(fpr[idx] + 0.1, tpr[idx] - 0.1),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, 'g-', linewidth=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {model_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add baseline (random classifier)
    baseline_precision = np.sum(y_true) / len(y_true)
    ax2.axhline(y=baseline_precision, color='r', linestyle='--', 
                label=f'Baseline = {baseline_precision:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    return roc_auc, pr_auc

# Plot ROC and PR curves
roc_auc, pr_auc = plot_roc_and_pr_curves(y_test, y_prob, 'Logistic Regression')

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()
```

### Multi-class Logistic Regression

```python
# Generate multi-class data
X_multi, y_multi = make_classification(n_samples=1000, n_features=2, 
                                      n_informative=2, n_redundant=0,
                                      n_clusters_per_class=1, n_classes=3,
                                      random_state=42)

# Split data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi,
                                                            test_size=0.3, 
                                                            random_state=42)

# Fit multi-class logistic regression
log_reg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                  max_iter=1000, random_state=42)
log_reg_multi.fit(X_train_m, y_train_m)

# Visualize multi-class decision boundaries
def plot_multiclass_decision_boundary(X, y, model, title):
    """Plot decision boundary for multi-class classification"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)
    
    # Plot points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis,
                         edgecolor='black', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    
    # Add class probabilities visualization
    n_classes = len(np.unique(y))
    fig2, axes = plt.subplots(1, n_classes, figsize=(15, 5))
    
    for class_idx in range(n_classes):
        Z_prob = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, class_idx]
        Z_prob = Z_prob.reshape(xx.shape)
        
        ax = axes[class_idx]
        contour = ax.contourf(xx, yy, Z_prob, alpha=0.8, cmap=plt.cm.RdYlBu, levels=20)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis,
                  edgecolor='black', s=30, alpha=0.5)
        ax.set_title(f'P(y={class_idx}|x)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    plt.show()

plot_multiclass_decision_boundary(X_test_m, y_test_m, log_reg_multi,
                                 'Multi-class Logistic Regression')

# Multi-class metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred_multi = log_reg_multi.predict(X_test_m)
print("\nMulti-class Classification Report:")
print(classification_report(y_test_m, y_pred_multi))

# Multi-class confusion matrix
cm_multi = confusion_matrix(y_test_m, y_pred_multi)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Multi-class Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### Regularization in Logistic Regression

```python
# Compare different regularization strengths
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, C in enumerate(C_values):
    # Fit model with different regularization
    log_reg_C = LogisticRegression(C=C, random_state=42)
    log_reg_C.fit(X_train, y_train)
    
    # Plot decision boundary
    ax = axes[idx]
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = log_reg_C.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=20)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu,
               edgecolor='black', s=30)
    
    # Calculate accuracy
    accuracy = log_reg_C.score(X_test, y_test)
    ax.set_title(f'C = {C} (λ = {1/C:.3f})\nAccuracy: {accuracy:.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.suptitle('Effect of Regularization in Logistic Regression\n(C = 1/λ, smaller C = stronger regularization)', 
             fontsize=16)
plt.tight_layout()
plt.show()

# Compare L1 vs L2 regularization
from sklearn.preprocessing import StandardScaler

# Generate high-dimensional data
X_high_dim, y_high_dim = make_classification(n_samples=200, n_features=20,
                                            n_informative=5, n_redundant=15,
                                            random_state=42)

# Standardize features
scaler = StandardScaler()
X_high_scaled = scaler.fit_transform(X_high_dim)

# Split data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_high_scaled, y_high_dim,
                                                            test_size=0.3, random_state=42)

# Compare L1 and L2
penalties = ['l1', 'l2']
C_values = np.logspace(-3, 2, 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for penalty in penalties:
    train_scores = []
    test_scores = []
    n_nonzero_coefs = []
    
    for C in C_values:
        if penalty == 'l1':
            solver = 'liblinear'
        else:
            solver = 'lbfgs'
            
        log_reg = LogisticRegression(penalty=penalty, C=C, solver=solver, 
                                   max_iter=1000, random_state=42)
        log_reg.fit(X_train_h, y_train_h)
        
        train_scores.append(log_reg.score(X_train_h, y_train_h))
        test_scores.append(log_reg.score(X_test_h, y_test_h))
        n_nonzero_coefs.append(np.sum(np.abs(log_reg.coef_[0]) > 1e-5))
    
    # Plot accuracy
    ax1.semilogx(C_values, train_scores, label=f'{penalty} (train)', 
                linewidth=2, alpha=0.7)
    ax1.semilogx(C_values, test_scores, label=f'{penalty} (test)', 
                linewidth=2)
    
    # Plot number of features
    ax2.semilogx(C_values, n_nonzero_coefs, label=penalty, linewidth=2)

ax1.set_xlabel('C (inverse regularization)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance vs Regularization')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('C (inverse regularization)')
ax2.set_ylabel('Number of non-zero coefficients')
ax2.set_title('Feature Selection Effect')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Day 13: Naive Bayes

### Bayes' Theorem and Naive Bayes

**Bayes' Theorem:**
```
P(y|X) = P(X|y) × P(y) / P(X)
```

**Naive Assumption:**
Features are conditionally independent given the class:
```
P(X|y) = ∏ᵢ P(xᵢ|y)
```

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Gaussian Naive Bayes for continuous features
def gaussian_naive_bayes_demo():
    """Demonstrate Gaussian Naive Bayes"""
    
    # Generate data with clear class separation
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              class_sep=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Fit Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # Visualize class distributions and decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Class distributions
    ax = axes[0]
    for class_idx in [0, 1]:
        mask = y_train == class_idx
        ax.scatter(X_train[mask, 0], X_train[mask, 1], 
                  label=f'Class {class_idx}', alpha=0.6, s=50)
        
        # Plot Gaussian contours
        mean = gnb.theta_[class_idx]
        var = gnb.var_[class_idx]
        
        # Create ellipse for 2 standard deviations
        from matplotlib.patches import Ellipse
        ell = Ellipse(mean, 2*np.sqrt(var[0]), 2*np.sqrt(var[1]),
                     alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(ell)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Gaussian Naive Bayes: Class Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Decision boundary
    ax = axes[1]
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=20)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu,
               edgecolor='black', s=50)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Decision Boundary (Accuracy: {gnb.score(X_test, y_test):.3f})')
    
    plt.tight_layout()
    plt.show()
    
    # Print model parameters
    print("Gaussian Naive Bayes Parameters:")
    print(f"Class priors: {gnb.class_prior_}")
    print(f"Class means:\n{gnb.theta_}")
    print(f"Class variances:\n{gnb.var_}")
    
    return gnb

gnb_model = gaussian_naive_bayes_demo()
```

### Text Classification with Naive Bayes

```python
# Text classification example
def text_classification_demo():
    """Demonstrate text classification with Naive Bayes"""
    
    # Load a subset of 20 newsgroups dataset
    categories = ['sci.space', 'comp.graphics', 'rec.sport.baseball']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                         remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                        remove=('headers', 'footers', 'quotes'))
    
    # Vectorize text data
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X_train_counts = vectorizer.fit_transform(newsgroups_train.data)
    X_test_counts = vectorizer.transform(newsgroups_test.data)
    
    # Train Multinomial Naive Bayes
    mnb = MultinomialNB(alpha=0.1)
    mnb.fit(X_train_counts, newsgroups_train.target)
    
    # Predictions
    y_pred = mnb.predict(X_test_counts)
    
    # Evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Text Classification Results:")
    print("\nClassification Report:")
    print(classification_report(newsgroups_test.target, y_pred,
                              target_names=categories))
    
    # Confusion Matrix
    cm = confusion_matrix(newsgroups_test.target, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix - Text Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance (top words per class)
    feature_names = vectorizer.get_feature_names_out()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (category, ax) in enumerate(zip(categories, axes)):
        # Get top features for this class
        class_log_prob = mnb.feature_log_prob_[idx]
        top_indices = np.argsort(class_log_prob)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_probs = np.exp(class_log_prob[top_indices])
        
        ax.barh(range(len(top_features)), top_probs)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Probability')
        ax.set_title(f'Top Words: {category}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mnb, vectorizer

mnb_model, vectorizer = text_classification_demo()
```

### Comparing Naive Bayes Variants

```python
def compare_naive_bayes_variants():
    """Compare different Naive Bayes implementations"""
    
    # Generate different types of data
    n_samples = 1000
    
    # 1. Continuous data (Gaussian)
    X_continuous, y_continuous = make_classification(n_samples=n_samples, 
                                                    n_features=4, 
                                                    n_informative=3,
                                                    random_state=42)
    
    # 2. Count data (Multinomial)
    X_counts = np.random.poisson(3, size=(n_samples, 10))
    y_counts = (X_counts.sum(axis=1) > 30).astype(int)
    
    # 3. Binary data (Bernoulli)
    X_binary = np.random.binomial(1, 0.3, size=(n_samples, 10))
    y_binary = (X_binary.sum(axis=1) > 3).astype(int)
    
    # Compare models
    datasets = [
        ('Continuous Features', X_continuous, y_continuous, GaussianNB()),
        ('Count Features', X_counts, y_counts, MultinomialNB()),
        ('Binary Features', X_binary, y_binary, BernoulliNB())
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, X, y, model) in enumerate(datasets):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42)
        
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # ROC curve
        ax = axes[0, idx]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}\n{type(model).__name__}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Feature distributions
        ax = axes[1, idx]
        if name == 'Continuous Features':
            # Show first two features
            for class_idx in [0, 1]:
                mask = y_train == class_idx
                ax.scatter(X_train[mask, 0], X_train[mask, 1],
                          label=f'Class {class_idx}', alpha=0.6)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        else:
            # Show feature means
            feature_means_0 = X_train[y_train == 0].mean(axis=0)
            feature_means_1 = X_train[y_train == 1].mean(axis=0)
            
            x_pos = np.arange(len(feature_means_0))
            width = 0.35
            
            ax.bar(x_pos - width/2, feature_means_0, width, label='Class 0', alpha=0.7)
            ax.bar(x_pos + width/2, feature_means_1, width, label='Class 1', alpha=0.7)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Mean Value')
            ax.set_xticks(x_pos)
        
        ax.legend()
        ax.set_title('Feature Analysis')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_naive_bayes_variants()
```

### Naive Bayes Assumptions and Limitations

```python
def naive_bayes_assumptions_demo():
    """Demonstrate the impact of violating Naive Bayes assumptions"""
    
    # Create datasets with different levels of feature correlation
    n_samples = 500
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    correlation_levels = [0.0, 0.5, 0.9]
    
    for idx, correlation in enumerate(correlation_levels):
        # Generate correlated features
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        
        # Class 0
        X_0 = np.random.multivariate_normal(mean, cov, n_samples // 2)
        
        # Class 1 (shifted)
        mean_1 = [2, 2]
        X_1 = np.random.multivariate_normal(mean_1, cov, n_samples // 2)
        
        X = np.vstack([X_0, X_1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42)
        
        # Fit Naive Bayes and Logistic Regression
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)
        
        # Plot data
        ax = axes[0, idx]
        ax.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.6, label='Class 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.6, label='Class 1')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Feature Correlation: {correlation}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compare accuracies
        ax = axes[1, idx]
        nb_acc = gnb.score(X_test, y_test)
        lr_acc = log_reg.score(X_test, y_test)
        
        models = ['Naive Bayes', 'Logistic Reg']
        accuracies = [nb_acc, lr_acc]
        colors = ['blue', 'green']
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Model Comparison\nCorrelation: {correlation}')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Feature Correlation on Naive Bayes Performance', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\nNaive Bayes Assumptions and When to Use:")
    print("\nAssumptions:")
    print("1. Features are conditionally independent given the class")
    print("2. Each feature contributes independently to the probability")
    print("\nWhen to use Naive Bayes:")
    print("✓ Text classification")
    print("✓ High-dimensional sparse data")
    print("✓ Small training datasets")
    print("✓ Need for fast training/prediction")
    print("✓ When independence assumption approximately holds")
    print("\nWhen to avoid:")
    print("✗ Highly correlated features")
    print("✗ Complex feature interactions")
    print("✗ Need for calibrated probabilities")

naive_bayes_assumptions_demo()
```

## Day 14: Practice Session - Regression & Metrics

### Comprehensive Regression Project

```python
def regression_practice_project():
    """Complete regression project with all concepts from Week 2"""
    
    # Generate complex dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_informative = 5
    
    # Create dataset with some correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Add correlation between features
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] - 0.5 * np.random.randn(n_samples)
    
    # True coefficients (sparse)
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = np.random.randn(n_informative) * 2
    
    # Generate target with non-linear component
    y = X @ true_coef
    y += 0.5 * X[:, 0]**2  # Non-linear term
    y += np.random.normal(0, 0.5, n_samples)  # Noise
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare different models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Polynomial (deg=2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
    }
    
    results = []
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        # Handle polynomial features separately
        if 'Polynomial' in name:
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results.append({
            'Model': name,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R²': train_r2,
            'Test R²': test_r2
        })
        
        # Plot actual vs predicted
        ax = axes[idx]
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=20)
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\nTest R² = {test_r2:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Summary plot
    ax = axes[5]
    results_df = pd.DataFrame(results)
    
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['Train R²'], width, label='Train R²', alpha=0.7)
    ax.bar(x + width/2, results_df['Test R²'], width, label='Test R²', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    
    # Feature importance analysis
    print("\n\nFeature Importance Analysis:")
    
    # For linear models with direct coefficients
    linear_models = ['Ridge', 'Lasso', 'ElasticNet']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model_name in enumerate(linear_models):
        model = models[model_name]
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            
            ax = axes[idx]
            ax.bar(range(len(coefs)), coefs)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Coefficient')
            ax.set_title(f'{model_name} Coefficients')
            ax.grid(True, alpha=0.3)
            
            # Mark true non-zero coefficients
            for i in range(n_informative):
                ax.axvline(x=i, color='red', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Run the practice project
practice_results = regression_practice_project()

# Quiz questions
quiz = """
Week 2 Review Quiz:

1. What is the main difference between Ridge and Lasso regression?
   Answer: Ridge uses L2 penalty (squares of coefficients), Lasso uses L1 penalty 
   (absolute values). Lasso can shrink coefficients to exactly zero (feature selection).

2. When would you use gradient descent instead of the normal equation?
   Answer: When you have a large number of features (normal equation requires matrix 
   inversion which is O(n³)), or when you need online/incremental learning.

3. What does a negative R² value indicate?
   Answer: The model performs worse than a horizontal line at the mean of y. This 
   usually indicates a very poor model fit.

4. Why do we use log-odds (logit) in logistic regression?
   Answer: To transform the bounded probability [0,1] to an unbounded range (-∞,+∞), 
   allowing us to use linear regression techniques.

5. What is the "naive" assumption in Naive Bayes?
   Answer: Features are conditionally independent given the class label.

6. How does the learning rate affect gradient descent convergence?
   Answer: Too high: may overshoot and diverge. Too low: very slow convergence. 
   Optimal rate balances speed and stability.

7. What is the bias-variance tradeoff?
   Answer: Simple models have high bias (underfitting), complex models have high 
   variance (overfitting). We seek the sweet spot that minimizes total error.

8. When should you use MAE vs MSE as a loss function?
   Answer: MAE when you want robustness to outliers (equal penalty for all errors). 
   MSE when large errors are particularly bad (quadratic penalty).
"""

print(quiz)
```

This completes Week 2 of your ML interview preparation covering:

1. **Day 8**: Linear Regression fundamentals and Gradient Descent optimization
2. **Day 9**: Ridge and Lasso regularization techniques  
3. **Day 10**: Overfitting/Underfitting concepts and Bias-Variance tradeoff
4. **Day 11**: Comprehensive regression metrics (MSE, MAE, R², Adjusted R²)
5. **Day 12**: Logistic Regression for classification, ROC curves, and AUC
6. **Day 13**: Naive Bayes classifiers and their applications
7. **Day 14**: Practice session integrating all Week 2 concepts

Each topic includes mathematical formulations, detailed implementations, visualizations, and practical examples with advanced techniques.