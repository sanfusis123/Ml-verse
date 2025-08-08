# Day 14: Regression Practice - Comprehensive Problem Solving

## 📚 Table of Contents
1. [Introduction and Overview](#introduction)
2. [Practice Problem 1: Boston Housing Price Prediction](#problem1)
3. [Practice Problem 2: Polynomial Regression with Regularization](#problem2)
4. [Practice Problem 3: Multi-Output Regression](#problem3)
5. [Practice Problem 4: Time Series Regression](#problem4)
6. [Practice Problem 5: Feature Engineering Challenge](#problem5)
7. [Common Pitfalls and Solutions](#pitfalls)
8. [Comprehensive Interview Q&A](#interview-qa)

---

## 1. Introduction and Overview {#introduction}

### Today's Objectives

This practice session consolidates all regression concepts:
- Linear Regression (Day 8)
- Ridge and Lasso Regression (Day 9)
- Overfitting/Underfitting (Day 10)
- Evaluation Metrics (Day 11)

### Skills to Practice

1. **Data Preprocessing**: Handling missing values, scaling, encoding
2. **Feature Engineering**: Creating polynomial features, interactions
3. **Model Selection**: Choosing between Linear, Ridge, Lasso, Elastic Net
4. **Hyperparameter Tuning**: Finding optimal regularization strength
5. **Evaluation**: Using appropriate metrics for different problems
6. **Visualization**: Learning curves, residual plots, feature importance

### Approach for Each Problem

1. **Understand the data**: EDA and visualization
2. **Preprocess**: Handle missing values, scale features
3. **Baseline model**: Simple linear regression
4. **Improve**: Add regularization, polynomial features
5. **Evaluate**: Multiple metrics, cross-validation
6. **Interpret**: Coefficient analysis, feature importance

---

## 2. Practice Problem 1: Boston Housing Price Prediction {#problem1}

### Problem Statement

Predict median home values using the Boston Housing dataset with multiple regression techniques.

### Complete Solution

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load and explore data
boston = load_boston()
X, y = boston.data, boston.target
feature_names = boston.feature_names

# Create DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['PRICE'] = y

print("Dataset shape:", df.shape)
print("\nFeature statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:", df.isnull().sum().sum())

# Visualize target distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(y, bins=30, edgecolor='black')
plt.xlabel('Price ($1000s)')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')

# Correlation heatmap
plt.subplot(1, 3, 2)
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0)
plt.title('Feature Correlations')

# Feature importance (correlation with target)
plt.subplot(1, 3, 3)
target_corr = df.corr()['PRICE'].drop('PRICE').sort_values()
target_corr.plot(kind='barh')
plt.xlabel('Correlation with Price')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Baseline Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\n=== Linear Regression Results ===")
print(f"R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.4f}")

# 2. Ridge Regression with CV
ridge = Ridge()
ridge_params = {'alpha': np.logspace(-3, 3, 50)}
ridge_cv = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_cv.predict(X_test_scaled)

print("\n=== Ridge Regression Results ===")
print(f"Best alpha: {ridge_cv.best_params_['alpha']:.4f}")
print(f"R² Score: {r2_score(y_test, y_pred_ridge):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")

# 3. Lasso Regression with CV
lasso = Lasso(max_iter=10000)
lasso_params = {'alpha': np.logspace(-3, 1, 50)}
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_cv.predict(X_test_scaled)

print("\n=== Lasso Regression Results ===")
print(f"Best alpha: {lasso_cv.best_params_['alpha']:.4f}")
print(f"R² Score: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")
print(f"Features eliminated: {np.sum(lasso_cv.best_estimator_.coef_ == 0)}")

# 4. Elastic Net
elastic = ElasticNet(max_iter=10000)
elastic_params = {
    'alpha': np.logspace(-3, 1, 20),
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic_cv = GridSearchCV(elastic, elastic_params, cv=5, scoring='neg_mean_squared_error')
elastic_cv.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_cv.predict(X_test_scaled)

print("\n=== Elastic Net Results ===")
print(f"Best alpha: {elastic_cv.best_params_['alpha']:.4f}")
print(f"Best l1_ratio: {elastic_cv.best_params_['l1_ratio']:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_elastic):.4f}")

# Coefficient comparison
plt.figure(figsize=(12, 6))
models = {
    'Linear': lr.coef_,
    'Ridge': ridge_cv.best_estimator_.coef_,
    'Lasso': lasso_cv.best_estimator_.coef_,
    'Elastic Net': elastic_cv.best_estimator_.coef_
}

x_pos = np.arange(len(feature_names))
width = 0.2

for i, (name, coefs) in enumerate(models.items()):
    plt.bar(x_pos + i*width, coefs, width, label=name)

plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Comparison Across Models')
plt.xticks(x_pos + width*1.5, feature_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Residual analysis
plt.figure(figsize=(12, 4))
for i, (name, pred) in enumerate([('Linear', y_pred_lr), 
                                   ('Ridge', y_pred_ridge), 
                                   ('Lasso', y_pred_lasso)]):
    plt.subplot(1, 3, i+1)
    residuals = y_test - pred
    plt.scatter(pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{name} Regression Residuals')
plt.tight_layout()
plt.show()

# Learning curves
def plot_learning_curves(model, X, y, title):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        n_samples = int(size * len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        scores = cross_val_score(model, X_subset, y_subset, cv=5, 
                                scoring='neg_mean_squared_error')
        train_score = -model.fit(X_subset, y_subset).score(X_subset, y_subset)
        
        train_scores.append(train_score)
        val_scores.append(-scores.mean())
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * len(X), train_scores, 'o-', label='Training score')
    plt.plot(train_sizes * len(X), val_scores, 'o-', label='Validation score')
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.title(f'Learning Curves - {title}')
    plt.legend()
    plt.show()

# Plot learning curves for Ridge
plot_learning_curves(ridge_cv.best_estimator_, X_train_scaled, y_train, 'Ridge Regression')
```

### Key Insights from Problem 1

1. **Feature Scaling**: Essential for regularized models
2. **Hyperparameter Tuning**: GridSearchCV finds optimal regularization
3. **Model Comparison**: Ridge often best for correlated features
4. **Residual Analysis**: Check for heteroscedasticity
5. **Learning Curves**: Diagnose over/underfitting

---

## 3. Practice Problem 2: Polynomial Regression with Regularization {#problem2}

### Problem Statement

Generate synthetic non-linear data and compare polynomial regression with different regularization techniques.

### Complete Solution

```python
# Generate non-linear synthetic data
np.random.seed(42)
n_samples = 200
X = np.sort(np.random.uniform(-3, 3, n_samples))
y_true = 0.5 * X**3 - 2 * X**2 + X + 2
y = y_true + np.random.normal(0, 3, n_samples)  # Add noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y, test_size=0.3, random_state=42
)

# Create polynomial features of different degrees
degrees = [1, 3, 5, 10, 15]
alphas = [0, 0.01, 0.1, 1, 10]

results = {}

plt.figure(figsize=(15, 10))
plot_idx = 1

for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # Train models with different regularization
    for alpha in alphas:
        if alpha == 0:
            model = LinearRegression()
            model_name = f'Poly-{degree}'
        else:
            model = Ridge(alpha=alpha)
            model_name = f'Poly-{degree}-Ridge-{alpha}'
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Store results
        results[model_name] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Plot for selected combinations
        if degree in [1, 3, 10] and alpha in [0, 0.1, 10]:
            plt.subplot(3, 3, plot_idx)
            
            # Plot data
            plt.scatter(X_train, y_train, alpha=0.5, label='Train')
            plt.scatter(X_test, y_test, alpha=0.5, label='Test')
            
            # Plot predictions
            X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
            X_plot_poly = poly.transform(X_plot)
            X_plot_scaled = scaler.transform(X_plot_poly)
            y_plot = model.predict(X_plot_scaled)
            
            plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Prediction')
            plt.plot(X_plot, 0.5 * X_plot**3 - 2 * X_plot**2 + X_plot + 2, 
                    'g--', linewidth=1, label='True function')
            
            plt.title(f'Degree={degree}, α={alpha}')
            plt.xlabel('X')
            plt.ylabel('Y')
            if plot_idx == 1:
                plt.legend()
            
            plot_idx += 1

plt.tight_layout()
plt.show()

# Results summary
results_df = pd.DataFrame(results).T
results_df['overfit_score'] = results_df['train_r2'] - results_df['test_r2']

print("\n=== Model Performance Summary ===")
print(results_df.sort_values('test_rmse').head(10))

# Visualize bias-variance tradeoff
plt.figure(figsize=(12, 5))

# Plot 1: RMSE vs Polynomial Degree
plt.subplot(1, 2, 1)
for alpha in [0, 0.1, 1]:
    train_rmse = []
    test_rmse = []
    for degree in degrees:
        if alpha == 0:
            key = f'Poly-{degree}'
        else:
            key = f'Poly-{degree}-Ridge-{alpha}'
        if key in results:
            train_rmse.append(results[key]['train_rmse'])
            test_rmse.append(results[key]['test_rmse'])
    
    plt.plot(degrees[:len(train_rmse)], train_rmse, 'o--', label=f'Train (α={alpha})')
    plt.plot(degrees[:len(test_rmse)], test_rmse, 'o-', label=f'Test (α={alpha})')

plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.title('Model Complexity vs Error')
plt.legend()

# Plot 2: Regularization effect
plt.subplot(1, 2, 2)
for degree in [3, 5, 10]:
    test_rmse = []
    for alpha in alphas:
        if alpha == 0:
            key = f'Poly-{degree}'
        else:
            key = f'Poly-{degree}-Ridge-{alpha}'
        test_rmse.append(results[key]['test_rmse'])
    
    plt.plot(alphas, test_rmse, 'o-', label=f'Degree={degree}')

plt.xlabel('Regularization Strength (α)')
plt.ylabel('Test RMSE')
plt.xscale('log')
plt.title('Regularization Effect on Test Error')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 4. Practice Problem 3: Multi-Output Regression {#problem3}

### Problem Statement

Predict multiple correlated outputs simultaneously using multi-output regression techniques.

### Complete Solution

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Generate multi-output data
np.random.seed(42)
n_samples = 500
n_features = 10
n_outputs = 3

# Generate features
X = np.random.randn(n_samples, n_features)

# Create correlated outputs
W1 = np.random.randn(n_features, n_outputs)
W2 = np.random.randn(n_features, n_outputs) * 0.5
y = X @ W1 + (X**2) @ W2 + np.random.randn(n_samples, n_outputs) * 0.5

# Add some correlation between outputs
y[:, 1] += 0.5 * y[:, 0]
y[:, 2] += 0.3 * y[:, 0] + 0.4 * y[:, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Independent models for each output
print("=== Independent Models ===")
independent_scores = []

for i in range(n_outputs):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train[:, i])
    y_pred = ridge.predict(X_test_scaled)
    score = r2_score(y_test[:, i], y_pred)
    independent_scores.append(score)
    print(f"Output {i+1} R² Score: {score:.4f}")

# Method 2: Multi-output regression
print("\n=== Multi-Output Regression ===")
multi_ridge = MultiOutputRegressor(Ridge(alpha=1.0))
multi_ridge.fit(X_train_scaled, y_train)
y_pred_multi = multi_ridge.predict(X_test_scaled)

for i in range(n_outputs):
    score = r2_score(y_test[:, i], y_pred_multi[:, i])
    print(f"Output {i+1} R² Score: {score:.4f}")

# Method 3: Single model predicting all outputs (Ridge can do this natively)
print("\n=== Native Multi-Output Ridge ===")
ridge_multi = Ridge(alpha=1.0)
ridge_multi.fit(X_train_scaled, y_train)
y_pred_native = ridge_multi.predict(X_test_scaled)

for i in range(n_outputs):
    score = r2_score(y_test[:, i], y_pred_native[:, i])
    print(f"Output {i+1} R² Score: {score:.4f}")

# Visualize predictions vs actual
fig, axes = plt.subplots(n_outputs, 3, figsize=(12, 3*n_outputs))

methods = ['Independent', 'MultiOutput', 'Native']
predictions = [
    np.column_stack([ridge.predict(X_test_scaled) for ridge in 
                     [Ridge(alpha=1.0).fit(X_train_scaled, y_train[:, i]) 
                      for i in range(n_outputs)]]),
    y_pred_multi,
    y_pred_native
]

for i in range(n_outputs):
    for j, (method, pred) in enumerate(zip(methods, predictions)):
        ax = axes[i, j] if n_outputs > 1 else axes[j]
        ax.scatter(y_test[:, i], pred[:, i], alpha=0.5)
        ax.plot([y_test[:, i].min(), y_test[:, i].max()], 
                [y_test[:, i].min(), y_test[:, i].max()], 'r--')
        ax.set_xlabel(f'True Output {i+1}')
        ax.set_ylabel(f'Predicted Output {i+1}')
        ax.set_title(f'{method} Method')
        
        # Add R² score
        r2 = r2_score(y_test[:, i], pred[:, i])
        ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes, 
                verticalalignment='top')

plt.tight_layout()
plt.show()

# Analyze output correlations
print("\n=== Output Correlations ===")
print("True outputs correlation:")
print(pd.DataFrame(y_test).corr())
print("\nPredicted outputs correlation (Native Multi-Output):")
print(pd.DataFrame(y_pred_native).corr())
```

---

## 5. Practice Problem 4: Time Series Regression {#problem4}

### Problem Statement

Apply regression techniques to time series data with feature engineering.

### Complete Solution

```python
# Generate time series data
np.random.seed(42)
n_points = 365 * 2  # 2 years of daily data
time = np.arange(n_points)

# Create components
trend = 0.05 * time
seasonal = 10 * np.sin(2 * np.pi * time / 365.25)  # Yearly seasonality
weekly = 3 * np.sin(2 * np.pi * time / 7)  # Weekly seasonality
noise = np.random.normal(0, 2, n_points)

# Combine components
y = 50 + trend + seasonal + weekly + noise

# Create time series features
def create_time_features(time):
    features = pd.DataFrame()
    features['time'] = time
    features['day_of_year'] = time % 365
    features['day_of_week'] = time % 7
    features['month'] = (time % 365) // 30
    features['sin_yearly'] = np.sin(2 * np.pi * time / 365.25)
    features['cos_yearly'] = np.cos(2 * np.pi * time / 365.25)
    features['sin_weekly'] = np.sin(2 * np.pi * time / 7)
    features['cos_weekly'] = np.cos(2 * np.pi * time / 7)
    return features

X = create_time_features(time)

# Create lag features
for lag in [1, 7, 30]:
    X[f'lag_{lag}'] = pd.Series(y).shift(lag).fillna(method='bfill')

# Train-test split (time series split)
split_point = int(0.8 * len(time))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Polynomial': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10))
    ])
}

results = {}
predictions = {}

for name, model in models.items():
    if name == 'Polynomial':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    results[name] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    predictions[name] = y_pred

# Visualize results
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Original time series with predictions
ax1 = axes[0]
ax1.plot(time[:split_point], y_train, label='Training Data', alpha=0.7)
ax1.plot(time[split_point:], y_test, label='Test Data', alpha=0.7)

for name, pred in predictions.items():
    if name in ['Linear', 'Ridge']:  # Show only selected models
        ax1.plot(time[split_point:], pred, '--', label=f'{name} Prediction')

ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_title('Time Series Predictions')
ax1.legend()

# Plot 2: Residuals over time
ax2 = axes[1]
for name, pred in predictions.items():
    residuals = y_test - pred
    ax2.plot(time[split_point:], residuals, label=f'{name} Residuals')

ax2.axhline(y=0, color='black', linestyle='--')
ax2.set_xlabel('Time')
ax2.set_ylabel('Residuals')
ax2.set_title('Model Residuals Over Time')
ax2.legend()

# Plot 3: Feature importance (Ridge coefficients)
ax3 = axes[2]
ridge_model = models['Ridge']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': ridge_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

feature_importance.plot(x='feature', y='coefficient', kind='barh', ax=ax3)
ax3.set_xlabel('Coefficient Value')
ax3.set_title('Feature Importance (Ridge Regression)')

plt.tight_layout()
plt.show()

# Performance summary
print("\n=== Model Performance Summary ===")
performance_df = pd.DataFrame(results).T
print(performance_df.round(4))

# Forecast future values
future_time = np.arange(n_points, n_points + 30)  # 30 days forecast
X_future = create_time_features(future_time)

# Use last known values for lag features
for lag in [1, 7, 30]:
    if lag == 1:
        X_future[f'lag_{lag}'] = y[-1]
    else:
        X_future[f'lag_{lag}'] = y[-lag]

X_future_scaled = scaler.transform(X_future)
future_pred = models['Ridge'].predict(X_future_scaled)

plt.figure(figsize=(10, 5))
plt.plot(time[-60:], y[-60:], label='Historical')
plt.plot(future_time, future_pred, 'r--', label='Forecast')
plt.fill_between(future_time, 
                 future_pred - 2*results['Ridge']['rmse'],
                 future_pred + 2*results['Ridge']['rmse'],
                 alpha=0.3, color='red', label='Confidence Interval')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('30-Day Forecast')
plt.legend()
plt.show()
```

---

## 6. Practice Problem 5: Feature Engineering Challenge {#problem5}

### Problem Statement

Create a comprehensive feature engineering pipeline for a complex regression problem.

### Complete Solution

```python
# Generate complex synthetic dataset
np.random.seed(42)
n_samples = 1000

# Base features
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
X3 = np.random.exponential(1, n_samples)
X4 = np.random.uniform(-2, 2, n_samples)
X5 = np.random.choice([0, 1], n_samples)  # Binary feature

# Create target with complex relationships
y = (
    2 * X1 +                    # Linear
    3 * X2**2 +                 # Quadratic
    np.sin(X3) +                # Non-linear
    X1 * X2 +                   # Interaction
    4 * X4 * X5 +               # Conditional effect
    np.exp(-X1**2) +            # Gaussian-like
    np.random.normal(0, 0.5, n_samples)  # Noise
)

# Create DataFrame
df = pd.DataFrame({
    'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'y': y
})

# Feature engineering pipeline
class FeatureEngineer:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        features = pd.DataFrame()
        
        # Original features
        for col in ['X1', 'X2', 'X3', 'X4', 'X5']:
            features[col] = df[col]
        
        # Mathematical transformations
        features['X1_squared'] = df['X1']**2
        features['X2_squared'] = df['X2']**2
        features['X3_log'] = np.log1p(np.abs(df['X3']))
        features['X4_abs'] = np.abs(df['X4'])
        
        # Trigonometric features
        features['X3_sin'] = np.sin(df['X3'])
        features['X3_cos'] = np.cos(df['X3'])
        features['X4_sin'] = np.sin(df['X4'])
        
        # Interaction features
        features['X1_X2'] = df['X1'] * df['X2']
        features['X1_X3'] = df['X1'] * df['X3']
        features['X2_X3'] = df['X2'] * df['X3']
        features['X4_X5'] = df['X4'] * df['X5']
        
        # Conditional features
        features['X1_positive'] = (df['X1'] > 0).astype(int)
        features['X2_positive'] = (df['X2'] > 0).astype(int)
        features['X1_X2_same_sign'] = ((df['X1'] > 0) == (df['X2'] > 0)).astype(int)
        
        # Binning continuous features
        features['X1_bin'] = pd.cut(df['X1'], bins=5, labels=False)
        features['X3_bin'] = pd.qcut(df['X3'], q=5, labels=False)
        
        # Exponential features
        features['X1_exp'] = np.exp(-df['X1']**2)
        features['X2_exp'] = np.exp(-df['X2']**2)
        
        return features

# Create features
engineer = FeatureEngineer()
X_engineered = engineer.create_features(df)

print(f"Original features: 5")
print(f"Engineered features: {X_engineered.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models with different feature sets
results = {}

# 1. Original features only
X_train_orig = X_train[['X1', 'X2', 'X3', 'X4', 'X5']]
X_test_orig = X_test[['X1', 'X2', 'X3', 'X4', 'X5']]
scaler_orig = StandardScaler()
X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig)
X_test_orig_scaled = scaler_orig.transform(X_test_orig)

ridge_orig = Ridge(alpha=1.0)
ridge_orig.fit(X_train_orig_scaled, y_train)
y_pred_orig = ridge_orig.predict(X_test_orig_scaled)
results['Original Features'] = {
    'r2': r2_score(y_test, y_pred_orig),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_orig))
}

# 2. All engineered features
ridge_all = Ridge(alpha=1.0)
ridge_all.fit(X_train_scaled, y_train)
y_pred_all = ridge_all.predict(X_test_scaled)
results['All Features'] = {
    'r2': r2_score(y_test, y_pred_all),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_all))
}

# 3. Feature selection with Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
selected_features = X_engineered.columns[lasso.coef_ != 0]
print(f"\nLasso selected {len(selected_features)} features:")
print(selected_features.tolist())

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
scaler_selected = StandardScaler()
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
X_test_selected_scaled = scaler_selected.transform(X_test_selected)

ridge_selected = Ridge(alpha=1.0)
ridge_selected.fit(X_train_selected_scaled, y_train)
y_pred_selected = ridge_selected.predict(X_test_selected_scaled)
results['Lasso Selected'] = {
    'r2': r2_score(y_test, y_pred_selected),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_selected))
}

# 4. Recursive Feature Elimination
from sklearn.feature_selection import RFE

rfe = RFE(Ridge(alpha=1.0), n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)
X_train_rfe = X_train_scaled[:, rfe.support_]
X_test_rfe = X_test_scaled[:, rfe.support_]

ridge_rfe = Ridge(alpha=1.0)
ridge_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = ridge_rfe.predict(X_test_rfe)
results['RFE Selected'] = {
    'r2': r2_score(y_test, y_pred_rfe),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rfe))
}

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Feature importance
ax1 = axes[0, 0]
feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'coefficient': ridge_all.coef_
}).sort_values('coefficient', key=abs, ascending=False).head(15)

feature_importance.plot(x='feature', y='coefficient', kind='barh', ax=ax1)
ax1.set_xlabel('Coefficient Value')
ax1.set_title('Top 15 Feature Importance')

# Model comparison
ax2 = axes[0, 1]
results_df = pd.DataFrame(results).T
results_df.plot(kind='bar', ax=ax2)
ax2.set_ylabel('Score')
ax2.set_title('Model Performance Comparison')
ax2.legend(['R²', 'RMSE'])

# Actual vs Predicted
ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred_all, alpha=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
ax3.set_title('Actual vs Predicted (All Features)')

# Feature correlation heatmap
ax4 = axes[1, 1]
top_features = feature_importance.head(10)['feature'].values
corr_matrix = X_engineered[top_features].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax4)
ax4.set_title('Top Features Correlation')

plt.tight_layout()
plt.show()

print("\n=== Results Summary ===")
print(pd.DataFrame(results))
```

---

## 7. Common Pitfalls and Solutions {#pitfalls}

### 1. Data Leakage

**Problem**: Using future information in training data
```python
# Wrong: Creating lag features before split
df['lag_1'] = df['target'].shift(1)
X_train, X_test = train_test_split(df)

# Correct: Creating lag features after split
X_train, X_test = train_test_split(df)
X_train['lag_1'] = X_train['target'].shift(1)
X_test['lag_1'] = X_test['target'].shift(1)
```

### 2. Not Scaling Features

**Problem**: Regularization penalizes large-scale features more
```python
# Always scale for regularized models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

### 3. Ignoring Multicollinearity

**Problem**: Unstable coefficients with correlated features
```python
# Check VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif[vif['VIF'] > 10])  # Features with high multicollinearity
```

### 4. Wrong Cross-Validation for Time Series

**Problem**: Random splits break temporal order
```python
from sklearn.model_selection import TimeSeriesSplit

# Correct for time series
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

### 5. Over-regularization

**Problem**: Too much regularization leads to underfitting
```python
# Always use cross-validation to find optimal alpha
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Optimal alpha: {ridge_cv.alpha_}")
```

---

## 8. Comprehensive Interview Q&A {#interview-qa}

### Q1: When should you use Ridge vs Lasso regression?

**Answer:**
- **Ridge**: When you believe most features are relevant, dealing with multicollinearity, or want stable predictions
- **Lasso**: When you want feature selection, believe many features are irrelevant, or need interpretability
- **Elastic Net**: When you have correlated features in groups and want both selection and grouping

### Q2: How do you handle categorical variables in regression?

**Answer:**
```python
# One-hot encoding (dummy variables)
pd.get_dummies(df, columns=['category'], drop_first=True)

# Ordinal encoding for ordered categories
df['size_encoded'] = df['size'].map({'S': 1, 'M': 2, 'L': 3})

# Target encoding (be careful of overfitting)
mean_target = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(mean_target)
```

### Q3: Explain the bias-variance tradeoff in regression context.

**Answer:**
- **High Bias (Underfitting)**: Simple models like linear regression on non-linear data
- **High Variance (Overfitting)**: Complex models like high-degree polynomials
- **Regularization**: Adds bias to reduce variance
- **Optimal**: Minimize total error = bias² + variance + irreducible error

### Q4: How do you diagnose regression problems?

**Answer:**
1. **Residual plots**: Check for patterns, heteroscedasticity
2. **Q-Q plots**: Check normality of residuals
3. **Learning curves**: Diagnose over/underfitting
4. **Coefficient stability**: Check with bootstrap or cross-validation
5. **Influence plots**: Identify outliers and high-leverage points

### Q5: What's the difference between R² and Adjusted R²?

**Answer:**
- **R²**: Proportion of variance explained, always increases with more features
- **Adjusted R²**: Penalizes for number of features
- Formula: Adj R² = 1 - [(1-R²)(n-1)/(n-p-1)]
- Use Adjusted R² when comparing models with different numbers of features

### Q6: How do you handle outliers in regression?

**Answer:**
1. **Detection**: Cook's distance, leverage plots, standardized residuals
2. **Robust regression**: Huber regression, RANSAC
3. **Transformation**: Log transform can reduce outlier influence
4. **Regularization**: L1 (Lasso) is more robust to outliers than L2 (Ridge)
5. **Domain knowledge**: Understand if outliers are errors or important cases

### Q7: Explain multicollinearity and its effects.

**Answer:**
- **Definition**: High correlation between predictors
- **Effects**: Unstable coefficients, large standard errors, counterintuitive signs
- **Detection**: VIF > 10, correlation matrix, condition number
- **Solutions**: Remove features, PCA, Ridge regression, combine features

### Q8: How do you perform feature selection for regression?

**Answer:**
```python
# 1. Filter methods
correlation = df.corr()['target'].abs().sort_values(ascending=False)

# 2. Wrapper methods
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)

# 3. Embedded methods
lasso = Lasso(alpha=0.1)  # L1 regularization

# 4. Forward/Backward selection
from sklearn.feature_selection import SequentialFeatureSelector
sfs = SequentialFeatureSelector(LinearRegression(), direction='forward')
```

### Q9: What assumptions does linear regression make?

**Answer:**
1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Predictors are not highly correlated

### Q10: How do you handle missing values in regression?

**Answer:**
```python
# 1. Simple imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'

# 2. Advanced imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

# 3. Indicator for missingness
df['feature_missing'] = df['feature'].isna().astype(int)

# 4. Domain-specific
# Forward-fill for time series
df.fillna(method='ffill')
```

### Q11: Explain the difference between parametric and non-parametric regression.

**Answer:**
- **Parametric** (Linear, Ridge, Lasso):
  - Assumes functional form
  - Fixed number of parameters
  - Fast, interpretable
  - Can extrapolate

- **Non-parametric** (KNN, Random Forest):
  - No assumed form
  - Parameters grow with data
  - Flexible, can fit complex patterns
  - Cannot extrapolate well

### Q12: How do you handle non-linear relationships in regression?

**Answer:**
1. **Polynomial features**: X² X³, interactions
2. **Transformations**: Log, sqrt, inverse
3. **Splines**: Piecewise polynomials
4. **Kernel methods**: Kernel Ridge Regression
5. **Switch models**: Tree-based methods, neural networks

### Q13: What's the computational complexity of different regression methods?

**Answer:**
- **Linear Regression**: O(np²) for normal equation, O(np) per iteration for gradient descent
- **Ridge**: Same as linear regression
- **Lasso**: O(np) per iteration, more iterations due to L1
- **Elastic Net**: Similar to Lasso
- p = features, n = samples

### Q14: How do you interpret regression coefficients?

**Answer:**
- **Linear**: One unit increase in Xᵢ → βᵢ unit change in y (holding others constant)
- **Log-transformed**: Percentage changes
- **Standardized**: Compare importance across features
- **With interactions**: Effect depends on other variables
- **Confidence intervals**: Measure uncertainty

### Q15: Explain gradient descent for regression.

**Answer:**
```python
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta -= learning_rate * gradient
    
    return theta
```
- **Batch**: Use all samples
- **Stochastic**: Use one sample
- **Mini-batch**: Use subset

### Q16: How do you choose the regularization parameter?

**Answer:**
1. **Cross-validation**: Most common, unbiased estimate
2. **Information criteria**: AIC, BIC for model selection
3. **Validation set**: Fast but uses less data
4. **Regularization path**: Fit models for range of λ values
5. **Bayesian methods**: Prior on λ

### Q17: What's the difference between prediction and inference in regression?

**Answer:**
- **Prediction**: Focus on accurate ŷ, complex models OK
- **Inference**: Understanding relationships, need interpretability
- **Example**: Predicting house prices (prediction) vs understanding what drives prices (inference)

### Q18: How do you handle heteroscedasticity?

**Answer:**
1. **Detection**: Breusch-Pagan test, residual plots
2. **Transformation**: Log, square root of y
3. **Weighted Least Squares**: Give less weight to high-variance observations
4. **Robust standard errors**: Correct inference without fixing problem
5. **Generalized Least Squares**: Model the variance structure

### Q19: Explain the connection between regression and maximum likelihood.

**Answer:**
- OLS assumes: y = Xβ + ε, where ε ~ N(0, σ²)
- Log-likelihood: L(β) = -n/2 log(2πσ²) - 1/(2σ²) Σ(yᵢ - xᵢβ)²
- Maximizing likelihood = minimizing squared errors
- Regularization = adding prior on β (MAP estimation)

### Q20: How do you validate regression models?

**Answer:**
1. **Train-test split**: Simple, single estimate
2. **K-fold CV**: Better estimate, computationally intensive
3. **Leave-one-out CV**: Unbiased but high variance
4. **Time series split**: For temporal data
5. **Nested CV**: For hyperparameter tuning + evaluation
6. **Bootstrap**: Confidence intervals on predictions

### Q21: What are some advanced regression techniques?

**Answer:**
1. **Quantile Regression**: Predict percentiles, not just mean
2. **Robust Regression**: Huber, RANSAC for outliers
3. **Bayesian Regression**: Uncertainty quantification
4. **Mixed Effects Models**: Hierarchical/grouped data
5. **Gaussian Process Regression**: Non-parametric, uncertainty estimates
6. **Neural Networks**: Deep learning for complex patterns

### Q22: How do you debug a poorly performing regression model?

**Answer:**
1. **Check data quality**: Missing values, outliers, errors
2. **Visualize relationships**: Scatter plots, residual plots
3. **Feature engineering**: Create interactions, polynomials
4. **Try different models**: Linear → polynomial → non-parametric
5. **Regularization tuning**: Cross-validate hyperparameters
6. **Error analysis**: Where does model fail? Pattern in errors?
7. **More data**: Sometimes the only solution

---

## Summary

Today's practice session covered:
1. **Real-world application** of regression techniques
2. **Feature engineering** strategies
3. **Model selection** and comparison
4. **Diagnostic tools** for regression
5. **Common pitfalls** and solutions

Key takeaways:
- Always start simple, add complexity gradually
- Visualization is crucial for understanding
- Cross-validation for reliable evaluation
- Feature engineering often more important than model choice
- Consider the bias-variance tradeoff in every decision