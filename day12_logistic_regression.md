# Day 12: Logistic Regression, ROC Curve, and AUC

## 📚 Table of Contents
1. [Introduction and Theory](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Maximum Likelihood Estimation](#mle)
4. [ROC Curves and AUC](#roc-auc)
5. [Implementation from Scratch](#implementation)
6. [Advanced Topics](#advanced-topics)
7. [Comprehensive Interview Q&A](#interview-qa)

---

## 1. Introduction and Theory {#introduction}

### What is Logistic Regression?

Despite its name, logistic regression is a **classification** algorithm, not regression. It models the probability of a binary outcome using the logistic (sigmoid) function.

### Why Logistic Regression?

1. **Probabilistic output**: Provides probabilities, not just classifications
2. **Linear decision boundary**: Simple and interpretable
3. **No tuning required**: Works well out of the box
4. **Foundation for deep learning**: Logistic regression = single neuron
5. **Theoretical guarantees**: Convex optimization, unique solution

### The Sigmoid Function

The key to logistic regression is the sigmoid (logistic) function:

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties**:
- Range: (0, 1) - perfect for probabilities
- S-shaped curve
- σ(0) = 0.5
- σ'(z) = σ(z)(1 - σ(z))
- Symmetric: σ(-z) = 1 - σ(z)

### From Linear to Logistic

**Linear Regression**: y = β₀ + β₁x₁ + ... + βₚxₚ (unbounded output)

**Logistic Regression**: 
1. Linear combination: z = β₀ + β₁x₁ + ... + βₚxₚ
2. Apply sigmoid: p = σ(z) = 1 / (1 + e^(-z))
3. Interpret as probability: P(y=1|x) = p

### Odds and Log-Odds

**Odds**: ratio of probability of success to failure
```
odds = p / (1 - p)
```

**Log-odds (logit)**:
```
log(odds) = log(p / (1 - p)) = z = β₀ + β₁x₁ + ... + βₚxₚ
```

This shows logistic regression models **linear relationship in log-odds space**.

---

## 2. Mathematical Foundation {#mathematical-foundation}

### The Logistic Regression Model

For binary classification (y ∈ {0, 1}):

```
P(y=1|x; β) = σ(βᵀx) = 1 / (1 + e^(-βᵀx))
P(y=0|x; β) = 1 - σ(βᵀx) = e^(-βᵀx) / (1 + e^(-βᵀx))
```

Compact form:
```
P(y|x; β) = σ(βᵀx)^y × (1 - σ(βᵀx))^(1-y)
```

### Decision Boundary

The decision boundary is where P(y=1|x) = 0.5:
```
σ(βᵀx) = 0.5
βᵀx = 0
```

For 2D case: β₀ + β₁x₁ + β₂x₂ = 0
This is a **linear decision boundary**.

### Cost Function

We cannot use squared error (non-convex for logistic regression). Instead, use **negative log-likelihood**:

For single sample:
```
Cost(y, ŷ) = -y log(ŷ) - (1-y) log(1-ŷ)
```

Properties:
- If y = 1 and ŷ → 0: Cost → ∞
- If y = 0 and ŷ → 1: Cost → ∞
- Convex function

### Gradient Descent Update

The gradient of the cost function:
```
∂J/∂βⱼ = (1/m) Σᵢ (σ(βᵀxᵢ) - yᵢ)xᵢⱼ
```

Update rule:
```
β := β - α × (1/m) × Xᵀ(σ(Xβ) - y)
```

Remarkably similar to linear regression!

---

## 3. Maximum Likelihood Estimation {#mle}

### Likelihood Function

Given n independent samples, the likelihood is:
```
L(β) = ∏ᵢ P(yᵢ|xᵢ; β) = ∏ᵢ σ(βᵀxᵢ)^yᵢ × (1 - σ(βᵀxᵢ))^(1-yᵢ)
```

### Log-Likelihood

Taking the log (for numerical stability):
```
ℓ(β) = log L(β) = Σᵢ [yᵢ log σ(βᵀxᵢ) + (1-yᵢ) log(1 - σ(βᵀxᵢ))]
```

### MLE Solution

Maximize log-likelihood = Minimize negative log-likelihood:
```
J(β) = -(1/m) ℓ(β) = -(1/m) Σᵢ [yᵢ log σ(βᵀxᵢ) + (1-yᵢ) log(1 - σ(βᵀxᵢ))]
```

This is our cross-entropy loss function!

### Newton's Method

For faster convergence than gradient descent:
```
β_new = β_old - H⁻¹∇J
```

Where H is the Hessian matrix:
```
H_jk = ∂²J/∂βⱼ∂βₖ = (1/m) Σᵢ σ(βᵀxᵢ)(1 - σ(βᵀxᵢ))xᵢⱼxᵢₖ
```

---

## 4. ROC Curves and AUC {#roc-auc}

### Understanding ROC Curves

**ROC (Receiver Operating Characteristic)** curve plots:
- Y-axis: True Positive Rate (TPR) = Recall = TP/(TP+FN)
- X-axis: False Positive Rate (FPR) = FP/(FP+TN)

At various classification thresholds.

### Key Points on ROC Curve

1. **(0, 0)**: Threshold = 1 (classify nothing as positive)
2. **(1, 1)**: Threshold = 0 (classify everything as positive)
3. **(0, 1)**: Perfect classifier
4. **Diagonal line**: Random classifier

### AUC (Area Under the Curve)

**AUC** summarizes ROC curve into a single number:
- Range: [0, 1]
- 0.5 = Random classifier
- 1.0 = Perfect classifier
- < 0.5 = Worse than random (flip predictions)

### Interpretation of AUC

AUC has a probabilistic interpretation:
> Probability that the model ranks a random positive example higher than a random negative example.

### When to Use ROC-AUC

**Advantages**:
- Threshold-independent
- Single number for model comparison
- Robust to class imbalance (debated)

**Limitations**:
- Can be misleading for highly imbalanced data
- Doesn't reflect performance at specific operating point
- Equal weight to all thresholds

---

## 5. Implementation from Scratch {#implementation}

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix,
    log_loss, classification_report
)
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import pandas as pd
from scipy.special import expit  # Sigmoid function
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

class LogisticRegression:
    """
    Logistic Regression implementation from scratch
    Supports multiple optimization methods and regularization
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=None, lambda_reg=0.01, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.method = method
        self.costs = []
        self.theta = None
        
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, X, y, theta):
        """Calculate negative log-likelihood with optional regularization"""
        m = len(y)
        z = X @ theta
        predictions = self._sigmoid(z)
        
        # Clip predictions to prevent log(0)
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        cost = -(1/m) * np.sum(y * np.log(predictions) + 
                               (1 - y) * np.log(1 - predictions))
        
        # Add regularization
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2*m)) * np.sum(theta[1:]**2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(theta[1:]))
            
        return cost
    
    def _gradient(self, X, y, theta):
        """Calculate gradient of cost function"""
        m = len(y)
        predictions = self._sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (predictions - y)
        
        # Add regularization gradient
        if self.regularization == 'l2':
            reg_term = (self.lambda_reg / m) * theta
            reg_term[0] = 0  # Don't regularize intercept
            gradient += reg_term
        elif self.regularization == 'l1':
            reg_term = (self.lambda_reg / m) * np.sign(theta)
            reg_term[0] = 0
            gradient += reg_term
            
        return gradient
    
    def _gradient_descent(self, X, y):
        """Gradient descent optimization"""
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.n_iterations):
            # Calculate gradient
            gradient = self._gradient(X, y, self.theta)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Store cost
            cost = self._cost_function(X, y, self.theta)
            self.costs.append(cost)
            
            # Early stopping
            if i > 0 and abs(self.costs[-1] - self.costs[-2]) < 1e-7:
                print(f"Converged at iteration {i}")
                break
    
    def _newton_method(self, X, y):
        """Newton's method for faster convergence"""
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        
        for i in range(min(self.n_iterations, 20)):  # Newton converges fast
            # Predictions
            predictions = self._sigmoid(X @ self.theta)
            
            # Gradient
            gradient = self._gradient(X, y, self.theta)
            
            # Hessian
            R = np.diag(predictions * (1 - predictions))
            hessian = (1/m) * X.T @ R @ X
            
            # Add regularization to Hessian
            if self.regularization == 'l2':
                reg_matrix = (self.lambda_reg / m) * np.eye(X.shape[1])
                reg_matrix[0, 0] = 0
                hessian += reg_matrix
            
            # Newton update
            try:
                self.theta -= np.linalg.inv(hessian) @ gradient
            except np.linalg.LinAlgError:
                # If Hessian is singular, fall back to gradient descent
                self.theta -= self.learning_rate * gradient
            
            # Store cost
            cost = self._cost_function(X, y, self.theta)
            self.costs.append(cost)
    
    def fit(self, X, y):
        """Train the model"""
        # Add intercept term
        X = np.column_stack([np.ones(len(X)), X])
        
        # Ensure y is column vector
        y = y.reshape(-1)
        
        # Choose optimization method
        if self.method == 'gradient_descent':
            self._gradient_descent(X, y)
        elif self.method == 'newton':
            self._newton_method(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store coefficients separately
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.column_stack([np.ones(len(X)), X])
        probabilities = self._sigmoid(X @ self.theta)
        # Return probabilities for both classes
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Demonstration 1: Logistic Regression Basics
print("=== Demonstration 1: Logistic Regression Basics ===")

# Generate 2D data for visualization
X_2d, y_2d = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                  n_informative=2, random_state=42,
                                  n_clusters_per_class=1, flip_y=0.1)

# Split data
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)

# Train custom implementation
log_reg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
log_reg.fit(X_train_2d, y_train_2d)

# Predictions
y_pred_train = log_reg.predict(X_train_2d)
y_pred_test = log_reg.predict(X_test_2d)
y_proba_test = log_reg.predict_proba(X_test_2d)[:, 1]

print(f"Training Accuracy: {log_reg.score(X_train_2d, y_train_2d):.3f}")
print(f"Test Accuracy: {log_reg.score(X_test_2d, y_test_2d):.3f}")
print(f"Coefficients: {log_reg.coef_}")
print(f"Intercept: {log_reg.intercept_:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Data and Decision Boundary
ax = axes[0, 0]
# Create mesh for decision boundary
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot decision boundary
contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11),
                      cmap='RdBu_r', alpha=0.6)
ax.scatter(X_train_2d[y_train_2d == 0, 0], X_train_2d[y_train_2d == 0, 1],
           c='blue', edgecolor='black', s=50, label='Class 0 (Train)')
ax.scatter(X_train_2d[y_train_2d == 1, 0], X_train_2d[y_train_2d == 1, 1],
           c='red', edgecolor='black', s=50, label='Class 1 (Train)')
ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d,
           cmap='coolwarm', edgecolor='green', s=100, marker='s',
           label='Test Data')

# Decision boundary line
x_boundary = np.array([x_min, x_max])
y_boundary = -(log_reg.intercept_ + log_reg.coef_[0] * x_boundary) / log_reg.coef_[1]
ax.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Logistic Regression Decision Boundary')
ax.legend()
plt.colorbar(contour, ax=ax, label='P(y=1)')

# 2. Cost Function During Training
ax = axes[0, 1]
ax.plot(log_reg.costs)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost (Negative Log-Likelihood)')
ax.set_title('Cost Function During Training')
ax.grid(True, alpha=0.3)

# 3. Sigmoid Function Visualization
ax = axes[1, 0]
z = np.linspace(-10, 10, 100)
sigmoid_values = 1 / (1 + np.exp(-z))
ax.plot(z, sigmoid_values, 'b-', linewidth=2)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('z (linear combination)')
ax.set_ylabel('σ(z)')
ax.set_title('Sigmoid Function')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

# 4. Probability Distribution
ax = axes[1, 1]
# Histogram of predicted probabilities
ax.hist(y_proba_test[y_test_2d == 0], bins=20, alpha=0.5, 
        label='True Class 0', color='blue', density=True)
ax.hist(y_proba_test[y_test_2d == 1], bins=20, alpha=0.5, 
        label='True Class 1', color='red', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
ax.set_xlabel('Predicted Probability of Class 1')
ax.set_ylabel('Density')
ax.set_title('Distribution of Predicted Probabilities')
ax.legend()

plt.tight_layout()
plt.show()

# Demonstration 2: ROC Curve Analysis
print("\n=== Demonstration 2: ROC Curve Deep Dive ===")

# Generate larger dataset for better ROC analysis
X_roc, y_roc = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                   n_redundant=5, random_state=42, flip_y=0.05)

X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(
    X_roc, y_roc, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_roc_scaled = scaler.fit_transform(X_train_roc)
X_test_roc_scaled = scaler.transform(X_test_roc)

# Train model
log_reg_roc = LogisticRegression(learning_rate=0.1, n_iterations=500)
log_reg_roc.fit(X_train_roc_scaled, y_train_roc)

# Get probabilities
y_scores = log_reg_roc.predict_proba(X_test_roc_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_roc, y_scores)
roc_auc = auc(fpr, tpr)

# Manual ROC calculation for understanding
def calculate_roc_manually(y_true, y_scores):
    """Calculate ROC curve manually to understand the process"""
    # Sort by score descending
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Unique thresholds
    unique_scores = np.unique(y_scores_sorted)
    thresholds = np.concatenate([[np.inf], unique_scores, [-np.inf]])
    
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds[:-1]

# Calculate manually
fpr_manual, tpr_manual, thresholds_manual = calculate_roc_manually(y_test_roc, y_scores)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC Curve
ax = axes[0, 0]
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.2)

# Mark some thresholds
threshold_marks = [0.1, 0.3, 0.5, 0.7, 0.9]
for t_mark in threshold_marks:
    idx = np.argmin(np.abs(thresholds - t_mark))
    ax.plot(fpr[idx], tpr[idx], 'ro', markersize=8)
    ax.annotate(f't={t_mark:.1f}', (fpr[idx], tpr[idx]), 
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Threshold Impact
ax = axes[0, 1]
ax.plot(thresholds[::10], tpr[::10], 'g-', marker='o', label='TPR (Sensitivity)')
ax.plot(thresholds[::10], fpr[::10], 'r-', marker='s', label='FPR')
ax.plot(thresholds[::10], 1-fpr[::10], 'b-', marker='^', label='TNR (Specificity)')
ax.set_xlabel('Threshold')
ax.set_ylabel('Rate')
ax.set_title('TPR and FPR vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

# 3. Precision-Recall vs Threshold
from sklearn.metrics import precision_recall_curve
precision, recall, pr_thresholds = precision_recall_curve(y_test_roc, y_scores)

ax = axes[1, 0]
ax.plot(pr_thresholds, precision[:-1], 'b-', label='Precision')
ax.plot(pr_thresholds, recall[:-1], 'r-', label='Recall')
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
ax.plot(pr_thresholds, f1_scores, 'g-', label='F1-Score')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision, Recall, F1 vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

# Find optimal threshold (max F1)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx]
ax.axvline(x=optimal_threshold, color='red', linestyle='--', 
           label=f'Optimal: {optimal_threshold:.3f}')

# 4. AUC Interpretation
ax = axes[1, 1]
# Simulate ranking visualization
n_pos = np.sum(y_test_roc == 1)
n_neg = np.sum(y_test_roc == 0)

# Sort by predicted probability
sorted_idx = np.argsort(y_scores)[::-1]
y_sorted = y_test_roc[sorted_idx]

# Calculate how many negative examples are ranked below each positive
auc_interpretation = 0
for i, label in enumerate(y_sorted):
    if label == 1:
        # Count negatives ranked below this positive
        negatives_below = np.sum(y_sorted[i+1:] == 0)
        auc_interpretation += negatives_below

auc_interpretation /= (n_pos * n_neg)

# Visualization
ranks = np.arange(len(y_sorted))
colors = ['red' if y == 1 else 'blue' for y in y_sorted]
ax.scatter(ranks[:50], y_scores[sorted_idx][:50], c=colors[:50], alpha=0.6)
ax.set_xlabel('Rank (by predicted probability)')
ax.set_ylabel('Predicted Probability')
ax.set_title(f'AUC Interpretation\nP(rank(pos) > rank(neg)) = {auc_interpretation:.3f}')
ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10)],
          ['Positive', 'Negative'])

plt.tight_layout()
plt.show()

print(f"\nAUC calculated: {roc_auc:.3f}")
print(f"AUC interpretation (ranking): {auc_interpretation:.3f}")

# Demonstration 3: Multi-class Logistic Regression
print("\n=== Demonstration 3: Multi-class Extension ===")

# One-vs-Rest (OvR) implementation
class MulticlassLogisticRegression:
    """Multi-class logistic regression using One-vs-Rest strategy"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        # Train binary classifier for each class
        for class_label in self.classes:
            # Create binary labels
            y_binary = (y == class_label).astype(int)
            
            # Train classifier
            clf = LogisticRegression(**self.kwargs)
            clf.fit(X, y_binary)
            self.classifiers[class_label] = clf
            
        return self
    
    def predict_proba(self, X):
        # Get probabilities from each classifier
        probas = np.zeros((len(X), len(self.classes)))
        
        for i, class_label in enumerate(self.classes):
            probas[:, i] = self.classifiers[class_label].predict_proba(X)[:, 1]
        
        # Normalize probabilities
        probas = probas / probas.sum(axis=1, keepdims=True)
        return probas
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

# Generate multi-class data
X_multi, y_multi = make_classification(n_samples=300, n_features=2, n_informative=2,
                                      n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                      random_state=42)

# Train multi-class model
multi_log_reg = MulticlassLogisticRegression(learning_rate=0.1, n_iterations=500)
multi_log_reg.fit(X_multi, y_multi)

# Predictions
y_pred_multi = multi_log_reg.predict(X_multi)
accuracy_multi = np.mean(y_pred_multi == y_multi)
print(f"Multi-class Accuracy: {accuracy_multi:.3f}")

# Visualization
plt.figure(figsize=(10, 8))

# Create mesh
h = 0.02
x_min, x_max = X_multi[:, 0].min() - 1, X_multi[:, 0].max() + 1
y_min, y_max = X_multi[:, 1].min() - 1, X_multi[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for mesh
Z = multi_log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
scatter = plt.scatter(X_multi[:, 0], X_multi[:, 1], c=y_multi, 
                     cmap='viridis', edgecolor='black', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Multi-class Logistic Regression (OvR)\nAccuracy: {accuracy_multi:.3f}')
plt.colorbar(scatter, label='Class')
plt.show()

# Demonstration 4: Regularization Effects
print("\n=== Demonstration 4: Regularization in Logistic Regression ===")

# Generate dataset with more features
X_reg, y_reg = make_classification(n_samples=200, n_features=20, n_informative=10,
                                   n_redundant=10, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Standardize
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train models with different regularization
regularization_params = {
    'No Regularization': (None, 0),
    'L2 (λ=0.01)': ('l2', 0.01),
    'L2 (λ=0.1)': ('l2', 0.1),
    'L2 (λ=1.0)': ('l2', 1.0),
    'L1 (λ=0.1)': ('l1', 0.1)
}

results_reg = {}
coefficients = {}

for name, (reg_type, lambda_val) in regularization_params.items():
    model = LogisticRegression(
        regularization=reg_type, 
        lambda_reg=lambda_val,
        learning_rate=0.1,
        n_iterations=1000
    )
    
    model.fit(X_train_reg_scaled, y_train_reg)
    
    # Evaluate
    train_score = model.score(X_train_reg_scaled, y_train_reg)
    test_score = model.score(X_test_reg_scaled, y_test_reg)
    
    results_reg[name] = {
        'Train Accuracy': train_score,
        'Test Accuracy': test_score,
        'Non-zero Coef': np.sum(np.abs(model.coef_) > 1e-4)
    }
    
    coefficients[name] = model.coef_

# Display results
results_df = pd.DataFrame(results_reg).T
print("\nRegularization Effects:")
print(results_df)

# Visualize coefficients
plt.figure(figsize=(12, 8))

positions = np.arange(len(coefficients))
width = 0.15

for i, (name, coef) in enumerate(coefficients.items()):
    plt.bar(np.arange(len(coef)) + i * width, np.abs(coef), 
            width, label=name, alpha=0.8)

plt.xlabel('Feature Index')
plt.ylabel('|Coefficient|')
plt.title('Coefficient Magnitudes with Different Regularization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Demonstration 5: Calibration Analysis
print("\n=== Demonstration 5: Probability Calibration ===")

from sklearn.calibration import calibration_curve

# Train on breast cancer dataset
data = load_breast_cancer()
X_bc, y_bc = data.data, data.target

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.3, random_state=42
)

# Standardize
scaler_bc = StandardScaler()
X_train_bc_scaled = scaler_bc.fit_transform(X_train_bc)
X_test_bc_scaled = scaler_bc.transform(X_test_bc)

# Train models
models_calib = {
    'Logistic Regression': LogisticRegression(learning_rate=0.1, n_iterations=1000),
    'Logistic (High Reg)': LogisticRegression(learning_rate=0.1, n_iterations=1000,
                                              regularization='l2', lambda_reg=10)
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, model) in enumerate(models_calib.items()):
    model.fit(X_train_bc_scaled, y_train_bc)
    
    # Get probabilities
    y_proba = model.predict_proba(X_test_bc_scaled)[:, 1]
    
    # Calculate calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test_bc, y_proba, n_bins=10
    )
    
    # Brier score
    brier_score = np.mean((y_proba - y_test_bc) ** 2)
    
    # Plot
    ax = axes[idx]
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
            linewidth=2, markersize=8, label=f'{name}')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Add confidence bands
    ax.fill_between(mean_predicted_value, 
                    fraction_of_positives - 0.05, 
                    fraction_of_positives + 0.05, 
                    alpha=0.2)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{name}\nBrier Score: {brier_score:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Demonstration 6: Feature Importance and Interpretation
print("\n=== Demonstration 6: Feature Importance ===")

# Use breast cancer data with feature names
log_reg_bc = LogisticRegression(learning_rate=0.1, n_iterations=1000)
log_reg_bc.fit(X_train_bc_scaled, y_train_bc)

# Get feature importance (absolute coefficients)
feature_importance = np.abs(log_reg_bc.coef_)
feature_names = data.feature_names

# Sort by importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': log_reg_bc.coef_,
    'Abs_Coefficient': feature_importance
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10)[['Feature', 'Coefficient']].to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot of coefficients
ax = axes[0]
top_features = importance_df.head(15)
colors = ['red' if c < 0 else 'blue' for c in top_features['Coefficient']]
bars = ax.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Coefficient Value')
ax.set_title('Top 15 Feature Coefficients')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add interpretation
for bar, coef in zip(bars, top_features['Coefficient']):
    width = bar.get_width()
    label = 'Malignant' if width > 0 else 'Benign'
    ax.text(width + 0.01 if width > 0 else width - 0.01, 
            bar.get_y() + bar.get_height()/2,
            label, ha='left' if width > 0 else 'right', va='center',
            fontsize=8)

# Odds ratio interpretation
ax = axes[1]
odds_ratios = np.exp(log_reg_bc.coef_)
top_or_idx = np.argsort(np.abs(np.log(odds_ratios)))[-10:][::-1]

or_values = odds_ratios[top_or_idx]
or_features = [feature_names[i] for i in top_or_idx]

bars = ax.barh(range(len(or_values)), or_values)
ax.set_yticks(range(len(or_values)))
ax.set_yticklabels(or_features)
ax.set_xlabel('Odds Ratio')
ax.set_title('Top 10 Features by Odds Ratio')
ax.axvline(x=1, color='red', linestyle='--', linewidth=1)

# Add interpretation
for bar, or_val in zip(bars, or_values):
    width = bar.get_width()
    if or_val > 1:
        text = f'{or_val:.2f}x more likely'
    else:
        text = f'{1/or_val:.2f}x less likely'
    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
            text, ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.show()
```

---

## 6. Advanced Topics {#advanced-topics}

### Multinomial Logistic Regression

For K classes, model K-1 log-odds:
```
log(P(y=k)/P(y=K)) = βₖᵀx    for k = 1, ..., K-1
```

Probabilities via softmax:
```
P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)
```

### Ordinal Logistic Regression

For ordered categories, use cumulative logits:
```
logit(P(Y ≤ k)) = αₖ - βᵀx
```

### Logistic Regression Assumptions

1. **Linear relationship**: Between log-odds and features
2. **Independence**: Observations are independent
3. **No multicollinearity**: Features not perfectly correlated
4. **Large sample size**: Rule of thumb: 10-20 observations per feature

### Connection to Neural Networks

Logistic regression = Single-layer neural network:
- Linear combination: z = Wx + b
- Activation: σ(z)
- Loss: Binary cross-entropy

### Imbalanced Classes

Strategies:
1. **Class weights**: Weight loss by inverse frequency
2. **Threshold adjustment**: Optimize for desired metric
3. **Resampling**: SMOTE, undersampling
4. **Cost-sensitive learning**: Different costs for FP/FN

---

## 7. Comprehensive Interview Questions & Answers {#interview-qa}

### Fundamental Understanding

**Q1: Explain logistic regression. Why is it called "regression" when it's used for classification?**

**A:** Logistic regression models the probability of a binary outcome using the logistic (sigmoid) function. 

**Why "regression"?**
- It regresses the log-odds (logit) as a linear function of features
- Outputs continuous probabilities [0, 1], not discrete classes
- Historical naming from statistics

**Key points**:
- Models: P(y=1|x) = 1/(1 + e^(-βᵀx))
- Linear decision boundary
- Maximum likelihood estimation
- Outputs probabilities → threshold for classification

**Q2: What's the difference between linear and logistic regression?**

**A:** 

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Output** | Continuous (-∞, ∞) | Probability [0, 1] |
| **Function** | y = βᵀx | p = 1/(1 + e^(-βᵀx)) |
| **Use case** | Regression | Classification |
| **Loss** | Squared error | Cross-entropy |
| **Assumptions** | Normal residuals | Logistic distribution |
| **Decision boundary** | N/A | Linear |

**Key insight**: Logistic regression is linear regression on log-odds with sigmoid transformation.

**Q3: Why use sigmoid function in logistic regression?**

**A:** Several reasons:

1. **Maps to probabilities**: Transforms (-∞, ∞) → (0, 1)
2. **Smooth and differentiable**: Enables gradient-based optimization
3. **Nice derivative**: σ'(z) = σ(z)(1 - σ(z))
4. **Probabilistic interpretation**: Natural for log-odds
5. **Maximum likelihood**: Leads to convex optimization

**Mathematical justification**:
- If we model log-odds linearly: log(p/(1-p)) = βᵀx
- Solving for p: p = 1/(1 + e^(-βᵀx)) = sigmoid

### Mathematical Deep Dive

**Q4: Derive the gradient for logistic regression.**

**A:** Starting with negative log-likelihood loss:

```
J(β) = -(1/m) Σᵢ [yᵢ log(σ(βᵀxᵢ)) + (1-yᵢ) log(1-σ(βᵀxᵢ))]
```

For single sample:
```
∂J/∂β = ∂/∂β [-y log(σ(z)) - (1-y) log(1-σ(z))]    where z = βᵀx
```

Using chain rule and σ'(z) = σ(z)(1-σ(z)):
```
∂J/∂β = -y(1/σ(z))σ'(z)x - (1-y)(1/(1-σ(z)))(-σ'(z))x
      = -y(1-σ(z))x + (1-y)σ(z)x
      = (σ(z) - y)x
```

For all samples:
```
∇J = (1/m) Σᵢ (σ(βᵀxᵢ) - yᵢ)xᵢ = (1/m) Xᵀ(σ(Xβ) - y)
```

**Q5: Why can't we use MSE loss for logistic regression?**

**A:** Multiple reasons:

1. **Non-convex optimization**: MSE with sigmoid creates multiple local minima
2. **Gradient vanishing**: When predictions are very wrong, gradients → 0
3. **Probabilistic interpretation**: Cross-entropy is MLE for Bernoulli distribution
4. **Poor convergence**: Slow learning when predictions are confident but wrong

**Example**: If y=1 and ŷ→0:
- MSE gradient: 2(ŷ-1)σ'(z) → 0 as σ'(z)→0
- Cross-entropy gradient: (ŷ-1) → -1 (keeps learning)

### ROC and AUC Understanding

**Q6: Explain ROC curves and AUC. What does AUC actually measure?**

**A:** 

**ROC Curve**: Plots TPR vs FPR at all possible thresholds
- Shows tradeoff between catching positives (TPR) and false alarms (FPR)
- Threshold-independent evaluation

**AUC (Area Under ROC Curve)**: 
- Single number summary [0, 1]
- **Interpretation**: Probability that model ranks a random positive example higher than a random negative example

**Mathematical meaning**:
```
AUC = P(score(x⁺) > score(x⁻))
```

**Properties**:
- 0.5 = Random ranking
- 1.0 = Perfect ranking
- Invariant to class balance (controversial)
- Measures ranking quality, not calibration

**Q7: When is AUC misleading? What are alternatives?**

**A:** AUC can be misleading when:

1. **Highly imbalanced data**: Small improvements in FPR can seem significant
2. **Need specific operating point**: AUC averages across all thresholds
3. **Cost-sensitive problems**: Equal weight to all errors
4. **Calibration matters**: AUC ignores probability calibration

**Alternatives**:
- **Precision-Recall AUC**: Better for imbalanced data
- **Partial AUC**: Focus on specific FPR range
- **Cost curves**: Incorporate misclassification costs
- **Calibration plots**: Check probability reliability

**Example**: 99.9% negative class
- Random: AUC ≈ 0.5, PR-AUC ≈ 0.001
- PR-AUC better reflects difficulty

### Practical Applications

**Q8: How do you handle multi-class classification with logistic regression?**

**A:** Three main approaches:

1. **One-vs-Rest (OvR)**:
   - Train K binary classifiers
   - Class k vs all others
   - Predict: argmax of probabilities
   - Simple but can have ambiguous regions

2. **One-vs-One (OvO)**:
   - Train K(K-1)/2 binary classifiers
   - Each pair of classes
   - Predict: voting
   - More models but smaller training sets

3. **Multinomial (Softmax)**:
   - Single model with K-1 weight vectors
   - Softmax function for probabilities
   - More efficient, natural extension

**Implementation considerations**:
- OvR: Good for many classes
- Softmax: Preferred when feasible
- Handle class imbalance in each binary problem

**Q9: How do you interpret logistic regression coefficients?**

**A:** Multiple interpretations:

1. **Log-odds change**: βⱼ = change in log-odds for unit increase in xⱼ

2. **Odds ratio**: e^βⱼ = multiplicative change in odds
   - e^βⱼ > 1: Increases odds
   - e^βⱼ < 1: Decreases odds
   - e^βⱼ = 1: No effect

3. **Probability change**: Depends on current probability (non-linear)

**Example**: If β₁ = 0.693:
- Log-odds increase by 0.693
- Odds multiply by e^0.693 = 2
- "Doubles the odds"

**Important**: Must consider feature scaling and interactions!

### Algorithm Comparison

**Q10: When would you choose logistic regression over other classifiers?**

**A:** Choose logistic regression when:

**Advantages**:
1. **Need probabilities**: Calibrated probability outputs
2. **Interpretability**: Coefficients show feature effects
3. **Linear boundary sufficient**: Problem is linearly separable
4. **Limited data**: Works well with small datasets
5. **Baseline model**: Good starting point
6. **Feature importance**: Clear coefficient interpretation

**Prefer other methods when**:
- Non-linear boundaries (→ SVM, trees)
- Feature interactions important (→ trees, neural nets)
- Very high dimensions (→ regularized methods)
- No need for probabilities (→ SVM)

**Q11: Compare logistic regression with Naive Bayes.**

**A:** 

| Aspect | Logistic Regression | Naive Bayes |
|--------|-------------------|--------------|
| **Approach** | Discriminative | Generative |
| **Models** | P(y\|x) directly | P(x\|y) and P(y) |
| **Assumptions** | Log-linear | Feature independence |
| **Training** | Iterative (slower) | Closed-form (fast) |
| **Features** | Handles correlation | Assumes independence |
| **Data needed** | More | Less |
| **Calibration** | Better | Often poor |

**When to use which**:
- **Logistic**: Feature correlation, need calibration
- **Naive Bayes**: Quick baseline, text classification

### Advanced Topics

**Q12: Explain the connection between logistic regression and neural networks.**

**A:** Logistic regression is a single-layer neural network:

1. **Architecture**:
   - Input layer: Features
   - No hidden layers
   - Output: Single neuron with sigmoid

2. **Forward pass**:
   ```
   z = Wx + b
   a = σ(z)
   ```

3. **Loss**: Binary cross-entropy (same)

4. **Backprop**: Just gradient descent

**Extensions to neural networks**:
- Add hidden layers → non-linear boundaries
- Multiple outputs → multi-class
- Different activations → other functions

**Key insight**: Deep learning started here!

**Q13: How do you diagnose and improve a poorly performing logistic regression model?**

**A:** Systematic diagnosis:

1. **Check basics**:
   - Class balance
   - Feature scaling
   - Missing values
   - Data leakage

2. **Analyze predictions**:
   - Confusion matrix patterns
   - Probability distributions
   - Calibration plots
   - ROC/PR curves

3. **Feature analysis**:
   - Coefficient magnitudes
   - Multicollinearity (VIF)
   - Feature importance
   - Non-linear relationships

4. **Common fixes**:
   - **High bias**: Add features, polynomial terms, interactions
   - **High variance**: Regularization, fewer features
   - **Poor calibration**: Platt scaling, isotonic regression
   - **Class imbalance**: Weights, resampling, threshold adjustment

**Q14: Explain regularization in logistic regression.**

**A:** Similar to linear regression but applied to likelihood:

**L2 Regularization (Ridge)**:
```
J(β) = -ℓ(β) + (λ/2)||β||²
```
- Shrinks coefficients
- Handles multicollinearity
- Keeps all features

**L1 Regularization (Lasso)**:
```
J(β) = -ℓ(β) + λ||β||₁
```
- Sparse solutions
- Feature selection
- Less stable with correlated features

**Effects**:
- Prevents overfitting
- Improves generalization
- Controls model complexity

**Choice of λ**: Cross-validation

### Real-world Scenarios

**Q15: You're building a fraud detection system. How would you use logistic regression?**

**A:** Comprehensive approach:

1. **Problem formulation**:
   - Binary: Fraud/Not Fraud
   - Highly imbalanced (typically < 1% fraud)

2. **Feature engineering**:
   - Transaction amount, frequency
   - User behavior patterns
   - Time-based features
   - Device/location info

3. **Handle imbalance**:
   - Class weights: weight_fraud = n_total/n_fraud
   - Threshold optimization for business metric
   - Consider anomaly detection approach

4. **Model development**:
   ```python
   model = LogisticRegression(
       class_weight='balanced',
       penalty='l2',
       C=1.0
   )
   ```

5. **Evaluation strategy**:
   - Precision-Recall curve (not ROC)
   - Cost-based metrics
   - Temporal validation

6. **Production considerations**:
   - Real-time scoring
   - Threshold based on capacity
   - Monitoring for drift

**Q16: How do you explain logistic regression results to non-technical stakeholders?**

**A:** Layered explanation:

**Simple version**:
"The model calculates the probability of [outcome] based on [features]. Higher probability means more likely."

**Coefficient interpretation**:
"For every unit increase in [feature], the odds of [outcome] multiply by [e^β]. For example, if odds double, something twice as likely."

**Visual aids**:
1. Feature importance bar chart
2. Probability distributions
3. Example predictions with explanations

**Business impact**:
"Setting threshold at X gives:
- Catches Y% of [positive cases]
- Z% false alarms
- $A in expected value"

### Interview Problems

**Q17: Implement logistic regression gradient descent from scratch.**

**A:** Core implementation:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression_gd(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Forward pass
        z = X @ theta
        h = sigmoid(z)
        
        # Cost (optional, for monitoring)
        cost = -(1/m) * (y @ np.log(h + 1e-7) + 
                         (1-y) @ np.log(1-h + 1e-7))
        
        # Gradient
        gradient = (1/m) * X.T @ (h - y)
        
        # Update
        theta -= lr * gradient
        
    return theta
```

**Q18: The AUC of your model is 0.95 but it performs poorly in production. Why?**

**A:** Several possible reasons:

1. **Distribution shift**:
   - Training/production data mismatch
   - Temporal patterns changed
   - Different population

2. **Label quality**:
   - Training labels noisy
   - Different labeling in production

3. **Metric mismatch**:
   - AUC measures ranking
   - Production needs specific threshold
   - Costs not reflected in AUC

4. **Calibration issues**:
   - High AUC but poor probability calibration
   - Threshold chosen incorrectly

5. **Sample selection bias**:
   - Training on processed data
   - Production sees raw data

**Solution**: Monitor production metrics, A/B test, retrain regularly

**Q19: Design a real-time logistic regression system.**

**A:** Architecture considerations:

1. **Feature computation**:
   - Pre-compute expensive features
   - Cache user/item features
   - Real-time feature serving

2. **Model serving**:
   ```python
   class LogisticRegressionServer:
       def __init__(self, model_path):
           self.model = load_model(model_path)
           self.scaler = load_scaler(model_path)
           
       def predict(self, features):
           # Fast path
           X = self.scaler.transform(features)
           prob = self.model.predict_proba(X)[:, 1]
           return prob
   ```

3. **Optimization**:
   - Vectorized predictions
   - Model quantization
   - Caching predictions

4. **Monitoring**:
   - Prediction latency
   - Feature distributions
   - Score distributions

**Q20: How do you test if your logistic regression model is significantly better than random?**

**A:** Statistical testing approaches:

1. **DeLong's test** (for AUC):
   - Tests if AUC significantly > 0.5
   - Accounts for correlation in predictions

2. **Permutation test**:
   - Shuffle labels, retrain
   - Compare to actual performance
   - Empirical p-value

3. **McNemar's test**:
   - For paired predictions
   - Tests if error rates differ

4. **Likelihood ratio test**:
   - Compare to null model (intercept only)
   - Chi-squared distribution

**Implementation sketch**:
```python
def likelihood_ratio_test(model_full, model_null, df):
    lr_stat = 2 * (model_full.log_likelihood - 
                   model_null.log_likelihood)
    p_value = 1 - stats.chi2.cdf(lr_stat, df)
    return lr_stat, p_value
```

---

## Practice Problems

1. Implement multinomial logistic regression
2. Build Newton-Raphson optimizer
3. Create calibration diagnostic tools
4. Implement focal loss for imbalanced data
5. Build online learning logistic regression
6. Create feature importance with confidence intervals

## Key Takeaways

1. **Logistic regression models probabilities** using sigmoid function
2. **Linear decision boundary** in feature space
3. **Convex optimization** guarantees global optimum
4. **Coefficients represent log-odds** changes
5. **ROC/AUC evaluates ranking** quality
6. **Regularization prevents overfitting**
7. **Foundation for neural networks**
8. **Interpretable and probabilistic**