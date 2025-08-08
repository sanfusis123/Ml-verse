# Day 20: Support Vector Machines (SVM)

## Table of Contents
1. [Introduction](#1-introduction)
2. [The Intuition Behind SVM](#2-the-intuition-behind-svm)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Hard Margin vs Soft Margin SVM](#4-hard-margin-vs-soft-margin-svm)
5. [The Kernel Trick](#5-the-kernel-trick)
6. [Multi-class Classification with SVM](#6-multi-class-classification-with-svm)
7. [Implementation from Scratch](#7-implementation-from-scratch)
8. [Using Scikit-learn](#8-using-scikit-learn)
9. [Advanced Topics](#9-advanced-topics)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

Support Vector Machines (SVM) are powerful supervised learning algorithms used for both classification and regression tasks. SVMs are particularly effective in high-dimensional spaces and are memory efficient as they use only a subset of training points (support vectors) in the decision function.

### What Makes SVM Special?

1. **Maximum Margin Principle**: SVM finds the hyperplane that maximizes the margin between classes
2. **Kernel Trick**: Can handle non-linear classification without explicitly transforming features
3. **Robust to Outliers**: Uses only support vectors for decision boundary
4. **Theoretical Foundation**: Strong mathematical foundation in statistical learning theory

### When to Use SVM?

- **High-dimensional data**: Text classification, gene expression data
- **Non-linear problems**: When kernel trick can capture complex patterns
- **Binary classification**: Originally designed for two-class problems
- **Limited training data**: Effective even with small datasets

## 2. The Intuition Behind SVM

### Geometric Interpretation

Imagine you have two classes of points in 2D space. SVM tries to find the best line (or hyperplane in higher dimensions) that separates these classes. But what makes a line "best"?

```
Class A: o o o o
         o o o
           |← margin →|
    ------------------- (decision boundary)
           |← margin →|
         x x x
Class B: x x x x
```

### The Margin Concept

The **margin** is the distance between the decision boundary and the nearest data points from each class. These nearest points are called **support vectors**.

**Key Insight**: Among all possible separating hyperplanes, SVM chooses the one with the maximum margin. This provides better generalization to unseen data.

### Why Maximum Margin?

1. **Confidence**: Points far from the boundary are classified with high confidence
2. **Generalization**: Larger margins typically lead to lower generalization error
3. **Uniqueness**: For linearly separable data, the maximum margin hyperplane is unique

## 3. Mathematical Foundation

### Linear SVM Formulation

For a dataset with n samples {(x₁, y₁), ..., (xₙ, yₙ)} where xᵢ ∈ ℝᵈ and yᵢ ∈ {-1, +1}:

#### Decision Function
```
f(x) = wᵀx + b
```
Where:
- w is the weight vector (normal to the hyperplane)
- b is the bias term
- Classification: sign(f(x))

#### Optimization Problem

**Primal Form**:
```
minimize    ½||w||²
subject to  yᵢ(wᵀxᵢ + b) ≥ 1, for all i
```

**Geometric Margin**: γ = 1/||w||

**Intuition**: Minimizing ||w|| maximizes the margin

#### Lagrangian and Dual Form

Using Lagrange multipliers αᵢ ≥ 0:

```
L(w, b, α) = ½||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]
```

**Dual Problem**:
```
maximize    Σᵢ αᵢ - ½ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
subject to  Σᵢ αᵢyᵢ = 0 and αᵢ ≥ 0
```

#### KKT Conditions

The Karush-Kuhn-Tucker conditions give us:
- αᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0 (complementary slackness)
- Only points with αᵢ > 0 are support vectors
- Support vectors satisfy yᵢ(wᵀxᵢ + b) = 1

## 4. Hard Margin vs Soft Margin SVM

### Hard Margin SVM

**Assumption**: Data is linearly separable
**Constraint**: All points must be correctly classified

**Limitations**:
- May not exist for non-separable data
- Sensitive to outliers
- Can lead to overfitting

### Soft Margin SVM

Introduces slack variables ξᵢ to allow misclassification:

**Optimization Problem**:
```
minimize    ½||w||² + C Σᵢ ξᵢ
subject to  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0
```

Where:
- C is the regularization parameter
- ξᵢ measures the degree of misclassification

**C Parameter Trade-off**:
- **Large C**: Less regularization, smaller margins, fewer misclassifications
- **Small C**: More regularization, larger margins, more misclassifications

## 5. The Kernel Trick

### Motivation

Linear SVM can only find linear decision boundaries. For non-linear problems, we need to transform data to a higher-dimensional space where it becomes linearly separable.

### Kernel Function

Instead of explicitly computing φ(x) (feature transformation), we use kernel functions:
```
K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
```

### Common Kernels

1. **Linear Kernel**:
   ```
   K(xᵢ, xⱼ) = xᵢᵀxⱼ
   ```

2. **Polynomial Kernel**:
   ```
   K(xᵢ, xⱼ) = (γxᵢᵀxⱼ + r)ᵈ
   ```

3. **RBF (Gaussian) Kernel**:
   ```
   K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
   ```

4. **Sigmoid Kernel**:
   ```
   K(xᵢ, xⱼ) = tanh(γxᵢᵀxⱼ + r)
   ```

### Kernel Trick in Action

The dual optimization becomes:
```
maximize    Σᵢ αᵢ - ½ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ)
```

Decision function:
```
f(x) = Σᵢ αᵢyᵢK(xᵢ, x) + b
```

## 6. Multi-class Classification with SVM

### One-vs-Rest (OvR)

Train K binary classifiers, each separating one class from all others:
- For class k: positive examples from class k, negative examples from all other classes
- Prediction: class with highest decision function value

### One-vs-One (OvO)

Train K(K-1)/2 binary classifiers for all pairs of classes:
- Each classifier trained on data from two classes only
- Prediction: voting scheme (class with most "wins")

### Comparison

| Approach | # Classifiers | Training Complexity | Prediction Time |
|----------|---------------|--------------------|-----------------| 
| OvR      | K             | O(K·n)             | O(K)            |
| OvO      | K(K-1)/2      | O(K²·n/K²) = O(n)  | O(K²)           |

## 7. Implementation from Scratch

```python
import numpy as np
from cvxopt import matrix, solvers

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self.b = None
        
    def _kernel_function(self, x1, x2):
        """Compute kernel function between two vectors"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1, X2):
        """Compute kernel matrix between two sets of samples"""
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        return K
    
    def fit(self, X, y):
        """Train SVM using quadratic programming"""
        n_samples, n_features = X.shape
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X, X)
        
        # Setup QP problem: minimize 1/2 x^T P x + q^T x
        # subject to: G x <= h and A x = b
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        
        # Inequality constraints: -alpha_i <= 0 and alpha_i <= C
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = matrix(np.hstack([np.zeros(n_samples), self.C * np.ones(n_samples)]))
        
        # Equality constraint: sum(alpha_i * y_i) = 0
        A = matrix(y.reshape(1, -1).astype(np.float64))
        b = matrix(0.0)
        
        # Solve QP
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        # Extract alphas
        alphas = np.array(solution['x']).flatten()
        
        # Find support vectors (alphas > threshold)
        support_vector_indices = alphas > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.alphas = alphas[support_vector_indices]
        
        # Compute bias term b
        self._compute_bias(K, y, alphas, support_vector_indices)
        
        return self
    
    def _compute_bias(self, K, y, alphas, support_vector_indices):
        """Compute bias term using support vectors"""
        # Find support vectors with 0 < alpha < C
        sv_indices = np.where((alphas > 1e-5) & (alphas < self.C - 1e-5))[0]
        
        if len(sv_indices) > 0:
            # Compute b using these support vectors
            b_values = []
            for idx in sv_indices:
                b_val = y[idx] - np.sum(alphas * y * K[:, idx])
                b_values.append(b_val)
            self.b = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            self.b = np.mean(
                self.support_vector_labels - 
                np.sum(self.alphas * self.support_vector_labels * 
                       K[support_vector_indices][:, support_vector_indices], axis=1)
            )
    
    def decision_function(self, X):
        """Compute decision function values"""
        K = self._compute_kernel_matrix(X, self.support_vectors)
        return np.sum(self.alphas * self.support_vector_labels * K.T, axis=0) + self.b
    
    def predict(self, X):
        """Predict class labels"""
        return np.sign(self.decision_function(X))
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Example usage with different kernels
def demonstrate_svm():
    # Generate synthetic data
    np.random.seed(42)
    
    # Linearly separable data
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X_linear = np.vstack([X1, X2])
    y_linear = np.hstack([np.ones(50), -np.ones(50)])
    
    # Non-linearly separable data (circles)
    angles1 = np.random.uniform(0, 2*np.pi, 50)
    r1 = np.random.uniform(0, 2, 50)
    X1_circle = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    angles2 = np.random.uniform(0, 2*np.pi, 50)
    r2 = np.random.uniform(3, 5, 50)
    X2_circle = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    X_circle = np.vstack([X1_circle, X2_circle])
    y_circle = np.hstack([np.ones(50), -np.ones(50)])
    
    # Test different kernels
    print("Linear SVM on linearly separable data:")
    svm_linear = SVMClassifier(kernel='linear', C=1.0)
    svm_linear.fit(X_linear, y_linear)
    print(f"Accuracy: {svm_linear.score(X_linear, y_linear):.4f}")
    print(f"Number of support vectors: {len(svm_linear.support_vectors)}")
    
    print("\nRBF SVM on circular data:")
    svm_rbf = SVMClassifier(kernel='rbf', C=1.0, gamma=0.5)
    svm_rbf.fit(X_circle, y_circle)
    print(f"Accuracy: {svm_rbf.score(X_circle, y_circle):.4f}")
    print(f"Number of support vectors: {len(svm_rbf.support_vectors)}")

# Demonstrate soft margin SVM
def soft_margin_example():
    # Create data with outliers
    np.random.seed(42)
    X1 = np.random.randn(45, 2) + np.array([2, 2])
    X2 = np.random.randn(45, 2) + np.array([-2, -2])
    
    # Add outliers
    X1_outliers = np.random.randn(5, 2) + np.array([-2, -2])
    X2_outliers = np.random.randn(5, 2) + np.array([2, 2])
    
    X = np.vstack([X1, X1_outliers, X2, X2_outliers])
    y = np.hstack([np.ones(50), -np.ones(50)])
    
    # Test different C values
    for C in [0.1, 1.0, 10.0, 100.0]:
        svm = SVMClassifier(kernel='linear', C=C)
        svm.fit(X, y)
        print(f"\nC = {C}:")
        print(f"Accuracy: {svm.score(X, y):.4f}")
        print(f"Number of support vectors: {len(svm.support_vectors)}")
```

## 8. Using Scikit-learn

```python
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Complete SVM pipeline
class SVMPipeline:
    def __init__(self, kernel='rbf', scale=True):
        self.kernel = kernel
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.model = None
        
    def fit(self, X_train, y_train, param_grid=None):
        """Train SVM with optional hyperparameter tuning"""
        # Scale features
        if self.scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        if param_grid:
            # Grid search for best parameters
            svm_model = svm.SVC(kernel=self.kernel)
            self.model = GridSearchCV(svm_model, param_grid, cv=5, 
                                    scoring='accuracy', n_jobs=-1)
            self.model.fit(X_train_scaled, y_train)
            print(f"Best parameters: {self.model.best_params_}")
            print(f"Best CV score: {self.model.best_score_:.4f}")
        else:
            # Train with default parameters
            self.model = svm.SVC(kernel=self.kernel)
            self.model.fit(X_train_scaled, y_train)
            
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if self.scale:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        return self.model.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        accuracy = np.mean(predictions == y_test)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        return accuracy

# Example: Complete workflow
def svm_workflow_example():
    from sklearn.datasets import make_classification, make_moons
    from sklearn.model_selection import train_test_split
    
    # Generate datasets
    # 1. Linear dataset
    X_linear, y_linear = make_classification(n_samples=200, n_features=2, 
                                            n_redundant=0, n_informative=2,
                                            n_clusters_per_class=1, 
                                            class_sep=2.0, random_state=42)
    
    # 2. Non-linear dataset
    X_nonlinear, y_nonlinear = make_moons(n_samples=200, noise=0.15, 
                                          random_state=42)
    
    # Split data
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_linear, y_linear, test_size=0.3, random_state=42)
    
    X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
        X_nonlinear, y_nonlinear, test_size=0.3, random_state=42)
    
    # Linear SVM for linear data
    print("Linear SVM on linear data:")
    linear_pipeline = SVMPipeline(kernel='linear')
    param_grid_linear = {'C': [0.1, 1, 10, 100]}
    linear_pipeline.fit(X_train_l, y_train_l, param_grid_linear)
    linear_pipeline.evaluate(X_test_l, y_test_l)
    
    # RBF SVM for non-linear data
    print("\n" + "="*50)
    print("RBF SVM on non-linear data:")
    rbf_pipeline = SVMPipeline(kernel='rbf')
    param_grid_rbf = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    rbf_pipeline.fit(X_train_nl, y_train_nl, param_grid_rbf)
    rbf_pipeline.evaluate(X_test_nl, y_test_nl)
    
    # Visualize decision boundaries
    def plot_decision_boundary(X, y, model, title):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                   edgecolor='black')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    # Plot for linear data
    plot_decision_boundary(X_test_l, y_test_l, linear_pipeline, 
                          "Linear SVM Decision Boundary")
    
    # Plot for non-linear data
    plot_decision_boundary(X_test_nl, y_test_nl, rbf_pipeline, 
                          "RBF SVM Decision Boundary")

# Multi-class classification example
def multiclass_svm_example():
    from sklearn.datasets import load_iris
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-vs-Rest
    print("One-vs-Rest SVM:")
    ovr_svm = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1.0, gamma='auto'))
    ovr_svm.fit(X_train_scaled, y_train)
    ovr_pred = ovr_svm.predict(X_test_scaled)
    print(f"Accuracy: {np.mean(ovr_pred == y_test):.4f}")
    
    # One-vs-One
    print("\nOne-vs-One SVM:")
    ovo_svm = OneVsOneClassifier(svm.SVC(kernel='rbf', C=1.0, gamma='auto'))
    ovo_svm.fit(X_train_scaled, y_train)
    ovo_pred = ovo_svm.predict(X_test_scaled)
    print(f"Accuracy: {np.mean(ovo_pred == y_test):.4f}")
    
    # Default SVM (uses OvO for multi-class)
    print("\nDefault SVM (OvO):")
    default_svm = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
    default_svm.fit(X_train_scaled, y_train)
    default_pred = default_svm.predict(X_test_scaled)
    print(f"Accuracy: {np.mean(default_pred == y_test):.4f}")
```

## 9. Advanced Topics

### 9.1 SVM for Regression (SVR)

Support Vector Regression uses similar principles but with ε-insensitive loss:

```python
from sklearn.svm import SVR

def svr_example():
    # Generate regression data
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    
    # Train different SVR models
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_linear = SVR(kernel='linear', C=100, epsilon=0.1)
    svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1)
    
    # Fit models
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_linear = svr_linear.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=2, label='RBF model')
    plt.plot(X, y_linear, color='c', lw=2, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=2, label='Polynomial model')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
```

### 9.2 Online SVM (SGD)

For large-scale problems, use Stochastic Gradient Descent:

```python
from sklearn.linear_model import SGDClassifier

def online_svm_example():
    # Simulate streaming data
    from sklearn.datasets import make_classification
    
    # Generate initial batch
    X_batch1, y_batch1 = make_classification(n_samples=1000, n_features=20,
                                            n_classes=2, random_state=42)
    
    # Initialize SGD-based SVM
    sgd_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001,
                           random_state=42, max_iter=1000)
    
    # Initial training
    sgd_svm.fit(X_batch1, y_batch1)
    print(f"Initial accuracy: {sgd_svm.score(X_batch1, y_batch1):.4f}")
    
    # Simulate new batches
    for i in range(5):
        X_new, y_new = make_classification(n_samples=200, n_features=20,
                                          n_classes=2, random_state=i)
        # Partial fit on new data
        sgd_svm.partial_fit(X_new, y_new)
        print(f"After batch {i+2}, accuracy: {sgd_svm.score(X_new, y_new):.4f}")
```

### 9.3 Probability Calibration

SVM doesn't naturally provide probability estimates. Use calibration:

```python
from sklearn.calibration import CalibratedClassifierCV

def probability_calibration_example():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Standard SVM
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    
    # Calibrated SVM
    calibrated_svm = CalibratedClassifierCV(svm_model, cv=3, method='sigmoid')
    calibrated_svm.fit(X_train, y_train)
    
    # Compare predictions
    svm_pred = svm_model.predict(X_test[:5])
    prob_pred = calibrated_svm.predict_proba(X_test[:5])
    
    print("SVM predictions:", svm_pred)
    print("\nCalibrated probabilities:")
    for i, probs in enumerate(prob_pred):
        print(f"Sample {i}: Class 0: {probs[0]:.3f}, Class 1: {probs[1]:.3f}")
```

### 9.4 Custom Kernels

Create custom kernel functions:

```python
def custom_kernel_example():
    # Define custom kernel: histogram intersection kernel
    def histogram_intersection_kernel(X, Y):
        """Histogram intersection kernel for histogram features"""
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        K = np.zeros((n_samples_X, n_samples_Y))
        
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                K[i, j] = np.sum(np.minimum(X[i], Y[j]))
        return K
    
    # Generate histogram-like data (all positive values)
    np.random.seed(42)
    X_train = np.abs(np.random.randn(100, 50))
    y_train = np.random.choice([-1, 1], 100)
    X_test = np.abs(np.random.randn(20, 50))
    
    # Normalize to make histograms
    X_train = X_train / X_train.sum(axis=1, keepdims=True)
    X_test = X_test / X_test.sum(axis=1, keepdims=True)
    
    # Precompute kernel matrix
    K_train = histogram_intersection_kernel(X_train, X_train)
    K_test = histogram_intersection_kernel(X_test, X_train)
    
    # Train SVM with precomputed kernel
    svm_model = svm.SVC(kernel='precomputed')
    svm_model.fit(K_train, y_train)
    
    # Predict
    predictions = svm_model.predict(K_test)
    print(f"Custom kernel predictions: {predictions}")
```

## 10. Interview Questions

### Q1: What is the main idea behind Support Vector Machines?
**Answer**: SVM finds the optimal hyperplane that maximizes the margin between different classes. The margin is the distance between the hyperplane and the nearest data points from each class (support vectors). This maximum margin principle helps achieve better generalization.

### Q2: Explain the difference between hard margin and soft margin SVM.
**Answer**: 
- **Hard margin SVM**: Assumes data is linearly separable and requires all points to be correctly classified. No misclassifications allowed.
- **Soft margin SVM**: Introduces slack variables to allow some misclassifications. Uses parameter C to balance between maximizing margin and minimizing misclassifications. More practical for real-world data.

### Q3: What is the kernel trick and why is it important?
**Answer**: The kernel trick allows SVM to work in high-dimensional feature spaces without explicitly computing the transformations. Instead of computing φ(x), we use kernel functions K(xi, xj) = φ(xi)·φ(xj). This makes non-linear classification computationally feasible.

### Q4: How do you choose between different kernel functions?
**Answer**:
- **Linear kernel**: When data is linearly separable or high-dimensional (text classification)
- **RBF kernel**: General-purpose, good for non-linear problems, fewer hyperparameters than polynomial
- **Polynomial kernel**: When you suspect polynomial relationships
- **Sigmoid kernel**: Similar to neural networks, but RBF often performs better

Use cross-validation to compare performance.

### Q5: What are support vectors and why are they important?
**Answer**: Support vectors are the data points that lie closest to the decision boundary. They are the only points that determine the position of the hyperplane. This makes SVM memory efficient and robust to outliers that are far from the boundary.

### Q6: How does the C parameter affect SVM performance?
**Answer**: 
- **Large C**: Less regularization, tries to classify all training examples correctly, smaller margins, risk of overfitting
- **Small C**: More regularization, allows more misclassifications, larger margins, risk of underfitting

C controls the trade-off between smooth decision boundary and classifying training points correctly.

### Q7: What are the advantages and disadvantages of SVM?
**Answer**:
**Advantages**:
- Effective in high-dimensional spaces
- Memory efficient (only uses support vectors)
- Versatile through different kernels
- Strong theoretical foundation

**Disadvantages**:
- Computationally expensive for large datasets O(n²) to O(n³)
- Sensitive to feature scaling
- No probabilistic output by default
- Choosing right kernel and parameters can be challenging

### Q8: How does SVM handle multi-class classification?
**Answer**: SVM is inherently binary. For multi-class:
- **One-vs-Rest (OvR)**: Train K classifiers, each separating one class from all others
- **One-vs-One (OvO)**: Train K(K-1)/2 classifiers for all class pairs
OvO often performs better but requires more classifiers.

### Q9: What is the time complexity of training an SVM?
**Answer**: 
- Training: O(n²) to O(n³) depending on the algorithm and data
- Prediction: O(nsv × d) where nsv is number of support vectors, d is dimensions
- For linear SVM with specialized solvers: O(n × d)

### Q10: How do you handle imbalanced datasets with SVM?
**Answer**:
1. **Class weights**: Use class_weight='balanced' or custom weights
2. **Different C values**: Use larger C for minority class
3. **Sampling**: Over-sample minority or under-sample majority class
4. **Ensemble methods**: Combine multiple SVMs trained on balanced subsets

### Q11: Explain the role of gamma in RBF kernel.
**Answer**: Gamma defines the influence of a single training example:
- **Low gamma**: Far-reaching influence, softer decision boundary
- **High gamma**: Close-reach influence, more complex boundary, risk of overfitting

Gamma = 1/(2σ²) where σ is the Gaussian kernel width.

### Q12: Can SVM be used for regression? How?
**Answer**: Yes, Support Vector Regression (SVR) uses similar principles but with ε-insensitive loss. It tries to fit as many instances as possible within an ε-margin around the prediction while maximizing the margin. Points outside this tube become support vectors.

### Q13: What is the difference between SVM and logistic regression?
**Answer**:
- **Loss function**: SVM uses hinge loss, LR uses log loss
- **Decision boundary**: SVM maximizes margin, LR finds any separating boundary
- **Solution**: SVM solution depends only on support vectors, LR uses all data
- **Output**: SVM gives class labels, LR gives probabilities naturally

### Q14: How do you perform feature selection with SVM?
**Answer**:
1. **Recursive Feature Elimination (RFE)**: Remove features based on weights
2. **L1-regularized SVM**: Use LinearSVC with L1 penalty
3. **Mutual information**: Select features before training
4. **Embedded methods**: Use SVM weights to rank features

### Q15: What are the KKT conditions in SVM?
**Answer**: Karush-Kuhn-Tucker conditions are necessary for optimality:
1. Stationarity: ∇L = 0
2. Primal feasibility: yi(wᵀxi + b) ≥ 1
3. Dual feasibility: αi ≥ 0
4. Complementary slackness: αi[yi(wᵀxi + b) - 1] = 0

These conditions tell us that only support vectors have αi > 0.

### Q16: How do you handle missing values in SVM?
**Answer**:
1. **Imputation**: Fill with mean, median, or use advanced methods
2. **Deletion**: Remove samples with missing values
3. **Indicator variables**: Add binary indicators for missingness
4. **Special kernels**: Design kernels that handle missing values

SVM doesn't naturally handle missing values.

### Q17: What is the ν-SVM formulation?
**Answer**: ν-SVM uses parameter ν instead of C:
- ν ∈ (0, 1] controls the fraction of support vectors
- Upper bound on fraction of margin errors
- Lower bound on fraction of support vectors
- More interpretable than C parameter

### Q18: How do you speed up SVM training for large datasets?
**Answer**:
1. **Linear SVM**: Use specialized solvers (LIBLINEAR)
2. **Approximation**: Random Fourier features, Nyström method
3. **SGD**: Use SGDClassifier with hinge loss
4. **Subset training**: Train on representative subset
5. **Parallel processing**: Some implementations support parallelization

### Q19: Explain the concept of working set selection in SVM optimization.
**Answer**: SMO (Sequential Minimal Optimization) selects pairs of variables to optimize at each step:
1. **First variable**: Violates KKT conditions most
2. **Second variable**: Maximizes step size |E1 - E2|
3. **Update**: Optimize these two while keeping others fixed
4. **Repeat**: Until convergence

This decomposition makes large-scale SVM training feasible.

### Q20: How do you interpret SVM decision boundaries?
**Answer**:
1. **Linear SVM**: w gives feature importance, larger |wi| means more important
2. **Non-linear SVM**: Harder to interpret directly
3. **Support vectors**: Points that define the boundary
4. **Distance to hyperplane**: Confidence in prediction
5. **Visualization**: Plot 2D projections or use dimensionality reduction

### Q21: What is the relationship between SVM and neural networks?
**Answer**:
- SVM with sigmoid kernel ≈ two-layer neural network
- Both can learn non-linear boundaries
- SVM has convex optimization (global optimum)
- Neural networks more flexible but local optima
- SVM better for small datasets, NN better for large datasets

### Q22: How do you validate SVM hyperparameters efficiently?
**Answer**:
1. **Grid search**: Exhaustive but expensive
2. **Random search**: Often finds good parameters faster
3. **Bayesian optimization**: Smart parameter selection
4. **Gradient-based**: For some parameters (C, γ)
5. **Multi-scale search**: Coarse then fine grid
6. **Cross-validation**: Use k-fold CV for robust estimates

Best practice: Start with wide range, then refine around best values.