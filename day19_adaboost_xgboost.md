# Day 19: AdaBoost and XGBoost

## Table of Contents
1. [Introduction to Advanced Boosting](#introduction)
2. [AdaBoost Deep Dive](#adaboost)
3. [XGBoost Architecture](#xgboost)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Implementation from Scratch](#implementation)
6. [Optimization Techniques](#optimization)
7. [Hyperparameter Tuning](#hyperparameters)
8. [Practical Applications](#applications)
9. [Advanced Topics](#advanced-topics)
10. [Interview Questions & Answers](#interview-qa)

## 1. Introduction to Advanced Boosting <a id="introduction"></a>

### Evolution of Boosting Algorithms:
1. **AdaBoost (1995)**: First practical boosting algorithm
2. **Gradient Boosting (2001)**: Generalized boosting framework
3. **XGBoost (2014)**: Optimized gradient boosting
4. **LightGBM (2017)**: Microsoft's efficient implementation
5. **CatBoost (2017)**: Yandex's categorical-friendly version

### Why These Algorithms Dominate:
- **AdaBoost**: Simple, interpretable, theoretical guarantees
- **XGBoost**: Speed, accuracy, regularization, missing values
- Win most Kaggle competitions
- Industry standard for tabular data

## 2. AdaBoost Deep Dive <a id="adaboost"></a>

### Adaptive Boosting Algorithm:

#### Core Idea:
"Focus on mistakes by reweighting misclassified samples"

#### Algorithm Steps:
1. Initialize uniform weights: w_i = 1/n
2. For m = 1 to M:
   - Train weak learner h_m on weighted data
   - Calculate weighted error: ε_m = Σ w_i × I(y_i ≠ h_m(x_i))
   - Calculate learner weight: α_m = 0.5 × ln((1-ε_m)/ε_m)
   - Update sample weights: w_i = w_i × exp(-α_m × y_i × h_m(x_i))
   - Normalize weights: w_i = w_i / Σw_i

3. Final prediction: H(x) = sign(Σ α_m × h_m(x))

### Mathematical Insight:
AdaBoost minimizes exponential loss:
```
L(y, f(x)) = exp(-y × f(x))
```

### Key Properties:
- **Stagewise Additive Modeling**: Adds one function at a time
- **Exponential Loss**: Sensitive to outliers
- **Self-Regularizing**: Margins continue to increase
- **Feature Selection**: Implicit through weak learners

### AdaBoost Variants:

#### AdaBoost.M1 (Multiclass):
```python
# For K classes
# Uses SAMME algorithm
α_m = log((1-ε_m)/ε_m) + log(K-1)
```

#### AdaBoost.R2 (Regression):
```python
# For continuous targets
# Uses different loss functions
L = max|y_i - h_m(x_i)| or Σ(y_i - h_m(x_i))²
```

#### Real AdaBoost:
- Outputs class probabilities instead of labels
- Uses probability estimates from weak learners

## 3. XGBoost Architecture <a id="xgboost"></a>

### eXtreme Gradient Boosting:

#### Core Innovations:
1. **Regularized Learning**: Prevents overfitting
2. **Parallel Processing**: Column-wise parallelization
3. **Tree Pruning**: Depth-first, prune backwards
4. **Missing Values**: Learned default directions
5. **Built-in CV**: Integrated cross-validation
6. **Continued Training**: Can add trees to existing model

### Objective Function:
```
Obj = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
```
Where:
- l = loss function (customizable)
- Ω = regularization term
- f_k = k-th tree

### Regularization Term:
```
Ω(f) = γT + (1/2)λ Σ w_j²
```
Where:
- T = number of leaves
- w_j = leaf weights
- γ = complexity penalty
- λ = L2 regularization

### Key Components:

#### 1. Second-Order Approximation:
Uses both first and second derivatives:
```
Obj ≈ Σ [g_i × f_t(x_i) + (1/2) × h_i × f_t²(x_i)] + Ω(f_t)
```
Where:
- g_i = ∂l/∂ŷ (gradient)
- h_i = ∂²l/∂ŷ² (hessian)

#### 2. Split Finding:
- **Exact Greedy**: Try all possible splits
- **Approximate**: Use quantiles for efficiency
- **Weighted Quantile Sketch**: For distributed computing

#### 3. Sparsity-Aware:
- Default direction for missing values
- Learns best direction during training

## 4. Mathematical Foundations <a id="mathematical-foundations"></a>

### AdaBoost Mathematics:

#### Weight Update Derivation:
Minimizing exponential loss leads to:
```
∂L/∂α_m = -Σ y_i × h_m(x_i) × exp(-y_i × F_{m-1}(x_i) - α_m × y_i × h_m(x_i))
```
Setting to zero gives optimal α_m

#### Training Error Bound:
```
Training_Error ≤ Π_m 2√(ε_m(1-ε_m))
```

#### Margin Theory:
Margin for sample i: margin_i = y_i × F(x_i) / ||F||_1

### XGBoost Mathematics:

#### Loss Function Expansion:
Taylor expansion of loss:
```
l(y_i, ŷ_i^(t-1) + f_t(x_i)) ≈ l(y_i, ŷ_i^(t-1)) + g_i × f_t(x_i) + (1/2) × h_i × f_t²(x_i)
```

#### Optimal Leaf Weights:
For a given tree structure:
```
w_j* = -Σ(i∈I_j) g_i / (Σ(i∈I_j) h_i + λ)
```

#### Split Gain:
```
Gain = (1/2) × [(Σ g_L)²/(Σ h_L + λ) + (Σ g_R)²/(Σ h_R + λ) - (Σ g)²/(Σ h + λ)] - γ
```

## 5. Implementation from Scratch <a id="implementation"></a>

### AdaBoost Implementation:
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostFromScratch:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
    def fit(self, X, y):
        n_samples = len(X)
        
        # Initialize weights uniformly
        sample_weight = np.ones(n_samples) / n_samples
        
        # Convert labels to {-1, 1}
        y_encoded = np.where(y == 0, -1, 1)
        
        for iboost in range(self.n_estimators):
            # Train weak learner
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred = estimator.predict(X)
            y_pred_encoded = np.where(y_pred == 0, -1, 1)
            
            # Calculate weighted error
            incorrect = y_pred_encoded != y_encoded
            estimator_error = np.average(incorrect, weights=sample_weight)
            
            # If perfect classifier or worse than random, stop
            if estimator_error <= 0:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.)
                self.estimator_errors_.append(0.)
                break
                
            if estimator_error >= 0.5:
                break
            
            # Calculate alpha (estimator weight)
            alpha = self.learning_rate * 0.5 * np.log(
                (1.0 - estimator_error) / estimator_error
            )
            
            # Update sample weights
            sample_weight *= np.exp(
                alpha * incorrect * 2  # incorrect is 0/1, we need -1/1 effect
            )
            
            # Normalize weights
            sample_weight /= sample_weight.sum()
            
            # Save estimator
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(estimator_error)
            
        return self
    
    def predict(self, X):
        # Weighted voting
        n_classes = 2  # Binary classification
        decision = self.decision_function(X)
        return np.where(decision > 0, 1, 0)
    
    def decision_function(self, X):
        decision = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            predictions_encoded = np.where(predictions == 0, -1, 1)
            decision += weight * predictions_encoded
            
        return decision
    
    def staged_predict(self, X):
        """Make predictions at each boosting iteration"""
        decision = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            predictions_encoded = np.where(predictions == 0, -1, 1)
            decision += weight * predictions_encoded
            yield np.where(decision > 0, 1, 0)
    
    def feature_importances_(self):
        """Calculate feature importances"""
        importances = np.zeros(self.estimators_[0].n_features_)
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            importances += estimator.feature_importances_ * weight
            
        importances /= np.sum(self.estimator_weights_)
        return importances

# Multiclass AdaBoost (SAMME)
class AdaBoostMulticlass:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimator_weights_ = []
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples = len(X)
        
        # Initialize weights
        sample_weight = np.ones(n_samples) / n_samples
        
        for iboost in range(self.n_estimators):
            # Train estimator
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred = estimator.predict(X)
            
            # Error
            incorrect = y_pred != y
            estimator_error = np.average(incorrect, weights=sample_weight)
            
            # Stop if perfect
            if estimator_error <= 0:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.)
                break
            
            # SAMME.R would use probability estimates
            # SAMME uses discrete outputs
            
            # Alpha with multi-class correction
            alpha = np.log((1 - estimator_error) / estimator_error) + \
                    np.log(self.n_classes_ - 1)
            
            # Update weights
            sample_weight *= np.exp(alpha * incorrect)
            sample_weight /= sample_weight.sum()
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            
        return self
```

### Simplified XGBoost Implementation:
```python
class XGBoostTree:
    """Single XGBoost tree"""
    def __init__(self, max_depth=3, min_child_weight=1, 
                 lambda_=1, gamma=0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.gamma = gamma
        self.tree = {}
        
    def fit(self, X, g, h):
        """Fit tree to gradients and hessians"""
        self.tree = self._build_tree(X, g, h, depth=0)
        return self
    
    def _build_tree(self, X, g, h, depth):
        n_samples = len(X)
        
        # Calculate best leaf weight for this node
        G = np.sum(g)
        H = np.sum(h)
        leaf_weight = -G / (H + self.lambda_)
        
        # Check stopping criteria
        if depth >= self.max_depth or n_samples < self.min_child_weight:
            return {'leaf': True, 'weight': leaf_weight}
        
        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if child too small
                if (np.sum(left_mask) < self.min_child_weight or 
                    np.sum(right_mask) < self.min_child_weight):
                    continue
                
                # Calculate gain
                G_L, H_L = np.sum(g[left_mask]), np.sum(h[left_mask])
                G_R, H_R = np.sum(g[right_mask]), np.sum(h[right_mask])
                
                gain = 0.5 * (
                    G_L**2 / (H_L + self.lambda_) +
                    G_R**2 / (H_R + self.lambda_) -
                    G**2 / (H + self.lambda_)
                ) - self.gamma
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        # If no good split found, return leaf
        if best_feature is None:
            return {'leaf': True, 'weight': leaf_weight}
        
        # Build children
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree(
            X[left_mask], g[left_mask], h[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], g[right_mask], h[right_mask], depth + 1
        )
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        if node['leaf']:
            return node['weight']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

class SimpleXGBoost:
    """Simplified XGBoost for regression"""
    def __init__(self, n_estimators=100, learning_rate=0.3,
                 max_depth=3, lambda_=1, gamma=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.trees = []
        
    def fit(self, X, y):
        # Initialize predictions
        self.base_prediction = np.mean(y)
        f = np.full(len(y), self.base_prediction)
        
        for i in range(self.n_estimators):
            # Calculate gradients and hessians (for squared loss)
            g = f - y  # gradient
            h = np.ones(len(y))  # hessian
            
            # Build tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma
            )
            tree.fit(X, g, h)
            
            # Update predictions
            update = tree.predict(X)
            f += self.learning_rate * update
            
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        f = np.full(len(X), self.base_prediction)
        
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
            
        return f

# XGBoost with custom objective
class XGBoostCustom:
    def __init__(self, objective, n_estimators=100):
        self.objective = objective
        self.n_estimators = n_estimators
        self.trees = []
        
    def fit(self, X, y):
        # Custom objectives should return (gradient, hessian)
        f = np.zeros(len(y))
        
        for i in range(self.n_estimators):
            g, h = self.objective(y, f)
            
            tree = XGBoostTree()
            tree.fit(X, g, h)
            
            f += tree.predict(X)
            self.trees.append(tree)
            
        return self

# Example custom objectives
def squared_loss(y_true, y_pred):
    grad = y_pred - y_true
    hess = np.ones_like(y_true)
    return grad, hess

def logistic_loss(y_true, y_pred):
    # For binary classification
    pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = pred - y_true
    hess = pred * (1.0 - pred)
    return grad, hess
```

## 6. Optimization Techniques <a id="optimization"></a>

### XGBoost Optimizations:

#### 1. Approximate Algorithm:
```python
def weighted_quantile_sketch(X, weights, n_bins):
    """Create quantile bins for approximate splits"""
    # Sort by feature value
    sorted_idx = np.argsort(X)
    sorted_X = X[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    # Calculate cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    
    # Find quantile thresholds
    thresholds = []
    for i in range(1, n_bins):
        target_weight = i * total_weight / n_bins
        idx = np.searchsorted(cum_weights, target_weight)
        thresholds.append(sorted_X[idx])
    
    return thresholds
```

#### 2. Sparsity-Aware Split Finding:
```python
def sparsity_aware_split(X, g, h, feature_idx):
    """Handle missing values in split finding"""
    # Separate missing and non-missing
    missing_mask = np.isnan(X[:, feature_idx])
    non_missing_mask = ~missing_mask
    
    # Statistics for missing values
    g_missing = np.sum(g[missing_mask])
    h_missing = np.sum(h[missing_mask])
    
    # Try default direction: missing goes left
    best_gain_left = calculate_gain_with_missing_left(...)
    
    # Try default direction: missing goes right
    best_gain_right = calculate_gain_with_missing_right(...)
    
    # Choose better direction
    if best_gain_left > best_gain_right:
        return 'left', best_gain_left
    else:
        return 'right', best_gain_right
```

#### 3. Column Block Structure:
```python
class ColumnBlock:
    """Store data in column-wise blocks for cache efficiency"""
    def __init__(self, X):
        self.blocks = []
        n_features = X.shape[1]
        
        # Group features into blocks
        block_size = 16  # Features per block
        
        for i in range(0, n_features, block_size):
            end = min(i + block_size, n_features)
            block = X[:, i:end].T.copy()  # Transpose for column access
            self.blocks.append(block)
```

### Hardware Optimizations:

#### GPU Acceleration:
```python
# Pseudo-code for GPU histogram building
def gpu_build_histogram(X_gpu, g_gpu, h_gpu, bins):
    """Build histogram on GPU"""
    # Parallel histogram construction
    # Each thread handles subset of samples
    # Atomic operations for histogram updates
    pass
```

#### Parallel Tree Construction:
```python
from multiprocessing import Pool

def parallel_tree_build(X, g, h, features_per_thread=10):
    """Build tree with parallel split finding"""
    n_features = X.shape[1]
    n_threads = (n_features + features_per_thread - 1) // features_per_thread
    
    with Pool(n_threads) as pool:
        # Each thread finds best split for subset of features
        results = pool.map(
            find_best_split_subset,
            [(X, g, h, start, end) for start, end in feature_ranges]
        )
    
    # Select globally best split
    return max(results, key=lambda x: x['gain'])
```

## 7. Hyperparameter Tuning <a id="hyperparameters"></a>

### AdaBoost Hyperparameters:

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| n_estimators | Number of weak learners | 50-500 | More = better, slower |
| learning_rate | Shrinkage parameter | 0.01-1.0 | Lower = need more trees |
| base_estimator | Weak learner type | DecisionStump | Complexity trade-off |
| algorithm | SAMME or SAMME.R | - | R uses probabilities |

### XGBoost Hyperparameters:

#### Tree-specific:
| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| max_depth | Maximum tree depth | 3-10 | Deeper = more complex |
| min_child_weight | Minimum sum of hessian | 1-10 | Higher = more conservative |
| gamma | Minimum loss reduction | 0-5 | Higher = more conservative |
| subsample | Row subsampling | 0.5-1.0 | Lower = more robust |
| colsample_bytree | Column subsampling | 0.5-1.0 | Lower = more robust |

#### Regularization:
| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| alpha | L1 regularization | 0-10 | Sparse solutions |
| lambda | L2 regularization | 0-10 | Smooth solutions |
| max_delta_step | Max delta step | 0-10 | Helpful for imbalanced |

#### Learning:
| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| learning_rate | Step size shrinkage | 0.01-0.3 | Lower = need more trees |
| n_estimators | Number of trees | 100-1000 | More = better, slower |

### Tuning Strategy:
```python
# Grid search example
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300]
}

# Start with important parameters
# 1. Fix learning_rate and tune n_estimators
# 2. Tune max_depth and min_child_weight
# 3. Tune gamma
# 4. Tune subsample and colsample_bytree
# 5. Fine-tune learning_rate
```

## 8. Practical Applications <a id="applications"></a>

### When to Use AdaBoost:
1. **Binary classification** with clear decision boundary
2. **Feature selection** through weak learner analysis
3. **Interpretability** is important
4. **Theoretical guarantees** needed
5. **Face detection** (Viola-Jones)

### When to Use XGBoost:
1. **Kaggle competitions** - tabular data
2. **High accuracy** requirements
3. **Large datasets** with missing values
4. **Custom objectives** needed
5. **Feature importance** analysis

### Real-world Examples:

#### Credit Scoring with XGBoost:
```python
# Feature engineering for credit scoring
def create_credit_features(df):
    features = []
    
    # Ratios
    features.append(df['debt'] / (df['income'] + 1))
    features.append(df['credit_utilization'])
    
    # Historical
    features.append(df['months_since_last_default'])
    features.append(df['payment_history_score'])
    
    # Behavioral
    features.append(df['num_credit_inquiries'])
    
    return np.column_stack(features)

# Custom objective for profit optimization
def profit_objective(y_true, y_pred):
    # Asymmetric loss: false negatives cost more
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Cost matrix
    fn_cost = 10  # Cost of approving bad loan
    fp_cost = 1   # Cost of rejecting good loan
    
    grad = np.where(y_true == 1, 
                    (p - 1) * fn_cost,
                    p * fp_cost)
    
    hess = p * (1 - p) * (fn_cost + fp_cost)
    
    return grad, hess
```

#### Click-Through Rate Prediction:
```python
# Handle categorical features
import xgboost as xgb

# Prepare data with categorical encoding
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

# Parameters for CTR prediction
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
}
```

## 9. Advanced Topics <a id="advanced-topics"></a>

### 1. XGBoost with Monotonic Constraints:
```python
# Ensure feature relationships make sense
params = {
    'monotone_constraints': '(1,-1,0)',  # increase, decrease, no constraint
}
```

### 2. Early Stopping with Validation:
```python
# Prevent overfitting
eval_set = [(X_train, y_train), (X_val, y_val)]
model = xgb.XGBClassifier()
model.fit(X_train, y_train,
          eval_set=eval_set,
          early_stopping_rounds=10,
          verbose=True)
```

### 3. Feature Interactions:
```python
# Limit interaction depth
params = {
    'interaction_constraints': [
        [0, 1],  # Features 0 and 1 can interact
        [2, 3, 4]  # Features 2, 3, 4 can interact
    ]
}
```

### 4. Dart Booster (Dropouts):
```python
# Prevent over-specialization
params = {
    'booster': 'dart',
    'sample_type': 'uniform',
    'normalize_type': 'tree',
    'rate_drop': 0.1,
    'skip_drop': 0.5
}
```

### 5. Multi-output XGBoost:
```python
class MultiOutputXGBoost:
    def __init__(self, **kwargs):
        self.models = []
        self.kwargs = kwargs
        
    def fit(self, X, Y):
        # Y is n_samples x n_outputs
        n_outputs = Y.shape[1]
        
        for i in range(n_outputs):
            model = xgb.XGBRegressor(**self.kwargs)
            model.fit(X, Y[:, i])
            self.models.append(model)
            
        return self
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.column_stack(predictions)
```

## 10. Interview Questions & Answers <a id="interview-qa"></a>

### Q1: What is AdaBoost and how does it work?
**Answer**: AdaBoost (Adaptive Boosting) is an ensemble method that:
1. Trains weak learners sequentially
2. Each learner focuses on mistakes of previous ones by reweighting samples
3. Misclassified samples get higher weights
4. Combines learners with weighted voting based on accuracy
5. Minimizes exponential loss: exp(-y × f(x))

Key insight: Converts weak learners (>50% accuracy) into strong ensemble

### Q2: How does XGBoost differ from traditional Gradient Boosting?
**Answer**: XGBoost improvements:
1. **Regularization**: L1/L2 penalties on leaf weights
2. **Second-order approximation**: Uses Hessian, not just gradient
3. **Sparsity handling**: Built-in missing value support
4. **Parallel processing**: Column-wise parallelization
5. **Tree pruning**: Backward pruning after max_depth
6. **Hardware optimization**: Cache-aware, GPU support
7. **Built-in CV**: Integrated cross-validation

Result: 10x+ faster, better accuracy, less overfitting

### Q3: Explain the weight update formula in AdaBoost
**Answer**: 
```
w_i^(t+1) = w_i^(t) × exp(-α_t × y_i × h_t(x_i))
```
Breaking it down:
- If correctly classified: y_i × h_t(x_i) = 1, weight multiplied by exp(-α_t) < 1
- If misclassified: y_i × h_t(x_i) = -1, weight multiplied by exp(α_t) > 1
- α_t larger for better classifiers, so they have more influence
- Exponential increase ensures focus on hard examples

### Q4: What is the mathematical objective of XGBoost?
**Answer**: XGBoost minimizes:
```
Obj = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
```
Where:
- l(y_i, ŷ_i) = loss function (customizable)
- Ω(f_k) = γT + (1/2)λΣw_j² (regularization)
- T = number of leaves
- w_j = leaf weights

This penalizes complex trees and prevents overfitting

### Q5: How does XGBoost handle missing values?
**Answer**: XGBoost learns optimal default directions:
1. During training, tries both directions (left/right) for missing values
2. Chooses direction that gives maximum gain
3. Stores default direction in model
4. At prediction, missing values follow learned direction
5. No imputation needed - handles sparsity natively

Advantage: Captures patterns in missingness

### Q6: Compare AdaBoost and XGBoost performance characteristics
**Answer**:
| Aspect | AdaBoost | XGBoost |
|--------|----------|---------|
| Speed | Moderate | Very fast |
| Accuracy | Good | Excellent |
| Overfitting | Resistant | Need regularization |
| Missing values | No support | Native support |
| Parallel | No | Yes |
| Memory | Low | High |
| Interpretability | Higher | Lower |
| Hyperparameters | Few | Many |

### Q7: What are the key hyperparameters in XGBoost?
**Answer**: Most important:
1. **n_estimators**: Number of trees (more = better but slower)
2. **max_depth**: Tree depth (3-10, deeper = more complex)
3. **learning_rate**: Shrinkage (0.01-0.3, lower = need more trees)
4. **min_child_weight**: Minimum hessian sum (higher = conservative)
5. **subsample**: Row sampling (0.5-1.0, lower = robust)
6. **colsample_bytree**: Column sampling (0.5-1.0)
7. **gamma**: Minimum split gain (0-5, pruning threshold)
8. **reg_alpha/lambda**: L1/L2 regularization

### Q8: When would you use AdaBoost vs XGBoost?
**Answer**:
**Use AdaBoost when**:
- Need interpretability
- Simple binary classification
- Theoretical guarantees important
- Limited computational resources
- Working with clean data

**Use XGBoost when**:
- Need maximum accuracy
- Have missing values
- Large datasets
- Custom loss functions
- Kaggle competitions
- Need speed and scalability

### Q9: Explain the concept of "stage-wise additive modeling"
**Answer**: Stage-wise additive modeling:
1. Start with initial prediction (often constant)
2. Iteratively add new functions: F_m(x) = F_{m-1}(x) + γ_m × h_m(x)
3. Each h_m is fit to improve current model
4. γ_m is step size/shrinkage
5. Final model is sum of all functions

Benefits: Simple optimization, interpretable process, natural regularization

### Q10: How does XGBoost achieve parallelization in tree building?
**Answer**: XGBoost parallelizes at multiple levels:
1. **Column-wise**: Each thread handles subset of features
2. **Split finding**: Parallel computation of gains
3. **Histogram building**: Distributed binning
4. **Prediction**: Parallel tree traversal

NOT parallelized: Sequential boosting (trees depend on previous)

Key: Pre-sorted data in column blocks for cache efficiency

### Q11: What is the role of second-order derivatives in XGBoost?
**Answer**: Hessian (second derivative) provides:
1. **Better approximation**: Taylor expansion more accurate
2. **Newton's method**: Faster convergence
3. **Adaptive learning**: Natural step size from H
4. **Leaf weights**: w* = -G/H (gradient/hessian)
5. **Custom objectives**: Any twice-differentiable loss

Example: For squared loss, H=1; for log loss, H=p(1-p)

### Q12: How do you handle imbalanced data in AdaBoost and XGBoost?
**Answer**:
**AdaBoost**:
- Initial sample weights proportional to class inverse frequency
- Cost-sensitive variant with different misclassification costs
- SMOTE before training

**XGBoost**:
- scale_pos_weight parameter: ratio of negative to positive class
- Custom objective with asymmetric loss
- max_delta_step for bounded updates
- AUC as eval_metric instead of accuracy

### Q13: Explain gradient and hessian calculation for common losses
**Answer**:
**Squared Loss (Regression)**:
- Gradient: g = ŷ - y
- Hessian: h = 1

**Logistic Loss (Binary)**:
- p = sigmoid(ŷ)
- Gradient: g = p - y
- Hessian: h = p(1-p)

**Poisson Loss (Count)**:
- Gradient: g = exp(ŷ) - y
- Hessian: h = exp(ŷ)

### Q14: What is early stopping and how is it implemented?
**Answer**: Early stopping prevents overfitting by:
1. Monitor validation metric during training
2. Stop when metric doesn't improve for N rounds
3. Return model from best iteration

Implementation:
```python
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=10,
          eval_metric='auc')
```

Benefits: Automatic regularization, saves computation

### Q15: How do you interpret XGBoost feature importance?
**Answer**: Three types:
1. **Gain**: Average gain when feature is used
2. **Cover**: Average coverage (# samples) when used
3. **Frequency**: How often feature is used

Interpretation:
- High gain = powerful when used
- High frequency = consistently useful
- High cover = affects many samples

Caution: Biased towards continuous features

### Q16: What are monotonic constraints in XGBoost?
**Answer**: Force feature relationships:
- Monotone increasing (+1): Higher feature → higher prediction
- Monotone decreasing (-1): Higher feature → lower prediction
- No constraint (0): Any relationship

Use cases:
- Price elasticity (price ↑ → demand ↓)
- Credit score (score ↑ → risk ↓)
- Age vs insurance (age ↑ → premium ↑)

Ensures model follows domain knowledge

### Q17: Explain the DART booster in XGBoost
**Answer**: DART (Dropouts meet Multiple Additive Regression Trees):
- Applies dropout to trees during training
- Prevents over-specialization of trees
- Each iteration: randomly drop previous trees
- Normalize contributions to maintain scale

Benefits: Better generalization, reduces overfitting
Trade-off: Slower training, more trees needed

### Q18: How would you debug poor XGBoost performance?
**Answer**: Systematic approach:
1. **Check data**: Missing values, outliers, scale
2. **Visualize predictions**: Plot predicted vs actual
3. **Feature importance**: Ensure sensible features selected
4. **Learning curves**: Check train/val gap
5. **Tree inspection**: Examine individual trees
6. **Hyperparameter sensitivity**: Grid search key params
7. **Cross-validation**: Ensure stable performance
8. **Error analysis**: Examine misclassified samples

### Q19: What is the computational complexity of AdaBoost and XGBoost?
**Answer**:
**AdaBoost**:
- Training: O(T × n × m × d) 
- Prediction: O(T × d)
Where T=trees, n=samples, m=features, d=depth

**XGBoost**:
- Training: O(T × n × m × log(n)) with sorting
- With histogram: O(T × n × m × B) where B=bins
- Prediction: O(T × d)

XGBoost optimizations make it much faster in practice

### Q20: What are some alternatives to AdaBoost and XGBoost?
**Answer**:
1. **LightGBM**: Leaf-wise growth, faster than XGBoost
2. **CatBoost**: Better categorical handling, less tuning
3. **HistGradientBoosting**: Scikit-learn's fast implementation
4. **NGBoost**: Natural gradient boosting for uncertainty
5. **Random Forest**: When interpretability needed
6. **Neural Networks**: Unstructured data, very large datasets
7. **Linear Models**: When relationships are linear

Choice depends on: data type, size, interpretability needs, deployment constraints