# Day 17: Random Forest

## Table of Contents
1. [Introduction to Random Forest](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Bootstrap Aggregating (Bagging)](#bagging)
4. [Random Feature Selection](#feature-selection)
5. [Implementation from Scratch](#implementation)
6. [Out-of-Bag (OOB) Error](#oob-error)
7. [Feature Importance](#feature-importance)
8. [Hyperparameters](#hyperparameters)
9. [Advanced Topics](#advanced-topics)
10. [Interview Questions & Answers](#interview-qa)

## 1. Introduction to Random Forest <a id="introduction"></a>

Random Forest is an **ensemble learning method** that constructs multiple decision trees during training and outputs the mode of classes (classification) or mean prediction (regression) of individual trees.

### Key Concepts:
- **Ensemble of Decision Trees**: Combines multiple weak learners
- **Bootstrap Sampling**: Each tree trained on different sample
- **Random Feature Selection**: Random subset of features at each split
- **Voting/Averaging**: Aggregate predictions from all trees
- **Variance Reduction**: Reduces overfitting compared to single tree

### Why Random Forest Works:
1. **Law of Large Numbers**: Average of many trees converges to expected value
2. **Decorrelation**: Random features make trees less correlated
3. **Bias-Variance Tradeoff**: Low bias (deep trees) + reduced variance (averaging)

## 2. Mathematical Foundation <a id="mathematical-foundation"></a>

### Ensemble Prediction:

#### For Classification:
```
ŷ = mode{h₁(x), h₂(x), ..., hₙ(x)}
```
Or with probabilities:
```
P(y=c|x) = (1/n) × Σᵢ₌₁ⁿ P(y=c|x, Tᵢ)
```

#### For Regression:
```
ŷ = (1/n) × Σᵢ₌₁ⁿ hᵢ(x)
```

### Variance Reduction:
For uncorrelated trees with variance σ²:
```
Var(RF) = σ²/n
```

For correlated trees with correlation ρ:
```
Var(RF) = ρσ² + (1-ρ)σ²/n
```

As n → ∞: Var(RF) → ρσ²

### Error Bound (Breiman):
```
PE* ≤ ρ̄(1-s²)/s²
```
Where:
- PE* = generalization error
- ρ̄ = mean correlation between trees
- s = strength of trees

## 3. Bootstrap Aggregating (Bagging) <a id="bagging"></a>

### Bootstrap Sampling:
- Sample n instances with replacement from dataset of size n
- Each bootstrap sample has ~63.2% unique instances
- Remaining ~36.8% are out-of-bag (OOB) samples

### Mathematical Probability:
```
P(instance not selected) = (1 - 1/n)ⁿ → 1/e ≈ 0.368 as n → ∞
```

### Bagging Algorithm:
```
for i = 1 to n_trees:
    D_i = bootstrap_sample(D)
    h_i = train_tree(D_i)
    
prediction = aggregate(h_1, h_2, ..., h_n)
```

## 4. Random Feature Selection <a id="feature-selection"></a>

### Feature Sampling Strategy:
At each node, randomly select m features from p total features:
- **Classification**: m = √p (square root of total features)
- **Regression**: m = p/3 (one-third of features)

### Effect on Trees:
- Reduces correlation between trees
- Forces trees to use different features
- Increases diversity in ensemble

## 5. Implementation from Scratch <a id="implementation"></a>

```python
import numpy as np
from collections import Counter
import random

class DecisionTreeRF:
    """Decision tree for Random Forest with random feature selection"""
    def __init__(self, max_depth=None, min_samples_split=2, 
                 n_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.criterion = criterion
        self.tree = None
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        # Set number of features to consider
        if self.n_features is None:
            self.n_features = int(np.sqrt(self.n_features_))
        
        self.tree = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth if self.max_depth else False) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}
        
        # Random feature selection
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        # Find best split among random features
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}
        
        # Split
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        
        # Grow children
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {'leaf': False, 'feature': best_feat, 'threshold': best_thresh,
                'left': left, 'right': right}
    
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        best_feat = None
        best_thresh = None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold
                    
        return best_feat, best_thresh
    
    def _information_gain(self, y, X_column, threshold):
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Split
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Weighted impurity
        n = len(y)
        n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
        e_l, e_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r
        
        return parent_impurity - child_impurity
    
    def _impurity(self, y):
        proportions = np.bincount(y) / len(y)
        
        if self.criterion == 'gini':
            return 1 - np.sum(proportions ** 2)
        else:  # entropy
            return -np.sum([p * np.log2(p) for p in proportions if p > 0])
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features='sqrt',
                 bootstrap=True, oob_score=False, n_jobs=1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees = []
        self.oob_score_ = None
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        
        # Determine max_features
        if self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            self.max_features_ = int(np.log2(self.n_features))
        else:
            self.max_features_ = self.max_features
        
        # OOB initialization
        if self.oob_score:
            self.oob_predictions = np.zeros((len(X), self.n_classes))
            self.oob_count = np.zeros(len(X))
        
        # Build trees
        for i in range(self.n_estimators):
            tree = DecisionTreeRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.max_features_
            )
            
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(len(X), len(X), replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
                
                # Track OOB samples
                if self.oob_score:
                    oob_indices = np.setdiff1d(np.arange(len(X)), indices)
                    if len(oob_indices) > 0:
                        predictions = tree.fit(X_sample, y_sample).predict(X[oob_indices])
                        for idx, pred in zip(oob_indices, predictions):
                            self.oob_predictions[idx, pred] += 1
                            self.oob_count[idx] += 1
                else:
                    tree.fit(X_sample, y_sample)
            else:
                tree.fit(X, y)
            
            self.trees.append(tree)
        
        # Calculate OOB score
        if self.oob_score:
            self.oob_score_ = self._calculate_oob_score(y)
            
        return self
    
    def _calculate_oob_score(self, y):
        oob_pred = []
        for i in range(len(y)):
            if self.oob_count[i] > 0:
                # Majority vote
                pred = np.argmax(self.oob_predictions[i])
                oob_pred.append(pred == y[i])
        
        return np.mean(oob_pred) if oob_pred else 0
    
    def predict(self, X):
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] 
                        for i in range(len(X))])
    
    def predict_proba(self, X):
        # Collect predictions
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate probabilities
        probas = np.zeros((len(X), self.n_classes))
        for i in range(len(X)):
            counts = np.bincount(predictions[:, i], minlength=self.n_classes)
            probas[i] = counts / self.n_estimators
            
        return probas

# Advanced Random Forest with parallel processing
from multiprocessing import Pool

class ParallelRandomForest(RandomForestClassifier):
    def _build_tree(self, args):
        X, y, tree_params = args
        tree = DecisionTreeRF(**tree_params)
        
        if self.bootstrap:
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree.fit(X_sample, y_sample)
            
            # Return tree and OOB info
            oob_indices = np.setdiff1d(np.arange(len(X)), indices)
            return tree, oob_indices
        else:
            tree.fit(X, y)
            return tree, None
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        
        # Tree parameters
        tree_params = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'n_features': self.max_features_
        }
        
        # Parallel tree building
        with Pool(self.n_jobs) as pool:
            args = [(X, y, tree_params) for _ in range(self.n_estimators)]
            results = pool.map(self._build_tree, args)
        
        self.trees = [tree for tree, _ in results]
        return self

# Random Forest Regressor
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features='auto'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        self.n_features = X.shape[1]
        
        # Determine max_features
        if self.max_features == 'auto':
            self.max_features_ = self.n_features // 3
        elif self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(self.n_features))
        else:
            self.max_features_ = self.max_features
        
        # Build trees
        for _ in range(self.n_estimators):
            tree = DecisionTreeRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.max_features_
            )
            
            # Bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
```

## 6. Out-of-Bag (OOB) Error <a id="oob-error"></a>

### OOB Score Calculation:
```python
def calculate_oob_error(forest, X, y):
    n_samples = len(X)
    oob_predictions = np.zeros((n_samples, forest.n_classes))
    oob_count = np.zeros(n_samples)
    
    for tree, oob_indices in zip(forest.trees, forest.oob_indices_list):
        if len(oob_indices) > 0:
            predictions = tree.predict(X[oob_indices])
            for idx, pred in zip(oob_indices, predictions):
                oob_predictions[idx, pred] += 1
                oob_count[idx] += 1
    
    # Get predictions
    oob_pred_labels = []
    for i in range(n_samples):
        if oob_count[i] > 0:
            pred = np.argmax(oob_predictions[i])
            oob_pred_labels.append(pred)
        else:
            oob_pred_labels.append(-1)  # No OOB prediction
    
    # Calculate accuracy
    valid_predictions = [i for i in range(n_samples) if oob_pred_labels[i] != -1]
    accuracy = np.mean([oob_pred_labels[i] == y[i] for i in valid_predictions])
    
    return accuracy, oob_pred_labels
```

### Properties:
- Unbiased estimate of generalization error
- No need for separate validation set
- Each sample used for validation ~37% of trees

## 7. Feature Importance <a id="feature-importance"></a>

### Mean Decrease Impurity (MDI):
```python
def calculate_feature_importance(forest):
    importances = np.zeros(forest.n_features)
    
    for tree in forest.trees:
        tree_importances = tree.feature_importances_
        importances += tree_importances
    
    # Average over all trees
    importances /= forest.n_estimators
    
    # Normalize
    importances /= np.sum(importances)
    
    return importances
```

### Permutation Importance:
```python
def permutation_importance(forest, X, y, n_repeats=10):
    baseline_score = forest.score(X, y)
    importances = []
    
    for feature in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])
            score = forest.score(X_permuted, y)
            scores.append(baseline_score - score)
        
        importances.append(np.mean(scores))
    
    return np.array(importances)
```

## 8. Hyperparameters <a id="hyperparameters"></a>

### Key Hyperparameters:

1. **n_estimators**: Number of trees
   - More trees = better performance (diminishing returns)
   - More trees = more computation
   - Default: 100

2. **max_features**: Features to consider at each split
   - sqrt: √p features (classification)
   - log2: log₂(p) features
   - auto: p/3 features (regression)
   - Lower values = more randomness

3. **max_depth**: Maximum tree depth
   - None: nodes expanded until pure
   - Controls overfitting
   - Deeper = more complex

4. **min_samples_split**: Minimum samples to split node
   - Higher values = simpler trees
   - Default: 2

5. **min_samples_leaf**: Minimum samples in leaf
   - Smooths model
   - Default: 1

6. **bootstrap**: Whether to use bootstrap samples
   - True: standard Random Forest
   - False: uses entire dataset

### Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

## 9. Advanced Topics <a id="advanced-topics"></a>

### 1. Extremely Randomized Trees:
```python
class ExtraTreesClassifier:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        
    def _build_tree(self, X, y):
        # Random splits instead of best splits
        # Faster training, more randomness
        pass
```

### 2. Feature Selection with RF:
```python
def select_features_rf(X, y, threshold=0.01):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.where(importances > threshold)[0]
    
    return X[:, indices], indices
```

### 3. Proximity Matrix:
```python
def compute_proximity_matrix(forest, X):
    n_samples = len(X)
    proximity = np.zeros((n_samples, n_samples))
    
    for tree in forest.trees:
        # Get leaf indices for all samples
        leaves = tree.apply(X)
        
        # Samples in same leaf are proximate
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if leaves[i] == leaves[j]:
                    proximity[i, j] += 1
                    proximity[j, i] += 1
    
    # Normalize by number of trees
    proximity /= forest.n_estimators
    
    return proximity
```

### 4. Handling Imbalanced Data:
```python
class BalancedRandomForest:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        # Balance each bootstrap sample
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Balanced bootstrap
            indices = []
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                sampled = np.random.choice(cls_indices, min_class_count)
                indices.extend(sampled)
            
            # Train tree on balanced sample
            tree = DecisionTreeRF()
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
```

## 10. Interview Questions & Answers <a id="interview-qa"></a>

### Q1: What is Random Forest and how does it work?
**Answer**: Random Forest is an ensemble learning method that:
1. Creates multiple decision trees using bootstrap samples
2. At each split, considers random subset of features (not all)
3. Makes predictions by averaging (regression) or voting (classification)
4. Reduces overfitting through averaging and decorrelation

Key innovations: Bootstrap aggregating + random feature selection

### Q2: Why does Random Forest work better than a single Decision Tree?
**Answer**: 
1. **Variance Reduction**: Averaging multiple trees reduces variance
2. **Decorrelation**: Random features make trees less correlated
3. **Overfitting Protection**: Individual trees overfit, but average doesn't
4. **Bias-Variance**: Maintains low bias of deep trees while reducing variance
5. **Robustness**: Less sensitive to outliers and noise

### Q3: Explain the role of randomness in Random Forest
**Answer**: Two sources of randomness:
1. **Bootstrap Sampling**: Each tree sees different data subset
   - Creates diversity in trees
   - Enables OOB error estimation
2. **Random Feature Selection**: Each split considers feature subset
   - Decorrelates trees
   - Forces trees to use different features
   - Reduces impact of strong predictors

### Q4: What is Out-of-Bag (OOB) error?
**Answer**: OOB error is a way to estimate generalization error without separate validation set:
- Each tree trained on ~63% of data (bootstrap)
- Remaining ~37% are "out-of-bag" samples
- Use each tree to predict its OOB samples
- Average predictions across all trees where sample was OOB
- Provides unbiased error estimate

### Q5: How does Random Forest calculate feature importance?
**Answer**: Two main methods:

1. **Mean Decrease Impurity (MDI)**:
   - Sum of impurity decreases for each feature across all trees
   - Weighted by number of samples reaching node
   - Fast but biased towards high-cardinality features

2. **Permutation Importance**:
   - Shuffle feature values and measure accuracy decrease
   - More reliable but computationally expensive
   - Shows actual impact on predictions

### Q6: Compare Random Forest with Gradient Boosting
**Answer**:
| Aspect | Random Forest | Gradient Boosting |
|--------|--------------|-------------------|
| Training | Parallel | Sequential |
| Tree depth | Deep | Shallow |
| Bias-Variance | Low bias, reduces variance | Reduces bias |
| Speed | Faster training | Slower training |
| Overfitting | More robust | Prone to overfitting |
| Interpretability | Feature importance | Less interpretable |
| Hyperparameters | Fewer critical | More sensitive |

### Q7: What are the key hyperparameters in Random Forest?
**Answer**:
1. **n_estimators**: Number of trees (more is generally better)
2. **max_features**: Features per split (sqrt for classification, /3 for regression)
3. **max_depth**: Tree depth (None for fully grown)
4. **min_samples_split**: Minimum samples to split
5. **min_samples_leaf**: Minimum samples in leaf
6. **bootstrap**: Whether to use bootstrap sampling

Most important: n_estimators and max_features

### Q8: How do you handle imbalanced datasets with Random Forest?
**Answer**:
1. **Class weights**: Weight samples inversely to class frequency
2. **Balanced bootstrap**: Equal samples from each class per tree
3. **Threshold adjustment**: Modify decision threshold for classification
4. **Stratified sampling**: Maintain class distribution in bootstrap
5. **SMOTE + RF**: Synthetic oversampling before training
6. **Cost-sensitive**: Modify split criterion to consider costs

### Q9: What are the advantages and disadvantages of Random Forest?
**Answer**:
**Advantages**:
- High accuracy without tuning
- Handles non-linear relationships
- Robust to outliers and noise
- No feature scaling needed
- Feature importance built-in
- Can handle missing values
- Parallel training possible

**Disadvantages**:
- Less interpretable than single tree
- Large memory footprint
- Slow prediction for many trees
- Can overfit with noisy data
- Biased towards categorical variables with many levels

### Q10: When should you use Random Forest vs other algorithms?
**Answer**:
**Use Random Forest when**:
- Need high accuracy with minimal tuning
- Have mixed numerical/categorical features
- Want feature importance
- Dataset has non-linear relationships
- Need robust predictions

**Consider alternatives when**:
- Need interpretability (use single tree)
- Have very high-dimensional sparse data (use linear models)
- Need fast predictions (use simpler models)
- Have sequential/time-series data (use RNN/ARIMA)

### Q11: Explain how Random Forest handles missing values
**Answer**: Several approaches:
1. **Surrogate splits**: Find correlated features for splitting
2. **Imputation before training**: Fill missing values
3. **Missing value branch**: Separate branch for missing
4. **Proximity-based imputation**: Use RF proximity matrix
5. **Built-in handling**: Some implementations handle natively

Best practice: Understand your data and choose appropriate method

### Q12: What is the difference between Random Forest and Extra Trees?
**Answer**:
| Feature | Random Forest | Extra Trees |
|---------|--------------|-------------|
| Split selection | Best among random features | Random splits |
| Bootstrap | Yes | No (uses full dataset) |
| Training speed | Slower | Faster |
| Variance | Lower | Higher |
| Bias | Similar | Similar |
| Randomness | Moderate | Maximum |

Extra Trees: More randomness, faster training, sometimes better generalization

### Q13: How do you interpret Random Forest predictions?
**Answer**: Several methods:
1. **Feature importance**: Which features matter most
2. **Partial dependence plots**: Feature effect on predictions
3. **SHAP values**: Individual prediction explanations
4. **Proximity analysis**: Similar instances
5. **Tree extraction**: Look at individual trees
6. **LIME**: Local interpretable explanations

### Q14: What is the computational complexity of Random Forest?
**Answer**:
**Training**:
- Single tree: O(n × log n × m × d)
- Forest: O(k × n × log n × m × d)
Where: k=trees, n=samples, m=features considered, d=depth

**Prediction**:
- Single instance: O(k × d)
- n instances: O(n × k × d)

**Memory**: O(k × n) for storing all trees

### Q15: How does max_features affect Random Forest performance?
**Answer**:
- **Lower max_features**: More randomness, less correlation, higher bias
- **Higher max_features**: Less randomness, more correlation, lower bias
- **sqrt(p)**: Good for classification (high dimensionality)
- **p/3**: Good for regression
- **log2(p)**: Very high dimensionality

Trade-off: Decorrelation vs individual tree strength

### Q16: Explain bootstrap aggregating (bagging)
**Answer**: Bagging is:
1. **Bootstrap**: Sample n instances with replacement
2. **Aggregate**: Combine predictions (vote/average)

Benefits:
- Reduces variance without increasing bias
- Each model sees different data
- Natural way to estimate uncertainty
- Enables parallel training

Mathematical: If trees have variance σ², bagged ensemble has variance ≈ σ²/n

### Q17: Can Random Forest be used for feature selection?
**Answer**: Yes, several methods:
1. **Feature importance threshold**: Select features above threshold
2. **Recursive elimination**: Remove least important iteratively
3. **Boruta algorithm**: Compare with shadow features
4. **Permutation importance**: More reliable selection

Example approach:
- Train RF, get importances
- Select top-k features
- Retrain model with selected features

### Q18: How do you optimize Random Forest performance?
**Answer**:
1. **More trees**: Generally improves until plateau
2. **Feature engineering**: Better features help
3. **Hyperparameter tuning**: Grid/random search
4. **Parallel processing**: Use all cores
5. **Reduce tree size**: For faster predictions
6. **Feature selection**: Remove irrelevant features
7. **Data preprocessing**: Handle outliers, missing values

### Q19: What are some variations of Random Forest?
**Answer**:
1. **Extremely Randomized Trees**: Random splits
2. **Conditional Inference Forests**: Statistical stopping
3. **Quantile Regression Forests**: Predict distributions
4. **Survival Random Forests**: Time-to-event data
5. **Rotation Forests**: PCA-based feature extraction
6. **Oblique Random Forests**: Linear combination splits

### Q20: What are real-world applications of Random Forest?
**Answer**:
1. **Banking**: Credit scoring, fraud detection
2. **Healthcare**: Disease prediction, drug discovery
3. **E-commerce**: Recommendation systems, customer churn
4. **Finance**: Stock prediction, risk assessment
5. **Marketing**: Customer segmentation, campaign response
6. **Manufacturing**: Quality control, predictive maintenance
7. **Ecology**: Species distribution modeling
8. **Remote Sensing**: Land cover classification