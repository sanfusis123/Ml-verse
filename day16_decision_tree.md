# Day 16: Decision Trees

## Table of Contents
1. [Introduction to Decision Trees](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Splitting Criteria](#splitting-criteria)
4. [Tree Building Algorithm](#tree-building)
5. [Implementation from Scratch](#implementation)
6. [Pruning Techniques](#pruning)
7. [Handling Different Data Types](#data-types)
8. [Advantages and Disadvantages](#pros-cons)
9. [Advanced Topics](#advanced-topics)
10. [Interview Questions & Answers](#interview-qa)

## 1. Introduction to Decision Trees <a id="introduction"></a>

Decision Trees are **non-parametric** supervised learning algorithms used for both classification and regression. They learn simple decision rules inferred from data features to create a model that predicts target values.

### Key Concepts:
- **Root Node**: Top node representing entire dataset
- **Internal Nodes**: Decision nodes that split data
- **Leaf Nodes**: Terminal nodes with predictions
- **Branches**: Outcomes of decisions
- **Depth**: Length of longest path from root to leaf

### Visual Structure:
```
                 [Age > 30?]
                /           \
              Yes            No
              /               \
      [Income > 50k?]     [Student?]
         /      \           /      \
       Yes      No        Yes      No
       /         \        /         \
    Approve    Reject  Reject    Approve
```

## 2. Mathematical Foundation <a id="mathematical-foundation"></a>

### Information Theory Basics:

#### Entropy (Measure of Impurity):
```
H(S) = -∑(i=1 to c) p_i × log₂(p_i)
```
Where:
- S = dataset
- c = number of classes
- p_i = proportion of samples in class i

#### Information Gain:
```
IG(S, A) = H(S) - ∑(v∈Values(A)) |S_v|/|S| × H(S_v)
```
Where:
- A = attribute to split on
- S_v = subset where attribute A has value v

#### Gini Impurity:
```
Gini(S) = 1 - ∑(i=1 to c) p_i²
```

#### Gain Ratio (handles attribute bias):
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

SplitInfo(S, A) = -∑(v∈Values(A)) |S_v|/|S| × log₂(|S_v|/|S|)
```

## 3. Splitting Criteria <a id="splitting-criteria"></a>

### For Classification:

1. **Information Gain (ID3)**:
   - Measures reduction in entropy
   - Biased towards attributes with many values

2. **Gain Ratio (C4.5)**:
   - Normalizes information gain
   - Reduces bias towards multi-valued attributes

3. **Gini Index (CART)**:
   - Measures impurity
   - Computationally efficient
   - Creates binary splits only

### For Regression:

1. **Mean Squared Error (MSE)**:
```
MSE = (1/n) × ∑(y_i - ŷ)²
```

2. **Mean Absolute Error (MAE)**:
```
MAE = (1/n) × ∑|y_i - ŷ|
```

3. **Variance Reduction**:
```
VR = Var(parent) - ∑(|child|/|parent| × Var(child))
```

## 4. Tree Building Algorithm <a id="tree-building"></a>

### ID3 Algorithm (Iterative Dichotomiser 3):
```
function ID3(examples, attributes):
    if all examples have same class:
        return leaf with that class
    if attributes is empty:
        return leaf with majority class
    
    best_attribute = attribute with highest information gain
    tree = new decision node for best_attribute
    
    for each value v of best_attribute:
        examples_v = examples with best_attribute = v
        subtree = ID3(examples_v, attributes - {best_attribute})
        add branch to tree with label v and subtree
    
    return tree
```

### CART Algorithm (Classification and Regression Trees):
- Creates binary trees
- Uses Gini index for classification
- Uses MSE for regression
- Handles missing values

## 5. Implementation from Scratch <a id="implementation"></a>

```python
import numpy as np
from collections import Counter
import math

class DecisionNode:
    """Class for internal nodes"""
    def __init__(self, feature_idx, threshold, left, right, info_gain):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

class LeafNode:
    """Class for leaf nodes"""
    def __init__(self, value):
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        
    def fit(self, X, y):
        """Build decision tree"""
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        
    def _build_tree(self, X, y, depth=0):
        """Recursive tree building"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_label(y)
            return LeafNode(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # No improvement possible
        if best_gain == 0:
            leaf_value = self._most_common_label(y)
            return LeafNode(value=leaf_value)
        
        # Split data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        
        # Recursive build
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return DecisionNode(best_feature, best_threshold, left, right, best_gain)
    
    def _best_split(self, X, y):
        """Find best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(self.n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def _information_gain(self, X_column, y, threshold):
        """Calculate information gain"""
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Split
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Weighted average impurity of children
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        impurity_left = self._impurity(y[left_idxs])
        impurity_right = self._impurity(y[right_idxs])
        
        weighted_impurity = (n_left/n) * impurity_left + (n_right/n) * impurity_right
        
        # Information gain
        info_gain = parent_impurity - weighted_impurity
        return info_gain
    
    def _impurity(self, y):
        """Calculate impurity (Gini or Entropy)"""
        proportions = np.bincount(y) / len(y)
        
        if self.criterion == 'gini':
            return 1 - np.sum(proportions ** 2)
        elif self.criterion == 'entropy':
            # Avoid log(0)
            proportions = proportions[proportions > 0]
            return -np.sum(proportions * np.log2(proportions))
    
    def _split(self, X_column, threshold):
        """Split data based on threshold"""
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]
        return left_idxs, right_idxs
    
    def _most_common_label(self, y):
        """Get most common class label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self._traverse_tree(x, self.root))
        return np.array(predictions)
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction"""
        if isinstance(node, LeafNode):
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Advanced Decision Tree with additional features
class AdvancedDecisionTree(DecisionTreeClassifier):
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None,
                 criterion='gini'):
        super().__init__(max_depth, min_samples_split, criterion)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
    def _best_split(self, X, y):
        """Find best split with feature sampling"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Feature sampling
        n_features = X.shape[1]
        if self.max_features is None:
            features = range(n_features)
        else:
            n_sample_features = min(self.max_features, n_features)
            features = np.random.choice(n_features, n_sample_features, replace=False)
        
        for feature_idx in features:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                
                # Check minimum samples in leaf
                left_idxs, right_idxs = self._split(X_column, threshold)
                if len(left_idxs) < self.min_samples_leaf or \
                   len(right_idxs) < self.min_samples_leaf:
                    continue
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain

# Decision Tree Regressor
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return LeafNode(value=np.mean(y))
        
        # Find best split
        best_feature, best_threshold, best_mse = self._best_split(X, y)
        
        if best_mse is None:
            return LeafNode(value=np.mean(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursive build
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(best_feature, best_threshold, left, right, None)
    
    def _best_split(self, X, y):
        """Find split that minimizes MSE"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                mse = self._calculate_mse(X[:, feature_idx], y, threshold)
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        if best_feature is None:
            return None, None, None
            
        return best_feature, best_threshold, best_mse
    
    def _calculate_mse(self, X_column, y, threshold):
        """Calculate MSE for a split"""
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return float('inf')
        
        left_mse = np.mean((y[left_mask] - np.mean(y[left_mask])) ** 2)
        right_mse = np.mean((y[right_mask] - np.mean(y[right_mask])) ** 2)
        
        # Weighted average
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = n_left + n_right
        
        weighted_mse = (n_left/n_total) * left_mse + (n_right/n_total) * right_mse
        return weighted_mse
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._traverse_tree(x, self.root))
        return np.array(predictions)
    
    def _traverse_tree(self, x, node):
        if isinstance(node, LeafNode):
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
```

## 6. Pruning Techniques <a id="pruning"></a>

### Pre-pruning (Early Stopping):
- Set maximum depth
- Minimum samples for split
- Minimum samples in leaf
- Minimum impurity decrease

### Post-pruning:
```python
def prune_tree(node, X_val, y_val):
    """Cost complexity pruning"""
    if isinstance(node, LeafNode):
        return node
    
    # Recursively prune children
    node.left = prune_tree(node.left, X_val, y_val)
    node.right = prune_tree(node.right, X_val, y_val)
    
    # Check if both children are leaves
    if isinstance(node.left, LeafNode) and isinstance(node.right, LeafNode):
        # Calculate error with and without pruning
        error_no_prune = calculate_error(node, X_val, y_val)
        
        # Create leaf with majority class
        leaf = LeafNode(value=majority_class(y_val))
        error_prune = calculate_error(leaf, X_val, y_val)
        
        # Prune if error doesn't increase
        if error_prune <= error_no_prune:
            return leaf
    
    return node
```

## 7. Handling Different Data Types <a id="data-types"></a>

### Categorical Features:
```python
def split_categorical(X_column, categories):
    """Multi-way split for categorical features"""
    splits = {}
    for category in categories:
        mask = X_column == category
        splits[category] = np.where(mask)[0]
    return splits
```

### Missing Values:
1. **Surrogate splits**: Use correlated features
2. **Separate branch**: Create branch for missing values
3. **Imputation**: Fill before training

### Continuous Features:
- Binary splits at thresholds
- Consider all unique values as potential splits

## 8. Advantages and Disadvantages <a id="pros-cons"></a>

### Advantages:
1. **Interpretability**: Easy to understand and visualize
2. **No preprocessing**: Handles numerical and categorical data
3. **Non-linear relationships**: Captures complex patterns
4. **Feature importance**: Built-in feature selection
5. **Handles missing values**: With proper techniques
6. **Fast prediction**: O(log n) with balanced tree

### Disadvantages:
1. **Overfitting**: Prone to creating complex trees
2. **Instability**: Small data changes → different trees
3. **Biased**: Towards features with more levels
4. **Axis-aligned splits**: Can't capture diagonal boundaries
5. **Difficulty with linear relationships**: Requires many splits
6. **Single tree limitations**: Often need ensembles

## 9. Advanced Topics <a id="advanced-topics"></a>

### 1. Feature Importance:
```python
def calculate_feature_importance(tree, n_features):
    """Calculate importance based on impurity decrease"""
    importances = np.zeros(n_features)
    
    def traverse(node):
        if isinstance(node, LeafNode):
            return
        
        # Add importance for this split
        importances[node.feature_idx] += node.info_gain
        
        # Recursive traverse
        traverse(node.left)
        traverse(node.right)
    
    traverse(tree.root)
    
    # Normalize
    importances = importances / np.sum(importances)
    return importances
```

### 2. Oblique Decision Trees:
- Allow linear combinations of features
- More flexible decision boundaries
- Higher computational cost

### 3. Fuzzy Decision Trees:
- Soft boundaries instead of hard splits
- Membership functions for splits
- Better handling of uncertainty

### 4. Incremental Learning:
```python
class IncrementalDecisionTree:
    def partial_fit(self, X, y):
        """Update tree with new data"""
        # Implement Hoeffding trees or similar
        pass
```

## 10. Interview Questions & Answers <a id="interview-qa"></a>

### Q1: What is a Decision Tree and how does it work?
**Answer**: A Decision Tree is a tree-structured classifier where:
- Internal nodes represent features/attributes
- Branches represent decision rules
- Leaf nodes represent outcomes/predictions

It works by recursively splitting data based on feature values that best separate the classes (or reduce variance for regression), creating a flowchart-like structure for making predictions.

### Q2: Explain the difference between Gini Index and Entropy
**Answer**:
- **Entropy**: H(S) = -Σ p_i × log₂(p_i)
  - Measures information/uncertainty
  - Ranges from 0 (pure) to log₂(n) (equal distribution)
  - More computationally expensive (logarithm)
  
- **Gini Index**: Gini(S) = 1 - Σ p_i²
  - Measures impurity
  - Ranges from 0 (pure) to 1-1/n
  - Computationally efficient
  - Default in CART, similar results to entropy

### Q3: What is Information Gain?
**Answer**: Information Gain measures the reduction in entropy/uncertainty after splitting on an attribute:
```
IG(S, A) = Entropy(S) - Weighted_Average_Entropy(S_after_split)
```
Higher information gain means the attribute provides more information about the class. ID3 algorithm selects attributes with highest information gain.

### Q4: What causes overfitting in Decision Trees?
**Answer**: 
1. **Growing tree too deep**: Memorizes training data
2. **No pruning**: Keeps noise and outliers
3. **Small leaf size**: Creates leaves for individual samples
4. **No regularization**: Unconstrained growth
5. **High variance**: Sensitive to small data changes

**Solutions**: Pruning, max depth, min samples split/leaf, ensemble methods

### Q5: Explain pre-pruning vs post-pruning
**Answer**:
**Pre-pruning (Early Stopping)**:
- Stop growing tree before it becomes too complex
- Parameters: max_depth, min_samples_split, min_impurity_decrease
- Faster, but might stop too early
- Risk of underfitting

**Post-pruning**:
- Grow full tree, then remove nodes
- Methods: Cost complexity, reduced error pruning
- More computationally expensive
- Generally better performance
- Can find optimal tree size

### Q6: How do Decision Trees handle categorical variables?
**Answer**:
1. **Multi-way splits**: Create branch for each category (ID3, C4.5)
2. **Binary splits**: Group categories into two sets (CART)
3. **One-hot encoding**: Convert to binary features
4. **Ordinal encoding**: If natural ordering exists
5. **Target encoding**: Based on target variable statistics

CART creates optimal binary splits by trying all possible groupings.

### Q7: Compare Decision Trees with other algorithms
**Answer**:
| Aspect | Decision Trees | Random Forest | SVM | Neural Networks |
|--------|---------------|---------------|-----|-----------------|
| Interpretability | High | Medium | Low | Very Low |
| Training Speed | Fast | Medium | Slow | Very Slow |
| Prediction Speed | Very Fast | Fast | Fast | Fast |
| Handles Non-linear | Yes | Yes | Yes (kernel) | Yes |
| Feature Scaling | Not needed | Not needed | Required | Required |
| Overfitting Risk | High | Low | Low | High |

### Q8: What is the time complexity of Decision Trees?
**Answer**:
**Training**:
- Average: O(n × m × log n)
- Worst case: O(n² × m)
Where n = samples, m = features

**Prediction**:
- Average: O(log n) - balanced tree
- Worst case: O(n) - degenerate tree

**Space Complexity**: O(n) for storing the tree

### Q9: How do you choose the best attribute to split on?
**Answer**: Depends on algorithm:
1. **ID3**: Information Gain (highest)
2. **C4.5**: Gain Ratio (normalized IG)
3. **CART**: Gini Index (lowest impurity)
4. **Regression**: MSE reduction (highest)

Process: Try all features and thresholds, select one giving best metric improvement.

### Q10: What is pruning and why is it important?
**Answer**: Pruning removes parts of the tree that don't provide power to classify instances. Important because:
1. **Reduces overfitting**: Removes noise-fitting branches
2. **Improves generalization**: Better test performance
3. **Simplifies model**: Easier interpretation
4. **Reduces size**: Faster predictions
5. **Handles noise**: More robust to outliers

### Q11: Explain CART algorithm
**Answer**: CART (Classification and Regression Trees):
1. **Binary splits only**: Each node splits into exactly two branches
2. **Handles both tasks**: Classification (Gini) and Regression (MSE)
3. **Greedy algorithm**: Selects best split at each node
4. **Missing values**: Uses surrogate splits
5. **Pruning**: Cost-complexity pruning
6. **Feature selection**: Automatic during training

### Q12: How do Decision Trees calculate feature importance?
**Answer**: Feature importance is calculated as:
```
Importance(feature) = Σ (impurity_decrease × n_samples_split) / n_total_samples
```
Normalized so all importances sum to 1. Features used higher in tree and causing larger impurity decreases get higher importance.

### Q13: What are the limitations of Decision Trees?
**Answer**:
1. **Axis-aligned boundaries**: Can't capture diagonal decision boundaries efficiently
2. **Instability**: Small data changes can result in different trees
3. **Biased**: Towards features with more levels
4. **Poor extrapolation**: Can't predict beyond training data range
5. **Single tree accuracy**: Often needs ensembles for competitive performance
6. **Linear relationships**: Inefficient for linear patterns

### Q14: How do you handle imbalanced classes in Decision Trees?
**Answer**:
1. **Class weights**: Weight samples inversely to class frequency
2. **Balanced splits**: Consider class distribution in split criterion
3. **Sampling**: Over/under-sampling before training
4. **Modified metrics**: Use balanced accuracy or F1 for splitting
5. **Cost-sensitive**: Assign different misclassification costs
6. **Ensemble methods**: Combine with techniques like SMOTE

### Q15: Explain the bias-variance tradeoff in Decision Trees
**Answer**:
- **High variance**: Deep trees overfit, sensitive to training data
- **Low bias**: Can capture complex patterns
- **Shallow trees**: Higher bias, lower variance
- **Deep trees**: Lower bias, higher variance

Balance through:
- Pruning parameters
- Ensemble methods (Random Forest reduces variance)
- Cross-validation for hyperparameter tuning

### Q16: What is the difference between ID3, C4.5, and CART?
**Answer**:
| Feature | ID3 | C4.5 | CART |
|---------|-----|------|------|
| Split criterion | Information Gain | Gain Ratio | Gini/MSE |
| Handles continuous | No | Yes | Yes |
| Handles missing | No | Yes | Yes |
| Tree type | Multi-way | Multi-way | Binary |
| Pruning | No | Yes | Yes |
| Regression | No | No | Yes |

### Q17: How would you implement a Decision Tree from scratch?
**Answer**: Key components:
1. **Node classes**: DecisionNode and LeafNode
2. **Splitting criteria**: Implement Gini/Entropy/MSE
3. **Best split finder**: Try all features and thresholds
4. **Recursive builder**: Build tree top-down
5. **Prediction**: Traverse tree based on feature values
6. **Stopping criteria**: Max depth, min samples

### Q18: When should you use Decision Trees?
**Answer**:
**Use when**:
- Need interpretable model
- Have mixed data types
- Non-linear relationships
- Want feature importance
- Need fast predictions

**Avoid when**:
- Need high accuracy (use ensembles)
- Data has linear relationships
- Very high dimensional data
- Small dataset with high variance
- Need stable model

### Q19: How do you visualize a Decision Tree?
**Answer**:
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Method 1: Using plot_tree
plot_tree(clf, feature_names=features, class_names=classes, filled=True)

# Method 2: Export to Graphviz
from sklearn.tree import export_graphviz
export_graphviz(clf, out_file='tree.dot', feature_names=features)

# Method 3: Text representation
from sklearn.tree import export_text
tree_rules = export_text(clf, feature_names=features)
```

### Q20: What are some real-world applications of Decision Trees?
**Answer**:
1. **Credit scoring**: Loan approval decisions
2. **Medical diagnosis**: Disease classification based on symptoms
3. **Customer churn**: Predict customer retention
4. **Fraud detection**: Identify suspicious transactions
5. **Manufacturing**: Quality control and defect prediction
6. **Marketing**: Customer segmentation
7. **Real estate**: Property valuation
8. **HR**: Employee retention and hiring decisions