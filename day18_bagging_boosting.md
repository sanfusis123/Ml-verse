# Day 18: Bagging vs Boosting

## Table of Contents
1. [Introduction to Ensemble Methods](#introduction)
2. [Bagging (Bootstrap Aggregating)](#bagging)
3. [Boosting Fundamentals](#boosting)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Key Differences](#differences)
6. [Implementation from Scratch](#implementation)
7. [Variants and Algorithms](#variants)
8. [Practical Considerations](#practical)
9. [Advanced Topics](#advanced-topics)
10. [Interview Questions & Answers](#interview-qa)

## 1. Introduction to Ensemble Methods <a id="introduction"></a>

Ensemble methods combine predictions from multiple models to create a stronger predictor than any individual model.

### Two Main Paradigms:
1. **Bagging**: Train models independently, combine predictions
2. **Boosting**: Train models sequentially, each correcting previous errors

### Key Principle:
- **Wisdom of Crowds**: Aggregating diverse opinions leads to better decisions
- **Error Reduction**: Different models make different errors

## 2. Bagging (Bootstrap Aggregating) <a id="bagging"></a>

### Core Concept:
- Create diverse models by training on different data subsets
- Combine predictions through voting/averaging

### Algorithm:
```
for i = 1 to M:
    D_i = bootstrap_sample(D)
    h_i = train_model(D_i)

H(x) = aggregate(h_1(x), h_2(x), ..., h_M(x))
```

### Variance Reduction:
For M independent models with variance σ²:
```
Var(Bagging) = σ²/M
```

### Key Characteristics:
- **Parallel Training**: Models trained independently
- **Equal Weights**: All models contribute equally
- **Reduces Variance**: Effective for high-variance models
- **Bootstrap Sampling**: ~63.2% unique samples per model

### Popular Algorithms:
- Random Forest
- Extra Trees
- Bagged Decision Trees

## 3. Boosting Fundamentals <a id="boosting"></a>

### Core Concept:
- Train models sequentially
- Each model focuses on errors of previous models
- Combine with weighted voting

### Generic Boosting Algorithm:
```
Initialize weights w_i = 1/n for all samples
for m = 1 to M:
    h_m = train_model(D, weights=w)
    ε_m = weighted_error(h_m)
    α_m = compute_model_weight(ε_m)
    Update sample weights based on errors
    
H(x) = sign(Σ α_m × h_m(x))
```

### Bias Reduction:
- Converts weak learners to strong learner
- Reduces both bias and variance
- Can achieve very low training error

### Key Characteristics:
- **Sequential Training**: Each model depends on previous
- **Weighted Models**: Better models get higher weights
- **Focus on Errors**: Emphasizes misclassified samples
- **Adaptive**: Adjusts to data difficulty

### Popular Algorithms:
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

## 4. Mathematical Foundations <a id="mathematical-foundations"></a>

### Bagging Mathematics:

#### Expected Error:
```
E[(y - ŷ_bag)²] = Bias² + Var/M + σ²
```
Where M is number of models

#### Correlation Effect:
With correlation ρ between models:
```
Var(Bagging) = ρσ² + (1-ρ)σ²/M
```

### Boosting Mathematics:

#### AdaBoost Objective:
Minimize exponential loss:
```
L = Σᵢ exp(-yᵢ × f(xᵢ))
```

#### Model Weight (AdaBoost):
```
α_m = 0.5 × log((1 - ε_m) / ε_m)
```

#### Sample Weight Update:
```
w_i^(m+1) = w_i^(m) × exp(-α_m × yᵢ × h_m(xᵢ))
```

#### Gradient Boosting:
Minimize loss using gradient descent in function space:
```
f_m(x) = f_{m-1}(x) + γ_m × h_m(x)
h_m = argmin_h Σᵢ L(yᵢ, f_{m-1}(xᵢ) + h(xᵢ))
```

## 5. Key Differences <a id="differences"></a>

### Comparison Table:

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Model Independence | Independent | Dependent |
| Weight Assignment | Equal | Based on performance |
| Focus | Variance reduction | Bias reduction |
| Error Type | Reduces variance | Reduces bias & variance |
| Overfitting | More robust | Prone to overfitting |
| Base Models | Complex (low bias) | Simple (high bias) |
| Sample Weights | Not used | Adaptively updated |
| Outlier Sensitivity | Robust | Sensitive |
| Training Speed | Fast (parallel) | Slow (sequential) |

### When to Use:

**Use Bagging when**:
- High variance models (e.g., deep trees)
- Need parallel training
- Want robustness to outliers
- Have unstable base models

**Use Boosting when**:
- High bias models (e.g., shallow trees)
- Need maximum accuracy
- Clean dataset (few outliers)
- Can afford sequential training

## 6. Implementation from Scratch <a id="implementation"></a>

### Bagging Implementation:
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

class BaggingClassifier:
    def __init__(self, base_estimator=None, n_estimators=10, 
                 max_samples=1.0, bootstrap=True):
        self.base_estimator = base_estimator or DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.estimators_ = []
        self.estimator_samples_ = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(n_samples, 
                                         size=int(self.max_samples * n_samples),
                                         replace=True)
            else:
                indices = np.random.permutation(n_samples)[:int(self.max_samples * n_samples)]
            
            # Train estimator
            estimator = clone(self.base_estimator)
            estimator.fit(X[indices], y[indices])
            
            self.estimators_.append(estimator)
            self.estimator_samples_.append(indices)
            
        return self
    
    def predict(self, X):
        # Collect predictions from all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_])
        
        # Majority vote
        return np.array([np.bincount(predictions[:, i]).argmax() 
                        for i in range(X.shape[0])])
    
    def predict_proba(self, X):
        # Average probabilities
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return np.mean(probas, axis=0)

# Bagging Regressor
class BaggingRegressor(BaggingClassifier):
    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.estimators_])
        return np.mean(predictions, axis=0)
```

### AdaBoost Implementation:
```python
class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Initialize weights
        sample_weight = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            # Train weak learner
            estimator = clone(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = y_pred != y
            estimator_error = np.sum(sample_weight[incorrect]) / np.sum(sample_weight)
            
            # Skip if perfect prediction
            if estimator_error <= 0:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.)
                self.estimator_errors_.append(0.)
                break
                
            # Skip if error is too high
            if estimator_error >= 0.5:
                break
                
            # Calculate alpha (estimator weight)
            alpha = self.learning_rate * 0.5 * np.log((1 - estimator_error) / estimator_error)
            
            # Update sample weights
            sample_weight *= np.exp(-alpha * y * y_pred)
            sample_weight /= np.sum(sample_weight)
            
            # Store estimator and weight
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(estimator_error)
            
        return self
    
    def predict(self, X):
        # Weighted voting
        decision = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            decision += weight * estimator.predict(X)
            
        return np.sign(decision).astype(int)
    
    def staged_predict(self, X):
        """Predictions at each boosting iteration"""
        decision = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            decision += weight * estimator.predict(X)
            yield np.sign(decision).astype(int)

# Gradient Boosting Implementation
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []
        self.init_prediction_ = None
        
    def fit(self, X, y):
        # Initialize with mean
        self.init_prediction_ = np.mean(y)
        f = np.full(y.shape, self.init_prediction_)
        
        for _ in range(self.n_estimators):
            # Calculate negative gradient (residuals for squared loss)
            residuals = y - f
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            f += self.learning_rate * predictions
            
            self.estimators_.append(tree)
            
        return self
    
    def predict(self, X):
        f = np.full(X.shape[0], self.init_prediction_)
        
        for tree in self.estimators_:
            f += self.learning_rate * tree.predict(X)
            
        return f

# Advanced: Gradient Boosting for Classification
class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []
        self.init_prediction_ = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        # Initialize with log odds
        pos_ratio = np.mean(y)
        self.init_prediction_ = np.log(pos_ratio / (1 - pos_ratio))
        f = np.full(y.shape, self.init_prediction_)
        
        for _ in range(self.n_estimators):
            # Calculate probability
            p = self._sigmoid(f)
            
            # Calculate negative gradient
            residuals = y - p
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            f += self.learning_rate * predictions
            
            self.estimators_.append(tree)
            
        return self
    
    def predict_proba(self, X):
        f = np.full(X.shape[0], self.init_prediction_)
        
        for tree in self.estimators_:
            f += self.learning_rate * tree.predict(X)
            
        probas = self._sigmoid(f)
        return np.vstack([1 - probas, probas]).T
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
```

## 7. Variants and Algorithms <a id="variants"></a>

### Bagging Variants:
1. **Random Forest**: Bagging + random feature selection
2. **Extra Trees**: Extreme randomization
3. **Pasting**: Sampling without replacement
4. **Random Subspaces**: Sample features, not instances
5. **Random Patches**: Sample both features and instances

### Boosting Variants:
1. **AdaBoost**: Adaptive Boosting
2. **Gradient Boosting**: Optimize any differentiable loss
3. **XGBoost**: Extreme Gradient Boosting
4. **LightGBM**: Gradient boosting with histogram-based learning
5. **CatBoost**: Handles categorical features naturally
6. **LogitBoost**: Logistic regression + boosting

### Hybrid Approaches:
```python
# Stacking: Use predictions as features
class StackingClassifier:
    def __init__(self, base_estimators, meta_estimator):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        
    def fit(self, X, y):
        # Train base estimators
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        
        # Create meta features
        meta_features = np.column_stack([
            est.predict_proba(X)[:, 1] for est in self.base_estimators
        ])
        
        # Train meta estimator
        self.meta_estimator.fit(meta_features, y)
        
        return self
```

## 8. Practical Considerations <a id="practical"></a>

### Hyperparameter Tuning:

**Bagging**:
- n_estimators: More is generally better
- max_samples: Controls diversity
- max_features: For random subspaces

**Boosting**:
- n_estimators: Risk of overfitting
- learning_rate: Smaller = need more estimators
- base_estimator complexity: Usually simple

### Computational Aspects:
```python
# Parallel Bagging
from joblib import Parallel, delayed

def train_estimator(X, y, indices, base_estimator):
    estimator = clone(base_estimator)
    estimator.fit(X[indices], y[indices])
    return estimator

class ParallelBagging:
    def fit(self, X, y):
        indices_list = [bootstrap_indices(len(X)) for _ in range(self.n_estimators)]
        
        self.estimators_ = Parallel(n_jobs=-1)(
            delayed(train_estimator)(X, y, indices, self.base_estimator)
            for indices in indices_list
        )
```

### Early Stopping:
```python
def train_with_early_stopping(X_train, y_train, X_val, y_val):
    gbm = GradientBoostingClassifier(n_estimators=1000)
    
    best_score = -np.inf
    patience = 10
    no_improvement = 0
    
    for i, y_pred in enumerate(gbm.staged_predict(X_val)):
        score = accuracy_score(y_val, y_pred)
        
        if score > best_score:
            best_score = score
            best_iteration = i
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            break
            
    return best_iteration
```

## 9. Advanced Topics <a id="advanced-topics"></a>

### 1. Multi-class Boosting:
```python
class MultiClassAdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # SAMME algorithm for multi-class
        # Implementation details...
```

### 2. Online Boosting:
```python
class OnlineBooster:
    def partial_fit(self, X, y):
        # Update weak learners incrementally
        pass
```

### 3. Confidence-rated Predictions:
```python
def predict_with_confidence(ensemble, X):
    predictions = [est.predict_proba(X) for est in ensemble.estimators_]
    mean_proba = np.mean(predictions, axis=0)
    std_proba = np.std(predictions, axis=0)
    
    # High std = low confidence
    confidence = 1 - std_proba.max(axis=1)
    
    return mean_proba.argmax(axis=1), confidence
```

## 10. Interview Questions & Answers <a id="interview-qa"></a>

### Q1: What is the fundamental difference between Bagging and Boosting?
**Answer**: 
- **Bagging**: Trains models independently in parallel, reduces variance by averaging diverse models. Each model has equal weight.
- **Boosting**: Trains models sequentially, each focusing on errors of previous models. Reduces bias by combining weak learners into a strong one. Models are weighted by performance.

Key insight: Bagging = parallel variance reduction, Boosting = sequential bias reduction

### Q2: Why does Bagging reduce variance?
**Answer**: Bagging reduces variance through averaging:
- If we have M independent models with variance σ², averaging reduces variance to σ²/M
- Bootstrap sampling creates diverse models that make different errors
- Errors cancel out when averaged
- Mathematical: Var(Average) = Var(Individual)/M + covariance terms

### Q3: How does Boosting convert weak learners to strong learners?
**Answer**: Boosting combines weak learners (slightly better than random) into a strong learner by:
1. Sequential training where each model corrects previous errors
2. Upweighting misclassified samples
3. Weighted combination of all models
4. Theoretical guarantee: Error decreases exponentially with iterations

Example: Each weak learner has 60% accuracy, but combination can achieve 95%+

### Q4: What is the risk of overfitting in Bagging vs Boosting?
**Answer**:
**Bagging**:
- Generally reduces overfitting
- Individual models can overfit, but averaging helps
- More trees usually better
- Very robust to overfitting

**Boosting**:
- Can overfit with too many iterations
- Sensitive to noise and outliers
- Learning rate helps control overfitting
- Requires careful tuning and monitoring

### Q5: Explain the sample weight update in AdaBoost
**Answer**: AdaBoost updates weights to focus on misclassified samples:
```
w_i^(t+1) = w_i^(t) × exp(-α_t × y_i × h_t(x_i))
```
- Correctly classified: y_i × h_t(x_i) = 1, weight decreases
- Misclassified: y_i × h_t(x_i) = -1, weight increases by exp(2α_t)
- α_t is larger for better classifiers, so they have more influence

### Q6: What is Out-of-Bag (OOB) error and why is it useful?
**Answer**: OOB error is unique to bagging:
- Each bootstrap sample uses ~63.2% of data
- Remaining ~36.8% are "out-of-bag"
- Use OOB samples to evaluate each model
- Provides unbiased error estimate without separate validation set
- Free cross-validation during training

### Q7: Compare Random Forest (Bagging) with Gradient Boosting
**Answer**:
| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Base approach | Bagging | Boosting |
| Training | Parallel | Sequential |
| Tree depth | Deep | Shallow |
| Feature sampling | Yes | Optional |
| Speed | Faster | Slower |
| Interpretability | Feature importance | Less interpretable |
| Overfitting | Robust | Sensitive |
| Hyperparameters | Few critical | Many critical |

### Q8: What is Gradient Boosting and how does it differ from AdaBoost?
**Answer**:
**AdaBoost**:
- Specific to exponential loss
- Updates sample weights
- Works with any classifier
- Simple weight calculation

**Gradient Boosting**:
- Works with any differentiable loss
- Fits to negative gradient (pseudo-residuals)
- Typically uses regression trees
- More flexible and powerful

Both are boosting but GB is more general framework

### Q9: When would you choose Bagging over Boosting?
**Answer**:
**Choose Bagging when**:
- Need parallel training (time constraints)
- Have high-variance base models
- Dataset has outliers/noise
- Want robust, stable predictions
- Limited tuning time

**Choose Boosting when**:
- Need maximum accuracy
- Have high-bias base models
- Clean, well-prepared dataset
- Can afford sequential training
- Can carefully tune hyperparameters

### Q10: How do you implement early stopping in Gradient Boosting?
**Answer**:
```python
best_score = -inf
patience_counter = 0

for iteration in range(max_iterations):
    # Train next tree
    score = evaluate(validation_set)
    
    if score > best_score:
        best_score = score
        best_iteration = iteration
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break
```
Monitor validation performance and stop when no improvement

### Q11: What are weak learners and why are they used in Boosting?
**Answer**: Weak learners are models slightly better than random guessing (>50% accuracy for binary classification).

**Why use them**:
1. **Computational efficiency**: Simple models train fast
2. **Avoid overfitting**: Simple models less likely to overfit
3. **Theoretical guarantees**: Boosting theory assumes weak learners
4. **Flexibility**: Easier to find diverse weak learners

Common choices: Decision stumps (1-level trees), shallow trees

### Q12: Explain the bias-variance tradeoff in ensemble methods
**Answer**:
**Bagging**:
- Reduces variance, maintains bias
- Individual models: Low bias, high variance (e.g., deep trees)
- Ensemble: Low bias, low variance

**Boosting**:
- Reduces both bias and variance
- Individual models: High bias, low variance (e.g., stumps)
- Ensemble: Low bias, moderate variance

Trade-off: Boosting can achieve lower bias but risks overfitting

### Q13: How does learning rate affect Boosting algorithms?
**Answer**: Learning rate (shrinkage) controls contribution of each model:
```
F_m = F_{m-1} + learning_rate × h_m
```

Effects:
- **Smaller rate**: Need more iterations, better generalization
- **Larger rate**: Fewer iterations, risk overfitting
- **Trade-off**: learning_rate × n_estimators
- **Best practice**: Small rate (0.01-0.1) with early stopping

Analogous to step size in gradient descent

### Q14: What is Stacking and how does it relate to Bagging/Boosting?
**Answer**: Stacking is another ensemble method:
1. Train multiple diverse base models (level-0)
2. Use their predictions as features for meta-model (level-1)
3. Meta-model learns how to combine base predictions

Differences:
- **Bagging/Boosting**: Fixed combination rule
- **Stacking**: Learns optimal combination
- Can combine different algorithm types
- Often wins competitions

### Q15: How do you handle imbalanced data in ensemble methods?
**Answer**:
**Bagging**:
- Balanced Random Forest: Equal samples per class
- BalanceBagging: Combines with under/over-sampling

**Boosting**:
- RUSBoost: Random undersampling + AdaBoost
- SMOTEBoost: SMOTE + AdaBoost
- Cost-sensitive boosting: Modify loss function

General: Boosting naturally focuses on minority class errors

### Q16: What are the key hyperparameters for ensemble methods?
**Answer**:
**Bagging**:
- n_estimators: Number of models (more is better)
- max_samples: Bootstrap sample size
- max_features: Feature sampling

**Boosting**:
- n_estimators: Risk of overfitting
- learning_rate: Shrinkage parameter
- base_estimator params: max_depth, min_samples_split
- subsample: Stochastic gradient boosting

Most important: n_estimators and learning_rate for boosting

### Q17: Explain the theoretical error bounds for AdaBoost
**Answer**: AdaBoost training error decreases exponentially:
```
Training_error ≤ exp(-2 × Σ γ_t²)
```
Where γ_t = 0.5 - ε_t (edge over random)

Generalization bound depends on:
- Margin distribution
- VC dimension of weak learners
- Number of iterations

Key insight: Large margins → better generalization

### Q18: How do modern boosting libraries (XGBoost, LightGBM) improve on traditional methods?
**Answer**:
1. **Regularization**: L1/L2 penalties on leaf weights
2. **Sparsity handling**: Efficient missing value treatment
3. **Parallel processing**: Column-wise parallelization
4. **Histogram-based**: Faster split finding
5. **Leaf-wise growth**: More efficient than level-wise
6. **Built-in CV**: Integrated cross-validation
7. **GPU support**: Hardware acceleration

Result: 10-100x faster with better accuracy

### Q19: What is the difference between hard and soft voting in ensembles?
**Answer**:
**Hard Voting**:
- Each model makes class prediction
- Take majority vote
- Ignores confidence
```python
predictions = [model.predict(X) for model in models]
final_pred = mode(predictions)
```

**Soft Voting**:
- Average predicted probabilities
- Then select class with highest average
- Uses confidence information
```python
probas = [model.predict_proba(X) for model in models]
final_pred = np.mean(probas, axis=0).argmax(axis=1)
```

Soft voting usually performs better

### Q20: What are some real-world applications of Bagging and Boosting?
**Answer**:
**Bagging Applications**:
1. **Credit scoring**: Random Forest for default prediction
2. **Bioinformatics**: Gene expression analysis
3. **Remote sensing**: Land cover classification
4. **Recommendation**: Collaborative filtering

**Boosting Applications**:
1. **Web search**: Ranking (XGBoost at scale)
2. **Fraud detection**: Transaction classification
3. **Computer vision**: Object detection (cascade classifiers)
4. **Kaggle competitions**: XGBoost/LightGBM dominate
5. **Ad click prediction**: Real-time bidding systems