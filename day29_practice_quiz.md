# Day 29: Practice Quiz - Comprehensive ML Interview Questions

## Table of Contents
1. [Fundamentals Quiz](#fundamentals)
2. [Statistics & Probability Quiz](#statistics)
3. [Supervised Learning Quiz](#supervised)
4. [Unsupervised Learning Quiz](#unsupervised)
5. [Model Evaluation Quiz](#evaluation)
6. [Feature Engineering Quiz](#features)
7. [Practical Scenarios Quiz](#practical)
8. [Coding Challenges](#coding)
9. [Advanced Topics Quiz](#advanced)
10. [Solutions & Explanations](#solutions)

## Introduction
This comprehensive quiz covers all topics from Days 1-28. Each section contains questions of varying difficulty levels to test your ML interview readiness.

## 1. Fundamentals Quiz <a id="fundamentals"></a>

### Q1: What is the difference between parametric and non-parametric models?
a) Parametric models are faster to train
b) Non-parametric models have fixed number of parameters
c) Parametric models make assumptions about data distribution
d) Non-parametric models cannot handle large datasets

### Q2: Which of the following is NOT a supervised learning algorithm?
a) Support Vector Machines
b) K-Means Clustering
c) Random Forest
d) Logistic Regression

### Q3: What is the curse of dimensionality?
a) Models become more accurate with more features
b) Computational cost increases linearly with dimensions
c) Data becomes sparse in high-dimensional space
d) Feature selection becomes easier

### Q4: Match the algorithm with its primary optimization technique:
- Linear Regression → ?
- SVM → ?
- Neural Networks → ?
- K-Means → ?

Options: Gradient Descent, Quadratic Programming, EM Algorithm, Normal Equation

### Q5: What is the time complexity of training a decision tree?
a) O(n log n)
b) O(n²)
c) O(n × m × log n)
d) O(2^n)

## 2. Statistics & Probability Quiz <a id="statistics"></a>

### Q6: Calculate the entropy of a dataset with 60 positive and 40 negative examples.

### Q7: What is the relationship between standard error and sample size?
a) SE = σ/√n
b) SE = σ × n
c) SE = σ²/n
d) SE = √(σ/n)

### Q8: In hypothesis testing, Type I error is:
a) Failing to reject a false null hypothesis
b) Rejecting a true null hypothesis
c) Accepting an alternative hypothesis
d) Having insufficient sample size

### Q9: If events A and B are independent, which is true?
a) P(A|B) = P(A) + P(B)
b) P(A ∩ B) = P(A) × P(B)
c) P(A ∪ B) = P(A) × P(B)
d) P(A|B) = P(B|A)

### Q10: Calculate the Gini impurity for a node with 30 samples of class A, 20 of class B, and 10 of class C.

## 3. Supervised Learning Quiz <a id="supervised"></a>

### Q11: In logistic regression, what does the sigmoid function output represent?
a) The class label
b) The distance from decision boundary
c) The probability of belonging to positive class
d) The log-odds ratio

### Q12: Which regularization technique sets some coefficients exactly to zero?
a) L2 (Ridge)
b) L1 (Lasso)
c) Elastic Net with α=0
d) Early stopping

### Q13: In Random Forest, if we have 100 features, how many features are typically considered at each split for classification?
a) 100
b) 50
c) 10
d) 33

### Q14: What is the key difference between AdaBoost and Gradient Boosting?
a) AdaBoost uses decision trees only
b) Gradient Boosting is faster
c) AdaBoost updates sample weights, GB fits to residuals
d) Gradient Boosting cannot do classification

### Q15: Code Challenge: Implement the predict function for a trained decision tree
```python
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def predict(node, sample):
    # Your implementation here
    pass
```

## 4. Unsupervised Learning Quiz <a id="unsupervised"></a>

### Q16: In K-Means clustering, what is the objective function being minimized?
a) Between-cluster variance
b) Within-cluster sum of squares
c) Silhouette coefficient
d) Davies-Bouldin index

### Q17: What is the main advantage of DBSCAN over K-Means?
a) Faster computation
b) Always finds global optimum
c) Can find arbitrary-shaped clusters
d) Requires fewer parameters

### Q18: In PCA, the first principal component:
a) Has the minimum variance
b) Is orthogonal to all other components
c) Captures the maximum variance in data
d) Is always along the x-axis

### Q19: Calculate the new coordinates of point (4, 3) after PCA transformation with eigenvectors [0.6, 0.8] and [-0.8, 0.6].

### Q20: Which clustering algorithm would be best for finding clusters of varying densities?
a) K-Means
b) Hierarchical Clustering
c) DBSCAN
d) Gaussian Mixture Models

## 5. Model Evaluation Quiz <a id="evaluation"></a>

### Q21: Calculate precision, recall, and F1-score:
- True Positives: 85
- False Positives: 15
- False Negatives: 20
- True Negatives: 80

### Q22: What does AUC-ROC of 0.5 indicate?
a) Perfect classifier
b) Random classifier
c) Worst possible classifier
d) Need more data

### Q23: Which metric is most appropriate for imbalanced classification?
a) Accuracy
b) F1-score
c) Mean Squared Error
d) R-squared

### Q24: In k-fold cross-validation with k=5 and 1000 samples, how many samples are in each validation fold?
a) 200
b) 250
c) 100
d) 800

### Q25: What is the relationship between bias, variance, and model complexity?
Draw or describe the typical curves.

## 6. Feature Engineering Quiz <a id="features"></a>

### Q26: Which feature scaling method is robust to outliers?
a) StandardScaler (z-score normalization)
b) MinMaxScaler
c) RobustScaler
d) Normalizer

### Q27: One-hot encoding of a categorical variable with 50 unique values will create:
a) 1 new feature
b) 49 new features
c) 50 new features
d) 51 new features

### Q28: What is the main disadvantage of using polynomial features?
a) Cannot capture non-linear relationships
b) Exponential increase in feature space
c) Always leads to underfitting
d) Reduces model interpretability only

### Q29: Code Challenge: Implement a function to detect multicollinearity
```python
def detect_multicollinearity(X, threshold=0.9):
    # Return pairs of highly correlated features
    # Your implementation here
    pass
```

### Q30: When should you use target encoding for categorical variables?
a) Always, it's the best method
b) When cardinality is high and data is sufficient
c) Only for ordinal variables
d) Never, it causes overfitting

## 7. Practical Scenarios Quiz <a id="practical"></a>

### Q31: Scenario: Your model has 95% training accuracy but 60% test accuracy. What would you try first?
a) Collect more features
b) Use a more complex model
c) Add regularization
d) Increase learning rate

### Q32: You have 1 million samples and 10 features. Which algorithm would likely train fastest?
a) SVM with RBF kernel
b) Random Forest with 100 trees
c) Logistic Regression
d) Deep Neural Network

### Q33: Scenario: Building a fraud detection system with 99.9% legitimate transactions. What's your approach?
List at least 3 techniques you would use.

### Q34: Your feature importance shows one feature dominates all others. What could this indicate?
a) The feature is highly predictive
b) Possible data leakage
c) Need to remove other features
d) Model is overfitting

### Q35: Design a simple ML pipeline for predicting customer churn. Include all major steps.

## 8. Coding Challenges <a id="coding"></a>

### Q36: Implement gradient descent for linear regression
```python
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    # Return trained weights
    # Your implementation here
    pass
```

### Q37: Implement the Gini impurity calculation
```python
def gini_impurity(y):
    # Calculate Gini impurity for array y
    # Your implementation here
    pass
```

### Q38: Implement k-fold cross-validation from scratch
```python
def k_fold_cv(X, y, k=5):
    # Yield train and validation indices for each fold
    # Your implementation here
    pass
```

### Q39: Implement softmax function
```python
def softmax(x):
    # Apply softmax to vector or matrix x
    # Your implementation here
    pass
```

### Q40: Implement cosine similarity
```python
def cosine_similarity(a, b):
    # Calculate cosine similarity between vectors a and b
    # Your implementation here
    pass
```

## 9. Advanced Topics Quiz <a id="advanced"></a>

### Q41: In XGBoost, what is the purpose of the gamma parameter?
a) Learning rate control
b) Minimum loss reduction for split
c) L2 regularization
d) Maximum tree depth

### Q42: What is the key innovation in attention mechanisms?
a) Parallel processing
b) Dynamic weighting of inputs
c) Reduced parameters
d) Faster training

### Q43: In ensemble methods, why does bagging reduce variance?
Provide mathematical intuition.

### Q44: Explain the vanishing gradient problem and one solution.

### Q45: What is the difference between batch normalization and layer normalization?

### Q46: In reinforcement learning, what is the exploration-exploitation tradeoff?

### Q47: How does dropout help prevent overfitting in neural networks?

### Q48: What is transfer learning and when is it useful?

### Q49: Explain the concept of few-shot learning.

### Q50: What are Graph Neural Networks used for? Give 2 examples.

## 10. Solutions & Explanations <a id="solutions"></a>

### Solutions for Fundamentals (Q1-Q5)

**Q1: Answer: c**
Explanation: Parametric models assume a specific functional form (like linear) and have fixed number of parameters. Non-parametric models (like KNN, Decision Trees) don't make such assumptions and can have varying complexity.

**Q2: Answer: b**
Explanation: K-Means is unsupervised - it finds patterns without labeled data. The others require labeled training data.

**Q3: Answer: c**
Explanation: In high dimensions, data points become sparse and distances become less meaningful. This affects algorithms relying on distance metrics.

**Q4: Answers:**
- Linear Regression → Normal Equation (or Gradient Descent)
- SVM → Quadratic Programming
- Neural Networks → Gradient Descent
- K-Means → EM Algorithm

**Q5: Answer: c**
Explanation: O(n × m × log n) where n = samples, m = features. The log n comes from sorting at each node.

### Solutions for Statistics (Q6-Q10)

**Q6: Solution:**
```
p_positive = 60/100 = 0.6
p_negative = 40/100 = 0.4
Entropy = -0.6×log₂(0.6) - 0.4×log₂(0.4)
        = -0.6×(-0.737) - 0.4×(-1.322)
        = 0.442 + 0.529 = 0.971
```

**Q7: Answer: a**
Explanation: Standard error decreases with square root of sample size.

**Q8: Answer: b**
Explanation: Type I error (α) is rejecting H₀ when it's actually true (false positive).

**Q9: Answer: b**
Explanation: Independence means P(A ∩ B) = P(A) × P(B).

**Q10: Solution:**
```
Total = 60
p_A = 30/60 = 0.5
p_B = 20/60 = 0.333
p_C = 10/60 = 0.167
Gini = 1 - (0.5² + 0.333² + 0.167²)
     = 1 - (0.25 + 0.111 + 0.028)
     = 1 - 0.389 = 0.611
```

### Solutions for Supervised Learning (Q11-Q15)

**Q11: Answer: c**
Explanation: Sigmoid transforms log-odds to probability between 0 and 1.

**Q12: Answer: b**
Explanation: L1 (Lasso) creates sparse solutions by setting some coefficients to exactly zero.

**Q13: Answer: c**
Explanation: For classification, typically √p features are considered, so √100 = 10.

**Q14: Answer: c**
Explanation: AdaBoost reweights samples, Gradient Boosting fits to pseudo-residuals (negative gradients).

**Q15: Solution:**
```python
def predict(node, sample):
    # Leaf node - return value
    if node.value is not None:
        return node.value
    
    # Internal node - traverse based on feature threshold
    if sample[node.feature] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)
```

### Solutions for Unsupervised Learning (Q16-Q20)

**Q16: Answer: b**
Explanation: K-Means minimizes within-cluster sum of squares (WCSS).

**Q17: Answer: c**
Explanation: DBSCAN can find clusters of arbitrary shape, while K-Means assumes spherical clusters.

**Q18: Answer: c**
Explanation: First PC is direction of maximum variance.

**Q19: Solution:**
```
Original point: [4, 3]
PC1 = [0.6, 0.8], PC2 = [-0.8, 0.6]
New coordinates:
x' = 4×0.6 + 3×0.8 = 2.4 + 2.4 = 4.8
y' = 4×(-0.8) + 3×0.6 = -3.2 + 1.8 = -1.4
Transformed point: (4.8, -1.4)
```

**Q20: Answer: d**
Explanation: GMM can model clusters with different densities using different covariance matrices.

### Solutions for Model Evaluation (Q21-Q25)

**Q21: Solution:**
```
Precision = TP/(TP+FP) = 85/(85+15) = 0.85
Recall = TP/(TP+FN) = 85/(85+20) = 0.809
F1 = 2×(P×R)/(P+R) = 2×(0.85×0.809)/(0.85+0.809) = 0.829
```

**Q22: Answer: b**
Explanation: AUC of 0.5 indicates random performance - no better than chance.

**Q23: Answer: b**
Explanation: F1-score balances precision and recall, better for imbalanced data than accuracy.

**Q24: Answer: a**
Explanation: Each fold contains 1000/5 = 200 samples.

**Q25: Solution:**
- Low complexity: High bias, low variance
- High complexity: Low bias, high variance
- Optimal: Balance between bias and variance
- Total error = Bias² + Variance + Irreducible error

### Solutions for Feature Engineering (Q26-Q30)

**Q26: Answer: c**
Explanation: RobustScaler uses median and IQR, robust to outliers.

**Q27: Answer: c**
Explanation: One-hot encoding creates one binary feature per unique value.

**Q28: Answer: b**
Explanation: Polynomial features grow exponentially (n choose k for degree k).

**Q29: Solution:**
```python
def detect_multicollinearity(X, threshold=0.9):
    import numpy as np
    corr_matrix = np.corrcoef(X.T)
    n_features = X.shape[1]
    
    highly_correlated = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                highly_correlated.append((i, j, corr_matrix[i, j]))
    
    return highly_correlated
```

**Q30: Answer: b**
Explanation: Target encoding works well with high cardinality and sufficient data per category to avoid overfitting.

### Solutions for Practical Scenarios (Q31-Q35)

**Q31: Answer: c**
Explanation: Large gap between train/test accuracy indicates overfitting - add regularization.

**Q32: Answer: c**
Explanation: Logistic Regression is linear in samples and features, fastest for this size.

**Q33: Solution:**
1. Use appropriate metrics (Precision, Recall, F1, not accuracy)
2. Apply SMOTE or undersampling for balance
3. Use anomaly detection algorithms
4. Cost-sensitive learning with higher penalty for missing fraud
5. Ensemble methods for robustness

**Q34: Answer: b**
Explanation: One dominant feature often indicates data leakage - the feature may contain information about the target.

**Q35: Solution:**
```
1. Data Collection & Cleaning
   - Handle missing values
   - Remove duplicates
   
2. Feature Engineering
   - Create recency, frequency, monetary features
   - Aggregate transaction history
   - Engineer interaction features
   
3. Preprocessing
   - Scale numerical features
   - Encode categorical variables
   - Handle class imbalance
   
4. Model Selection
   - Try Logistic Regression (baseline)
   - Random Forest/XGBoost
   - Evaluate with cross-validation
   
5. Model Evaluation
   - Use appropriate metrics (F1, AUC)
   - Check feature importance
   
6. Deployment
   - Create prediction pipeline
   - Monitor performance
   - Retrain periodically
```

### Solutions for Coding Challenges (Q36-Q40)

**Q36: Gradient Descent Solution:**
```python
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(n_iterations):
        # Forward pass
        y_pred = X.dot(weights) + bias
        
        # Compute gradients
        dw = (1/n_samples) * X.T.dot(y_pred - y)
        db = (1/n_samples) * np.sum(y_pred - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias
```

**Q37: Gini Impurity Solution:**
```python
def gini_impurity(y):
    if len(y) == 0:
        return 0
    
    counts = np.bincount(y)
    probabilities = counts / len(y)
    
    gini = 1 - np.sum(probabilities ** 2)
    return gini
```

**Q38: K-Fold CV Solution:**
```python
def k_fold_cv(X, y, k=5):
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        yield train_indices, val_indices
```

**Q39: Softmax Solution:**
```python
def softmax(x):
    # Handle numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

**Q40: Cosine Similarity Solution:**
```python
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)
```

### Solutions for Advanced Topics (Q41-Q50)

**Q41: Answer: b**
Explanation: Gamma is minimum loss reduction required to make a split.

**Q42: Answer: b**
Explanation: Attention dynamically weights different parts of input based on relevance.

**Q43: Solution:**
If individual models have variance σ², averaging M independent models reduces variance to σ²/M. Even with correlation ρ, variance reduces to ρσ² + (1-ρ)σ²/M.

**Q44: Solution:**
Vanishing gradient: Gradients become exponentially small in deep networks, preventing learning in early layers.
Solutions: ReLU activation, batch normalization, residual connections, gradient clipping.

**Q45: Solution:**
- Batch Norm: Normalizes across batch dimension for each feature
- Layer Norm: Normalizes across feature dimension for each sample
- BN depends on batch, LN doesn't - better for RNNs, small batches

**Q46: Solution:**
Exploration: Try new actions to discover rewards
Exploitation: Use known best actions
Tradeoff: Too much exploration wastes time, too much exploitation misses better options
Methods: ε-greedy, UCB, Thompson sampling

**Q47: Solution:**
Dropout randomly deactivates neurons during training:
- Prevents co-adaptation of neurons
- Acts like ensemble of sub-networks
- Forces redundant representations
- Reduces overfitting

**Q48: Solution:**
Transfer learning: Use pre-trained model features for new task
Useful when:
- Limited training data
- Similar domains
- Expensive to train from scratch
Example: Use ImageNet CNN for medical imaging

**Q49: Solution:**
Few-shot learning: Learning from very few examples (1-5 per class)
Approaches:
- Meta-learning (learning to learn)
- Metric learning (similarity-based)
- Data augmentation
Applications: Rare disease diagnosis, new product recognition

**Q50: Solution:**
GNNs process graph-structured data.
Examples:
1. Social network analysis (friend recommendations)
2. Molecular property prediction (drug discovery)
3. Traffic prediction (road networks)
4. Knowledge graphs (question answering)