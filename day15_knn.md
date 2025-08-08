# Day 15: K-Nearest Neighbors (KNN)

## Table of Contents
1. [Introduction to KNN](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Distance Metrics](#distance-metrics)
4. [KNN Algorithm](#knn-algorithm)
5. [Implementation from Scratch](#implementation)
6. [Choosing K Value](#choosing-k)
7. [KNN Variations](#variations)
8. [Advantages and Disadvantages](#pros-cons)
9. [Advanced Topics](#advanced-topics)
10. [Interview Questions & Answers](#interview-qa)

## 1. Introduction to KNN <a id="introduction"></a>

K-Nearest Neighbors is a **non-parametric**, **lazy learning** algorithm used for both classification and regression. It makes predictions based on the K closest training examples in the feature space.

### Key Characteristics:
- **Instance-based learning**: Stores all training data
- **Non-parametric**: Makes no assumptions about data distribution
- **Lazy learning**: No explicit training phase
- **Local approximation**: Makes predictions based on local neighborhood

### Intuition:
"Tell me who your neighbors are, and I'll tell you who you are"

## 2. Mathematical Foundation <a id="mathematical-foundation"></a>

### For Classification:
Given a query point x_q, the predicted class is:

```
ŷ = argmax_c ∑(i∈N_k(x_q)) I(y_i = c)
```

Where:
- N_k(x_q) = set of k nearest neighbors
- I(·) = indicator function
- c = class label

### For Regression:
```
ŷ = (1/k) ∑(i∈N_k(x_q)) y_i
```

### Weighted KNN:
```
ŷ = ∑(i∈N_k(x_q)) w_i · y_i / ∑(i∈N_k(x_q)) w_i
```

Where w_i = 1/d(x_q, x_i) or w_i = exp(-d(x_q, x_i))

## 3. Distance Metrics <a id="distance-metrics"></a>

### Euclidean Distance:
```
d(x, y) = √(∑(i=1 to n) (x_i - y_i)²)
```

### Manhattan Distance:
```
d(x, y) = ∑(i=1 to n) |x_i - y_i|
```

### Minkowski Distance:
```
d(x, y) = (∑(i=1 to n) |x_i - y_i|^p)^(1/p)
```
- p=1: Manhattan distance
- p=2: Euclidean distance
- p→∞: Chebyshev distance

### Cosine Similarity:
```
similarity(x, y) = (x·y) / (||x|| × ||y||)
distance(x, y) = 1 - similarity(x, y)
```

### Hamming Distance (for categorical):
```
d(x, y) = ∑(i=1 to n) I(x_i ≠ y_i)
```

## 4. KNN Algorithm <a id="knn-algorithm"></a>

### Training Phase:
1. Store all training data points and their labels

### Prediction Phase:
1. Calculate distance from query point to all training points
2. Select K nearest neighbors
3. For classification: Vote among K neighbors
4. For regression: Average the K neighbors' values

### Time Complexity:
- Training: O(1) - just store data
- Prediction: O(n·d) where n = number of samples, d = dimensions
- With optimizations (KD-Tree): O(log n) average case

## 5. Implementation from Scratch <a id="implementation"></a>

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance"""
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2, p=3):
        """Calculate Minkowski distance"""
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
    def _get_distance(self, x1, x2):
        """Get distance based on metric"""
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
    
    def predict(self, X):
        """Predict labels for test data"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._get_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Get labels of k nearest neighbors
            k_labels = [label for _, label in k_nearest]
            
            # Most common label (classification)
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
            
        return np.array(predictions)

# Advanced KNN with weighted voting
class WeightedKNN(KNN):
    def __init__(self, k=5, distance_metric='euclidean', weights='distance'):
        super().__init__(k, distance_metric)
        self.weights = weights
        
    def predict(self, X):
        """Predict with weighted voting"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._get_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            if self.weights == 'uniform':
                k_labels = [label for _, label in k_nearest]
                prediction = Counter(k_labels).most_common(1)[0][0]
            elif self.weights == 'distance':
                # Weight by inverse distance
                weighted_votes = {}
                for dist, label in k_nearest:
                    weight = 1 / (dist + 1e-8)  # avoid division by zero
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                
                prediction = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(prediction)
            
        return np.array(predictions)

# KNN Regressor
class KNNRegressor(KNN):
    def predict(self, X):
        """Predict continuous values"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._get_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Average of k nearest neighbors
            k_values = [value for _, value in k_nearest]
            prediction = np.mean(k_values)
            predictions.append(prediction)
            
        return np.array(predictions)

# Optimized KNN using vectorized operations
class OptimizedKNN:
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Vectorized distance computation
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_labels = self.y_train[k_indices]
            
            # Most common label
            unique, counts = np.unique(k_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions.append(prediction)
            
        return np.array(predictions)
```

## 6. Choosing K Value <a id="choosing-k"></a>

### Methods to Select K:

1. **Square Root Rule**: k = √n (where n is number of samples)
2. **Odd Number Rule**: Choose odd k to avoid ties in binary classification
3. **Cross-Validation**: Test different k values and choose best

```python
from sklearn.model_selection import cross_val_score

def find_optimal_k(X_train, y_train, k_range=range(1, 31)):
    """Find optimal k using cross-validation"""
    k_scores = []
    
    for k in k_range:
        knn = KNN(k=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        k_scores.append((k, scores.mean()))
    
    # Find k with highest score
    optimal_k = max(k_scores, key=lambda x: x[1])[0]
    return optimal_k, k_scores
```

### Effect of K:
- **Small K**: More sensitive to noise, overfitting
- **Large K**: Smoother decision boundary, underfitting
- **K = N**: Always predicts majority class

## 7. KNN Variations <a id="variations"></a>

### 1. Ball Tree Algorithm:
```python
from sklearn.neighbors import BallTree

class BallTreeKNN:
    def __init__(self, k=5, leaf_size=30):
        self.k = k
        self.leaf_size = leaf_size
        
    def fit(self, X, y):
        self.y_train = np.array(y)
        self.tree = BallTree(X, leaf_size=self.leaf_size)
        
    def predict(self, X):
        distances, indices = self.tree.query(X, k=self.k)
        predictions = []
        
        for idx_list in indices:
            k_labels = self.y_train[idx_list]
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
            
        return np.array(predictions)
```

### 2. KD-Tree Algorithm:
```python
from sklearn.neighbors import KDTree

class KDTreeKNN:
    def __init__(self, k=5, leaf_size=30):
        self.k = k
        self.leaf_size = leaf_size
        
    def fit(self, X, y):
        self.y_train = np.array(y)
        self.tree = KDTree(X, leaf_size=self.leaf_size)
        
    def predict(self, X):
        distances, indices = self.tree.query(X, k=self.k)
        # Similar prediction logic
        pass
```

### 3. Locality Sensitive Hashing (LSH):
For very high-dimensional data

## 8. Advantages and Disadvantages <a id="pros-cons"></a>

### Advantages:
1. **Simple and Intuitive**: Easy to understand and implement
2. **No Training Period**: Can add new data easily
3. **Multi-class Natural**: Handles multiple classes naturally
4. **Non-linear Boundaries**: Can capture complex patterns
5. **Few Hyperparameters**: Only k and distance metric

### Disadvantages:
1. **Computationally Expensive**: O(n) predictions
2. **Memory Intensive**: Stores all training data
3. **Curse of Dimensionality**: Performance degrades in high dimensions
4. **Sensitive to Feature Scale**: Requires normalization
5. **Imbalanced Data Issues**: Majority class bias

## 9. Advanced Topics <a id="advanced-topics"></a>

### 1. Feature Scaling:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)
```

### 2. Dimensionality Reduction:
```python
from sklearn.decomposition import PCA

# Reduce dimensions before KNN
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
```

### 3. Handling Imbalanced Data:
```python
class BalancedKNN:
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # Calculate class weights
        unique, counts = np.unique(y, return_counts=True)
        self.class_weights = {}
        total = len(y)
        
        for cls, count in zip(unique, counts):
            self.class_weights[cls] = total / (len(unique) * count)
```

### 4. Online KNN:
```python
class OnlineKNN:
    def __init__(self, k=5, max_samples=10000):
        self.k = k
        self.max_samples = max_samples
        self.X_train = []
        self.y_train = []
        
    def partial_fit(self, X, y):
        """Add new samples incrementally"""
        self.X_train.extend(X)
        self.y_train.extend(y)
        
        # Keep only most recent samples if exceeded max
        if len(self.X_train) > self.max_samples:
            self.X_train = self.X_train[-self.max_samples:]
            self.y_train = self.y_train[-self.max_samples:]
```

## 10. Interview Questions & Answers <a id="interview-qa"></a>

### Q1: What is KNN and how does it work?
**Answer**: KNN is a non-parametric, instance-based learning algorithm that classifies new data points based on the majority class of their k nearest neighbors in the feature space. For a query point, it:
1. Calculates distances to all training points
2. Selects k closest points
3. Assigns the most frequent class among these k neighbors (or averages for regression)

### Q2: Why is KNN called a "lazy" learning algorithm?
**Answer**: KNN is called lazy because it doesn't build an explicit model during training. Instead, it simply stores all training data and defers computation until prediction time. This contrasts with "eager" learners like decision trees or neural networks that build a model during training.

### Q3: What are the key hyperparameters in KNN?
**Answer**: 
1. **k**: Number of neighbors to consider
2. **Distance metric**: Euclidean, Manhattan, Minkowski, etc.
3. **Weights**: Uniform or distance-based weighting
4. **Algorithm**: Brute force, Ball Tree, KD Tree
5. **Leaf size**: For tree-based algorithms

### Q4: How do you choose the optimal value of k?
**Answer**: Several methods:
1. **Cross-validation**: Test different k values and choose one with best validation score
2. **Square root rule**: k = √n (simple heuristic)
3. **Elbow method**: Plot error vs k and look for elbow point
4. **Odd number**: For binary classification to avoid ties
5. Generally, k should be: small enough to capture local patterns, large enough to reduce noise

### Q5: What is the curse of dimensionality in KNN?
**Answer**: As dimensions increase:
1. **Distance becomes meaningless**: All points become approximately equidistant
2. **Sparsity**: Data becomes sparse, neighbors aren't truly "near"
3. **Computational cost**: Distance calculations become expensive
4. **More features needed**: Exponentially more data required

Solutions: Feature selection, dimensionality reduction (PCA), use specialized distance metrics

### Q6: How does KNN handle categorical features?
**Answer**: Several approaches:
1. **One-hot encoding**: Convert to binary features
2. **Label encoding**: With caution, as it introduces ordering
3. **Hamming distance**: For purely categorical data
4. **Gower distance**: Handles mixed data types
5. **Custom distance functions**: Domain-specific metrics

### Q7: Compare KNN with other algorithms like Decision Trees
**Answer**:
| Aspect | KNN | Decision Trees |
|--------|-----|----------------|
| Training time | O(1) | O(n log n) |
| Prediction time | O(n) | O(log d) |
| Interpretability | Low | High |
| Feature scaling | Required | Not required |
| Non-linear boundaries | Yes | Yes |
| Outlier sensitivity | High | Medium |
| Memory usage | High | Low |

### Q8: What are the advantages and disadvantages of KNN?
**Answer**:
**Advantages**:
- Simple and intuitive
- No assumptions about data distribution
- Can capture complex, non-linear patterns
- Naturally handles multi-class problems
- Easy to implement

**Disadvantages**:
- Computationally expensive predictions
- Memory intensive (stores all data)
- Sensitive to feature scaling
- Performance degrades with high dimensions
- Sensitive to noisy data and outliers

### Q9: How can you speed up KNN?
**Answer**:
1. **Data structures**: KD-Tree (low dimensions), Ball Tree (medium dimensions)
2. **Approximate methods**: LSH (Locality Sensitive Hashing)
3. **Dimensionality reduction**: PCA, feature selection
4. **Parallelization**: Compute distances in parallel
5. **Indexing**: Pre-compute and index distances
6. **Sampling**: Use representative subset for large datasets

### Q10: Explain weighted KNN
**Answer**: In weighted KNN, closer neighbors have more influence on the prediction:
- **Uniform weights**: All k neighbors vote equally
- **Distance weights**: Weight = 1/distance (inverse distance)
- **Gaussian weights**: Weight = exp(-distance²/2σ²)

Benefits: Smoother decision boundaries, better handling of varying densities

### Q11: How does KNN work for regression?
**Answer**: For regression, KNN predicts continuous values by:
1. Finding k nearest neighbors
2. Taking average (or weighted average) of their target values
3. Can use mean, median, or weighted mean

Formula: ŷ = (1/k) × Σ(yi) for uniform weights

### Q12: What preprocessing steps are important for KNN?
**Answer**:
1. **Feature scaling**: Standardization or normalization (critical!)
2. **Handle missing values**: Imputation or removal
3. **Outlier detection**: Remove or cap extreme values
4. **Feature selection**: Reduce irrelevant features
5. **Dimensionality reduction**: PCA for high-dimensional data
6. **Encoding categoricals**: One-hot or other appropriate encoding

### Q13: When should you use KNN vs when should you avoid it?
**Answer**:
**Use KNN when**:
- Local patterns are important
- Non-linear relationships exist
- Dataset is not too large
- Low to medium dimensionality
- You need a simple baseline

**Avoid KNN when**:
- Very large datasets (slow predictions)
- High-dimensional data (curse of dimensionality)
- Need interpretable model
- Real-time predictions required
- Imbalanced datasets (without modifications)

### Q14: How do you handle imbalanced classes in KNN?
**Answer**:
1. **Weighted voting**: Weight by inverse class frequency
2. **Different k for each class**: Adaptive k based on class distribution
3. **SMOTE with KNN**: Generate synthetic samples
4. **Distance weighting**: Give more weight to minority class neighbors
5. **Ensemble methods**: Combine with sampling techniques

### Q15: Explain the difference between KD-Tree and Ball Tree
**Answer**:
**KD-Tree**:
- Partitions space using axis-aligned hyperplanes
- Efficient for low dimensions (d < 20)
- O(log n) average query time
- Construction: O(n log n)
- Fails in high dimensions

**Ball Tree**:
- Partitions using hyperspheres
- Better for medium dimensions
- More robust to high dimensions
- Higher construction cost
- More flexible partitioning

### Q16: How does feature scaling affect KNN?
**Answer**: Feature scaling is crucial because:
1. **Distance dominated by large-scale features**: Without scaling, features with larger ranges dominate distance calculations
2. **Example**: Age (0-100) vs Income (0-100000) - income would dominate
3. **Solution**: Standardization (z-score) or normalization (min-max)
4. **Equal importance**: Scaling ensures all features contribute equally

### Q17: Can KNN handle missing values? How?
**Answer**: Several approaches:
1. **Remove samples**: With missing values (data loss)
2. **Imputation**: Fill with mean/median/mode before KNN
3. **Modified distance**: Ignore missing features in distance calculation
4. **KNN imputation**: Use KNN itself to impute missing values
5. **Weighted distance**: Adjust weights based on available features

### Q18: What is the time and space complexity of KNN?
**Answer**:
**Time Complexity**:
- Training: O(1) - just store data
- Prediction (brute force): O(n×d×k) 
- With KD-Tree: O(d×log n) average, O(d×n) worst case
- With Ball Tree: O(d×log n) average

**Space Complexity**:
- Storage: O(n×d) for data
- KD-Tree/Ball Tree: O(n) additional

### Q19: How do you implement KNN from scratch?
**Answer**: Key steps:
```python
1. Store training data
2. For each test point:
   - Calculate distances to all training points
   - Sort distances and get k nearest
   - Vote (classification) or average (regression)
   - Return prediction
```
Important: Vectorize operations for efficiency

### Q20: What are some real-world applications of KNN?
**Answer**:
1. **Recommendation systems**: Find similar users/items
2. **Image recognition**: Classify images based on similar ones
3. **Credit scoring**: Classify loan applicants
4. **Medical diagnosis**: Disease prediction based on similar cases
5. **Text classification**: Document categorization
6. **Anomaly detection**: Identify outliers
7. **Collaborative filtering**: Netflix, Amazon recommendations
8. **Pattern recognition**: Handwriting, speech recognition