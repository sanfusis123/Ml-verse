# Day 13: Naive Bayes

## 📚 Table of Contents
1. [Introduction and Theory](#introduction)
2. [Bayes' Theorem Foundation](#bayes-theorem)
3. [Types of Naive Bayes](#types)
4. [Mathematical Derivation](#mathematical-derivation)
5. [Implementation from Scratch](#implementation)
6. [Advanced Topics](#advanced-topics)
7. [Comprehensive Interview Q&A](#interview-qa)

---

## 1. Introduction and Theory {#introduction}

### What is Naive Bayes?

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with the "naive" assumption of conditional independence between features.

### Key Characteristics

1. **Generative Model**: Models P(X|Y) and P(Y), not P(Y|X) directly
2. **Naive Assumption**: Features are conditionally independent given class
3. **Probabilistic**: Outputs true probabilities (often poorly calibrated)
4. **Fast**: Training is just counting, prediction is multiplication
5. **Works with small data**: Needs less training data than discriminative models

### Why "Naive"?

The conditional independence assumption:
```
P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

This is rarely true in practice, but the classifier often works well anyway!

### When to Use Naive Bayes

**Excellent for**:
- Text classification (spam filtering, sentiment analysis)
- Real-time prediction (very fast)
- Multi-class problems
- Baseline models

**Not suitable for**:
- Highly correlated features
- Regression problems
- When calibrated probabilities needed

---

## 2. Bayes' Theorem Foundation {#bayes-theorem}

### Bayes' Theorem

For classification:
```
P(y|X) = P(X|y) × P(y) / P(X)
```

Where:
- P(y|X): Posterior probability of class given features
- P(X|y): Likelihood of features given class
- P(y): Prior probability of class
- P(X): Evidence (normalizing constant)

### Classification Rule

Since P(X) is constant for all classes:
```
ŷ = argmax_y P(y|X) = argmax_y P(X|y) × P(y)
```

### The Naive Assumption

With conditional independence:
```
P(X|y) = P(x₁, x₂, ..., xₙ|y) = ∏ᵢ P(xᵢ|y)
```

Therefore:
```
ŷ = argmax_y P(y) × ∏ᵢ P(xᵢ|y)
```

### Log-Space Computation

To avoid numerical underflow:
```
log P(y|X) ∝ log P(y) + Σᵢ log P(xᵢ|y)
```

---

## 3. Types of Naive Bayes {#types}

### Gaussian Naive Bayes

For continuous features, assumes Gaussian distribution:
```
P(xᵢ|y) = (1/√(2πσ²ᵧᵢ)) × exp(-(xᵢ - μᵧᵢ)²/(2σ²ᵧᵢ))
```

**Parameters**: μᵧᵢ and σ²ᵧᵢ for each feature i and class y

### Multinomial Naive Bayes

For discrete counts (e.g., word frequencies):
```
P(xᵢ|y) = (Nᵧᵢ + α) / (Nᵧ + α × |V|)
```

Where:
- Nᵧᵢ: Count of feature i in class y
- Nᵧ: Total count in class y
- α: Smoothing parameter (Laplace smoothing)
- |V|: Vocabulary size

### Bernoulli Naive Bayes

For binary features (present/absent):
```
P(xᵢ|y) = P(xᵢ=1|y)^xᵢ × (1-P(xᵢ=1|y))^(1-xᵢ)
```

### Complement Naive Bayes

For imbalanced datasets, uses complement class statistics:
```
ŷ = argmin_y Σᵢ tᵢ × log P(xᵢ|¬y)
```

---

## 4. Mathematical Derivation {#mathematical-derivation}

### Maximum Likelihood Estimation

For Gaussian Naive Bayes:

**Prior estimation**:
```
P(y = k) = nₖ / n
```

**Mean estimation**:
```
μₖⱼ = (1/nₖ) Σᵢ:yᵢ=k xᵢⱼ
```

**Variance estimation**:
```
σ²ₖⱼ = (1/nₖ) Σᵢ:yᵢ=k (xᵢⱼ - μₖⱼ)²
```

### Laplace Smoothing

To handle zero probabilities:
```
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × |features|)
```

Common choices: α = 1 (Laplace), α < 1 (Lidstone)

### Decision Boundary

For binary classification with Gaussian NB:
```
log(P(y=1|x)/P(y=0|x)) = 0
```

This creates a quadratic decision boundary (unlike linear for logistic regression).

---

## 5. Implementation from Scratch {#implementation}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, fetch_20newsgroups, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# Implementation of different Naive Bayes variants
class GaussianNaiveBayes:
    """Gaussian Naive Bayes implementation from scratch"""
    
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.variances = {}
        
    def fit(self, X, y):
        """Fit Gaussian NB by calculating statistics"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            # Filter samples for this class
            X_c = X[y == c]
            
            # Calculate prior
            self.priors[c] = len(X_c) / n_samples
            
            # Calculate mean and variance for each feature
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-6  # Add small value for stability
            
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        """Calculate Gaussian probability density"""
        # Use log space for numerical stability
        const = -0.5 * np.log(2 * np.pi * var)
        exp_term = -0.5 * ((x - mean) ** 2) / var
        return const + exp_term
    
    def predict_log_proba(self, X):
        """Predict log probabilities for each class"""
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            # Log prior
            log_prior = np.log(self.priors[c])
            
            # Log likelihood for each feature
            log_likelihood = np.sum(
                self._gaussian_pdf(X, self.means[c], self.variances[c]), 
                axis=1
            )
            
            log_probs[:, idx] = log_prior + log_likelihood
            
        return log_probs
    
    def predict_proba(self, X):
        """Predict probabilities (normalized)"""
        log_probs = self.predict_log_proba(X)
        # Subtract max for numerical stability
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        # Normalize
        return probs / np.sum(probs, axis=1, keepdims=True)
    
    def predict(self, X):
        """Predict classes"""
        log_probs = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probs, axis=1)]
    
    def score(self, X, y):
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes implementation from scratch"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes = None
        self.class_counts = {}
        self.feature_counts = {}
        self.class_log_priors = {}
        
    def fit(self, X, y):
        """Fit Multinomial NB by counting"""
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        for c in self.classes:
            # Get samples for this class
            X_c = X[y == c]
            
            # Count features
            self.feature_counts[c] = np.sum(X_c, axis=0) + self.alpha
            self.class_counts[c] = np.sum(self.feature_counts[c])
            
            # Log prior
            self.class_log_priors[c] = np.log(len(X_c) / len(y))
            
        return self
    
    def predict_log_proba(self, X):
        """Predict log probabilities"""
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            # Log probabilities of features
            feature_log_probs = np.log(self.feature_counts[c] / self.class_counts[c])
            
            # Log likelihood: sum of feature_count * log_prob
            log_likelihood = X @ feature_log_probs
            
            log_probs[:, idx] = self.class_log_priors[c] + log_likelihood
            
        return log_probs
    
    def predict_proba(self, X):
        """Predict probabilities (normalized)"""
        log_probs = self.predict_log_proba(X)
        # Subtract max for numerical stability
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        # Normalize
        return probs / np.sum(probs, axis=1, keepdims=True)
    
    def predict(self, X):
        """Predict classes"""
        log_probs = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probs, axis=1)]
    
    def score(self, X, y):
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)


# Demonstration 1: Gaussian Naive Bayes
print("=== Demonstration 1: Gaussian Naive Bayes ===")

# Generate 2D data for visualization
X_gauss, y_gauss = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42, class_sep=2
)

# Split data
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_gauss, y_gauss, test_size=0.3, random_state=42
)

# Train custom implementation
gnb_custom = GaussianNaiveBayes()
gnb_custom.fit(X_train_g, y_train_g)

# Train sklearn implementation for comparison
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train_g, y_train_g)

# Compare results
print(f"Custom implementation accuracy: {gnb_custom.score(X_test_g, y_test_g):.3f}")
print(f"Sklearn implementation accuracy: {gnb_sklearn.score(X_test_g, y_test_g):.3f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Data distribution and decision boundary
ax = axes[0, 0]
h = 0.02
x_min, x_max = X_gauss[:, 0].min() - 1, X_gauss[:, 0].max() + 1
y_min, y_max = X_gauss[:, 1].min() - 1, X_gauss[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict probabilities for mesh
Z = gnb_custom.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot decision boundary
contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), 
                      cmap='RdBu_r', alpha=0.6)
ax.scatter(X_train_g[y_train_g == 0, 0], X_train_g[y_train_g == 0, 1], 
           c='blue', label='Class 0', edgecolor='black', s=50)
ax.scatter(X_train_g[y_train_g == 1, 0], X_train_g[y_train_g == 1, 1], 
           c='red', label='Class 1', edgecolor='black', s=50)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Gaussian NB Decision Boundary')
ax.legend()
plt.colorbar(contour, ax=ax)

# 2. Feature distributions by class
ax = axes[0, 1]
for i, c in enumerate(gnb_custom.classes):
    mask = y_train_g == c
    ax.hist(X_train_g[mask, 0], bins=20, alpha=0.5, 
            label=f'Class {c} - Feature 1', density=True)
    
    # Plot fitted Gaussian
    x_range = np.linspace(X_train_g[:, 0].min(), X_train_g[:, 0].max(), 100)
    pdf = norm.pdf(x_range, gnb_custom.means[c][0], 
                   np.sqrt(gnb_custom.variances[c][0]))
    ax.plot(x_range, pdf, linewidth=2)

ax.set_xlabel('Feature 1 Value')
ax.set_ylabel('Density')
ax.set_title('Feature 1 Distribution by Class')
ax.legend()

# 3. 2D Gaussian contours
ax = axes[0, 2]
for i, c in enumerate(gnb_custom.classes):
    mask = y_train_g == c
    ax.scatter(X_train_g[mask, 0], X_train_g[mask, 1], alpha=0.5, 
              label=f'Class {c}')
    
    # Plot Gaussian contours
    mean = gnb_custom.means[c]
    cov = np.diag(gnb_custom.variances[c])
    
    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform to ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    ellipse = mean[:, np.newaxis] + 2 * eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle
    
    ax.plot(ellipse[0], ellipse[1], linewidth=2)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Gaussian Distributions (2σ contours)')
ax.legend()
ax.axis('equal')

# 4. Predicted probabilities histogram
ax = axes[1, 0]
probs = gnb_custom.predict_proba(X_test_g)[:, 1]
ax.hist(probs[y_test_g == 0], bins=20, alpha=0.5, label='True Class 0', 
        color='blue', density=True)
ax.hist(probs[y_test_g == 1], bins=20, alpha=0.5, label='True Class 1', 
        color='red', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold')
ax.set_xlabel('Predicted P(Class 1)')
ax.set_ylabel('Density')
ax.set_title('Prediction Probabilities Distribution')
ax.legend()

# 5. Confusion Matrix
ax = axes[1, 1]
y_pred = gnb_custom.predict(X_test_g)
cm = confusion_matrix(y_test_g, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# 6. Comparison with other classifiers
ax = axes[1, 2]
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    'Gaussian NB': GaussianNB(),
    'Logistic Reg': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(max_depth=3)
}

scores = {}
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_gauss, y_gauss, cv=5)
    scores[name] = cv_scores

positions = np.arange(len(scores))
ax.boxplot(scores.values(), labels=scores.keys())
ax.set_ylabel('Cross-validation Accuracy')
ax.set_title('Classifier Comparison')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration 2: Text Classification with Multinomial NB
print("\n=== Demonstration 2: Text Classification ===")

# Load sample text data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                               remove=('headers', 'footers', 'quotes'))

# Prepare data
X_text = newsgroups.data[:1000]  # Limit for speed
y_text = newsgroups.target[:1000]

# Split data
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y_text, test_size=0.3, random_state=42
)

# Vectorize text
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# Train Multinomial NB
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_vec, y_train_text)

# Predictions
y_pred_text = mnb.predict(X_test_vec)
accuracy = accuracy_score(y_test_text, y_pred_text)
print(f"\nText classification accuracy: {accuracy:.3f}")

# Analyze feature importance
feature_names = vectorizer.get_feature_names_out()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, category_idx in enumerate(range(len(categories))):
    ax = axes[idx]
    
    # Get log probabilities for this class
    log_prob = mnb.feature_log_prob_[category_idx]
    
    # Get top 20 features
    top_indices = np.argsort(log_prob)[-20:]
    top_features = [feature_names[i] for i in top_indices]
    top_probs = np.exp(log_prob[top_indices])
    
    # Plot
    ax.barh(range(len(top_features)), top_probs)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Probability')
    ax.set_title(f'Top Features for: {categories[category_idx]}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration 3: Comparing NB Variants
print("\n=== Demonstration 3: Comparing Naive Bayes Variants ===")

# Generate data with different characteristics
n_samples = 1000
n_features = 20

# Continuous features (Gaussian)
X_continuous = np.random.randn(n_samples, n_features)
# Count features (Multinomial) 
X_counts = np.random.poisson(2, (n_samples, n_features))
# Binary features (Bernoulli)
X_binary = np.random.binomial(1, 0.3, (n_samples, n_features))

# Create target based on different feature relationships
y_continuous = (X_continuous[:, 0] + X_continuous[:, 1] > 0).astype(int)
y_counts = (X_counts[:, 0] + X_counts[:, 1] > 4).astype(int)
y_binary = (X_binary[:, 0] + X_binary[:, 1] > 0).astype(int)

# Test different NB variants on each data type
datasets = {
    'Continuous': (X_continuous, y_continuous),
    'Counts': (X_counts, y_counts),
    'Binary': (X_binary, y_binary)
}

nb_variants = {
    'Gaussian': GaussianNB(),
    'Multinomial': MultinomialNB(),
    'Bernoulli': BernoulliNB()
}

results = pd.DataFrame(index=datasets.keys(), columns=nb_variants.keys())

for data_name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    for nb_name, nb_model in nb_variants.items():
        try:
            nb_model.fit(X_train, y_train)
            score = nb_model.score(X_test, y_test)
            results.loc[data_name, nb_name] = score
        except:
            results.loc[data_name, nb_name] = np.nan

print("\nNaive Bayes Variants Performance:")
print(results)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
results_numeric = results.astype(float)
sns.heatmap(results_numeric, annot=True, fmt='.3f', cmap='YlOrRd', 
            cbar_kws={'label': 'Accuracy'}, ax=ax)
ax.set_title('NB Variant Performance on Different Data Types')
plt.tight_layout()
plt.show()

# Demonstration 4: Handling Imbalanced Data
print("\n=== Demonstration 4: Imbalanced Data ===")

# Create imbalanced dataset
X_imb, y_imb = make_classification(n_samples=1000, n_features=20, 
                                   n_informative=15, n_redundant=5,
                                   n_classes=2, weights=[0.9, 0.1],
                                   random_state=42)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# Standard Gaussian NB
gnb_standard = GaussianNB()
gnb_standard.fit(X_train_imb, y_train_imb)

# Complement NB
from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()
X_train_imb_pos = X_train_imb + abs(X_train_imb.min()) + 1  # Make positive
X_test_imb_pos = X_test_imb + abs(X_test_imb.min()) + 1
cnb.fit(X_train_imb_pos, y_train_imb)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score

y_pred_gnb = gnb_standard.predict(X_test_imb)
y_pred_cnb = cnb.predict(X_test_imb_pos)

print("\nGaussian NB on Imbalanced Data:")
print(classification_report(y_test_imb, y_pred_gnb, 
                          target_names=['Majority', 'Minority']))

print("\nComplement NB on Imbalanced Data:")
print(classification_report(y_test_imb, y_pred_cnb,
                          target_names=['Majority', 'Minority']))

# Demonstration 5: Calibration Analysis
print("\n=== Demonstration 5: Probability Calibration ===")

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

# Generate dataset
X_cal, y_cal = make_classification(n_samples=2000, n_features=20, 
                                  n_informative=10, random_state=42)

X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(
    X_cal, y_cal, test_size=0.5, random_state=42
)

# Train models
models_cal = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression()
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, model) in enumerate(models_cal.items()):
    model.fit(X_train_cal, y_train_cal)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test_cal)[:, 1]
    else:
        probs = model.decision_function(X_test_cal)
        probs = (probs - probs.min()) / (probs.max() - probs.min())
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test_cal, probs, n_bins=10
    )
    
    # Brier score
    brier_score = np.mean((probs - y_test_cal) ** 2)
    
    # Plot
    ax = axes[idx]
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
            linewidth=2, markersize=8, label=f'{name}')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{name}\nBrier Score: {brier_score:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration 6: Feature Independence Analysis
print("\n=== Demonstration 6: Feature Independence Assumption ===")

# Create dataset with varying feature correlations
n_samples = 1000

# Independent features
X_indep = np.random.randn(n_samples, 2)
y_indep = (X_indep[:, 0] + X_indep[:, 1] > 0).astype(int)

# Correlated features
mean = [0, 0]
cov_matrix = [[1, 0.9], [0.9, 1]]  # High correlation
X_corr = np.random.multivariate_normal(mean, cov_matrix, n_samples)
y_corr = (X_corr[:, 0] + X_corr[:, 1] > 0).astype(int)

# Evaluate NB performance
datasets_indep = {
    'Independent Features': (X_indep, y_indep),
    'Correlated Features': (X_corr, y_corr)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (name, (X, y)) in enumerate(datasets_indep.items()):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Train models
    gnb = GaussianNB()
    lr = LogisticRegression()
    
    gnb.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    # Plot data
    ax = axes[idx, 0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.5, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.5, label='Class 1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'{name}\nCorrelation: {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.3f}')
    ax.legend()
    
    # Compare decision boundaries
    ax = axes[idx, 1]
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    
    # NB predictions
    Z_nb = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_nb = Z_nb.reshape(xx.shape)
    
    # LR predictions
    Z_lr = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_lr = Z_lr.reshape(xx.shape)
    
    # Plot difference
    diff = Z_nb - Z_lr
    im = ax.contourf(xx, yy, diff, levels=20, cmap='RdBu_r', alpha=0.8)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=20, alpha=0.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=20, alpha=0.5)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'P(NB) - P(LR)\nNB: {gnb.score(X_test, y_test):.3f}, '
                 f'LR: {lr.score(X_test, y_test):.3f}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

---

## 6. Advanced Topics {#advanced-topics}

### Semi-Supervised Naive Bayes

Incorporates unlabeled data:
1. Initialize with labeled data
2. Predict on unlabeled data
3. Use predictions to update parameters
4. Iterate (EM algorithm)

### Online/Incremental Learning

Update parameters with new data:
```python
# For Gaussian NB
n_new = len(X_new)
n_old = self.n_samples_seen_[c]
n_total = n_old + n_new

# Update mean
self.theta_[c] = (n_old * self.theta_[c] + n_new * mean_new) / n_total

# Update variance
self.sigma_[c] = (n_old * self.sigma_[c] + n_new * var_new) / n_total
```

### Weighted Naive Bayes

Weight features differently:
```
P(y|X) ∝ P(y) × ∏ᵢ P(xᵢ|y)^wᵢ
```

### Feature Selection for NB

1. **Mutual Information**: I(X; Y)
2. **Chi-squared test**: For categorical features
3. **Information Gain**: Entropy reduction

### Handling Continuous Features

1. **Gaussian assumption**: Standard approach
2. **Kernel Density Estimation**: Non-parametric
3. **Discretization**: Convert to bins
4. **Gaussian Mixture Models**: Multiple modes

---

## 7. Comprehensive Interview Questions & Answers {#interview-qa}

### Fundamental Understanding

**Q1: Explain Naive Bayes classifier. Why is it called "naive"?**

**A:** Naive Bayes is a probabilistic classifier based on Bayes' theorem that assumes conditional independence between features given the class label.

**Why "naive"?**
The conditional independence assumption is naive because features are rarely independent in real data. For example, in text classification, words like "machine" and "learning" often appear together, violating independence.

**Despite being naive**, it works well because:
1. The decision boundary may still be approximately correct
2. We need the correct ranking of probabilities, not exact values
3. Dependencies often cancel out across many features

**Core equation**:
```
P(y|x₁,...,xₙ) ∝ P(y) × ∏ᵢ P(xᵢ|y)
```

**Q2: What's the difference between generative and discriminative classifiers? Where does Naive Bayes fit?**

**A:** 

**Generative models** (like Naive Bayes):
- Model P(X|Y) and P(Y)
- Can generate new samples
- Learn class distributions
- Work well with small data

**Discriminative models** (like Logistic Regression):
- Model P(Y|X) directly
- Focus on decision boundary
- Don't model data distribution
- Usually better with large data

**Naive Bayes is generative** because it:
1. Models how data is generated for each class
2. Can sample new data points
3. Provides insight into feature distributions

**Q3: When would you use different variants of Naive Bayes?**

**A:** Choose based on feature types:

**Gaussian NB**:
- Continuous features
- Assumes normal distribution
- Example: Sensor readings, measurements

**Multinomial NB**:
- Discrete counts
- Text classification (word counts)
- Example: Document classification, recommendation systems

**Bernoulli NB**:
- Binary features
- Presence/absence
- Example: Text with binary word occurrence

**Complement NB**:
- Imbalanced datasets
- Better performance on minority class
- Uses complement class statistics

### Mathematical Deep Dive

**Q4: Derive the Naive Bayes classification rule from Bayes' theorem.**

**A:** Starting with Bayes' theorem:

```
P(y|X) = P(X|y)P(y) / P(X)
```

For classification, we want:
```
ŷ = argmax_y P(y|X)
```

Since P(X) is constant across classes:
```
ŷ = argmax_y P(X|y)P(y)
```

With naive assumption:
```
P(X|y) = P(x₁,...,xₙ|y) = ∏ᵢ P(xᵢ|y)
```

Therefore:
```
ŷ = argmax_y P(y) ∏ᵢ P(xᵢ|y)
```

In log space (to avoid underflow):
```
ŷ = argmax_y [log P(y) + Σᵢ log P(xᵢ|y)]
```

**Q5: Why do we use Laplace smoothing? What happens without it?**

**A:** **Laplace smoothing** prevents zero probabilities:

**Without smoothing**: If a word never appears with a class in training:
- P(word|class) = 0
- Entire product becomes 0
- Class can never be predicted

**With smoothing**:
```
P(xᵢ|y) = (count(xᵢ,y) + α) / (count(y) + α|V|)
```

**Effects**:
- No zero probabilities
- Slight bias toward uniform distribution
- α = 1 (Laplace), α < 1 (Lidstone)

**Example**: New word in test
- Without: P(new_word|class) = 0/100 = 0 → kills prediction
- With: P(new_word|class) = 1/101 ≈ 0.01 → allows prediction

### Practical Applications

**Q6: How does Naive Bayes handle missing values?**

**A:** Several strategies:

1. **Ignore missing features**:
   ```python
   # Skip feature in product
   P(y|X) ∝ P(y) × ∏ᵢ:xᵢ≠missing P(xᵢ|y)
   ```

2. **Imputation**:
   - Mean/mode for training
   - Class-conditional mean/mode

3. **Missing as separate category**:
   - Treat "missing" as feature value
   - Learn P(missing|y)

4. **Probabilistic imputation**:
   - Sample from P(xᵢ|y)
   - Multiple imputation

**Best practice**: Method 1 (ignore) often works well due to NB's probabilistic nature.

**Q7: Your Naive Bayes classifier has 90% accuracy on training but 60% on test. What's wrong?**

**A:** Several possible issues:

1. **Overfitting to training data**:
   - Too little smoothing
   - Solution: Increase α parameter

2. **Feature distribution mismatch**:
   - Test data has different distribution
   - Solution: Check feature statistics

3. **Curse of dimensionality**:
   - Too many features, sparse data
   - Solution: Feature selection

4. **Wrong NB variant**:
   - Using Gaussian for discrete data
   - Solution: Choose appropriate variant

5. **Class imbalance change**:
   - Different class ratios in test
   - Solution: Adjust priors

**Diagnosis steps**:
- Plot feature distributions
- Check calibration
- Try different smoothing
- Reduce features

### Algorithm Comparison

**Q8: Compare Naive Bayes with Logistic Regression.**

**A:** 

| Aspect | Naive Bayes | Logistic Regression |
|--------|-------------|-------------------|
| **Type** | Generative | Discriminative |
| **Assumptions** | Feature independence | Log-linear relationship |
| **Training** | Closed-form (counting) | Iterative optimization |
| **Speed** | Very fast | Slower |
| **Data needed** | Works with less data | Needs more data |
| **Interpretability** | Feature distributions | Feature weights |
| **Calibration** | Often poor | Usually better |
| **Correlated features** | Problematic | Handles well |

**When to use NB over LR**:
- Limited training data
- Need fast training/prediction
- Features reasonably independent
- Baseline model

**Q9: Why does Naive Bayes often work well for text classification despite violated assumptions?**

**A:** Several reasons:

1. **High dimensionality**:
   - Many features (words)
   - Dependencies average out
   - Decision boundary approximately correct

2. **Sparse features**:
   - Most words absent in document
   - Reduces impact of dependencies

3. **Task characteristics**:
   - Often just need correct ranking
   - Not exact probabilities

4. **Empirical success**:
   - Works well for spam, sentiment
   - Fast and simple baseline

5. **Zipf's law**:
   - Few common words, many rare
   - Rare words often independent

### Advanced Topics

**Q10: How do you implement incremental/online Naive Bayes?**

**A:** Update statistics with new data:

```python
class IncrementalGaussianNB:
    def partial_fit(self, X, y):
        for c in np.unique(y):
            X_c = X[y == c]
            
            if c not in self.class_counts:
                # First time seeing this class
                self.class_counts[c] = len(X_c)
                self.means[c] = np.mean(X_c, axis=0)
                self.vars[c] = np.var(X_c, axis=0)
            else:
                # Update existing statistics
                n_old = self.class_counts[c]
                n_new = len(X_c)
                n_total = n_old + n_new
                
                # Update mean
                old_mean = self.means[c]
                new_mean = np.mean(X_c, axis=0)
                self.means[c] = (n_old * old_mean + n_new * new_mean) / n_total
                
                # Update variance (parallel algorithm)
                old_var = self.vars[c]
                new_var = np.var(X_c, axis=0)
                self.vars[c] = (n_old * old_var + n_new * new_var + 
                               n_old * n_new * (old_mean - new_mean)**2 / n_total) / n_total
                
                self.class_counts[c] = n_total
```

**Applications**:
- Streaming data
- Large datasets
- Adaptive systems

**Q11: Explain the decision boundary of Gaussian Naive Bayes.**

**A:** The decision boundary is **quadratic** (not linear like logistic regression).

For binary classification:
```
P(y=1|x) = P(y=0|x)
```

Taking log and expanding:
```
log P(y=1) - 0.5 Σᵢ[(xᵢ-μ₁ᵢ)²/σ²₁ᵢ + log(2πσ²₁ᵢ)] = 
log P(y=0) - 0.5 Σᵢ[(xᵢ-μ₀ᵢ)²/σ²₀ᵢ + log(2πσ²₀ᵢ)]
```

This simplifies to a quadratic function of x when σ₁ᵢ ≠ σ₀ᵢ.

**Special case**: When variances are equal (σ₁ᵢ = σ₀ᵢ), boundary becomes linear.

### Real-world Scenarios

**Q12: You're building a spam filter. How would you use Naive Bayes?**

**A:** Complete approach:

1. **Data preparation**:
   ```python
   # Tokenization
   tokens = word_tokenize(email.lower())
   # Remove stopwords
   tokens = [t for t in tokens if t not in stopwords]
   ```

2. **Feature extraction**:
   - Bag of words or TF-IDF
   - Include header features
   - URL/attachment indicators

3. **Model choice**:
   - Multinomial NB for word counts
   - Bernoulli NB for word presence

4. **Training**:
   ```python
   vectorizer = CountVectorizer(max_features=10000)
   X = vectorizer.fit_transform(emails)
   mnb = MultinomialNB(alpha=1.0)
   mnb.fit(X, y)
   ```

5. **Handling challenges**:
   - Imbalanced data: Adjust priors
   - Evolving spam: Online learning
   - False positives: Tune threshold

6. **Production**:
   - Fast prediction
   - Update with user feedback
   - Monitor performance

**Q13: How do you debug poor Naive Bayes performance?**

**A:** Systematic approach:

1. **Check assumptions**:
   - Plot feature distributions by class
   - Calculate feature correlations
   - Verify feature types match NB variant

2. **Analyze predictions**:
   ```python
   # Get top features for misclassified
   log_probs = model.feature_log_prob_
   for misclassified in errors:
       feature_contributions = X[misclassified] * log_probs
   ```

3. **Common fixes**:
   - Wrong variant → Switch
   - Zero probabilities → Increase smoothing
   - Correlated features → Feature selection
   - Poor calibration → Calibration methods

4. **Diagnostic plots**:
   - Calibration curves
   - Feature importance
   - Confusion matrix patterns

**Q14: Explain how Naive Bayes can be used for feature selection.**

**A:** NB naturally provides feature importance:

1. **Likelihood ratios**:
   ```python
   # For binary classification
   log_ratio = log(P(feature|class1) / P(feature|class0))
   ```

2. **Mutual information**:
   ```python
   MI(X,Y) = Σ P(x,y) log(P(x,y) / (P(x)P(y)))
   ```

3. **Class-conditional differences**:
   - Large |μ₁ - μ₀| / σ for Gaussian
   - High |P(word|spam) - P(word|ham)| for text

4. **Implementation**:
   ```python
   # Select top k features
   feature_scores = np.abs(nb.theta_[1] - nb.theta_[0])
   top_k_idx = np.argsort(feature_scores)[-k:]
   ```

### Interview Problems

**Q15: Implement Gaussian Naive Bayes with feature weighting.**

**A:** 
```python
class WeightedGaussianNB:
    def __init__(self, feature_weights=None):
        self.feature_weights = feature_weights
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-6,
                'prior': len(X_c) / len(y)
            }
            
        if self.feature_weights is None:
            self.feature_weights = np.ones(X.shape[1])
            
    def predict_proba(self, X):
        probs = []
        
        for c in self.classes:
            prior = np.log(self.params[c]['prior'])
            
            # Weighted log likelihood
            log_likelihood = -0.5 * np.sum(
                self.feature_weights * (
                    np.log(2 * np.pi * self.params[c]['var']) +
                    (X - self.params[c]['mean'])**2 / self.params[c]['var']
                ), axis=1
            )
            
            probs.append(prior + log_likelihood)
            
        probs = np.array(probs).T
        # Normalize
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)
```

**Q16: Your NB classifier outputs probabilities of 0.99 for all positive predictions but many are wrong. Why?**

**A:** **Poor calibration** - common in NB:

**Causes**:
1. Independence assumption violated
2. Feature distributions don't match assumed (e.g., not Gaussian)
3. Multiplying many small probabilities

**Solutions**:

1. **Calibration methods**:
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated_nb = CalibratedClassifierCV(nb, method='isotonic')
   ```

2. **Adjust decision threshold**:
   - Don't use 0.5
   - Optimize for metric

3. **Use for ranking only**:
   - NB good at ordering
   - Poor at exact probabilities

4. **Alternative scoring**:
   - Use log probabilities
   - Normalized scores

**Q17: How would you modify NB for multi-label classification?**

**A:** Treat each label independently:

```python
class MultiLabelNB:
    def __init__(self, base_nb=GaussianNB):
        self.base_nb = base_nb
        
    def fit(self, X, Y):
        # Y is n_samples x n_labels
        self.n_labels = Y.shape[1]
        self.classifiers = []
        
        for i in range(self.n_labels):
            clf = self.base_nb()
            clf.fit(X, Y[:, i])
            self.classifiers.append(clf)
            
    def predict_proba(self, X):
        # Return probabilities for each label
        probs = np.zeros((len(X), self.n_labels))
        
        for i, clf in enumerate(self.classifiers):
            probs[:, i] = clf.predict_proba(X)[:, 1]
            
        return probs
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
```

**Assumptions**: Labels are independent (often violated but works okay).

**Q18: Explain how NB handles continuous vs discrete features differently.**

**A:** Different probability distributions:

**Continuous (Gaussian NB)**:
- Assumes normal distribution
- Parameters: μ, σ² per feature per class
- PDF: Complex calculation

**Discrete (Multinomial/Bernoulli)**:
- Count-based probabilities
- Parameters: P(xᵢ=v|y) for each value v
- Simple lookup

**Mixed features**:
```python
class MixedNB:
    def __init__(self, continuous_features, discrete_features):
        self.continuous_idx = continuous_features
        self.discrete_idx = discrete_features
        self.gaussian_nb = GaussianNB()
        self.multinomial_nb = MultinomialNB()
        
    def fit(self, X, y):
        X_cont = X[:, self.continuous_idx]
        X_disc = X[:, self.discrete_idx]
        
        self.gaussian_nb.fit(X_cont, y)
        self.multinomial_nb.fit(X_disc, y)
        
    def predict_log_proba(self, X):
        X_cont = X[:, self.continuous_idx]
        X_disc = X[:, self.discrete_idx]
        
        log_proba_cont = self.gaussian_nb.predict_log_proba(X_cont)
        log_proba_disc = self.multinomial_nb.predict_log_proba(X_disc)
        
        # Combine (sum in log space)
        return log_proba_cont + log_proba_disc
```

**Q19: How do you determine if the independence assumption is severely violated?**

**A:** Several diagnostic approaches:

1. **Feature correlation analysis**:
   ```python
   corr_matrix = np.corrcoef(X.T)
   high_corr_pairs = np.where(np.abs(corr_matrix) > 0.8)
   ```

2. **Compare with discriminative model**:
   - If LR >> NB, likely dependence issues

3. **Conditional independence tests**:
   ```python
   # Chi-squared test for discrete features
   from sklearn.feature_selection import chi2
   chi2_stats = chi2(X, y)
   ```

4. **Error analysis**:
   - Systematic errors indicate assumption violation

5. **Visualization**:
   - Plot feature pairs by class
   - Look for clear dependencies

**Q20: Design a Naive Bayes system for real-time recommendation.**

**A:** Architecture:

1. **Feature engineering**:
   - User features: demographics, history
   - Item features: category, popularity
   - Context: time, location

2. **Model design**:
   ```python
   class RecommenderNB:
       def __init__(self):
           self.user_nb = GaussianNB()  # User features
           self.item_nb = MultinomialNB()  # Item features
           self.context_nb = GaussianNB()  # Context
           
       def train(self, interactions):
           # Extract features
           user_features = extract_user_features(interactions)
           item_features = extract_item_features(interactions)
           context_features = extract_context_features(interactions)
           
           # Train on positive/negative interactions
           labels = interactions['clicked']
           
           self.user_nb.fit(user_features, labels)
           self.item_nb.fit(item_features, labels)
           self.context_nb.fit(context_features, labels)
           
       def score(self, user, item, context):
           # Combine probabilities
           user_score = self.user_nb.predict_proba([user])[0, 1]
           item_score = self.item_nb.predict_proba([item])[0, 1]
           context_score = self.context_nb.predict_proba([context])[0, 1]
           
           # Weighted combination
           return 0.4 * user_score + 0.4 * item_score + 0.2 * context_score
   ```

3. **Real-time requirements**:
   - Precompute user/item statistics
   - Cache predictions
   - Online updates for new items

4. **Advantages for recommendations**:
   - Fast inference
   - Handles cold start (with priors)
   - Interpretable (which features matter)

---

## Practice Problems

1. Implement semi-supervised Naive Bayes
2. Create streaming NB with concept drift detection
3. Build feature selection using NB scores
4. Implement kernel density NB for continuous features
5. Create hierarchical NB for taxonomic classification
6. Build confidence intervals for NB predictions

## Key Takeaways

1. **NB is simple but effective** - Often great baseline
2. **Independence assumption rarely holds** - But often works anyway
3. **Different variants for different data** - Choose wisely
4. **Very fast training and prediction** - Just counting
5. **Poor probability calibration** - Better for ranking
6. **Works well with high dimensions** - Text classification
7. **Handles missing data naturally** - Probabilistic framework
8. **Small data requirements** - Generative advantage