# Day 11: Evaluation Metrics - R², Adjusted R², F1-score, Precision, Recall

## 📚 Table of Contents
1. [Introduction to Model Evaluation](#introduction)
2. [Regression Metrics](#regression-metrics)
3. [Classification Metrics](#classification-metrics)
4. [Advanced Metrics and Considerations](#advanced-metrics)
5. [Implementation and Visualization](#implementation)
6. [Metric Selection Strategy](#metric-selection)
7. [Comprehensive Interview Q&A](#interview-qa)

---

## 1. Introduction to Model Evaluation {#introduction}

### Why Metrics Matter

Choosing the right evaluation metric is crucial because:
1. **Guides optimization**: What you measure is what you optimize
2. **Business alignment**: Metrics should reflect business goals
3. **Model comparison**: Need consistent way to compare models
4. **Problem-specific**: Different problems need different metrics

### Categories of Metrics

1. **Regression Metrics**: Continuous target variables
   - R², Adjusted R², MSE, MAE, MAPE

2. **Classification Metrics**: Categorical target variables
   - Accuracy, Precision, Recall, F1-score, ROC-AUC

3. **Probabilistic Metrics**: Probability outputs
   - Log loss, Brier score, Calibration

4. **Ranking Metrics**: Order matters
   - NDCG, MAP, MRR

### The Importance of Context

No metric is universally best - selection depends on:
- Problem domain
- Cost of different errors
- Class imbalance
- Business requirements

---

## 2. Regression Metrics {#regression-metrics}

### R-Squared (Coefficient of Determination)

**Definition**: Proportion of variance in target variable explained by the model.

**Formula**:
```
R² = 1 - (SS_res / SS_tot)
R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

Where:
- SS_res: Residual sum of squares
- SS_tot: Total sum of squares
- ȳ: Mean of y

**Properties**:
- Range: (-∞, 1] for test set, [0, 1] for training
- R² = 1: Perfect fit
- R² = 0: Model performs as well as mean
- R² < 0: Model worse than predicting mean

**Interpretation**:
- 0.7 means 70% of variance is explained by model

### Adjusted R-Squared

**Problem with R²**: Always increases when adding features, even irrelevant ones.

**Formula**:
```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]
```

Where:
- n: Number of samples
- p: Number of features

**Properties**:
- Penalizes for adding features
- Can decrease if feature doesn't improve model enough
- Better for model comparison with different feature counts

### Mean Squared Error (MSE) and Variants

**MSE**:
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Root Mean Squared Error (RMSE)**:
```
RMSE = √MSE
```
- Same units as target variable
- More interpretable than MSE

**Mean Absolute Error (MAE)**:
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```
- Less sensitive to outliers than MSE
- Same units as target

**Mean Absolute Percentage Error (MAPE)**:
```
MAPE = (100/n) Σ|yᵢ - ŷᵢ| / |yᵢ|
```
- Scale-independent
- Issues when y ≈ 0

### When to Use Which Regression Metric

1. **R²**: General model explanation power
2. **Adjusted R²**: Comparing models with different features
3. **MSE/RMSE**: When large errors are particularly bad
4. **MAE**: When errors scale linearly with impact
5. **MAPE**: Comparing across different scales

---

## 3. Classification Metrics {#classification-metrics}

### Confusion Matrix Foundation

For binary classification:
```
                 Predicted
              Positive  Negative
Actual  Pos     TP       FN
        Neg     FP       TN
```

- **TP**: True Positives (correctly predicted positive)
- **TN**: True Negatives (correctly predicted negative)
- **FP**: False Positives (Type I error)
- **FN**: False Negatives (Type II error)

### Accuracy

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Limitations**:
- Misleading for imbalanced datasets
- Doesn't distinguish between error types

**When to use**:
- Balanced datasets
- All errors equally important

### Precision (Positive Predictive Value)

**Formula**:
```
Precision = TP / (TP + FP)
```

**Interpretation**: Of all positive predictions, what fraction is correct?

**When to use**:
- False positives are costly
- Spam detection (don't want to mark legitimate emails as spam)
- Medical diagnosis (avoid unnecessary treatment)

### Recall (Sensitivity, True Positive Rate)

**Formula**:
```
Recall = TP / (TP + FN)
```

**Interpretation**: Of all actual positives, what fraction did we catch?

**When to use**:
- False negatives are costly
- Disease screening (don't want to miss sick patients)
- Fraud detection (catch all fraudulent transactions)

### F1-Score

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Properties**:
- Harmonic mean of precision and recall
- Balanced metric between precision and recall
- Range: [0, 1]

**F-beta Score** (generalization):
```
Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```
- β < 1: Weights precision higher
- β > 1: Weights recall higher
- β = 1: F1-score (equal weight)

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic)**:
- Plot of TPR (Recall) vs FPR at various thresholds
- FPR = FP / (FP + TN)

**AUC (Area Under Curve)**:
- Single number summary of ROC curve
- Range: [0, 1]
- 0.5: Random classifier
- 1.0: Perfect classifier

**Properties**:
- Threshold-independent
- Good for ranking quality
- Robust to class imbalance (debated)

### Precision-Recall Curve

Alternative to ROC for imbalanced datasets:
- Plot Precision vs Recall
- Better for imbalanced datasets
- Area under PR curve (AP) as summary

---

## 4. Advanced Metrics and Considerations {#advanced-metrics}

### Multi-class Classification Metrics

**Macro Average**: Calculate metric for each class, then average
```
Macro-F1 = (1/k) Σ F1ᵢ
```

**Weighted Average**: Weight by class frequency
```
Weighted-F1 = Σ (nᵢ/n) × F1ᵢ
```

**Micro Average**: Calculate globally
```
Micro-F1 = F1(ΣTPᵢ, ΣFPᵢ, ΣFNᵢ)
```

### Matthews Correlation Coefficient (MCC)

**Formula**:
```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Properties**:
- Range: [-1, 1]
- Balanced metric for binary classification
- Works well with imbalanced datasets

### Cohen's Kappa

Measures agreement correcting for chance:
```
κ = (p₀ - pₑ) / (1 - pₑ)
```
Where:
- p₀: Observed agreement
- pₑ: Expected agreement by chance

### Log Loss (Cross-Entropy)

For probabilistic predictions:
```
LogLoss = -(1/n) Σ[yᵢ log(p̂ᵢ) + (1-yᵢ)log(1-p̂ᵢ)]
```

**Properties**:
- Penalizes confident wrong predictions heavily
- Proper scoring rule
- Used in logistic regression optimization

### Calibration Metrics

**Brier Score**:
```
BS = (1/n) Σ(p̂ᵢ - yᵢ)²
```

**Expected Calibration Error (ECE)**:
Measures how well predicted probabilities match actual frequencies.

---

## 5. Implementation and Visualization {#implementation}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    # Regression metrics
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    # Advanced metrics
    matthews_corrcoef, cohen_kappa_score, log_loss,
    # Multi-class
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# Demonstration 1: Regression Metrics Deep Dive
print("=== Demonstration 1: Regression Metrics ===")

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                               noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train models with different complexities
models_reg = {
    'Linear': LinearRegression(),
    'Linear + L2': LinearRegression(),  # Will add polynomial features
}

# Custom implementation of metrics
def calculate_r2(y_true, y_pred):
    """Calculate R² from scratch"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_adjusted_r2(y_true, y_pred, n_features):
    """Calculate Adjusted R² from scratch"""
    n = len(y_true)
    r2 = calculate_r2(y_true, y_pred)
    return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))

# Train and evaluate
results_reg = {}
for name, model in models_reg.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_train = model.predict(X_train_reg)
    y_pred_test = model.predict(X_test_reg)
    
    # Calculate all metrics
    results_reg[name] = {
        'R² (train)': r2_score(y_train_reg, y_pred_train),
        'R² (test)': r2_score(y_test_reg, y_pred_test),
        'Adjusted R² (test)': calculate_adjusted_r2(y_test_reg, y_pred_test, X_reg.shape[1]),
        'MSE': mean_squared_error(y_test_reg, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_test)),
        'MAE': mean_absolute_error(y_test_reg, y_pred_test),
        'MAPE': mean_absolute_percentage_error(y_test_reg, y_pred_test) * 100
    }

# Display results
results_df = pd.DataFrame(results_reg).T
print("\nRegression Metrics Comparison:")
print(results_df.round(4))

# Visualization of regression metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² visualization
ax = axes[0, 0]
model = LinearRegression().fit(X_train_reg, y_train_reg)
y_pred = model.predict(X_test_reg)

ax.scatter(y_test_reg, y_pred, alpha=0.5)
ax.plot([y_test_reg.min(), y_test_reg.max()], 
        [y_test_reg.min(), y_test_reg.max()], 
        'r--', lw=2)
ax.set_xlabel('True Values')
ax.set_ylabel('Predictions')
ax.set_title(f'Predicted vs Actual\nR² = {r2_score(y_test_reg, y_pred):.3f}')

# Residual plot
ax = axes[0, 1]
residuals = y_test_reg - y_pred
ax.scatter(y_pred, residuals, alpha=0.5)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')

# Distribution of errors
ax = axes[1, 0]
ax.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--')
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title(f'Residual Distribution\nMean: {np.mean(residuals):.3f}, Std: {np.std(residuals):.3f}')

# Different error metrics visualization
ax = axes[1, 1]
errors = {
    'MSE': mean_squared_error(y_test_reg, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred)),
    'MAE': mean_absolute_error(y_test_reg, y_pred),
}
bars = ax.bar(errors.keys(), errors.values())
ax.set_ylabel('Error Value')
ax.set_title('Comparison of Error Metrics')
for bar, value in zip(bars, errors.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Demonstration 2: R² vs Adjusted R² with Feature Addition
print("\n=== Demonstration 2: R² vs Adjusted R² ===")

# Simulate adding random features
n_samples = 100
n_true_features = 5
X_true = np.random.randn(n_samples, n_true_features)
true_coeffs = np.array([2, -1.5, 0.5, 1, -0.5])
y_true = X_true @ true_coeffs + np.random.randn(n_samples) * 0.5

# Add random features progressively
max_random_features = 50
r2_scores = []
adj_r2_scores = []
n_features_range = range(n_true_features, n_true_features + max_random_features + 1, 5)

for n_total_features in n_features_range:
    # Add random features
    n_random = n_total_features - n_true_features
    if n_random > 0:
        X_random = np.random.randn(n_samples, n_random)
        X_combined = np.hstack([X_true, X_random])
    else:
        X_combined = X_true
    
    # Fit model
    model = LinearRegression()
    model.fit(X_combined, y_true)
    y_pred = model.predict(X_combined)
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    adj_r2 = calculate_adjusted_r2(y_true, y_pred, n_total_features)
    
    r2_scores.append(r2)
    adj_r2_scores.append(adj_r2)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(n_features_range, r2_scores, 'o-', label='R²', markersize=8)
plt.plot(n_features_range, adj_r2_scores, 's-', label='Adjusted R²', markersize=8)
plt.axvline(x=n_true_features, color='red', linestyle='--', 
            label='True number of features')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('R² vs Adjusted R²: Effect of Adding Random Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Demonstration 3: Classification Metrics Deep Dive
print("\n=== Demonstration 3: Classification Metrics ===")

# Generate imbalanced classification data
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                   n_redundant=5, n_classes=2, weights=[0.9, 0.1],
                                   flip_y=0.05, random_state=42)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

# Scale features
scaler = StandardScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Train different models
models_clf = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Custom implementation of classification metrics
def calculate_precision_recall_f1(y_true, y_pred):
    """Calculate precision, recall, and F1 from scratch"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Evaluate models
results_clf = {}
for name, model in models_clf.items():
    model.fit(X_train_clf_scaled, y_train_clf)
    y_pred = model.predict(X_test_clf_scaled)
    y_proba = model.predict_proba(X_test_clf_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    precision, recall, f1 = calculate_precision_recall_f1(y_test_clf, y_pred)
    
    results_clf[name] = {
        'Accuracy': accuracy_score(y_test_clf, y_pred),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'MCC': matthews_corrcoef(y_test_clf, y_pred),
        'Cohen\'s Kappa': cohen_kappa_score(y_test_clf, y_pred),
        'Log Loss': log_loss(y_test_clf, y_proba) if y_proba is not None else np.nan
    }

# Display results
results_clf_df = pd.DataFrame(results_clf).T
print("\nClassification Metrics Comparison:")
print(results_clf_df.round(4))

# Confusion Matrix Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models_clf.items()):
    y_pred = model.predict(X_test_clf_scaled)
    cm = confusion_matrix(y_test_clf, y_pred)
    
    # Plot confusion matrix
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test_clf, y_pred):.3f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# Hide extra subplots
for idx in range(len(models_clf), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Demonstration 4: Precision-Recall Tradeoff
print("\n=== Demonstration 4: Precision-Recall Tradeoff ===")

# Use logistic regression for probability outputs
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_clf_scaled, y_train_clf)
y_scores = log_reg.predict_proba(X_test_clf_scaled)[:, 1]

# Calculate precision-recall for different thresholds
thresholds = np.linspace(0, 1, 100)
precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_scores >= threshold).astype(int)
    prec = precision_score(y_test_clf, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test_clf, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test_clf, y_pred_thresh, zero_division=0)
    
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

# Find optimal threshold (max F1)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Metrics vs Threshold
ax = axes[0]
ax.plot(thresholds, precisions, label='Precision', linewidth=2)
ax.plot(thresholds, recalls, label='Recall', linewidth=2)
ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
ax.axvline(x=optimal_threshold, color='red', linestyle='--', 
           label=f'Optimal threshold: {optimal_threshold:.2f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Metrics vs Decision Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# Precision-Recall Curve
ax = axes[1]
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test_clf, y_scores)
avg_precision = average_precision_score(y_test_clf, y_scores)

ax.plot(recall_curve, precision_curve, linewidth=2, 
        label=f'AP = {avg_precision:.3f}')
ax.plot(recalls[optimal_idx], precisions[optimal_idx], 'ro', markersize=10,
        label=f'Optimal F1 point')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration 5: ROC Curves and AUC
print("\n=== Demonstration 5: ROC Curves and AUC ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC curves for all models
ax = axes[0]
for name, model in models_clf.items():
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test_clf_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_clf, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# ROC vs PR curve comparison
ax = axes[1]
# For imbalanced dataset
y_scores = log_reg.predict_proba(X_test_clf_scaled)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_test_clf, y_scores)
roc_auc = auc(fpr, tpr)

# PR
precision_curve, recall_curve, _ = precision_recall_curve(y_test_clf, y_scores)
pr_auc = average_precision_score(y_test_clf, y_scores)

# Baseline for PR curve (positive class proportion)
baseline_pr = np.sum(y_test_clf) / len(y_test_clf)

ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot(recall_curve, precision_curve, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
ax.axhline(y=baseline_pr, color='gray', linestyle='--', 
           label=f'Baseline PR = {baseline_pr:.3f}')
ax.set_xlabel('Recall / FPR')
ax.set_ylabel('Precision / TPR')
ax.set_title('ROC vs PR Curve (Imbalanced Data)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration 6: Multi-class Classification Metrics
print("\n=== Demonstration 6: Multi-class Metrics ===")

# Generate multi-class data
X_multi, y_multi = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                       n_classes=4, n_clusters_per_class=2,
                                       random_state=42)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

# Train model
rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = rf_multi.predict(X_test_multi)

# Calculate metrics for each class
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_multi, y_pred_multi, average=None
)

# Calculate different averaging methods
metrics_multi = {
    'Macro': precision_recall_fscore_support(y_test_multi, y_pred_multi, average='macro'),
    'Weighted': precision_recall_fscore_support(y_test_multi, y_pred_multi, average='weighted'),
    'Micro': precision_recall_fscore_support(y_test_multi, y_pred_multi, average='micro')
}

# Display per-class metrics
class_metrics_df = pd.DataFrame({
    'Class': range(4),
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
print("\nPer-Class Metrics:")
print(class_metrics_df.round(3))

print("\nAveraging Methods:")
for method, (prec, rec, f1, _) in metrics_multi.items():
    print(f"{method:10} - Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# Confusion matrix for multi-class
plt.figure(figsize=(8, 6))
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(4)],
            yticklabels=[f'Class {i}' for i in range(4)])
plt.title('Multi-class Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Demonstration 7: Calibration Analysis
print("\n=== Demonstration 7: Model Calibration ===")

# Compare calibration of different models
models_calib = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, model) in enumerate(models_calib.items()):
    model.fit(X_train_clf_scaled, y_train_clf)
    y_proba = model.predict_proba(X_test_clf_scaled)[:, 1]
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test_clf, y_proba, n_bins=10
    )
    
    # Brier score
    brier_score = np.mean((y_proba - y_test_clf) ** 2)
    
    ax = axes[idx]
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
            label=f'{name}\nBrier Score: {brier_score:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Plot: {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstration 8: Custom Metric Implementation
print("\n=== Demonstration 8: Custom Metrics ===")

def weighted_f1_score(y_true, y_pred, weight_precision=0.3, weight_recall=0.7):
    """Custom F-score with different weights for precision and recall"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0
    
    weighted_f1 = (1 + (weight_recall/weight_precision)**2) * precision * recall / \
                  ((weight_recall/weight_precision)**2 * precision + recall)
    
    return weighted_f1

def profit_score(y_true, y_pred, tp_profit=100, fp_cost=20, fn_cost=50, tn_profit=5):
    """Custom metric based on business costs/profits"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_profit = tp * tp_profit - fp * fp_cost - fn * fn_cost + tn * tn_profit
    max_profit = (tp + fn) * tp_profit + (tn + fp) * tn_profit  # Perfect classifier
    
    return total_profit / max_profit if max_profit > 0 else 0

# Apply custom metrics
y_pred = log_reg.predict(X_test_clf_scaled)

custom_metrics = {
    'Standard F1': f1_score(y_test_clf, y_pred),
    'Recall-weighted F1': weighted_f1_score(y_test_clf, y_pred, 0.3, 0.7),
    'Precision-weighted F1': weighted_f1_score(y_test_clf, y_pred, 0.7, 0.3),
    'Profit Score': profit_score(y_test_clf, y_pred)
}

print("\nCustom Metrics:")
for metric, value in custom_metrics.items():
    print(f"{metric:20}: {value:.3f}")

# Visualize impact of different metrics
thresholds = np.linspace(0.1, 0.9, 20)
metrics_by_threshold = {metric: [] for metric in custom_metrics.keys()}

for threshold in thresholds:
    y_pred_thresh = (y_scores >= threshold).astype(int)
    
    metrics_by_threshold['Standard F1'].append(
        f1_score(y_test_clf, y_pred_thresh))
    metrics_by_threshold['Recall-weighted F1'].append(
        weighted_f1_score(y_test_clf, y_pred_thresh, 0.3, 0.7))
    metrics_by_threshold['Precision-weighted F1'].append(
        weighted_f1_score(y_test_clf, y_pred_thresh, 0.7, 0.3))
    metrics_by_threshold['Profit Score'].append(
        profit_score(y_test_clf, y_pred_thresh))

plt.figure(figsize=(10, 6))
for metric, values in metrics_by_threshold.items():
    plt.plot(thresholds, values, 'o-', label=metric, markersize=6)

plt.xlabel('Decision Threshold')
plt.ylabel('Metric Value')
plt.title('Custom Metrics vs Decision Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. Metric Selection Strategy {#metric-selection}

### Decision Framework

1. **Problem Type**:
   - Regression → R², MSE, MAE
   - Binary Classification → Precision, Recall, F1, AUC
   - Multi-class → Macro/Weighted F1, Accuracy
   - Ranking → NDCG, MAP

2. **Data Characteristics**:
   - Imbalanced → Precision-Recall, F1, MCC
   - Balanced → Accuracy, ROC-AUC
   - Outliers present → MAE over MSE

3. **Business Requirements**:
   - Cost-sensitive → Custom profit metric
   - High stakes → Optimize for recall (medical)
   - User-facing → Optimize for precision (recommendations)

4. **Model Purpose**:
   - Screening → High recall
   - Decision making → High precision
   - Ranking → AUC, NDCG

### Common Pitfalls

1. **Using accuracy on imbalanced data**
2. **Ignoring business context**
3. **Optimizing wrong metric**
4. **Not considering metric properties**
5. **Single metric fixation**

---

## 7. Comprehensive Interview Questions & Answers {#interview-qa}

### Fundamental Understanding

**Q1: Explain R² in simple terms. What are its limitations?**

**A:** R² measures the proportion of variance in the target variable that's predictable from the features. 

**Interpretation**: 
- R² = 0.8 means 80% of the variance in y is explained by the model
- Remaining 20% is unexplained (could be noise or missing features)

**Limitations**:
1. **Always increases with more features**: Even random features increase R²
2. **Can be negative on test set**: When model is worse than predicting mean
3. **Doesn't indicate causation**: High R² doesn't mean features cause target
4. **Sensitive to outliers**: Few extreme points can inflate R²
5. **Scale dependent**: Can't compare across different targets

**When to use Adjusted R²**: When comparing models with different numbers of features, as it penalizes complexity.

**Q2: When would you use precision vs recall?**

**A:** It depends on the relative cost of false positives vs false negatives:

**Use Precision when false positives are costly**:
- Spam detection: Don't want to mark important emails as spam
- Legal document classification: Wrong classification has serious consequences
- High-precision manufacturing: Passing defective items is expensive

**Use Recall when false negatives are costly**:
- Disease screening: Missing a sick patient is dangerous
- Fraud detection: Missing fraudulent transactions costs money
- Security threats: Missing a threat could be catastrophic

**Example**: 
- Cancer screening → Optimize recall (catch all cases, even if some false alarms)
- Cancer diagnosis → Balance both (need accuracy for treatment decisions)

**Q3: Explain the F1 score. When is it not appropriate?**

**A:** F1 score is the harmonic mean of precision and recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Properties**:
- Balances precision and recall
- Ranges from 0 to 1
- Harmonic mean penalizes extreme values

**When NOT appropriate**:
1. **Heavily imbalanced costs**: When FP and FN have very different impacts
2. **Need interpretability**: F1 is harder to explain than precision/recall
3. **Multi-class with different importances**: Some classes matter more
4. **Threshold selection**: F1 is threshold-dependent

Use F-beta score for weighted importance, or custom metrics for business needs.

### Mathematical Deep Dive

**Q4: Why do we use RMSE instead of MSE?**

**A:** Several reasons:

1. **Unit consistency**: RMSE has same units as target variable
   - MSE units are squared (e.g., dollars² is not interpretable)
   - RMSE in dollars is interpretable

2. **Scale interpretation**: Easier to understand
   - RMSE = 10 means average error of ~10 units
   - MSE = 100 is harder to interpret

3. **Outlier sensitivity**: Both are sensitive, but RMSE is more interpretable
   - RMSE gives "typical" error magnitude

**When to use MSE**: 
- Optimization (mathematical convenience)
- When you want to heavily penalize large errors

**Q5: Derive the relationship between R² and correlation coefficient.**

**A:** For simple linear regression (one feature):

R² equals the square of Pearson correlation coefficient (r):
```
R² = r²
```

**Proof**:
- Correlation: r = Cov(X,Y) / (σ_X × σ_Y)
- In simple linear regression: β = r × (σ_Y / σ_X)
- R² = Explained variance / Total variance
- After substitution: R² = r²

**For multiple regression**: R² is the square of multiple correlation coefficient (correlation between y and ŷ).

### Advanced Concepts

**Q6: What is the Matthews Correlation Coefficient (MCC) and why use it?**

**A:** MCC is a balanced metric for binary classification:

```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Advantages**:
1. **Works well on imbalanced data**: Unlike accuracy
2. **Single score**: Considers all confusion matrix cells
3. **Symmetric**: No need to define positive class
4. **Range [-1, 1]**: -1 (total disagreement), 0 (random), 1 (perfect)

**When to use**:
- Imbalanced datasets
- When you need a single balanced metric
- Comparing classifiers across different datasets

**Q7: Explain AUC-ROC vs AUC-PR. When to use each?**

**A:** 

**AUC-ROC (Area Under ROC Curve)**:
- Plots TPR vs FPR at various thresholds
- Measures ranking quality
- Robust to class imbalance (controversial)

**AUC-PR (Area Under Precision-Recall Curve)**:
- Plots Precision vs Recall
- Better for imbalanced datasets
- Focuses on positive class performance

**When to use**:
- **AUC-ROC**: Balanced datasets, need overall ranking quality
- **AUC-PR**: Imbalanced datasets, care about positive class

**Example**: 
- Dataset with 1% positive class
- Random classifier: AUC-ROC ≈ 0.5, AUC-PR ≈ 0.01
- AUC-PR better reflects poor performance

### Practical Applications

**Q8: How do you handle metrics for multi-class classification?**

**A:** Three main approaches:

1. **Macro-averaging**:
   ```
   Macro-F1 = (F1_class1 + F1_class2 + ... + F1_classN) / N
   ```
   - Treats all classes equally
   - Good when all classes are important

2. **Weighted-averaging**:
   ```
   Weighted-F1 = Σ(n_i / n_total) × F1_i
   ```
   - Weights by class frequency
   - Accounts for class imbalance

3. **Micro-averaging**:
   ```
   Micro-F1 = F1(Σ TP_i, Σ FP_i, Σ FN_i)
   ```
   - Aggregates contributions across classes
   - Dominated by frequent classes

**Choice depends on**:
- Equal class importance → Macro
- Natural class imbalance → Weighted
- Overall performance → Micro

**Q9: You're building a fraud detection system. Which metrics would you track?**

**A:** Multiple metrics for different stakeholders:

**Primary Metrics**:
1. **Recall @ fixed precision**: Catch fraud while controlling false alarms
2. **Precision @ fixed recall**: Minimize customer friction
3. **Custom profit metric**: Incorporates business costs

**Secondary Metrics**:
1. **AUC-PR**: Overall model quality on imbalanced data
2. **Precision-Recall curve**: For threshold selection
3. **Alert rate**: Percentage flagged (operational capacity)

**Monitoring Metrics**:
1. **Daily fraud catch rate**: Business impact
2. **False positive rate**: Customer experience
3. **Model calibration**: Probability reliability

**Implementation**:
```python
def fraud_profit_metric(y_true, y_pred, fraud_loss=1000, review_cost=10):
    # Custom metric based on business costs
    savings = caught_fraud * fraud_loss
    costs = false_positives * review_cost
    return savings - costs
```

### Model Evaluation Strategy

**Q10: How do you design a comprehensive evaluation strategy?**

**A:** Multi-faceted approach:

1. **Define success criteria**:
   - Business metrics (revenue, user satisfaction)
   - ML metrics aligned with business goals
   - Minimum acceptable performance

2. **Choose primary metric**:
   - Single metric for optimization
   - Must align with business objective
   - Consider custom metrics

3. **Select guardrail metrics**:
   - Ensure no critical degradation
   - Example: Maintain precision > 0.9 while optimizing recall

4. **Evaluation protocol**:
   - Proper train/val/test split
   - Cross-validation for stability
   - Time-based splits if temporal

5. **Statistical significance**:
   - Confidence intervals
   - A/B testing in production
   - Multiple random seeds

**Q11: How do you handle changing metrics in production?**

**A:** 

**Challenges**:
- Distribution shift
- Changing business requirements
- Delayed feedback

**Solutions**:

1. **Monitor multiple metrics**:
   - Track distribution of predictions
   - Monitor feature distributions
   - Alert on significant changes

2. **Online evaluation**:
   - A/B testing
   - Bandit algorithms
   - Gradual rollout

3. **Feedback loops**:
   - Collect ground truth when possible
   - User feedback as proxy
   - Business metrics correlation

4. **Retraining triggers**:
   - Performance degradation
   - Distribution shift detection
   - Scheduled updates

### Common Pitfalls and Solutions

**Q12: What are common mistakes in model evaluation?**

**A:** 

1. **Data leakage**:
   - Test data information in training
   - Solution: Proper pipeline design

2. **Overfitting to validation set**:
   - Too many experiments on same validation
   - Solution: Keep true holdout set

3. **Wrong metric optimization**:
   - Accuracy on imbalanced data
   - Solution: Understand problem domain

4. **Ignoring confidence intervals**:
   - Single point estimates
   - Solution: Bootstrap confidence intervals

5. **Temporal issues**:
   - Random splits on time-series
   - Solution: Time-based splits

6. **Sample selection bias**:
   - Test set not representative
   - Solution: Stratified sampling

**Q13: How do you communicate model performance to stakeholders?**

**A:** Tailor communication to audience:

**For Technical Team**:
- Full metrics dashboard
- Confusion matrices
- ROC/PR curves
- Statistical significance

**For Business Stakeholders**:
- Business metric impact
- Simple visualizations
- Concrete examples
- Cost-benefit analysis

**Example template**:
"The model correctly identifies 85% of fraudulent transactions (recall) while maintaining 90% precision. This translates to:
- $X saved per month
- Y% reduction in fraud losses
- Z false alarms per 1000 transactions"

### Advanced Topics

**Q14: Explain calibration and its importance.**

**A:** Calibration measures how well predicted probabilities match actual frequencies.

**Well-calibrated model**: If it predicts 70% probability for 100 cases, ~70 should be positive.

**Importance**:
1. **Decision making**: Reliable probabilities for thresholds
2. **Risk assessment**: Accurate probability estimates
3. **Model combining**: Calibrated probabilities can be compared

**Checking calibration**:
- Calibration plots
- Brier score
- Expected Calibration Error (ECE)

**Improving calibration**:
- Platt scaling
- Isotonic regression
- Beta calibration

**Q15: How do metrics guide model improvement?**

**A:** Metrics provide diagnostic information:

1. **High bias (underfitting)**:
   - Low training and test scores
   - Action: Increase model complexity

2. **High variance (overfitting)**:
   - High training, low test scores
   - Action: Regularization, more data

3. **Class imbalance issues**:
   - High accuracy, low recall
   - Action: Resampling, class weights

4. **Threshold issues**:
   - Suboptimal precision-recall balance
   - Action: Adjust decision threshold

5. **Calibration issues**:
   - Poor probability estimates
   - Action: Calibration techniques

### Real-world Scenarios

**Q16: A model has 99% accuracy but performs poorly. What could be wrong?**

**A:** Classic imbalanced data problem:

**Diagnosis**:
1. Check class distribution (likely 99% one class)
2. Calculate per-class metrics
3. Look at confusion matrix

**Example**:
- 99% negative, 1% positive class
- Predicting all negative: 99% accuracy, 0% recall
- Useless for finding positive cases

**Solutions**:
1. Use appropriate metrics (F1, AUC-PR, MCC)
2. Resampling techniques
3. Class weights
4. Anomaly detection approach
5. Cost-sensitive learning

**Q17: How do you evaluate a recommendation system?**

**A:** Multiple metrics needed:

**Offline Metrics**:
1. **Ranking metrics**: NDCG, MAP, MRR
2. **Classification metrics**: Precision@K, Recall@K
3. **Rating prediction**: RMSE, MAE
4. **Coverage**: Catalog percentage recommended
5. **Diversity**: Intra-list diversity

**Online Metrics**:
1. **CTR**: Click-through rate
2. **Conversion rate**: Actual purchases
3. **User engagement**: Time spent, return rate
4. **Revenue**: Direct business impact

**Challenges**:
- Offline-online metric gap
- Position bias
- Feedback loops

**Q18: How do you handle metrics when ground truth is expensive?**

**A:** Several strategies:

1. **Proxy metrics**:
   - User engagement as proxy for satisfaction
   - Correlate with true metric on small sample

2. **Active learning**:
   - Selectively label most informative samples
   - Focus on uncertain predictions

3. **Weak supervision**:
   - Heuristics for approximate labels
   - Multiple noisy labelers

4. **Semi-supervised evaluation**:
   - Use unlabeled data patterns
   - Consistency checks

5. **Human-in-the-loop**:
   - Expert evaluation on sample
   - Crowd-sourcing for scale

### Interview Problem Solving

**Q19: Design a metric for a loan default prediction system.**

**A:** Consider multiple stakeholders:

**Business Requirements**:
- Minimize defaults (losses)
- Maintain loan volume (revenue)
- Fair lending practices

**Custom Metric Design**:
```python
def loan_profit_metric(y_true, y_pred, loan_amounts,
                      interest_rate=0.05, default_loss=0.8):
    """
    Calculate expected profit from loan decisions
    """
    # True positives: Correctly approved good loans
    tp_mask = (y_true == 0) & (y_pred == 0)
    tp_profit = np.sum(loan_amounts[tp_mask] * interest_rate)
    
    # False negatives: Missed good loans
    fn_mask = (y_true == 0) & (y_pred == 1)
    fn_loss = np.sum(loan_amounts[fn_mask] * interest_rate)
    
    # False positives: Approved bad loans
    fp_mask = (y_true == 1) & (y_pred == 0)
    fp_loss = np.sum(loan_amounts[fp_mask] * default_loss)
    
    total_profit = tp_profit - fn_loss - fp_loss
    return total_profit
```

**Additional Metrics**:
- Approval rate by demographic (fairness)
- Portfolio risk metrics
- Customer satisfaction scores

**Q20: You need to compare models trained on different datasets. How?**

**A:** Challenging but possible:

**Approaches**:

1. **Normalized metrics**:
   - Use percentile ranks
   - Normalize by baseline performance
   - Relative improvement over naive model

2. **Cross-dataset evaluation**:
   - Test each model on all datasets
   - Create combined test set
   - Domain adaptation techniques

3. **Meta-metrics**:
   - Model complexity vs performance
   - Training time vs accuracy
   - Robustness measures

4. **Statistical testing**:
   - Paired t-tests across splits
   - Wilcoxon signed-rank test
   - Confidence intervals

**Example framework**:
```python
def compare_across_datasets(models, datasets):
    results = {}
    for model_name, model in models.items():
        for dataset_name, (X, y) in datasets.items():
            # Evaluate model on dataset
            score = cross_val_score(model, X, y)
            # Normalize by baseline
            baseline_score = dummy_classifier_score(X, y)
            relative_improvement = (score - baseline_score) / baseline_score
            results[(model_name, dataset_name)] = relative_improvement
    return results
```

---

## Practice Problems

1. Implement custom F-beta score with numpy
2. Create a profit-based metric for your domain
3. Build a calibration diagnostic tool
4. Design metrics for fairness evaluation
5. Implement confidence intervals for all metrics
6. Create automated metric selection based on data characteristics

## Key Takeaways

1. **No universal best metric** - choose based on problem context
2. **Multiple metrics** provide complete picture
3. **Business alignment** is crucial
4. **Class imbalance** requires special attention
5. **Threshold selection** affects metric values
6. **Calibration** matters for probability interpretation
7. **Custom metrics** often needed for real problems
8. **Monitor metrics** in production continuously