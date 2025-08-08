# Day 21: Classification Practice - Comprehensive Guide

## Table of Contents
1. [Introduction](#1-introduction)
2. [Classification Problem Framework](#2-classification-problem-framework)
3. [Dataset Preparation and EDA](#3-dataset-preparation-and-eda)
4. [Feature Engineering for Classification](#4-feature-engineering-for-classification)
5. [Model Selection Strategy](#5-model-selection-strategy)
6. [Implementation: Binary Classification](#6-implementation-binary-classification)
7. [Implementation: Multi-class Classification](#7-implementation-multi-class-classification)
8. [Advanced Techniques](#8-advanced-techniques)
9. [Real-world Case Studies](#9-real-world-case-studies)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

Today we'll consolidate our knowledge of classification algorithms through practical implementation. We'll work through complete classification pipelines, compare different algorithms, and learn best practices for real-world applications.

### What We'll Cover

1. **End-to-end classification workflow**
2. **Comparative analysis of classifiers**
3. **Handling real-world challenges**
4. **Performance optimization techniques**
5. **Production-ready code patterns**

### Key Learning Objectives

- Master the complete classification pipeline
- Understand when to use which classifier
- Learn to handle imbalanced datasets
- Implement ensemble methods effectively
- Debug and optimize classification models

## 2. Classification Problem Framework

### 2.1 Problem Types

```python
# Classification taxonomy
classification_types = {
    'Binary': {
        'examples': ['Spam detection', 'Fraud detection', 'Churn prediction'],
        'metrics': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'],
        'challenges': ['Class imbalance', 'Cost-sensitive learning']
    },
    'Multi-class': {
        'examples': ['Image classification', 'Document categorization'],
        'metrics': ['Accuracy', 'Macro/Micro F1', 'Confusion matrix'],
        'challenges': ['One-vs-Rest vs One-vs-One', 'Class imbalance']
    },
    'Multi-label': {
        'examples': ['Tag prediction', 'Gene function prediction'],
        'metrics': ['Hamming loss', 'Subset accuracy', 'F1 per label'],
        'challenges': ['Label correlation', 'Threshold selection']
    },
    'Imbalanced': {
        'examples': ['Rare disease detection', 'Anomaly detection'],
        'metrics': ['Precision-Recall curve', 'F1', 'Matthews correlation'],
        'challenges': ['Sampling strategies', 'Cost-sensitive learning']
    }
}
```

### 2.2 Algorithm Selection Guide

```python
def select_classifier(data_characteristics):
    """Guide for selecting appropriate classifier"""
    
    selection_rules = {
        'linear_separable': {
            'small_data': ['Logistic Regression', 'Linear SVM'],
            'large_data': ['SGD Classifier', 'Passive Aggressive']
        },
        'non_linear': {
            'small_data': ['SVM with RBF', 'Random Forest', 'XGBoost'],
            'large_data': ['Neural Networks', 'Gradient Boosting']
        },
        'interpretability_required': {
            'any_size': ['Decision Tree', 'Logistic Regression', 'Naive Bayes']
        },
        'high_dimensional': {
            'text_data': ['Naive Bayes', 'Linear SVM', 'Logistic Regression'],
            'other': ['Random Forest', 'Regularized Linear Models']
        },
        'realtime_prediction': {
            'any_size': ['Naive Bayes', 'Logistic Regression', 'Linear SVM']
        }
    }
    
    return selection_rules
```

## 3. Dataset Preparation and EDA

### 3.1 Comprehensive EDA Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ClassificationEDA:
    """Comprehensive EDA for classification problems"""
    
    def __init__(self, df, target_column):
        self.df = df
        self.target = target_column
        self.numerical_features = []
        self.categorical_features = []
        self.report = {}
        
    def analyze(self):
        """Perform complete EDA"""
        self._identify_feature_types()
        self._check_missing_values()
        self._analyze_target_distribution()
        self._analyze_feature_distributions()
        self._check_correlations()
        self._detect_outliers()
        self._generate_report()
        return self.report
    
    def _identify_feature_types(self):
        """Identify numerical and categorical features"""
        for col in self.df.columns:
            if col == self.target:
                continue
            if self.df[col].dtype in ['int64', 'float64']:
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
                
        self.report['feature_types'] = {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features
        }
    
    def _check_missing_values(self):
        """Analyze missing values"""
        missing = self.df.isnull().sum()
        missing_pct = 100 * missing / len(self.df)
        
        self.report['missing_values'] = pd.DataFrame({
            'count': missing,
            'percentage': missing_pct
        }).sort_values('percentage', ascending=False)
    
    def _analyze_target_distribution(self):
        """Analyze target variable distribution"""
        target_dist = self.df[self.target].value_counts()
        target_pct = 100 * target_dist / len(self.df)
        
        self.report['target_distribution'] = pd.DataFrame({
            'count': target_dist,
            'percentage': target_pct
        })
        
        # Check for imbalance
        min_class_pct = target_pct.min()
        self.report['is_imbalanced'] = min_class_pct < 20
        
        # Visualize
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        target_dist.plot(kind='bar')
        plt.title('Target Distribution (Counts)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        target_pct.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Target Distribution (Percentage)')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()
    
    def _analyze_feature_distributions(self):
        """Analyze distributions of features"""
        # Numerical features
        if self.numerical_features:
            n_cols = min(3, len(self.numerical_features))
            n_rows = (len(self.numerical_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(self.numerical_features):
                if idx < len(axes):
                    self.df[col].hist(ax=axes[idx], bins=30)
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
        
        # Categorical features
        if self.categorical_features:
            for col in self.categorical_features[:5]:  # Limit to first 5
                plt.figure(figsize=(10, 6))
                value_counts = self.df[col].value_counts()[:10]  # Top 10
                value_counts.plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    def _check_correlations(self):
        """Check feature correlations"""
        if len(self.numerical_features) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[self.numerical_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.show()
            
            # Find highly correlated features
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            self.report['high_correlations'] = high_corr
    
    def _detect_outliers(self):
        """Detect outliers using IQR method"""
        outlier_report = {}
        
        for col in self.numerical_features:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | 
                              (self.df[col] > upper_bound)]
            outlier_pct = 100 * len(outliers) / len(self.df)
            
            outlier_report[col] = {
                'count': len(outliers),
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        self.report['outliers'] = outlier_report
    
    def _generate_report(self):
        """Generate summary report"""
        print("=== EDA Summary Report ===")
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Target variable: {self.target}")
        print(f"Number of classes: {self.df[self.target].nunique()}")
        print(f"Imbalanced dataset: {self.report['is_imbalanced']}")
        
        print(f"\nFeature types:")
        print(f"- Numerical: {len(self.numerical_features)}")
        print(f"- Categorical: {len(self.categorical_features)}")
        
        print("\nMissing values:")
        missing_features = self.report['missing_values'][
            self.report['missing_values']['count'] > 0
        ]
        if len(missing_features) > 0:
            print(missing_features.head())
        else:
            print("No missing values found")
        
        if self.report.get('high_correlations'):
            print(f"\nHighly correlated features: {len(self.report['high_correlations'])}")

# Example usage
def demonstrate_eda():
    # Generate synthetic dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=15, n_redundant=5,
                              n_classes=3, class_sep=1.0,
                              flip_y=0.1, random_state=42)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['category_B'] = np.random.choice(['X', 'Y', 'Z'], size=len(df))
    
    # Perform EDA
    eda = ClassificationEDA(df, 'target')
    report = eda.analyze()
    
    return df, report
```

## 4. Feature Engineering for Classification

### 4.1 Comprehensive Feature Engineering Pipeline

```python
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion

class FeatureEngineer:
    """Advanced feature engineering for classification"""
    
    def __init__(self):
        self.transformers = {}
        self.feature_names = []
        
    def create_features(self, X, y=None):
        """Create various types of features"""
        features = []
        
        # 1. Polynomial features for numerical columns
        if hasattr(X, 'select_dtypes'):
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                poly_features = self._create_polynomial_features(
                    X[numerical_cols], degree=2
                )
                features.append(poly_features)
        
        # 2. Interaction features
        interaction_features = self._create_interaction_features(X)
        if interaction_features is not None:
            features.append(interaction_features)
        
        # 3. Statistical features
        stat_features = self._create_statistical_features(X)
        if stat_features is not None:
            features.append(stat_features)
        
        # 4. Domain-specific features (example)
        domain_features = self._create_domain_features(X)
        if domain_features is not None:
            features.append(domain_features)
        
        # Combine all features
        if features:
            return np.hstack(features)
        return X
    
    def _create_polynomial_features(self, X, degree=2):
        """Create polynomial features"""
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(X)
        
        # Store feature names
        if hasattr(X, 'columns'):
            feature_names = poly.get_feature_names_out(X.columns)
            self.feature_names.extend(feature_names)
        
        return poly_features
    
    def _create_interaction_features(self, X):
        """Create interaction features between columns"""
        if not hasattr(X, 'columns') or len(X.columns) < 2:
            return None
        
        interactions = []
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Create pairwise interactions
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                # Multiplication
                interaction = X[col1] * X[col2]
                interactions.append(interaction.values.reshape(-1, 1))
                self.feature_names.append(f'{col1}_x_{col2}')
                
                # Division (with small epsilon to avoid division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = X[col1] / (X[col2] + 1e-8)
                    ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
                interactions.append(ratio.values.reshape(-1, 1))
                self.feature_names.append(f'{col1}_div_{col2}')
        
        if interactions:
            return np.hstack(interactions)
        return None
    
    def _create_statistical_features(self, X):
        """Create statistical aggregation features"""
        if not hasattr(X, 'select_dtypes'):
            return None
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return None
        
        stat_features = []
        
        # Row-wise statistics
        X_num = X[numerical_cols]
        
        # Mean
        row_mean = X_num.mean(axis=1).values.reshape(-1, 1)
        stat_features.append(row_mean)
        self.feature_names.append('row_mean')
        
        # Standard deviation
        row_std = X_num.std(axis=1).values.reshape(-1, 1)
        stat_features.append(row_std)
        self.feature_names.append('row_std')
        
        # Min and Max
        row_min = X_num.min(axis=1).values.reshape(-1, 1)
        row_max = X_num.max(axis=1).values.reshape(-1, 1)
        stat_features.extend([row_min, row_max])
        self.feature_names.extend(['row_min', 'row_max'])
        
        # Skewness indicator
        row_skew = X_num.apply(lambda x: x.skew(), axis=1).values.reshape(-1, 1)
        stat_features.append(row_skew)
        self.feature_names.append('row_skew')
        
        return np.hstack(stat_features)
    
    def _create_domain_features(self, X):
        """Create domain-specific features (example for e-commerce)"""
        domain_features = []
        
        # Example: if dealing with user behavior data
        if hasattr(X, 'columns'):
            # Price-related features
            if 'price' in X.columns and 'quantity' in X.columns:
                total_value = X['price'] * X['quantity']
                domain_features.append(total_value.values.reshape(-1, 1))
                self.feature_names.append('total_value')
            
            # Time-based features
            if 'timestamp' in X.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
                    timestamps = pd.to_datetime(X['timestamp'])
                else:
                    timestamps = X['timestamp']
                
                # Extract time features
                hour = timestamps.dt.hour.values.reshape(-1, 1)
                day_of_week = timestamps.dt.dayofweek.values.reshape(-1, 1)
                is_weekend = (day_of_week >= 5).astype(int).reshape(-1, 1)
                
                domain_features.extend([hour, day_of_week, is_weekend])
                self.feature_names.extend(['hour', 'day_of_week', 'is_weekend'])
        
        if domain_features:
            return np.hstack(domain_features)
        return None

# Feature selection utilities
class FeatureSelector:
    """Advanced feature selection techniques"""
    
    def __init__(self, method='mutual_info', k=20):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
        
    def fit_transform(self, X, y):
        """Select best features"""
        if self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=self.k)
        elif self.method == 'chi2':
            # Ensure non-negative values for chi2
            X_positive = X - X.min() + 1e-8
            self.selector = SelectKBest(chi2, k=self.k)
            return self.selector.fit_transform(X_positive, y)
        elif self.method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            self.selector = VarianceThreshold(threshold=0.01)
            return self.selector.fit_transform(X)
        
        return self.selector.fit_transform(X, y)
    
    def get_feature_scores(self, feature_names=None):
        """Get feature importance scores"""
        if hasattr(self.selector, 'scores_'):
            scores = self.selector.scores_
            if feature_names is not None:
                return pd.DataFrame({
                    'feature': feature_names,
                    'score': scores
                }).sort_values('score', ascending=False)
            return scores
        return None
```

## 5. Model Selection Strategy

### 5.1 Comprehensive Model Comparison Framework

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time

class ModelComparison:
    """Compare multiple classification models"""
    
    def __init__(self, models=None):
        if models is None:
            self.models = self._get_default_models()
        else:
            self.models = models
        self.results = {}
        
    def _get_default_models(self):
        """Get default set of classifiers"""
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
    
    def compare(self, X_train, y_train, X_test, y_test, cv=5):
        """Compare all models"""
        results = []
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Prediction time
            start_time = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # ROC AUC for binary classification
            if len(np.unique(y_train)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                      scoring='accuracy')
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Train Time': train_time,
                'Pred Time': pred_time
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def plot_comparison(self):
        """Visualize model comparison"""
        if self.results.empty:
            print("No results to plot. Run compare() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        ax = axes[0, 0]
        self.results.set_index('Model')['Accuracy'].plot(kind='bar', ax=ax)
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_xticklabels(self.results['Model'], rotation=45)
        
        # F1-Score comparison
        ax = axes[0, 1]
        self.results.set_index('Model')['F1-Score'].plot(kind='bar', ax=ax, color='orange')
        ax.set_title('Model F1-Score Comparison')
        ax.set_ylabel('F1-Score')
        ax.set_xticklabels(self.results['Model'], rotation=45)
        
        # Cross-validation scores
        ax = axes[1, 0]
        self.results.set_index('Model')['CV Mean'].plot(kind='bar', ax=ax, 
                                                       color='green', yerr=self.results['CV Std'])
        ax.set_title('Cross-Validation Scores')
        ax.set_ylabel('CV Accuracy')
        ax.set_xticklabels(self.results['Model'], rotation=45)
        
        # Training time
        ax = axes[1, 1]
        self.results.set_index('Model')['Train Time'].plot(kind='bar', ax=ax, color='red')
        ax.set_title('Training Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticklabels(self.results['Model'], rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric='F1-Score'):
        """Get best performing model"""
        if self.results.empty:
            print("No results available. Run compare() first.")
            return None
        
        best_idx = self.results[metric].idxmax()
        best_model_name = self.results.loc[best_idx, 'Model']
        best_score = self.results.loc[best_idx, metric]
        
        print(f"\nBest model based on {metric}: {best_model_name}")
        print(f"{metric}: {best_score:.4f}")
        
        return self.models[best_model_name], best_model_name
```

## 6. Implementation: Binary Classification

### 6.1 Complete Binary Classification Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class BinaryClassificationPipeline:
    """Complete pipeline for binary classification"""
    
    def __init__(self, handle_imbalance='none'):
        self.handle_imbalance = handle_imbalance
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model = None
        self.pipeline = None
        
    def build_pipeline(self, model):
        """Build complete preprocessing and modeling pipeline"""
        steps = []
        
        # Scaling
        steps.append(('scaler', self.scaler))
        
        # Handle imbalanced data
        if self.handle_imbalance == 'smote':
            steps.append(('sampler', SMOTE(random_state=42)))
        elif self.handle_imbalance == 'undersample':
            steps.append(('sampler', RandomUnderSampler(random_state=42)))
        
        # Model
        steps.append(('classifier', model))
        
        # Create pipeline
        if self.handle_imbalance != 'none':
            self.pipeline = ImbPipeline(steps)
        else:
            self.pipeline = Pipeline(steps)
        
        return self.pipeline
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train model and provide comprehensive evaluation"""
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluation
        print("=== Model Evaluation ===")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Visualizations
        self._plot_confusion_matrix(cm, ['Class 0', 'Class 1'])
        self._plot_roc_curve(y_test, y_proba)
        self._plot_precision_recall_curve(y_test, y_proba)
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def _plot_roc_curve(self, y_true, y_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def _plot_precision_recall_curve(self, y_true, y_proba):
        """Plot precision-recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Avg Precision = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()
    
    def optimize_threshold(self, X_val, y_val, metric='f1'):
        """Find optimal classification threshold"""
        y_proba = self.pipeline.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        print(f"\nOptimal threshold for {metric}: {optimal_threshold:.3f}")
        print(f"Score at optimal threshold: {optimal_score:.4f}")
        
        # Plot threshold vs metric
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores)
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal = {optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Threshold')
        plt.legend()
        plt.show()
        
        return optimal_threshold

# Example usage
def binary_classification_example():
    # Generate imbalanced binary dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, weights=[0.9, 0.1],
                              flip_y=0.05, random_state=42)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, 
                                                        stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                     stratify=y_temp, random_state=42)
    
    # Compare different approaches
    approaches = {
        'No balancing': BinaryClassificationPipeline(handle_imbalance='none'),
        'SMOTE': BinaryClassificationPipeline(handle_imbalance='smote'),
        'Undersampling': BinaryClassificationPipeline(handle_imbalance='undersample')
    }
    
    results = {}
    for name, pipeline in approaches.items():
        print(f"\n{'='*50}")
        print(f"Approach: {name}")
        print('='*50)
        
        # Build and train
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        pipeline.build_pipeline(model)
        eval_results = pipeline.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # Optimize threshold
        optimal_threshold = pipeline.optimize_threshold(X_val, y_val, metric='f1')
        
        results[name] = {
            'pipeline': pipeline,
            'evaluation': eval_results,
            'optimal_threshold': optimal_threshold
        }
    
    return results
```

## 7. Implementation: Multi-class Classification

### 7.1 Complete Multi-class Classification Pipeline

```python
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

class MultiClassClassificationPipeline:
    """Complete pipeline for multi-class classification"""
    
    def __init__(self, strategy='ovr'):
        self.strategy = strategy  # 'ovr' or 'ovo'
        self.scaler = StandardScaler()
        self.label_binarizer = LabelBinarizer()
        self.model = None
        
    def build_model(self, base_estimator):
        """Build multi-class classifier"""
        if self.strategy == 'ovr':
            self.model = OneVsRestClassifier(base_estimator)
        elif self.strategy == 'ovo':
            self.model = OneVsOneClassifier(base_estimator)
        else:
            self.model = base_estimator
        
        return self.model
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train and evaluate multi-class model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        # Evaluation
        print("=== Multi-class Model Evaluation ===")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Per-class metrics
        self._evaluate_per_class(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        # Multi-class ROC
        self._plot_multiclass_roc(X_test_scaled, y_test)
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': cm
        }
    
    def _evaluate_per_class(self, y_true, y_pred):
        """Evaluate metrics per class"""
        classes = np.unique(y_true)
        
        print("\nPer-class metrics:")
        print("-" * 50)
        
        for class_label in classes:
            # Binary indicators for current class
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            precision = precision_score(y_true_binary, y_pred_binary)
            recall = recall_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary)
            
            print(f"Class {class_label}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix for multi-class"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Multi-class Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def _plot_multiclass_roc(self, X_test, y_test):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Get probabilities
        y_score = self.model.predict_proba(X_test)
        
        # Compute ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.show()

# Multi-label classification
class MultiLabelClassificationPipeline:
    """Pipeline for multi-label classification"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def train_binary_relevance(self, X_train, y_train, base_estimator):
        """Train using binary relevance approach"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        n_labels = y_train.shape[1]
        
        for label_idx in range(n_labels):
            print(f"Training classifier for label {label_idx}...")
            model = base_estimator()
            model.fit(X_train_scaled, y_train[:, label_idx])
            self.models[label_idx] = model
    
    def predict(self, X_test):
        """Predict multiple labels"""
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = []
        for label_idx, model in self.models.items():
            pred = model.predict(X_test_scaled)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def evaluate(self, X_test, y_test):
        """Evaluate multi-label predictions"""
        y_pred = self.predict(X_test)
        
        # Hamming loss
        from sklearn.metrics import hamming_loss
        h_loss = hamming_loss(y_test, y_pred)
        
        # Subset accuracy
        subset_acc = np.mean(np.all(y_pred == y_test, axis=1))
        
        # Per-label metrics
        n_labels = y_test.shape[1]
        label_metrics = []
        
        for label_idx in range(n_labels):
            precision = precision_score(y_test[:, label_idx], y_pred[:, label_idx])
            recall = recall_score(y_test[:, label_idx], y_pred[:, label_idx])
            f1 = f1_score(y_test[:, label_idx], y_pred[:, label_idx])
            
            label_metrics.append({
                'label': label_idx,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        print(f"\nHamming Loss: {h_loss:.4f}")
        print(f"Subset Accuracy: {subset_acc:.4f}")
        print("\nPer-label metrics:")
        print(pd.DataFrame(label_metrics))
        
        return {
            'predictions': y_pred,
            'hamming_loss': h_loss,
            'subset_accuracy': subset_acc,
            'label_metrics': label_metrics
        }
```

## 8. Advanced Techniques

### 8.1 Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

class AdvancedEnsemble:
    """Advanced ensemble techniques for classification"""
    
    def create_voting_ensemble(self, X_train, y_train, X_test, y_test):
        """Create and evaluate voting ensemble"""
        # Define base models
        base_models = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('svm', SVC(probability=True)),
            ('gb', GradientBoostingClassifier(n_estimators=100))
        ]
        
        # Soft voting ensemble
        soft_voting = VotingClassifier(estimators=base_models, voting='soft')
        soft_voting.fit(X_train, y_train)
        soft_pred = soft_voting.predict(X_test)
        soft_acc = accuracy_score(y_test, soft_pred)
        
        # Hard voting ensemble
        hard_voting = VotingClassifier(estimators=base_models, voting='hard')
        hard_voting.fit(X_train, y_train)
        hard_pred = hard_voting.predict(X_test)
        hard_acc = accuracy_score(y_test, hard_pred)
        
        print(f"Soft Voting Accuracy: {soft_acc:.4f}")
        print(f"Hard Voting Accuracy: {hard_acc:.4f}")
        
        # Compare with individual models
        print("\nIndividual Model Performance:")
        for name, model in base_models:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            print(f"{name}: {acc:.4f}")
        
        return soft_voting, hard_voting
    
    def create_stacking_ensemble(self, X_train, y_train, X_test, y_test):
        """Create and evaluate stacking ensemble"""
        # Base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('svm', SVC(probability=True, kernel='rbf')),
            ('gb', GradientBoostingClassifier(n_estimators=100))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000)
        
        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5  # Use cross-validation to train meta-learner
        )
        
        # Train and evaluate
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_pred)
        
        print(f"\nStacking Ensemble Accuracy: {stacking_acc:.4f}")
        
        # Feature importances from meta-learner
        if hasattr(meta_learner, 'coef_'):
            meta_weights = meta_learner.coef_[0]
            print("\nMeta-learner weights:")
            for (name, _), weight in zip(base_models, meta_weights):
                print(f"{name}: {weight:.4f}")
        
        return stacking
    
    def create_blending_ensemble(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create ensemble using blending"""
        # Split training data for blending
        blend_features_train = []
        blend_features_val = []
        blend_features_test = []
        
        # Train base models and create blend features
        base_models = {
            'rf': RandomForestClassifier(n_estimators=100),
            'gb': GradientBoostingClassifier(n_estimators=100),
            'svm': SVC(probability=True)
        }
        
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Get predictions
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            
            blend_features_train.append(train_pred)
            blend_features_val.append(val_pred)
            blend_features_test.append(test_pred)
        
        # Create blend datasets
        X_blend_train = np.column_stack(blend_features_train)
        X_blend_val = np.column_stack(blend_features_val)
        X_blend_test = np.column_stack(blend_features_test)
        
        # Train meta-model
        meta_model = LogisticRegression()
        meta_model.fit(X_blend_val, y_val)
        
        # Final predictions
        final_pred = meta_model.predict(X_blend_test)
        final_acc = accuracy_score(y_test, final_pred)
        
        print(f"\nBlending Ensemble Accuracy: {final_acc:.4f}")
        
        return meta_model, base_models

### 8.2 Calibration Techniques

class CalibrationTechniques:
    """Probability calibration for better predictions"""
    
    def calibrate_probabilities(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """Apply different calibration methods"""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression
        
        # Train base model
        model.fit(X_train, y_train)
        
        # Get uncalibrated probabilities
        prob_uncalibrated = model.predict_proba(X_test)[:, 1]
        
        # Platt scaling (sigmoid)
        platt_calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        platt_calibrated.fit(X_val, y_val)
        prob_platt = platt_calibrated.predict_proba(X_test)[:, 1]
        
        # Isotonic regression
        isotonic_calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        isotonic_calibrated.fit(X_val, y_val)
        prob_isotonic = isotonic_calibrated.predict_proba(X_test)[:, 1]
        
        # Compare calibration
        self._plot_calibration_curve(y_test, prob_uncalibrated, prob_platt, prob_isotonic)
        
        # Evaluate
        from sklearn.metrics import brier_score_loss, log_loss
        
        print("Calibration Results:")
        print(f"Uncalibrated - Brier Score: {brier_score_loss(y_test, prob_uncalibrated):.4f}")
        print(f"Platt - Brier Score: {brier_score_loss(y_test, prob_platt):.4f}")
        print(f"Isotonic - Brier Score: {brier_score_loss(y_test, prob_isotonic):.4f}")
        
        return platt_calibrated, isotonic_calibrated
    
    def _plot_calibration_curve(self, y_true, prob_uncalibrated, prob_platt, prob_isotonic):
        """Plot calibration curves"""
        from sklearn.calibration import calibration_curve
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration plot
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(y_true, prob_uncalibrated, n_bins=10)
        fraction_pos_platt, mean_pred_platt = calibration_curve(y_true, prob_platt, n_bins=10)
        fraction_pos_iso, mean_pred_iso = calibration_curve(y_true, prob_isotonic, n_bins=10)
        
        ax1.plot(mean_pred_uncal, fraction_pos_uncal, 's-', label='Uncalibrated')
        ax1.plot(mean_pred_platt, fraction_pos_platt, 's-', label='Platt')
        ax1.plot(mean_pred_iso, fraction_pos_iso, 's-', label='Isotonic')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curves')
        ax1.legend()
        
        # Histogram
        ax2.hist(prob_uncalibrated, bins=50, alpha=0.5, label='Uncalibrated')
        ax2.hist(prob_platt, bins=50, alpha=0.5, label='Platt')
        ax2.hist(prob_isotonic, bins=50, alpha=0.5, label='Isotonic')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
```

## 9. Real-world Case Studies

### 9.1 Customer Churn Prediction

```python
def customer_churn_case_study():
    """Complete customer churn prediction pipeline"""
    
    # Simulate customer data
    np.random.seed(42)
    n_customers = 5000
    
    # Generate features
    data = {
        'tenure': np.random.randint(1, 72, n_customers),
        'monthly_charges': np.random.uniform(20, 120, n_customers),
        'total_charges': np.random.uniform(100, 8000, n_customers),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Credit card'], n_customers),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
        'online_security': np.random.choice(['Yes', 'No', 'No service'], n_customers),
        'tech_support': np.random.choice(['Yes', 'No', 'No service'], n_customers),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No service'], n_customers),
        'num_services': np.random.randint(1, 9, n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on realistic patterns
    churn_probability = (
        0.5 * (df['contract_type'] == 'Month-to-month').astype(int) +
        0.3 * (df['payment_method'] == 'Electronic check').astype(int) +
        0.2 * (df['tenure'] < 12).astype(int) +
        0.1 * (df['monthly_charges'] > 80).astype(int) -
        0.3 * (df['contract_type'] == 'Two year').astype(int)
    )
    
    df['churn'] = (churn_probability + np.random.normal(0, 0.3, n_customers) > 0.7).astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Feature engineering
    # 1. Create derived features
    df['charges_per_tenure'] = df['total_charges'] / (df['tenure'] + 1)
    df['high_value_customer'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
    df['new_customer'] = (df['tenure'] <= 6).astype(int)
    
    # 2. Encode categorical variables
    categorical_columns = ['contract_type', 'payment_method', 'internet_service', 
                          'online_security', 'tech_support', 'streaming_tv']
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Prepare for modeling
    X = df_encoded.drop('churn', axis=1)
    y = df_encoded['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Model comparison with business metrics
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': GradientBoostingClassifier(n_estimators=100)
    }
    
    results = []
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate business metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Cost assumptions
        cost_false_positive = 50  # Cost of unnecessary retention effort
        cost_false_negative = 500  # Cost of losing a customer
        
        total_cost = fp * cost_false_positive + fn * cost_false_negative
        
        # Lift at 20% (top 20% most likely to churn)
        n_top20 = int(0.2 * len(y_test))
        top20_indices = np.argsort(y_proba)[-n_top20:]
        lift_20 = y_test.iloc[top20_indices].mean() / y_test.mean()
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'Total Cost': total_cost,
            'Lift@20%': lift_20
        })
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)
    
    # Feature importance analysis
    best_model = models['Random Forest']
    best_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Actionable insights
    print("\n=== Actionable Insights ===")
    print("1. High-risk segments:")
    print("   - Month-to-month contracts with electronic check payments")
    print("   - New customers (tenure < 6 months) with high monthly charges")
    print("\n2. Retention strategies:")
    print("   - Offer contract upgrades to month-to-month customers")
    print("   - Provide payment method incentives for electronic check users")
    print("   - Enhanced onboarding for new customers")
    
    return df, results_df, feature_importance

### 9.2 Fraud Detection

def fraud_detection_case_study():
    """Credit card fraud detection with extreme imbalance"""
    
    # Simulate transaction data
    np.random.seed(42)
    n_transactions = 10000
    fraud_rate = 0.002  # 0.2% fraud rate
    
    # Normal transactions
    n_normal = int(n_transactions * (1 - fraud_rate))
    normal_data = {
        'amount': np.random.lognormal(3, 1.5, n_normal),
        'days_since_last': np.random.exponential(2, n_normal),
        'merchant_risk_score': np.random.beta(2, 5, n_normal),
        'user_history_score': np.random.beta(5, 2, n_normal),
        'time_of_day': np.random.uniform(0, 24, n_normal),
        'location_risk': np.random.beta(2, 8, n_normal)
    }
    
    # Fraudulent transactions
    n_fraud = n_transactions - n_normal
    fraud_data = {
        'amount': np.random.lognormal(4, 2, n_fraud),  # Higher amounts
        'days_since_last': np.random.exponential(0.5, n_fraud),  # More frequent
        'merchant_risk_score': np.random.beta(5, 2, n_fraud),  # Riskier merchants
        'user_history_score': np.random.beta(2, 5, n_fraud),  # Unusual for user
        'time_of_day': np.concatenate([
            np.random.uniform(0, 6, n_fraud//2),  # Late night
            np.random.uniform(22, 24, n_fraud//2)
        ]),
        'location_risk': np.random.beta(5, 3, n_fraud)  # Riskier locations
    }
    
    # Combine data
    X_normal = pd.DataFrame(normal_data)
    X_fraud = pd.DataFrame(fraud_data)
    
    X = pd.concat([X_normal, X_fraud], ignore_index=True)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y[shuffle_idx]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Feature engineering
    X['amount_zscore'] = (X['amount'] - X['amount'].mean()) / X['amount'].std()
    X['high_risk_time'] = ((X['time_of_day'] < 6) | (X['time_of_day'] > 22)).astype(int)
    X['risk_composite'] = X['merchant_risk_score'] * X['location_risk']
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    
    # Different approaches for extreme imbalance
    approaches = {
        'Baseline': RandomForestClassifier(n_estimators=100, random_state=42),
        'Class Weight': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SMOTE': Pipeline([
            ('sampler', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Isolation Forest': None  # Will implement separately
    }
    
    results = []
    
    for name, approach in approaches.items():
        if name == 'Isolation Forest':
            from sklearn.ensemble import IsolationForest
            # Unsupervised approach
            iso_forest = IsolationForest(contamination=fraud_rate, random_state=42)
            y_pred = iso_forest.fit_predict(X_test)
            y_pred = (y_pred == -1).astype(int)  # -1 indicates anomaly
        else:
            approach.fit(X_train, y_train)
            y_pred = approach.predict(X_test)
        
        # Metrics for imbalanced data
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Business metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Cost analysis
        avg_fraud_amount = X_test[y_test == 1]['amount'].mean()
        avg_normal_amount = X_test[y_test == 0]['amount'].mean()
        
        fraud_prevented = tp * avg_fraud_amount
        false_positive_cost = fp * avg_normal_amount * 0.02  # 2% of transaction as investigation cost
        fraud_missed = fn * avg_fraud_amount
        
        net_benefit = fraud_prevented - false_positive_cost - fraud_missed
        
        results.append({
            'Approach': name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Fraud Caught': f"{tp}/{tp+fn}",
            'False Alarms': fp,
            'Net Benefit': f"${net_benefit:,.2f}"
        })
    
    results_df = pd.DataFrame(results)
    print("\nFraud Detection Results:")
    print(results_df)
    
    # Precision-Recall analysis
    best_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    best_model.fit(X_train, y_train)
    
    y_scores = best_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold for business objective
    thresholds = np.linspace(0, 1, 100)
    net_benefits = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        
        fraud_prevented = tp * avg_fraud_amount
        false_positive_cost = fp * avg_normal_amount * 0.02
        fraud_missed = fn * avg_fraud_amount
        
        net_benefit = fraud_prevented - false_positive_cost - fraud_missed
        net_benefits.append(net_benefit)
    
    optimal_threshold = thresholds[np.argmax(net_benefits)]
    
    print(f"\nOptimal threshold for maximum net benefit: {optimal_threshold:.3f}")
    print(f"Maximum net benefit: ${max(net_benefits):,.2f}")
    
    # Plot net benefit vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal = {optimal_threshold:.3f}')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Net Benefit ($)')
    plt.title('Net Benefit vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return X, y, results_df
```

## 10. Interview Questions

### Q1: What factors should you consider when choosing a classification algorithm?
**Answer**: Key factors include:
1. **Data characteristics**: Size, dimensionality, linearity, noise level
2. **Interpretability requirements**: Simple models vs black boxes
3. **Training time constraints**: Real-time vs batch processing
4. **Prediction speed requirements**: Online serving constraints
5. **Handling of categorical variables**: Native support vs encoding
6. **Class imbalance**: Algorithm's robustness to imbalance
7. **Probabilistic outputs**: Need for calibrated probabilities
8. **Scalability**: Performance with increasing data size

### Q2: How do you handle severely imbalanced datasets in classification?
**Answer**: 
1. **Data-level approaches**:
   - Oversampling minority (SMOTE, ADASYN)
   - Undersampling majority (Random, Tomek links)
   - Synthetic data generation
2. **Algorithm-level approaches**:
   - Class weights/cost-sensitive learning
   - Threshold optimization
   - Ensemble methods focused on minority class
3. **Evaluation metrics**:
   - Use Precision-Recall instead of ROC-AUC
   - F1-score, Matthews Correlation Coefficient
   - Business-specific metrics (cost/benefit analysis)

### Q3: Explain the difference between hard and soft voting in ensemble classifiers.
**Answer**:
- **Hard voting**: Each classifier makes a class prediction, final prediction is majority vote. Simple but ignores confidence.
- **Soft voting**: Averages predicted probabilities from each classifier, predicts class with highest average probability. Generally performs better as it considers prediction confidence.

Example: For 3 classifiers predicting [A, A, B] with probabilities [(0.51, 0.49), (0.55, 0.45), (0.40, 0.60)]:
- Hard voting: A (2 votes vs 1)
- Soft voting: B (avg probabilities: A=0.49, B=0.51)

### Q4: What is calibration in classification and why is it important?
**Answer**: Calibration ensures predicted probabilities reflect true likelihood of positive class. Important because:
1. **Decision making**: Accurate probabilities needed for risk assessment
2. **Threshold selection**: Meaningful probabilities allow principled threshold choice
3. **Model combination**: Calibrated probabilities can be meaningfully averaged

Methods:
- **Platt scaling**: Fits sigmoid to map scores to probabilities
- **Isotonic regression**: Non-parametric, more flexible but needs more data

### Q5: How do you evaluate a multi-class classifier?
**Answer**:
1. **Overall metrics**:
   - Accuracy (can be misleading if imbalanced)
   - Macro/Micro/Weighted F1-scores
2. **Per-class analysis**:
   - Confusion matrix visualization
   - Per-class precision, recall, F1
   - Class-wise ROC curves
3. **Advanced metrics**:
   - Cohen's Kappa (accounts for chance agreement)
   - Matthews Correlation Coefficient
   - Top-k accuracy for ranking problems

### Q6: What's the difference between OneVsRest and OneVsOne for multi-class classification?
**Answer**:
- **OneVsRest (OvR)**: 
  - Trains K binary classifiers (K = number of classes)
  - Each separates one class from all others
  - Faster training, natural probability scores
  - Can suffer from imbalanced training sets

- **OneVsOne (OvO)**:
  - Trains K(K-1)/2 binary classifiers
  - Each trained on pairs of classes
  - More balanced training sets
  - Slower prediction, voting needed for final class

### Q7: How do you handle missing values in classification problems?
**Answer**:
1. **Understanding missingness**:
   - MCAR (Missing Completely At Random)
   - MAR (Missing At Random)
   - MNAR (Missing Not At Random)
2. **Strategies**:
   - Simple imputation (mean, median, mode)
   - Advanced imputation (KNN, MICE, deep learning)
   - Indicator variables for missingness
   - Tree-based models (can handle naturally)
3. **Model-specific approaches**:
   - Some algorithms handle missing values natively
   - Create "missing" category for categorical variables

### Q8: Explain feature importance in Random Forests for classification.
**Answer**: Random Forest calculates feature importance using:
1. **Gini importance**: Average decrease in node impurity (Gini index) when splitting on feature, weighted by number of samples reaching node
2. **Permutation importance**: Decrease in model performance when feature values are randomly shuffled

Considerations:
- Biased towards high-cardinality features
- Correlated features share importance
- Different from coefficients in linear models
- Use SHAP values for better interpretation

### Q9: How do you detect and handle overfitting in classification?
**Answer**:
**Detection**:
1. Large gap between training and validation accuracy
2. Perfect or near-perfect training accuracy
3. High variance in cross-validation scores
4. Poor performance on hold-out test set

**Prevention/Handling**:
1. **Regularization**: L1/L2 penalties, dropout
2. **Simplify model**: Reduce complexity, fewer features
3. **More data**: Collect more samples, data augmentation
4. **Ensemble methods**: Bagging, boosting
5. **Early stopping**: Stop training when validation performance plateaus
6. **Cross-validation**: Better estimate of generalization

### Q10: What are the advantages of using ensemble methods for classification?
**Answer**:
1. **Improved accuracy**: Combines multiple weak learners
2. **Reduced overfitting**: Especially with bagging
3. **Better generalization**: Captures different aspects of data
4. **Robustness**: Less sensitive to outliers/noise
5. **Flexibility**: Can combine different algorithm types

Types:
- **Bagging**: Reduces variance (Random Forest)
- **Boosting**: Reduces bias (AdaBoost, XGBoost)
- **Stacking**: Learns optimal combination

### Q11: How do you choose between precision and recall in classification?
**Answer**: Depends on business context and cost of errors:

**Favor Precision when**:
- False positives are costly (spam detection - don't want to miss important emails)
- Limited resources for follow-up
- User trust is critical

**Favor Recall when**:
- False negatives are costly (disease detection - don't want to miss sick patients)
- Safety-critical applications
- Initial screening before human review

Use F1-score for balance, or weighted F-beta score for custom trade-offs.

### Q12: Explain the concept of class weights in classification.
**Answer**: Class weights adjust the loss function to penalize misclassification of certain classes more heavily:

```
weighted_loss = weight[class] * original_loss
```

Applications:
1. **Imbalanced datasets**: Higher weight for minority class
2. **Cost-sensitive learning**: Weight proportional to misclassification cost
3. **Business priorities**: Emphasize important classes

Implementation:
- `class_weight='balanced'`: Automatically adjusts inversely proportional to class frequencies
- Custom weights: Dictionary mapping class to weight

### Q13: What is stratified sampling and why is it important in classification?
**Answer**: Stratified sampling ensures each split (train/test/validation) maintains the same class distribution as the original dataset.

**Importance**:
1. **Representative splits**: Each set reflects overall class balance
2. **Reliable evaluation**: Test set performance more stable
3. **Better for imbalanced data**: Ensures minority class appears in all splits
4. **Consistent cross-validation**: Each fold has similar class distribution

Implementation: Use `stratify` parameter in `train_test_split` or `StratifiedKFold` for CV.

### Q14: How do you handle multi-label classification problems?
**Answer**:
1. **Problem transformation**:
   - Binary Relevance: Independent binary classifier per label
   - Classifier Chains: Sequential prediction considering correlations
   - Label Powerset: Each unique label combination as one class

2. **Algorithm adaptation**:
   - Multi-label specific algorithms (ML-kNN, ML-ARAM)
   - Modify existing algorithms

3. **Evaluation metrics**:
   - Hamming Loss: Fraction of wrong labels
   - Subset Accuracy: Exact match ratio
   - Micro/Macro F1 scores

4. **Challenges**:
   - Label correlations
   - Imbalanced label distributions
   - Computational complexity

### Q15: What are the key differences between generative and discriminative classifiers?
**Answer**:

**Generative classifiers** (e.g., Naive Bayes, LDA):
- Model P(X|Y) and P(Y), then use Bayes' rule for P(Y|X)
- Can generate new data samples
- Work well with small datasets
- Handle missing data naturally

**Discriminative classifiers** (e.g., Logistic Regression, SVM):
- Directly model P(Y|X) or decision boundary
- Generally better accuracy with sufficient data
- More robust to model assumptions
- Focus on classification boundary

### Q16: How do you validate a classification model in production?
**Answer**:
1. **A/B testing**: Compare new model against current
2. **Shadow mode**: Run parallel, log predictions without serving
3. **Gradual rollout**: Start with small traffic percentage
4. **Monitoring**:
   - Prediction distribution shifts
   - Feature distribution changes
   - Business metric tracking
   - Error analysis and edge cases
5. **Feedback loops**: Collect labels for ongoing evaluation
6. **Performance tracking**: Latency, throughput, resource usage

### Q17: Explain concept drift in classification and how to handle it.
**Answer**: Concept drift occurs when the statistical properties of target variable change over time.

**Types**:
1. **Sudden drift**: Abrupt change
2. **Gradual drift**: Slow change over time
3. **Incremental drift**: Continuous small changes
4. **Recurring concepts**: Seasonal patterns

**Detection**:
- Monitor prediction confidence distribution
- Track feature distributions
- Performance metrics on recent data
- Statistical tests (Kolmogorov-Smirnov, Page-Hinkley)

**Handling**:
- Regular model retraining
- Online learning algorithms
- Ensemble with different time windows
- Adaptive learning rates

### Q18: What role does feature scaling play in classification?
**Answer**: Feature scaling importance varies by algorithm:

**Critical for**:
- **SVM**: Uses distance in kernel calculations
- **Neural Networks**: Gradient descent convergence
- **KNN**: Distance-based classification
- **Regularized models**: Penalty applies equally to all features

**Not required for**:
- **Tree-based models**: Split based on values, not distances
- **Naive Bayes**: Assumes feature independence

**Methods**:
- StandardScaler: Zero mean, unit variance
- MinMaxScaler: Scale to [0,1]
- RobustScaler: Robust to outliers
- Normalizer: Unit norm per sample

### Q19: How do you interpret classification model predictions for stakeholders?
**Answer**:
1. **Global interpretation**:
   - Feature importance plots
   - Partial dependence plots
   - Model-agnostic methods (LIME, SHAP)

2. **Local interpretation**:
   - Individual prediction explanations
   - Counterfactual examples
   - Similar cases from training data

3. **Visualization**:
   - Decision boundaries (2D/3D projections)
   - Confusion matrix heatmaps
   - ROC/PR curves with operating points

4. **Business context**:
   - Translate metrics to business impact
   - Cost-benefit analysis
   - Confidence intervals
   - Limitations and assumptions

### Q20: What are common pitfalls in classification projects and how to avoid them?
**Answer**:
1. **Data leakage**: 
   - Future information in features
   - *Solution*: Careful feature engineering, temporal validation

2. **Inappropriate metrics**:
   - Using accuracy for imbalanced data
   - *Solution*: Choose metrics aligned with business goals

3. **Overfitting to validation set**:
   - Multiple iterations on same validation data
   - *Solution*: Keep truly held-out test set

4. **Ignoring class imbalance**:
   - Model predicts only majority class
   - *Solution*: Appropriate sampling, metrics, thresholds

5. **Not considering prediction costs**:
   - Expensive features at inference time
   - *Solution*: Feature importance vs cost analysis

6. **Poor preprocessing**:
   - Inconsistent train/test preprocessing
   - *Solution*: Use pipelines, save preprocessors

### Q21: How do you choose the optimal decision threshold for a binary classifier?
**Answer**: 
1. **Define objective**: Maximize F1, minimize cost, target specific precision/recall
2. **Methods**:
   - Grid search over thresholds
   - ROC curve: Choose point closest to (0,1)
   - Precision-Recall curve: Based on requirements
   - Cost-benefit analysis: Maximize expected value
3. **Validation**: Use separate validation set, not training data
4. **Considerations**: May need different thresholds for different segments

### Q22: Explain the role of cross-validation in classification model selection.
**Answer**: Cross-validation provides robust performance estimates by training/testing on multiple data splits:

**Benefits**:
1. Better generalization estimate than single split
2. Uses all data for training and validation
3. Reduces variance in performance metrics
4. Helps detect overfitting

**Types**:
- **K-Fold**: Standard approach, k iterations
- **Stratified K-Fold**: Maintains class distribution
- **Leave-One-Out**: K = n samples (expensive)
- **Time Series Split**: For temporal data

**Best practices**:
- Use same CV strategy for all models in comparison
- Stratify for imbalanced datasets
- Consider computational cost vs. robustness