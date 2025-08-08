# Day 27: Machine Learning Pipeline

## Table of Contents
1. [Introduction](#1-introduction)
2. [Understanding ML Pipelines](#2-understanding-ml-pipelines)
3. [Building Pipelines with Scikit-learn](#3-building-pipelines-with-scikit-learn)
4. [Advanced Pipeline Components](#4-advanced-pipeline-components)
5. [Custom Pipeline Components](#5-custom-pipeline-components)
6. [Production Pipeline Design](#6-production-pipeline-design)
7. [Pipeline Optimization](#7-pipeline-optimization)
8. [MLOps and Pipeline Management](#8-mlops-and-pipeline-management)
9. [Case Studies](#9-case-studies)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

A machine learning pipeline is an end-to-end construct that orchestrates the flow of data into, through, and out of a machine learning model. It encompasses all steps from raw data to predictions, ensuring reproducibility, maintainability, and scalability.

### Why ML Pipelines Matter

1. **Reproducibility**: Same results every time
2. **Maintainability**: Easy to update and debug
3. **Scalability**: Handle increasing data volumes
4. **Automation**: Reduce manual intervention
5. **Consistency**: Standardized workflow across projects

### Pipeline Components Overview

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipelineOverview:
    """Overview of ML pipeline components"""
    
    @staticmethod
    def visualize_pipeline_flow():
        """Visualize typical ML pipeline flow"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define pipeline stages
        stages = [
            'Raw Data',
            'Data Validation',
            'Data Preprocessing',
            'Feature Engineering',
            'Feature Selection',
            'Model Training',
            'Model Validation',
            'Model Deployment',
            'Monitoring'
        ]
        
        # Define connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (6, 7), (7, 8), (8, 2)  # Feedback loop
        ]
        
        # Position nodes
        positions = {
            0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (3, 4),
            4: (4, 4), 5: (5, 4), 6: (6, 4), 7: (7, 4),
            8: (7, 2)
        }
        
        # Draw connections
        for start, end in connections:
            x_values = [positions[start][0], positions[end][0]]
            y_values = [positions[start][1], positions[end][1]]
            
            if start == 8 and end == 2:  # Feedback loop
                ax.annotate('', xy=positions[end], xytext=positions[start],
                           arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                         connectionstyle="arc3,rad=0.3"))
            else:
                ax.plot(x_values, y_values, 'b-', lw=2)
                ax.annotate('', xy=positions[end], xytext=(positions[end][0]-0.1, positions[end][1]),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Draw nodes
        for i, (stage, pos) in enumerate(zip(stages, positions.values())):
            if i < len(stages) - 1:
                circle = plt.Circle(pos, 0.3, color='lightblue', ec='darkblue', lw=2)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], str(i+1), ha='center', va='center', fontsize=12, fontweight='bold')
                ax.text(pos[0], pos[1]-0.6, stage, ha='center', va='top', fontsize=10)
        
        # Special handling for monitoring (feedback)
        pos = positions[8]
        circle = plt.Circle(pos, 0.3, color='lightcoral', ec='darkred', lw=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], '9', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(pos[0], pos[1]-0.6, 'Monitoring', ha='center', va='top', fontsize=10)
        
        ax.set_xlim(-1, 8)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Machine Learning Pipeline Flow', fontsize=16, fontweight='bold')
        
        # Add legend
        ax.text(4, 0.5, 'Blue arrows: Forward flow', ha='center', color='blue')
        ax.text(4, 0, 'Red arrow: Feedback loop', ha='center', color='red')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def demonstrate_simple_pipeline():
        """Demonstrate a simple sklearn pipeline"""
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'numeric_1': np.random.normal(100, 15, n_samples),
            'numeric_2': np.random.exponential(50, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Add some missing values
        data.loc[data.sample(frac=0.1).index, 'numeric_1'] = np.nan
        data.loc[data.sample(frac=0.05).index, 'categorical'] = np.nan
        
        print("Sample data:")
        print(data.head())
        print(f"\nMissing values:\n{data.isnull().sum()}")
        
        # Define preprocessing for numeric and categorical features
        numeric_features = ['numeric_1', 'numeric_2']
        categorical_features = ['categorical']
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create full pipeline
        from sklearn.ensemble import RandomForestClassifier
        
        clf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit pipeline
        clf_pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = clf_pipeline.score(X_train, y_train)
        test_score = clf_pipeline.score(X_test, y_test)
        
        print(f"\nPipeline Performance:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        return clf_pipeline, data
```

## 2. Understanding ML Pipelines

### 2.1 Pipeline Architecture

```python
class PipelineArchitecture:
    """Understanding different pipeline architectures"""
    
    @staticmethod
    def sequential_pipeline():
        """Traditional sequential pipeline"""
        
        # Linear flow: A -> B -> C -> D
        sequential = Pipeline([
            ('step1', StandardScaler()),
            ('step2', PCA(n_components=10)),
            ('step3', RandomForestClassifier())
        ])
        
        return sequential
    
    @staticmethod
    def parallel_pipeline():
        """Parallel processing pipeline using FeatureUnion"""
        
        # Parallel branches that merge
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest
        
        parallel = FeatureUnion([
            ('pca', PCA(n_components=10)),
            ('select_best', SelectKBest(k=5)),
            ('raw_features', 'passthrough')  # Include original features
        ])
        
        # Combine with classifier
        full_pipeline = Pipeline([
            ('features', parallel),
            ('classifier', RandomForestClassifier())
        ])
        
        return full_pipeline
    
    @staticmethod
    def nested_pipeline():
        """Nested pipeline with complex structure"""
        
        # Preprocessing pipeline
        preprocessing = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
        ])
        
        # Feature engineering pipeline
        feature_engineering = Pipeline([
            ('preprocessor', preprocessing),
            ('polynomial', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        # Feature selection pipeline
        feature_selection = Pipeline([
            ('engineer', feature_engineering),
            ('select', SelectKBest(k=20))
        ])
        
        # Final pipeline
        final_pipeline = Pipeline([
            ('features', feature_selection),
            ('classifier', RandomForestClassifier())
        ])
        
        return final_pipeline
    
    @staticmethod
    def conditional_pipeline():
        """Pipeline with conditional logic"""
        
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class ConditionalTransformer(BaseEstimator, TransformerMixin):
            """Apply different transformations based on data characteristics"""
            
            def __init__(self, threshold=0.5):
                self.threshold = threshold
                self.transformer_ = None
                
            def fit(self, X, y=None):
                # Decide which transformer to use based on data
                skewness = np.abs(X).mean()
                
                if skewness > self.threshold:
                    self.transformer_ = PowerTransformer()
                else:
                    self.transformer_ = StandardScaler()
                
                self.transformer_.fit(X, y)
                return self
            
            def transform(self, X):
                return self.transformer_.transform(X)
        
        conditional_pipeline = Pipeline([
            ('conditional', ConditionalTransformer()),
            ('classifier', RandomForestClassifier())
        ])
        
        return conditional_pipeline

### 2.2 Pipeline Design Patterns

class PipelinePatterns:
    """Common pipeline design patterns"""
    
    @staticmethod
    def data_validation_pattern():
        """Pipeline with data validation steps"""
        
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class DataValidator(BaseEstimator, TransformerMixin):
            """Validate data quality"""
            
            def __init__(self, max_missing_ratio=0.1, min_samples=100):
                self.max_missing_ratio = max_missing_ratio
                self.min_samples = min_samples
                
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                # Check sample size
                if len(X) < self.min_samples:
                    raise ValueError(f"Insufficient samples: {len(X)} < {self.min_samples}")
                
                # Check missing values
                if hasattr(X, 'isnull'):
                    missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
                else:
                    missing_ratio = np.isnan(X).sum() / X.size
                
                if missing_ratio > self.max_missing_ratio:
                    raise ValueError(f"Too many missing values: {missing_ratio:.2%}")
                
                return X
        
        validation_pipeline = Pipeline([
            ('validator', DataValidator()),
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        
        return validation_pipeline
    
    @staticmethod
    def caching_pattern():
        """Pipeline with caching for expensive operations"""
        
        # Use memory parameter for caching
        from joblib import Memory
        
        # Create cache directory
        cachedir = './pipeline_cache'
        memory = Memory(location=cachedir, verbose=0)
        
        # Expensive transformer
        class ExpensiveTransformer(BaseEstimator, TransformerMixin):
            """Simulates expensive computation"""
            
            def fit(self, X, y=None):
                # Expensive fitting operation
                import time
                time.sleep(2)  # Simulate expensive operation
                return self
            
            def transform(self, X):
                # Expensive transformation
                return X ** 2
        
        cached_pipeline = Pipeline([
            ('expensive', ExpensiveTransformer()),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ], memory=memory)
        
        return cached_pipeline
    
    @staticmethod
    def logging_pattern():
        """Pipeline with comprehensive logging"""
        
        import logging
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class LoggingTransformer(BaseEstimator, TransformerMixin):
            """Transformer that logs its operations"""
            
            def __init__(self, name="transformer"):
                self.name = name
                self.logger = logging.getLogger(self.name)
                
            def fit(self, X, y=None):
                self.logger.info(f"Fitting {self.name} on {X.shape} data")
                self.n_features_in_ = X.shape[1]
                return self
            
            def transform(self, X):
                self.logger.info(f"Transforming {X.shape} data with {self.name}")
                
                if X.shape[1] != self.n_features_in_:
                    self.logger.warning(f"Feature count mismatch: {X.shape[1]} != {self.n_features_in_}")
                
                return X
        
        logging_pipeline = Pipeline([
            ('log_input', LoggingTransformer('input_logger')),
            ('scaler', StandardScaler()),
            ('log_scaled', LoggingTransformer('scaled_logger')),
            ('classifier', RandomForestClassifier())
        ])
        
        return logging_pipeline
```

## 3. Building Pipelines with Scikit-learn

### 3.1 Basic Pipeline Construction

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class BasicPipelineConstruction:
    """Building basic pipelines with scikit-learn"""
    
    @staticmethod
    def manual_pipeline():
        """Manually constructed pipeline"""
        
        # Explicit construction
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
            ('selector', SelectKBest(f_classif, k=10)),
            ('svm', SVC(kernel='rbf'))
        ])
        
        return pipe
    
    @staticmethod
    def make_pipeline_example():
        """Using make_pipeline for automatic naming"""
        
        # Automatic naming
        pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            SelectKBest(f_classif, k=10),
            SVC(kernel='rbf')
        )
        
        # Check step names
        print("Pipeline steps:")
        for name, step in pipe.steps:
            print(f"  {name}: {type(step).__name__}")
        
        return pipe
    
    @staticmethod
    def column_transformer_pipeline():
        """Pipeline with ColumnTransformer for heterogeneous data"""
        
        # Sample heterogeneous data
        data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [30000, 45000, 55000, 60000, 80000],
            'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
            'years_experience': [1, 5, 8, 10, 15]
        })
        
        # Define column groups
        numeric_features = ['age', 'salary', 'years_experience']
        categorical_features = ['city', 'department']
        
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop other columns
        )
        
        # Full pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])
        
        return full_pipeline, data
    
    @staticmethod
    def feature_union_pipeline():
        """Pipeline using FeatureUnion for parallel processing"""
        
        from sklearn.pipeline import FeatureUnion
        from sklearn.decomposition import PCA, TruncatedSVD
        
        # Parallel feature extraction
        combined_features = FeatureUnion([
            # PCA features
            ('pca', Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=10))
            ])),
            
            # Polynomial features
            ('poly', Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ])),
            
            # Original features (scaled)
            ('original', StandardScaler())
        ])
        
        # Complete pipeline
        pipeline = Pipeline([
            ('features', combined_features),
            ('classifier', RandomForestClassifier())
        ])
        
        return pipeline

### 3.2 Pipeline Parameter Access and Modification

class PipelineParameterHandling:
    """Handling pipeline parameters"""
    
    @staticmethod
    def access_pipeline_params():
        """Access and modify pipeline parameters"""
        
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=5)),
            ('svm', SVC(kernel='rbf', C=1.0))
        ])
        
        # Access parameters
        print("Initial parameters:")
        print(f"PCA components: {pipe.named_steps['pca'].n_components}")
        print(f"SVM C: {pipe.named_steps['svm'].C}")
        
        # Modify parameters
        pipe.set_params(pca__n_components=10, svm__C=10.0)
        
        print("\nModified parameters:")
        print(f"PCA components: {pipe.named_steps['pca'].n_components}")
        print(f"SVM C: {pipe.named_steps['svm'].C}")
        
        # Get all parameters
        all_params = pipe.get_params()
        print(f"\nTotal parameters: {len(all_params)}")
        
        return pipe
    
    @staticmethod
    def grid_search_pipeline():
        """Grid search over pipeline parameters"""
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import make_classification
        
        # Generate data
        X, y = make_classification(n_samples=1000, n_features=20, 
                                  n_informative=15, random_state=42)
        
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('svm', SVC())
        ])
        
        # Parameter grid
        param_grid = {
            'pca__n_components': [5, 10, 15],
            'svm__kernel': ['rbf', 'linear'],
            'svm__C': [0.1, 1, 10]
        }
        
        # Grid search
        grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        
        print("Best parameters:")
        print(grid_search.best_params_)
        print(f"\nBest score: {grid_search.best_score_:.3f}")
        
        return grid_search
```

## 4. Advanced Pipeline Components

### 4.1 Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class AdvancedTransformers:
    """Advanced custom transformers for pipelines"""
    
    class OutlierRemover(BaseEstimator, TransformerMixin):
        """Remove outliers using IQR method"""
        
        def __init__(self, factor=1.5):
            self.factor = factor
            self.bounds_ = {}
            
        def fit(self, X, y=None):
            """Calculate bounds for each feature"""
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            
            for col in X_df.columns:
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                self.bounds_[col] = {
                    'lower': Q1 - self.factor * IQR,
                    'upper': Q3 + self.factor * IQR
                }
            
            return self
        
        def transform(self, X):
            """Remove outliers"""
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
            
            for col in X_df.columns:
                if col in self.bounds_:
                    bounds = self.bounds_[col]
                    X_df[col] = X_df[col].clip(lower=bounds['lower'], 
                                               upper=bounds['upper'])
            
            return X_df.values if not isinstance(X, pd.DataFrame) else X_df
    
    class FeatureInteractionCreator(BaseEstimator, TransformerMixin):
        """Create interaction features"""
        
        def __init__(self, interaction_only=True, include_bias=False):
            self.interaction_only = interaction_only
            self.include_bias = include_bias
            self.feature_names_ = None
            
        def fit(self, X, y=None):
            n_features = X.shape[1]
            self.n_features_in_ = n_features
            
            # Generate feature names for interactions
            self.feature_names_ = []
            
            if self.include_bias:
                self.feature_names_.append('bias')
            
            # Original features
            for i in range(n_features):
                self.feature_names_.append(f'x{i}')
            
            # Interactions
            for i in range(n_features):
                for j in range(i+1, n_features):
                    self.feature_names_.append(f'x{i}_x{j}')
                
                # Include powers if not interaction_only
                if not self.interaction_only:
                    self.feature_names_.append(f'x{i}^2')
            
            return self
        
        def transform(self, X):
            """Create interaction features"""
            n_samples, n_features = X.shape
            
            features = []
            
            if self.include_bias:
                features.append(np.ones((n_samples, 1)))
            
            # Original features
            features.append(X)
            
            # Interactions
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)
                
                # Powers
                if not self.interaction_only:
                    power = (X[:, i] ** 2).reshape(-1, 1)
                    features.append(power)
            
            return np.hstack(features)
        
        def get_feature_names_out(self, input_features=None):
            """Get output feature names"""
            return self.feature_names_
    
    class TargetEncoder(BaseEstimator, TransformerMixin):
        """Target encoding for categorical variables"""
        
        def __init__(self, smoothing=1.0, min_samples_leaf=1):
            self.smoothing = smoothing
            self.min_samples_leaf = min_samples_leaf
            self.encodings_ = {}
            
        def fit(self, X, y):
            """Learn target encodings"""
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            
            # Global target mean
            self.global_mean_ = np.mean(y)
            
            for col in X_df.columns:
                # Calculate target statistics for each category
                stats = pd.DataFrame({'cat': X_df[col], 'target': y})
                stats = stats.groupby('cat')['target'].agg(['mean', 'count'])
                
                # Apply smoothing
                smoothed_mean = (
                    (stats['mean'] * stats['count'] + 
                     self.global_mean_ * self.smoothing) /
                    (stats['count'] + self.smoothing)
                )
                
                self.encodings_[col] = smoothed_mean.to_dict()
            
            return self
        
        def transform(self, X):
            """Apply target encodings"""
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
            
            for col in X_df.columns:
                if col in self.encodings_:
                    # Map encodings
                    X_df[col] = X_df[col].map(self.encodings_[col])
                    # Fill unknown categories with global mean
                    X_df[col].fillna(self.global_mean_, inplace=True)
            
            return X_df.values if not isinstance(X, pd.DataFrame) else X_df

### 4.2 Pipeline Monitoring Components

class PipelineMonitoring:
    """Components for monitoring pipeline execution"""
    
    class PerformanceMonitor(BaseEstimator, TransformerMixin):
        """Monitor transformation performance"""
        
        def __init__(self, name="transformer"):
            self.name = name
            self.metrics_ = {
                'fit_times': [],
                'transform_times': [],
                'data_shapes': []
            }
            
        def fit(self, X, y=None):
            """Fit and record metrics"""
            import time
            
            start_time = time.time()
            # Actual fitting logic would go here
            fit_time = time.time() - start_time
            
            self.metrics_['fit_times'].append(fit_time)
            self.metrics_['data_shapes'].append(X.shape)
            
            print(f"{self.name} - Fit time: {fit_time:.3f}s on shape {X.shape}")
            
            return self
        
        def transform(self, X):
            """Transform and record metrics"""
            import time
            
            start_time = time.time()
            # Actual transformation would go here
            result = X  # Placeholder
            transform_time = time.time() - start_time
            
            self.metrics_['transform_times'].append(transform_time)
            
            print(f"{self.name} - Transform time: {transform_time:.3f}s on shape {X.shape}")
            
            return result
        
        def get_report(self):
            """Get performance report"""
            report = {
                'name': self.name,
                'avg_fit_time': np.mean(self.metrics_['fit_times']),
                'avg_transform_time': np.mean(self.metrics_['transform_times']),
                'total_transforms': len(self.metrics_['transform_times'])
            }
            return report
    
    class DataDriftDetector(BaseEstimator, TransformerMixin):
        """Detect data drift in pipeline"""
        
        def __init__(self, threshold=0.1, method='ks'):
            self.threshold = threshold
            self.method = method
            self.reference_stats_ = {}
            
        def fit(self, X, y=None):
            """Store reference statistics"""
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            
            for col in X_df.columns:
                if X_df[col].dtype in ['int64', 'float64']:
                    self.reference_stats_[col] = {
                        'mean': X_df[col].mean(),
                        'std': X_df[col].std(),
                        'distribution': X_df[col].values
                    }
            
            return self
        
        def transform(self, X):
            """Check for drift and transform"""
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            
            drift_detected = False
            drift_scores = {}
            
            for col in X_df.columns:
                if col in self.reference_stats_:
                    if self.method == 'ks':
                        # Kolmogorov-Smirnov test
                        from scipy.stats import ks_2samp
                        
                        stat, p_value = ks_2samp(
                            self.reference_stats_[col]['distribution'],
                            X_df[col].values
                        )
                        
                        drift_scores[col] = p_value
                        if p_value < self.threshold:
                            drift_detected = True
                            print(f"Warning: Drift detected in {col} (p-value: {p_value:.4f})")
            
            if drift_detected:
                print("Data drift detected! Consider retraining the pipeline.")
            
            return X_df.values if not isinstance(X, pd.DataFrame) else X_df
```

## 5. Custom Pipeline Components

### 5.1 Building Custom Estimators

```python
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class CustomPipelineComponents:
    """Custom components for ML pipelines"""
    
    class CustomClassifier(BaseEstimator, ClassifierMixin):
        """Template for custom classifier"""
        
        def __init__(self, param1=1.0, param2='auto'):
            self.param1 = param1
            self.param2 = param2
            
        def fit(self, X, y):
            """Fit the classifier"""
            # Check inputs
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)
            
            # Store training data info
            self.n_features_in_ = X.shape[1]
            self.n_samples_fit_ = X.shape[0]
            
            # Actual fitting logic
            # For demo, just store mean of each class
            self.class_means_ = {}
            for cls in self.classes_:
                mask = y == cls
                self.class_means_[cls] = X[mask].mean(axis=0)
            
            # Return self for sklearn compatibility
            return self
        
        def predict(self, X):
            """Predict classes"""
            # Check if fitted
            check_is_fitted(self)
            
            # Input validation
            X = check_array(X)
            
            # Simple nearest mean classifier
            predictions = []
            for sample in X:
                distances = {}
                for cls, mean in self.class_means_.items():
                    distances[cls] = np.linalg.norm(sample - mean)
                predictions.append(min(distances, key=distances.get))
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            """Predict class probabilities"""
            check_is_fitted(self)
            X = check_array(X)
            
            # Simple softmax over negative distances
            probas = []
            for sample in X:
                distances = []
                for cls in self.classes_:
                    dist = np.linalg.norm(sample - self.class_means_[cls])
                    distances.append(-dist)  # Negative for softmax
                
                # Softmax
                exp_distances = np.exp(distances - np.max(distances))
                proba = exp_distances / exp_distances.sum()
                probas.append(proba)
            
            return np.array(probas)
    
    class MultiOutputTransformer(BaseEstimator, TransformerMixin):
        """Transformer that creates multiple output features"""
        
        def __init__(self, n_outputs=3):
            self.n_outputs = n_outputs
            
        def fit(self, X, y=None):
            """Learn transformation parameters"""
            X = check_array(X)
            
            # Learn random projection matrix
            self.n_features_in_ = X.shape[1]
            self.projection_matrix_ = np.random.randn(
                self.n_features_in_, 
                self.n_outputs
            )
            
            # Normalize columns
            self.projection_matrix_ /= np.linalg.norm(
                self.projection_matrix_, 
                axis=0
            )
            
            return self
        
        def transform(self, X):
            """Apply transformation"""
            check_is_fitted(self)
            X = check_array(X)
            
            # Apply projection
            X_transformed = X @ self.projection_matrix_
            
            # Add non-linear transformations
            X_combined = np.hstack([
                X_transformed,
                np.sin(X_transformed),
                np.exp(-np.abs(X_transformed))
            ])
            
            return X_combined
    
    class AdaptivePreprocessor(BaseEstimator, TransformerMixin):
        """Preprocessor that adapts based on data characteristics"""
        
        def __init__(self, strategy='auto'):
            self.strategy = strategy
            self.preprocessor_ = None
            
        def fit(self, X, y=None):
            """Determine and fit appropriate preprocessor"""
            X = check_array(X)
            
            if self.strategy == 'auto':
                # Analyze data characteristics
                data_stats = {
                    'mean': np.mean(X),
                    'std': np.std(X),
                    'skewness': self._calculate_skewness(X),
                    'has_negative': (X < 0).any()
                }
                
                # Choose preprocessor based on stats
                if abs(data_stats['skewness']) > 1:
                    from sklearn.preprocessing import PowerTransformer
                    self.preprocessor_ = PowerTransformer()
                elif data_stats['std'] > 10 * data_stats['mean']:
                    from sklearn.preprocessing import RobustScaler
                    self.preprocessor_ = RobustScaler()
                else:
                    from sklearn.preprocessing import StandardScaler
                    self.preprocessor_ = StandardScaler()
                
                print(f"Selected preprocessor: {type(self.preprocessor_).__name__}")
            
            # Fit selected preprocessor
            self.preprocessor_.fit(X)
            
            return self
        
        def transform(self, X):
            """Apply selected preprocessing"""
            check_is_fitted(self)
            X = check_array(X)
            
            return self.preprocessor_.transform(X)
        
        def _calculate_skewness(self, X):
            """Calculate average skewness"""
            from scipy.stats import skew
            return np.mean([skew(X[:, i]) for i in range(X.shape[1])])

### 5.2 Complex Pipeline Structures

class ComplexPipelineStructures:
    """Building complex pipeline structures"""
    
    @staticmethod
    def branching_pipeline():
        """Pipeline with conditional branching"""
        
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class ConditionalPipeline(BaseEstimator, TransformerMixin):
            """Apply different pipelines based on conditions"""
            
            def __init__(self, condition_func, pipeline_a, pipeline_b):
                self.condition_func = condition_func
                self.pipeline_a = pipeline_a
                self.pipeline_b = pipeline_b
                self.selected_pipeline_ = None
                
            def fit(self, X, y=None):
                # Evaluate condition
                if self.condition_func(X, y):
                    self.selected_pipeline_ = self.pipeline_a
                    print("Selected Pipeline A")
                else:
                    self.selected_pipeline_ = self.pipeline_b
                    print("Selected Pipeline B")
                
                # Fit selected pipeline
                self.selected_pipeline_.fit(X, y)
                return self
            
            def transform(self, X):
                return self.selected_pipeline_.transform(X)
        
        # Example condition function
        def has_many_features(X, y=None):
            return X.shape[1] > 50
        
        # Define two different pipelines
        pipeline_high_dim = Pipeline([
            ('pca', PCA(n_components=0.95)),
            ('scaler', StandardScaler())
        ])
        
        pipeline_low_dim = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler())
        ])
        
        # Create conditional pipeline
        conditional = ConditionalPipeline(
            condition_func=has_many_features,
            pipeline_a=pipeline_high_dim,
            pipeline_b=pipeline_low_dim
        )
        
        return conditional
    
    @staticmethod
    def ensemble_pipeline():
        """Pipeline that ensembles multiple models"""
        
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        
        # Define base pipelines
        pipe1 = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression())
        ])
        
        pipe2 = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])
        
        pipe3 = Pipeline([
            ('tree', DecisionTreeClassifier())
        ])
        
        # Combine in voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('lr_pipe', pipe1),
                ('svm_pipe', pipe2),
                ('tree_pipe', pipe3)
            ],
            voting='soft'
        )
        
        # Wrap in pipeline with preprocessing
        full_pipeline = Pipeline([
            ('preprocessor', StandardScaler()),
            ('ensemble', ensemble)
        ])
        
        return full_pipeline
    
    @staticmethod
    def multi_level_pipeline():
        """Multi-level hierarchical pipeline"""
        
        # Level 1: Basic preprocessing
        level1 = Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
        ])
        
        # Level 2: Feature engineering
        level2 = FeatureUnion([
            ('original', 'passthrough'),
            ('pca', PCA(n_components=10)),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        # Level 3: Feature selection
        level3 = SelectKBest(f_classif, k=50)
        
        # Level 4: Model
        level4 = RandomForestClassifier(n_estimators=100)
        
        # Combine all levels
        multi_level = Pipeline([
            ('level1_preprocessing', level1),
            ('level2_features', level2),
            ('level3_selection', level3),
            ('level4_model', level4)
        ])
        
        return multi_level
```

## 6. Production Pipeline Design

### 6.1 Production-Ready Pipelines

```python
import joblib
import json
from datetime import datetime
import logging

class ProductionPipeline:
    """Production-ready ML pipeline implementation"""
    
    def __init__(self, model_name="ml_model", version="1.0"):
        self.model_name = model_name
        self.version = version
        self.pipeline = None
        self.metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'features': None,
            'performance_metrics': {}
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"{model_name}_v{version}")
        
    def build_pipeline(self, config):
        """Build pipeline from configuration"""
        
        # Example configuration-driven pipeline
        preprocessing_steps = []
        
        # Add imputation if configured
        if config.get('imputation', {}).get('enabled', True):
            strategy = config['imputation'].get('strategy', 'median')
            preprocessing_steps.append(
                ('imputer', SimpleImputer(strategy=strategy))
            )
        
        # Add scaling if configured
        if config.get('scaling', {}).get('enabled', True):
            scaler_type = config['scaling'].get('type', 'standard')
            if scaler_type == 'standard':
                preprocessing_steps.append(
                    ('scaler', StandardScaler())
                )
            elif scaler_type == 'robust':
                preprocessing_steps.append(
                    ('scaler', RobustScaler())
                )
        
        # Create preprocessing pipeline
        preprocessor = Pipeline(preprocessing_steps)
        
        # Add model
        model_config = config.get('model', {})
        model_type = model_config.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(**model_config.get('params', {}))
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**model_config.get('params', {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Complete pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        self.metadata['config'] = config
        
        return self
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit pipeline with validation"""
        
        self.logger.info(f"Starting training for {self.model_name} v{self.version}")
        
        # Record feature names
        if hasattr(X_train, 'columns'):
            self.metadata['features'] = X_train.columns.tolist()
        
        # Fit pipeline
        start_time = datetime.now()
        self.pipeline.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.metadata['training_time'] = training_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_score = self.pipeline.score(X_val, y_val)
            self.metadata['performance_metrics']['validation_score'] = val_score
            self.logger.info(f"Validation score: {val_score:.4f}")
        
        return self
    
    def predict(self, X, return_proba=False):
        """Make predictions with logging"""
        
        self.logger.info(f"Making predictions on {len(X)} samples")
        
        # Validate input features
        if hasattr(X, 'columns') and self.metadata.get('features'):
            missing_features = set(self.metadata['features']) - set(X.columns)
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
        
        # Make predictions
        start_time = datetime.now()
        
        if return_proba and hasattr(self.pipeline, 'predict_proba'):
            predictions = self.pipeline.predict_proba(X)
        else:
            predictions = self.pipeline.predict(X)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Predictions completed in {prediction_time:.3f} seconds")
        
        return predictions
    
    def save(self, path):
        """Save pipeline and metadata"""
        
        # Save pipeline
        model_path = f"{path}/{self.model_name}_v{self.version}.pkl"
        joblib.dump(self.pipeline, model_path)
        
        # Save metadata
        metadata_path = f"{path}/{self.model_name}_v{self.version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
        
        return model_path, metadata_path
    
    def load(self, path):
        """Load pipeline and metadata"""
        
        # Load pipeline
        model_path = f"{path}/{self.model_name}_v{self.version}.pkl"
        self.pipeline = joblib.load(model_path)
        
        # Load metadata
        metadata_path = f"{path}/{self.model_name}_v{self.version}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.logger.info(f"Model loaded from {model_path}")
        
        return self
    
    def validate_input(self, X):
        """Validate input data"""
        
        issues = []
        
        # Check features
        if hasattr(X, 'columns') and self.metadata.get('features'):
            expected_features = set(self.metadata['features'])
            actual_features = set(X.columns)
            
            missing = expected_features - actual_features
            extra = actual_features - expected_features
            
            if missing:
                issues.append(f"Missing features: {missing}")
            if extra:
                issues.append(f"Extra features: {extra}")
        
        # Check data types
        if hasattr(X, 'dtypes'):
            for col in X.columns:
                if X[col].dtype == 'object':
                    issues.append(f"Categorical column {col} needs encoding")
        
        # Check for nulls
        if hasattr(X, 'isnull'):
            null_counts = X.isnull().sum()
            if null_counts.any():
                issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        if issues:
            self.logger.warning(f"Input validation issues: {issues}")
        
        return len(issues) == 0, issues

### 6.2 Pipeline Versioning and Deployment

class PipelineVersionControl:
    """Version control for ML pipelines"""
    
    def __init__(self, model_registry_path="./model_registry"):
        self.registry_path = model_registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self):
        """Load model registry"""
        registry_file = f"{self.registry_path}/registry.json"
        
        try:
            with open(registry_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'models': {},
                'active_versions': {}
            }
    
    def register_model(self, model_name, version, pipeline, metadata):
        """Register a new model version"""
        
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {}
        
        self.registry['models'][model_name][version] = {
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata,
            'status': 'registered'
        }
        
        # Save pipeline
        model_path = f"{self.registry_path}/{model_name}/v{version}"
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(pipeline, f"{model_path}/pipeline.pkl")
        
        with open(f"{model_path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._save_registry()
        
        return f"{model_name} v{version} registered successfully"
    
    def promote_version(self, model_name, version, stage='production'):
        """Promote model version to stage"""
        
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.registry['models'][model_name]:
            raise ValueError(f"Version {version} not found for {model_name}")
        
        # Update status
        self.registry['models'][model_name][version]['status'] = stage
        self.registry['active_versions'][f"{model_name}_{stage}"] = version
        
        self._save_registry()
        
        return f"{model_name} v{version} promoted to {stage}"
    
    def get_model(self, model_name, version=None, stage='production'):
        """Retrieve model by name and version/stage"""
        
        if version is None:
            # Get version for stage
            version_key = f"{model_name}_{stage}"
            if version_key not in self.registry['active_versions']:
                raise ValueError(f"No {stage} version found for {model_name}")
            version = self.registry['active_versions'][version_key]
        
        # Load pipeline
        model_path = f"{self.registry_path}/{model_name}/v{version}"
        pipeline = joblib.load(f"{model_path}/pipeline.pkl")
        
        with open(f"{model_path}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return pipeline, metadata
    
    def compare_versions(self, model_name, version1, version2):
        """Compare two model versions"""
        
        meta1 = self.registry['models'][model_name][version1]['metadata']
        meta2 = self.registry['models'][model_name][version2]['metadata']
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'performance_diff': {},
            'config_changes': {}
        }
        
        # Compare performance metrics
        metrics1 = meta1.get('performance_metrics', {})
        metrics2 = meta2.get('performance_metrics', {})
        
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            comparison['performance_diff'][metric] = val2 - val1
        
        # Compare configurations
        config1 = meta1.get('config', {})
        config2 = meta2.get('config', {})
        
        # Find differences
        for key in set(config1.keys()) | set(config2.keys()):
            if config1.get(key) != config2.get(key):
                comparison['config_changes'][key] = {
                    'v1': config1.get(key),
                    'v2': config2.get(key)
                }
        
        return comparison
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = f"{self.registry_path}/registry.json"
        os.makedirs(self.registry_path, exist_ok=True)
        
        with open(registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
```

## 7. Pipeline Optimization

### 7.1 Performance Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import numpy as np

class PipelineOptimization:
    """Techniques for optimizing ML pipelines"""
    
    @staticmethod
    def optimize_hyperparameters(pipeline, param_distributions, X, y, n_iter=50):
        """Optimize pipeline hyperparameters"""
        
        # Random search for efficiency
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        
        print("Starting hyperparameter optimization...")
        random_search.fit(X, y)
        
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best score: {random_search.best_score_:.4f}")
        
        # Analyze parameter importance
        results_df = pd.DataFrame(random_search.cv_results_)
        
        # Find most important parameters
        param_importance = {}
        for param in param_distributions.keys():
            if param in results_df.columns:
                correlation = results_df[param].corr(
                    results_df['mean_test_score']
                )
                param_importance[param] = abs(correlation)
        
        print("\nParameter importance (correlation with score):")
        for param, importance in sorted(param_importance.items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {param}: {importance:.3f}")
        
        return random_search.best_estimator_
    
    @staticmethod
    def optimize_feature_selection(pipeline, X, y):
        """Optimize feature selection in pipeline"""
        
        from sklearn.feature_selection import RFECV
        
        # Extract classifier from pipeline
        classifier = pipeline.named_steps['classifier']
        
        # Create new pipeline with RFECV
        optimized_pipeline = Pipeline([
            ('preprocessor', pipeline.named_steps['preprocessor']),
            ('feature_selection', RFECV(
                estimator=classifier,
                step=1,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )),
            ('classifier', clone(classifier))
        ])
        
        # Fit and find optimal features
        optimized_pipeline.fit(X, y)
        
        n_features = optimized_pipeline.named_steps['feature_selection'].n_features_
        print(f"Optimal number of features: {n_features}")
        
        return optimized_pipeline
    
    @staticmethod
    def optimize_memory_usage(pipeline):
        """Optimize pipeline memory usage"""
        
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class MemoryEfficientTransformer(BaseEstimator, TransformerMixin):
            """Transformer that reduces memory usage"""
            
            def __init__(self, dtype='float32'):
                self.dtype = dtype
                
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                # Convert to more memory-efficient dtype
                if hasattr(X, 'astype'):
                    return X.astype(self.dtype)
                else:
                    return np.array(X, dtype=self.dtype)
        
        # Add memory optimization step
        steps = [('memory_optimizer', MemoryEfficientTransformer())]
        steps.extend(pipeline.steps)
        
        optimized_pipeline = Pipeline(steps)
        
        return optimized_pipeline
    
    @staticmethod
    def profile_pipeline(pipeline, X, y):
        """Profile pipeline performance"""
        
        import time
        import psutil
        import os
        
        results = {
            'step_times': {},
            'memory_usage': {},
            'output_shapes': {}
        }
        
        # Initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Profile each step
        X_current = X
        
        for step_name, step in pipeline.steps[:-1]:  # All but final estimator
            # Time measurement
            start_time = time.time()
            
            if hasattr(step, 'fit_transform'):
                X_current = step.fit_transform(X_current, y)
            else:
                step.fit(X_current, y)
                X_current = step.transform(X_current)
            
            step_time = time.time() - start_time
            
            # Memory measurement
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Store results
            results['step_times'][step_name] = step_time
            results['memory_usage'][step_name] = memory_increase
            results['output_shapes'][step_name] = (
                X_current.shape if hasattr(X_current, 'shape') else 'unknown'
            )
            
            print(f"{step_name}: {step_time:.3f}s, "
                  f"+{memory_increase:.1f}MB, "
                  f"shape={results['output_shapes'][step_name]}")
        
        # Final estimator
        final_name, final_step = pipeline.steps[-1]
        start_time = time.time()
        final_step.fit(X_current, y)
        results['step_times'][final_name] = time.time() - start_time
        
        # Total time
        total_time = sum(results['step_times'].values())
        print(f"\nTotal pipeline time: {total_time:.3f}s")
        
        return results

### 7.2 Pipeline Parallelization

class ParallelPipeline:
    """Parallel execution strategies for pipelines"""
    
    @staticmethod
    def create_parallel_feature_pipeline():
        """Create pipeline with parallel feature processing"""
        
        from joblib import Parallel, delayed
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class ParallelFeatureUnion(BaseEstimator, TransformerMixin):
            """Parallel version of FeatureUnion"""
            
            def __init__(self, transformer_list, n_jobs=-1):
                self.transformer_list = transformer_list
                self.n_jobs = n_jobs
                
            def fit(self, X, y=None):
                """Fit all transformers in parallel"""
                
                def fit_transformer(name, transformer, X, y):
                    return name, transformer.fit(X, y)
                
                # Parallel fitting
                fitted = Parallel(n_jobs=self.n_jobs)(
                    delayed(fit_transformer)(name, trans, X, y)
                    for name, trans in self.transformer_list
                )
                
                self.transformer_list = fitted
                return self
            
            def transform(self, X):
                """Transform with all transformers in parallel"""
                
                def transform_one(name, transformer, X):
                    return transformer.transform(X)
                
                # Parallel transformation
                Xs = Parallel(n_jobs=self.n_jobs)(
                    delayed(transform_one)(name, trans, X)
                    for name, trans in self.transformer_list
                )
                
                # Concatenate results
                return np.hstack(Xs)
        
        # Example usage
        parallel_features = ParallelFeatureUnion([
            ('pca', PCA(n_components=10)),
            ('select_best', SelectKBest(k=10)),
            ('poly', PolynomialFeatures(degree=2))
        ])
        
        pipeline = Pipeline([
            ('features', parallel_features),
            ('classifier', RandomForestClassifier())
        ])
        
        return pipeline
    
    @staticmethod
    def distributed_pipeline_training(pipeline, X, y, n_splits=5):
        """Distributed training using data parallelism"""
        
        from sklearn.base import clone
        from joblib import Parallel, delayed
        
        def train_fold(pipeline, X_train, y_train, X_val, y_val):
            """Train pipeline on one fold"""
            # Clone to avoid shared state
            fold_pipeline = clone(pipeline)
            
            # Train
            fold_pipeline.fit(X_train, y_train)
            
            # Evaluate
            score = fold_pipeline.score(X_val, y_val)
            
            return fold_pipeline, score
        
        # Create folds
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Parallel training
        results = Parallel(n_jobs=n_splits)(
            delayed(train_fold)(
                pipeline,
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx]
            )
            for train_idx, val_idx in kf.split(X)
        )
        
        # Extract models and scores
        models = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Return best model
        best_idx = np.argmax(scores)
        return models[best_idx], scores
```

## 8. MLOps and Pipeline Management

### 8.1 Pipeline Monitoring

```python
class PipelineMonitoringSystem:
    """Comprehensive pipeline monitoring"""
    
    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        self.metrics_history = []
        self.alerts = []
        
    def monitor_prediction_drift(self, predictions, reference_predictions=None):
        """Monitor prediction distribution drift"""
        
        from scipy.stats import ks_2samp
        
        current_stats = {
            'timestamp': datetime.now(),
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions)
        }
        
        self.metrics_history.append(current_stats)
        
        # Check for drift if we have reference
        if reference_predictions is not None:
            ks_stat, p_value = ks_2samp(predictions, reference_predictions)
            
            if p_value < 0.05:
                alert = {
                    'type': 'prediction_drift',
                    'severity': 'high',
                    'message': f'Prediction distribution drift detected (p={p_value:.4f})',
                    'timestamp': datetime.now()
                }
                self.alerts.append(alert)
                print(f"ALERT: {alert['message']}")
        
        return current_stats
    
    def monitor_feature_drift(self, X, reference_X=None):
        """Monitor feature distribution drift"""
        
        feature_stats = {}
        
        for col in range(X.shape[1]):
            feature_data = X[:, col]
            
            stats = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'missing_rate': np.isnan(feature_data).mean()
            }
            
            feature_stats[f'feature_{col}'] = stats
            
            # Check for drift
            if reference_X is not None:
                ref_data = reference_X[:, col]
                ks_stat, p_value = ks_2samp(feature_data, ref_data)
                
                if p_value < 0.01:
                    alert = {
                        'type': 'feature_drift',
                        'feature': f'feature_{col}',
                        'severity': 'medium',
                        'message': f'Feature {col} distribution drift (p={p_value:.4f})',
                        'timestamp': datetime.now()
                    }
                    self.alerts.append(alert)
        
        return feature_stats
    
    def monitor_performance(self, y_true, y_pred, metric='accuracy'):
        """Monitor model performance over time"""
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, average='weighted')
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, average='weighted')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        perf_record = {
            'timestamp': datetime.now(),
            'metric': metric,
            'score': score,
            'n_samples': len(y_true)
        }
        
        # Check for performance degradation
        if len(self.metrics_history) > 0:
            recent_scores = [m.get('score', 0) for m in self.metrics_history[-10:]]
            avg_recent = np.mean(recent_scores)
            
            if score < avg_recent * 0.95:  # 5% degradation
                alert = {
                    'type': 'performance_degradation',
                    'severity': 'high',
                    'message': f'Performance degradation: {score:.4f} < {avg_recent:.4f}',
                    'timestamp': datetime.now()
                }
                self.alerts.append(alert)
                print(f"ALERT: {alert['message']}")
        
        self.metrics_history.append(perf_record)
        
        return perf_record
    
    def generate_monitoring_report(self):
        """Generate monitoring report"""
        
        report = {
            'pipeline_name': self.pipeline_name,
            'report_time': datetime.now().isoformat(),
            'total_predictions': len(self.metrics_history),
            'alerts_summary': {
                'total': len(self.alerts),
                'by_type': {},
                'by_severity': {}
            },
            'performance_summary': {},
            'recent_alerts': self.alerts[-10:]  # Last 10 alerts
        }
        
        # Summarize alerts
        for alert in self.alerts:
            alert_type = alert['type']
            severity = alert['severity']
            
            report['alerts_summary']['by_type'][alert_type] = \
                report['alerts_summary']['by_type'].get(alert_type, 0) + 1
            
            report['alerts_summary']['by_severity'][severity] = \
                report['alerts_summary']['by_severity'].get(severity, 0) + 1
        
        # Performance summary
        if self.metrics_history:
            scores = [m.get('score', 0) for m in self.metrics_history if 'score' in m]
            if scores:
                report['performance_summary'] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'trend': 'declining' if scores[-1] < scores[0] else 'stable'
                }
        
        return report

### 8.2 A/B Testing Pipelines

class PipelineABTesting:
    """A/B testing framework for ML pipelines"""
    
    def __init__(self, pipeline_a, pipeline_b, test_name="ab_test"):
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
        self.test_name = test_name
        self.results = {
            'a': {'predictions': [], 'scores': []},
            'b': {'predictions': [], 'scores': []}
        }
        
    def run_test(self, X_test, y_test, n_iterations=100, test_size=0.1):
        """Run A/B test with statistical analysis"""
        
        from scipy import stats
        
        for i in range(n_iterations):
            # Sample test data
            sample_size = int(len(X_test) * test_size)
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            
            X_sample = X_test[indices]
            y_sample = y_test[indices]
            
            # Test both pipelines
            pred_a = self.pipeline_a.predict(X_sample)
            pred_b = self.pipeline_b.predict(X_sample)
            
            score_a = accuracy_score(y_sample, pred_a)
            score_b = accuracy_score(y_sample, pred_b)
            
            self.results['a']['scores'].append(score_a)
            self.results['b']['scores'].append(score_b)
        
        # Statistical analysis
        scores_a = np.array(self.results['a']['scores'])
        scores_b = np.array(self.results['b']['scores'])
        
        # T-test
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(scores_a)**2 + np.std(scores_b)**2) / 2)
        cohens_d = (np.mean(scores_b) - np.mean(scores_a)) / pooled_std
        
        # Results summary
        summary = {
            'test_name': self.test_name,
            'n_iterations': n_iterations,
            'pipeline_a_mean': np.mean(scores_a),
            'pipeline_a_std': np.std(scores_a),
            'pipeline_b_mean': np.mean(scores_b),
            'pipeline_b_std': np.std(scores_b),
            'difference': np.mean(scores_b) - np.mean(scores_a),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'winner': 'B' if np.mean(scores_b) > np.mean(scores_a) else 'A'
        }
        
        self._plot_results(scores_a, scores_b, summary)
        
        return summary
    
    def _plot_results(self, scores_a, scores_b, summary):
        """Visualize A/B test results"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution plot
        ax1.hist(scores_a, bins=20, alpha=0.5, label='Pipeline A', density=True)
        ax1.hist(scores_b, bins=20, alpha=0.5, label='Pipeline B', density=True)
        ax1.axvline(np.mean(scores_a), color='blue', linestyle='--', 
                   label=f'A mean: {np.mean(scores_a):.4f}')
        ax1.axvline(np.mean(scores_b), color='orange', linestyle='--', 
                   label=f'B mean: {np.mean(scores_b):.4f}')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distributions')
        ax1.legend()
        
        # Box plot
        ax2.boxplot([scores_a, scores_b], labels=['Pipeline A', 'Pipeline B'])
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Performance Comparison (p={summary["p_value"]:.4f})')
        
        if summary['significant']:
            ax2.text(1.5, np.max([scores_a.max(), scores_b.max()]) * 1.02,
                    f'Significant difference!\nEffect size: {summary["cohens_d"]:.3f}',
                    ha='center', fontweight='bold')
        
        plt.suptitle(f'A/B Test Results: {self.test_name}')
        plt.tight_layout()
        plt.show()
```

## 9. Case Studies

### 9.1 End-to-End Classification Pipeline

```python
def classification_pipeline_case_study():
    """Complete classification pipeline example"""
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {data.DESCR.split('.')[0]}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Build comprehensive pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    
    # Create feature preprocessing pipeline
    preprocessing = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=20))
    ])
    
    # Create multiple models with preprocessing
    rf_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    lr_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    svm_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Ensemble pipeline
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_pipeline),
            ('lr', lr_pipeline),
            ('svm', svm_pipeline)
        ],
        voting='soft'
    )
    
    # Train and evaluate
    print("\nTraining ensemble pipeline...")
    ensemble.fit(X_train, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    # Evaluation
    from sklearn.metrics import classification_report, roc_auc_score
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['malignant', 'benign']))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
    
    # Feature importance analysis
    # Get the random forest model from the ensemble
    rf_model = ensemble.estimators_[0].named_steps['rf']
    preprocessing_step = ensemble.estimators_[0].named_steps['preprocessing']
    selector = preprocessing_step.named_steps['feature_selection']
    
    # Get selected features
    selected_features = np.array(feature_names)[selector.get_support()]
    importances = rf_model.feature_importances_
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1][:10]
    plt.bar(range(10), importances[indices])
    plt.xticks(range(10), selected_features[indices], rotation=45, ha='right')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.show()
    
    return ensemble

### 9.2 Time Series Pipeline

def time_series_pipeline_case_study():
    """Time series forecasting pipeline"""
    
    # Generate synthetic time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Create time series with trend and seasonality
    trend = np.linspace(100, 200, 1000)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
    noise = np.random.normal(0, 5, 1000)
    
    values = trend + seasonal + noise
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    print("Time Series Data:")
    print(df.head())
    
    # Feature engineering for time series
    from sklearn.base import BaseEstimator, TransformerMixin
    
    class TimeSeriesFeaturizer(BaseEstimator, TransformerMixin):
        """Create time series features"""
        
        def __init__(self, lags=[1, 7, 30], rolling_windows=[7, 30]):
            self.lags = lags
            self.rolling_windows = rolling_windows
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            df = X.copy()
            
            # Lag features
            for lag in self.lags:
                df[f'lag_{lag}'] = df['value'].shift(lag)
            
            # Rolling statistics
            for window in self.rolling_windows:
                df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
                df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
            
            # Time features
            df['dayofweek'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['dayofyear'] = df['date'].dt.dayofyear
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Drop original date and value columns
            df = df.drop(['date', 'value'], axis=1)
            
            # Handle NaN from lagging
            df = df.fillna(method='bfill')
            
            return df
    
    # Create pipeline
    ts_pipeline = Pipeline([
        ('featurizer', TimeSeriesFeaturizer()),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Prepare data for supervised learning
    # Use past values to predict future
    lookback = 60  # Use 60 days of history
    forecast_horizon = 30  # Predict 30 days ahead
    
    X_list, y_list = [], []
    
    for i in range(lookback, len(df) - forecast_horizon):
        X_list.append(df.iloc[i-lookback:i])
        y_list.append(df.iloc[i+forecast_horizon]['value'])
    
    # Split data
    split_idx = int(len(X_list) * 0.8)
    X_train = X_list[:split_idx]
    X_test = X_list[split_idx:]
    y_train = y_list[:split_idx]
    y_test = y_list[split_idx:]
    
    # Train pipeline
    print(f"\nTraining on {len(X_train)} samples...")
    
    # Note: This is a simplified example. In practice, you'd need to handle
    # the sequential nature of the data more carefully
    
    # For demonstration, we'll use the last entry of each sequence
    X_train_last = pd.DataFrame([x.iloc[-1] for x in X_train])
    X_test_last = pd.DataFrame([x.iloc[-1] for x in X_test])
    
    ts_pipeline.fit(X_train_last, y_train)
    
    # Predictions
    y_pred = ts_pipeline.predict(X_test_last)
    
    # Evaluation
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Time Series Predictions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return ts_pipeline
```

## 10. Interview Questions

### Q1: What is an ML pipeline and why is it important?
**Answer**: An ML pipeline is an end-to-end workflow that automates the machine learning process from raw data to predictions. It includes:
- Data ingestion and validation
- Preprocessing and feature engineering
- Model training and evaluation
- Deployment and monitoring

**Importance**:
1. **Reproducibility**: Same results every time
2. **Automation**: Reduces manual errors
3. **Scalability**: Handle production workloads
4. **Maintainability**: Easy to update components
5. **Consistency**: Standardized process across projects

### Q2: What are the key components of a production ML pipeline?
**Answer**:
1. **Data Ingestion**: Collecting data from various sources
2. **Data Validation**: Checking data quality and schema
3. **Feature Engineering**: Creating and transforming features
4. **Model Training**: Training with hyperparameter tuning
5. **Model Validation**: Evaluating performance and fairness
6. **Model Registry**: Versioning and storing models
7. **Deployment**: Serving predictions
8. **Monitoring**: Tracking performance and drift
9. **Feedback Loop**: Continuous improvement

Each component should be modular, testable, and scalable.

### Q3: How do you handle data preprocessing in a pipeline?
**Answer**: Use sklearn's preprocessing tools:

```python
# Numeric features
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features  
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

**Best practices**:
- Fit preprocessors only on training data
- Handle unknown categories in production
- Save preprocessing parameters with model

### Q4: What is the difference between Pipeline and FeatureUnion in scikit-learn?
**Answer**:

**Pipeline**: Sequential execution
- Steps execute one after another
- Output of step n is input to step n+1
- Example: Scaler → PCA → Classifier

**FeatureUnion**: Parallel execution
- Transformers execute in parallel
- Outputs are concatenated
- Example: PCA || SelectKBest || PolynomialFeatures

```python
# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

# FeatureUnion
union = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('kbest', SelectKBest(k=5))
])
```

### Q5: How do you prevent data leakage in ML pipelines?
**Answer**:

1. **Proper splitting**: Split before any preprocessing
2. **Fit only on train**: Never fit transformers on test data
3. **Cross-validation**: Include preprocessing in CV loop
4. **Target encoding**: Use out-of-fold encoding
5. **Time series**: Respect temporal order

```python
# Wrong
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# Correct
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Q6: How do you optimize hyperparameters in a pipeline?
**Answer**: Use GridSearchCV or RandomizedSearchCV with pipeline parameters:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC())
])

param_grid = {
    'pca__n_components': [5, 10, 20],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**Tips**:
- Use pipeline__parameter notation
- Start with RandomizedSearchCV for many parameters
- Consider Bayesian optimization for expensive models

### Q7: How do you handle custom transformers in scikit-learn pipelines?
**Answer**: Inherit from BaseEstimator and TransformerMixin:

```python
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param
    
    def fit(self, X, y=None):
        # Learn parameters from X
        self.mean_ = X.mean()
        return self
    
    def transform(self, X):
        # Apply transformation
        return X - self.mean_
```

**Requirements**:
- Implement fit() and transform()
- Return self from fit()
- Store learned parameters with trailing _
- Use check_array() for input validation

### Q8: What are best practices for ML pipeline versioning?
**Answer**:

1. **Code versioning**: Git for pipeline code
2. **Data versioning**: DVC, Delta Lake
3. **Model versioning**: MLflow, model registry
4. **Configuration**: Version control configs
5. **Environment**: Docker, requirements.txt

```python
metadata = {
    'model_version': '1.2.3',
    'pipeline_version': '2.0.0',
    'training_date': '2024-01-01',
    'git_commit': 'abc123',
    'data_version': 'v3',
    'metrics': {...}
}
```

### Q9: How do you monitor ML pipelines in production?
**Answer**: Monitor multiple aspects:

1. **Data quality**:
   - Schema validation
   - Distribution drift
   - Missing values
   - Outliers

2. **Model performance**:
   - Accuracy metrics
   - Latency
   - Throughput
   - Error rates

3. **System health**:
   - CPU/memory usage
   - API response times
   - Error logs

4. **Business metrics**:
   - Downstream impact
   - User engagement

Tools: Prometheus, Grafana, custom dashboards

### Q10: How do you handle streaming data in ML pipelines?
**Answer**:

1. **Online learning**: Update model incrementally
2. **Micro-batching**: Process small batches
3. **Window-based**: Sliding/tumbling windows
4. **Feature stores**: Real-time feature serving

```python
class StreamingPipeline:
    def __init__(self, model, window_size=100):
        self.model = model
        self.buffer = []
        self.window_size = window_size
    
    def process(self, data_point):
        # Add to buffer
        self.buffer.append(data_point)
        
        # Process when window full
        if len(self.buffer) >= self.window_size:
            batch = np.array(self.buffer)
            predictions = self.model.predict(batch)
            self.buffer = []
            return predictions
```

### Q11: What is the difference between batch and real-time pipelines?
**Answer**:

**Batch pipelines**:
- Process data in large batches
- Higher latency (hours/days)
- More complex transformations
- Example: Daily model retraining

**Real-time pipelines**:
- Process data as it arrives
- Low latency (ms/seconds)
- Simpler transformations
- Example: Fraud detection

**Hybrid approach**: Lambda architecture combining both

### Q12: How do you ensure pipeline reproducibility?
**Answer**:

1. **Random seeds**: Set everywhere
2. **Data snapshots**: Version datasets
3. **Environment**: Docker/conda
4. **Code version**: Git commits
5. **Configuration**: Track all parameters

```python
# Reproducibility checklist
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Save environment
pip freeze > requirements.txt

# Track experiments
mlflow.log_param("random_state", 42)
mlflow.log_metric("accuracy", 0.95)
```

### Q13: How do you debug ML pipelines?
**Answer**:

1. **Logging**: Comprehensive logging at each step
2. **Assertions**: Validate assumptions
3. **Unit tests**: Test individual components
4. **Integration tests**: Test full pipeline
5. **Data validation**: Check intermediate outputs

```python
class DebuggablePipeline:
    def transform(self, X):
        logger.info(f"Input shape: {X.shape}")
        
        # Validate input
        assert X.shape[1] == self.n_features_
        
        # Transform
        X_transformed = self._transform(X)
        
        # Validate output
        assert not np.isnan(X_transformed).any()
        
        logger.info(f"Output shape: {X_transformed.shape}")
        return X_transformed
```

### Q14: How do you handle model deployment in pipelines?
**Answer**:

1. **Containerization**: Docker for consistency
2. **API endpoints**: REST/gRPC
3. **Model serving**: TensorFlow Serving, Seldon
4. **Load balancing**: Handle traffic
5. **A/B testing**: Gradual rollout

```python
# Simple Flask deployment
from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = joblib.load('pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = pd.DataFrame(data)
    predictions = pipeline.predict(X)
    return jsonify(predictions.tolist())
```

### Q15: What are common pitfalls in ML pipeline design?
**Answer**:

1. **Data leakage**: Information from test in training
2. **Training-serving skew**: Different preprocessing
3. **Silent failures**: No error but wrong results
4. **Performance degradation**: Slow pipelines
5. **Memory issues**: Large intermediate results
6. **Version mismatches**: Different library versions
7. **Missing error handling**: Crashes in production

**Prevention**:
- Comprehensive testing
- Monitoring and alerting
- Regular audits
- Documentation

### Q16: How do you scale ML pipelines?
**Answer**:

1. **Horizontal scaling**: Distribute across machines
2. **Vertical scaling**: Bigger machines
3. **Parallelization**: Process in parallel
4. **Caching**: Store intermediate results
5. **Batch processing**: Group predictions
6. **Feature stores**: Precompute features

```python
# Parallel processing example
from joblib import Parallel, delayed

def process_batch(batch, pipeline):
    return pipeline.transform(batch)

results = Parallel(n_jobs=-1)(
    delayed(process_batch)(batch, pipeline) 
    for batch in data_batches
)
```

### Q17: What is MLOps and how does it relate to pipelines?
**Answer**: MLOps applies DevOps principles to ML:

**Components**:
1. **CI/CD**: Automated testing and deployment
2. **Monitoring**: Performance tracking
3. **Versioning**: Models, data, code
4. **Experimentation**: A/B testing
5. **Governance**: Compliance, fairness

**Pipeline's role**:
- Standardizes ML workflow
- Enables automation
- Facilitates monitoring
- Supports versioning

### Q18: How do you handle feature drift in production pipelines?
**Answer**:

1. **Detection**:
   - Statistical tests (KS, Chi-square)
   - Distribution monitoring
   - Feature importance changes

2. **Response**:
   - Alert stakeholders
   - Retrain model
   - Update features
   - Rollback if needed

```python
def detect_drift(reference_data, current_data):
    from scipy.stats import ks_2samp
    
    drift_detected = False
    for col in reference_data.columns:
        stat, p_value = ks_2samp(
            reference_data[col], 
            current_data[col]
        )
        if p_value < 0.05:
            drift_detected = True
            log_alert(f"Drift in {col}: p={p_value}")
    
    return drift_detected
```

### Q19: How do you implement pipeline testing?
**Answer**: Multiple levels of testing:

1. **Unit tests**: Individual components
```python
def test_custom_transformer():
    transformer = CustomTransformer()
    X = np.array([[1, 2], [3, 4]])
    X_transformed = transformer.fit_transform(X)
    assert X_transformed.shape == X.shape
```

2. **Integration tests**: Full pipeline
```python
def test_pipeline():
    pipeline = create_pipeline()
    X, y = make_classification(n_samples=100)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)
```

3. **Performance tests**: Speed and resource usage
4. **Data validation tests**: Schema and quality

### Q20: How do you design pipelines for different model types?
**Answer**: Adapt pipeline to model requirements:

**Tree-based models**:
- No scaling needed
- Can handle missing values
- Feature engineering important

**Linear models**:
- Scaling crucial
- Handle multicollinearity
- Polynomial features helpful

**Neural networks**:
- Normalization important
- Batch processing
- GPU optimization

**Example**:
```python
def create_model_specific_pipeline(model_type):
    if model_type == 'tree':
        return Pipeline([
            ('imputer', SimpleImputer()),
            ('model', RandomForestClassifier())
        ])
    elif model_type == 'linear':
        return Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])
```

### Q21: What are pipeline design patterns?
**Answer**: Common patterns include:

1. **Sequential**: Step-by-step processing
2. **Parallel**: Feature union
3. **Conditional**: Different paths based on data
4. **Ensemble**: Multiple models combined
5. **Hierarchical**: Nested pipelines
6. **Feedback**: Results influence next iteration

Each pattern suits different use cases and requirements.

### Q22: How do you optimize pipeline performance?
**Answer**:

1. **Profiling**: Identify bottlenecks
2. **Caching**: Store intermediate results
3. **Parallelization**: Use multiple cores
4. **Vectorization**: Numpy operations
5. **Feature selection**: Reduce dimensionality
6. **Approximate methods**: Trade accuracy for speed

```python
# Example: Caching expensive operations
from joblib import Memory

memory = Memory('./cache', verbose=0)

@memory.cache
def expensive_transform(X):
    # Expensive computation
    return result

pipeline = Pipeline([
    ('cached_transform', FunctionTransformer(expensive_transform)),
    ('model', RandomForestClassifier())
])
```