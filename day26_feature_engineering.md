# Day 26: Feature Engineering

## Table of Contents
1. [Introduction](#1-introduction)
2. [Understanding Features](#2-understanding-features)
3. [Feature Creation Techniques](#3-feature-creation-techniques)
4. [Feature Transformation](#4-feature-transformation)
5. [Feature Selection](#5-feature-selection)
6. [Automated Feature Engineering](#6-automated-feature-engineering)
7. [Domain-Specific Feature Engineering](#7-domain-specific-feature-engineering)
8. [Advanced Techniques](#8-advanced-techniques)
9. [Best Practices](#9-best-practices)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

Feature engineering is the process of creating new features or transforming existing ones to improve machine learning model performance. It's often said that "feature engineering is the key to success in machine learning" because the right features can make simple models perform exceptionally well.

### Why Feature Engineering Matters

1. **Model Performance**: Good features can dramatically improve accuracy
2. **Simplicity**: Better features allow simpler models
3. **Interpretability**: Well-designed features are easier to understand
4. **Domain Knowledge**: Incorporates expert understanding
5. **Data Efficiency**: Can reduce the amount of data needed

### The Feature Engineering Process

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif

class FeatureEngineeringPipeline:
    """Complete feature engineering workflow"""
    
    def __init__(self):
        self.feature_history = []
        self.transformations = {}
        
    def analyze_features(self, df, target_col):
        """Initial feature analysis"""
        print("=== Feature Analysis ===")
        print(f"Dataset shape: {df.shape}")
        print(f"\nFeature types:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\nMissing values:")
            print(missing[missing > 0])
        
        # Cardinality for categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical feature cardinality:")
            for col in categorical_cols:
                print(f"  {col}: {df[col].nunique()} unique values")
        
        # Basic statistics
        print(f"\nNumerical features summary:")
        print(df.describe())
        
        # Target distribution
        if target_col in df.columns:
            print(f"\nTarget distribution:")
            print(df[target_col].value_counts())
        
        return self._generate_feature_report(df, target_col)
    
    def _generate_feature_report(self, df, target_col):
        """Generate detailed feature report"""
        report = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'high_cardinality': [],
            'low_variance': [],
            'highly_correlated': []
        }
        
        # Classify features
        for col in df.columns:
            if col == target_col:
                continue
                
            dtype = df[col].dtype
            nunique = df[col].nunique()
            
            if dtype in ['int64', 'float64']:
                report['numerical'].append(col)
                
                # Check for low variance
                if df[col].std() < 0.01:
                    report['low_variance'].append(col)
                    
            elif dtype == 'object':
                report['categorical'].append(col)
                
                # Check for high cardinality
                if nunique > 50:
                    report['high_cardinality'].append(col)
                    
            elif 'datetime' in str(dtype):
                report['datetime'].append(col)
        
        # Check correlations
        if len(report['numerical']) > 1:
            corr_matrix = df[report['numerical']].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )
            
            report['highly_correlated'] = high_corr_pairs
        
        return report
```

## 2. Understanding Features

### 2.1 Types of Features

```python
class FeatureTypes:
    """Understanding different types of features"""
    
    @staticmethod
    def demonstrate_feature_types():
        """Show examples of different feature types"""
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            # Numerical features
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'temperature': np.random.normal(20, 5, n_samples),
            
            # Categorical features
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
            'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n_samples),
            
            # Binary features
            'has_car': np.random.choice([0, 1], n_samples),
            'is_employed': np.random.choice([True, False], n_samples),
            
            # Ordinal features
            'satisfaction': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], n_samples),
            
            # Date/Time features
            'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            
            # Text features
            'comment': ['Good product'] * (n_samples // 2) + ['Bad product'] * (n_samples // 2),
            
            # Composite features
            'address': [f"{np.random.randint(1, 999)} Main St, City{np.random.randint(1, 10)}" 
                       for _ in range(n_samples)]
        }
        
        df = pd.DataFrame(data)
        
        # Analyze each type
        feature_analysis = {
            'Numerical': {
                'examples': ['age', 'income', 'temperature'],
                'properties': 'Continuous or discrete values, can be used directly in models',
                'transformations': ['Scaling', 'Binning', 'Log transform', 'Polynomial']
            },
            'Categorical': {
                'examples': ['city', 'education'],
                'properties': 'Discrete values without inherent order',
                'transformations': ['One-hot encoding', 'Target encoding', 'Embeddings']
            },
            'Binary': {
                'examples': ['has_car', 'is_employed'],
                'properties': 'Two possible values',
                'transformations': ['Direct use', 'Interaction with other features']
            },
            'Ordinal': {
                'examples': ['satisfaction'],
                'properties': 'Discrete values with inherent order',
                'transformations': ['Label encoding', 'Target encoding', 'Binning']
            },
            'DateTime': {
                'examples': ['signup_date'],
                'properties': 'Temporal information',
                'transformations': ['Extract components', 'Cyclical encoding', 'Time differences']
            },
            'Text': {
                'examples': ['comment'],
                'properties': 'Unstructured text data',
                'transformations': ['Bag of words', 'TF-IDF', 'Embeddings', 'Length features']
            }
        }
        
        return df, feature_analysis
    
    @staticmethod
    def visualize_feature_relationships(df, target_col):
        """Visualize relationships between features and target"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Numerical vs target
        ax = axes[0, 0]
        numerical_col = df.select_dtypes(include=[np.number]).columns[0]
        if df[target_col].dtype in ['int64', 'float64']:
            ax.scatter(df[numerical_col], df[target_col], alpha=0.5)
            ax.set_xlabel(numerical_col)
            ax.set_ylabel(target_col)
        else:
            df.boxplot(column=numerical_col, by=target_col, ax=ax)
        ax.set_title('Numerical Feature vs Target')
        
        # Categorical vs target
        ax = axes[0, 1]
        categorical_col = df.select_dtypes(include=['object']).columns[0]
        if df[target_col].dtype == 'object':
            pd.crosstab(df[categorical_col], df[target_col]).plot(kind='bar', ax=ax)
        else:
            df.groupby(categorical_col)[target_col].mean().plot(kind='bar', ax=ax)
        ax.set_title('Categorical Feature vs Target')
        
        # Feature correlation heatmap
        ax = axes[1, 0]
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:5]
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlations')
        
        # Feature distributions
        ax = axes[1, 1]
        df[numerical_col].hist(bins=30, ax=ax)
        ax.set_xlabel(numerical_col)
        ax.set_title('Feature Distribution')
        
        plt.tight_layout()
        plt.show()
```

### 2.2 Feature Quality Metrics

```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import spearmanr

class FeatureQualityAnalyzer:
    """Analyze feature quality and importance"""
    
    def __init__(self, task='classification'):
        self.task = task
        self.feature_scores = {}
        
    def calculate_feature_importance(self, X, y, feature_names):
        """Calculate various feature importance metrics"""
        
        importance_metrics = {}
        
        # 1. Mutual Information
        if self.task == 'classification':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        importance_metrics['mutual_info'] = dict(zip(feature_names, mi_scores))
        
        # 2. Correlation (for numerical targets)
        if y.dtype in ['int64', 'float64']:
            correlations = {}
            for i, feature in enumerate(feature_names):
                corr, _ = spearmanr(X[:, i], y)
                correlations[feature] = abs(corr)
            importance_metrics['correlation'] = correlations
        
        # 3. Variance
        variances = {}
        for i, feature in enumerate(feature_names):
            variances[feature] = np.var(X[:, i])
        importance_metrics['variance'] = variances
        
        # 4. Unique value ratio
        unique_ratios = {}
        for i, feature in enumerate(feature_names):
            unique_ratio = len(np.unique(X[:, i])) / len(X[:, i])
            unique_ratios[feature] = unique_ratio
        importance_metrics['unique_ratio'] = unique_ratios
        
        return importance_metrics
    
    def visualize_importance(self, importance_metrics, top_n=20):
        """Visualize feature importance metrics"""
        
        n_metrics = len(importance_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, scores) in enumerate(importance_metrics.items()):
            # Sort and get top features
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, values = zip(*sorted_features)
            
            ax = axes[idx]
            ax.barh(range(len(features)), values)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'Top {top_n} Features by {metric_name}')
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(v, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
        
        return importance_metrics
```

## 3. Feature Creation Techniques

### 3.1 Mathematical Transformations

```python
class MathematicalTransformations:
    """Mathematical feature transformations"""
    
    @staticmethod
    def apply_transformations(df, numerical_cols):
        """Apply various mathematical transformations"""
        
        transformed_features = pd.DataFrame()
        
        for col in numerical_cols:
            # Original
            transformed_features[col] = df[col]
            
            # Log transform (handling zeros)
            if (df[col] > 0).all():
                transformed_features[f'{col}_log'] = np.log(df[col])
            else:
                transformed_features[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1)
            
            # Square root
            if (df[col] >= 0).all():
                transformed_features[f'{col}_sqrt'] = np.sqrt(df[col])
            
            # Square
            transformed_features[f'{col}_squared'] = df[col] ** 2
            
            # Reciprocal (avoiding division by zero)
            with np.errstate(divide='ignore'):
                reciprocal = 1 / df[col].replace(0, np.nan)
            transformed_features[f'{col}_reciprocal'] = reciprocal.fillna(0)
            
            # Exponential
            if df[col].max() < 10:  # Avoid overflow
                transformed_features[f'{col}_exp'] = np.exp(df[col])
            
            # Binning
            transformed_features[f'{col}_binned'] = pd.qcut(
                df[col], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                duplicates='drop'
            )
            
        return transformed_features
    
    @staticmethod
    def create_polynomial_features(df, numerical_cols, degree=2, include_bias=False):
        """Create polynomial features"""
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[numerical_cols])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Create DataFrame
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Remove original features (they're included in polynomial)
        poly_df = poly_df.drop(columns=numerical_cols)
        
        return poly_df
    
    @staticmethod
    def create_interaction_features(df, cols1, cols2=None):
        """Create interaction features between columns"""
        
        if cols2 is None:
            cols2 = cols1
        
        interaction_features = pd.DataFrame()
        
        for col1 in cols1:
            for col2 in cols2:
                if col1 != col2:
                    # Multiplication
                    interaction_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    
                    # Addition
                    interaction_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    
                    # Subtraction
                    interaction_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    
                    # Division (avoiding division by zero)
                    with np.errstate(divide='ignore'):
                        ratio = df[col1] / df[col2].replace(0, np.nan)
                    interaction_features[f'{col1}_div_{col2}'] = ratio.fillna(0)
        
        return interaction_features
```

### 3.2 Domain-Based Feature Creation

```python
class DomainFeatureCreator:
    """Create domain-specific features"""
    
    @staticmethod
    def create_temporal_features(df, date_col):
        """Extract features from datetime columns"""
        
        temporal_features = pd.DataFrame()
        dates = pd.to_datetime(df[date_col])
        
        # Basic components
        temporal_features['year'] = dates.dt.year
        temporal_features['month'] = dates.dt.month
        temporal_features['day'] = dates.dt.day
        temporal_features['dayofweek'] = dates.dt.dayofweek
        temporal_features['dayofyear'] = dates.dt.dayofyear
        temporal_features['weekofyear'] = dates.dt.isocalendar().week
        temporal_features['quarter'] = dates.dt.quarter
        
        # Time-based (if datetime)
        if dates.dt.hour.notna().any():
            temporal_features['hour'] = dates.dt.hour
            temporal_features['minute'] = dates.dt.minute
            temporal_features['is_morning'] = (dates.dt.hour >= 6) & (dates.dt.hour < 12)
            temporal_features['is_afternoon'] = (dates.dt.hour >= 12) & (dates.dt.hour < 18)
            temporal_features['is_evening'] = (dates.dt.hour >= 18) & (dates.dt.hour < 24)
            temporal_features['is_night'] = (dates.dt.hour >= 0) & (dates.dt.hour < 6)
        
        # Cyclical encoding
        temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['day_sin'] = np.sin(2 * np.pi * temporal_features['day'] / 31)
        temporal_features['day_cos'] = np.cos(2 * np.pi * temporal_features['day'] / 31)
        
        # Special dates
        temporal_features['is_weekend'] = temporal_features['dayofweek'].isin([5, 6])
        temporal_features['is_month_start'] = temporal_features['day'] == 1
        temporal_features['is_month_end'] = dates.dt.is_month_end
        temporal_features['is_quarter_start'] = dates.dt.is_quarter_start
        temporal_features['is_quarter_end'] = dates.dt.is_quarter_end
        temporal_features['is_year_start'] = dates.dt.is_year_start
        temporal_features['is_year_end'] = dates.dt.is_year_end
        
        # Days since/until
        if len(dates) > 1:
            temporal_features['days_since_start'] = (dates - dates.min()).dt.days
            temporal_features['days_until_end'] = (dates.max() - dates).dt.days
        
        return temporal_features
    
    @staticmethod
    def create_text_features(df, text_col):
        """Extract features from text columns"""
        
        text_features = pd.DataFrame()
        texts = df[text_col].fillna('')
        
        # Basic statistics
        text_features['length'] = texts.str.len()
        text_features['word_count'] = texts.str.split().str.len()
        text_features['unique_word_count'] = texts.apply(lambda x: len(set(x.split())))
        text_features['char_count'] = texts.str.replace(' ', '').str.len()
        
        # Character-based
        text_features['uppercase_count'] = texts.str.count('[A-Z]')
        text_features['lowercase_count'] = texts.str.count('[a-z]')
        text_features['digit_count'] = texts.str.count('[0-9]')
        text_features['space_count'] = texts.str.count(' ')
        text_features['punctuation_count'] = texts.str.count('[!"\#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]')
        
        # Ratios
        text_features['uppercase_ratio'] = text_features['uppercase_count'] / (text_features['length'] + 1)
        text_features['digit_ratio'] = text_features['digit_count'] / (text_features['length'] + 1)
        text_features['punctuation_ratio'] = text_features['punctuation_count'] / (text_features['length'] + 1)
        
        # Average word length
        text_features['avg_word_length'] = text_features['char_count'] / (text_features['word_count'] + 1)
        
        # Sentiment indicators (simple)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing']
        
        text_features['positive_word_count'] = texts.str.lower().apply(
            lambda x: sum(word in x for word in positive_words)
        )
        text_features['negative_word_count'] = texts.str.lower().apply(
            lambda x: sum(word in x for word in negative_words)
        )
        
        # Questions and exclamations
        text_features['question_count'] = texts.str.count('\?')
        text_features['exclamation_count'] = texts.str.count('!')
        
        return text_features
    
    @staticmethod
    def create_geographical_features(df, lat_col, lon_col, reference_points=None):
        """Create features from geographical coordinates"""
        
        geo_features = pd.DataFrame()
        
        # Basic coordinates
        geo_features['latitude'] = df[lat_col]
        geo_features['longitude'] = df[lon_col]
        
        # Distances from reference points
        if reference_points:
            for name, (ref_lat, ref_lon) in reference_points.items():
                # Haversine distance
                geo_features[f'distance_to_{name}'] = haversine_distance(
                    df[lat_col], df[lon_col], ref_lat, ref_lon
                )
        
        # Clustering-based features
        from sklearn.cluster import KMeans
        coords = df[[lat_col, lon_col]].values
        
        # Cluster assignment
        kmeans = KMeans(n_clusters=5, random_state=42)
        geo_features['geo_cluster'] = kmeans.fit_predict(coords)
        
        # Distance to cluster center
        distances = kmeans.transform(coords)
        geo_features['distance_to_cluster_center'] = distances.min(axis=1)
        
        # Density features (points within radius)
        # This is computationally expensive for large datasets
        if len(df) < 10000:
            from scipy.spatial.distance import cdist
            distances_matrix = cdist(coords, coords, metric='euclidean')
            
            for radius in [0.01, 0.1, 1.0]:  # Degrees
                geo_features[f'points_within_{radius}deg'] = (distances_matrix < radius).sum(axis=1)
        
        return geo_features

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between coordinates"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c
```

### 3.3 Statistical Features

```python
class StatisticalFeatureCreator:
    """Create statistical features"""
    
    @staticmethod
    def create_aggregation_features(df, group_cols, agg_cols, operations=['mean', 'std', 'min', 'max']):
        """Create group-based aggregation features"""
        
        agg_features = pd.DataFrame(index=df.index)
        
        for group_col in group_cols:
            for agg_col in agg_cols:
                for operation in operations:
                    # Calculate group statistics
                    group_stats = df.groupby(group_col)[agg_col].agg(operation)
                    
                    # Map back to original dataframe
                    feature_name = f'{agg_col}_by_{group_col}_{operation}'
                    agg_features[feature_name] = df[group_col].map(group_stats)
                    
                    # Difference from group statistic
                    if operation in ['mean', 'median']:
                        diff_name = f'{agg_col}_diff_from_{group_col}_{operation}'
                        agg_features[diff_name] = df[agg_col] - agg_features[feature_name]
        
        return agg_features
    
    @staticmethod
    def create_rolling_features(df, columns, windows=[7, 30], operations=['mean', 'std', 'min', 'max']):
        """Create rolling window features (for time series)"""
        
        rolling_features = pd.DataFrame(index=df.index)
        
        for col in columns:
            for window in windows:
                for operation in operations:
                    feature_name = f'{col}_rolling_{window}_{operation}'
                    
                    if operation == 'mean':
                        rolling_features[feature_name] = df[col].rolling(window).mean()
                    elif operation == 'std':
                        rolling_features[feature_name] = df[col].rolling(window).std()
                    elif operation == 'min':
                        rolling_features[feature_name] = df[col].rolling(window).min()
                    elif operation == 'max':
                        rolling_features[feature_name] = df[col].rolling(window).max()
                    elif operation == 'sum':
                        rolling_features[feature_name] = df[col].rolling(window).sum()
        
        return rolling_features
    
    @staticmethod
    def create_lag_features(df, columns, lags=[1, 7, 30]):
        """Create lag features (for time series)"""
        
        lag_features = pd.DataFrame(index=df.index)
        
        for col in columns:
            for lag in lags:
                lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Difference from lag
                lag_features[f'{col}_diff_lag_{lag}'] = df[col] - df[col].shift(lag)
                
                # Ratio to lag (avoiding division by zero)
                with np.errstate(divide='ignore'):
                    ratio = df[col] / df[col].shift(lag).replace(0, np.nan)
                lag_features[f'{col}_ratio_lag_{lag}'] = ratio.fillna(1)
        
        return lag_features
```

## 4. Feature Transformation

### 4.1 Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class CategoricalEncoder:
    """Advanced categorical encoding techniques"""
    
    def __init__(self):
        self.encoders = {}
        
    def one_hot_encode(self, df, columns, drop_first=True, handle_unknown='ignore'):
        """One-hot encoding"""
        
        encoded_dfs = []
        
        for col in columns:
            # Create encoder
            encoder = OneHotEncoder(drop='first' if drop_first else None,
                                  handle_unknown=handle_unknown,
                                  sparse_output=False)
            
            # Fit and transform
            encoded = encoder.fit_transform(df[[col]])
            
            # Create column names
            if drop_first:
                feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0][1:]]
            else:
                feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0]]
            
            # Create DataFrame
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            encoded_dfs.append(encoded_df)
            
            # Store encoder
            self.encoders[col] = encoder
        
        return pd.concat(encoded_dfs, axis=1)
    
    def target_encode(self, df, columns, target, smoothing=1.0):
        """Target encoding with smoothing"""
        
        encoded_df = pd.DataFrame(index=df.index)
        
        for col in columns:
            # Calculate global mean
            global_mean = target.mean()
            
            # Calculate category statistics
            category_stats = df.groupby(col)[target.name].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_means = (
                (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
                (category_stats['count'] + smoothing)
            )
            
            # Map to original dataframe
            encoded_df[f'{col}_target_encoded'] = df[col].map(smoothed_means)
            
            # Store mapping
            self.encoders[f'{col}_target'] = smoothed_means.to_dict()
        
        return encoded_df
    
    def frequency_encode(self, df, columns):
        """Frequency encoding"""
        
        encoded_df = pd.DataFrame(index=df.index)
        
        for col in columns:
            # Calculate frequencies
            frequencies = df[col].value_counts(normalize=True)
            
            # Map to original dataframe
            encoded_df[f'{col}_frequency'] = df[col].map(frequencies)
            
            # Store mapping
            self.encoders[f'{col}_frequency'] = frequencies.to_dict()
        
        return encoded_df
    
    def ordinal_encode(self, df, columns, ordinal_mappings=None):
        """Ordinal encoding with custom ordering"""
        
        encoded_df = pd.DataFrame(index=df.index)
        
        for col in columns:
            if ordinal_mappings and col in ordinal_mappings:
                # Use custom mapping
                mapping = ordinal_mappings[col]
                encoded_df[f'{col}_ordinal'] = df[col].map(mapping)
            else:
                # Use default ordering
                encoder = LabelEncoder()
                encoded_df[f'{col}_ordinal'] = encoder.fit_transform(df[col])
                self.encoders[f'{col}_ordinal'] = encoder
        
        return encoded_df
    
    def binary_encode(self, df, columns):
        """Binary encoding for high cardinality features"""
        
        import category_encoders as ce
        
        encoder = ce.BinaryEncoder(cols=columns)
        encoded_df = encoder.fit_transform(df[columns])
        
        # Rename columns
        encoded_df.columns = [f'{col}_binary' for col in encoded_df.columns]
        
        self.encoders['binary'] = encoder
        
        return encoded_df
```

### 4.2 Scaling and Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

class FeatureScaler:
    """Advanced feature scaling techniques"""
    
    def __init__(self):
        self.scalers = {}
        
    def apply_scaling(self, df, numerical_cols, method='standard'):
        """Apply various scaling methods"""
        
        scaled_features = pd.DataFrame(index=df.index)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        scaled_data = scaler.fit_transform(df[numerical_cols])
        
        # Create DataFrame
        for i, col in enumerate(numerical_cols):
            scaled_features[f'{col}_{method}_scaled'] = scaled_data[:, i]
        
        # Store scaler
        self.scalers[method] = scaler
        
        return scaled_features
    
    def compare_scaling_methods(self, df, numerical_cols):
        """Compare different scaling methods visually"""
        
        methods = ['standard', 'minmax', 'robust', 'quantile']
        n_methods = len(methods)
        n_features = min(3, len(numerical_cols))  # Show max 3 features
        
        fig, axes = plt.subplots(n_features, n_methods + 1, figsize=(15, 4*n_features))
        
        for i, col in enumerate(numerical_cols[:n_features]):
            # Original distribution
            ax = axes[i, 0] if n_features > 1 else axes[0]
            df[col].hist(bins=30, ax=ax)
            ax.set_title(f'{col} - Original')
            ax.set_ylabel('Frequency')
            
            # Scaled distributions
            for j, method in enumerate(methods):
                ax = axes[i, j+1] if n_features > 1 else axes[j+1]
                
                scaled_df = self.apply_scaling(df, [col], method)
                scaled_df.iloc[:, 0].hist(bins=30, ax=ax)
                ax.set_title(f'{col} - {method.capitalize()}')
        
        plt.tight_layout()
        plt.show()
```

## 5. Feature Selection

### 5.1 Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

class FilterFeatureSelector:
    """Filter-based feature selection methods"""
    
    def __init__(self, task='classification'):
        self.task = task
        self.selectors = {}
        
    def variance_threshold_selection(self, X, threshold=0.01):
        """Remove low variance features"""
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        selected_features = selector.get_support()
        self.selectors['variance'] = selector
        
        return X_selected, selected_features
    
    def univariate_selection(self, X, y, k=10, method='f_classif'):
        """Select k best features using univariate tests"""
        
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'chi2':
            score_func = chi2
        elif method == 'mutual_info':
            score_func = mutual_info_classif if self.task == 'classification' else mutual_info_regression
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get scores and selected features
        scores = selector.scores_
        selected_features = selector.get_support()
        
        self.selectors[method] = selector
        
        return X_selected, selected_features, scores
    
    def correlation_threshold_selection(self, df, threshold=0.95):
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Select upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Return features to keep
        features_to_keep = [col for col in df.columns if col not in to_drop]
        
        return features_to_keep, to_drop
```

### 5.2 Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class WrapperFeatureSelector:
    """Wrapper-based feature selection methods"""
    
    def __init__(self, estimator=None):
        self.estimator = estimator or RandomForestClassifier(n_estimators=100, random_state=42)
        self.selectors = {}
        
    def recursive_feature_elimination(self, X, y, n_features_to_select=10):
        """Recursive Feature Elimination"""
        
        selector = RFE(estimator=self.estimator, 
                      n_features_to_select=n_features_to_select)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get rankings
        rankings = selector.ranking_
        selected_features = selector.support_
        
        self.selectors['rfe'] = selector
        
        return X_selected, selected_features, rankings
    
    def recursive_feature_elimination_cv(self, X, y, cv=5):
        """RFE with cross-validation"""
        
        selector = RFECV(estimator=self.estimator, cv=cv, scoring='accuracy')
        X_selected = selector.fit_transform(X, y)
        
        # Get results
        n_features = selector.n_features_
        rankings = selector.ranking_
        cv_scores = selector.cv_results_['mean_test_score']
        
        # Plot CV scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'b-')
        plt.axvline(x=n_features, color='r', linestyle='--', 
                   label=f'Optimal: {n_features} features')
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation Score')
        plt.title('RFECV: Feature Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        self.selectors['rfecv'] = selector
        
        return X_selected, selector.support_, rankings
    
    def forward_selection(self, X, y, max_features=10):
        """Forward feature selection"""
        
        n_features = X.shape[1]
        selected_features = []
        remaining_features = list(range(n_features))
        
        scores_history = []
        
        for i in range(min(max_features, n_features)):
            best_score = -np.inf
            best_feature = None
            
            # Try each remaining feature
            for feature in remaining_features:
                # Current feature set
                current_features = selected_features + [feature]
                
                # Evaluate
                X_subset = X[:, current_features]
                scores = cross_val_score(self.estimator, X_subset, y, cv=5)
                score = scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            # Add best feature
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            scores_history.append(best_score)
            
            print(f"Step {i+1}: Added feature {best_feature}, Score: {best_score:.4f}")
        
        return selected_features, scores_history
```

### 5.3 Embedded Methods

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class EmbeddedFeatureSelector:
    """Embedded feature selection methods"""
    
    def __init__(self):
        self.selectors = {}
        self.importances = {}
        
    def lasso_selection(self, X, y, alpha=0.01):
        """L1 regularization for feature selection"""
        
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        
        # Get coefficients
        coefficients = lasso.coef_
        selected_features = coefficients != 0
        
        self.selectors['lasso'] = lasso
        self.importances['lasso'] = coefficients
        
        return selected_features, coefficients
    
    def tree_based_selection(self, X, y, model_type='random_forest', threshold='median'):
        """Tree-based feature importance"""
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        model.fit(X, y)
        
        # Get importances
        importances = model.feature_importances_
        
        # Select features
        if threshold == 'median':
            threshold_value = np.median(importances)
        elif threshold == 'mean':
            threshold_value = np.mean(importances)
        else:
            threshold_value = threshold
        
        selected_features = importances > threshold_value
        
        self.selectors[model_type] = model
        self.importances[model_type] = importances
        
        return selected_features, importances
    
    def plot_feature_importances(self, feature_names, top_n=20):
        """Plot feature importances from different methods"""
        
        n_methods = len(self.importances)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, importances) in enumerate(self.importances.items()):
            # Sort features by importance
            indices = np.argsort(np.abs(importances))[::-1][:top_n]
            
            ax = axes[idx]
            ax.barh(range(top_n), np.abs(importances[indices]))
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{method.capitalize()} Feature Importance')
            
            # Add value labels
            for i, v in enumerate(np.abs(importances[indices])):
                ax.text(v, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
```

## 6. Automated Feature Engineering

### 6.1 Featuretools Implementation

```python
# Note: This is a conceptual implementation
# For actual use, install featuretools: pip install featuretools

class AutomatedFeatureEngineering:
    """Automated feature engineering techniques"""
    
    def __init__(self):
        self.generated_features = {}
        
    def generate_features_automatically(self, df, target_col, max_depth=2):
        """Generate features automatically"""
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        generated_features = pd.DataFrame(index=df.index)
        
        # 1. Numerical transformations
        for col in numerical_cols:
            # Basic transformations
            if (df[col] > 0).all():
                generated_features[f'{col}_log'] = np.log(df[col])
            generated_features[f'{col}_squared'] = df[col] ** 2
            generated_features[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            
            # Binning
            try:
                generated_features[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=False)
            except:
                pass
        
        # 2. Pairwise interactions (depth 2)
        if max_depth >= 2:
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    generated_features[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                    generated_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    generated_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    
                    # Safe division
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = df[col1] / df[col2].replace(0, np.nan)
                    generated_features[f'{col1}_div_{col2}'] = ratio.fillna(0)
        
        # 3. Aggregation features
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            for cat_col in categorical_cols[:3]:  # Limit to avoid explosion
                for num_col in numerical_cols[:5]:
                    # Group statistics
                    for agg in ['mean', 'std', 'min', 'max']:
                        group_stat = df.groupby(cat_col)[num_col].transform(agg)
                        generated_features[f'{num_col}_by_{cat_col}_{agg}'] = group_stat
        
        # 4. Count encoding for categoricals
        for col in categorical_cols:
            generated_features[f'{col}_count'] = df[col].map(df[col].value_counts())
        
        self.generated_features = generated_features
        
        return generated_features
    
    def select_best_features(self, X, y, n_features=50):
        """Select best automatically generated features"""
        
        from sklearn.feature_selection import SelectKBest, f_classif
        
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        print(f"Selected {len(selected_features)} features from {X.shape[1]} generated features")
        print("\nTop 10 features:")
        print(scores.head(10))
        
        return X[selected_features], scores
```

## 7. Domain-Specific Feature Engineering

### 7.1 E-commerce Features

```python
class EcommerceFeatureEngineering:
    """Feature engineering for e-commerce data"""
    
    @staticmethod
    def create_customer_features(df):
        """Create customer behavior features"""
        
        features = pd.DataFrame(index=df.index)
        
        # RFM features (Recency, Frequency, Monetary)
        if all(col in df.columns for col in ['customer_id', 'order_date', 'order_value']):
            # Convert to datetime
            df['order_date'] = pd.to_datetime(df['order_date'])
            current_date = df['order_date'].max()
            
            # Customer level aggregations
            customer_features = df.groupby('customer_id').agg({
                'order_date': lambda x: (current_date - x.max()).days,  # Recency
                'order_value': ['count', 'sum', 'mean'],  # Frequency & Monetary
            })
            
            customer_features.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg']
            
            # Map back to orders
            for col in customer_features.columns:
                features[f'customer_{col}'] = df['customer_id'].map(
                    customer_features[col].to_dict()
                )
        
        # Product features
        if 'product_id' in df.columns:
            # Product popularity
            product_counts = df['product_id'].value_counts()
            features['product_popularity'] = df['product_id'].map(product_counts)
            
            # Product price features
            if 'price' in df.columns:
                product_avg_price = df.groupby('product_id')['price'].mean()
                features['product_avg_price'] = df['product_id'].map(product_avg_price)
                features['price_vs_avg'] = df['price'] - features['product_avg_price']
        
        # Time-based features
        if 'order_date' in df.columns:
            features['order_hour'] = df['order_date'].dt.hour
            features['order_dayofweek'] = df['order_date'].dt.dayofweek
            features['is_weekend'] = features['order_dayofweek'].isin([5, 6])
            features['is_month_end'] = df['order_date'].dt.is_month_end
            
            # Seasonality
            features['order_month'] = df['order_date'].dt.month
            features['order_quarter'] = df['order_date'].dt.quarter
            features['is_holiday_season'] = features['order_month'].isin([11, 12])
        
        # Cart features
        if 'items_in_cart' in df.columns and 'order_value' in df.columns:
            features['avg_item_value'] = df['order_value'] / df['items_in_cart'].replace(0, 1)
        
        # Session features
        if 'session_duration' in df.columns:
            features['session_duration_binned'] = pd.qcut(
                df['session_duration'], 
                q=5, 
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        return features

### 7.2 Financial Features

class FinancialFeatureEngineering:
    """Feature engineering for financial data"""
    
    @staticmethod
    def create_technical_indicators(df, price_col='close', volume_col='volume'):
        """Create technical analysis features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based indicators
        if price_col in df.columns:
            prices = df[price_col]
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f'ma_{window}'] = prices.rolling(window).mean()
                features[f'ma_{window}_ratio'] = prices / features[f'ma_{window}']
            
            # Exponential moving average
            for span in [12, 26]:
                features[f'ema_{span}'] = prices.ewm(span=span).mean()
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_diff'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            ma_20 = prices.rolling(20).mean()
            std_20 = prices.rolling(20).std()
            features['bb_upper'] = ma_20 + 2 * std_20
            features['bb_lower'] = ma_20 - 2 * std_20
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (prices - features['bb_lower']) / features['bb_width']
            
            # RSI (Relative Strength Index)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Price changes
            for lag in [1, 5, 20]:
                features[f'return_{lag}d'] = prices.pct_change(lag)
                features[f'log_return_{lag}d'] = np.log(prices / prices.shift(lag))
        
        # Volume-based indicators
        if volume_col in df.columns and price_col in df.columns:
            volumes = df[volume_col]
            
            # Volume moving average
            features['volume_ma_10'] = volumes.rolling(10).mean()
            features['volume_ratio'] = volumes / features['volume_ma_10']
            
            # Price-Volume trend
            features['pvt'] = ((prices.diff() / prices.shift()) * volumes).cumsum()
            
            # On-Balance Volume
            obv = np.where(prices.diff() > 0, volumes, 
                          np.where(prices.diff() < 0, -volumes, 0))
            features['obv'] = np.cumsum(obv)
        
        # Volatility features
        if price_col in df.columns:
            # Historical volatility
            for window in [5, 20]:
                features[f'volatility_{window}d'] = prices.pct_change().rolling(window).std()
            
            # Average True Range (ATR)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features['atr'] = true_range.rolling(14).mean()
        
        return features
```

## 8. Advanced Techniques

### 8.1 Feature Learning

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class FeatureLearning:
    """Learn new features from data"""
    
    def __init__(self):
        self.models = {}
        
    def pca_features(self, X, n_components=10, prefix='pca'):
        """Create PCA features"""
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(X)
        
        # Create DataFrame
        feature_names = [f'{prefix}_{i}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_features, columns=feature_names, index=X.index if hasattr(X, 'index') else None)
        
        # Store model and info
        self.models['pca'] = {
            'model': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_variance_explained': sum(pca.explained_variance_ratio_)
        }
        
        print(f"PCA: {n_components} components explain {self.models['pca']['total_variance_explained']:.2%} of variance")
        
        return pca_df
    
    def clustering_features(self, X, n_clusters=10, prefix='cluster'):
        """Create clustering-based features"""
        
        features = pd.DataFrame(index=X.index if hasattr(X, 'index') else None)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        features[f'{prefix}_label'] = cluster_labels
        
        # Distance to each cluster center
        distances = kmeans.transform(X)
        for i in range(n_clusters):
            features[f'{prefix}_dist_{i}'] = distances[:, i]
        
        # Distance to nearest and second nearest cluster
        sorted_distances = np.sort(distances, axis=1)
        features[f'{prefix}_nearest_dist'] = sorted_distances[:, 0]
        features[f'{prefix}_second_nearest_dist'] = sorted_distances[:, 1]
        features[f'{prefix}_dist_ratio'] = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-10)
        
        self.models['kmeans'] = kmeans
        
        return features
    
    def autoencoder_features(self, X, encoding_dim=10):
        """Create features using autoencoder (simplified version)"""
        
        from sklearn.neural_network import MLPRegressor
        
        # Simple autoencoder using MLPRegressor
        # For real applications, use keras/pytorch
        
        hidden_layer_sizes = (encoding_dim,)
        
        # Train autoencoder
        autoencoder = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes + hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        autoencoder.fit(X, X)  # Reconstruct input
        
        # Extract encoded features (activations of middle layer)
        # This is a simplified approach
        # For actual implementation, use deep learning frameworks
        
        # Use the model to transform data
        encoded_features = autoencoder.predict(X)[:, :encoding_dim]
        
        # Create DataFrame
        feature_names = [f'autoencoder_{i}' for i in range(encoding_dim)]
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
        
        self.models['autoencoder'] = autoencoder
        
        return encoded_df
```

### 8.2 Feature Interaction Discovery

```python
from itertools import combinations
from scipy.stats import spearmanr

class FeatureInteractionDiscovery:
    """Discover meaningful feature interactions"""
    
    def __init__(self):
        self.interactions = {}
        
    def find_nonlinear_interactions(self, X, y, top_k=10):
        """Find nonlinear interactions between features"""
        
        feature_names = X.columns if hasattr(X, 'columns') else [f'f_{i}' for i in range(X.shape[1])]
        interaction_scores = []
        
        # Try all pairs
        for feat1, feat2 in combinations(range(X.shape[1]), 2):
            if hasattr(X, 'iloc'):
                x1, x2 = X.iloc[:, feat1], X.iloc[:, feat2]
            else:
                x1, x2 = X[:, feat1], X[:, feat2]
            
            # Create interaction
            interaction = x1 * x2
            
            # Measure interaction strength
            # Compare individual correlations vs interaction correlation
            corr1 = abs(spearmanr(x1, y)[0])
            corr2 = abs(spearmanr(x2, y)[0])
            corr_interaction = abs(spearmanr(interaction, y)[0])
            
            # Interaction gain
            gain = corr_interaction - max(corr1, corr2)
            
            interaction_scores.append({
                'feature1': feature_names[feat1],
                'feature2': feature_names[feat2],
                'corr1': corr1,
                'corr2': corr2,
                'corr_interaction': corr_interaction,
                'gain': gain
            })
        
        # Sort by gain
        interaction_df = pd.DataFrame(interaction_scores)
        interaction_df = interaction_df.sort_values('gain', ascending=False)
        
        # Store top interactions
        self.interactions['top_k'] = interaction_df.head(top_k)
        
        return interaction_df.head(top_k)
    
    def create_interaction_features(self, X, interactions_df):
        """Create features based on discovered interactions"""
        
        interaction_features = pd.DataFrame(index=X.index if hasattr(X, 'index') else None)
        
        for _, row in interactions_df.iterrows():
            feat1, feat2 = row['feature1'], row['feature2']
            
            # Multiplication
            interaction_features[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
            
            # Other interactions based on gain
            if row['gain'] > 0.1:
                # Strong interaction - create more features
                interaction_features[f'{feat1}_plus_{feat2}'] = X[feat1] + X[feat2]
                interaction_features[f'{feat1}_minus_{feat2}'] = X[feat1] - X[feat2]
                
                # Safe division
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = X[feat1] / X[feat2].replace(0, np.nan)
                interaction_features[f'{feat1}_div_{feat2}'] = ratio.fillna(0)
        
        return interaction_features
```

## 9. Best Practices

### 9.1 Feature Engineering Pipeline

```python
class FeatureEngineeringBestPractices:
    """Best practices for feature engineering"""
    
    @staticmethod
    def create_robust_pipeline():
        """Create a robust feature engineering pipeline"""
        
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
        
        # Full pipeline with feature selection
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(k=50)),
            ('classifier', RandomForestClassifier())
        ])
        
        return full_pipeline
    
    @staticmethod
    def feature_engineering_checklist():
        """Comprehensive checklist for feature engineering"""
        
        checklist = {
            'Data Understanding': [
                'Analyze feature distributions',
                'Check for missing values',
                'Identify feature types',
                'Understand business context',
                'Check for data leakage'
            ],
            
            'Feature Creation': [
                'Create domain-specific features',
                'Extract from datetime columns',
                'Generate statistical features',
                'Create interaction features',
                'Apply mathematical transformations'
            ],
            
            'Feature Transformation': [
                'Handle missing values appropriately',
                'Encode categorical variables',
                'Scale numerical features',
                'Handle outliers',
                'Create binned features'
            ],
            
            'Feature Selection': [
                'Remove zero/low variance features',
                'Remove highly correlated features',
                'Use statistical tests',
                'Apply model-based selection',
                'Consider business importance'
            ],
            
            'Validation': [
                'Check for data leakage',
                'Validate on hold-out set',
                'Cross-validate feature importance',
                'Monitor feature stability',
                'Document feature definitions'
            ],
            
            'Production': [
                'Ensure reproducibility',
                'Handle new categories',
                'Monitor feature drift',
                'Version control features',
                'Optimize for inference speed'
            ]
        }
        
        return checklist
    
    @staticmethod
    def avoid_common_pitfalls():
        """Common pitfalls to avoid"""
        
        pitfalls = {
            'Data Leakage': {
                'description': 'Using future information or target information',
                'examples': [
                    'Including target in aggregations',
                    'Using test set statistics',
                    'Time-based leakage'
                ],
                'prevention': [
                    'Careful feature review',
                    'Temporal validation',
                    'Pipeline approach'
                ]
            },
            
            'Overfitting': {
                'description': 'Creating too specific features',
                'examples': [
                    'Too many polynomial features',
                    'Overly specific interactions',
                    'Target encoding without regularization'
                ],
                'prevention': [
                    'Cross-validation',
                    'Regularization',
                    'Feature selection'
                ]
            },
            
            'Inconsistency': {
                'description': 'Different preprocessing in train/test',
                'examples': [
                    'Fitting scalers on test data',
                    'Different imputation strategies',
                    'Inconsistent encoding'
                ],
                'prevention': [
                    'Use pipelines',
                    'Save preprocessing objects',
                    'Consistent validation'
                ]
            }
        }
        
        return pitfalls
```

## 10. Interview Questions

### Q1: What is feature engineering and why is it important?
**Answer**: Feature engineering is the process of creating new features or transforming existing ones to improve model performance. It's important because:
- The right features can make simple models perform very well
- It incorporates domain knowledge into the model
- Can reduce dimensionality while preserving information
- Helps models learn patterns more easily
- Often provides bigger performance gains than algorithm tuning

"Feature engineering is the key to success in applied machine learning."

### Q2: What are the main types of feature engineering techniques?
**Answer**:
1. **Feature Creation**:
   - Domain-based features
   - Mathematical transformations (log, polynomial)
   - Interaction features
   - Aggregation features

2. **Feature Transformation**:
   - Scaling/Normalization
   - Encoding categorical variables
   - Binning continuous variables
   - Handling missing values

3. **Feature Extraction**:
   - PCA, autoencoders
   - Text features (TF-IDF, embeddings)
   - Image features (edges, textures)

4. **Feature Selection**:
   - Filter methods (correlation, mutual information)
   - Wrapper methods (RFE)
   - Embedded methods (L1, tree importance)

### Q3: How do you handle categorical variables with high cardinality?
**Answer**: Several approaches:

1. **Frequency encoding**: Replace with occurrence count
2. **Target encoding**: Replace with target mean (with smoothing)
3. **Binary encoding**: Represent as binary numbers
4. **Hashing**: Hash to fixed number of buckets
5. **Embedding**: Learn dense representations (for neural networks)
6. **Grouping**: Combine rare categories
7. **PCA on one-hot**: Dimensionality reduction after encoding

Choice depends on:
- Model type (tree-based vs linear)
- Cardinality level
- Relationship with target
- Training data size

### Q4: What is target encoding and what are its risks?
**Answer**: Target encoding replaces categorical values with target statistics (usually mean).

**Example**: 
- City "NYC" → average target value for NYC

**Risks**:
1. **Overfitting**: Especially with high cardinality/small samples
2. **Target leakage**: If not done properly
3. **Distribution shift**: New categories in test set

**Mitigation strategies**:
- **Smoothing**: Blend with global mean
- **Cross-validation encoding**: Out-of-fold encoding
- **Leave-one-out**: Exclude current row
- **Noise addition**: Add random noise
- **Regularization**: Bayesian target encoding

### Q5: How do you create features from datetime data?
**Answer**: Extract multiple aspects:

**Basic components**:
- Year, month, day, hour, minute
- Day of week, day of year
- Week of year, quarter

**Cyclical encoding**:
```python
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```

**Relative features**:
- Days since/until event
- Age calculations
- Time between events

**Special indicators**:
- Is weekend/holiday
- Is month/quarter start/end
- Business days

**Domain-specific**:
- Shopping seasons
- Financial quarters
- Academic terms

### Q6: What is feature scaling and when is it necessary?
**Answer**: Feature scaling transforms features to similar ranges.

**When necessary**:
- **Distance-based algorithms**: KNN, K-means, SVM
- **Gradient descent**: Neural networks, linear models
- **Regularization**: When penalty should apply equally
- **PCA**: To prevent dominance by scale

**When NOT necessary**:
- Tree-based models (Random Forest, XGBoost)
- Rule-based models

**Methods**:
1. **StandardScaler**: (x - μ) / σ
2. **MinMaxScaler**: (x - min) / (max - min)
3. **RobustScaler**: Uses median and IQR
4. **Normalizer**: Unit norm per sample
5. **QuantileTransformer**: Uniform or normal distribution

### Q7: How do you detect and handle feature interactions?
**Answer**:

**Detection methods**:
1. **Domain knowledge**: Business understanding
2. **Correlation analysis**: Between interaction and target
3. **Tree-based importance**: Trees naturally find interactions
4. **Statistical tests**: ANOVA for interaction effects
5. **Visualization**: Scatter plots, parallel coordinates

**Creation methods**:
1. **Arithmetic**: Multiply, add, subtract, divide
2. **Polynomial features**: All combinations up to degree n
3. **Domain-specific**: Price × quantity, rate × time
4. **Threshold-based**: Feature1 > X AND Feature2 < Y

**Selection**:
- Mutual information gain
- Forward selection
- L1 regularization

### Q8: What is the curse of dimensionality in feature engineering?
**Answer**: As dimensions increase:

**Problems**:
1. **Sparsity**: Data becomes sparse in high dimensions
2. **Distance concentration**: All points become equidistant
3. **Overfitting**: More parameters than observations
4. **Computational cost**: Exponential increase

**In feature engineering context**:
- Creating too many features (polynomial, interactions)
- One-hot encoding high cardinality
- Text features (bag of words)

**Solutions**:
1. **Feature selection**: Keep only important features
2. **Dimensionality reduction**: PCA, autoencoders
3. **Regularization**: L1/L2 penalties
4. **Domain knowledge**: Create meaningful features only
5. **Sparse representations**: For text/categorical

### Q9: How do you handle missing values in feature engineering?
**Answer**: Depends on missing mechanism:

**Types**:
1. **MCAR** (Missing Completely At Random): Random missingness
2. **MAR** (Missing At Random): Depends on observed features
3. **MNAR** (Missing Not At Random): Depends on unobserved

**Strategies**:
1. **Deletion**: If MCAR and few missing
2. **Simple imputation**: Mean, median, mode, forward-fill
3. **Advanced imputation**: KNN, MICE, deep learning
4. **Indicator variable**: IsMissing feature
5. **Domain-specific**: Business logic imputation
6. **Model-based**: Some algorithms handle naturally

**Best practices**:
- Never impute target variable
- Impute within cross-validation
- Consider multiple imputation
- Document imputation strategy

### Q10: What is feature leakage and how do you prevent it?
**Answer**: Feature leakage occurs when training features contain information about the target that won't be available at prediction time.

**Common sources**:
1. **Direct leakage**: Including target-derived features
2. **Temporal leakage**: Future information in features
3. **Statistical leakage**: Using test set statistics
4. **Group leakage**: Information from same group/user

**Prevention**:
1. **Careful feature review**: Understand each feature
2. **Temporal validation**: Respect time order
3. **Pipeline approach**: Fit preprocessors on train only
4. **Cross-validation**: Proper nested CV
5. **Business understanding**: Know data generation process

**Example**: Predicting loan default using "account_closed_reason" - only populated after default!

### Q11: How do you perform feature selection for different types of models?
**Answer**:

**Linear models**:
- Correlation analysis
- L1 regularization (Lasso)
- Statistical tests (t-test, F-test)
- VIF for multicollinearity

**Tree-based models**:
- Built-in feature importance
- Permutation importance
- SHAP values
- Can handle many features

**Neural networks**:
- Automatic feature learning
- Dropout as feature selection
- L1/L2 regularization
- Attention mechanisms

**General approaches**:
- Mutual information
- Recursive feature elimination
- Forward/backward selection
- Genetic algorithms

### Q12: What are embedding features and when do you use them?
**Answer**: Embeddings are learned dense representations of categorical/discrete data.

**When to use**:
1. **High cardinality categoricals**: User IDs, product IDs
2. **Text data**: Word/sentence embeddings
3. **Network data**: Node embeddings
4. **Sequential data**: Time series embeddings

**Methods**:
1. **Neural network embeddings**: Learned end-to-end
2. **Word2Vec/GloVe**: For text
3. **Entity embeddings**: For categoricals
4. **Pre-trained embeddings**: Transfer learning

**Advantages**:
- Captures semantic similarity
- Reduces dimensionality
- Can transfer between tasks
- Handles new categories (with defaults)

### Q13: How do you engineer features for time series data?
**Answer**:

**Lag features**:
- Previous values: lag_1, lag_7, lag_30
- Differences: value - lag_1
- Ratios: value / lag_1

**Rolling statistics**:
- Moving averages: MA_7, MA_30
- Rolling std, min, max
- Exponential weighted stats

**Time-based**:
- Hour, day, month effects
- Seasonality indicators
- Holiday/event flags

**Domain-specific**:
- Technical indicators (finance)
- Weather features
- Economic indicators

**Advanced**:
- Fourier transforms
- Wavelet features
- Change point detection

**Important**: Avoid look-ahead bias!

### Q14: What is polynomial feature generation and when is it useful?
**Answer**: Creating features by multiplying existing features (including with themselves).

**Example**:
- Features: [a, b]
- Polynomial degree 2: [1, a, b, a², ab, b²]

**When useful**:
1. **Non-linear relationships**: Model can't capture naturally
2. **Linear models**: Add non-linearity
3. **Interaction effects**: Important cross-terms
4. **Domain knowledge**: Known polynomial relationships

**Cautions**:
- Exponential growth in features
- Increased overfitting risk
- Multicollinearity issues
- Scale sensitivity

**Best practices**:
- Use with regularization
- Consider only interactions (no powers)
- Domain-guided selection
- Cross-validate degree

### Q15: How do you validate feature engineering choices?
**Answer**:

**Validation strategies**:
1. **Cross-validation**: Most important features consistent?
2. **Hold-out test**: Performance improvement?
3. **Feature stability**: Consistent across time/samples?
4. **Business validation**: Do features make sense?

**Metrics**:
- Model performance gain
- Feature importance scores
- Correlation with target
- Mutual information

**Techniques**:
1. **Ablation study**: Remove features, measure impact
2. **Permutation importance**: Shuffle feature, measure degradation
3. **Partial dependence**: Feature effect on predictions
4. **SHAP values**: Feature contributions

**Red flags**:
- Too good to be true performance
- Unstable feature importance
- Non-intuitive top features
- Different behavior in production

### Q16: What are some domain-specific feature engineering techniques?
**Answer**:

**Text**:
- TF-IDF, n-grams
- Sentiment scores
- Named entity counts
- Readability metrics

**Images**:
- Edge detection
- Color histograms
- Texture features
- Pre-trained CNN features

**E-commerce**:
- RFM (Recency, Frequency, Monetary)
- Session features
- Product interactions
- Seasonal patterns

**Finance**:
- Technical indicators
- Risk metrics
- Market correlations
- Fundamental ratios

**Healthcare**:
- Vital sign variability
- Lab test trends
- Comorbidity indices
- Treatment patterns

### Q17: How do you handle cyclical features?
**Answer**: Cyclical features (like hour, day, month) need special handling because 23:00 is close to 00:00.

**Approach**: Transform to sin/cos representation
```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

**Benefits**:
- Preserves cyclical nature
- 23 and 0 are close
- Smooth transitions
- No arbitrary boundaries

**Applications**:
- Time: hour, minute
- Calendar: day of week, month
- Angles: wind direction
- Periodic events: seasons

**Alternative**: Binning with overlap or splines

### Q18: What is feature hashing and when do you use it?
**Answer**: Feature hashing (hash trick) maps features to fixed-size vector using hash function.

**How it works**:
```python
hash_index = hash(feature_name) % n_buckets
```

**When to use**:
1. **High cardinality**: Millions of categories
2. **Memory constraints**: Fixed memory usage
3. **Online learning**: Handle new categories
4. **Text features**: Bag of words alternative

**Advantages**:
- O(1) memory
- No vocabulary needed
- Handles new values

**Disadvantages**:
- Hash collisions
- No interpretability
- Can't reverse mapping

**Best practices**:
- Use enough buckets (reduce collisions)
- Include sign hash for better distribution
- Combine with other methods

### Q19: How do you create features for recommendation systems?
**Answer**:

**User features**:
- Demographics
- Historical behavior
- Preference vectors
- Activity patterns
- Social network features

**Item features**:
- Content features
- Popularity metrics
- Category embeddings
- Quality indicators
- Age/freshness

**Interaction features**:
- User-item affinity
- Collaborative filtering signals
- Time since last interaction
- Context (device, time, location)
- Session features

**Advanced**:
- Matrix factorization features
- Graph-based features
- Cross-features (user type × item type)
- Sequential patterns

### Q20: What are some advanced feature engineering techniques?
**Answer**:

**1. Representation learning**:
- Autoencoders
- VAEs (Variational Autoencoders)
- Self-supervised learning

**2. Feature learning**:
- Deep feature synthesis
- Genetic programming
- Neural architecture search for features

**3. Graph features**:
- Node embeddings (Node2Vec)
- Graph neural networks
- Centrality measures

**4. Meta-features**:
- Features about features
- Dataset characteristics
- Statistical properties

**5. Transfer features**:
- Pre-trained embeddings
- Domain adaptation
- Multi-task features

**6. Automated engineering**:
- Featuretools
- AutoML platforms
- Reinforcement learning for features