# Day 28: Machine Learning Algorithms Summary

## Table of Contents
1. [Introduction](#1-introduction)
2. [Algorithm Taxonomy](#2-algorithm-taxonomy)
3. [Supervised Learning Algorithms](#3-supervised-learning-algorithms)
4. [Unsupervised Learning Algorithms](#4-unsupervised-learning-algorithms)
5. [Algorithm Comparison Framework](#5-algorithm-comparison-framework)
6. [When to Use Which Algorithm](#6-when-to-use-which-algorithm)
7. [Complexity Analysis](#7-complexity-analysis)
8. [Ensemble Methods](#8-ensemble-methods)
9. [Algorithm Selection Guide](#9-algorithm-selection-guide)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

This comprehensive summary covers all major machine learning algorithms, their strengths, weaknesses, and appropriate use cases. Understanding when and how to apply different algorithms is crucial for solving real-world problems effectively.

### The Algorithm Landscape

Machine learning algorithms can be broadly categorized by:
- **Learning Style**: Supervised, Unsupervised, Semi-supervised, Reinforcement
- **Function**: Classification, Regression, Clustering, Dimensionality Reduction
- **Approach**: Instance-based, Model-based, Tree-based, Neural-based

### Key Considerations for Algorithm Selection

1. **Problem Type**: Classification vs Regression vs Clustering
2. **Data Characteristics**: Size, dimensionality, noise level
3. **Interpretability Requirements**: Black box vs explainable
4. **Computational Resources**: Training time and memory
5. **Performance Metrics**: Accuracy vs speed vs robustness

## 2. Algorithm Taxonomy

### 2.1 Comprehensive Algorithm Classification

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AlgorithmTaxonomy:
    """Organize and visualize ML algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'Supervised': {
                'Regression': {
                    'Linear': ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net'],
                    'Non-linear': ['Polynomial Regression', 'SVR', 'Decision Tree Regressor', 
                                  'Random Forest Regressor', 'Neural Networks'],
                    'Instance-based': ['KNN Regressor']
                },
                'Classification': {
                    'Linear': ['Logistic Regression', 'Linear SVM', 'Naive Bayes'],
                    'Non-linear': ['SVM with Kernel', 'Decision Tree', 'Random Forest',
                                  'Neural Networks', 'XGBoost'],
                    'Instance-based': ['KNN Classifier'],
                    'Probabilistic': ['Naive Bayes', 'Gaussian Process']
                }
            },
            'Unsupervised': {
                'Clustering': {
                    'Partitioning': ['K-Means', 'K-Medoids'],
                    'Hierarchical': ['Agglomerative', 'Divisive'],
                    'Density-based': ['DBSCAN', 'OPTICS'],
                    'Model-based': ['Gaussian Mixture Models']
                },
                'Dimensionality Reduction': {
                    'Linear': ['PCA', 'LDA', 'Factor Analysis'],
                    'Non-linear': ['t-SNE', 'UMAP', 'Kernel PCA', 'Autoencoders']
                }
            },
            'Semi-supervised': ['Label Propagation', 'Co-training', 'Self-training'],
            'Reinforcement': ['Q-Learning', 'SARSA', 'Policy Gradient', 'Actor-Critic']
        }
    
    def visualize_taxonomy(self):
        """Create visual representation of algorithm taxonomy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Supervised learning tree
        self._plot_tree(ax1, self.algorithms['Supervised'], 'Supervised Learning Algorithms')
        
        # Unsupervised learning tree
        self._plot_tree(ax2, self.algorithms['Unsupervised'], 'Unsupervised Learning Algorithms')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_tree(self, ax, data, title):
        """Helper to plot hierarchical structure"""
        y_pos = 0
        x_levels = {0: 0.1, 1: 0.4, 2: 0.7}
        positions = {}
        
        ax.text(0.5, 0.95, title, ha='center', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        for category, subcategories in data.items():
            y_pos -= 0.15
            ax.text(x_levels[0], y_pos, category, fontsize=12, fontweight='bold')
            positions[category] = (x_levels[0], y_pos)
            
            if isinstance(subcategories, dict):
                for subcat, algorithms in subcategories.items():
                    y_pos -= 0.1
                    ax.text(x_levels[1], y_pos, subcat, fontsize=10)
                    
                    # Draw connection
                    ax.plot([x_levels[0] + 0.1, x_levels[1]], 
                           [positions[category][1], y_pos], 'k-', alpha=0.3)
                    
                    if isinstance(algorithms, dict):
                        for algo_type, algo_list in algorithms.items():
                            y_pos -= 0.08
                            ax.text(x_levels[2], y_pos, algo_type + ':', fontsize=9, style='italic')
                            for algo in algo_list:
                                y_pos -= 0.06
                                ax.text(x_levels[2] + 0.05, y_pos, '• ' + algo, fontsize=8)
                    else:
                        for algo in algorithms:
                            y_pos -= 0.06
                            ax.text(x_levels[2], y_pos, '• ' + algo, fontsize=8)
                            
                            # Draw connection
                            ax.plot([x_levels[1] + 0.1, x_levels[2]], 
                                   [positions[category][1], y_pos], 'k-', alpha=0.2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(y_pos - 0.1, 1)
        ax.axis('off')

# Visualize algorithm taxonomy
taxonomy = AlgorithmTaxonomy()
taxonomy.visualize_taxonomy()
```

### 2.2 Algorithm Characteristics Matrix

```python
class AlgorithmCharacteristics:
    """Compare algorithm characteristics"""
    
    def __init__(self):
        self.characteristics = pd.DataFrame({
            'Algorithm': ['Linear Regression', 'Logistic Regression', 'Decision Tree',
                         'Random Forest', 'SVM', 'Neural Network', 'K-Means', 'PCA',
                         'Naive Bayes', 'KNN'],
            'Type': ['Regression', 'Classification', 'Both', 'Both', 'Both', 'Both',
                    'Clustering', 'Dim Reduction', 'Classification', 'Both'],
            'Interpretability': ['High', 'High', 'Medium', 'Low', 'Low', 'Very Low',
                               'Medium', 'Medium', 'Medium', 'Low'],
            'Training_Speed': ['Very Fast', 'Fast', 'Fast', 'Medium', 'Slow', 'Very Slow',
                             'Fast', 'Fast', 'Very Fast', 'Very Fast'],
            'Prediction_Speed': ['Very Fast', 'Very Fast', 'Very Fast', 'Fast', 'Fast', 'Fast',
                               'N/A', 'Very Fast', 'Very Fast', 'Slow'],
            'Memory_Usage': ['Low', 'Low', 'Medium', 'High', 'Medium', 'High',
                           'Low', 'Medium', 'Low', 'High'],
            'Handles_Nonlinearity': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
                                   'Limited', 'No', 'No', 'Yes'],
            'Requires_Scaling': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes',
                               'Yes', 'Yes', 'No', 'Yes'],
            'Robust_to_Outliers': ['No', 'Medium', 'Yes', 'Yes', 'Medium', 'Medium',
                                 'No', 'No', 'Yes', 'No']
        })
    
    def create_comparison_heatmap(self):
        """Create heatmap comparing algorithm characteristics"""
        # Convert categorical to numerical for heatmap
        char_numeric = self.characteristics.copy()
        
        # Define mappings
        speed_map = {'Very Slow': 1, 'Slow': 2, 'Medium': 3, 'Fast': 4, 'Very Fast': 5, 'N/A': 0}
        level_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4}
        binary_map = {'No': 0, 'Limited': 0.5, 'Yes': 1}
        
        char_numeric['Interpretability'] = char_numeric['Interpretability'].map(level_map)
        char_numeric['Training_Speed'] = char_numeric['Training_Speed'].map(speed_map)
        char_numeric['Prediction_Speed'] = char_numeric['Prediction_Speed'].map(speed_map)
        char_numeric['Memory_Usage'] = char_numeric['Memory_Usage'].map(level_map)
        char_numeric['Handles_Nonlinearity'] = char_numeric['Handles_Nonlinearity'].map(binary_map)
        char_numeric['Requires_Scaling'] = char_numeric['Requires_Scaling'].map(binary_map)
        char_numeric['Robust_to_Outliers'] = char_numeric['Robust_to_Outliers'].map(binary_map)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        numeric_cols = ['Interpretability', 'Training_Speed', 'Prediction_Speed', 
                       'Memory_Usage', 'Handles_Nonlinearity', 'Requires_Scaling', 
                       'Robust_to_Outliers']
        
        heatmap_data = char_numeric[numeric_cols].T
        heatmap_data.columns = char_numeric['Algorithm']
        
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=2.5, 
                   linewidths=0.5, cbar_kws={'label': 'Score'})
        plt.title('Algorithm Characteristics Comparison', fontsize=16, pad=20)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Characteristic', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return self.characteristics

# Create and display characteristics comparison
char_analyzer = AlgorithmCharacteristics()
characteristics_df = char_analyzer.create_comparison_heatmap()
print("\nAlgorithm Characteristics Table:")
print(characteristics_df.to_string(index=False))
```

## 3. Supervised Learning Algorithms

### 3.1 Regression Algorithms

#### Linear Regression
```python
class RegressionAlgorithmsSummary:
    """Summary of regression algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'Linear Regression': {
                'equation': 'y = β₀ + β₁x₁ + ... + βₙxₙ + ε',
                'loss': 'MSE = (1/n)Σ(yᵢ - ŷᵢ)²',
                'assumptions': ['Linearity', 'Independence', 'Homoscedasticity', 
                              'Normality of residuals'],
                'pros': ['Simple and interpretable', 'Fast training', 'No hyperparameters'],
                'cons': ['Assumes linear relationship', 'Sensitive to outliers', 
                        'Can overfit with many features'],
                'use_cases': ['Trend analysis', 'Forecasting', 'Feature importance'],
                'complexity': {'training': 'O(n³)', 'prediction': 'O(d)'}
            },
            'Ridge Regression': {
                'equation': 'Loss = MSE + α||β||²',
                'regularization': 'L2 penalty',
                'key_param': 'α (regularization strength)',
                'pros': ['Handles multicollinearity', 'Prevents overfitting', 
                        'Stable coefficients'],
                'cons': ['Less interpretable', 'Includes all features'],
                'use_cases': ['High-dimensional data', 'Correlated features'],
                'complexity': {'training': 'O(n³)', 'prediction': 'O(d)'}
            },
            'Lasso Regression': {
                'equation': 'Loss = MSE + α||β||₁',
                'regularization': 'L1 penalty',
                'key_param': 'α (regularization strength)',
                'pros': ['Feature selection', 'Sparse models', 'Interpretable'],
                'cons': ['Can be unstable', 'Arbitrary selection among correlated features'],
                'use_cases': ['Feature selection', 'Sparse data', 'Interpretability needed'],
                'complexity': {'training': 'O(n³)', 'prediction': 'O(d)'}
            },
            'Support Vector Regression': {
                'equation': 'f(x) = Σᵢ(αᵢ - αᵢ*)K(xᵢ, x) + b',
                'key_concept': 'ε-insensitive tube',
                'key_params': ['C (regularization)', 'ε (tube width)', 'kernel'],
                'pros': ['Handles non-linearity', 'Robust to outliers', 'Effective in high dimensions'],
                'cons': ['Computationally expensive', 'Memory intensive', 'Black box'],
                'use_cases': ['Non-linear relationships', 'High-dimensional data'],
                'complexity': {'training': 'O(n²) to O(n³)', 'prediction': 'O(n_sv × d)'}
            },
            'Decision Tree Regression': {
                'algorithm': 'Recursive partitioning',
                'splitting_criterion': 'MSE reduction',
                'key_params': ['max_depth', 'min_samples_split', 'min_samples_leaf'],
                'pros': ['Handles non-linearity', 'No scaling needed', 'Feature importance'],
                'cons': ['Prone to overfitting', 'Unstable', 'Cannot extrapolate'],
                'use_cases': ['Non-linear data', 'Mixed data types', 'Rule extraction'],
                'complexity': {'training': 'O(n log n × d)', 'prediction': 'O(log n)'}
            },
            'Random Forest Regression': {
                'base': 'Ensemble of decision trees',
                'key_innovation': 'Bootstrap + random feature selection',
                'key_params': ['n_estimators', 'max_features', 'max_depth'],
                'pros': ['Reduces overfitting', 'Handles non-linearity', 'Feature importance'],
                'cons': ['Black box', 'Computationally expensive', 'Memory intensive'],
                'use_cases': ['General purpose', 'Feature importance', 'Robust predictions'],
                'complexity': {'training': 'O(n log n × d × k)', 'prediction': 'O(k log n)'}
            }
        }
    
    def compare_regression_algorithms(self, X, y):
        """Compare performance of regression algorithms"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        algorithms = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'SVR': Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))]),
            'Decision Tree': DecisionTreeRegressor(max_depth=5),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5)
        }
        
        results = {}
        for name, model in algorithms.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            results[name] = {
                'RMSE': np.sqrt(-scores.mean()),
                'Std': np.sqrt(scores.std())
            }
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Performance comparison
        names = list(results.keys())
        rmse_values = [results[name]['RMSE'] for name in names]
        std_values = [results[name]['Std'] for name in names]
        
        x_pos = np.arange(len(names))
        ax1.bar(x_pos, rmse_values, yerr=std_values, capsize=5, 
                color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'peachpuff'])
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Regression Algorithm Performance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Complexity visualization
        train_complexity = {'Linear': 1, 'Ridge': 1, 'Lasso': 1, 
                          'SVR': 3, 'Decision Tree': 2, 'Random Forest': 3}
        predict_complexity = {'Linear': 1, 'Ridge': 1, 'Lasso': 1, 
                            'SVR': 3, 'Decision Tree': 1, 'Random Forest': 2}
        
        shortened_names = ['Linear', 'Ridge', 'Lasso', 'SVR', 'DT', 'RF']
        train_vals = [train_complexity[name] for name in train_complexity.keys()]
        predict_vals = [predict_complexity[name] for name in predict_complexity.keys()]
        
        x = np.arange(len(shortened_names))
        width = 0.35
        
        ax2.bar(x - width/2, train_vals, width, label='Training', color='steelblue')
        ax2.bar(x + width/2, predict_vals, width, label='Prediction', color='darkorange')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Relative Complexity')
        ax2.set_title('Computational Complexity Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(shortened_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results

# Generate sample data and compare algorithms
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)

reg_summary = RegressionAlgorithmsSummary()
reg_results = reg_summary.compare_regression_algorithms(X_reg, y_reg)
```

### 3.2 Classification Algorithms

```python
class ClassificationAlgorithmsSummary:
    """Summary of classification algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'Logistic Regression': {
                'function': 'p(y=1|x) = 1/(1 + exp(-βᵀx))',
                'loss': 'Cross-entropy',
                'decision_boundary': 'Linear',
                'pros': ['Probabilistic output', 'Interpretable', 'Fast'],
                'cons': ['Linear boundary only', 'Requires feature engineering'],
                'use_cases': ['Binary classification', 'Probability estimation', 'Baseline model'],
                'extensions': ['Multinomial (softmax)', 'Ordinal regression']
            },
            'Support Vector Machine': {
                'objective': 'Maximize margin',
                'key_concept': 'Support vectors',
                'kernels': ['Linear', 'RBF', 'Polynomial', 'Sigmoid'],
                'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
                'cons': ['Black box with kernels', 'Sensitive to scaling', 'Binary by nature'],
                'use_cases': ['High-dimensional data', 'Non-linear boundaries', 'Text classification']
            },
            'Naive Bayes': {
                'assumption': 'Feature independence',
                'variants': ['Gaussian', 'Multinomial', 'Bernoulli'],
                'formula': 'P(y|x) ∝ P(y)∏P(xᵢ|y)',
                'pros': ['Fast training and prediction', 'Works with small data', 'Probabilistic'],
                'cons': ['Independence assumption rarely holds', 'Can be overconfident'],
                'use_cases': ['Text classification', 'Spam filtering', 'Real-time prediction']
            },
            'Decision Tree': {
                'algorithm': 'Recursive partitioning',
                'splitting': ['Gini impurity', 'Entropy', 'Information gain'],
                'pros': ['Interpretable', 'Handles non-linearity', 'No scaling needed'],
                'cons': ['Overfitting', 'Unstable', 'Biased to dominant classes'],
                'use_cases': ['Rule extraction', 'Mixed data types', 'Feature importance']
            },
            'Random Forest': {
                'ensemble_method': 'Bagging + random subspace',
                'key_params': ['n_estimators', 'max_features', 'max_depth'],
                'pros': ['Reduces overfitting', 'Feature importance', 'Handles missing values'],
                'cons': ['Black box', 'Memory intensive', 'Slower prediction'],
                'use_cases': ['General purpose', 'Feature selection', 'Imbalanced data']
            },
            'K-Nearest Neighbors': {
                'principle': 'Local similarity',
                'key_param': 'k (number of neighbors)',
                'distance_metrics': ['Euclidean', 'Manhattan', 'Minkowski'],
                'pros': ['Simple', 'Non-parametric', 'Multi-class naturally'],
                'cons': ['Slow prediction', 'Memory intensive', 'Curse of dimensionality'],
                'use_cases': ['Non-linear boundaries', 'Local patterns', 'Recommendation systems']
            },
            'Neural Networks': {
                'architecture': 'Layers of neurons',
                'activation': ['ReLU', 'Sigmoid', 'Tanh', 'Softmax'],
                'training': 'Backpropagation',
                'pros': ['Universal approximator', 'Feature learning', 'State-of-the-art'],
                'cons': ['Black box', 'Requires large data', 'Computationally expensive'],
                'use_cases': ['Complex patterns', 'Image/text/speech', 'Deep learning']
            }
        }
    
    def decision_boundary_comparison(self):
        """Visualize decision boundaries of different classifiers"""
        from sklearn.datasets import make_moons, make_circles, make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        
        # Generate datasets
        datasets = [
            make_moons(n_samples=200, noise=0.3, random_state=0),
            make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
            make_classification(n_samples=200, n_features=2, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, random_state=2)
        ]
        
        dataset_names = ['Moons', 'Circles', 'Linear']
        
        # Initialize classifiers
        classifiers = [
            ('Logistic Regression', LogisticRegression()),
            ('SVM (RBF)', SVC(kernel='rbf', gamma=2)),
            ('Decision Tree', DecisionTreeClassifier(max_depth=5)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5)),
            ('KNN (k=15)', KNeighborsClassifier(n_neighbors=15)),
            ('Naive Bayes', GaussianNB())
        ]
        
        fig, axes = plt.subplots(len(datasets), len(classifiers), figsize=(18, 10))
        
        for ds_idx, (X, y) in enumerate(datasets):
            # Standardize features
            X = StandardScaler().fit_transform(X)
            
            # Create mesh
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            for clf_idx, (name, clf) in enumerate(classifiers):
                ax = axes[ds_idx, clf_idx]
                
                # Train classifier
                clf.fit(X, y)
                
                # Predict on mesh
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
                
                # Plot training points
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                          edgecolor='black', s=20)
                
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                
                if ds_idx == 0:
                    ax.set_title(name)
                if clf_idx == 0:
                    ax.set_ylabel(dataset_names[ds_idx], fontsize=12)
        
        plt.suptitle('Decision Boundaries Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()

# Compare classification algorithms
clf_summary = ClassificationAlgorithmsSummary()
clf_summary.decision_boundary_comparison()
```

## 4. Unsupervised Learning Algorithms

### 4.1 Clustering Algorithms

```python
class ClusteringAlgorithmsSummary:
    """Summary of clustering algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'K-Means': {
                'type': 'Partitioning',
                'objective': 'Minimize within-cluster sum of squares',
                'parameters': ['k (number of clusters)', 'initialization method'],
                'pros': ['Fast', 'Scalable', 'Simple'],
                'cons': ['Assumes spherical clusters', 'Sensitive to initialization', 'Fixed k'],
                'use_cases': ['Customer segmentation', 'Image compression', 'Document clustering'],
                'complexity': 'O(n × k × d × i)'
            },
            'DBSCAN': {
                'type': 'Density-based',
                'principle': 'Core points and density reachability',
                'parameters': ['eps (neighborhood radius)', 'min_samples'],
                'pros': ['Finds arbitrary shapes', 'Robust to outliers', 'No k needed'],
                'cons': ['Sensitive to parameters', 'Varying densities', 'Not fully deterministic'],
                'use_cases': ['Spatial data', 'Anomaly detection', 'Non-convex clusters'],
                'complexity': 'O(n log n) with index'
            },
            'Hierarchical': {
                'types': ['Agglomerative (bottom-up)', 'Divisive (top-down)'],
                'linkage': ['Single', 'Complete', 'Average', 'Ward'],
                'pros': ['Dendrogram visualization', 'No k needed', 'Deterministic'],
                'cons': ['Computationally expensive', 'Memory intensive', 'Cannot undo merges'],
                'use_cases': ['Taxonomy creation', 'Small datasets', 'Exploring hierarchies'],
                'complexity': 'O(n³) naive, O(n² log n) optimized'
            },
            'Gaussian Mixture Models': {
                'type': 'Model-based',
                'assumption': 'Data from mixture of Gaussians',
                'estimation': 'Expectation-Maximization (EM)',
                'pros': ['Soft clustering', 'Captures elliptical shapes', 'Probabilistic'],
                'cons': ['Sensitive to initialization', 'Assumes Gaussian', 'Can converge to local optima'],
                'use_cases': ['Overlapping clusters', 'Density estimation', 'Missing data'],
                'complexity': 'O(n × k × d²) per iteration'
            }
        }
    
    def compare_clustering_algorithms(self):
        """Compare different clustering algorithms visually"""
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.datasets import make_blobs, make_moons, make_circles
        from sklearn.preprocessing import StandardScaler
        
        # Generate different cluster shapes
        n_samples = 300
        
        # Dataset 1: Well-separated blobs
        X1, y1 = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                           cluster_std=0.5, random_state=0)
        
        # Dataset 2: Moons (non-convex)
        X2, y2 = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
        
        # Dataset 3: Concentric circles
        X3, y3 = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=0)
        
        datasets = [(X1, y1, 'Blobs'), (X2, y2, 'Moons'), (X3, y3, 'Circles')]
        
        # Clustering algorithms
        clustering_algorithms = [
            ('K-Means', KMeans(n_clusters=3)),
            ('DBSCAN', DBSCAN(eps=0.3)),
            ('Hierarchical', AgglomerativeClustering(n_clusters=3)),
            ('GMM', GaussianMixture(n_components=3))
        ]
        
        fig, axes = plt.subplots(len(datasets), len(clustering_algorithms) + 1, 
                                figsize=(18, 10))
        
        for i, (X, y_true, dataset_name) in enumerate(datasets):
            # Standardize features
            X = StandardScaler().fit_transform(X)
            
            # Plot original data
            axes[i, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30)
            axes[i, 0].set_title(f'{dataset_name} (True Labels)' if i == 0 else '')
            axes[i, 0].set_ylabel(dataset_name, fontsize=12)
            
            # Apply clustering algorithms
            for j, (name, algorithm) in enumerate(clustering_algorithms):
                if name == 'DBSCAN' and dataset_name == 'Blobs':
                    algorithm = DBSCAN(eps=0.5)  # Adjust for blobs
                elif name == 'DBSCAN' and dataset_name == 'Circles':
                    algorithm = DBSCAN(eps=0.15)  # Adjust for circles
                
                # Fit and predict
                if name == 'GMM':
                    y_pred = algorithm.fit_predict(X)
                else:
                    y_pred = algorithm.fit_predict(X)
                
                # Plot results
                axes[i, j+1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30)
                if i == 0:
                    axes[i, j+1].set_title(name)
                
                # Mark noise points for DBSCAN
                if name == 'DBSCAN':
                    noise_mask = y_pred == -1
                    axes[i, j+1].scatter(X[noise_mask, 0], X[noise_mask, 1], 
                                       c='red', marker='x', s=50)
        
        plt.suptitle('Clustering Algorithms Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()

# Compare clustering algorithms
clustering_summary = ClusteringAlgorithmsSummary()
clustering_summary.compare_clustering_algorithms()
```

### 4.2 Dimensionality Reduction Algorithms

```python
class DimensionalityReductionSummary:
    """Summary of dimensionality reduction algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'PCA': {
                'full_name': 'Principal Component Analysis',
                'type': 'Linear',
                'objective': 'Maximize variance',
                'method': 'Eigendecomposition of covariance matrix',
                'pros': ['Fast', 'No parameters', 'Optimal linear projection'],
                'cons': ['Linear only', 'Sensitive to scaling', 'Less interpretable'],
                'use_cases': ['Feature extraction', 'Visualization', 'Noise reduction']
            },
            't-SNE': {
                'full_name': 't-Distributed Stochastic Neighbor Embedding',
                'type': 'Non-linear',
                'objective': 'Preserve local structure',
                'key_param': 'perplexity',
                'pros': ['Excellent for visualization', 'Reveals clusters', 'Non-linear'],
                'cons': ['Slow', 'Non-deterministic', 'Not for feature extraction'],
                'use_cases': ['Data visualization', 'Cluster analysis', 'High-dim exploration']
            },
            'UMAP': {
                'full_name': 'Uniform Manifold Approximation and Projection',
                'type': 'Non-linear',
                'objective': 'Preserve both local and global structure',
                'pros': ['Faster than t-SNE', 'Preserves global structure', 'Scalable'],
                'cons': ['Many hyperparameters', 'Theory is complex', 'Non-deterministic'],
                'use_cases': ['Visualization', 'Feature extraction', 'Large datasets']
            },
            'LDA': {
                'full_name': 'Linear Discriminant Analysis',
                'type': 'Linear supervised',
                'objective': 'Maximize class separation',
                'requirement': 'Labeled data',
                'pros': ['Supervised', 'Good for classification', 'Interpretable'],
                'cons': ['Linear only', 'Assumes Gaussian', 'Limited components'],
                'use_cases': ['Feature extraction for classification', 'Face recognition']
            },
            'Autoencoders': {
                'type': 'Non-linear neural',
                'architecture': 'Encoder-decoder',
                'variants': ['Vanilla', 'Denoising', 'Variational', 'Sparse'],
                'pros': ['Very flexible', 'Non-linear', 'Can generate data'],
                'cons': ['Requires tuning', 'Computationally expensive', 'Black box'],
                'use_cases': ['Feature learning', 'Denoising', 'Anomaly detection']
            }
        }
    
    def visualize_dim_reduction_comparison(self):
        """Compare dimensionality reduction techniques"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.datasets import load_digits
        import umap
        
        # Load dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Apply different techniques
        techniques = [
            ('PCA', PCA(n_components=2)),
            ('t-SNE', TSNE(n_components=2, perplexity=30)),
            ('UMAP', umap.UMAP(n_components=2)),
            ('LDA', LinearDiscriminantAnalysis(n_components=2))
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, technique) in enumerate(techniques):
            # Transform data
            X_transformed = technique.fit_transform(X, y) if name == 'LDA' else technique.fit_transform(X)
            
            # Plot
            scatter = axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                      c=y, cmap='tab10', s=5, alpha=0.7)
            axes[idx].set_title(f'{name} Projection of Digits Dataset', fontsize=12)
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
            
            # Add colorbar
            if idx == 1:
                cbar = plt.colorbar(scatter, ax=axes[idx])
                cbar.set_label('Digit')
        
        plt.suptitle('Dimensionality Reduction Techniques Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()

# Compare dimensionality reduction
dim_red_summary = DimensionalityReductionSummary()
dim_red_summary.visualize_dim_reduction_comparison()
```

## 5. Algorithm Comparison Framework

### 5.1 Comprehensive Comparison Metrics

```python
class AlgorithmComparisonFramework:
    """Framework for comparing algorithms across multiple dimensions"""
    
    def __init__(self):
        self.comparison_criteria = {
            'Performance': ['Accuracy', 'Precision/Recall', 'F1-Score', 'AUC-ROC'],
            'Computational': ['Training Time', 'Prediction Time', 'Memory Usage'],
            'Data Requirements': ['Sample Size', 'Feature Types', 'Preprocessing'],
            'Interpretability': ['Model Transparency', 'Feature Importance', 'Decision Rules'],
            'Robustness': ['Outlier Handling', 'Noise Tolerance', 'Missing Data'],
            'Scalability': ['Large n', 'High dimensions', 'Streaming data']
        }
    
    def create_algorithm_scorecard(self):
        """Create comprehensive algorithm scorecard"""
        algorithms = ['Linear/Logistic Reg', 'Decision Tree', 'Random Forest', 
                     'SVM', 'Neural Network', 'KNN', 'Naive Bayes', 'K-Means']
        
        criteria = ['Accuracy', 'Speed', 'Interpretability', 'Scalability', 
                   'Robustness', 'Ease of Use']
        
        # Scores (1-5 scale)
        scores = np.array([
            [3, 5, 5, 5, 2, 5],  # Linear/Logistic
            [4, 4, 5, 3, 4, 4],  # Decision Tree
            [5, 3, 2, 4, 5, 4],  # Random Forest
            [5, 2, 1, 3, 4, 3],  # SVM
            [5, 1, 1, 5, 3, 2],  # Neural Network
            [3, 2, 3, 1, 2, 5],  # KNN
            [3, 5, 4, 4, 3, 5],  # Naive Bayes
            [3, 4, 3, 4, 2, 4]   # K-Means
        ])
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
        scores_normalized = scores / 5  # Normalize to 0-1
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(projection='polar'))
        axes = axes.ravel()
        
        for idx, (algorithm, score) in enumerate(zip(algorithms, scores_normalized)):
            ax = axes[idx]
            
            # Close the plot
            values = score.tolist()
            values += values[:1]
            angles_plot = angles + angles[:1]
            
            # Plot
            ax.plot(angles_plot, values, 'o-', linewidth=2)
            ax.fill(angles_plot, values, alpha=0.25)
            ax.set_xticks(angles)
            ax.set_xticklabels(criteria, size=8)
            ax.set_ylim(0, 1)
            ax.set_title(algorithm, size=10, y=1.1)
            ax.grid(True)
        
        plt.suptitle('Algorithm Scorecard - Multi-Criteria Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame(scores, index=algorithms, columns=criteria)
    
    def algorithm_selection_flowchart(self):
        """Create algorithm selection flowchart logic"""
        print("=== Algorithm Selection Guide ===\n")
        
        selection_logic = """
        1. What is your problem type?
           ├─ Regression → Go to 2a
           ├─ Classification → Go to 2b
           └─ Clustering → Go to 2c
        
        2a. Regression Problem:
            ├─ Linear relationship? → Linear Regression
            ├─ Need feature selection? → Lasso
            ├─ Multicollinearity? → Ridge
            ├─ Non-linear patterns? → 
            │   ├─ Interpretability needed? → Decision Tree
            │   └─ Best performance? → Random Forest / XGBoost
            └─ Very high dimensions? → SVR with linear kernel
        
        2b. Classification Problem:
            ├─ Linear separable? → Logistic Regression
            ├─ Small dataset? → 
            │   ├─ Text data? → Naive Bayes
            │   └─ General? → SVM
            ├─ Need probability? → Logistic Regression / Naive Bayes
            ├─ Non-linear boundary? →
            │   ├─ Interpretability? → Decision Tree
            │   └─ Performance? → Random Forest / XGBoost
            └─ Image/Complex patterns? → Neural Networks
        
        2c. Clustering Problem:
            ├─ Know number of clusters? → K-Means
            ├─ Arbitrary shapes? → DBSCAN
            ├─ Hierarchical structure? → Agglomerative Clustering
            └─ Probabilistic assignment? → Gaussian Mixture Models
        """
        
        print(selection_logic)
        
        # Create visual decision tree
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define nodes and connections for simplified view
        nodes = {
            'Start': (7, 10),
            'Regression': (3, 8),
            'Classification': (7, 8),
            'Clustering': (11, 8),
            'Linear Reg': (1, 6),
            'Tree/Forest': (5, 6),
            'Log Reg': (5, 6),
            'SVM/NN': (9, 6),
            'K-Means': (9, 6),
            'DBSCAN': (13, 6)
        }
        
        connections = [
            ('Start', 'Regression'),
            ('Start', 'Classification'),
            ('Start', 'Clustering'),
            ('Regression', 'Linear Reg'),
            ('Regression', 'Tree/Forest'),
            ('Classification', 'Log Reg'),
            ('Classification', 'SVM/NN'),
            ('Clustering', 'K-Means'),
            ('Clustering', 'DBSCAN')
        ]
        
        # Draw connections
        for start, end in connections:
            x_values = [nodes[start][0], nodes[end][0]]
            y_values = [nodes[start][1], nodes[end][1]]
            ax.plot(x_values, y_values, 'k-', alpha=0.5, linewidth=2)
        
        # Draw nodes
        for node, (x, y) in nodes.items():
            if node == 'Start':
                circle = plt.Circle((x, y), 0.5, color='lightgreen', ec='darkgreen', linewidth=2)
            elif node in ['Regression', 'Classification', 'Clustering']:
                circle = plt.Circle((x, y), 0.5, color='lightblue', ec='darkblue', linewidth=2)
            else:
                circle = plt.Circle((x, y), 0.5, color='lightyellow', ec='orange', linewidth=2)
            
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlim(0, 14)
        ax.set_ylim(5, 11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Simplified Algorithm Selection Decision Tree', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.show()

# Create comparison framework
comparison = AlgorithmComparisonFramework()
scorecard = comparison.create_algorithm_scorecard()
print("\nAlgorithm Scorecard:")
print(scorecard)
comparison.algorithm_selection_flowchart()
```

## 6. When to Use Which Algorithm

### 6.1 Decision Matrix

```python
class AlgorithmSelectionGuide:
    """Comprehensive guide for algorithm selection"""
    
    def __init__(self):
        self.use_case_matrix = {
            'Linear/Logistic Regression': {
                'when_to_use': [
                    'Linear relationships expected',
                    'Interpretability is crucial',
                    'Baseline model needed',
                    'Feature coefficients needed',
                    'Probabilistic output required'
                ],
                'when_not_to_use': [
                    'Non-linear patterns',
                    'Complex interactions',
                    'High-dimensional sparse data'
                ],
                'typical_applications': [
                    'Price prediction',
                    'Risk scoring',
                    'A/B testing',
                    'Feature importance'
                ]
            },
            'Decision Trees': {
                'when_to_use': [
                    'Non-linear relationships',
                    'Rule extraction needed',
                    'Mixed data types',
                    'No scaling required',
                    'Feature interactions'
                ],
                'when_not_to_use': [
                    'Smooth functions',
                    'Extrapolation needed',
                    'Small datasets',
                    'High variance concern'
                ],
                'typical_applications': [
                    'Credit approval',
                    'Medical diagnosis',
                    'Customer segmentation'
                ]
            },
            'Random Forest': {
                'when_to_use': [
                    'High accuracy needed',
                    'Non-linear patterns',
                    'Feature importance',
                    'Robust predictions',
                    'Handle missing values'
                ],
                'when_not_to_use': [
                    'Real-time prediction',
                    'Model transparency needed',
                    'Limited memory',
                    'Extrapolation required'
                ],
                'typical_applications': [
                    'Fraud detection',
                    'Customer churn',
                    'Feature selection',
                    'Bioinformatics'
                ]
            },
            'SVM': {
                'when_to_use': [
                    'High-dimensional data',
                    'Clear margin of separation',
                    'Binary classification',
                    'Text classification',
                    'Non-linear boundaries (with kernels)'
                ],
                'when_not_to_use': [
                    'Large datasets',
                    'Probability estimates crucial',
                    'Multi-class with many classes',
                    'Online learning needed'
                ],
                'typical_applications': [
                    'Text categorization',
                    'Image classification',
                    'Bioinformatics',
                    'Face recognition'
                ]
            },
            'Neural Networks': {
                'when_to_use': [
                    'Complex patterns',
                    'Large datasets available',
                    'Feature learning needed',
                    'State-of-the-art required',
                    'Unstructured data'
                ],
                'when_not_to_use': [
                    'Small datasets',
                    'Interpretability crucial',
                    'Limited computational resources',
                    'Quick training needed'
                ],
                'typical_applications': [
                    'Image recognition',
                    'Natural language processing',
                    'Speech recognition',
                    'Game playing'
                ]
            },
            'K-Means': {
                'when_to_use': [
                    'Spherical clusters expected',
                    'Known number of clusters',
                    'Large datasets',
                    'Even cluster sizes',
                    'Vector quantization'
                ],
                'when_not_to_use': [
                    'Non-convex shapes',
                    'Varying densities',
                    'Unknown cluster count',
                    'Categorical data only'
                ],
                'typical_applications': [
                    'Customer segmentation',
                    'Document clustering',
                    'Image compression',
                    'Preprocessing step'
                ]
            }
        }
    
    def create_use_case_heatmap(self):
        """Create heatmap showing algorithm suitability for different scenarios"""
        scenarios = [
            'Small dataset (<1000)',
            'Large dataset (>1M)',
            'High dimensions (>100)',
            'Real-time prediction',
            'Interpretability needed',
            'Non-linear patterns',
            'Missing data',
            'Categorical features',
            'Imbalanced classes',
            'Probabilistic output'
        ]
        
        algorithms = ['Linear/Log Reg', 'Decision Tree', 'Random Forest', 
                     'SVM', 'Neural Net', 'KNN', 'Naive Bayes']
        
        # Suitability scores (0-3: not suitable to highly suitable)
        suitability = np.array([
            [3, 0, 2, 3, 3, 1, 2, 2, 1, 3],  # Linear/Logistic
            [2, 1, 1, 3, 3, 3, 2, 3, 2, 2],  # Decision Tree
            [1, 3, 3, 1, 1, 3, 3, 3, 3, 2],  # Random Forest
            [2, 1, 3, 2, 1, 3, 1, 1, 2, 2],  # SVM
            [0, 3, 3, 2, 0, 3, 2, 2, 2, 3],  # Neural Network
            [2, 0, 0, 0, 2, 3, 1, 1, 1, 2],  # KNN
            [3, 2, 2, 3, 2, 0, 2, 3, 1, 3]   # Naive Bayes
        ])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(suitability, annot=True, cmap='RdYlGn', 
                   xticklabels=scenarios, yticklabels=algorithms,
                   cbar_kws={'label': 'Suitability (0=Poor, 3=Excellent)'})
        plt.title('Algorithm Suitability for Different Scenarios', fontsize=16, pad=20)
        plt.xlabel('Scenario', fontsize=12)
        plt.ylabel('Algorithm', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return suitability
    
    def generate_decision_rules(self):
        """Generate practical decision rules for algorithm selection"""
        decision_rules = {
            'Start Simple': """
            Always start with simple algorithms as baseline:
            - Regression: Linear Regression
            - Classification: Logistic Regression
            - Clustering: K-Means
            """,
            
            'Data Size Rules': """
            Small data (<1000 samples):
            - Prefer simple models (Linear, Naive Bayes)
            - Use cross-validation extensively
            - Consider ensemble methods carefully
            
            Large data (>100K samples):
            - Neural networks become viable
            - Consider mini-batch algorithms
            - Random sampling for prototyping
            """,
            
            'Dimensionality Rules': """
            High dimensions (p > n):
            - Regularization essential (Ridge, Lasso)
            - SVM often works well
            - Consider dimensionality reduction
            
            Very high dimensions (>10K):
            - Linear models often sufficient
            - Feature selection crucial
            - Sparse methods preferred
            """,
            
            'Interpretability Rules': """
            High interpretability needed:
            1. Linear/Logistic Regression
            2. Decision Trees (shallow)
            3. Naive Bayes
            
            Medium interpretability:
            1. Random Forest (with feature importance)
            2. GAM (Generalized Additive Models)
            
            Low interpretability acceptable:
            1. Neural Networks
            2. SVM with non-linear kernels
            3. Ensemble methods
            """,
            
            'Performance Rules': """
            When accuracy is paramount:
            1. Start with Random Forest/XGBoost
            2. Try neural networks if data is large
            3. Ensemble multiple models
            4. Extensive hyperparameter tuning
            
            When speed is crucial:
            1. Linear models for prediction
            2. Naive Bayes for classification
            3. Pre-computed/approximate methods
            4. Avoid kernel methods and deep networks
            """
        }
        
        for rule_type, rules in decision_rules.items():
            print(f"\n{'='*50}")
            print(f"{rule_type}")
            print('='*50)
            print(rules)
        
        return decision_rules

# Create selection guide
selection_guide = AlgorithmSelectionGuide()
suitability_matrix = selection_guide.create_use_case_heatmap()
decision_rules = selection_guide.generate_decision_rules()
```

## 7. Complexity Analysis

### 7.1 Time and Space Complexity

```python
class ComplexityAnalysis:
    """Analyze computational complexity of algorithms"""
    
    def __init__(self):
        self.complexity_data = {
            'Linear Regression': {
                'training_time': 'O(n·d²) to O(n·d³)',
                'prediction_time': 'O(d)',
                'space': 'O(d²)',
                'notes': 'Matrix inversion dominates'
            },
            'Logistic Regression': {
                'training_time': 'O(n·d·i)',
                'prediction_time': 'O(d)',
                'space': 'O(d)',
                'notes': 'i = iterations for convergence'
            },
            'Decision Tree': {
                'training_time': 'O(n·log(n)·d)',
                'prediction_time': 'O(log(n))',
                'space': 'O(n)',
                'notes': 'Depth affects complexity'
            },
            'Random Forest': {
                'training_time': 'O(k·n·log(n)·d·m)',
                'prediction_time': 'O(k·log(n))',
                'space': 'O(k·n)',
                'notes': 'k trees, m features per split'
            },
            'SVM': {
                'training_time': 'O(n²) to O(n³)',
                'prediction_time': 'O(n_sv·d)',
                'space': 'O(n²)',
                'notes': 'n_sv = number of support vectors'
            },
            'Neural Network': {
                'training_time': 'O(n·d·h·o·i)',
                'prediction_time': 'O(d·h·o)',
                'space': 'O(d·h + h·o)',
                'notes': 'h = hidden units, o = outputs'
            },
            'K-Means': {
                'training_time': 'O(n·k·d·i)',
                'prediction_time': 'O(k·d)',
                'space': 'O(n·d + k·d)',
                'notes': 'i = iterations to converge'
            },
            'KNN': {
                'training_time': 'O(1)',
                'prediction_time': 'O(n·d)',
                'space': 'O(n·d)',
                'notes': 'Can use KD-tree for O(log(n))'
            }
        }
    
    def plot_complexity_comparison(self):
        """Visualize complexity growth rates"""
        n_values = np.logspace(2, 6, 100)  # 100 to 1M samples
        
        # Define complexity functions
        complexity_functions = {
            'O(n)': lambda n: n,
            'O(n log n)': lambda n: n * np.log(n),
            'O(n²)': lambda n: n**2,
            'O(n³)': lambda n: n**3,
            'O(log n)': lambda n: np.log(n),
            'O(n·d·k)': lambda n: n * 20 * 5  # d=20, k=5
        }
        
        plt.figure(figsize=(12, 8))
        
        for label, func in complexity_functions.items():
            if label != 'O(n³)':  # Skip n³ for visualization
                plt.loglog(n_values, func(n_values), label=label, linewidth=2)
        
        # Add algorithm annotations
        algorithm_complexities = {
            'Linear Reg': (1000, 1e6, 'O(n·d²)'),
            'Tree': (10000, 1e6, 'O(n log n)'),
            'SVM': (1000, 1e9, 'O(n²)'),
            'KNN Predict': (10000, 1e8, 'O(n·d)')
        }
        
        for algo, (x, y, complexity) in algorithm_complexities.items():
            plt.annotate(f'{algo}\n{complexity}', xy=(x, y), 
                        xytext=(x*2, y*2), fontsize=9,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        plt.xlabel('Number of Samples (n)', fontsize=12)
        plt.ylabel('Time Complexity', fontsize=12)
        plt.title('Algorithm Complexity Growth Rates', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_complexity_table(self):
        """Create detailed complexity comparison table"""
        df = pd.DataFrame.from_dict(self.complexity_data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Algorithm'}, inplace=True)
        
        # Display formatted table
        print("\n" + "="*80)
        print("ALGORITHM COMPLEXITY COMPARISON")
        print("="*80)
        print(f"{'Algorithm':<20} {'Training':<20} {'Prediction':<15} {'Space':<15}")
        print("-"*80)
        
        for _, row in df.iterrows():
            print(f"{row['Algorithm']:<20} {row['training_time']:<20} "
                  f"{row['prediction_time']:<15} {row['space']:<15}")
        
        print("\nLegend:")
        print("- n: number of samples")
        print("- d: number of features")
        print("- k: number of clusters/trees")
        print("- i: iterations")
        print("- h: hidden units")
        print("- o: output units")
        
        return df

# Analyze complexity
complexity = ComplexityAnalysis()
complexity.plot_complexity_comparison()
complexity_df = complexity.create_complexity_table()
```

## 8. Ensemble Methods

### 8.1 Ensemble Strategies

```python
class EnsembleMethodsSummary:
    """Summary of ensemble learning methods"""
    
    def __init__(self):
        self.ensemble_types = {
            'Bagging': {
                'principle': 'Bootstrap aggregating',
                'variance_reduction': 'High',
                'bias_reduction': 'Low',
                'parallel': True,
                'examples': ['Random Forest', 'Extra Trees'],
                'when_to_use': 'High variance base learners'
            },
            'Boosting': {
                'principle': 'Sequential learning from mistakes',
                'variance_reduction': 'Medium',
                'bias_reduction': 'High',
                'parallel': False,
                'examples': ['AdaBoost', 'XGBoost', 'LightGBM'],
                'when_to_use': 'High bias base learners'
            },
            'Stacking': {
                'principle': 'Meta-learning from base models',
                'variance_reduction': 'High',
                'bias_reduction': 'High',
                'parallel': True,
                'examples': ['Stacked Generalization'],
                'when_to_use': 'Diverse base learners available'
            },
            'Voting': {
                'principle': 'Democratic decision',
                'types': ['Hard voting', 'Soft voting'],
                'variance_reduction': 'Medium',
                'bias_reduction': 'Low',
                'examples': ['Majority Vote', 'Weighted Average'],
                'when_to_use': 'Multiple good models'
            }
        }
    
    def visualize_ensemble_methods(self):
        """Visualize how different ensemble methods work"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Bagging visualization
        ax = axes[0, 0]
        ax.text(0.5, 0.9, 'Bagging', ha='center', fontsize=14, fontweight='bold')
        
        # Draw bootstrap samples
        for i in range(3):
            rect = plt.Rectangle((0.1 + i*0.3, 0.6), 0.2, 0.2, 
                               fill=True, alpha=0.3, color=f'C{i}')
            ax.add_patch(rect)
            ax.text(0.2 + i*0.3, 0.7, f'Sample {i+1}', ha='center', fontsize=10)
            ax.arrow(0.2 + i*0.3, 0.55, 0, -0.1, head_width=0.03, color=f'C{i}')
            ax.text(0.2 + i*0.3, 0.4, f'Model {i+1}', ha='center', fontsize=10)
        
        ax.arrow(0.5, 0.3, 0, -0.1, head_width=0.05, color='black')
        ax.text(0.5, 0.1, 'Average/Vote', ha='center', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Boosting visualization
        ax = axes[0, 1]
        ax.text(0.5, 0.9, 'Boosting', ha='center', fontsize=14, fontweight='bold')
        
        # Sequential models
        for i in range(3):
            y_pos = 0.7 - i*0.2
            rect = plt.Rectangle((0.3, y_pos), 0.4, 0.1, 
                               fill=True, alpha=0.3, color=f'C{i}')
            ax.add_patch(rect)
            ax.text(0.5, y_pos + 0.05, f'Model {i+1}', ha='center', fontsize=10)
            if i < 2:
                ax.arrow(0.5, y_pos - 0.02, 0, -0.08, head_width=0.03, color='black')
        
        ax.text(0.5, 0.1, 'Weighted Sum', ha='center', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Stacking visualization
        ax = axes[1, 0]
        ax.text(0.5, 0.9, 'Stacking', ha='center', fontsize=14, fontweight='bold')
        
        # Base models
        for i in range(3):
            rect = plt.Rectangle((0.1 + i*0.3, 0.6), 0.2, 0.15, 
                               fill=True, alpha=0.3, color=f'C{i}')
            ax.add_patch(rect)
            ax.text(0.2 + i*0.3, 0.675, f'Base {i+1}', ha='center', fontsize=9)
            ax.arrow(0.2 + i*0.3, 0.55, 0, -0.1, head_width=0.02, color=f'C{i}')
        
        # Meta model
        rect = plt.Rectangle((0.25, 0.25), 0.5, 0.15, 
                           fill=True, alpha=0.3, color='gray')
        ax.add_patch(rect)
        ax.text(0.5, 0.325, 'Meta Model', ha='center', fontsize=11, fontweight='bold')
        ax.arrow(0.5, 0.2, 0, -0.05, head_width=0.03, color='black')
        ax.text(0.5, 0.1, 'Final Prediction', ha='center', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Voting visualization
        ax = axes[1, 1]
        ax.text(0.5, 0.9, 'Voting', ha='center', fontsize=14, fontweight='bold')
        
        # Models with predictions
        predictions = ['A', 'B', 'A']
        for i in range(3):
            rect = plt.Rectangle((0.1 + i*0.3, 0.5), 0.2, 0.2, 
                               fill=True, alpha=0.3, color=f'C{i}')
            ax.add_patch(rect)
            ax.text(0.2 + i*0.3, 0.6, f'Model {i+1}', ha='center', fontsize=10)
            ax.text(0.2 + i*0.3, 0.45, f'Pred: {predictions[i]}', ha='center', fontsize=9)
        
        ax.text(0.5, 0.25, 'Majority: A (2/3)', ha='center', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.suptitle('Ensemble Methods Visualization', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def ensemble_performance_comparison(self):
        """Compare ensemble vs single model performance"""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                     VotingClassifier, BaggingClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Generate dataset
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                 n_redundant=5, random_state=42)
        
        # Define models
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=5),
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(probability=True),
            'Bagging (DT)': BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=50),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'AdaBoost': AdaBoostClassifier(n_estimators=50),
            'Voting (Soft)': VotingClassifier([
                ('dt', DecisionTreeClassifier(max_depth=5)),
                ('lr', LogisticRegression()),
                ('svm', SVC(probability=True))
            ], voting='soft')
        }
        
        # Evaluate models
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        # Visualize results
        plt.figure(figsize=(10, 6))
        
        names = list(results.keys())
        means = [results[name]['mean'] for name in names]
        stds = [results[name]['std'] for name in names]
        
        # Color code: single models vs ensembles
        colors = ['lightblue', 'lightblue', 'lightblue', 'lightgreen', 
                 'lightgreen', 'lightgreen', 'lightgreen']
        
        bars = plt.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Single Models vs Ensemble Methods Performance', fontsize=14)
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        single_patch = plt.Rectangle((0, 0), 1, 1, fc='lightblue', label='Single Model')
        ensemble_patch = plt.Rectangle((0, 0), 1, 1, fc='lightgreen', label='Ensemble')
        plt.legend(handles=[single_patch, ensemble_patch], loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        return results

# Analyze ensemble methods
ensemble_summary = EnsembleMethodsSummary()
ensemble_summary.visualize_ensemble_methods()
ensemble_results = ensemble_summary.ensemble_performance_comparison()
```

## 9. Algorithm Selection Guide

### 9.1 Practical Selection Framework

```python
class PracticalAlgorithmSelection:
    """Practical framework for algorithm selection"""
    
    def create_selection_checklist(self):
        """Create comprehensive selection checklist"""
        checklist = """
        ╔════════════════════════════════════════════════════════════════╗
        ║              ALGORITHM SELECTION CHECKLIST                      ║
        ╚════════════════════════════════════════════════════════════════╝
        
        1. DEFINE THE PROBLEM
        □ Supervised or Unsupervised?
        □ Classification, Regression, or Clustering?
        □ What is the business objective?
        □ What are the constraints (time, resources)?
        
        2. UNDERSTAND YOUR DATA
        □ How many samples? _______
        □ How many features? _______
        □ Data types (numerical/categorical)?
        □ Missing values present?
        □ Class balance (if classification)?
        □ Linear or non-linear relationships?
        
        3. REQUIREMENTS
        □ Interpretability: High / Medium / Low
        □ Prediction speed: Critical / Important / Not Important
        □ Training time: Hours / Days / Flexible
        □ Accuracy: State-of-art / Good enough / Baseline
        
        4. ALGORITHM SHORT LIST
        Based on above, consider:
        
        For Regression:
        □ Linear Regression (baseline, interpretable)
        □ Random Forest (non-linear, robust)
        □ XGBoost (performance)
        □ Neural Network (complex patterns)
        
        For Classification:
        □ Logistic Regression (baseline, probability)
        □ Random Forest (general purpose)
        □ SVM (high dimensions)
        □ XGBoost (performance)
        
        For Clustering:
        □ K-Means (spherical, known k)
        □ DBSCAN (arbitrary shapes)
        □ Hierarchical (dendrogram needed)
        □ GMM (soft clustering)
        
        5. EVALUATION PLAN
        □ Cross-validation strategy
        □ Metrics to optimize
        □ Baseline performance
        □ Statistical significance testing
        
        6. IMPLEMENTATION STEPS
        1. Start with simplest baseline
        2. Evaluate against requirements
        3. Try 2-3 algorithms from shortlist
        4. Tune hyperparameters for best
        5. Consider ensemble if needed
        6. Validate on holdout set
        """
        
        print(checklist)
        return checklist
    
    def algorithm_recommendation_system(self, 
                                      problem_type='classification',
                                      n_samples=10000,
                                      n_features=20,
                                      interpretability='medium',
                                      speed_requirement='medium',
                                      data_type='mixed'):
        """Recommend algorithms based on requirements"""
        
        recommendations = []
        reasoning = []
        
        # Base recommendations by problem type
        if problem_type == 'classification':
            if n_samples < 1000:
                recommendations.append('Logistic Regression')
                reasoning.append('Small dataset - simple model preferred')
                if interpretability == 'low':
                    recommendations.append('SVM')
                    reasoning.append('SVM works well with small data')
            elif n_samples > 100000:
                recommendations.append('XGBoost')
                reasoning.append('Large dataset - can leverage complex model')
                if interpretability == 'low':
                    recommendations.append('Neural Network')
                    reasoning.append('Sufficient data for deep learning')
            else:
                recommendations.append('Random Forest')
                reasoning.append('Medium dataset - robust general purpose')
        
        elif problem_type == 'regression':
            if interpretability == 'high':
                recommendations.append('Linear Regression')
                reasoning.append('High interpretability requirement')
            if n_features > n_samples:
                recommendations.append('Ridge/Lasso')
                reasoning.append('High dimensional - regularization needed')
            if data_type == 'non-linear':
                recommendations.append('Random Forest')
                reasoning.append('Non-linear patterns detected')
        
        elif problem_type == 'clustering':
            if speed_requirement == 'high':
                recommendations.append('K-Means')
                reasoning.append('Fast clustering algorithm')
            recommendations.append('DBSCAN')
            reasoning.append('No assumptions about cluster shape')
        
        # Display recommendations
        print("\n" + "="*60)
        print("ALGORITHM RECOMMENDATIONS")
        print("="*60)
        print(f"Problem Type: {problem_type}")
        print(f"Dataset Size: {n_samples} samples, {n_features} features")
        print(f"Requirements: Interpretability={interpretability}, Speed={speed_requirement}")
        print("\nRecommended Algorithms:")
        print("-"*60)
        
        for algo, reason in zip(recommendations, reasoning):
            print(f"• {algo}")
            print(f"  Reason: {reason}")
        
        return recommendations, reasoning

# Create practical selection guide
selection = PracticalAlgorithmSelection()
selection.create_selection_checklist()

# Example recommendation
recommendations, reasoning = selection.algorithm_recommendation_system(
    problem_type='classification',
    n_samples=50000,
    n_features=100,
    interpretability='medium',
    speed_requirement='high',
    data_type='mixed'
)
```

## 10. Interview Questions

### Conceptual Questions

1. **Q: Explain the bias-variance tradeoff in the context of different algorithms.**
   A: The bias-variance tradeoff represents the balance between a model's ability to fit training data (low bias) and generalize to new data (low variance). Linear models like linear regression have high bias but low variance - they make strong assumptions about data relationships but produce consistent predictions. Complex models like neural networks have low bias but high variance - they can fit complex patterns but may overfit. Ensemble methods like Random Forest reduce variance through averaging while maintaining low bias.

2. **Q: When would you choose a generative model over a discriminative model?**
   A: Generative models (like Naive Bayes, GMM) model P(X,Y) and can generate new data samples, making them useful when:
   - You need to handle missing data
   - You want to generate synthetic data
   - You have small datasets (they often need less data)
   - You need to model the data distribution
   
   Discriminative models (like SVM, logistic regression) model P(Y|X) directly and are preferred when:
   - You only care about classification/regression
   - You have sufficient labeled data
   - You want potentially better performance on the specific task

3. **Q: Explain why Random Forest typically outperforms individual decision trees.**
   A: Random Forest improves upon decision trees through:
   - **Variance Reduction**: By averaging multiple trees trained on different bootstrap samples
   - **Decorrelation**: Random feature selection at each split ensures trees are different
   - **Overfitting Resistance**: Individual trees can overfit, but averaging reduces this
   - **Stability**: Less sensitive to small changes in training data
   - **Feature Importance**: Provides robust feature importance through permutation

4. **Q: What factors determine whether to use L1 or L2 regularization?**
   A: Choose based on:
   - **L1 (Lasso)**: When you want feature selection, expect many irrelevant features, need sparse models, or want interpretability
   - **L2 (Ridge)**: When all features are somewhat relevant, you have multicollinearity, or you want to shrink coefficients smoothly
   - **Elastic Net**: When you have correlated features and want both selection and grouping

5. **Q: How do you decide between parametric and non-parametric algorithms?**
   A: Consider:
   - **Parametric** (linear regression, logistic regression): Use when you have prior knowledge about data distribution, limited data, need fast prediction, or require interpretability
   - **Non-parametric** (KNN, decision trees, kernel SVM): Use when data relationships are complex/unknown, you have sufficient data, or flexibility is more important than interpretability

### Algorithm Comparison Questions

6. **Q: Compare and contrast SVM with logistic regression.**
   A: 
   - **Objective**: SVM maximizes margin; logistic regression maximizes likelihood
   - **Loss**: SVM uses hinge loss; logistic uses log loss
   - **Output**: SVM gives class labels; logistic gives probabilities
   - **Outliers**: SVM more robust due to support vectors; logistic sensitive
   - **Kernels**: SVM easily extends to non-linear; logistic needs feature engineering
   - **Multi-class**: Logistic natural with softmax; SVM needs OvR or OvO

7. **Q: When would you use PCA vs t-SNE vs UMAP?**
   A:
   - **PCA**: Linear dimensionality reduction, preserves global structure, fast, deterministic. Use for feature extraction, denoising, or when interpretability matters
   - **t-SNE**: Non-linear, preserves local structure, great for visualization. Use for exploring clusters in high-dimensional data, not for feature extraction
   - **UMAP**: Non-linear, preserves both local and global structure, faster than t-SNE. Use for visualization or feature extraction when non-linear relationships exist

8. **Q: Explain the differences between bagging and boosting.**
   A:
   - **Training**: Bagging trains in parallel; boosting trains sequentially
   - **Data Sampling**: Bagging uses bootstrap samples; boosting uses weighted samples
   - **Focus**: Bagging reduces variance; boosting reduces bias
   - **Base Learners**: Bagging uses strong learners; boosting uses weak learners
   - **Combination**: Bagging averages/votes; boosting uses weighted sum
   - **Overfitting**: Bagging resistant; boosting can overfit with too many iterations

9. **Q: Compare K-means with DBSCAN for clustering.**
   A:
   - **Cluster Shape**: K-means assumes spherical; DBSCAN finds arbitrary shapes
   - **Number of Clusters**: K-means requires k; DBSCAN determines automatically
   - **Outliers**: K-means assigns all points; DBSCAN identifies outliers
   - **Parameters**: K-means needs k; DBSCAN needs eps and min_samples
   - **Scalability**: K-means is O(nkd); DBSCAN is O(n log n) with index
   - **Determinism**: K-means depends on initialization; DBSCAN mostly deterministic

10. **Q: When would you choose a tree-based model over a neural network?**
    A: Choose tree-based when:
    - Limited training data (<10k samples)
    - Tabular/structured data
    - Need feature importance
    - Interpretability matters
    - Quick training needed
    - No specialized hardware (GPU)
    
    Choose neural networks when:
    - Large datasets available
    - Unstructured data (images, text, audio)
    - Complex patterns exist
    - State-of-the-art performance needed
    - Resources available for tuning

### Complexity and Scalability Questions

11. **Q: How do different algorithms scale with the number of features?**
    A:
    - **Well-scaling**: Linear models (O(d)), Naive Bayes (O(d)), Random Forest (O(d√d))
    - **Moderate**: SVM (O(d) but kernel computation can be expensive), Neural Networks (depends on architecture)
    - **Poor**: KNN (curse of dimensionality), K-means (distance calculations)
    - **Solutions**: Feature selection, dimensionality reduction, regularization, sparse representations

12. **Q: Explain the computational trade-offs in ensemble methods.**
    A:
    - **Training Time**: Increases linearly with number of base models (parallelizable in bagging)
    - **Prediction Time**: Increases linearly (critical for real-time systems)
    - **Memory**: Stores multiple models (can be prohibitive)
    - **Trade-offs**: Better performance vs resource usage
    - **Optimizations**: Model compression, early stopping, selective ensemble

13. **Q: How do you handle algorithms that don't scale to big data?**
    A:
    - **Sampling**: Train on representative subset
    - **Mini-batch**: Use stochastic/mini-batch versions
    - **Distributed**: Use distributed implementations (MLlib, Dask)
    - **Approximations**: Use approximate algorithms (approximate KNN)
    - **Feature Engineering**: Reduce dimensionality first
    - **Different Algorithm**: Switch to scalable alternative

### Practical Application Questions

14. **Q: How do you select algorithms for imbalanced classification?**
    A: Consider:
    - **Algorithms**: Tree-based methods handle imbalance better; SVM with class weights; 
    - **Sampling**: SMOTE, undersampling, or combination
    - **Metrics**: Use precision-recall, F1, AUC-ROC instead of accuracy
    - **Ensemble**: Bagging with balanced bootstrap samples
    - **Cost-sensitive**: Algorithms that support class weights
    - **Threshold**: Adjust decision threshold based on costs

15. **Q: What's your algorithm selection process for a new problem?**
    A:
    1. **Understand**: Problem type, constraints, data characteristics
    2. **Baseline**: Start with simple model (linear/logistic regression)
    3. **Iterate**: Try 2-3 algorithms from different families
    4. **Validate**: Use appropriate CV strategy
    5. **Tune**: Optimize hyperparameters for best performers
    6. **Ensemble**: Consider combining if performance critical
    7. **Trade-offs**: Balance performance with requirements

16. **Q: How do you handle mixed data types (numerical + categorical)?**
    A:
    - **Tree-based**: Handle naturally without preprocessing
    - **Linear models**: One-hot encode categorical, scale numerical
    - **Neural networks**: Embeddings for categorical, normalization for numerical
    - **SVM**: Kernel for categorical, scaling essential
    - **Distance-based**: Careful encoding and scaling critical
    - **Pipeline**: Use ColumnTransformer for consistent preprocessing

17. **Q: Explain your approach to algorithm selection for time series.**
    A:
    - **Classical**: ARIMA, SARIMA for univariate with clear patterns
    - **Machine Learning**: Random Forest, XGBoost with lag features
    - **Deep Learning**: LSTM, GRU for complex patterns, multiple series
    - **Considerations**: Stationarity, seasonality, external variables
    - **Validation**: Time-based splits, no random shuffling
    - **Feature Engineering**: Lag features, rolling statistics, time-based features

### Advanced Questions

18. **Q: How do you handle concept drift in algorithm selection?**
    A:
    - **Detection**: Monitor performance metrics over time
    - **Adaptive Algorithms**: Online learning, incremental models
    - **Retraining**: Periodic retraining with recent data
    - **Ensemble**: Weighted ensemble with recent models weighted higher
    - **Algorithm Choice**: Prefer adaptable algorithms (SGD, incremental trees)
    - **Validation**: Use time-based validation to detect drift early

19. **Q: Compare algorithms for multi-label classification.**
    A:
    - **Problem Transformation**: Binary Relevance, Classifier Chains, Label Powerset
    - **Algorithm Adaptation**: Multi-label k-NN, Multi-label trees
    - **Neural Networks**: Single network with sigmoid outputs
    - **Considerations**: Label correlation, computational complexity
    - **Evaluation**: Hamming loss, subset accuracy, macro/micro F1

20. **Q: How do you select algorithms for online learning scenarios?**
    A:
    - **Requirements**: Incremental updates, bounded memory, fast adaptation
    - **Algorithms**: SGD variants, Hoeffding trees, incremental clustering
    - **Trade-offs**: Stability vs plasticity
    - **Evaluation**: Prequential evaluation, regret bounds
    - **Challenges**: Concept drift, verification latency
    - **Applications**: Recommendation systems, fraud detection, IoT

21. **Q: Explain algorithm selection for interpretable ML.**
    A:
    - **Linear Models**: Coefficients directly interpretable
    - **GAMs**: Additive models show individual feature effects
    - **Decision Trees**: Simple trees with path rules
    - **Rule-Based**: RIPPER, decision lists
    - **Post-hoc**: LIME, SHAP for black-box models
    - **Trade-offs**: Performance vs interpretability

22. **Q: How do you approach algorithm selection for edge computing?**
    A:
    - **Constraints**: Memory, computation, power consumption
    - **Model Compression**: Quantization, pruning, distillation
    - **Algorithms**: Linear models, small trees, compressed networks
    - **Trade-offs**: Accuracy vs resource usage
    - **Deployment**: Consider prediction latency and model size
    - **Updates**: How to handle model updates on edge