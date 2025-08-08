# Day 25: Hyperparameter Tuning

## Table of Contents
1. [Introduction](#1-introduction)
2. [Understanding Hyperparameters](#2-understanding-hyperparameters)
3. [Grid Search](#3-grid-search)
4. [Random Search](#4-random-search)
5. [Bayesian Optimization](#5-bayesian-optimization)
6. [Advanced Techniques](#6-advanced-techniques)
7. [Practical Implementation](#7-practical-implementation)
8. [Best Practices](#8-best-practices)
9. [Case Studies](#9-case-studies)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

Hyperparameter tuning is the process of finding the optimal configuration of hyperparameters that results in the best model performance. Unlike model parameters that are learned during training, hyperparameters are set before training begins and control the learning process itself.

### Why Hyperparameter Tuning Matters

1. **Performance Impact**: Can dramatically improve model performance
2. **Generalization**: Proper tuning helps avoid overfitting/underfitting
3. **Resource Optimization**: Find the best trade-off between performance and computational cost
4. **Model Behavior**: Hyperparameters control fundamental aspects of model learning
5. **Competitive Edge**: Often the difference between good and state-of-the-art results

### Challenges in Hyperparameter Tuning

- **High Dimensionality**: Many hyperparameters to tune
- **Expensive Evaluation**: Training can be computationally costly
- **Non-convex Optimization**: Multiple local optima
- **Interdependencies**: Hyperparameters often interact
- **Discrete and Continuous**: Mixed types of hyperparameters

## 2. Understanding Hyperparameters

### 2.1 Types of Hyperparameters

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

class HyperparameterAnalyzer:
    """Analyze effects of different hyperparameters"""
    
    def __init__(self):
        self.hyperparameter_types = {
            'Model Capacity': {
                'description': 'Controls model complexity',
                'examples': ['n_estimators', 'max_depth', 'n_hidden_units'],
                'effect': 'Higher values → more complex model → risk of overfitting'
            },
            'Regularization': {
                'description': 'Prevents overfitting',
                'examples': ['alpha', 'C', 'l1_ratio', 'dropout_rate'],
                'effect': 'Higher regularization → simpler model → risk of underfitting'
            },
            'Learning Dynamics': {
                'description': 'Controls optimization process',
                'examples': ['learning_rate', 'momentum', 'batch_size'],
                'effect': 'Affects convergence speed and stability'
            },
            'Algorithmic': {
                'description': 'Algorithm-specific settings',
                'examples': ['kernel', 'metric', 'linkage'],
                'effect': 'Changes fundamental algorithm behavior'
            }
        }
    
    def plot_hyperparameter_effect(self, model, X, y, param_name, param_range, cv=5):
        """Visualize effect of a hyperparameter on performance"""
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'b-', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'r-', label='Validation score')
        plt.fill_between(param_range, val_mean - val_std, 
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Validation Curve: Effect of {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Mark optimal value
        optimal_idx = np.argmax(val_mean)
        optimal_value = param_range[optimal_idx]
        plt.axvline(x=optimal_value, color='green', linestyle='--', 
                   label=f'Optimal: {optimal_value}')
        
        plt.show()
        
        return optimal_value, val_mean[optimal_idx]
    
    def analyze_interactions(self, model, X, y, param1_name, param1_range, 
                           param2_name, param2_range, cv=3):
        """Analyze interaction between two hyperparameters"""
        scores = np.zeros((len(param1_range), len(param2_range)))
        
        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                model.set_params(**{param1_name: p1, param2_name: p2})
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                scores[i, j] = np.mean(cv_scores)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(scores, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Validation Score')
        
        # Set ticks
        plt.xticks(range(len(param2_range)), param2_range)
        plt.yticks(range(len(param1_range)), param1_range)
        plt.xlabel(param2_name)
        plt.ylabel(param1_name)
        plt.title(f'Hyperparameter Interaction: {param1_name} vs {param2_name}')
        
        # Mark optimal
        optimal_idx = np.unravel_index(scores.argmax(), scores.shape)
        plt.plot(optimal_idx[1], optimal_idx[0], 'b*', markersize=15)
        
        plt.tight_layout()
        plt.show()
        
        return scores, (param1_range[optimal_idx[0]], param2_range[optimal_idx[1]])

# Demonstrate hyperparameter effects
def demonstrate_hyperparameter_effects():
    """Show how different hyperparameters affect model performance"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    analyzer = HyperparameterAnalyzer()
    
    # Example 1: Tree depth in Random Forest
    print("Example 1: Effect of max_depth in Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    depth_range = [2, 5, 10, 15, 20, 30, None]
    optimal_depth, score = analyzer.plot_hyperparameter_effect(
        rf, X, y, 'max_depth', depth_range
    )
    
    # Example 2: Regularization in SVM
    print("\nExample 2: Effect of C in SVM")
    svm = SVC(kernel='rbf', random_state=42)
    C_range = np.logspace(-3, 3, 7)
    optimal_C, score = analyzer.plot_hyperparameter_effect(
        svm, X, y, 'C', C_range
    )
    
    # Example 3: Interaction between hyperparameters
    print("\nExample 3: Interaction between C and gamma in SVM")
    gamma_range = np.logspace(-3, 1, 5)
    scores, optimal_params = analyzer.analyze_interactions(
        svm, X, y, 'C', C_range[:5], 'gamma', gamma_range
    )
    
    print(f"Optimal parameters: C={optimal_params[0]}, gamma={optimal_params[1]}")
```

### 2.2 Hyperparameter Importance

```python
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

def analyze_hyperparameter_importance():
    """Analyze relative importance of different hyperparameters"""
    from sklearn.datasets import make_regression
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Define hyperparameter ranges
    param_ranges = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    # Analyze individual impact
    base_model = GradientBoostingRegressor(random_state=42)
    base_score = cross_val_score(base_model, X, y, cv=5, scoring='r2').mean()
    
    importance_results = {}
    
    for param, values in param_ranges.items():
        scores = []
        for value in values:
            model = GradientBoostingRegressor(random_state=42)
            model.set_params(**{param: value})
            score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
            scores.append(score)
        
        # Calculate importance as variance in scores
        importance = np.var(scores)
        importance_results[param] = {
            'importance': importance,
            'score_range': (min(scores), max(scores)),
            'best_value': values[np.argmax(scores)]
        }
    
    # Visualize importance
    params = list(importance_results.keys())
    importances = [importance_results[p]['importance'] for p in params]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(params, importances)
    plt.xlabel('Hyperparameter')
    plt.ylabel('Importance (Variance in Performance)')
    plt.title('Relative Importance of Hyperparameters')
    
    # Add value labels
    for bar, param in zip(bars, params):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{importance_results[param]["best_value"]}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return importance_results
```

## 3. Grid Search

### 3.1 Implementation from Scratch

```python
from itertools import product
from sklearn.model_selection import cross_val_score
import time

class GridSearchFromScratch:
    """Grid search implementation from scratch"""
    
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy', n_jobs=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = []
        
    def _generate_candidates(self):
        """Generate all parameter combinations"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def fit(self, X, y):
        """Perform grid search"""
        candidates = list(self._generate_candidates())
        n_candidates = len(candidates)
        
        print(f"Fitting {self.cv} folds for each of {n_candidates} candidates, "
              f"totalling {self.cv * n_candidates} fits")
        
        start_time = time.time()
        
        for i, params in enumerate(candidates):
            # Set parameters
            model = clone(self.estimator)
            model.set_params(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                   scoring=self.scoring, n_jobs=self.n_jobs)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Store results
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            }
            self.cv_results_.append(result)
            
            # Update best
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
            
            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {i+1}/{n_candidates} "
                      f"({elapsed:.1f}s, best score: {self.best_score_:.4f})")
        
        # Refit best model
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
        total_time = time.time() - start_time
        print(f"\nGrid search completed in {total_time:.1f}s")
        print(f"Best score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return self
    
    def predict(self, X):
        """Predict using best model"""
        return self.best_estimator_.predict(X)
    
    def get_search_results(self):
        """Get detailed search results as DataFrame"""
        results_df = pd.DataFrame(self.cv_results_)
        results_df = results_df.sort_values('mean_score', ascending=False)
        return results_df

# Visualize grid search results
def visualize_grid_search(param_grid, cv_results):
    """Visualize grid search results"""
    if len(param_grid) == 1:
        # 1D parameter search
        param_name = list(param_grid.keys())[0]
        param_values = [r['params'][param_name] for r in cv_results]
        mean_scores = [r['mean_score'] for r in cv_results]
        std_scores = [r['std_score'] for r in cv_results]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(param_values, mean_scores, yerr=std_scores, 
                    marker='o', capsize=5)
        plt.xlabel(param_name)
        plt.ylabel('CV Score')
        plt.title('Grid Search Results')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    elif len(param_grid) == 2:
        # 2D parameter search
        param_names = list(param_grid.keys())
        param1_values = sorted(set(r['params'][param_names[0]] for r in cv_results))
        param2_values = sorted(set(r['params'][param_names[1]] for r in cv_results))
        
        # Create score matrix
        scores = np.zeros((len(param1_values), len(param2_values)))
        
        for result in cv_results:
            i = param1_values.index(result['params'][param_names[0]])
            j = param2_values.index(result['params'][param_names[1]])
            scores[i, j] = result['mean_score']
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(scores, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='CV Score')
        
        plt.xticks(range(len(param2_values)), param2_values)
        plt.yticks(range(len(param1_values)), param1_values)
        plt.xlabel(param_names[1])
        plt.ylabel(param_names[0])
        plt.title('Grid Search Results Heatmap')
        
        # Mark best
        best_idx = np.unravel_index(scores.argmax(), scores.shape)
        plt.plot(best_idx[1], best_idx[0], 'b*', markersize=15)
        
        plt.tight_layout()
        plt.show()
```

### 3.2 Efficient Grid Search with Scikit-learn

```python
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

class EfficientGridSearch:
    """Efficient grid search with various optimizations"""
    
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy',
                 n_jobs=-1, cache_cv=True, verbose=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cache_cv = cache_cv
        self.verbose = verbose
        
    def search_with_pruning(self, X, y, early_stopping_rounds=10):
        """Grid search with early stopping for poor configurations"""
        from sklearn.model_selection import StratifiedKFold
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(self.param_grid))
        
        for params in param_combinations:
            model = clone(self.estimator)
            model.set_params(**params)
            
            # Quick evaluation with fewer folds
            quick_cv = StratifiedKFold(n_splits=min(3, self.cv), shuffle=True, random_state=42)
            quick_scores = cross_val_score(model, X, y, cv=quick_cv, scoring=self.scoring)
            quick_mean = np.mean(quick_scores)
            
            # Skip if clearly worse than best
            if quick_mean < best_score - 0.05:  # 5% tolerance
                results.append({
                    'params': params,
                    'score': quick_mean,
                    'pruned': True
                })
                continue
            
            # Full evaluation
            full_scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            full_mean = np.mean(full_scores)
            
            results.append({
                'params': params,
                'score': full_mean,
                'pruned': False
            })
            
            if full_mean > best_score:
                best_score = full_mean
                best_params = params
        
        # Summary
        n_pruned = sum(1 for r in results if r['pruned'])
        print(f"Pruned {n_pruned}/{len(param_combinations)} configurations")
        print(f"Best score: {best_score:.4f}")
        print(f"Best params: {best_params}")
        
        return best_params, results
    
    def multi_metric_grid_search(self, X, y, scoring_dict):
        """Grid search optimizing multiple metrics"""
        from sklearn.model_selection import cross_validate
        
        results = []
        
        param_combinations = list(ParameterGrid(self.param_grid))
        
        for params in param_combinations:
            model = clone(self.estimator)
            model.set_params(**params)
            
            # Evaluate multiple metrics
            cv_results = cross_validate(model, X, y, cv=self.cv, 
                                      scoring=scoring_dict, n_jobs=self.n_jobs)
            
            result = {'params': params}
            for metric_name in scoring_dict:
                scores = cv_results[f'test_{metric_name}']
                result[f'{metric_name}_mean'] = np.mean(scores)
                result[f'{metric_name}_std'] = np.std(scores)
            
            results.append(result)
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Find Pareto optimal solutions
        pareto_optimal = self._find_pareto_optimal(results_df, list(scoring_dict.keys()))
        
        return results_df, pareto_optimal
    
    def _find_pareto_optimal(self, results_df, metrics):
        """Find Pareto optimal parameter configurations"""
        pareto_optimal = []
        
        for idx, row in results_df.iterrows():
            dominated = False
            
            for other_idx, other_row in results_df.iterrows():
                if idx != other_idx:
                    # Check if other_row dominates row
                    if all(other_row[f'{m}_mean'] >= row[f'{m}_mean'] for m in metrics) and \
                       any(other_row[f'{m}_mean'] > row[f'{m}_mean'] for m in metrics):
                        dominated = True
                        break
            
            if not dominated:
                pareto_optimal.append(idx)
        
        return results_df.iloc[pareto_optimal]
```

## 4. Random Search

### 4.1 Random Search Implementation

```python
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint, loguniform

class RandomSearchOptimizer:
    """Advanced random search implementation"""
    
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, 
                 scoring='accuracy', n_jobs=-1, random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def _sample_params(self):
        """Sample parameters from distributions"""
        sampler = ParameterSampler(self.param_distributions, 
                                 n_iter=self.n_iter, 
                                 random_state=self.random_state)
        return list(sampler)
    
    def search(self, X, y):
        """Perform random search"""
        param_samples = self._sample_params()
        results = []
        
        best_score = -np.inf
        best_params = None
        
        print(f"Random search: {self.n_iter} parameter settings")
        
        for i, params in enumerate(param_samples):
            model = clone(self.estimator)
            model.set_params(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=self.cv, 
                                   scoring=self.scoring, n_jobs=self.n_jobs)
            mean_score = np.mean(scores)
            
            results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': np.std(scores),
                'iteration': i
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{self.n_iter}, "
                      f"best score: {best_score:.4f}")
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        
        # Refit best model
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
        return self
    
    def plot_convergence(self):
        """Plot convergence of random search"""
        iterations = [r['iteration'] for r in self.cv_results_]
        scores = [r['mean_score'] for r in self.cv_results_]
        
        # Calculate running best
        running_best = []
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            running_best.append(current_best)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(iterations, scores, alpha=0.5, label='Sampled scores')
        plt.plot(iterations, running_best, 'r-', linewidth=2, label='Best score')
        plt.xlabel('Iteration')
        plt.ylabel('CV Score')
        plt.title('Random Search Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Compare Grid vs Random Search
def compare_search_methods():
    """Compare efficiency of grid search vs random search"""
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    # Define parameter space
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }
    
    param_distributions = {
        'C': loguniform(0.001, 100),
        'gamma': loguniform(0.0001, 1),
        'kernel': ['rbf', 'poly']
    }
    
    # Grid search
    print("Running Grid Search...")
    start_time = time.time()
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    grid_time = time.time() - start_time
    
    # Random search with same budget
    n_iter = len(ParameterGrid(param_grid))
    print(f"\nRunning Random Search ({n_iter} iterations)...")
    start_time = time.time()
    random_search = RandomizedSearchCV(SVC(), param_distributions, 
                                     n_iter=n_iter, cv=5, scoring='accuracy', 
                                     n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    random_time = time.time() - start_time
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Grid Search:")
    print(f"  Time: {grid_time:.1f}s")
    print(f"  Best Score: {grid_search.best_score_:.4f}")
    print(f"  Best Params: {grid_search.best_params_}")
    print(f"\nRandom Search:")
    print(f"  Time: {random_time:.1f}s")
    print(f"  Best Score: {random_search.best_score_:.4f}")
    print(f"  Best Params: {random_search.best_params_}")
    
    # Plot parameter exploration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Grid search coverage
    grid_params = pd.DataFrame(grid_search.cv_results_['params'])
    ax1.scatter(grid_params['C'], grid_params['gamma'], 
               c=grid_search.cv_results_['mean_test_score'],
               cmap='viridis', s=100, alpha=0.6)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('C')
    ax1.set_ylabel('gamma')
    ax1.set_title('Grid Search Coverage')
    
    # Random search coverage
    random_params = pd.DataFrame(random_search.cv_results_['params'])
    ax2.scatter(random_params['C'], random_params['gamma'],
               c=random_search.cv_results_['mean_test_score'],
               cmap='viridis', s=100, alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('C')
    ax2.set_ylabel('gamma')
    ax2.set_title('Random Search Coverage')
    
    # Add colorbars
    for ax in [ax1, ax2]:
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(grid_search.cv_results_['mean_test_score'])
        plt.colorbar(sm, ax=ax, label='CV Score')
    
    plt.suptitle('Parameter Space Exploration: Grid vs Random Search')
    plt.tight_layout()
    plt.show()
```

## 5. Bayesian Optimization

### 5.1 Gaussian Process-based Optimization

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    """Bayesian optimization using Gaussian Processes"""
    
    def __init__(self, func, bounds, n_calls=50, n_initial=5, 
                 acq_func='ei', xi=0.01, random_state=None):
        self.func = func
        self.bounds = bounds
        self.n_calls = n_calls
        self.n_initial = n_initial
        self.acq_func = acq_func
        self.xi = xi  # Exploration parameter
        self.random_state = random_state
        
        # Initialize GP
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                         normalize_y=True, n_restarts_optimizer=5)
        
        # Storage
        self.X_obs = []
        self.y_obs = []
        
    def _acquisition(self, X, gp, y_max):
        """Compute acquisition function"""
        mean, std = gp.predict(X.reshape(-1, len(self.bounds)), return_std=True)
        
        if self.acq_func == 'ei':  # Expected Improvement
            z = (mean - y_max - self.xi) / std
            ei = std * (z * norm.cdf(z) + norm.pdf(z))
            return ei
        
        elif self.acq_func == 'ucb':  # Upper Confidence Bound
            return mean + 2 * std
        
        elif self.acq_func == 'pi':  # Probability of Improvement
            z = (mean - y_max - self.xi) / std
            return norm.cdf(z)
    
    def _propose_location(self):
        """Propose next sampling location"""
        if len(self.X_obs) < self.n_initial:
            # Random sampling for initial points
            return np.random.uniform(low=[b[0] for b in self.bounds],
                                   high=[b[1] for b in self.bounds])
        
        # Fit GP to observations
        self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
        
        # Find point with maximum acquisition value
        y_max = max(self.y_obs)
        
        # Multi-start optimization
        best_x = None
        best_acq = -np.inf
        
        for _ in range(10):
            x0 = np.random.uniform(low=[b[0] for b in self.bounds],
                                 high=[b[1] for b in self.bounds])
            
            res = minimize(lambda x: -self._acquisition(x, self.gp, y_max),
                         x0, bounds=self.bounds, method='L-BFGS-B')
            
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
        
        return best_x
    
    def optimize(self):
        """Run Bayesian optimization"""
        for i in range(self.n_calls):
            # Propose next point
            x_next = self._propose_location()
            
            # Evaluate function
            y_next = self.func(x_next)
            
            # Store observation
            self.X_obs.append(x_next)
            self.y_obs.append(y_next)
            
            # Progress
            if (i + 1) % 10 == 0:
                best_idx = np.argmax(self.y_obs)
                print(f"Iteration {i+1}: Best value = {self.y_obs[best_idx]:.4f}")
        
        # Return best found
        best_idx = np.argmax(self.y_obs)
        return self.X_obs[best_idx], self.y_obs[best_idx]
    
    def plot_optimization(self):
        """Visualize optimization process (for 1D or 2D)"""
        X_obs = np.array(self.X_obs)
        y_obs = np.array(self.y_obs)
        
        if len(self.bounds) == 1:
            # 1D visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot GP posterior
            x_plot = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
            X_plot = x_plot.reshape(-1, 1)
            
            self.gp.fit(X_obs, y_obs)
            y_mean, y_std = self.gp.predict(X_plot, return_std=True)
            
            ax1.plot(x_plot, y_mean, 'k-', label='GP mean')
            ax1.fill_between(x_plot, y_mean - 2*y_std, y_mean + 2*y_std,
                           alpha=0.3, color='blue', label='95% CI')
            ax1.scatter(X_obs, y_obs, c='red', s=50, zorder=5, label='Observations')
            ax1.set_xlabel('Parameter')
            ax1.set_ylabel('Objective')
            ax1.set_title('Gaussian Process Posterior')
            ax1.legend()
            
            # Plot acquisition function
            y_max = max(y_obs)
            acq_values = [self._acquisition(np.array([x]), self.gp, y_max) 
                         for x in x_plot]
            ax2.plot(x_plot, acq_values, 'g-')
            ax2.set_xlabel('Parameter')
            ax2.set_ylabel('Acquisition Value')
            ax2.set_title(f'Acquisition Function ({self.acq_func.upper()})')
            
            plt.tight_layout()
            plt.show()
            
        elif len(self.bounds) == 2:
            # 2D visualization
            fig = plt.figure(figsize=(15, 5))
            
            # Create grid
            x1_range = np.linspace(self.bounds[0][0], self.bounds[0][1], 50)
            x2_range = np.linspace(self.bounds[1][0], self.bounds[1][1], 50)
            X1, X2 = np.meshgrid(x1_range, x2_range)
            
            # GP predictions
            ax1 = fig.add_subplot(131)
            X_grid = np.column_stack([X1.ravel(), X2.ravel()])
            self.gp.fit(X_obs, y_obs)
            y_mean = self.gp.predict(X_grid).reshape(X1.shape)
            
            im1 = ax1.contourf(X1, X2, y_mean, levels=20, cmap='viridis')
            ax1.scatter(X_obs[:, 0], X_obs[:, 1], c='red', s=50, edgecolor='white')
            ax1.set_xlabel('Parameter 1')
            ax1.set_ylabel('Parameter 2')
            ax1.set_title('GP Mean Prediction')
            plt.colorbar(im1, ax=ax1)
            
            # Uncertainty
            ax2 = fig.add_subplot(132)
            _, y_std = self.gp.predict(X_grid, return_std=True)
            y_std = y_std.reshape(X1.shape)
            
            im2 = ax2.contourf(X1, X2, y_std, levels=20, cmap='Blues')
            ax2.scatter(X_obs[:, 0], X_obs[:, 1], c='red', s=50, edgecolor='white')
            ax2.set_xlabel('Parameter 1')
            ax2.set_ylabel('Parameter 2')
            ax2.set_title('GP Uncertainty (Std)')
            plt.colorbar(im2, ax=ax2)
            
            # Acquisition function
            ax3 = fig.add_subplot(133)
            y_max = max(y_obs)
            acq_values = np.array([self._acquisition(x, self.gp, y_max) 
                                 for x in X_grid]).reshape(X1.shape)
            
            im3 = ax3.contourf(X1, X2, acq_values, levels=20, cmap='Reds')
            ax3.scatter(X_obs[:, 0], X_obs[:, 1], c='blue', s=50, edgecolor='white')
            ax3.set_xlabel('Parameter 1')
            ax3.set_ylabel('Parameter 2')
            ax3.set_title(f'Acquisition Function ({self.acq_func.upper()})')
            plt.colorbar(im3, ax=ax3)
            
            plt.suptitle('Bayesian Optimization Visualization')
            plt.tight_layout()
            plt.show()

# Example: Optimize hyperparameters using Bayesian optimization
def bayesian_hyperparameter_tuning():
    """Use Bayesian optimization for hyperparameter tuning"""
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    # Define objective function
    def objective(params):
        C = 10 ** params[0]  # Log scale
        gamma = 10 ** params[1]  # Log scale
        
        model = SVC(C=C, gamma=gamma, kernel='rbf')
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return np.mean(scores)
    
    # Define bounds (in log scale)
    bounds = [(-3, 2), (-4, 0)]  # C: [0.001, 100], gamma: [0.0001, 1]
    
    # Run Bayesian optimization
    optimizer = BayesianOptimizer(objective, bounds, n_calls=30, n_initial=5)
    best_params, best_score = optimizer.optimize()
    
    print(f"\nBest parameters found:")
    print(f"C = {10**best_params[0]:.4f}")
    print(f"gamma = {10**best_params[1]:.4f}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Visualize optimization process
    optimizer.plot_optimization()
    
    # Compare with random search
    random_scores = []
    for _ in range(30):
        random_params = [np.random.uniform(b[0], b[1]) for b in bounds]
        score = objective(random_params)
        random_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), [max(optimizer.y_obs[:i]) for i in range(1, 31)], 
            'b-', label='Bayesian Optimization', linewidth=2)
    plt.plot(range(1, 31), [max(random_scores[:i]) for i in range(1, 31)], 
            'r--', label='Random Search', linewidth=2)
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Best Score Found')
    plt.title('Bayesian Optimization vs Random Search')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## 6. Advanced Techniques

### 6.1 Multi-fidelity Optimization

```python
class MultiFidelityOptimizer:
    """Multi-fidelity hyperparameter optimization"""
    
    def __init__(self, estimator, param_distributions, max_budget=100):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.max_budget = max_budget
        
    def successive_halving(self, X, y, n_configs=16, min_budget=1, eta=3):
        """Successive halving algorithm"""
        # Initialize
        configs = self._sample_configurations(n_configs)
        
        # Successive halving rounds
        budget = min_budget
        results = []
        
        while len(configs) > 1 and budget < self.max_budget:
            print(f"\nRound: {len(configs)} configs, budget={budget}")
            
            # Evaluate all configs with current budget
            scores = []
            for config in configs:
                score = self._evaluate_config(config, X, y, budget)
                scores.append(score)
                results.append({
                    'config': config,
                    'budget': budget,
                    'score': score
                })
            
            # Keep top 1/eta configs
            n_keep = max(1, int(len(configs) / eta))
            top_indices = np.argsort(scores)[-n_keep:]
            configs = [configs[i] for i in top_indices]
            
            # Increase budget
            budget = min(budget * eta, self.max_budget)
        
        # Final evaluation with full budget
        best_config = configs[0]
        final_score = self._evaluate_config(best_config, X, y, self.max_budget)
        
        return best_config, final_score, results
    
    def _sample_configurations(self, n_configs):
        """Sample random configurations"""
        sampler = ParameterSampler(self.param_distributions, 
                                 n_iter=n_configs, random_state=42)
        return list(sampler)
    
    def _evaluate_config(self, config, X, y, budget):
        """Evaluate configuration with given budget"""
        # Budget can mean different things:
        # - Subset of data
        # - Number of iterations
        # - Reduced model complexity
        
        # Example: Use subset of data
        n_samples = int(len(X) * min(budget / self.max_budget, 1.0))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        model = clone(self.estimator)
        model.set_params(**config)
        
        scores = cross_val_score(model, X_subset, y_subset, cv=3, scoring='accuracy')
        return np.mean(scores)
    
    def hyperband(self, X, y, max_iter=81, eta=3):
        """Hyperband algorithm"""
        logeta = lambda x: np.log(x) / np.log(eta)
        s_max = int(logeta(max_iter))
        B = (s_max + 1) * max_iter
        
        results = []
        best_config = None
        best_score = -np.inf
        
        for s in reversed(range(s_max + 1)):
            # Initial number of configurations
            n = int(np.ceil(B / max_iter / (s + 1) * eta ** s))
            # Initial budget per configuration
            r = max_iter * eta ** (-s)
            
            print(f"\nBracket s={s}: n={n}, r={r:.1f}")
            
            # Successive halving with (n, r)
            configs = self._sample_configurations(n)
            
            for i in range(s + 1):
                # Current budget
                r_i = r * eta ** i
                n_i = int(n * eta ** (-i))
                
                print(f"  Round {i}: {n_i} configs, budget={r_i:.1f}")
                
                # Evaluate configurations
                scores = []
                for config in configs[:n_i]:
                    score = self._evaluate_config(config, X, y, int(r_i))
                    scores.append(score)
                    results.append({
                        'config': config,
                        'budget': r_i,
                        'score': score,
                        'bracket': s
                    })
                
                # Select top configurations
                if i < s:
                    n_keep = int(n_i / eta)
                    top_indices = np.argsort(scores)[-n_keep:]
                    configs = [configs[j] for j in top_indices]
                else:
                    # Last round - update best
                    if scores and max(scores) > best_score:
                        best_idx = np.argmax(scores)
                        best_score = scores[best_idx]
                        best_config = configs[best_idx]
        
        return best_config, best_score, results

### 6.2 Population-based Training

class PopulationBasedTraining:
    """Population-based training for hyperparameter optimization"""
    
    def __init__(self, estimator, param_ranges, population_size=10, 
                 exploit_factor=0.2, explore_factor=1.2):
        self.estimator = estimator
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.exploit_factor = exploit_factor
        self.explore_factor = explore_factor
        
    def train(self, X, y, n_generations=20):
        """Run population-based training"""
        # Initialize population
        population = self._initialize_population()
        history = []
        
        for generation in range(n_generations):
            print(f"\nGeneration {generation + 1}/{n_generations}")
            
            # Evaluate population
            scores = []
            for member in population:
                score = self._evaluate_member(member, X, y)
                scores.append(score)
                member['score'] = score
            
            # Sort by performance
            population = sorted(population, key=lambda x: x['score'], reverse=True)
            
            # Log best
            best_member = population[0]
            print(f"Best score: {best_member['score']:.4f}")
            history.append({
                'generation': generation,
                'best_score': best_member['score'],
                'best_params': best_member['params'].copy()
            })
            
            # Exploit and explore
            if generation < n_generations - 1:
                population = self._exploit_explore(population)
        
        return population[0], history
    
    def _initialize_population(self):
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            params = {}
            for param, (low, high) in self.param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param] = np.random.randint(low, high + 1)
                else:
                    params[param] = np.random.uniform(low, high)
            
            population.append({
                'params': params,
                'score': None
            })
        
        return population
    
    def _evaluate_member(self, member, X, y):
        """Evaluate a population member"""
        model = clone(self.estimator)
        model.set_params(**member['params'])
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return np.mean(scores)
    
    def _exploit_explore(self, population):
        """Exploit good solutions and explore new ones"""
        n_exploit = int(len(population) * self.exploit_factor)
        
        # Bottom performers copy from top performers
        for i in range(len(population) - n_exploit, len(population)):
            # Copy from random top performer
            source_idx = np.random.randint(0, n_exploit)
            population[i]['params'] = population[source_idx]['params'].copy()
            
            # Explore by perturbing parameters
            for param, value in population[i]['params'].items():
                if np.random.rand() < 0.5:  # 50% chance to perturb
                    low, high = self.param_ranges[param]
                    
                    # Perturb by random factor
                    if isinstance(value, int):
                        delta = np.random.randint(-2, 3)
                        new_value = np.clip(value + delta, low, high)
                    else:
                        factor = np.random.uniform(1/self.explore_factor, 
                                                 self.explore_factor)
                        new_value = np.clip(value * factor, low, high)
                    
                    population[i]['params'][param] = new_value
        
        return population
```

## 7. Practical Implementation

### 7.1 Complete Hyperparameter Tuning Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class HyperparameterTuningPipeline:
    """Complete pipeline for hyperparameter tuning"""
    
    def __init__(self, task='classification'):
        self.task = task
        self.results = {}
        
    def create_preprocessing_pipeline(self, numeric_features, categorical_features):
        """Create preprocessing pipeline"""
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def tune_model(self, X, y, model, param_distributions, method='random', 
                   n_iter=50, cv=5):
        """Tune hyperparameters using specified method"""
        
        if method == 'grid':
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(model, param_distributions, cv=cv, 
                                scoring='accuracy' if self.task == 'classification' else 'neg_mean_squared_error',
                                n_jobs=-1, verbose=1)
        
        elif method == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter,
                                      cv=cv, scoring='accuracy' if self.task == 'classification' else 'neg_mean_squared_error',
                                      n_jobs=-1, verbose=1, random_state=42)
        
        elif method == 'bayesian':
            # Use skopt for Bayesian optimization
            from skopt import BayesSearchCV
            search = BayesSearchCV(model, param_distributions, n_iter=n_iter,
                                 cv=cv, scoring='accuracy' if self.task == 'classification' else 'neg_mean_squared_error',
                                 n_jobs=-1, verbose=1, random_state=42)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit
        search.fit(X, y)
        
        # Store results
        self.results[method] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'best_estimator': search.best_estimator_
        }
        
        return search
    
    def compare_tuning_methods(self, X, y, model, param_distributions):
        """Compare different tuning methods"""
        methods = ['grid', 'random', 'bayesian']
        comparison_results = []
        
        for method in methods:
            print(f"\nTuning with {method} search...")
            start_time = time.time()
            
            try:
                search = self.tune_model(X, y, model, param_distributions, 
                                       method=method, n_iter=30)
                elapsed_time = time.time() - start_time
                
                comparison_results.append({
                    'method': method,
                    'best_score': search.best_score_,
                    'time': elapsed_time,
                    'n_evaluations': len(search.cv_results_['mean_test_score'])
                })
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # Visualize comparison
        if comparison_results:
            self._plot_method_comparison(comparison_results)
        
        return comparison_results
    
    def _plot_method_comparison(self, results):
        """Plot comparison of tuning methods"""
        methods = [r['method'] for r in results]
        scores = [r['best_score'] for r in results]
        times = [r['time'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Best scores
        ax1.bar(methods, scores)
        ax1.set_ylabel('Best Score')
        ax1.set_title('Best Score by Method')
        ax1.set_ylim(min(scores) * 0.95, max(scores) * 1.05)
        
        # Time taken
        ax2.bar(methods, times)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Time Taken by Method')
        
        plt.suptitle('Hyperparameter Tuning Method Comparison')
        plt.tight_layout()
        plt.show()

# Example: Complete hyperparameter tuning workflow
def complete_tuning_example():
    """Complete example of hyperparameter tuning workflow"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    # Define model and parameters
    model = RandomForestClassifier(random_state=42)
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Random search with distributions
    param_distributions_random = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(10, 31)),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Initialize pipeline
    pipeline = HyperparameterTuningPipeline(task='classification')
    
    # Compare methods
    results = pipeline.compare_tuning_methods(X, y, model, param_distributions)
    
    # Get best model
    best_method = max(results, key=lambda x: x['best_score'])['method']
    best_model = pipeline.results[best_method]['best_estimator']
    
    print(f"\nBest method: {best_method}")
    print(f"Best parameters: {pipeline.results[best_method]['best_params']}")
    
    # Feature importance from best model
    feature_importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(feature_importances))],
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'][:10], 
            feature_importance_df['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances (Best Model)')
    plt.tight_layout()
    plt.show()
    
    return pipeline
```

## 8. Best Practices

### 8.1 Hyperparameter Tuning Best Practices

```python
def hyperparameter_tuning_best_practices():
    """Comprehensive guide to best practices"""
    
    practices = {
        'Before Tuning': [
            'Understand each hyperparameter\'s effect',
            'Start with recommended/default values',
            'Use domain knowledge to set reasonable ranges',
            'Consider computational budget',
            'Ensure proper train/validation/test splits'
        ],
        
        'During Tuning': [
            'Use appropriate search strategy for problem size',
            'Monitor for overfitting to validation set',
            'Use early stopping when applicable',
            'Log all experiments for reproducibility',
            'Consider multi-objective optimization'
        ],
        
        'Search Strategies': [
            'Few hyperparameters (<5): Grid search',
            'Many hyperparameters: Random search or Bayesian',
            'Limited budget: Multi-fidelity methods',
            'Continuous parameters: Bayesian optimization',
            'Mixed types: Consider specialized libraries'
        ],
        
        'Validation': [
            'Use nested CV for unbiased estimates',
            'Stratify folds for imbalanced data',
            'Time-based splits for temporal data',
            'Group-based splits for grouped data',
            'Multiple metrics for comprehensive evaluation'
        ],
        
        'Common Pitfalls': [
            'Tuning on test set',
            'Not using same preprocessing in CV',
            'Ignoring parameter interactions',
            'Over-tuning leading to overfitting',
            'Not considering inference time'
        ]
    }
    
    # Create visual guide
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(practices)))
    
    for idx, (category, items) in enumerate(practices.items()):
        # Category header
        ax.text(0, y_pos, category, fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[idx]))
        y_pos -= 1.5
        
        # Items
        for item in items:
            ax.text(0.5, y_pos, f"• {item}", fontsize=11)
            y_pos -= 1
        
        y_pos -= 0.5
    
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(y_pos, 2)
    ax.axis('off')
    ax.set_title('Hyperparameter Tuning Best Practices', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return practices

### 8.2 Hyperparameter Range Guidelines

def get_hyperparameter_ranges():
    """Common hyperparameter ranges for different algorithms"""
    
    ranges = {
        'RandomForest': {
            'n_estimators': [100, 500],
            'max_depth': [None, 3, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        
        'GradientBoosting': {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'min_samples_split': [2, 5, 10]
        },
        
        'SVM': {
            'C': np.logspace(-3, 3, 7),
            'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 6)),
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        
        'NeuralNetwork': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128],
            'max_iter': [200, 500]
        },
        
        'XGBoost': {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    }
    
    return ranges
```

## 9. Case Studies

### 9.1 Case Study: Image Classification

```python
def image_classification_tuning():
    """Hyperparameter tuning for image classification"""
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Create pipeline with PCA and MLP
    pipeline = Pipeline([
        ('pca', PCA()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=42))
    ])
    
    # Define parameter distributions
    param_distributions = {
        'pca__n_components': randint(10, 50),
        'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'mlp__learning_rate_init': loguniform(0.0001, 0.1),
        'mlp__alpha': loguniform(0.0001, 0.1),
        'mlp__activation': ['relu', 'tanh']
    }
    
    # Bayesian optimization
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    
    search_spaces = {
        'pca__n_components': Integer(10, 50),
        'mlp__hidden_layer_sizes': Categorical([(50,), (100,), (100, 50), (100, 100)]),
        'mlp__learning_rate_init': Real(0.0001, 0.1, prior='log-uniform'),
        'mlp__alpha': Real(0.0001, 0.1, prior='log-uniform'),
        'mlp__activation': Categorical(['relu', 'tanh'])
    }
    
    bayes_search = BayesSearchCV(
        pipeline,
        search_spaces,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    # Fit
    bayes_search.fit(X, y)
    
    # Results
    print(f"Best score: {bayes_search.best_score_:.4f}")
    print(f"Best parameters: {bayes_search.best_params_}")
    
    # Convergence plot
    from skopt.plots import plot_convergence
    plot_convergence(bayes_search.optimizer_results_[0])
    plt.title('Bayesian Optimization Convergence')
    plt.show()
    
    return bayes_search

### 9.2 Case Study: Time Series Forecasting

def time_series_hyperparameter_tuning():
    """Hyperparameter tuning for time series"""
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Generate synthetic time series data
    n_samples = 1000
    time = np.arange(n_samples)
    
    # Create features from time
    X = np.column_stack([
        np.sin(2 * np.pi * time / 365),  # Yearly seasonality
        np.sin(2 * np.pi * time / 30),   # Monthly seasonality
        time / n_samples,                  # Trend
        np.random.randn(n_samples) * 0.1  # Noise
    ])
    
    # Target with complex pattern
    y = (10 + 0.01 * time + 
         5 * np.sin(2 * np.pi * time / 365) + 
         2 * np.sin(2 * np.pi * time / 30) + 
         np.random.randn(n_samples))
    
    # Time series cross-validation
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Model
    model = GradientBoostingRegressor(random_state=42)
    
    # Hyperparameters
    param_distributions = {
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'min_samples_split': randint(2, 20)
    }
    
    # Custom scorer for time series
    def custom_time_series_score(y_true, y_pred):
        """Custom scoring that penalizes recent errors more"""
        weights = np.linspace(0.5, 1.0, len(y_true))
        weighted_errors = weights * np.abs(y_true - y_pred)
        return -np.mean(weighted_errors)
    
    from sklearn.metrics import make_scorer
    custom_scorer = make_scorer(custom_time_series_score, greater_is_better=True)
    
    # Random search with time series CV
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=50,
        cv=tscv,
        scoring=custom_scorer,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit
    search.fit(X, y)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_:.4f}")
    
    # Visualize predictions
    best_model = search.best_estimator_
    y_pred = best_model.predict(X)
    
    plt.figure(figsize=(15, 6))
    plt.plot(time[-200:], y[-200:], label='Actual', alpha=0.7)
    plt.plot(time[-200:], y_pred[-200:], label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Prediction with Tuned Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return search
```

## 10. Interview Questions

### Q1: What are hyperparameters and how do they differ from parameters?
**Answer**: 
- **Parameters**: Learned during training (e.g., weights in neural networks, coefficients in linear regression)
- **Hyperparameters**: Set before training, control the learning process (e.g., learning rate, number of trees)

Key differences:
- Parameters are optimized by the learning algorithm
- Hyperparameters are optimized by the practitioner
- Parameters define the model, hyperparameters define how to learn the model

### Q2: Compare grid search and random search for hyperparameter tuning.
**Answer**:

**Grid Search**:
- Exhaustive search over all combinations
- Pros: Guaranteed to find best in grid, systematic
- Cons: Computationally expensive (exponential in # parameters), may miss optimal values between grid points

**Random Search**:
- Sample random combinations
- Pros: More efficient for many parameters, better for continuous parameters, can find good solutions faster
- Cons: No guarantee of finding optimal, may need many iterations

Rule of thumb: Use grid search for <4 hyperparameters, random search otherwise.

### Q3: Explain Bayesian optimization for hyperparameter tuning.
**Answer**: Bayesian optimization builds a probabilistic model (usually Gaussian Process) of the objective function:

1. **Prior**: Start with prior belief about function
2. **Surrogate model**: GP provides mean and uncertainty estimates
3. **Acquisition function**: Decides where to sample next (balances exploration vs exploitation)
4. **Update**: Incorporate new observation, update posterior

Advantages:
- Efficient for expensive evaluations
- Naturally handles uncertainty
- Can incorporate prior knowledge

Common acquisition functions: Expected Improvement (EI), Upper Confidence Bound (UCB), Probability of Improvement (PI).

### Q4: What is the curse of dimensionality in hyperparameter tuning?
**Answer**: As the number of hyperparameters increases:

1. **Search space grows exponentially**: n parameters with k values each = k^n combinations
2. **Sparse sampling**: Random/grid search covers tiny fraction of space
3. **Local optima**: More likely to get stuck
4. **Interactions**: Complex parameter interactions harder to capture

Solutions:
- Use random search or Bayesian optimization
- Focus on most important hyperparameters first
- Use domain knowledge to constrain ranges
- Multi-fidelity methods (Hyperband, BOHB)

### Q5: How do you avoid overfitting during hyperparameter tuning?
**Answer**:

1. **Nested cross-validation**: 
   - Outer loop for model evaluation
   - Inner loop for hyperparameter selection

2. **Separate validation set**: Don't use test set for any decisions

3. **Regularization in search space**: Include regularization hyperparameters

4. **Conservative choices**: Prefer simpler models when performance is similar

5. **Multiple metrics**: Consider generalization metrics, not just accuracy

6. **Early stopping**: For iterative algorithms

### Q6: What are multi-fidelity optimization methods?
**Answer**: Methods that use cheaper approximations to eliminate poor configurations early:

**Successive Halving**:
- Start with many configs, small budget
- Iteratively increase budget, keep top fraction

**Hyperband**:
- Multiple brackets of successive halving
- Different trade-offs between n_configs and budget

**Benefits**:
- More efficient than full evaluation
- Can try more configurations
- Adaptive resource allocation

**Examples of fidelities**:
- Subset of data
- Fewer iterations/epochs
- Smaller model size

### Q7: How do you handle different types of hyperparameters (continuous, discrete, categorical)?
**Answer**:

**Continuous** (e.g., learning rate):
- Grid: Define discrete values
- Random: Sample from distribution (uniform, log-uniform)
- Bayesian: Use appropriate kernel

**Discrete** (e.g., n_estimators):
- Grid: List specific values
- Random: Use randint
- Bayesian: Integer-valued space

**Categorical** (e.g., kernel type):
- Grid: List all options
- Random: Uniform choice
- Bayesian: Categorical distribution

**Mixed types**: Use appropriate search methods (random search, Bayesian optimization handle naturally).

### Q8: What is learning rate scheduling and how do you tune it?
**Answer**: Learning rate scheduling adjusts learning rate during training:

**Types**:
1. **Step decay**: Reduce by factor every n epochs
2. **Exponential decay**: lr = lr_0 * decay^epoch
3. **Cosine annealing**: Cosine function decrease
4. **Adaptive**: Based on validation performance

**Tuning approaches**:
- Tune initial rate and decay parameters
- Use callbacks to monitor and adjust
- Cyclical learning rates
- Learning rate range test

**Best practices**:
- Start with higher rate, decay gradually
- Monitor training/validation curves
- Consider warm-up period

### Q9: How do you tune hyperparameters for ensemble methods?
**Answer**:

**Key hyperparameters**:
1. **Base estimator parameters**: Tune individually first
2. **Number of estimators**: More is usually better (with diminishing returns)
3. **Subsampling**: Data/feature sampling rates
4. **Combination method**: Voting weights, stacking meta-learner

**Strategies**:
1. **Two-stage**: Tune base estimators, then ensemble parameters
2. **Joint optimization**: Tune all together (expensive)
3. **Fixed base**: Use default base estimators, tune ensemble only

**Special considerations**:
- Diversity vs accuracy trade-off
- Computational cost grows with ensemble size
- May need different parameters for different base estimators

### Q10: What are the computational considerations in hyperparameter tuning?
**Answer**:

**Time complexity**:
- Grid search: O(n^p × model_cost × cv_folds)
- Random search: O(n_iter × model_cost × cv_folds)
- Bayesian: O(n_iter × model_cost × cv_folds + GP_cost)

**Strategies for efficiency**:
1. **Parallelization**: Use n_jobs=-1
2. **Early stopping**: Stop poor configurations early
3. **Warm starts**: Reuse previous computations
4. **Caching**: Store intermediate results
5. **Progressive sampling**: Start with data subset

**Resource management**:
- Set time/iteration budgets
- Use cloud/cluster resources
- Consider multi-fidelity methods

### Q11: How do you tune hyperparameters for deep learning models?
**Answer**:

**Key hyperparameters**:
1. **Architecture**: Layers, units, activation functions
2. **Optimization**: Learning rate, batch size, optimizer
3. **Regularization**: Dropout, weight decay, batch norm

**Challenges**:
- Very high-dimensional search space
- Long training times
- Non-convex optimization

**Approaches**:
1. **Neural Architecture Search (NAS)**
2. **Population-based training**
3. **Hyperband/ASHA for early stopping**
4. **Bayesian optimization with early stopping**

**Best practices**:
- Use learning rate range test
- Start with proven architectures
- Monitor validation curves
- Use callbacks for adaptive tuning

### Q12: What is the difference between hyperparameter optimization and neural architecture search?
**Answer**:

**Hyperparameter Optimization**:
- Tunes fixed architecture parameters
- Examples: learning rate, regularization
- Search space is predefined

**Neural Architecture Search (NAS)**:
- Searches for optimal architecture
- Examples: number of layers, connections
- Search space includes topology

**Key differences**:
- NAS has much larger search space
- NAS may discover novel architectures
- NAS typically more computationally expensive
- Different optimization methods (RL, evolutionary, gradient-based)

### Q13: How do you incorporate domain knowledge into hyperparameter tuning?
**Answer**:

1. **Constrain search space**: Use reasonable ranges based on experience
2. **Prior distributions**: In Bayesian optimization
3. **Relative importance**: Focus on known important parameters
4. **Conditional parameters**: Some parameters only matter for certain values
5. **Meta-learning**: Learn from previous similar tasks

**Examples**:
- Learning rate typically in [1e-4, 1e-1]
- Tree depth rarely needs to exceed 20
- Regularization stronger for smaller datasets

### Q14: What are some common pitfalls in hyperparameter tuning?
**Answer**:

1. **Data leakage**: Using test set information
2. **Multiple testing**: Not accounting for trying many configurations
3. **Overfitting to validation set**: Too much tuning
4. **Ignoring variance**: Only looking at mean performance
5. **Scale issues**: Not standardizing continuous parameters
6. **Random seeds**: Not fixing seeds for reproducibility
7. **Interaction effects**: Assuming parameters are independent
8. **Computational waste**: Not using early stopping
9. **Local optima**: Getting stuck in suboptimal regions
10. **Production mismatch**: Tuning for metrics not aligned with business goals

### Q15: How do you tune hyperparameters for online learning algorithms?
**Answer**:

**Challenges**:
- No fixed dataset
- Concept drift
- Real-time constraints

**Approaches**:
1. **Progressive validation**: Evaluate on future data
2. **Bandits for hyperparameters**: Thompson sampling, UCB
3. **Meta-learning**: Learn parameter schedules
4. **Adaptive parameters**: Change based on performance

**Key hyperparameters**:
- Learning rate (often needs decay)
- Buffer/window size
- Update frequency
- Forgetting factor

### Q16: Explain the explore-exploit trade-off in hyperparameter optimization.
**Answer**:

**Exploration**: Try diverse parameters to find new regions
**Exploitation**: Focus on promising regions

**In different methods**:
- **Grid search**: Pure exploration (systematic)
- **Random search**: Pure exploration (random)
- **Bayesian optimization**: Balances via acquisition function
- **Bandit methods**: Explicit explore-exploit algorithms

**Acquisition functions balance this**:
- **Expected Improvement**: Natural balance
- **UCB**: Explicit exploration bonus
- **Probability of Improvement**: More exploitative
- **Entropy-based**: More exploratory

### Q17: How do you handle hyperparameter tuning with limited computational budget?
**Answer**:

1. **Multi-fidelity methods**: Hyperband, BOHB
2. **Transfer learning**: Start from previous good configurations
3. **Surrogate models**: Cheaper approximations
4. **Early stopping**: Halt poor performers
5. **Subset of data**: Initial screening on smaller data
6. **Feature sampling**: Reduce dimensionality
7. **Progressive refinement**: Coarse then fine search
8. **Parallel coordinates**: Visualize to guide search

**Prioritization**:
- Tune most impactful parameters first
- Use domain knowledge for good starting points
- Consider parameter sensitivity analysis

### Q18: What is population-based training (PBT)?
**Answer**: PBT combines parallel search with online adaptation:

**Process**:
1. Initialize population of models
2. Train in parallel for some steps
3. Evaluate performance
4. Replace poor performers with mutations of good ones
5. Continue training

**Advantages**:
- Adapts hyperparameters during training
- Efficient use of resources
- Can discover schedules
- Robust to initialization

**Use cases**:
- Deep learning
- Reinforcement learning
- Long training times

### Q19: How do you validate hyperparameter choices?
**Answer**:

1. **Hold-out test set**: Final evaluation only
2. **Nested cross-validation**: Unbiased estimate
3. **Time-based validation**: For temporal data
4. **Multiple random seeds**: Check stability
5. **Different data splits**: Ensure robustness
6. **Sensitivity analysis**: Small perturbations
7. **Learning curves**: Check for overfitting
8. **Production A/B test**: Real-world validation

**Red flags**:
- Large train-test gap
- Unstable across seeds
- Very sensitive to small changes
- Poor performance on slightly different data

### Q20: How has automated machine learning (AutoML) changed hyperparameter tuning?
**Answer**:

**Traditional approach**:
- Manual selection
- Expert knowledge required
- Time-consuming
- Suboptimal results

**AutoML approach**:
- Automated search
- Includes feature engineering
- End-to-end optimization
- Democratizes ML

**Key innovations**:
1. **Meta-learning**: Learn from previous tasks
2. **Neural architecture search**
3. **Automated feature engineering**
4. **Pipeline optimization**
5. **Multi-objective optimization**

**Popular frameworks**: Auto-sklearn, H2O, Google AutoML, TPOT

**Limitations**:
- Black box nature
- Computational cost
- May miss domain-specific insights
- Not always better than expert tuning