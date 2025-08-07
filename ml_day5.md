# Day 5: Mean, Median, Mode, Variance, Covariance

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Python Implementations](#python-implementations)
4. [ML Applications](#ml-applications)
5. [Interview Questions](#interview-questions)
6. [Practice Exercises](#practice-exercises)

## Core Concepts

### Measures of Central Tendency

**Central tendency** describes the center or typical value of a dataset. The three main measures are:

1. **Mean (μ)**: Average value
   - Arithmetic mean: Sum of all values divided by count
   - Sensitive to outliers
   - Best for symmetric distributions

2. **Median**: Middle value when sorted
   - Robust to outliers
   - Better for skewed distributions
   - 50th percentile

3. **Mode**: Most frequent value
   - Can have multiple modes (multimodal)
   - Useful for categorical data
   - Only measure for nominal data

### Measures of Dispersion

**Dispersion** quantifies the spread or variability in data:

1. **Variance (σ²)**: Average squared deviation from mean
   - Population variance: σ² = Σ(x - μ)² / N
   - Sample variance: s² = Σ(x - x̄)² / (n-1)
   - Units are squared

2. **Standard Deviation (σ)**: Square root of variance
   - Same units as original data
   - 68-95-99.7 rule for normal distribution

3. **Covariance**: Measure of joint variability
   - Cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]
   - Positive: variables move together
   - Negative: variables move oppositely
   - Zero: no linear relationship

## Mathematical Foundations

### Population vs Sample Statistics

#### Population Parameters
- Mean: μ = (1/N) Σᵢ xᵢ
- Variance: σ² = (1/N) Σᵢ (xᵢ - μ)²
- Covariance: σ_XY = (1/N) Σᵢ (xᵢ - μ_X)(yᵢ - μ_Y)

#### Sample Statistics
- Mean: x̄ = (1/n) Σᵢ xᵢ
- Variance: s² = (1/(n-1)) Σᵢ (xᵢ - x̄)²
- Covariance: s_XY = (1/(n-1)) Σᵢ (xᵢ - x̄)(yᵢ - ȳ)

### Why (n-1) in Sample Variance?

**Bessel's Correction**: Using (n-1) makes sample variance an unbiased estimator of population variance.

Mathematical proof:
```
E[s²] = E[(1/(n-1)) Σ(xᵢ - x̄)²]
     = σ² (when using n-1)
```

### Properties of Variance and Covariance

1. **Variance Properties**:
   - Var(aX + b) = a²Var(X)
   - Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
   - Var(X) ≥ 0

2. **Covariance Properties**:
   - Cov(X,X) = Var(X)
   - Cov(X,Y) = Cov(Y,X)
   - Cov(aX + b, cY + d) = ac·Cov(X,Y)

### Covariance Matrix

For multiple variables X₁, X₂, ..., Xₙ:
```
Σ = [Cov(X₁,X₁)  Cov(X₁,X₂)  ...  Cov(X₁,Xₙ)]
    [Cov(X₂,X₁)  Cov(X₂,X₂)  ...  Cov(X₂,Xₙ)]
    [    ...         ...      ...     ...    ]
    [Cov(Xₙ,X₁)  Cov(Xₙ,X₂)  ...  Cov(Xₙ,Xₙ)]
```

Properties:
- Symmetric: Σ = Σᵀ
- Positive semi-definite: xᵀΣx ≥ 0 for all x
- Diagonal elements are variances

## Python Implementations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict, Union, Optional
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class CentralTendencyMeasures:
    """Advanced implementations of central tendency measures"""
    
    @staticmethod
    def mean(data: np.ndarray, weights: Optional[np.ndarray] = None,
             axis: Optional[int] = None, trimmed: float = 0.0) -> float:
        """
        Calculate various types of means
        
        Args:
            data: Input array
            weights: Optional weights for weighted mean
            axis: Axis along which to compute
            trimmed: Fraction to trim from each end (trimmed mean)
        
        Returns:
            Mean value(s)
        """
        if trimmed > 0:
            # Trimmed mean (robust to outliers)
            return stats.trim_mean(data, trimmed, axis=axis)
        elif weights is not None:
            # Weighted mean
            return np.average(data, weights=weights, axis=axis)
        else:
            # Arithmetic mean
            return np.mean(data, axis=axis)
    
    @staticmethod
    def geometric_mean(data: np.ndarray, axis: Optional[int] = None) -> float:
        """Calculate geometric mean (for positive values)"""
        if np.any(data <= 0):
            raise ValueError("Geometric mean requires positive values")
        return np.exp(np.mean(np.log(data), axis=axis))
    
    @staticmethod
    def harmonic_mean(data: np.ndarray, axis: Optional[int] = None) -> float:
        """Calculate harmonic mean (for positive values)"""
        if np.any(data <= 0):
            raise ValueError("Harmonic mean requires positive values")
        return len(data) / np.sum(1.0 / data, axis=axis)
    
    @staticmethod
    def median(data: np.ndarray, axis: Optional[int] = None) -> float:
        """Calculate median with interpolation options"""
        return np.median(data, axis=axis)
    
    @staticmethod
    def mode(data: np.ndarray, return_counts: bool = False) -> Union[float, Tuple[float, int]]:
        """
        Calculate mode(s) of the data
        
        Args:
            data: Input array
            return_counts: Whether to return counts
        
        Returns:
            Mode value(s) and optionally counts
        """
        values, counts = np.unique(data, return_counts=True)
        max_count = np.max(counts)
        modes = values[counts == max_count]
        
        if return_counts:
            return modes, max_count
        return modes
    
    @staticmethod
    def robust_location_estimators(data: np.ndarray) -> Dict[str, float]:
        """
        Calculate various robust estimators of location
        
        Returns:
            Dictionary with different estimators
        """
        return {
            'median': np.median(data),
            'trimmed_mean_10': stats.trim_mean(data, 0.1),
            'trimmed_mean_20': stats.trim_mean(data, 0.2),
            'winsorized_mean_10': stats.mstats.winsorize(data, limits=[0.1, 0.1]).mean(),
            'hodges_lehmann': CentralTendencyMeasures._hodges_lehmann(data),
            'tukey_biweight': CentralTendencyMeasures._tukey_biweight(data)
        }
    
    @staticmethod
    def _hodges_lehmann(data: np.ndarray) -> float:
        """Hodges-Lehmann estimator (median of pairwise averages)"""
        n = len(data)
        pairwise_avg = []
        for i in range(n):
            for j in range(i, n):
                pairwise_avg.append((data[i] + data[j]) / 2)
        return np.median(pairwise_avg)
    
    @staticmethod
    def _tukey_biweight(data: np.ndarray, c: float = 4.685) -> float:
        """Tukey's biweight M-estimator"""
        # Iterative algorithm
        median = np.median(data)
        mad = stats.median_abs_deviation(data)
        
        for _ in range(10):  # Usually converges quickly
            u = (data - median) / (c * mad)
            weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
            median = np.sum(weights * data) / np.sum(weights)
        
        return median

class DispersionMeasures:
    """Advanced implementations of dispersion measures"""
    
    @staticmethod
    def variance(data: np.ndarray, ddof: int = 0, axis: Optional[int] = None,
                robust: bool = False) -> float:
        """
        Calculate variance with options
        
        Args:
            data: Input array
            ddof: Delta degrees of freedom (0 for population, 1 for sample)
            axis: Axis along which to compute
            robust: Use robust estimator (MAD-based)
        
        Returns:
            Variance
        """
        if robust:
            # Robust variance estimator using MAD
            mad = stats.median_abs_deviation(data, axis=axis)
            return (1.4826 * mad) ** 2  # Scale factor for normal distribution
        else:
            return np.var(data, ddof=ddof, axis=axis)
    
    @staticmethod
    def covariance(x: np.ndarray, y: np.ndarray, ddof: int = 0,
                  robust: bool = False) -> float:
        """
        Calculate covariance between two variables
        
        Args:
            x, y: Input arrays
            ddof: Delta degrees of freedom
            robust: Use robust estimator
        
        Returns:
            Covariance
        """
        if robust:
            # Robust covariance using Spearman correlation
            spearman_corr = stats.spearmanr(x, y)[0]
            return spearman_corr * DispersionMeasures.variance(x, robust=True)**0.5 * \
                   DispersionMeasures.variance(y, robust=True)**0.5
        else:
            return np.cov(x, y, ddof=ddof)[0, 1]
    
    @staticmethod
    def covariance_matrix(data: np.ndarray, ddof: int = 0,
                         robust: bool = False) -> np.ndarray:
        """
        Calculate covariance matrix
        
        Args:
            data: Input array (n_samples, n_features)
            ddof: Delta degrees of freedom
            robust: Use robust estimator
        
        Returns:
            Covariance matrix
        """
        if robust:
            # Robust covariance estimation
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(data)
            return robust_cov.covariance_
        else:
            return np.cov(data.T, ddof=ddof)
    
    @staticmethod
    def other_dispersion_measures(data: np.ndarray) -> Dict[str, float]:
        """Calculate various dispersion measures"""
        return {
            'variance': np.var(data, ddof=1),
            'std_dev': np.std(data, ddof=1),
            'mad': stats.median_abs_deviation(data),
            'iqr': stats.iqr(data),
            'range': np.ptp(data),
            'cv': np.std(data, ddof=1) / np.mean(data),  # Coefficient of variation
            'qcd': stats.iqr(data) / (np.percentile(data, 75) + np.percentile(data, 25))  # Quartile coefficient
        }

class StatisticalAnalyzer:
    """Comprehensive statistical analysis tools"""
    
    def __init__(self, data: Union[np.ndarray, pd.DataFrame]):
        """Initialize with data"""
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.columns = data.columns
        else:
            self.data = data
            self.columns = [f'Feature_{i}' for i in range(data.shape[1])]
        
        self.n_samples, self.n_features = self.data.shape
    
    def summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics"""
        stats_dict = {
            'mean': np.mean(self.data, axis=0),
            'median': np.median(self.data, axis=0),
            'mode': [stats.mode(self.data[:, i])[0][0] for i in range(self.n_features)],
            'std': np.std(self.data, axis=0, ddof=1),
            'variance': np.var(self.data, axis=0, ddof=1),
            'min': np.min(self.data, axis=0),
            'max': np.max(self.data, axis=0),
            'q1': np.percentile(self.data, 25, axis=0),
            'q3': np.percentile(self.data, 75, axis=0),
            'iqr': stats.iqr(self.data, axis=0),
            'skewness': stats.skew(self.data, axis=0),
            'kurtosis': stats.kurtosis(self.data, axis=0),
            'cv': np.std(self.data, axis=0, ddof=1) / np.mean(self.data, axis=0)
        }
        
        return pd.DataFrame(stats_dict, index=self.columns)
    
    def analyze_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Detect outliers using various methods
        
        Args:
            method: 'iqr', 'zscore', 'mad', or 'isolation_forest'
            threshold: Threshold for outlier detection
        
        Returns:
            Dictionary with outlier information
        """
        outliers = {}
        
        for i, col in enumerate(self.columns):
            feature_data = self.data[:, i]
            
            if method == 'iqr':
                Q1 = np.percentile(feature_data, 25)
                Q3 = np.percentile(feature_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(feature_data))
                outlier_mask = z_scores > threshold
                
            elif method == 'mad':
                median = np.median(feature_data)
                mad = stats.median_abs_deviation(feature_data)
                modified_z_scores = 0.6745 * (feature_data - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1)
                outlier_mask = iso_forest.fit_predict(feature_data.reshape(-1, 1)) == -1
            
            outliers[col] = {
                'n_outliers': np.sum(outlier_mask),
                'outlier_indices': np.where(outlier_mask)[0],
                'outlier_values': feature_data[outlier_mask]
            }
        
        return outliers
    
    def visualize_distributions(self, figsize: Tuple[int, int] = (15, 10)):
        """Visualize distributions of all features"""
        n_cols = min(3, self.n_features)
        n_rows = (self.n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if self.n_features > 1 else [axes]
        
        for i, col in enumerate(self.columns):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(self.data[:, i], bins=30, density=True, 
                           alpha=0.7, edgecolor='black')
                
                # KDE overlay
                kde_x = np.linspace(self.data[:, i].min(), 
                                  self.data[:, i].max(), 100)
                kde = stats.gaussian_kde(self.data[:, i])
                axes[i].plot(kde_x, kde(kde_x), 'r-', linewidth=2)
                
                # Add statistics
                mean_val = np.mean(self.data[:, i])
                median_val = np.median(self.data[:, i])
                
                axes[i].axvline(mean_val, color='green', linestyle='--', 
                              linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='blue', linestyle='--', 
                              linewidth=2, label=f'Median: {median_val:.2f}')
                
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(self.n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_covariance_matrix(self, annot: bool = True):
        """Visualize covariance matrix"""
        cov_matrix = np.cov(self.data.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cov_matrix, annot=annot, fmt='.2f', 
                   xticklabels=self.columns, yticklabels=self.columns,
                   cmap='coolwarm', center=0)
        plt.title('Covariance Matrix')
        plt.tight_layout()
        plt.show()
        
        return cov_matrix

class RobustStatistics:
    """Robust statistical methods for real-world data"""
    
    @staticmethod
    def breakdown_point_analysis(data: np.ndarray, estimators: Dict[str, Callable]) -> pd.DataFrame:
        """
        Analyze breakdown points of different estimators
        
        Args:
            data: Original data
            estimators: Dictionary of estimator names and functions
        
        Returns:
            DataFrame with breakdown analysis
        """
        results = []
        contamination_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for cont_level in contamination_levels:
            n_contaminate = int(len(data) * cont_level)
            contaminated_data = data.copy()
            
            # Add extreme outliers
            if n_contaminate > 0:
                contaminated_data[:n_contaminate] = np.max(data) * 10
            
            row = {'contamination': cont_level}
            for name, estimator in estimators.items():
                try:
                    row[name] = estimator(contaminated_data)
                except:
                    row[name] = np.nan
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def influence_function(data: np.ndarray, estimator: Callable, 
                          x_new: float) -> float:
        """
        Empirical influence function
        
        Args:
            data: Original data
            estimator: Statistical estimator
            x_new: New observation
        
        Returns:
            Influence value
        """
        n = len(data)
        original_estimate = estimator(data)
        
        # Add new observation
        data_with_new = np.append(data, x_new)
        new_estimate = estimator(data_with_new)
        
        # Empirical influence
        influence = (n + 1) * (new_estimate - original_estimate)
        
        return influence

class CovarianceApplications:
    """Applications of covariance in ML"""
    
    @staticmethod
    def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, 
                           cov_inv: np.ndarray) -> np.ndarray:
        """
        Calculate Mahalanobis distance
        
        Args:
            x: Points (n_samples, n_features)
            mean: Mean vector
            cov_inv: Inverse covariance matrix
        
        Returns:
            Mahalanobis distances
        """
        diff = x - mean
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    
    @staticmethod
    def whitening_transform(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply whitening transformation (decorrelation + unit variance)
        
        Returns:
            Whitened data, whitening matrix, mean
        """
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        # Covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Whitening matrix
        D = np.diag(1.0 / np.sqrt(eigenvalues + 1e-8))
        W = eigenvectors @ D @ eigenvectors.T
        
        # Apply transformation
        whitened = centered @ W
        
        return whitened, W, mean
    
    @staticmethod
    def portfolio_optimization(returns: np.ndarray, target_return: float) -> np.ndarray:
        """
        Markowitz portfolio optimization using covariance
        
        Args:
            returns: Asset returns (n_periods, n_assets)
            target_return: Target portfolio return
        
        Returns:
            Optimal weights
        """
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        n_assets = len(mean_returns)
        
        from scipy.optimize import minimize
        
        # Objective: minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
            {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}  # Target return
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(portfolio_variance, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x

# Example usage and demonstrations
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create correlated data
    mean = [10, 20]
    cov = [[4, 3], [3, 9]]  # Positive correlation
    data = np.random.multivariate_normal(mean, cov, 1000)
    
    # Add some outliers
    n_outliers = 50
    outliers = np.random.uniform(-50, 50, (n_outliers, 2))
    data_with_outliers = np.vstack([data, outliers])
    
    # 1. Central Tendency Analysis
    print("=== Central Tendency Analysis ===")
    ct = CentralTendencyMeasures()
    
    # Compare different means
    print(f"Arithmetic Mean: {ct.mean(data[:, 0]):.2f}")
    print(f"Trimmed Mean (10%): {ct.mean(data[:, 0], trimmed=0.1):.2f}")
    print(f"Median: {ct.median(data[:, 0]):.2f}")
    
    # With outliers
    print(f"\nWith outliers:")
    print(f"Arithmetic Mean: {ct.mean(data_with_outliers[:, 0]):.2f}")
    print(f"Trimmed Mean (10%): {ct.mean(data_with_outliers[:, 0], trimmed=0.1):.2f}")
    print(f"Median: {ct.median(data_with_outliers[:, 0]):.2f}")
    
    # Robust estimators
    robust_estimates = ct.robust_location_estimators(data_with_outliers[:, 0])
    print(f"\nRobust Estimators:")
    for name, value in robust_estimates.items():
        print(f"{name}: {value:.2f}")
    
    # 2. Dispersion Analysis
    print("\n=== Dispersion Analysis ===")
    disp = DispersionMeasures()
    
    print(f"Variance: {disp.variance(data[:, 0], ddof=1):.2f}")
    print(f"Robust Variance (MAD-based): {disp.variance(data[:, 0], robust=True):.2f}")
    print(f"Covariance: {disp.covariance(data[:, 0], data[:, 1], ddof=1):.2f}")
    
    # Other measures
    other_measures = disp.other_dispersion_measures(data[:, 0])
    print("\nOther Dispersion Measures:")
    for name, value in other_measures.items():
        print(f"{name}: {value:.4f}")
    
    # 3. Statistical Analysis
    print("\n=== Comprehensive Statistical Analysis ===")
    analyzer = StatisticalAnalyzer(data)
    
    # Summary statistics
    summary = analyzer.summary_statistics()
    print("\nSummary Statistics:")
    print(summary)
    
    # Visualizations
    analyzer.visualize_distributions()
    cov_matrix = analyzer.plot_covariance_matrix()
    
    # Outlier analysis
    outliers = analyzer.analyze_outliers(method='iqr')
    for feature, info in outliers.items():
        print(f"\n{feature}: {info['n_outliers']} outliers detected")
    
    # 4. Robust Statistics Demo
    print("\n=== Robust Statistics Demo ===")
    
    # Breakdown point analysis
    estimators = {
        'mean': np.mean,
        'median': np.median,
        'trimmed_mean_20': lambda x: stats.trim_mean(x, 0.2),
        'hodges_lehmann': CentralTendencyMeasures._hodges_lehmann
    }
    
    breakdown_df = RobustStatistics.breakdown_point_analysis(
        data[:, 0], estimators
    )
    print("\nBreakdown Point Analysis:")
    print(breakdown_df)
    
    # Influence function
    influence_mean = RobustStatistics.influence_function(
        data[:, 0], np.mean, 100
    )
    influence_median = RobustStatistics.influence_function(
        data[:, 0], np.median, 100
    )
    print(f"\nInfluence of outlier (100) on mean: {influence_mean:.2f}")
    print(f"Influence of outlier (100) on median: {influence_median:.2f}")
    
    # 5. Covariance Applications
    print("\n=== Covariance Applications ===")
    cov_app = CovarianceApplications()
    
    # Mahalanobis distance
    mean_vec = np.mean(data, axis=0)
    cov_matrix = np.cov(data.T)
    cov_inv = np.linalg.inv(cov_matrix)
    
    # Test points
    test_points = np.array([[10, 20], [15, 25], [0, 0], [30, 40]])
    mahal_distances = cov_app.mahalanobis_distance(test_points, mean_vec, cov_inv)
    
    print("\nMahalanobis Distances:")
    for i, (point, dist) in enumerate(zip(test_points, mahal_distances)):
        print(f"Point {point}: {dist:.2f}")
    
    # Whitening transformation
    whitened_data, W, mean = cov_app.whitening_transform(data)
    print(f"\nOriginal data covariance (off-diagonal): {np.cov(data.T)[0,1]:.2f}")
    print(f"Whitened data covariance (off-diagonal): {np.cov(whitened_data.T)[0,1]:.6f}")
    
    # Visualization of whitening
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.5)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.5)
    ax2.set_title('Whitened Data')
    ax2.set_xlabel('Feature 1 (whitened)')
    ax2.set_ylabel('Feature 2 (whitened)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## ML Applications

### 1. Feature Scaling and Normalization

```python
class FeatureEngineering:
    """Feature engineering using statistical measures"""
    
    @staticmethod
    def robust_scaler(X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Robust scaling using median and MAD
        
        Returns:
            Scaled data and parameters
        """
        median = np.median(X, axis=0)
        mad = stats.median_abs_deviation(X, axis=0)
        
        # Scale
        X_scaled = (X - median) / (mad + 1e-8)
        
        params = {'median': median, 'mad': mad}
        return X_scaled, params
    
    @staticmethod
    def variance_threshold_selection(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Feature selection based on variance
        
        Returns:
            Selected feature indices
        """
        variances = np.var(X, axis=0)
        return np.where(variances > threshold)[0]
    
    @staticmethod
    def decorrelate_features(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Remove highly correlated features
        
        Returns:
            Selected feature indices
        """
        corr_matrix = np.corrcoef(X.T)
        upper_tri = np.triu(np.abs(corr_matrix), k=1)
        
        # Find features to drop
        to_drop = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if upper_tri[i, j] > threshold:
                    # Drop feature with lower variance
                    if np.var(X[:, i]) < np.var(X[:, j]):
                        to_drop.add(i)
                    else:
                        to_drop.add(j)
        
        keep_features = [i for i in range(X.shape[1]) if i not in to_drop]
        return np.array(keep_features)
```

### 2. Anomaly Detection

```python
class AnomalyDetection:
    """Anomaly detection using statistical measures"""
    
    def __init__(self, method: str = 'mahalanobis'):
        self.method = method
        self.params = {}
    
    def fit(self, X: np.ndarray):
        """Fit the anomaly detector"""
        if self.method == 'mahalanobis':
            self.params['mean'] = np.mean(X, axis=0)
            self.params['cov'] = np.cov(X.T)
            self.params['cov_inv'] = np.linalg.inv(self.params['cov'] + 1e-6 * np.eye(X.shape[1]))
        
        elif self.method == 'robust_covariance':
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(X)
            self.params['mean'] = robust_cov.location_
            self.params['cov_inv'] = robust_cov.get_precision()
        
        elif self.method == 'zscore':
            self.params['mean'] = np.mean(X, axis=0)
            self.params['std'] = np.std(X, axis=0)
    
    def predict(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Predict anomalies"""
        if self.method in ['mahalanobis', 'robust_covariance']:
            distances = CovarianceApplications.mahalanobis_distance(
                X, self.params['mean'], self.params['cov_inv']
            )
            # Chi-squared threshold
            from scipy.stats import chi2
            threshold = chi2.ppf(0.95, df=X.shape[1])
            return distances > threshold
        
        elif self.method == 'zscore':
            z_scores = np.abs((X - self.params['mean']) / self.params['std'])
            return np.any(z_scores > threshold, axis=1)
```

### 3. Bayesian Statistics Applications

```python
class BayesianAnalysis:
    """Bayesian analysis using mean and variance"""
    
    @staticmethod
    def bayesian_update_normal(prior_mean: float, prior_var: float,
                              data: np.ndarray, known_var: float) -> Tuple[float, float]:
        """
        Bayesian update for normal distribution with known variance
        
        Returns:
            Posterior mean and variance
        """
        n = len(data)
        sample_mean = np.mean(data)
        
        # Posterior parameters
        posterior_var = 1 / (1/prior_var + n/known_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/known_var)
        
        return posterior_mean, posterior_var
    
    @staticmethod
    def conjugate_normal_inverse_gamma(data: np.ndarray, 
                                     prior_params: Dict[str, float]) -> Dict[str, float]:
        """
        Conjugate prior update for normal distribution with unknown mean and variance
        
        Prior: Normal-Inverse-Gamma(μ₀, λ, α, β)
        """
        n = len(data)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=0)
        
        # Extract prior parameters
        mu_0 = prior_params['mu_0']
        lambda_0 = prior_params['lambda']
        alpha_0 = prior_params['alpha']
        beta_0 = prior_params['beta']
        
        # Update parameters
        lambda_n = lambda_0 + n
        mu_n = (lambda_0 * mu_0 + n * sample_mean) / lambda_n
        alpha_n = alpha_0 + n/2
        beta_n = beta_0 + 0.5 * n * sample_var + \
                 0.5 * lambda_0 * n * (sample_mean - mu_0)**2 / lambda_n
        
        return {
            'mu_n': mu_n,
            'lambda_n': lambda_n,
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'posterior_mean': mu_n,
            'posterior_var': beta_n / (alpha_n * lambda_n)
        }
```

## Interview Questions

### Basic Level

1. **What is the difference between population and sample variance?**
   - Population: divide by N
   - Sample: divide by (n-1) for unbiased estimation
   - Bessel's correction accounts for using sample mean

2. **Why do we use (n-1) in sample variance calculation?**
   - Makes it an unbiased estimator
   - Accounts for loss of degree of freedom from estimating mean
   - E[s²] = σ² when using (n-1)

3. **What is covariance and what does it tell us?**
   - Measures linear relationship between variables
   - Positive: variables tend to increase together
   - Negative: one increases as other decreases
   - Zero: no linear relationship (but may have nonlinear)

### Intermediate Level

4. **How do mean, median, and mode behave with skewed distributions?**
   - Right-skewed: mode < median < mean
   - Left-skewed: mean < median < mode
   - Symmetric: mean ≈ median ≈ mode

5. **What is the covariance matrix and what are its properties?**
   - Contains all pairwise covariances
   - Symmetric: Σ = Σᵀ
   - Positive semi-definite
   - Diagonal elements are variances

6. **How does variance relate to machine learning model performance?**
   - High variance in predictions → overfitting
   - Bias-variance tradeoff
   - Feature variance affects scaling needs
   - Used in regularization (e.g., ridge regression)

### Advanced Level

7. **Explain the relationship between covariance and correlation.**
   ```
   ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
   ```
   - Correlation is normalized covariance
   - Bounded between -1 and 1
   - Scale-invariant

8. **How is the covariance matrix used in PCA?**
   - PCA finds eigenvectors of covariance matrix
   - Eigenvectors = principal components
   - Eigenvalues = variance explained
   - Transforms to uncorrelated features

9. **What are robust alternatives to mean and variance?**
   - Mean → Median, trimmed mean, Winsorized mean
   - Variance → MAD, IQR, robust covariance estimators
   - Important for outlier-contaminated data

## Practice Exercises

### Exercise 1: Implement Incremental Statistics
```python
class IncrementalStats:
    """Calculate statistics incrementally for streaming data"""
    
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0  # Sum of squared deviations
        
    def update(self, x: float):
        """Update statistics with new value"""
        # TODO: Implement Welford's algorithm
        pass
    
    @property
    def variance(self) -> float:
        """Return current variance"""
        # TODO: Calculate variance from M2
        pass
    
    @property
    def std(self) -> float:
        """Return current standard deviation"""
        # TODO: Calculate from variance
        pass
```

### Exercise 2: Multivariate Outlier Detection
```python
def detect_multivariate_outliers(X: np.ndarray, method: str = 'mahalanobis',
                                contamination: float = 0.1) -> np.ndarray:
    """
    Detect outliers in multivariate data
    
    Args:
        X: Data matrix (n_samples, n_features)
        method: Detection method
        contamination: Expected proportion of outliers
    
    Returns:
        Boolean mask of outliers
    """
    # TODO: Implement multiple detection methods
    # - Mahalanobis distance
    # - Minimum Covariance Determinant
    # - Local Outlier Factor
    pass
```

### Exercise 3: Weighted Statistics
```python
class WeightedStatistics:
    """Calculate weighted versions of statistical measures"""
    
    @staticmethod
    def weighted_mean(x: np.ndarray, weights: np.ndarray) -> float:
        # TODO: Implement weighted mean
        pass
    
    @staticmethod
    def weighted_variance(x: np.ndarray, weights: np.ndarray,
                         mean: Optional[float] = None) -> float:
        # TODO: Implement weighted variance
        pass
    
    @staticmethod
    def weighted_covariance(x: np.ndarray, y: np.ndarray,
                           weights: np.ndarray) -> float:
        # TODO: Implement weighted covariance
        pass
```

### Exercise 4: Shrinkage Covariance Estimation
```python
def shrinkage_covariance(X: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """
    Implement Ledoit-Wolf shrinkage for covariance estimation
    
    Shrinks sample covariance toward diagonal matrix
    
    Args:
        X: Data matrix
        shrinkage: Shrinkage parameter (0-1)
    
    Returns:
        Shrunk covariance matrix
    """
    # TODO: Implement shrinkage estimation
    # S_shrink = (1-δ)S + δF
    # where F is target (e.g., diagonal)
    pass
```

### Exercise 5: Bootstrap Confidence Intervals
```python
def bootstrap_ci(data: np.ndarray, statistic: Callable,
                n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for any statistic
    
    Args:
        data: Input data
        statistic: Function to calculate statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Lower and upper confidence bounds
    """
    # TODO: Implement bootstrap CI
    # 1. Resample with replacement
    # 2. Calculate statistic for each sample
    # 3. Find percentiles
    pass
```

## Advanced Topics

### 1. Robust Covariance Estimation
- **Minimum Covariance Determinant (MCD)**: Finds subset with minimum determinant covariance
- **M-estimators**: Iteratively reweighted estimation
- **Tyler's M-estimator**: Distribution-free robust covariance

### 2. High-Dimensional Covariance
- **Sparse covariance**: When p >> n
- **Graphical LASSO**: Sparse precision matrix estimation
- **Factor models**: Low-rank + sparse decomposition

### 3. Online/Streaming Statistics
- **Welford's algorithm**: Incremental mean and variance
- **Exponential moving statistics**: For non-stationary data
- **Sliding window statistics**: Fixed window size

## Summary

Key takeaways:
1. **Central tendency measures** have different robustness properties
2. **Variance and covariance** are fundamental to many ML algorithms
3. **Sample vs population** distinctions matter for unbiased estimation
4. **Robust alternatives** are crucial for real-world data
5. **Covariance matrix** properties enable techniques like PCA, Mahalanobis distance
6. Understanding these measures helps in:
   - Feature engineering and selection
   - Anomaly detection
   - Model diagnostics
   - Statistical inference