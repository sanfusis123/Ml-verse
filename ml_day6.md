# Day 6: Correlation & Multicollinearity

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Python Implementations](#python-implementations)
4. [ML Applications](#ml-applications)
5. [Interview Questions](#interview-questions)
6. [Practice Exercises](#practice-exercises)

## Core Concepts

### Correlation

**Correlation** measures the strength and direction of the linear relationship between two variables. Unlike covariance, correlation is standardized and scale-invariant.

### Types of Correlation

1. **Pearson Correlation (r)**
   - Measures linear relationship
   - Assumes normality
   - Sensitive to outliers
   - Range: [-1, 1]

2. **Spearman Correlation (ρ)**
   - Rank-based correlation
   - Non-parametric
   - Robust to outliers
   - Captures monotonic relationships

3. **Kendall's Tau (τ)**
   - Based on concordant/discordant pairs
   - More robust than Spearman
   - Better for small samples
   - Computationally intensive

### Multicollinearity

**Multicollinearity** occurs when independent variables in a regression model are highly correlated. This creates problems:

1. **Unstable coefficient estimates**
2. **Inflated standard errors**
3. **Difficult interpretation**
4. **Reduced predictive power**

### Detection Methods

1. **Correlation Matrix**: |r| > 0.8 indicates potential issue
2. **Variance Inflation Factor (VIF)**: VIF > 10 indicates severe multicollinearity
3. **Condition Number**: κ > 30 indicates multicollinearity
4. **Eigenvalue Analysis**: Near-zero eigenvalues indicate multicollinearity

## Mathematical Foundations

### Pearson Correlation Coefficient

For variables X and Y:
```
r = Cov(X,Y) / (σ_X × σ_Y)
  = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)² × Σ(y_i - ȳ)²]
```

Properties:
- Scale invariant: r(aX + b, cY + d) = sign(ac) × r(X,Y)
- Bounded: -1 ≤ r ≤ 1
- r = ±1 implies perfect linear relationship
- r = 0 implies no linear relationship

### Spearman Rank Correlation

Convert data to ranks, then apply Pearson correlation:
```
ρ = 1 - (6Σd_i²) / [n(n² - 1)]
```
where d_i = rank(x_i) - rank(y_i)

### Kendall's Tau

Based on concordant and discordant pairs:
```
τ = (n_c - n_d) / [n(n-1)/2]
```
where:
- n_c = number of concordant pairs
- n_d = number of discordant pairs

### Variance Inflation Factor (VIF)

For variable X_j in regression:
```
VIF_j = 1 / (1 - R_j²)
```
where R_j² is R-squared from regressing X_j on all other predictors

### Partial Correlation

Correlation between X and Y after removing effect of Z:
```
r_{XY.Z} = (r_{XY} - r_{XZ}r_{YZ}) / √[(1 - r_{XZ}²)(1 - r_{YZ}²)]
```

### Multiple Correlation

Correlation between Y and best linear combination of predictors:
```
R = √(R²)
```
where R² is coefficient of determination

## Python Implementations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalysis:
    """Comprehensive correlation analysis tools"""
    
    @staticmethod
    def calculate_correlations(X: np.ndarray, Y: np.ndarray,
                             methods: List[str] = ['pearson', 'spearman', 'kendall']) -> Dict:
        """
        Calculate multiple correlation coefficients
        
        Args:
            X, Y: Input arrays
            methods: List of correlation methods
        
        Returns:
            Dictionary with correlation coefficients and p-values
        """
        results = {}
        
        if 'pearson' in methods:
            r, p = pearsonr(X, Y)
            results['pearson'] = {'coefficient': r, 'p_value': p}
        
        if 'spearman' in methods:
            rho, p = spearmanr(X, Y)
            results['spearman'] = {'coefficient': rho, 'p_value': p}
        
        if 'kendall' in methods:
            tau, p = kendalltau(X, Y)
            results['kendall'] = {'coefficient': tau, 'p_value': p}
        
        # Additional measures
        results['distance_correlation'] = CorrelationAnalysis._distance_correlation(X, Y)
        results['maximal_information'] = CorrelationAnalysis._maximal_information_coefficient(X, Y)
        
        return results
    
    @staticmethod
    def _distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate distance correlation (captures nonlinear relationships)
        """
        n = len(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        
        # Distance matrices
        a = np.sqrt(np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))
        b = np.sqrt(np.sum((Y[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2))
        
        # Double-center
        A = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
        B = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
        
        # Distance covariance
        dcov2 = (A * B).sum() / n**2
        dvar_x = (A * A).sum() / n**2
        dvar_y = (B * B).sum() / n**2
        
        # Distance correlation
        dcor = np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
        return dcor
    
    @staticmethod
    def _maximal_information_coefficient(X: np.ndarray, Y: np.ndarray,
                                       n_bins: int = 10) -> float:
        """
        Simplified MIC calculation (approximation)
        """
        # Discretize data
        X_discrete = np.digitize(X, np.percentile(X, np.linspace(0, 100, n_bins)))
        Y_discrete = np.digitize(Y, np.percentile(Y, np.linspace(0, 100, n_bins)))
        
        # Calculate mutual information
        from sklearn.metrics import mutual_info_score
        mi = mutual_info_score(X_discrete, Y_discrete)
        
        # Normalize (approximation)
        return mi / np.log(n_bins)
    
    @staticmethod
    def partial_correlation(data: pd.DataFrame, x: str, y: str,
                          controlling: List[str]) -> float:
        """
        Calculate partial correlation controlling for other variables
        
        Args:
            data: DataFrame with variables
            x, y: Variables of interest
            controlling: Variables to control for
        
        Returns:
            Partial correlation coefficient
        """
        # Residualize x and y
        X_control = data[controlling].values
        
        # Fit linear models
        model_x = LinearRegression().fit(X_control, data[x])
        model_y = LinearRegression().fit(X_control, data[y])
        
        # Get residuals
        residuals_x = data[x] - model_x.predict(X_control)
        residuals_y = data[y] - model_y.predict(X_control)
        
        # Correlation of residuals
        return pearsonr(residuals_x, residuals_y)[0]
    
    @staticmethod
    def correlation_matrix_advanced(data: pd.DataFrame,
                                  method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix with significance testing
        
        Returns:
            Correlation matrix with significance stars
        """
        n_vars = len(data.columns)
        corr_matrix = np.zeros((n_vars, n_vars))
        p_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    if method == 'pearson':
                        corr, p = pearsonr(data.iloc[:, i], data.iloc[:, j])
                    elif method == 'spearman':
                        corr, p = spearmanr(data.iloc[:, i], data.iloc[:, j])
                    elif method == 'kendall':
                        corr, p = kendalltau(data.iloc[:, i], data.iloc[:, j])
                    
                    corr_matrix[i, j] = corr
                    p_matrix[i, j] = p
        
        # Create DataFrame with significance stars
        corr_df = pd.DataFrame(corr_matrix, 
                              index=data.columns, 
                              columns=data.columns)
        
        # Add significance stars
        def add_stars(val, p):
            if p < 0.001:
                return f"{val:.3f}***"
            elif p < 0.01:
                return f"{val:.3f}**"
            elif p < 0.05:
                return f"{val:.3f}*"
            else:
                return f"{val:.3f}"
        
        # Create annotated matrix
        annot_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        for i in range(n_vars):
            for j in range(n_vars):
                annot_matrix.iloc[i, j] = add_stars(corr_matrix[i, j], p_matrix[i, j])
        
        return corr_df, annot_matrix

class MulticollinearityAnalysis:
    """Tools for detecting and handling multicollinearity"""
    
    @staticmethod
    def calculate_vif(data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factors
        
        Args:
            data: DataFrame with features
            features: List of features to analyze (default: all)
        
        Returns:
            DataFrame with VIF values
        """
        if features is None:
            features = data.columns.tolist()
        
        X = data[features].values
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(len(features))]
        
        # Add interpretation
        vif_data["Severity"] = pd.cut(vif_data["VIF"], 
                                     bins=[0, 5, 10, np.inf],
                                     labels=["Low", "Moderate", "High"])
        
        return vif_data.sort_values('VIF', ascending=False)
    
    @staticmethod
    def condition_number(X: np.ndarray) -> Dict[str, float]:
        """
        Calculate condition number and related metrics
        
        Returns:
            Dictionary with condition metrics
        """
        # Standardize data
        X_std = StandardScaler().fit_transform(X)
        
        # Correlation matrix
        corr_matrix = np.corrcoef(X_std.T)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvals(corr_matrix)
        
        # Condition number
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        
        # Condition index
        condition_indices = np.sqrt(np.max(eigenvalues) / eigenvalues)
        
        return {
            'condition_number': condition_number,
            'eigenvalues': eigenvalues,
            'condition_indices': condition_indices,
            'min_eigenvalue': np.min(eigenvalues),
            'interpretation': 'Severe' if condition_number > 30 else 'Moderate' if condition_number > 10 else 'Low'
        }
    
    @staticmethod
    def detect_multicollinearity(X: pd.DataFrame, threshold: float = 0.9) -> Dict:
        """
        Comprehensive multicollinearity detection
        
        Returns:
            Dictionary with various diagnostics
        """
        results = {}
        
        # 1. Correlation matrix
        corr_matrix = X.corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        results['high_correlations'] = high_corr_pairs
        
        # 2. VIF
        results['vif'] = MulticollinearityAnalysis.calculate_vif(X)
        
        # 3. Condition number
        results['condition_analysis'] = MulticollinearityAnalysis.condition_number(X.values)
        
        # 4. Eigenvalue analysis
        eigenvalues = results['condition_analysis']['eigenvalues']
        results['near_zero_eigenvalues'] = np.sum(eigenvalues < 0.01)
        
        return results
    
    @staticmethod
    def remediate_multicollinearity(X: pd.DataFrame, y: np.ndarray,
                                  method: str = 'vif', threshold: float = 10) -> pd.DataFrame:
        """
        Remove multicollinear features
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: 'vif', 'correlation', or 'pca'
            threshold: Threshold for removal
        
        Returns:
            DataFrame with reduced features
        """
        if method == 'vif':
            # Iteratively remove highest VIF features
            features = X.columns.tolist()
            X_reduced = X.copy()
            
            while True:
                vif_df = MulticollinearityAnalysis.calculate_vif(X_reduced)
                max_vif = vif_df['VIF'].max()
                
                if max_vif < threshold:
                    break
                
                # Remove feature with highest VIF
                remove_feature = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
                X_reduced = X_reduced.drop(columns=[remove_feature])
                print(f"Removed {remove_feature} (VIF: {max_vif:.2f})")
            
            return X_reduced
        
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > threshold)]
            
            return X.drop(columns=to_drop)
        
        elif method == 'pca':
            # Use PCA to create orthogonal features
            pca = PCA(n_components=0.95)  # Keep 95% variance
            X_pca = pca.fit_transform(X)
            
            # Create DataFrame with PCA components
            columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            return pd.DataFrame(X_pca, columns=columns, index=X.index)

class CorrelationVisualization:
    """Visualization tools for correlation analysis"""
    
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame, method: str = 'pearson',
                              figsize: Tuple[int, int] = (10, 8),
                              mask_insignificant: bool = True,
                              alpha: float = 0.05):
        """
        Plot correlation matrix with significance masking
        """
        # Calculate correlations and p-values
        n_vars = len(data.columns)
        corr_matrix = np.zeros((n_vars, n_vars))
        p_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if method == 'pearson':
                    corr, p = pearsonr(data.iloc[:, i], data.iloc[:, j])
                elif method == 'spearman':
                    corr, p = spearmanr(data.iloc[:, i], data.iloc[:, j])
                
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p
        
        # Create mask for insignificant correlations
        if mask_insignificant:
            mask = p_matrix > alpha
        else:
            mask = None
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   xticklabels=data.columns,
                   yticklabels=data.columns,
                   cmap='coolwarm',
                   center=0,
                   vmin=-1, vmax=1)
        
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_scatter_matrix(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)):
        """
        Create scatter plot matrix with correlations
        """
        from pandas.plotting import scatter_matrix
        
        # Create scatter matrix
        axes = scatter_matrix(data, figsize=figsize, diagonal='kde', alpha=0.6)
        
        # Add correlation coefficients
        n_vars = len(data.columns)
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    ax = axes[i, j]
                    corr = pearsonr(data.iloc[:, j], data.iloc[:, i])[0]
                    ax.annotate(f'r={corr:.2f}', (0.05, 0.95), 
                              xycoords='axes fraction',
                              fontsize=10, ha='left', va='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Scatter Plot Matrix with Correlations')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_network(data: pd.DataFrame, threshold: float = 0.5,
                               figsize: Tuple[int, int] = (12, 10)):
        """
        Network visualization of correlations
        """
        import networkx as nx
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for col in data.columns:
            G.add_node(col)
        
        # Add edges for significant correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    G.add_edge(corr_matrix.columns[i], 
                             corr_matrix.columns[j],
                             weight=abs(corr),
                             correlation=corr)
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Plot
        plt.figure(figsize=figsize)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        
        # Draw edges with varying width
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        correlations = [G[u][v]['correlation'] for u, v in edges]
        
        # Color edges by positive/negative correlation
        edge_colors = ['red' if c < 0 else 'green' for c in correlations]
        
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                             edge_color=edge_colors, alpha=0.6)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Add correlation values on edges
        edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" 
                      for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        plt.title(f'Correlation Network (|r| > {threshold})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class RegressionWithMulticollinearity:
    """Demonstrate effects of multicollinearity on regression"""
    
    @staticmethod
    def compare_regression_methods(X: pd.DataFrame, y: np.ndarray,
                                 test_size: float = 0.3) -> pd.DataFrame:
        """
        Compare different regression methods under multicollinearity
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models to compare
        models = {
            'OLS': LinearRegression(),
            'Ridge (α=0.1)': Ridge(alpha=0.1),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Ridge (α=10.0)': Ridge(alpha=10.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'Lasso (α=1.0)': Lasso(alpha=1.0),
            'ElasticNet (α=1.0)': ElasticNet(alpha=1.0, l1_ratio=0.5)
        }
        
        results = []
        
        for name, model in models.items():
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Coefficient statistics
            coef_mean = np.mean(np.abs(model.coef_))
            coef_std = np.std(model.coef_)
            n_zero_coef = np.sum(np.abs(model.coef_) < 1e-5)
            
            results.append({
                'Model': name,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Mean |Coef|': coef_mean,
                'Std Coef': coef_std,
                'Zero Coefs': n_zero_coef
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def visualize_coefficient_paths(X: pd.DataFrame, y: np.ndarray,
                                  method: str = 'ridge',
                                  alphas: np.ndarray = None):
        """
        Visualize how coefficients change with regularization
        """
        if alphas is None:
            alphas = np.logspace(-3, 2, 100)
        
        # Standardize
        X_scaled = StandardScaler().fit_transform(X)
        
        # Store coefficients
        coefs = []
        
        for alpha in alphas:
            if method == 'ridge':
                model = Ridge(alpha=alpha)
            elif method == 'lasso':
                model = Lasso(alpha=alpha, max_iter=1000)
            
            model.fit(X_scaled, y)
            coefs.append(model.coef_)
        
        coefs = np.array(coefs)
        
        # Plot
        plt.figure(figsize=(10, 6))
        for i in range(coefs.shape[1]):
            plt.plot(alphas, coefs[:, i], label=X.columns[i])
        
        plt.xscale('log')
        plt.xlabel('Regularization Parameter (α)')
        plt.ylabel('Coefficient Value')
        plt.title(f'{method.capitalize()} Coefficient Paths')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example usage and demonstrations
if __name__ == "__main__":
    # Generate example data with multicollinearity
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated features
    # Base features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    
    # Correlated features
    X3 = 0.8 * X1 + 0.2 * np.random.normal(0, 1, n_samples)  # Highly correlated with X1
    X4 = 0.7 * X2 + 0.3 * np.random.normal(0, 1, n_samples)  # Highly correlated with X2
    X5 = 0.5 * X1 + 0.5 * X2 + np.random.normal(0, 0.5, n_samples)  # Linear combination
    
    # Independent feature
    X6 = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'X6': X6
    })
    
    # Create target variable
    y = 2*X1 + 3*X2 + X6 + np.random.normal(0, 0.5, n_samples)
    
    # 1. Correlation Analysis
    print("=== Correlation Analysis ===")
    corr_analyzer = CorrelationAnalysis()
    
    # Calculate different correlations for X1 and X3
    correlations = corr_analyzer.calculate_correlations(X1, X3)
    print("\nCorrelation between X1 and X3:")
    for method, result in correlations.items():
        if isinstance(result, dict):
            print(f"{method}: {result['coefficient']:.4f} (p={result['p_value']:.4f})")
        else:
            print(f"{method}: {result:.4f}")
    
    # Correlation matrix with significance
    corr_matrix, annot_matrix = corr_analyzer.correlation_matrix_advanced(data)
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # 2. Multicollinearity Detection
    print("\n=== Multicollinearity Detection ===")
    mc_analyzer = MulticollinearityAnalysis()
    
    # VIF calculation
    vif_df = mc_analyzer.calculate_vif(data)
    print("\nVariance Inflation Factors:")
    print(vif_df)
    
    # Comprehensive detection
    mc_results = mc_analyzer.detect_multicollinearity(data, threshold=0.7)
    print("\nHigh Correlations:")
    for pair in mc_results['high_correlations']:
        print(f"{pair['var1']} - {pair['var2']}: {pair['correlation']:.3f}")
    
    print(f"\nCondition Number: {mc_results['condition_analysis']['condition_number']:.2f}")
    print(f"Interpretation: {mc_results['condition_analysis']['interpretation']}")
    print(f"Near-zero eigenvalues: {mc_results['near_zero_eigenvalues']}")
    
    # 3. Visualizations
    print("\n=== Generating Visualizations ===")
    viz = CorrelationVisualization()
    
    # Correlation heatmap
    viz.plot_correlation_matrix(data, method='pearson')
    
    # Scatter matrix
    viz.plot_scatter_matrix(data.iloc[:, :4])  # First 4 features
    
    # Correlation network
    viz.plot_correlation_network(data, threshold=0.5)
    
    # 4. Regression Comparison
    print("\n=== Regression with Multicollinearity ===")
    reg_analyzer = RegressionWithMulticollinearity()
    
    # Compare methods
    comparison_df = reg_analyzer.compare_regression_methods(data, y)
    print("\nRegression Methods Comparison:")
    print(comparison_df)
    
    # Coefficient paths
    reg_analyzer.visualize_coefficient_paths(data, y, method='ridge')
    reg_analyzer.visualize_coefficient_paths(data, y, method='lasso')
    
    # 5. Remediation
    print("\n=== Multicollinearity Remediation ===")
    
    # Remove features with high VIF
    data_reduced_vif = mc_analyzer.remediate_multicollinearity(
        data, y, method='vif', threshold=10
    )
    print(f"\nFeatures after VIF reduction: {list(data_reduced_vif.columns)}")
    
    # PCA transformation
    data_pca = mc_analyzer.remediate_multicollinearity(
        data, y, method='pca'
    )
    print(f"\nPCA components: {list(data_pca.columns)}")
    
    # 6. Advanced Correlation Analysis
    print("\n=== Advanced Correlation Analysis ===")
    
    # Create non-linear relationship
    X_nonlinear = np.random.uniform(-3, 3, 500)
    Y_nonlinear = X_nonlinear**2 + np.random.normal(0, 1, 500)
    
    # Compare correlation measures
    nonlinear_corrs = corr_analyzer.calculate_correlations(X_nonlinear, Y_nonlinear)
    print("\nNon-linear relationship (Y = X²):")
    for method, result in nonlinear_corrs.items():
        if isinstance(result, dict):
            print(f"{method}: {result['coefficient']:.4f}")
        else:
            print(f"{method}: {result:.4f}")
    
    # Partial correlation example
    print("\n=== Partial Correlation ===")
    # X5 is influenced by both X1 and X2
    partial_corr = corr_analyzer.partial_correlation(
        data, 'X5', 'X1', controlling=['X2']
    )
    print(f"Correlation X5-X1: {data['X5'].corr(data['X1']):.3f}")
    print(f"Partial correlation X5-X1 (controlling for X2): {partial_corr:.3f}")
```

## ML Applications

### 1. Feature Selection Using Correlation

```python
class CorrelationBasedFeatureSelection:
    """Feature selection methods based on correlation"""
    
    @staticmethod
    def select_features_target_correlation(X: pd.DataFrame, y: np.ndarray,
                                         threshold: float = 0.3,
                                         method: str = 'pearson') -> List[str]:
        """
        Select features based on correlation with target
        """
        correlations = {}
        
        for col in X.columns:
            if method == 'pearson':
                corr = pearsonr(X[col], y)[0]
            elif method == 'spearman':
                corr = spearmanr(X[col], y)[0]
            
            correlations[col] = abs(corr)
        
        # Select features above threshold
        selected = [col for col, corr in correlations.items() 
                   if corr > threshold]
        
        return sorted(selected, key=lambda x: correlations[x], reverse=True)
    
    @staticmethod
    def remove_redundant_features(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove redundant features based on inter-feature correlation
        """
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to keep
        to_drop = set()
        
        for col in upper_tri.columns:
            # Find highly correlated features
            correlated_features = list(upper_tri.index[upper_tri[col] > threshold])
            
            if correlated_features and col not in to_drop:
                # Keep the feature with highest average correlation to all others
                avg_corr = {col: corr_matrix[col].mean()}
                for feat in correlated_features:
                    if feat not in to_drop:
                        avg_corr[feat] = corr_matrix[feat].mean()
                
                # Keep feature with highest average correlation
                keep_feature = max(avg_corr, key=avg_corr.get)
                
                # Drop others
                for feat in avg_corr:
                    if feat != keep_feature:
                        to_drop.add(feat)
        
        return [col for col in X.columns if col not in to_drop]
```

### 2. Handling Multicollinearity in Different ML Algorithms

```python
class MulticollinearityInML:
    """How different ML algorithms handle multicollinearity"""
    
    @staticmethod
    def algorithm_sensitivity_analysis(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Test how different algorithms handle multicollinearity
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X)
        
        # Algorithms to test
        algorithms = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(hidden_layers=(50, 50), random_state=42, max_iter=1000)
        }
        
        results = []
        
        for name, algorithm in algorithms.items():
            # Cross-validation scores
            scores = cross_val_score(algorithm, X_scaled, y, cv=5, 
                                   scoring='neg_mean_squared_error')
            
            results.append({
                'Algorithm': name,
                'Mean CV Score': -scores.mean(),
                'Std CV Score': scores.std(),
                'Handles Multicollinearity': name in ['Ridge Regression', 'Lasso Regression', 
                                                     'Random Forest', 'Gradient Boosting']
            })
        
        return pd.DataFrame(results).sort_values('Mean CV Score')
```

### 3. Correlation in Time Series

```python
class TimeSeriesCorrelation:
    """Correlation analysis for time series data"""
    
    @staticmethod
    def autocorrelation_analysis(series: np.ndarray, max_lag: int = 40) -> Dict:
        """
        Compute autocorrelation and partial autocorrelation
        """
        from statsmodels.tsa.stattools import acf, pacf
        
        # Autocorrelation
        acf_values = acf(series, nlags=max_lag)
        
        # Partial autocorrelation
        pacf_values = pacf(series, nlags=max_lag)
        
        # Find significant lags (outside 95% confidence interval)
        n = len(series)
        confidence_interval = 1.96 / np.sqrt(n)
        
        significant_acf_lags = np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
        significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > confidence_interval)[0] + 1
        
        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'significant_acf_lags': significant_acf_lags,
            'significant_pacf_lags': significant_pacf_lags,
            'confidence_interval': confidence_interval
        }
    
    @staticmethod
    def cross_correlation(series1: np.ndarray, series2: np.ndarray, 
                         max_lag: int = 20) -> Dict:
        """
        Compute cross-correlation between two time series
        """
        # Normalize series
        series1 = (series1 - np.mean(series1)) / np.std(series1)
        series2 = (series2 - np.mean(series2)) / np.std(series2)
        
        # Cross-correlation
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag < 0:
                corr = pearsonr(series1[:lag], series2[-lag:])[0]
            elif lag > 0:
                corr = pearsonr(series1[lag:], series2[:-lag])[0]
            else:
                corr = pearsonr(series1, series2)[0]
            
            correlations.append(corr)
        
        # Find optimal lag
        max_corr_idx = np.argmax(np.abs(correlations))
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlations[max_corr_idx]
        
        return {
            'lags': list(lags),
            'correlations': correlations,
            'optimal_lag': optimal_lag,
            'max_correlation': max_correlation
        }
```

## Interview Questions

### Basic Level

1. **What is the difference between correlation and covariance?**
   - Covariance: measures joint variability, scale-dependent
   - Correlation: normalized covariance, scale-invariant, bounded [-1, 1]
   - Correlation = Covariance / (σ_X × σ_Y)

2. **What does a correlation of 0 mean?**
   - No linear relationship
   - Variables may still have non-linear relationship
   - Independent variables have zero correlation (but not vice versa)

3. **What is multicollinearity and why is it problematic?**
   - High correlation among independent variables
   - Problems: unstable coefficients, inflated standard errors
   - Makes interpretation difficult

### Intermediate Level

4. **Compare Pearson, Spearman, and Kendall correlations.**
   - Pearson: linear relationships, assumes normality
   - Spearman: monotonic relationships, rank-based, robust
   - Kendall: concordant pairs, more robust, better for small samples

5. **How do you detect multicollinearity?**
   - Correlation matrix (|r| > 0.8)
   - VIF (> 10 indicates problem)
   - Condition number (> 30 indicates problem)
   - Eigenvalue analysis

6. **What methods can handle multicollinearity in regression?**
   - Ridge regression (L2 regularization)
   - Lasso regression (L1 regularization)
   - Principal Component Regression
   - Partial Least Squares

### Advanced Level

7. **Explain the relationship between VIF and R-squared.**
   ```
   VIF_j = 1 / (1 - R_j²)
   ```
   - R_j² from regressing X_j on other predictors
   - Higher R² means higher multicollinearity
   - VIF quantifies inflation in variance

8. **How does multicollinearity affect different ML algorithms?**
   - Linear models: severely affected
   - Tree-based models: relatively robust
   - Neural networks: can handle but may slow convergence
   - SVM: kernel trick can mitigate issues

9. **What is partial correlation and when is it useful?**
   - Correlation between X and Y after removing effect of Z
   - Useful for causal inference
   - Identifies direct relationships
   - Used in graphical models

## Practice Exercises

### Exercise 1: Implement Distance Correlation
```python
def distance_correlation_full(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """
    Implement full distance correlation with test statistic
    
    Returns:
        Dictionary with dCor, dCov, dVarX, dVarY, and test statistic
    """
    # TODO: Implement complete distance correlation
    # Include permutation test for significance
    pass
```

### Exercise 2: Robust Correlation Estimation
```python
def robust_correlation_matrix(data: pd.DataFrame, 
                            method: str = 'mcd') -> np.ndarray:
    """
    Calculate robust correlation matrix
    
    Methods:
        - 'mcd': Minimum Covariance Determinant
        - 'tyler': Tyler's M-estimator
        - 'spearman': Rank-based
    """
    # TODO: Implement robust correlation estimation
    pass
```

### Exercise 3: Dynamic Correlation Analysis
```python
class DynamicCorrelation:
    """
    Analyze time-varying correlations
    """
    
    def rolling_correlation(self, x: np.ndarray, y: np.ndarray,
                          window: int) -> np.ndarray:
        # TODO: Implement rolling window correlation
        pass
    
    def dcc_garch(self, returns: np.ndarray) -> np.ndarray:
        # TODO: Implement Dynamic Conditional Correlation GARCH
        pass
```

### Exercise 4: Multicollinearity-Robust Feature Selection
```python
def select_features_multicollinearity_aware(X: pd.DataFrame, y: np.ndarray,
                                          n_features: int) -> List[str]:
    """
    Select features considering both target correlation and multicollinearity
    
    Strategy:
    1. Rank by correlation with target
    2. Iteratively add features if VIF remains acceptable
    """
    # TODO: Implement smart feature selection
    pass
```

### Exercise 5: Correlation Network Analysis
```python
class CorrelationNetwork:
    """
    Network-based correlation analysis
    """
    
    def find_correlation_clusters(self, corr_matrix: np.ndarray,
                                threshold: float = 0.7) -> List[List[int]]:
        # TODO: Find clusters of highly correlated variables
        pass
    
    def minimum_spanning_tree(self, corr_matrix: np.ndarray) -> nx.Graph:
        # TODO: Create MST from correlation distances
        pass
```

## Advanced Topics

### 1. Copulas for Dependency Modeling
- Model complex dependencies beyond linear correlation
- Separate marginal distributions from dependency structure
- Useful for risk modeling and simulation

### 2. Information-Theoretic Measures
- Mutual Information: I(X;Y)
- Transfer Entropy: directional information flow
- Maximal Information Coefficient (MIC)

### 3. Causal Discovery
- PC Algorithm: constraint-based causal discovery
- GES: score-based causal discovery
- LiNGAM: Linear Non-Gaussian Acyclic Model

## Summary

Key takeaways:
1. **Correlation** measures linear relationships; consider non-linear alternatives
2. **Multiple correlation types** serve different purposes (Pearson, Spearman, Kendall)
3. **Multicollinearity** can severely impact model interpretation and stability
4. **Detection methods** include VIF, condition number, and eigenvalue analysis
5. **Remediation strategies** include regularization, PCA, and feature selection
6. **Different ML algorithms** have varying sensitivity to multicollinearity
7. Understanding these concepts is crucial for:
   - Feature engineering
   - Model diagnostics
   - Algorithm selection
   - Result interpretation