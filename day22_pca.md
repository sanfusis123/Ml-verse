# Day 22: Principal Component Analysis (PCA)

## Table of Contents
1. [Introduction](#1-introduction)
2. [The Intuition Behind PCA](#2-the-intuition-behind-pca)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Step-by-Step PCA Algorithm](#4-step-by-step-pca-algorithm)
5. [Implementation from Scratch](#5-implementation-from-scratch)
6. [Using Scikit-learn](#6-using-scikit-learn)
7. [Choosing the Number of Components](#7-choosing-the-number-of-components)
8. [PCA Variants and Extensions](#8-pca-variants-and-extensions)
9. [Applications and Case Studies](#9-applications-and-case-studies)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance as possible. It's one of the most widely used techniques in machine learning for feature extraction, data visualization, and noise reduction.

### Why PCA?

1. **Curse of Dimensionality**: Many ML algorithms struggle with high-dimensional data
2. **Visualization**: Project high-dimensional data to 2D/3D for visualization
3. **Noise Reduction**: Principal components often capture signal, not noise
4. **Computational Efficiency**: Fewer features mean faster training
5. **Feature Extraction**: Create new features that are linear combinations of original features

### When to Use PCA?

- **High-dimensional data**: Text data, image data, genomic data
- **Multicollinearity**: When features are highly correlated
- **Visualization needs**: Explore data structure in 2D/3D
- **Preprocessing**: Before applying distance-based algorithms
- **Compression**: Reduce storage requirements

### When NOT to Use PCA?

- **Interpretability is crucial**: PCA components are hard to interpret
- **Non-linear relationships**: PCA assumes linear relationships
- **Sparse data**: Can destroy sparsity structure
- **All features are important**: When domain knowledge says all features matter

## 2. The Intuition Behind PCA

### Geometric Interpretation

Imagine you have a cloud of points in 3D space that roughly forms an ellipsoid. The data varies most along one direction (length of ellipsoid), less in another (width), and least in the third (height).

PCA finds these directions of maximum variance:
- **First PC**: Direction of maximum variance
- **Second PC**: Direction of maximum variance orthogonal to first
- **Third PC**: Direction orthogonal to both previous PCs

### Variance and Information

**Key Insight**: Directions with high variance contain more information about the data structure.

Consider height and weight data:
- Both vary significantly → both contain information
- If everyone had same height → height has zero variance → no information

PCA identifies and keeps the directions with highest variance (most information).

### Change of Basis

PCA is fundamentally a change of basis operation:
- **Original basis**: Standard coordinate axes (features)
- **New basis**: Principal components (orthogonal directions of maximum variance)

This new basis is optimal in the sense that it decorrelates the data and orders dimensions by importance.

## 3. Mathematical Foundation

### 3.1 Covariance Matrix

For a centered data matrix X (n samples × p features), the covariance matrix is:

```
C = (1/(n-1)) * X^T * X
```

The covariance matrix:
- Is symmetric (C = C^T)
- Contains variances on diagonal
- Contains covariances off-diagonal
- Is positive semi-definite

### 3.2 Eigendecomposition

PCA involves finding eigenvalues and eigenvectors of the covariance matrix:

```
C * v = λ * v
```

Where:
- v is an eigenvector (principal component direction)
- λ is the corresponding eigenvalue (variance along that direction)

### 3.3 Properties of Principal Components

1. **Orthogonality**: All PCs are mutually orthogonal
2. **Variance**: First PC has maximum variance, second has maximum variance among directions orthogonal to first, etc.
3. **Uncorrelated**: Projections onto different PCs are uncorrelated

### 3.4 Singular Value Decomposition (SVD)

Alternative computation using SVD of centered data matrix X:

```
X = U * Σ * V^T
```

Where:
- U: Left singular vectors (n × n)
- Σ: Diagonal matrix of singular values (n × p)
- V: Right singular vectors (p × p) - these are the principal components

Relationship: V contains eigenvectors of X^T*X, and Σ² contains eigenvalues (up to scaling).

## 4. Step-by-Step PCA Algorithm

### Algorithm Steps:

1. **Standardize the data** (optional but recommended)
   - Subtract mean from each feature
   - Divide by standard deviation

2. **Compute covariance matrix**
   - C = (1/(n-1)) * X^T * X

3. **Compute eigenvalues and eigenvectors**
   - Solve: C * v = λ * v
   - Or use SVD: X = U * Σ * V^T

4. **Sort eigenvectors by eigenvalues**
   - Order from highest to lowest eigenvalue

5. **Select top k eigenvectors**
   - These are your k principal components

6. **Transform data**
   - Project data onto selected principal components
   - X_reduced = X * W, where W contains top k eigenvectors

### Mathematical Derivation

**Objective**: Find projection that maximizes variance

For projection onto unit vector w:
- Projected data: z = X * w
- Variance of projection: Var(z) = w^T * C * w

**Optimization problem**:
```
maximize    w^T * C * w
subject to  w^T * w = 1
```

Using Lagrange multipliers:
```
L = w^T * C * w - λ(w^T * w - 1)
```

Taking derivative and setting to zero:
```
∂L/∂w = 2*C*w - 2*λ*w = 0
```

This gives us: C * w = λ * w (eigenvalue equation!)

## 5. Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

class PCAFromScratch:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X, standardize=True):
        """Fit PCA model to data"""
        # Convert to numpy array
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Standardize data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        if standardize:
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1  # Avoid division by zero
            X_centered = X_centered / self.std_
        
        # Method 1: Eigendecomposition of covariance matrix
        # cov_matrix = np.cov(X_centered.T)
        # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Method 2: SVD (more numerically stable)
        U, S, Vt = svd(X_centered, full_matrices=False)
        
        # Eigenvalues from singular values
        eigenvalues = (S ** 2) / (n_samples - 1)
        eigenvectors = Vt.T
        
        # Sort by eigenvalues (already sorted by SVD, but let's be explicit)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """Transform data using fitted PCA"""
        X = np.array(X, dtype=np.float64)
        
        # Apply same preprocessing
        X_centered = X - self.mean_
        if self.std_ is not None:
            X_centered = X_centered / self.std_
        
        # Project onto principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X, standardize=True):
        """Fit and transform in one step"""
        self.fit(X, standardize)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Transform data back to original space"""
        # Project back
        X_original = np.dot(X_transformed, self.components_)
        
        # Reverse preprocessing
        if self.std_ is not None:
            X_original = X_original * self.std_
        X_original = X_original + self.mean_
        
        return X_original
    
    def get_covariance_matrix(self, X):
        """Compute and return covariance matrix"""
        X_centered = X - np.mean(X, axis=0)
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        return cov_matrix
    
    def plot_explained_variance(self):
        """Plot explained variance ratio"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model not fitted yet!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Individual explained variance
        ax1.bar(range(1, len(self.explained_variance_ratio_) + 1), 
                self.explained_variance_ratio_)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        
        # Cumulative explained variance
        cumsum = np.cumsum(self.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Demonstrate PCA step by step
def demonstrate_pca():
    # Generate synthetic data
    np.random.seed(42)
    
    # Create correlated features
    n_samples = 300
    mean = [0, 0, 0]
    cov = [[1, 0.8, 0.2],
           [0.8, 1, 0.3],
           [0.2, 0.3, 1]]
    
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Add some noise
    X += np.random.normal(0, 0.1, X.shape)
    
    # Apply PCA
    pca = PCAFromScratch(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    # Visualize original data and transformed data
    fig = plt.figure(figsize=(15, 5))
    
    # Original 3D data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.arange(n_samples), cmap='viridis')
    ax1.set_title('Original 3D Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    
    # Transformed 2D data
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                         c=np.arange(n_samples), cmap='viridis')
    ax2.set_title('PCA Transformed (2D)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True)
    
    # Explained variance
    ax3 = fig.add_subplot(133)
    ax3.bar(['PC1', 'PC2', 'PC3'], pca.explained_variance_ratio_)
    ax3.set_title('Explained Variance Ratio')
    ax3.set_ylabel('Variance Ratio')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Principal Components (as row vectors):")
    print(pca.components_)
    print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    print(f"Total variance preserved: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Demonstrate reconstruction
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\nReconstruction MSE: {reconstruction_error:.6f}")
    
    return pca, X, X_transformed

# Visualize PCA geometrically
def visualize_pca_geometry():
    # Generate 2D data with clear principal components
    np.random.seed(42)
    
    # Create elongated data
    n_points = 200
    t = np.random.randn(n_points)
    
    # Original data along diagonal
    X = np.column_stack([t + 0.5 * np.random.randn(n_points),
                        2 * t + 0.5 * np.random.randn(n_points)])
    
    # Rotate data
    angle = np.pi / 6
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    X = X @ rotation.T
    
    # Fit PCA
    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Original data
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original data')
    
    # Plot principal component directions
    mean = np.mean(X, axis=0)
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        # Scale by sqrt of variance for visualization
        scale = 3 * np.sqrt(var)
        plt.arrow(mean[0], mean[1], 
                 scale * comp[0], scale * comp[1],
                 head_width=0.2, head_length=0.2,
                 fc=f'C{i+1}', ec=f'C{i+1}',
                 label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%} var)')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('PCA: Principal Component Directions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
```

## 6. Using Scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, fetch_olivetti_faces
import seaborn as sns

# Comprehensive PCA workflow
class PCAWorkflow:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def analyze_dataset(self, X, feature_names=None):
        """Complete PCA analysis of dataset"""
        print(f"Dataset shape: {X.shape}")
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA with all components
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Find number of components for 95% variance
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        print(f"Components for 95% variance: {n_components_95}")
        print(f"Dimension reduction: {X.shape[1]} → {n_components_95} "
              f"({n_components_95/X.shape[1]:.1%})")
        
        # Visualize
        self._plot_analysis(pca_full, n_components_95)
        
        # Fit PCA with optimal components
        self.pca = PCA(n_components=n_components_95)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Component interpretation
        if feature_names is not None:
            self._interpret_components(self.pca, feature_names)
        
        return X_reduced
    
    def _plot_analysis(self, pca, n_components_95):
        """Comprehensive PCA visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scree plot
        ax = axes[0, 0]
        ax.plot(range(1, len(pca.explained_variance_) + 1),
                pca.explained_variance_, 'bo-')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        ax.set_yscale('log')
        ax.grid(True)
        
        # 2. Explained variance
        ax = axes[0, 1]
        ax.bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
               pca.explained_variance_ratio_[:20])
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Variance Explained by Each Component')
        
        # 3. Cumulative variance
        ax = axes[1, 0]
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
        ax.axvline(x=n_components_95, color='g', linestyle='--')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained')
        ax.legend()
        ax.grid(True)
        
        # 4. Component correlation heatmap (first 10)
        ax = axes[1, 1]
        n_show = min(10, pca.components_.shape[0])
        comp_corr = np.corrcoef(pca.components_[:n_show])
        sns.heatmap(comp_corr, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Principal Component Correlations')
        
        plt.tight_layout()
        plt.show()
    
    def _interpret_components(self, pca, feature_names, n_components=3):
        """Interpret principal components"""
        print("\nPrincipal Component Interpretation:")
        print("-" * 50)
        
        for i in range(min(n_components, pca.n_components_)):
            print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]:.1%} variance):")
            
            # Get component loadings
            loadings = pca.components_[i]
            
            # Sort by absolute value
            idx = np.argsort(np.abs(loadings))[::-1]
            
            # Show top contributing features
            print("Top contributing features:")
            for j in idx[:5]:
                print(f"  {feature_names[j]}: {loadings[j]:.3f}")

# Example: PCA on high-dimensional data
def pca_digits_example():
    """PCA on handwritten digits dataset"""
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print("Digits Dataset PCA Analysis")
    print("=" * 50)
    
    # Apply PCA workflow
    workflow = PCAWorkflow()
    X_reduced = workflow.analyze_dataset(X)
    
    # Visualize in 2D
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA projection colored by digit
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X))
    
    scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', 
                         alpha=0.6, edgecolors='k', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('Digits Dataset - 2D PCA Projection')
    plt.colorbar(scatter, ax=ax1, label='Digit')
    
    # Reconstruction quality
    reconstruction_errors = []
    n_components_range = range(1, 65, 5)
    
    for n_comp in n_components_range:
        pca_temp = PCA(n_components=n_comp)
        X_scaled = StandardScaler().fit_transform(X)
        X_reduced_temp = pca_temp.fit_transform(X_scaled)
        X_reconstructed = pca_temp.inverse_transform(X_reduced_temp)
        error = np.mean((X_scaled - X_reconstructed) ** 2)
        reconstruction_errors.append(error)
    
    ax2.plot(n_components_range, reconstruction_errors, 'bo-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('Reconstruction Error vs Components')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize principal components as images
    pca_vis = PCA(n_components=16)
    pca_vis.fit(StandardScaler().fit_transform(X))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(pca_vis.components_[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'PC{i+1}')
        ax.axis('off')
    
    plt.suptitle('First 16 Principal Components as Images')
    plt.tight_layout()
    plt.show()

# Example: PCA for face recognition
def pca_faces_example():
    """PCA on face images (Eigenfaces)"""
    # Load face data
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target
    
    n_samples, n_features = X.shape
    image_shape = (64, 64)
    
    print(f"Face dataset: {n_samples} samples, {n_features} features")
    
    # Apply PCA
    n_components = 150
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X)
    
    print(f"Reduced from {n_features} to {n_components} dimensions")
    print(f"Variance preserved: {sum(pca.explained_variance_ratio_):.1%}")
    
    # Visualize eigenfaces
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    for i, ax in enumerate(axes.ravel()):
        if i < n_components:
            ax.imshow(pca.components_[i].reshape(image_shape), cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
        ax.axis('off')
    
    plt.suptitle('Top 15 Eigenfaces')
    plt.tight_layout()
    plt.show()
    
    # Face reconstruction with different numbers of components
    n_faces_to_show = 5
    n_components_list = [10, 50, 100, 150]
    
    fig, axes = plt.subplots(len(n_components_list) + 1, n_faces_to_show, 
                            figsize=(12, 10))
    
    # Original faces
    for i in range(n_faces_to_show):
        axes[0, i].imshow(X[i].reshape(image_shape), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
    
    # Reconstructions
    for row, n_comp in enumerate(n_components_list):
        pca_temp = PCA(n_components=n_comp)
        X_reduced = pca_temp.fit_transform(X)
        X_reconstructed = pca_temp.inverse_transform(X_reduced)
        
        for i in range(n_faces_to_show):
            axes[row+1, i].imshow(X_reconstructed[i].reshape(image_shape), 
                                 cmap='gray')
            if i == 0:
                axes[row+1, i].set_ylabel(f'{n_comp} PCs')
            axes[row+1, i].axis('off')
    
    plt.suptitle('Face Reconstruction with Different Numbers of Components')
    plt.tight_layout()
    plt.show()
```

## 7. Choosing the Number of Components

### 7.1 Methods for Selecting Components

```python
class ComponentSelection:
    """Methods for choosing optimal number of PCA components"""
    
    @staticmethod
    def variance_threshold(X, threshold=0.95):
        """Select components to preserve given variance threshold"""
        pca = PCA()
        pca.fit(X)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= threshold) + 1
        
        return n_components, cumsum[n_components-1]
    
    @staticmethod
    def elbow_method(X, max_components=None):
        """Find elbow in scree plot"""
        if max_components is None:
            max_components = min(X.shape)
        
        pca = PCA(n_components=max_components)
        pca.fit(X)
        
        # Calculate second derivative
        explained_var = pca.explained_variance_ratio_
        first_diff = np.diff(explained_var)
        second_diff = np.diff(first_diff)
        
        # Find elbow (maximum second derivative)
        elbow = np.argmax(second_diff) + 2  # +2 because of double diff
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
        plt.axvline(x=elbow, color='r', linestyle='--', 
                   label=f'Elbow at {elbow} components')
        plt.xlabel('Component Number')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Elbow Method for Component Selection')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return elbow
    
    @staticmethod
    def kaiser_criterion(X):
        """Select components with eigenvalues > 1 (for standardized data)"""
        # Standardize data
        X_std = StandardScaler().fit_transform(X)
        
        pca = PCA()
        pca.fit(X_std)
        
        # Components with eigenvalue > 1
        n_components = np.sum(pca.explained_variance_ > 1)
        
        return n_components
    
    @staticmethod
    def cross_validation(X, y, cv=5):
        """Select components using reconstruction error or downstream task performance"""
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        n_components_range = range(1, min(X.shape[0], X.shape[1], 50), 2)
        cv_scores = []
        
        for n_comp in n_components_range:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_comp)),
                ('classifier', LogisticRegression(max_iter=1000))
            ])
            
            scores = cross_val_score(pipeline, X, y, cv=cv)
            cv_scores.append(scores.mean())
        
        # Find optimal
        optimal_idx = np.argmax(cv_scores)
        optimal_components = n_components_range[optimal_idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(n_components_range, cv_scores, 'bo-')
        plt.axvline(x=optimal_components, color='r', linestyle='--',
                   label=f'Optimal: {optimal_components} components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cross-validation Score')
        plt.title('Component Selection via Cross-validation')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return optimal_components
    
    @staticmethod
    def mle_estimation(X):
        """Minka's MLE for estimating intrinsic dimensionality"""
        pca_mle = PCA(n_components='mle')
        pca_mle.fit(X)
        return pca_mle.n_components_

# Demonstrate different selection methods
def compare_selection_methods():
    # Generate data with known intrinsic dimension
    np.random.seed(42)
    
    # 5D data embedded in 20D space
    n_samples = 500
    intrinsic_dim = 5
    ambient_dim = 20
    
    # Generate low-dimensional data
    low_dim_data = np.random.randn(n_samples, intrinsic_dim)
    
    # Random projection to high dimension
    projection = np.random.randn(intrinsic_dim, ambient_dim)
    projection = projection / np.linalg.norm(projection, axis=0)
    
    # Project and add noise
    X = low_dim_data @ projection
    X += 0.1 * np.random.randn(n_samples, ambient_dim)
    
    # Add some dummy labels for CV method
    y = (low_dim_data[:, 0] > 0).astype(int)
    
    print(f"True intrinsic dimension: {intrinsic_dim}")
    print(f"Ambient dimension: {ambient_dim}")
    print("\nComponent selection results:")
    
    # Apply different methods
    selector = ComponentSelection()
    
    # Variance threshold
    n_var, var_preserved = selector.variance_threshold(X, 0.95)
    print(f"Variance threshold (95%): {n_var} components")
    
    # Elbow method
    n_elbow = selector.elbow_method(X)
    print(f"Elbow method: {n_elbow} components")
    
    # Kaiser criterion
    n_kaiser = selector.kaiser_criterion(X)
    print(f"Kaiser criterion: {n_kaiser} components")
    
    # Cross-validation
    n_cv = selector.cross_validation(X, y)
    print(f"Cross-validation: {n_cv} components")
```

## 8. PCA Variants and Extensions

### 8.1 Incremental PCA

```python
from sklearn.decomposition import IncrementalPCA

def incremental_pca_example():
    """PCA for datasets that don't fit in memory"""
    # Simulate large dataset processed in batches
    n_samples = 10000
    n_features = 100
    batch_size = 1000
    n_components = 20
    
    # Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        # Simulate loading batch from disk
        batch = np.random.randn(batch_size, n_features)
        
        # Partial fit
        ipca.partial_fit(batch)
        
        if i % 2000 == 0:
            print(f"Processed {i + batch_size} samples...")
    
    print(f"\nFinal explained variance ratio: {ipca.explained_variance_ratio_[:5]}")
    
    # Compare with regular PCA on subset
    subset_size = 2000
    subset = np.random.randn(subset_size, n_features)
    
    regular_pca = PCA(n_components=n_components)
    regular_pca.fit(subset)
    
    print(f"Regular PCA variance ratio: {regular_pca.explained_variance_ratio_[:5]}")

### 8.2 Sparse PCA

from sklearn.decomposition import SparsePCA

def sparse_pca_example():
    """Sparse PCA for interpretable components"""
    # Generate data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Create data with sparse structure
    X = np.random.randn(n_samples, n_features)
    X[:, :5] += 2 * np.random.randn(n_samples, 5)  # First 5 features important
    X[:, 10:15] += 1.5 * np.random.randn(n_samples, 5)  # Next 5 moderately important
    
    # Regular PCA
    regular_pca = PCA(n_components=3)
    regular_pca.fit(X)
    
    # Sparse PCA
    sparse_pca = SparsePCA(n_components=3, alpha=1.0, random_state=42)
    sparse_pca.fit(X)
    
    # Compare components
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Regular PCA components
    im1 = ax1.imshow(regular_pca.components_, aspect='auto', cmap='RdBu_r')
    ax1.set_title('Regular PCA Components')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Components')
    plt.colorbar(im1, ax=ax1)
    
    # Sparse PCA components
    im2 = ax2.imshow(sparse_pca.components_, aspect='auto', cmap='RdBu_r')
    ax2.set_title('Sparse PCA Components')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Components')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Count non-zero loadings
    regular_nonzero = np.sum(np.abs(regular_pca.components_) > 0.01)
    sparse_nonzero = np.sum(np.abs(sparse_pca.components_) > 0.01)
    
    print(f"Non-zero loadings - Regular PCA: {regular_nonzero}")
    print(f"Non-zero loadings - Sparse PCA: {sparse_nonzero}")

### 8.3 Kernel PCA

from sklearn.decomposition import KernelPCA

def kernel_pca_example():
    """Kernel PCA for non-linear dimensionality reduction"""
    # Generate non-linear data (Swiss roll)
    from sklearn.datasets import make_swiss_roll
    
    n_samples = 1000
    X, color = make_swiss_roll(n_samples, noise=0.2, random_state=42)
    
    # Apply different PCA variants
    transformations = {
        'Linear PCA': PCA(n_components=2),
        'Kernel PCA (RBF)': KernelPCA(n_components=2, kernel='rbf', gamma=0.04),
        'Kernel PCA (Polynomial)': KernelPCA(n_components=2, kernel='poly', degree=3),
        'Kernel PCA (Sigmoid)': KernelPCA(n_components=2, kernel='sigmoid')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, transformer) in enumerate(transformations.items()):
        # Transform data
        X_transformed = transformer.fit_transform(X)
        
        # Plot
        scatter = axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                   c=color, cmap='viridis', alpha=0.6)
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Component 1')
        axes[idx].set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.show()

### 8.4 Randomized PCA

def randomized_pca_example():
    """Randomized PCA for large datasets"""
    # Generate large dataset
    n_samples = 5000
    n_features = 1000
    n_components = 50
    
    print(f"Dataset size: {n_samples} x {n_features}")
    
    # Create data with decaying variance
    variances = 1.0 / np.arange(1, n_features + 1)
    X = np.random.randn(n_samples, n_features) * np.sqrt(variances)
    
    # Time comparison
    import time
    
    # Full PCA
    start = time.time()
    pca_full = PCA(n_components=n_components, svd_solver='full')
    pca_full.fit(X)
    time_full = time.time() - start
    
    # Randomized PCA
    start = time.time()
    pca_random = PCA(n_components=n_components, svd_solver='randomized', 
                     random_state=42)
    pca_random.fit(X)
    time_random = time.time() - start
    
    print(f"\nTime - Full SVD: {time_full:.2f}s")
    print(f"Time - Randomized SVD: {time_random:.2f}s")
    print(f"Speedup: {time_full/time_random:.1f}x")
    
    # Compare results
    print(f"\nExplained variance ratio difference: "
          f"{np.mean(np.abs(pca_full.explained_variance_ratio_ - pca_random.explained_variance_ratio_)):.6f}")
```

## 9. Applications and Case Studies

### 9.1 PCA for Data Visualization

```python
def pca_visualization_pipeline():
    """Complete pipeline for visualizing high-dimensional data"""
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    
    # Load dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca_all = pca.fit_transform(X_scaled)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 2D PCA plot
    ax1 = plt.subplot(2, 3, 1)
    colors = ['navy', 'turquoise', 'darkorange']
    for target, color in zip(np.unique(y), colors):
        indices = y == target
        ax1.scatter(X_pca_all[indices, 0], X_pca_all[indices, 1], 
                   c=color, label=target_names[target], alpha=0.7)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('2D PCA Projection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 3D PCA plot
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    for target, color in zip(np.unique(y), colors):
        indices = y == target
        ax2.scatter(X_pca_all[indices, 0], X_pca_all[indices, 1], 
                   X_pca_all[indices, 2], c=color, label=target_names[target], 
                   alpha=0.7)
    ax2.set_xlabel(f'PC1')
    ax2.set_ylabel(f'PC2')
    ax2.set_zlabel(f'PC3')
    ax2.set_title('3D PCA Projection')
    ax2.legend()
    
    # 3. Biplot
    ax3 = plt.subplot(2, 3, 3)
    # Plot samples
    for target, color in zip(np.unique(y), colors):
        indices = y == target
        ax3.scatter(X_pca_all[indices, 0], X_pca_all[indices, 1], 
                   c=color, alpha=0.5)
    
    # Plot feature vectors
    for i, feature in enumerate(feature_names):
        ax3.arrow(0, 0, 
                 pca.components_[0, i] * 3,
                 pca.components_[1, i] * 3,
                 color='r', alpha=0.5, head_width=0.1)
        ax3.text(pca.components_[0, i] * 3.2,
                pca.components_[1, i] * 3.2,
                feature, fontsize=8)
    
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('PCA Biplot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loading plot
    ax4 = plt.subplot(2, 3, 4)
    loadings = pca.components_[:2].T
    ax4.scatter(loadings[:, 0], loadings[:, 1])
    for i, feature in enumerate(feature_names):
        ax4.annotate(feature, (loadings[i, 0], loadings[i, 1]), fontsize=8)
    ax4.set_xlabel('PC1 Loadings')
    ax4.set_ylabel('PC2 Loadings')
    ax4.set_title('Feature Loadings')
    ax4.grid(True, alpha=0.3)
    
    # 5. Correlation circle
    ax5 = plt.subplot(2, 3, 5)
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=1)
    ax5.add_patch(circle)
    
    for i, feature in enumerate(feature_names):
        ax5.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 color='blue', alpha=0.5, head_width=0.05)
        ax5.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, 
                feature, fontsize=8)
    
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title('Correlation Circle')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Contribution plot
    ax6 = plt.subplot(2, 3, 6)
    contributions = np.abs(pca.components_[:3])
    im = ax6.imshow(contributions, cmap='YlOrRd', aspect='auto')
    ax6.set_yticks(range(3))
    ax6.set_yticklabels([f'PC{i+1}' for i in range(3)])
    ax6.set_xticks(range(len(feature_names)))
    ax6.set_xticklabels(feature_names, rotation=45, ha='right')
    ax6.set_title('Feature Contributions to PCs')
    plt.colorbar(im, ax=ax6)
    
    plt.suptitle('Comprehensive PCA Analysis - Wine Dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

### 9.2 PCA for Anomaly Detection

def pca_anomaly_detection():
    """Using PCA reconstruction error for anomaly detection"""
    from sklearn.datasets import make_blobs
    
    # Generate normal data
    n_samples = 300
    n_outliers = 30
    
    X_normal, _ = make_blobs(n_samples=n_samples, centers=1, n_features=10,
                             center_box=(-10, 10), random_state=42)
    
    # Generate outliers
    X_outliers = np.random.uniform(-20, 20, (n_outliers, 10))
    
    # Combine data
    X = np.vstack([X_normal, X_outliers])
    y_true = np.hstack([np.zeros(n_samples), np.ones(n_outliers)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y_true = y_true[indices]
    
    # Fit PCA on normal data assumption (unsupervised)
    pca = PCA(n_components=5)
    pca.fit(X)
    
    # Calculate reconstruction errors
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
    
    # Determine threshold (e.g., 95th percentile)
    threshold = np.percentile(reconstruction_errors, 95)
    
    # Detect anomalies
    y_pred = (reconstruction_errors > threshold).astype(int)
    
    # Evaluate
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                              average='binary')
    auc = roc_auc_score(y_true, reconstruction_errors)
    
    print("PCA Anomaly Detection Results:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reconstruction error distribution
    ax1.hist(reconstruction_errors[y_true == 0], bins=30, alpha=0.5, 
             label='Normal', density=True)
    ax1.hist(reconstruction_errors[y_true == 1], bins=30, alpha=0.5, 
             label='Anomaly', density=True)
    ax1.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Reconstruction Error Distribution')
    ax1.legend()
    
    # 2D visualization
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=reconstruction_errors,
                         cmap='YlOrRd', alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Mark detected anomalies
    anomaly_indices = np.where(y_pred == 1)[0]
    ax2.scatter(X_2d[anomaly_indices, 0], X_2d[anomaly_indices, 1], 
               marker='x', s=100, c='red', label='Detected Anomaly')
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA Projection with Anomaly Detection')
    plt.colorbar(scatter, ax=ax2, label='Reconstruction Error')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

### 9.3 PCA for Feature Engineering

def pca_feature_engineering():
    """Using PCA for feature engineering in ML pipeline"""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Load data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare different feature engineering approaches
    results = []
    
    # 1. Original features
    rf_original = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_original.fit(X_train_scaled, y_train)
    y_pred_original = rf_original.predict(X_test_scaled)
    
    results.append({
        'Method': 'Original Features',
        'n_features': X_train.shape[1],
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_original)),
        'R2': r2_score(y_test, y_pred_original)
    })
    
    # 2. PCA features only
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    rf_pca = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_pca.fit(X_train_pca, y_train)
    y_pred_pca = rf_pca.predict(X_test_pca)
    
    results.append({
        'Method': 'PCA Features Only',
        'n_features': 5,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_pca)),
        'R2': r2_score(y_test, y_pred_pca)
    })
    
    # 3. Original + PCA features
    X_train_combined = np.hstack([X_train_scaled, X_train_pca])
    X_test_combined = np.hstack([X_test_scaled, X_test_pca])
    
    rf_combined = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_combined.fit(X_train_combined, y_train)
    y_pred_combined = rf_combined.predict(X_test_combined)
    
    results.append({
        'Method': 'Original + PCA Features',
        'n_features': X_train_scaled.shape[1] + 5,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_combined)),
        'R2': r2_score(y_test, y_pred_combined)
    })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("Feature Engineering Comparison:")
    print(results_df.to_string(index=False))
    
    # Feature importance analysis
    feature_names = list(housing.feature_names) + [f'PC{i+1}' for i in range(5)]
    importances = rf_combined.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance: Original + PCA Features')
    plt.tight_layout()
    plt.show()
    
    return results_df
```

## 10. Interview Questions

### Q1: Explain PCA in simple terms to a non-technical stakeholder.
**Answer**: PCA is like taking a photograph of a 3D sculpture from the best angle. Imagine you have a complex sculpture and can only take 2D photos. PCA finds the angles that capture the most information - showing the most important features and variations. Similarly, when we have data with many dimensions (like customer data with age, income, purchase history, etc.), PCA finds the most informative "views" that capture the essence of the data in fewer dimensions.

### Q2: What are the assumptions of PCA?
**Answer**:
1. **Linearity**: PCA assumes linear relationships between variables
2. **Large variances = important structure**: Assumes directions of large variance are important
3. **Orthogonal components**: Assumes principal components are orthogonal
4. **Statistical assumptions**: Works best when variables are continuous and follow roughly normal distributions
5. **Scale sensitivity**: Sensitive to variable scales (usually need standardization)

Note: PCA is quite robust and often works well even when assumptions are violated.

### Q3: When should you standardize data before PCA?
**Answer**: Almost always standardize when:
- Features have different units (e.g., dollars vs. years)
- Features have vastly different scales
- You want all features to contribute equally initially

Don't standardize when:
- All features are already on same scale
- The natural scale differences are meaningful
- Working with spectral data or images where scale matters

### Q4: How do you interpret principal components?
**Answer**: 
1. **Look at loadings**: Component loadings show how much each original feature contributes
2. **Identify patterns**: Features with high loadings in same component are related
3. **Sign matters**: Positive vs negative loadings indicate opposite relationships
4. **Name components**: Try to identify themes (e.g., "size" if height and weight load together)
5. **Check variance explained**: More important components explain more variance

Example: PC1 with high loadings on income, spending, and credit limit might represent "financial capacity"

### Q5: What's the difference between PCA and Factor Analysis?
**Answer**:
- **PCA**: Finds directions of maximum variance, purely mathematical transformation
- **Factor Analysis**: Assumes underlying latent factors, models observed variables as linear combinations of factors plus noise

Key differences:
- PCA: No error term, all variance explained
- FA: Includes unique variance/error for each variable
- PCA: Components are data transformations
- FA: Factors are theoretical constructs

### Q6: How does PCA relate to eigendecomposition and SVD?
**Answer**: 
- PCA uses eigendecomposition of the covariance matrix: C = VΛV^T
- Eigenvectors (V) are principal component directions
- Eigenvalues (Λ) represent variance along each PC

SVD relationship:
- For centered data matrix X: X = UΣV^T
- V contains the principal components
- Σ²/(n-1) gives eigenvalues
- More numerically stable than eigendecomposition

### Q7: Can PCA be used for feature selection?
**Answer**: Not directly, but indirectly:
- **Direct use**: PCA creates new features (components), doesn't select original features
- **Indirect use**: 
  1. Examine loadings to identify important original features
  2. Use PCA for dimensionality reduction before modeling
  3. Sparse PCA can zero out loadings (closer to feature selection)

Better alternatives for feature selection: Mutual information, LASSO, Random Forest importance

### Q8: What are the limitations of PCA?
**Answer**:
1. **Linear only**: Can't capture non-linear relationships
2. **Interpretability**: Components are linear combinations, hard to interpret
3. **Outlier sensitive**: Outliers can heavily influence components
4. **Variance ≠ Predictive power**: High variance doesn't always mean useful for prediction
5. **Dense components**: All original features contribute to each PC

### Q9: How do you choose the number of principal components?
**Answer**: Multiple methods:
1. **Variance threshold**: Keep components explaining 95% variance
2. **Elbow method**: Look for "elbow" in scree plot
3. **Kaiser criterion**: Keep components with eigenvalues > 1
4. **Cross-validation**: Choose based on downstream task performance
5. **Parallel analysis**: Compare with random data eigenvalues
6. **Business requirements**: Based on visualization needs or computational constraints

### Q10: Explain the difference between PCA and LDA.
**Answer**:
- **PCA**: Unsupervised, maximizes variance
- **LDA**: Supervised, maximizes class separability

Key differences:
- PCA: Doesn't use class labels, finds directions of maximum variance
- LDA: Uses class labels, finds directions that maximize between-class variance relative to within-class variance
- PCA: Can find up to min(n-1, p) components
- LDA: Can find at most (c-1) components where c is number of classes

### Q11: How do you handle missing values with PCA?
**Answer**: Several approaches:
1. **Deletion**: Remove rows/columns with missing values (loses information)
2. **Imputation first**: Fill missing values (mean, median, KNN, etc.) then apply PCA
3. **Iterative PCA**: Alternately impute and compute PCA
4. **Probabilistic PCA**: Handles missing values naturally through EM algorithm
5. **Skip missing**: Some implementations can ignore missing values in calculations

Best practice: Understand missing data mechanism and choose accordingly

### Q12: What is the computational complexity of PCA?
**Answer**:
- **Covariance matrix computation**: O(p²n) where p=features, n=samples
- **Eigendecomposition**: O(p³)
- **SVD approach**: O(min(p²n, pn²))
- **Overall**: O(p²n + p³) for full PCA

For large datasets:
- Randomized PCA: O(p²k) where k=components
- Incremental PCA: O(npk) per batch

### Q13: Can PCA be used for categorical variables?
**Answer**: Not directly, but alternatives exist:
1. **Encode first**: One-hot encode then apply PCA (creates many sparse features)
2. **MCA**: Multiple Correspondence Analysis for categorical data
3. **PCA-MIX**: Handles mixed continuous/categorical data
4. **FAMD**: Factor Analysis of Mixed Data
5. **Consider alternatives**: t-SNE, UMAP for mixed data

### Q14: What's the relationship between PCA and autoencoders?
**Answer**: 
- **Linear autoencoder** with MSE loss = PCA
- Both perform dimensionality reduction
- Both learn representations

Differences:
- PCA: Linear only, closed-form solution
- Autoencoders: Can be non-linear, learned iteratively
- PCA: Orthogonal components
- Autoencoders: No orthogonality constraint

### Q15: How do you validate PCA results?
**Answer**:
1. **Reconstruction error**: Check how well data can be reconstructed
2. **Stability**: Apply to bootstrap samples, check component stability
3. **Interpretability**: Do components make domain sense?
4. **Downstream performance**: Does PCA improve model performance?
5. **Visual inspection**: Plot data in PC space, check separation/structure
6. **Statistical tests**: Test for sphericity, sampling adequacy (KMO test)

### Q16: What is Kernel PCA and when to use it?
**Answer**: Kernel PCA performs PCA in a high-dimensional feature space using kernel trick:
- Maps data to higher dimension where it might be linearly separable
- Computes PCA without explicitly computing the mapping

When to use:
- Non-linear relationships in data
- Circular or spiral patterns
- When linear PCA gives poor results
- Before non-linear classifiers for better features

Common kernels: RBF, polynomial, sigmoid

### Q17: How does PCA handle multicollinearity?
**Answer**: PCA naturally handles multicollinearity:
1. **Combines correlated variables**: PCs are linear combinations of original features
2. **Orthogonal output**: PCs are uncorrelated by construction
3. **Dimension reduction**: Removes redundant information
4. **Stabilizes models**: Reduces condition number of data matrix

This makes PCA useful as preprocessing for algorithms sensitive to multicollinearity (e.g., linear regression).

### Q18: What's the difference between PCA and ICA?
**Answer**:
- **PCA**: Finds orthogonal components maximizing variance
- **ICA**: Finds independent components maximizing statistical independence

Key differences:
- PCA: Second-order statistics (covariance)
- ICA: Higher-order statistics (requires non-Gaussianity)
- PCA: Components are orthogonal
- ICA: Components are independent (stronger than uncorrelated)
- PCA: Unique solution (up to sign)
- ICA: Multiple solutions possible

### Q19: How do you implement PCA for streaming data?
**Answer**: Use Incremental PCA approaches:
1. **Moving window**: Apply PCA to recent window of data
2. **Incremental PCA**: Update components as new data arrives
3. **Forgetting factor**: Weight recent data more heavily
4. **CCIPCA**: Candid Covariance-free Incremental PCA
5. **Online SVD**: Update SVD incrementally

Challenges: Concept drift, computational efficiency, memory constraints

### Q20: What are common mistakes when using PCA?
**Answer**:
1. **Not standardizing**: When features have different scales
2. **Information loss**: Removing too many components
3. **Interpretation**: Over-interpreting component meanings
4. **Using for prediction**: PCA optimizes for variance, not prediction
5. **Applying blindly**: Not checking if PCA is appropriate
6. **Training/test leakage**: Fitting PCA on test data
7. **Ignoring signs**: PC signs are arbitrary, can flip
8. **Assuming normality**: PCA doesn't require normality but works better with it

### Q21: How do you explain the proportion of variance explained?
**Answer**: 
"Proportion of variance explained" tells us how much of the total variability in the data is captured by each principal component.

Example: If PC1 explains 60% variance:
- 60% of the differences between data points can be explained by their positions along PC1
- The remaining 40% is spread across other components
- It's like saying "this one direction captures 60% of what makes these data points different from each other"

Cumulative variance shows total information retained when using multiple PCs.

### Q22: Can PCA be used for time series data?
**Answer**: Yes, but with considerations:
1. **Standard PCA**: Treats time points as independent (ignores temporal structure)
2. **Functional PCA**: For smooth time series
3. **Dynamic PCA**: Incorporates time lags
4. **SSA**: Singular Spectrum Analysis for time series decomposition

Best practices:
- Consider temporal dependencies
- May need to detrend/deseasonalize first
- Window-based PCA for local patterns
- Combine with time series specific methods