# Day 23: K-Means Clustering

## Table of Contents
1. [Introduction](#1-introduction)
2. [The Intuition Behind K-Means](#2-the-intuition-behind-kmeans)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [The K-Means Algorithm](#4-the-kmeans-algorithm)
5. [Implementation from Scratch](#5-implementation-from-scratch)
6. [Using Scikit-learn](#6-using-scikit-learn)
7. [Choosing the Optimal K](#7-choosing-the-optimal-k)
8. [K-Means Variants and Improvements](#8-kmeans-variants-and-improvements)
9. [Applications and Case Studies](#9-applications-and-case-studies)
10. [Interview Questions](#10-interview-questions)

## 1. Introduction

K-Means is one of the most popular unsupervised learning algorithms used for clustering. It partitions n observations into k clusters where each observation belongs to the cluster with the nearest mean (cluster center or centroid).

### Why K-Means?

1. **Simplicity**: Easy to understand and implement
2. **Efficiency**: Computationally efficient for large datasets
3. **Scalability**: Works well with many data points
4. **Versatility**: Applicable to various domains
5. **Foundation**: Basis for more advanced clustering methods

### When to Use K-Means?

- **Customer segmentation**: Group customers by behavior
- **Image compression**: Reduce colors in images
- **Document clustering**: Organize similar documents
- **Anomaly detection**: Identify outliers
- **Feature learning**: Learn representations

### When NOT to Use K-Means?

- **Non-spherical clusters**: K-means assumes spherical clusters
- **Varying cluster sizes**: Struggles with very different cluster sizes
- **Varying densities**: Assumes similar density across clusters
- **High dimensions**: Curse of dimensionality affects distance metrics
- **Categorical data**: Designed for continuous data

## 2. The Intuition Behind K-Means

### The Basic Idea

Imagine you have a room full of people and want to organize them into k groups based on their positions. K-means would:

1. **Pick k people randomly** as initial group leaders
2. **Everyone joins the nearest leader** forming groups
3. **Find the center of each group** and make that the new leader position
4. **Repeat** until groups stabilize

### Visual Intuition

```
Initial:        After Step 1:     After Step 2:      Final:
x x   x x       [x x] [x x]      [x x] [x x]       [xxx] [xxx]
 x x x x    →    x x x x     →    •x •x x      →    •    •
x   x   x       x   x   x        x   x   x         [xxx] [xxx]

x: data point, []: cluster assignment, •: centroid
```

### Key Concepts

1. **Centroid**: The mean position of all points in a cluster
2. **Assignment**: Each point belongs to exactly one cluster
3. **Iteration**: Alternates between assignment and update steps
4. **Convergence**: Stops when assignments don't change

## 3. Mathematical Foundation

### 3.1 Objective Function

K-means minimizes the within-cluster sum of squares (WCSS):

```
J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ ||xᵢ - μⱼ||²
```

Where:
- n = number of data points
- k = number of clusters
- wᵢⱼ = 1 if xᵢ belongs to cluster j, 0 otherwise
- xᵢ = i-th data point
- μⱼ = centroid of cluster j
- ||·|| = Euclidean distance

### 3.2 The Two-Step Optimization

K-means alternates between two steps:

**Step 1: Assignment** (fix centroids, optimize assignments)
```
wᵢⱼ = 1 if j = argmin ||xᵢ - μⱼ||²
      0 otherwise
```

**Step 2: Update** (fix assignments, optimize centroids)
```
μⱼ = (Σᵢ wᵢⱼ xᵢ) / (Σᵢ wᵢⱼ)
```

### 3.3 Convergence Properties

1. **Monotonic decrease**: Objective function decreases each iteration
2. **Finite convergence**: Must converge in finite steps (finite possible assignments)
3. **Local minimum**: Only guaranteed to find local, not global minimum
4. **Initialization dependent**: Different starts → different results

### 3.4 Complexity Analysis

- **Time Complexity**: O(n × k × d × i)
  - n = number of points
  - k = number of clusters
  - d = dimensions
  - i = iterations
- **Space Complexity**: O(n × d + k × d)

## 4. The K-Means Algorithm

### Algorithm Steps:

```
1. Initialize k cluster centers randomly
2. Repeat until convergence:
   a. Assign each point to nearest center
   b. Update centers as mean of assigned points
3. Return cluster assignments and centers
```

### Detailed Algorithm:

```python
Algorithm: K-Means Clustering
Input: Data X = {x₁, x₂, ..., xₙ}, Number of clusters k
Output: Cluster assignments C = {c₁, c₂, ..., cₙ}, Centroids M = {μ₁, μ₂, ..., μₖ}

1. Initialize: Randomly select k points from X as initial centroids M
2. Repeat:
   3. Assignment Step:
      For each xᵢ in X:
         cᵢ = argmin_j ||xᵢ - μⱼ||²
   
   4. Update Step:
      For each cluster j = 1 to k:
         μⱼ = mean({xᵢ : cᵢ = j})
   
   5. Check Convergence:
      If no change in assignments, stop
      
6. Return C and M
```

### Initialization Methods

1. **Random**: Select k random points as initial centers
2. **K-means++**: Smart initialization for better results
3. **Random partition**: Randomly assign points, then compute centers
4. **Furthest first**: Greedily select points far from existing centers

## 5. Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, init='random', random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centers(self, X):
        """Initialize cluster centers"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        if self.init == 'random':
            # Random initialization
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centers = X[indices].copy()
            
        elif self.init == 'k-means++':
            # K-means++ initialization
            centers = []
            
            # Choose first center randomly
            centers.append(X[np.random.randint(n_samples)])
            
            for _ in range(1, self.n_clusters):
                # Calculate distances to nearest center
                distances = np.min(cdist(X, centers, 'euclidean'), axis=1)
                
                # Probability proportional to squared distance
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                
                # Choose next center
                cumsum = probabilities.cumsum()
                r = np.random.rand()
                centers.append(X[np.searchsorted(cumsum, r)])
            
            centers = np.array(centers)
            
        else:
            raise ValueError(f"Unknown init method: {self.init}")
        
        return centers
    
    def _assign_clusters(self, X, centers):
        """Assign each point to nearest center"""
        distances = cdist(X, centers, 'euclidean')
        return np.argmin(distances, axis=1)
    
    def _update_centers(self, X, labels):
        """Update centers as mean of assigned points"""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centers[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster - reinitialize randomly
                centers[k] = X[np.random.randint(X.shape[0])]
        
        return centers
    
    def _calculate_inertia(self, X, labels, centers):
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-means clustering"""
        X = np.array(X)
        
        # Initialize centers
        self.cluster_centers_ = self._initialize_centers(X)
        
        prev_labels = None
        
        for iteration in range(self.max_iters):
            # Assignment step
            self.labels_ = self._assign_clusters(X, self.cluster_centers_)
            
            # Check convergence
            if prev_labels is not None and np.array_equal(self.labels_, prev_labels):
                break
            
            # Update step
            self.cluster_centers_ = self._update_centers(X, self.labels_)
            
            prev_labels = self.labels_.copy()
            self.n_iter_ = iteration + 1
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X, self.labels_, self.cluster_centers_)
        
        return self
    
    def predict(self, X):
        """Predict cluster for new points"""
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        """Fit and return cluster assignments"""
        self.fit(X)
        return self.labels_
    
    def transform(self, X):
        """Transform X to cluster-distance space"""
        X = np.array(X)
        return cdist(X, self.cluster_centers_, 'euclidean')

# Demonstrate K-means step by step
def demonstrate_kmeans_steps():
    """Visualize K-means algorithm steps"""
    np.random.seed(42)
    
    # Generate sample data
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                          cluster_std=0.60, random_state=42)
    
    # Initialize K-means
    kmeans = KMeansFromScratch(n_clusters=3, init='random', max_iters=1, random_state=42)
    
    # Store intermediate results
    steps = []
    
    # Initial state
    kmeans.cluster_centers_ = kmeans._initialize_centers(X)
    initial_centers = kmeans.cluster_centers_.copy()
    steps.append({
        'centers': initial_centers.copy(),
        'labels': None,
        'title': 'Initial Centers'
    })
    
    # Run algorithm step by step
    for i in range(6):
        # Assignment step
        labels = kmeans._assign_clusters(X, kmeans.cluster_centers_)
        steps.append({
            'centers': kmeans.cluster_centers_.copy(),
            'labels': labels.copy(),
            'title': f'Step {i+1}: Assignment'
        })
        
        # Update step
        kmeans.cluster_centers_ = kmeans._update_centers(X, labels)
        steps.append({
            'centers': kmeans.cluster_centers_.copy(),
            'labels': labels.copy(),
            'title': f'Step {i+1}: Update Centers'
        })
        
        # Check convergence
        if i > 0 and np.array_equal(labels, steps[-4]['labels']):
            break
    
    # Visualize steps
    n_steps = len(steps)
    n_cols = 4
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.ravel()
    
    for idx, step in enumerate(steps):
        ax = axes[idx]
        
        # Plot points
        if step['labels'] is not None:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=step['labels'], 
                               cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        else:
            ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, 
                      edgecolors='k', linewidth=0.5)
        
        # Plot centers
        ax.scatter(step['centers'][:, 0], step['centers'][:, 1], 
                  c='red', s=200, marker='*', edgecolors='k', linewidth=2)
        
        # Add center movements
        if idx > 0 and idx % 2 == 0:  # Update steps
            prev_centers = steps[idx-1]['centers']
            for j in range(len(step['centers'])):
                ax.arrow(prev_centers[j, 0], prev_centers[j, 1],
                        step['centers'][j, 0] - prev_centers[j, 0],
                        step['centers'][j, 1] - prev_centers[j, 1],
                        head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
        
        ax.set_title(step['title'])
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('K-Means Algorithm Steps', fontsize=16)
    plt.tight_layout()
    plt.show()

# Compare initialization methods
def compare_initialization_methods():
    """Compare different initialization strategies"""
    np.random.seed(42)
    
    # Generate challenging data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                          cluster_std=[0.5, 1.0, 0.5, 1.0], random_state=42)
    
    # Try different initializations
    init_methods = ['random', 'k-means++']
    n_runs = 10
    
    results = {method: {'inertias': [], 'best_kmeans': None, 'best_inertia': float('inf')} 
              for method in init_methods}
    
    for method in init_methods:
        for run in range(n_runs):
            kmeans = KMeansFromScratch(n_clusters=4, init=method, 
                                     random_state=run, max_iters=100)
            kmeans.fit(X)
            
            results[method]['inertias'].append(kmeans.inertia_)
            
            if kmeans.inertia_ < results[method]['best_inertia']:
                results[method]['best_inertia'] = kmeans.inertia_
                results[method]['best_kmeans'] = kmeans
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot of inertias
    ax = axes[0]
    ax.boxplot([results[method]['inertias'] for method in init_methods],
               labels=init_methods)
    ax.set_ylabel('Inertia (WCSS)')
    ax.set_title('Initialization Method Comparison')
    
    # Best result for each method
    for idx, method in enumerate(init_methods):
        ax = axes[idx + 1]
        kmeans = results[method]['best_kmeans']
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, 
                           cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  c='red', s=200, marker='*', edgecolors='k', linewidth=2)
        
        ax.set_title(f'{method} (Inertia: {kmeans.inertia_:.2f})')
        ax.set_aspect('equal')
    
    plt.suptitle('K-Means Initialization Methods', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    for method in init_methods:
        inertias = results[method]['inertias']
        print(f"\n{method}:")
        print(f"  Mean inertia: {np.mean(inertias):.2f}")
        print(f"  Std inertia: {np.std(inertias):.2f}")
        print(f"  Best inertia: {np.min(inertias):.2f}")
        print(f"  Worst inertia: {np.max(inertias):.2f}")

# Demonstrate K-means properties
def demonstrate_kmeans_properties():
    """Show various properties and limitations of K-means"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Works well on spherical clusters
    ax = axes[0, 0]
    X1, y1 = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    kmeans1 = KMeansFromScratch(n_clusters=3, init='k-means++', random_state=42)
    labels1 = kmeans1.fit_predict(X1)
    ax.scatter(X1[:, 0], X1[:, 1], c=labels1, cmap='viridis', alpha=0.6)
    ax.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='k', linewidth=2)
    ax.set_title('Works Well: Spherical Clusters')
    
    # 2. Struggles with elongated clusters
    ax = axes[0, 1]
    X2, y2 = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X2 = np.dot(X2, transformation)
    kmeans2 = KMeansFromScratch(n_clusters=3, init='k-means++', random_state=42)
    labels2 = kmeans2.fit_predict(X2)
    ax.scatter(X2[:, 0], X2[:, 1], c=labels2, cmap='viridis', alpha=0.6)
    ax.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='k', linewidth=2)
    ax.set_title('Struggles: Elongated Clusters')
    
    # 3. Fails on concentric circles
    ax = axes[0, 2]
    from sklearn.datasets import make_circles
    X3, y3 = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
    kmeans3 = KMeansFromScratch(n_clusters=2, init='k-means++', random_state=42)
    labels3 = kmeans3.fit_predict(X3)
    ax.scatter(X3[:, 0], X3[:, 1], c=labels3, cmap='viridis', alpha=0.6)
    ax.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='k', linewidth=2)
    ax.set_title('Fails: Concentric Circles')
    
    # 4. Sensitive to scale
    ax = axes[1, 0]
    X4 = np.random.randn(300, 2)
    X4[:100, 0] += 2
    X4[100:200, 0] -= 2
    X4[200:, 1] += 2
    X4[:, 1] *= 0.1  # Scale down y-axis
    kmeans4 = KMeansFromScratch(n_clusters=3, init='k-means++', random_state=42)
    labels4 = kmeans4.fit_predict(X4)
    ax.scatter(X4[:, 0], X4[:, 1], c=labels4, cmap='viridis', alpha=0.6)
    ax.scatter(kmeans4.cluster_centers_[:, 0], kmeans4.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='k', linewidth=2)
    ax.set_title('Issue: Scale Sensitivity')
    
    # 5. Different density clusters
    ax = axes[1, 1]
    X5_1, _ = make_blobs(n_samples=100, centers=[[2, 2]], cluster_std=0.3, random_state=42)
    X5_2, _ = make_blobs(n_samples=200, centers=[[-2, -2]], cluster_std=1.0, random_state=42)
    X5 = np.vstack([X5_1, X5_2])
    kmeans5 = KMeansFromScratch(n_clusters=2, init='k-means++', random_state=42)
    labels5 = kmeans5.fit_predict(X5)
    ax.scatter(X5[:, 0], X5[:, 1], c=labels5, cmap='viridis', alpha=0.6)
    ax.scatter(kmeans5.cluster_centers_[:, 0], kmeans5.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='k', linewidth=2)
    ax.set_title('Issue: Different Densities')
    
    # 6. Different sized clusters
    ax = axes[1, 2]
    X6_1, _ = make_blobs(n_samples=50, centers=[[2, 2]], cluster_std=0.5, random_state=42)
    X6_2, _ = make_blobs(n_samples=250, centers=[[-2, -2]], cluster_std=0.5, random_state=42)
    X6 = np.vstack([X6_1, X6_2])
    kmeans6 = KMeansFromScratch(n_clusters=2, init='k-means++', random_state=42)
    labels6 = kmeans6.fit_predict(X6)
    ax.scatter(X6[:, 0], X6[:, 1], c=labels6, cmap='viridis', alpha=0.6)
    ax.scatter(kmeans6.cluster_centers_[:, 0], kmeans6.cluster_centers_[:, 1],
              c='red', s=200, marker='*', edgecolors='k', linewidth=2)
    ax.set_title('Issue: Different Sizes')
    
    plt.suptitle('K-Means Properties and Limitations', fontsize=16)
    plt.tight_layout()
    plt.show()
```

## 6. Using Scikit-learn

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import seaborn as sns

# Comprehensive K-means workflow
class KMeansWorkflow:
    def __init__(self, n_clusters=3, scale=True):
        self.n_clusters = n_clusters
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.kmeans = None
        self.metrics = {}
        
    def fit_predict(self, X, **kwargs):
        """Fit K-means and return predictions with analysis"""
        # Scale if needed
        if self.scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, **kwargs)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        self.metrics = {
            'inertia': self.kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels)
        }
        
        return labels
    
    def analyze_clusters(self, X, labels, feature_names=None):
        """Analyze cluster characteristics"""
        X_df = pd.DataFrame(X, columns=feature_names if feature_names else 
                           [f'Feature_{i}' for i in range(X.shape[1])])
        X_df['Cluster'] = labels
        
        # Cluster sizes
        cluster_sizes = X_df['Cluster'].value_counts().sort_index()
        print("Cluster Sizes:")
        print(cluster_sizes)
        print(f"\nSize ratio (max/min): {cluster_sizes.max() / cluster_sizes.min():.2f}")
        
        # Cluster statistics
        print("\nCluster Centers:")
        centers_df = pd.DataFrame(self.kmeans.cluster_centers_, 
                                columns=X_df.columns[:-1])
        centers_df.index = [f'Cluster {i}' for i in range(self.n_clusters)]
        print(centers_df)
        
        # Feature means by cluster
        print("\nFeature Means by Cluster:")
        cluster_means = X_df.groupby('Cluster').mean()
        print(cluster_means)
        
        return X_df, cluster_means
    
    def visualize_clusters(self, X, labels, feature_names=None):
        """Comprehensive cluster visualization"""
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            X_2d = X
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster plot
        ax = axes[0, 0]
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                           cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Plot centers
        if X.shape[1] > 2:
            centers_2d = pca.transform(self.kmeans.cluster_centers_)
        else:
            centers_2d = self.kmeans.cluster_centers_
        
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, 
                  marker='*', edgecolors='k', linewidth=2, label='Centers')
        
        ax.set_xlabel('Component 1' if X.shape[1] > 2 else 'Feature 1')
        ax.set_ylabel('Component 2' if X.shape[1] > 2 else 'Feature 2')
        ax.set_title('K-Means Clustering Results')
        ax.legend()
        
        # 2. Silhouette plot
        ax = axes[0, 1]
        from sklearn.metrics import silhouette_samples
        
        silhouette_vals = silhouette_samples(X, labels)
        y_lower = 10
        
        for i in range(self.n_clusters):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / self.n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, cluster_silhouette_vals,
                           facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title("Silhouette Plot")
        ax.axvline(x=self.metrics['silhouette'], color="red", linestyle="--",
                  label=f'Average: {self.metrics["silhouette"]:.3f}')
        ax.legend()
        
        # 3. Distance to nearest cluster
        ax = axes[1, 0]
        distances = self.kmeans.transform(X)
        min_distances = np.min(distances, axis=1)
        
        ax.hist(min_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Distance to Nearest Cluster Center')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Distances to Cluster Centers')
        ax.axvline(x=np.mean(min_distances), color='red', linestyle='--',
                  label=f'Mean: {np.mean(min_distances):.3f}')
        ax.legend()
        
        # 4. Cluster quality metrics
        ax = axes[1, 1]
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title('Clustering Quality Metrics')
        ax.set_xticklabels(['Metrics'], rotation=0)
        ax.legend(loc='best')
        
        plt.suptitle(f'K-Means Analysis (k={self.n_clusters})', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig

# Example: Customer Segmentation
def customer_segmentation_example():
    """Complete customer segmentation pipeline"""
    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Create customer features
    customer_data = {
        'annual_spending': np.concatenate([
            np.random.normal(30000, 10000, 300),  # Low spenders
            np.random.normal(60000, 15000, 400),  # Medium spenders
            np.random.normal(100000, 20000, 300)  # High spenders
        ]),
        'frequency': np.concatenate([
            np.random.poisson(5, 300),   # Infrequent
            np.random.poisson(15, 400),  # Regular
            np.random.poisson(30, 300)   # Frequent
        ]),
        'recency_days': np.concatenate([
            np.random.exponential(60, 300),  # Not recent
            np.random.exponential(20, 400),  # Somewhat recent
            np.random.exponential(5, 300)    # Very recent
        ]),
        'avg_basket_size': np.concatenate([
            np.random.gamma(2, 50, 300),   # Small baskets
            np.random.gamma(4, 75, 400),   # Medium baskets
            np.random.gamma(6, 100, 300)   # Large baskets
        ])
    }
    
    # Create DataFrame
    df = pd.DataFrame(customer_data)
    
    # Add some correlation
    df['total_value'] = df['annual_spending'] * (1 + 0.1 * np.random.randn(n_customers))
    
    print("Customer Data Summary:")
    print(df.describe())
    
    # Apply K-means workflow
    workflow = KMeansWorkflow(n_clusters=3, scale=True)
    
    # Fit and predict
    X = df.values
    feature_names = df.columns.tolist()
    labels = workflow.fit_predict(X, init='k-means++', n_init=10, random_state=42)
    
    # Analyze clusters
    df_clustered, cluster_means = workflow.analyze_clusters(X, labels, feature_names)
    
    # Visualize
    workflow.visualize_clusters(X, labels, feature_names)
    
    # Business interpretation
    print("\n=== Customer Segment Interpretation ===")
    for i in range(3):
        print(f"\nSegment {i}:")
        segment_means = cluster_means.iloc[i]
        
        # Determine segment characteristics
        if segment_means['annual_spending'] > 80000:
            spending_level = "High"
        elif segment_means['annual_spending'] > 45000:
            spending_level = "Medium"
        else:
            spending_level = "Low"
        
        if segment_means['frequency'] > 20:
            frequency_level = "Frequent"
        elif segment_means['frequency'] > 10:
            frequency_level = "Regular"
        else:
            frequency_level = "Infrequent"
        
        if segment_means['recency_days'] < 15:
            recency_level = "Very Recent"
        elif segment_means['recency_days'] < 40:
            recency_level = "Recent"
        else:
            recency_level = "Not Recent"
        
        print(f"  - Spending: {spending_level} (${segment_means['annual_spending']:,.0f})")
        print(f"  - Frequency: {frequency_level} ({segment_means['frequency']:.0f} purchases)")
        print(f"  - Recency: {recency_level} ({segment_means['recency_days']:.0f} days)")
        print(f"  - Avg Basket: ${segment_means['avg_basket_size']:,.0f}")
        
        # Suggest strategies
        if spending_level == "High" and frequency_level == "Frequent":
            print("  - Strategy: VIP treatment, exclusive offers")
        elif spending_level == "Medium" and recency_level == "Very Recent":
            print("  - Strategy: Upsell opportunities, loyalty programs")
        elif frequency_level == "Infrequent" or recency_level == "Not Recent":
            print("  - Strategy: Re-engagement campaigns, win-back offers")
    
    return df_clustered, workflow

# Mini-batch K-means for large datasets
def minibatch_kmeans_example():
    """Demonstrate Mini-batch K-means for large datasets"""
    # Generate large dataset
    n_samples = 100000
    n_features = 50
    n_clusters = 10
    
    print(f"Generating large dataset: {n_samples:,} samples, {n_features} features")
    X_large, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                           centers=n_clusters, random_state=42)
    
    # Time comparison
    import time
    
    # Standard K-means
    print("\nStandard K-means:")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_large)
    kmeans_time = time.time() - start_time
    kmeans_inertia = kmeans.inertia_
    
    # Mini-batch K-means
    print("\nMini-batch K-means:")
    batch_sizes = [100, 500, 1000, 5000]
    results = []
    
    for batch_size in batch_sizes:
        start_time = time.time()
        mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                  init='k-means++', n_init=10, random_state=42)
        mbkmeans.fit(X_large)
        mb_time = time.time() - start_time
        
        results.append({
            'batch_size': batch_size,
            'time': mb_time,
            'inertia': mbkmeans.inertia_,
            'speedup': kmeans_time / mb_time
        })
        
        print(f"  Batch size {batch_size}: {mb_time:.2f}s (speedup: {kmeans_time/mb_time:.1f}x)")
    
    # Compare results
    results_df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax1.bar(['K-means'] + [f'MB-{bs}' for bs in batch_sizes],
           [kmeans_time] + results_df['time'].tolist())
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Quality comparison
    ax2.bar(['K-means'] + [f'MB-{bs}' for bs in batch_sizes],
           [kmeans_inertia] + results_df['inertia'].tolist())
    ax2.set_ylabel('Inertia (WCSS)')
    ax2.set_title('Clustering Quality Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Mini-batch K-means Performance', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return results_df
```

## 7. Choosing the Optimal K

### 7.1 Methods for Selecting K

```python
class OptimalKSelection:
    """Methods for selecting optimal number of clusters"""
    
    @staticmethod
    def elbow_method(X, k_range=range(2, 11)):
        """Elbow method for selecting k"""
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Calculate elbow point
        # Using the "knee" point detection
        from kneed import KneeLocator
        kn = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        elbow_k = kn.elbow
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.axvline(x=elbow_k, color='r', linestyle='--', 
                   label=f'Elbow at k={elbow_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method for Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return elbow_k, inertias
    
    @staticmethod
    def silhouette_method(X, k_range=range(2, 11)):
        """Silhouette method for selecting k"""
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                   label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method for Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return optimal_k, silhouette_scores
    
    @staticmethod
    def gap_statistic(X, k_range=range(1, 11), n_refs=10):
        """Gap statistic method for selecting k"""
        gaps = []
        stds = []
        
        for k in k_range:
            # Cluster original data
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            inertia = kmeans.inertia_
            
            # Generate reference datasets
            ref_inertias = []
            for _ in range(n_refs):
                # Random data within bounds of original
                X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), 
                                        size=X.shape)
                kmeans_ref = KMeans(n_clusters=k, init='k-means++', 
                                   n_init=1, random_state=42)
                kmeans_ref.fit(X_ref)
                ref_inertias.append(kmeans_ref.inertia_)
            
            # Calculate gap
            gap = np.log(np.mean(ref_inertias)) - np.log(inertia)
            gaps.append(gap)
            stds.append(np.std(np.log(ref_inertias)))
        
        # Find optimal k (first k where gap[k] >= gap[k+1] - std[k+1])
        gaps = np.array(gaps)
        stds = np.array(stds)
        
        optimal_k = 1
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - stds[i + 1]:
                optimal_k = k_range[i]
                break
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(k_range, gaps, yerr=stds, fmt='bo-', capsize=5)
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                   label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic Method for Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return optimal_k, gaps, stds
    
    @staticmethod
    def davies_bouldin_method(X, k_range=range(2, 11)):
        """Davies-Bouldin index for selecting k (lower is better)"""
        db_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            score = davies_bouldin_score(X, labels)
            db_scores.append(score)
        
        # Find optimal k (minimum score)
        optimal_k = k_range[np.argmin(db_scores)]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, db_scores, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                   label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Davies-Bouldin Score (lower is better)')
        plt.title('Davies-Bouldin Method for Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return optimal_k, db_scores
    
    @staticmethod
    def compare_all_methods(X, k_range=range(2, 11)):
        """Compare all methods for selecting k"""
        selector = OptimalKSelection()
        
        # Apply all methods
        elbow_k, inertias = selector.elbow_method(X, k_range)
        silhouette_k, silhouette_scores = selector.silhouette_method(X, k_range)
        gap_k, gaps, _ = selector.gap_statistic(X, k_range)
        db_k, db_scores = selector.davies_bouldin_method(X, k_range)
        
        # Summary
        print("Optimal k Selection Summary:")
        print(f"  Elbow Method: k = {elbow_k}")
        print(f"  Silhouette Method: k = {silhouette_k}")
        print(f"  Gap Statistic: k = {gap_k}")
        print(f"  Davies-Bouldin: k = {db_k}")
        
        # Consensus
        all_k = [elbow_k, silhouette_k, gap_k, db_k]
        consensus_k = max(set(all_k), key=all_k.count)
        print(f"\nConsensus: k = {consensus_k}")
        
        return {
            'elbow': elbow_k,
            'silhouette': silhouette_k,
            'gap': gap_k,
            'davies_bouldin': db_k,
            'consensus': consensus_k
        }

# Demonstrate optimal k selection
def demonstrate_optimal_k():
    """Show how to find optimal k for different datasets"""
    datasets = {
        'Well-separated': make_blobs(n_samples=300, centers=4, n_features=2, 
                                   cluster_std=0.5, random_state=42),
        'Overlapping': make_blobs(n_samples=300, centers=4, n_features=2,
                                cluster_std=1.5, random_state=42),
        'Different sizes': None  # Will create custom
    }
    
    # Create custom dataset for different sizes
    X1, _ = make_blobs(n_samples=50, centers=[[0, 0]], cluster_std=0.5)
    X2, _ = make_blobs(n_samples=150, centers=[[5, 5]], cluster_std=0.5)
    X3, _ = make_blobs(n_samples=100, centers=[[-5, 5]], cluster_std=0.5)
    datasets['Different sizes'] = (np.vstack([X1, X2, X3]), None)
    
    selector = OptimalKSelection()
    
    for name, (X, _) in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print('='*50)
        
        # Standardize
        X_scaled = StandardScaler().fit_transform(X)
        
        # Find optimal k
        results = selector.compare_all_methods(X_scaled, range(2, 8))
        
        # Visualize with optimal k
        optimal_k = results['consensus']
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', 
                       n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                            alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.scatter(kmeans.cluster_centers_[:, 0] * np.std(X[:, 0]) + np.mean(X[:, 0]),
                   kmeans.cluster_centers_[:, 1] * np.std(X[:, 1]) + np.mean(X[:, 1]),
                   c='red', s=200, marker='*', edgecolors='k', linewidth=2)
        plt.title(f'{name} - Optimal k={optimal_k}')
        plt.colorbar(scatter)
        plt.show()
```

## 8. K-Means Variants and Improvements

### 8.1 K-Means++

Already implemented in the scratch implementation above.

### 8.2 K-Medoids (PAM)

```python
def k_medoids(X, n_clusters, max_iter=100):
    """K-medoids clustering (simplified implementation)"""
    n_samples = X.shape[0]
    
    # Initialize medoids randomly
    medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    medoids = X[medoid_indices].copy()
    
    for iteration in range(max_iter):
        # Assignment step
        distances = cdist(X, medoids)
        labels = np.argmin(distances, axis=1)
        
        # Update step (find best medoid for each cluster)
        new_medoid_indices = []
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                # Find point that minimizes sum of distances to all other points
                cluster_distances = cdist(cluster_points, cluster_points)
                total_distances = cluster_distances.sum(axis=1)
                best_medoid = np.argmin(total_distances)
                # Find index in original data
                cluster_indices = np.where(labels == k)[0]
                new_medoid_indices.append(cluster_indices[best_medoid])
            else:
                # Keep old medoid if cluster is empty
                new_medoid_indices.append(medoid_indices[k])
        
        # Check convergence
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        
        medoid_indices = new_medoid_indices
        medoids = X[medoid_indices]
    
    return labels, medoid_indices

### 8.3 Fuzzy C-Means

from sklearn.base import BaseEstimator, ClusterMixin

class FuzzyCMeans(BaseEstimator, ClusterMixin):
    """Fuzzy C-Means clustering"""
    
    def __init__(self, n_clusters=3, fuzziness=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = fuzziness  # Fuzziness parameter (m > 1)
        self.max_iter = max_iter
        self.tol = tol
        self.centers_ = None
        self.membership_ = None
        
    def fit(self, X):
        """Fit Fuzzy C-Means"""
        n_samples = X.shape[0]
        
        # Initialize membership matrix randomly
        self.membership_ = np.random.rand(n_samples, self.n_clusters)
        self.membership_ /= self.membership_.sum(axis=1, keepdims=True)
        
        for iteration in range(self.max_iter):
            # Update centers
            centers = []
            for j in range(self.n_clusters):
                numerator = np.sum((self.membership_[:, j:j+1] ** self.m) * X, axis=0)
                denominator = np.sum(self.membership_[:, j] ** self.m)
                centers.append(numerator / denominator)
            self.centers_ = np.array(centers)
            
            # Update membership
            distances = cdist(X, self.centers_)
            new_membership = np.zeros_like(self.membership_)
            
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    denominator = 0
                    for k in range(self.n_clusters):
                        if distances[i, k] > 0:
                            denominator += (distances[i, j] / distances[i, k]) ** (2 / (self.m - 1))
                        else:
                            # Handle zero distance
                            new_membership[i, k] = 1 if k == j else 0
                            denominator = 1
                            break
                    if denominator > 0:
                        new_membership[i, j] = 1 / denominator
            
            # Check convergence
            if np.max(np.abs(new_membership - self.membership_)) < self.tol:
                break
            
            self.membership_ = new_membership
        
        return self
    
    def predict(self, X):
        """Predict crisp clusters"""
        membership = self.predict_proba(X)
        return np.argmax(membership, axis=1)
    
    def predict_proba(self, X):
        """Predict membership probabilities"""
        distances = cdist(X, self.centers_)
        membership = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                denominator = 0
                for k in range(self.n_clusters):
                    if distances[i, k] > 0:
                        denominator += (distances[i, j] / distances[i, k]) ** (2 / (self.m - 1))
                if denominator > 0:
                    membership[i, j] = 1 / denominator
        
        return membership

### 8.4 Bisecting K-Means

class BisectingKMeans:
    """Hierarchical divisive clustering using K-means"""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_hierarchy_ = []
        
    def fit(self, X):
        """Fit Bisecting K-means"""
        n_samples = X.shape[0]
        
        # Start with all points in one cluster
        self.labels_ = np.zeros(n_samples, dtype=int)
        clusters = {0: np.arange(n_samples)}
        
        # Iteratively bisect clusters
        while len(clusters) < self.n_clusters:
            # Find cluster to split (largest or highest SSE)
            cluster_to_split = max(clusters.keys(), 
                                 key=lambda k: len(clusters[k]))
            
            if len(clusters[cluster_to_split]) <= 1:
                break
            
            # Get points in cluster
            points_indices = clusters[cluster_to_split]
            points = X[points_indices]
            
            # Bisect using K-means with k=2
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            sub_labels = kmeans.fit_predict(points)
            
            # Update labels and clusters
            new_cluster_id = max(clusters.keys()) + 1
            
            # Points assigned to sub-cluster 0 stay in original cluster
            # Points assigned to sub-cluster 1 go to new cluster
            for i, point_idx in enumerate(points_indices):
                if sub_labels[i] == 1:
                    self.labels_[point_idx] = new_cluster_id
            
            # Update cluster dictionary
            cluster0_indices = points_indices[sub_labels == 0]
            cluster1_indices = points_indices[sub_labels == 1]
            
            clusters[cluster_to_split] = cluster0_indices
            clusters[new_cluster_id] = cluster1_indices
            
            # Record hierarchy
            self.cluster_hierarchy_.append({
                'parent': cluster_to_split,
                'children': [cluster_to_split, new_cluster_id],
                'iteration': len(self.cluster_hierarchy_)
            })
        
        # Relabel clusters to be consecutive
        unique_labels = np.unique(self.labels_)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels_ = np.array([label_map[label] for label in self.labels_])
        
        return self
    
    def predict(self, X):
        """Predict clusters for new data"""
        # Simple approach: assign to nearest cluster center
        centers = []
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                centers.append(cluster_points.mean(axis=0))
        
        centers = np.array(centers)
        distances = cdist(X, centers)
        return np.argmin(distances, axis=1)

# Compare K-means variants
def compare_kmeans_variants():
    """Compare different K-means variants"""
    # Generate test data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                          cluster_std=0.7, random_state=42)
    
    # Algorithms to compare
    algorithms = {
        'K-Means': KMeans(n_clusters=4, init='k-means++', random_state=42),
        'K-Means (random init)': KMeans(n_clusters=4, init='random', random_state=42),
        'MiniBatch K-Means': MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42),
        'Fuzzy C-Means': FuzzyCMeans(n_clusters=4, fuzziness=2),
        'Bisecting K-Means': BisectingKMeans(n_clusters=4)
    }
    
    # Fit and evaluate
    results = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (name, algorithm) in enumerate(algorithms.items()):
        # Fit
        import time
        start_time = time.time()
        
        if name == 'K-Medoids':
            labels, medoid_indices = algorithm(X, n_clusters=4)
        else:
            algorithm.fit(X)
            labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(X)
        
        fit_time = time.time() - start_time
        
        # Evaluate
        silhouette = silhouette_score(X, labels)
        
        results.append({
            'Algorithm': name,
            'Silhouette Score': silhouette,
            'Fit Time': fit_time
        })
        
        # Plot
        ax = axes[idx]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                           alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Plot centers if available
        if hasattr(algorithm, 'cluster_centers_'):
            ax.scatter(algorithm.cluster_centers_[:, 0], 
                      algorithm.cluster_centers_[:, 1],
                      c='red', s=200, marker='*', edgecolors='k', linewidth=2)
        elif hasattr(algorithm, 'centers_'):
            ax.scatter(algorithm.centers_[:, 0], algorithm.centers_[:, 1],
                      c='red', s=200, marker='*', edgecolors='k', linewidth=2)
        
        ax.set_title(f'{name}\nSilhouette: {silhouette:.3f}')
        ax.set_aspect('equal')
    
    # Hide extra subplot
    axes[-1].set_visible(False)
    
    plt.suptitle('K-Means Variants Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nAlgorithm Comparison:")
    print(results_df.to_string(index=False))
    
    return results_df
```

## 9. Applications and Case Studies

### 9.1 Image Compression

```python
def image_compression_kmeans():
    """Use K-means for image compression"""
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Load sample image (or use local image)
    try:
        # Try to load a sample image
        response = requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg')
        img = Image.open(BytesIO(response.content))
    except:
        # Create synthetic image if download fails
        img = Image.new('RGB', (400, 300))
        pixels = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixels[i, j] = (
                    int(255 * np.sin(i/20) ** 2),
                    int(255 * np.sin(j/20) ** 2),
                    int(255 * np.sin((i+j)/20) ** 2)
                )
    
    # Convert to array
    img_array = np.array(img)
    original_shape = img_array.shape
    
    # Reshape to pixels x channels
    pixels = img_array.reshape(-1, 3)
    
    # Apply K-means with different k values
    k_values = [4, 8, 16, 32, 64]
    compressed_images = []
    
    for k in k_values:
        print(f"Compressing with k={k} colors...")
        
        # Apply K-means
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Replace pixels with cluster centers
        compressed_pixels = kmeans.cluster_centers_[labels]
        compressed_image = compressed_pixels.reshape(original_shape).astype(np.uint8)
        compressed_images.append(compressed_image)
        
        # Calculate compression ratio
        original_colors = len(np.unique(pixels.reshape(-1, pixels.shape[1]), axis=0))
        compression_ratio = original_colors / k
        print(f"  Original colors: {original_colors}, Compression ratio: {compression_ratio:.1f}:1")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title(f'Original Image\n{len(np.unique(pixels, axis=0))} colors')
    axes[0].axis('off')
    
    # Compressed images
    for idx, (k, compressed) in enumerate(zip(k_values, compressed_images)):
        axes[idx + 1].imshow(compressed)
        axes[idx + 1].set_title(f'K-Means Compressed\nk={k} colors')
        axes[idx + 1].axis('off')
    
    plt.suptitle('Image Compression using K-Means', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Show file size comparison (approximate)
    print("\nApproximate file sizes:")
    original_size = img_array.nbytes
    print(f"Original: {original_size / 1024:.1f} KB")
    
    for k in k_values:
        # Compressed size = cluster centers + labels
        centers_size = k * 3 * 4  # k centers, 3 channels, 4 bytes per float
        labels_size = pixels.shape[0] * np.ceil(np.log2(k) / 8)  # bits per label
        compressed_size = centers_size + labels_size
        print(f"k={k}: {compressed_size / 1024:.1f} KB "
              f"(compression ratio: {original_size / compressed_size:.1f}:1)")

### 9.2 Document Clustering

def document_clustering_example():
    """Cluster documents using K-means"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample documents (in practice, load from files)
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "The dog chased the cat around the garden",
        "Supervised learning requires labeled training data",
        "Unsupervised learning finds patterns without labels",
        "The cat sat on the mat in the sunny garden",
        "Reinforcement learning uses rewards and penalties",
        "Computer vision enables machines to interpret images"
    ]
    
    # More documents for better clustering
    documents.extend([
        "The brown cat jumped over the fence",
        "Dogs and cats are common household pets",
        "Machine learning algorithms can classify images",
        "Neural networks are inspired by biological brains",
        "Clustering is an unsupervised learning technique",
        "Classification is a supervised learning task",
        "The quick cat ran through the garden",
        "Artificial intelligence is transforming technology",
        "Deep neural networks can learn complex patterns",
        "Natural language processing powers chatbots"
    ])
    
    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(documents)
    
    # Apply K-means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                   n_init=10, random_state=42)
    labels = kmeans.fit_predict(X.toarray())
    
    # Analyze clusters
    print("Document Clusters:")
    print("=" * 50)
    
    for cluster_id in range(n_clusters):
        print(f"\nCluster {cluster_id}:")
        cluster_docs = [doc for doc, label in zip(documents, labels) 
                       if label == cluster_id]
        
        # Show sample documents
        print("Sample documents:")
        for doc in cluster_docs[:3]:
            print(f"  - {doc}")
        
        # Find top terms for cluster
        cluster_center = kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"Top terms: {', '.join(top_terms)}")
    
    # Visualize using PCA
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    
    for cluster_id in range(n_clusters):
        cluster_points = X_pca[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[cluster_id], label=f'Cluster {cluster_id}',
                   alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Plot cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='black', s=200, marker='*', edgecolors='white', linewidth=2)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Document Clusters Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return documents, labels, vectorizer

### 9.3 Anomaly Detection with K-Means

def anomaly_detection_kmeans():
    """Use K-means for anomaly detection"""
    # Generate normal data with some anomalies
    np.random.seed(42)
    
    # Normal data (3 clusters)
    normal_data = []
    centers = [[0, 0], [5, 5], [-5, 5]]
    
    for center in centers:
        cluster = np.random.randn(100, 2) + center
        normal_data.append(cluster)
    
    normal_data = np.vstack(normal_data)
    
    # Add anomalies
    n_anomalies = 20
    anomalies = np.random.uniform(-10, 10, (n_anomalies, 2))
    
    # Combine data
    X = np.vstack([normal_data, anomalies])
    true_labels = np.hstack([np.zeros(len(normal_data)), 
                           np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    true_labels = true_labels[indices]
    
    # Apply K-means
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    
    # Calculate distances to nearest cluster center
    distances = kmeans.transform(X)
    min_distances = np.min(distances, axis=1)
    
    # Determine threshold (e.g., 95th percentile)
    threshold_percentile = 95
    threshold = np.percentile(min_distances, threshold_percentile)
    
    # Detect anomalies
    predicted_anomalies = (min_distances > threshold).astype(int)
    
    # Evaluate
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_anomalies, average='binary'
    )
    auc = roc_auc_score(true_labels, min_distances)
    
    print("Anomaly Detection Results:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True labels
    ax1.scatter(X[true_labels == 0, 0], X[true_labels == 0, 1], 
               c='blue', label='Normal', alpha=0.6)
    ax1.scatter(X[true_labels == 1, 0], X[true_labels == 1, 1], 
               c='red', label='Anomaly', alpha=0.6, s=100, marker='x')
    ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='green', s=200, marker='*', edgecolors='k', linewidth=2,
               label='Cluster Centers')
    ax1.set_title('True Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Detected anomalies
    ax2.scatter(X[predicted_anomalies == 0, 0], X[predicted_anomalies == 0, 1],
               c='blue', label='Normal', alpha=0.6)
    ax2.scatter(X[predicted_anomalies == 1, 0], X[predicted_anomalies == 1, 1],
               c='red', label='Detected Anomaly', alpha=0.6, s=100, marker='x')
    
    # Draw threshold circle around each center
    for center in kmeans.cluster_centers_:
        circle = plt.Circle(center, threshold, fill=False, color='red', 
                          linestyle='--', linewidth=2)
        ax2.add_patch(circle)
    
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='green', s=200, marker='*', edgecolors='k', linewidth=2,
               label='Cluster Centers')
    ax2.set_title(f'Detected Anomalies (threshold={threshold:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle('K-Means for Anomaly Detection', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(min_distances[true_labels == 0], bins=30, alpha=0.5, 
            label='Normal', density=True)
    plt.hist(min_distances[true_labels == 1], bins=30, alpha=0.5, 
            label='Anomaly', density=True)
    plt.axvline(x=threshold, color='red', linestyle='--', 
               label=f'Threshold ({threshold_percentile}th percentile)')
    plt.xlabel('Distance to Nearest Cluster Center')
    plt.ylabel('Density')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## 10. Interview Questions

### Q1: Explain K-means clustering in simple terms.
**Answer**: K-means groups similar data points together into k clusters. It works like organizing people at a party into k groups based on where they're standing - each person joins the group whose center is closest to them. The algorithm repeatedly adjusts the group centers and reassigns people until the groups stabilize.

### Q2: What are the assumptions of K-means clustering?
**Answer**:
1. **Spherical clusters**: Assumes clusters are roughly spherical/circular
2. **Similar sizes**: Works best when clusters have similar numbers of points
3. **Similar densities**: Assumes similar point density across clusters
4. **Euclidean distance**: Assumes Euclidean distance is meaningful
5. **Continuous features**: Designed for numerical, continuous data
6. **Known k**: Number of clusters must be specified beforehand

### Q3: How does K-means algorithm work step by step?
**Answer**:
1. **Initialize**: Randomly select k points as initial cluster centers
2. **Assign**: Assign each data point to the nearest cluster center
3. **Update**: Calculate new cluster centers as mean of assigned points
4. **Repeat**: Continue steps 2-3 until assignments don't change
5. **Converge**: Algorithm stops when clusters stabilize

The algorithm minimizes within-cluster sum of squares (WCSS).

### Q4: What is the time complexity of K-means?
**Answer**: 
- **Per iteration**: O(n × k × d) where:
  - n = number of data points
  - k = number of clusters  
  - d = number of dimensions
- **Total complexity**: O(n × k × d × i) where i = number of iterations
- **Space complexity**: O(n × d + k × d)

In practice, the number of iterations is often small, making K-means efficient for large datasets.

### Q5: How do you choose the optimal number of clusters (k)?
**Answer**: Several methods:
1. **Elbow method**: Plot WCSS vs k, look for "elbow" where decrease slows
2. **Silhouette analysis**: Maximize average silhouette coefficient
3. **Gap statistic**: Compare with random data, find largest gap
4. **Domain knowledge**: Use business/domain understanding
5. **Davies-Bouldin index**: Minimize ratio of within to between cluster distances
6. **Cross-validation**: If using for downstream task

### Q6: What are the limitations of K-means?
**Answer**:
1. **Fixed k**: Must specify number of clusters beforehand
2. **Spherical assumption**: Fails on non-spherical clusters
3. **Sensitivity to outliers**: Outliers significantly affect centroids
4. **Local minima**: May converge to suboptimal solution
5. **Scale sensitivity**: Features on different scales dominate
6. **Categorical data**: Not designed for categorical features
7. **Equal cluster assumption**: Struggles with very different cluster sizes

### Q7: How do you handle categorical variables in K-means?
**Answer**:
1. **One-hot encoding**: Convert to binary features (increases dimensionality)
2. **K-modes**: Use mode instead of mean, Hamming distance
3. **K-prototypes**: Combine K-means and K-modes for mixed data
4. **Embedding**: Convert categories to numerical embeddings
5. **Gower distance**: Use distance metric that handles mixed types
6. **Separate clustering**: Cluster numerical and categorical separately

### Q8: What is K-means++ and why is it better?
**Answer**: K-means++ is a smart initialization method:
- **Standard K-means**: Random initialization can lead to poor results
- **K-means++**: 
  1. Choose first center randomly
  2. Choose next center with probability proportional to squared distance from nearest center
  3. Repeat until k centers chosen

**Benefits**:
- Better final clusters
- Faster convergence
- Theoretical guarantees on solution quality
- More consistent results

### Q9: Compare K-means with hierarchical clustering.
**Answer**:

| Aspect | K-Means | Hierarchical |
|--------|---------|--------------|
| Approach | Partitioning | Agglomerative/Divisive |
| K required | Yes, beforehand | No, can cut dendrogram |
| Scalability | Good O(nkdi) | Poor O(n²) to O(n³) |
| Cluster shape | Spherical | Any shape |
| Interpretability | Centers meaningful | Dendrogram shows hierarchy |
| Deterministic | No (random init) | Yes |
| Update capability | Can update | Must recompute |

### Q10: How do you evaluate K-means clustering results?
**Answer**:
**Internal metrics** (no ground truth):
- **Inertia/WCSS**: Lower is better (but decreases with k)
- **Silhouette coefficient**: -1 to 1, higher is better
- **Davies-Bouldin index**: Lower is better
- **Calinski-Harabasz index**: Higher is better

**External metrics** (with ground truth):
- **Adjusted Rand Index (ARI)**: Adjusted for chance
- **Normalized Mutual Information (NMI)**: Information theory based
- **Fowlkes-Mallows index**: Geometric mean of precision/recall

**Practical evaluation**:
- Visualization (PCA/t-SNE)
- Domain expert validation
- Downstream task performance

### Q11: What is the difference between K-means and K-medoids?
**Answer**:
- **K-means**: Uses mean of points as center (may not be actual data point)
- **K-medoids**: Uses actual data point as center (medoid)

**K-medoids advantages**:
- More robust to outliers
- Works with any distance metric
- Centers are interpretable (actual data points)

**K-medoids disadvantages**:
- More computationally expensive O(k(n-k)²)
- May not find optimal center for cluster

### Q12: How does Mini-batch K-means work?
**Answer**: Mini-batch K-means uses small random batches to update centers:
1. Sample batch of data (e.g., 100 points)
2. Assign batch points to nearest centers
3. Update centers using batch only
4. Repeat with new batches

**Benefits**:
- Much faster for large datasets
- Can handle streaming data
- Similar quality to standard K-means

**Trade-offs**:
- Slightly worse clusters
- More iterations needed
- Batch size affects quality/speed

### Q13: Can K-means handle non-spherical clusters?
**Answer**: No, K-means struggles with non-spherical clusters because:
- Uses Euclidean distance (assumes spherical)
- Centers may not represent non-spherical shapes well

**Solutions**:
1. **Kernel K-means**: Map to higher dimension where clusters are spherical
2. **Spectral clustering**: Uses eigenvectors, handles non-convex shapes
3. **DBSCAN**: Density-based, finds arbitrary shapes
4. **Gaussian Mixture Models**: Allows elliptical clusters
5. **Transform data**: Sometimes PCA/scaling helps

### Q14: How do you make K-means more robust to outliers?
**Answer**:
1. **Preprocessing**:
   - Remove outliers using IQR or z-score
   - Use robust scaling (median/MAD)
   
2. **Algorithm modifications**:
   - K-medoids instead of K-means
   - Trimmed K-means (ignore furthest points)
   - K-means with outlier detection
   
3. **Post-processing**:
   - Identify points far from all centers
   - Create separate outlier cluster
   
4. **Alternative algorithms**:
   - DBSCAN (has noise concept)
   - Robust clustering algorithms

### Q15: Explain the relationship between K-means and Lloyd's algorithm.
**Answer**: Lloyd's algorithm is the standard algorithm used to solve the K-means problem:
- **K-means**: The problem/objective (minimize WCSS)
- **Lloyd's algorithm**: The iterative solution method

Lloyd's algorithm = the assign-update iteration we typically call "K-means algorithm"

Other algorithms for K-means problem:
- MacQueen's online update
- Hartigan-Wong algorithm
- Genetic algorithms

### Q16: How do you implement K-means for streaming data?
**Answer**:
1. **Sequential K-means**: Update centers with each new point
   ```
   center_new = center_old + α(x - center_old)
   ```

2. **Mini-batch approach**: Collect small batches, update periodically

3. **Sliding window**: Maintain clusters for recent window of data

4. **Adaptive K-means**: Allow k to change, split/merge clusters

5. **Core-sets**: Maintain weighted summary of data

Challenges: Concept drift, memory limitations, real-time constraints

### Q17: What is the mathematical objective function of K-means?
**Answer**: K-means minimizes the Within-Cluster Sum of Squares (WCSS):

```
J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ ||xᵢ - μⱼ||²
```

Where:
- wᵢⱼ = 1 if point i assigned to cluster j, 0 otherwise
- xᵢ = data point i
- μⱼ = center of cluster j
- ||·|| = Euclidean norm

This is equivalent to minimizing variance within clusters.

### Q18: Can K-means be kernelized? How?
**Answer**: Yes, Kernel K-means maps data to higher-dimensional space:

1. **Standard K-means**: Minimize Σᵢ ||xᵢ - μ_c(i)||²

2. **Kernel K-means**: Minimize Σᵢ ||φ(xᵢ) - μ_c(i)||² in feature space

3. **Kernel trick**: Don't compute φ explicitly, use kernel function K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)

**Benefits**:
- Handles non-linear boundaries
- Finds spherical clusters in feature space

**Drawbacks**:
- Computationally expensive
- Must store kernel matrix
- Loses interpretability of centers

### Q19: How do you parallelize K-means?
**Answer**:
1. **Data parallelism**:
   - Partition data across nodes
   - Each node assigns its points to centers
   - Aggregate to compute new centers
   
2. **MapReduce implementation**:
   - Map: Assign points to nearest center
   - Reduce: Compute new centers
   
3. **GPU acceleration**:
   - Distance calculations highly parallel
   - Good for large n, moderate k
   
4. **Distributed frameworks**:
   - Spark MLlib
   - Distributed mini-batch

**Challenges**: Communication overhead, load balancing, synchronization

### Q20: What are soft clustering variants of K-means?
**Answer**:
1. **Fuzzy C-means**: 
   - Points have membership degrees to all clusters
   - Minimizes Σᵢ Σⱼ uᵢⱼᵐ ||xᵢ - cⱼ||²
   - m > 1 controls fuzziness

2. **Gaussian Mixture Models**:
   - Probabilistic cluster assignments
   - Models clusters as Gaussians
   - EM algorithm optimization

3. **Soft K-means**:
   - Uses softmax for assignments
   - Temperature parameter controls hardness

**Benefits**: Captures uncertainty, handles overlapping clusters better

### Q21: How do you determine if K-means is appropriate for your data?
**Answer**: Check these criteria:
1. **Hopkins statistic**: Test clustering tendency (> 0.5 suggests clusters)
2. **Visual inspection**: Plot data, look for natural groups
3. **Feature types**: Ensure numerical, continuous features
4. **Scale**: Check if features are on similar scales
5. **Cluster shape**: Look for roughly spherical patterns
6. **Domain knowledge**: Consider if partitioning makes sense
7. **Try and evaluate**: Run K-means and check if results are meaningful

If not appropriate, consider: DBSCAN, hierarchical clustering, or GMM.

### Q22: What happens if K is greater than the number of distinct data points?
**Answer**: 
- Some clusters will be empty
- Algorithm may crash or handle gracefully depending on implementation
- Common handling:
  1. Reinitialize empty cluster centers
  2. Reduce k automatically
  3. Throw error

**Best practice**: Always check n_unique ≥ k before running K-means

**Related issue**: Even if n > k, can get empty clusters if initialization is poor or data has strong structure.