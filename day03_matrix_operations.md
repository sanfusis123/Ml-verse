# Day 3: Matrix Operations, Eigenvalues & Eigenvectors

## 📚 Topics
- Matrix fundamentals and operations
- Linear transformations
- Eigenvalues and eigenvectors
- Matrix decompositions (SVD, LU, QR)
- Applications in ML

---

## 1. Matrix Fundamentals

### 📖 Core Concepts

#### What is a Matrix?
A matrix is a rectangular array of numbers arranged in rows and columns. In ML, matrices represent:
- Dataset (samples × features)
- Weights in neural networks
- Transformation operations
- Covariance relationships

#### Matrix Notation
- **A ∈ ℝ^(m×n)**: Matrix A with m rows and n columns
- **A_ij**: Element at row i, column j
- **A^T**: Transpose of matrix A
- **A^(-1)**: Inverse of matrix A

### 🔢 Mathematical Foundation

#### Basic Operations

1. **Addition/Subtraction**: Element-wise, same dimensions required
   ```
   C_ij = A_ij ± B_ij
   ```

2. **Scalar Multiplication**: 
   ```
   (kA)_ij = k × A_ij
   ```

3. **Matrix Multiplication**: 
   ```
   C = AB where C_ij = Σ(k=1 to n) A_ik × B_kj
   ```
   Dimensions: (m×n) × (n×p) = (m×p)

4. **Transpose Properties**:
   - (A^T)^T = A
   - (AB)^T = B^T A^T
   - (A + B)^T = A^T + B^T

### 💻 Matrix Operations Code

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, svd, inv, det, matrix_rank, norm
from scipy.linalg import lu, qr, cholesky
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 1. Basic Matrix Operations
print("=== Basic Matrix Operations ===")

# Create matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print(f"Matrix A:\n{A}")
print(f"\nMatrix B:\n{B}")

# Basic operations
print(f"\nA + B:\n{A + B}")
print(f"\nA - B:\n{A - B}")
print(f"\nElement-wise multiplication (Hadamard):\n{A * B}")
print(f"\nMatrix multiplication (A @ B):\n{A @ B}")
print(f"\nMatrix multiplication (B @ A):\n{B @ A}")  # Note: AB ≠ BA

# Transpose
print(f"\nA transpose:\n{A.T}")
print(f"\nVerify (AB)^T = B^T @ A^T:\n{(A @ B).T}\n{B.T @ A.T}")

# 2. Special Matrices
print("\n=== Special Matrices ===")

# Identity matrix
I = np.eye(3)
print(f"Identity matrix:\n{I}")
print(f"A @ I = A: {np.allclose(A @ I, A)}")

# Diagonal matrix
D = np.diag([1, 2, 3])
print(f"\nDiagonal matrix:\n{D}")

# Symmetric matrix
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print(f"\nSymmetric matrix (S = S^T):\n{S}")
print(f"Is symmetric? {np.allclose(S, S.T)}")

# Orthogonal matrix
Q, _ = qr(np.random.randn(3, 3))
print(f"\nOrthogonal matrix Q:\n{Q}")
print(f"Q @ Q^T = I? {np.allclose(Q @ Q.T, I)}")

# 3. Matrix Properties
print("\n=== Matrix Properties ===")

# Rank
print(f"Rank of A: {matrix_rank(A)}")

# Determinant
A_full_rank = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 10]])
print(f"\nDeterminant of A: {det(A):.6f}")  # Near 0 (singular)
print(f"Determinant of modified A: {det(A_full_rank):.6f}")

# Trace
print(f"\nTrace of A (sum of diagonal): {np.trace(A)}")

# Norms
print(f"\nFrobenius norm of A: {norm(A, 'fro'):.4f}")
print(f"L2 norm of A: {norm(A, 2):.4f}")
print(f"L1 norm of A: {norm(A, 1):.4f}")

# Condition number
print(f"\nCondition number of A: {np.linalg.cond(A):.2e}")

# 4. Matrix Inverse
print("\n=== Matrix Inverse ===")

# For non-singular matrix
A_inv = inv(A_full_rank)
print(f"Inverse of modified A:\n{A_inv}")
print(f"\nVerification A @ A^(-1) = I:\n{A_full_rank @ A_inv}")
print(f"Is identity? {np.allclose(A_full_rank @ A_inv, I)}")

# Pseudo-inverse for singular matrix
A_pinv = np.linalg.pinv(A)
print(f"\nPseudo-inverse of singular A:\n{A_pinv}")

# 5. Linear System Solving
print("\n=== Solving Linear Systems ===")

# Solve Ax = b
A_system = np.array([[3, 1, -1],
                     [1, 4, 1],
                     [2, 1, 5]])
b = np.array([2, 12, 10])

# Method 1: Using inverse (not recommended for large systems)
x_inv = inv(A_system) @ b
print(f"Solution using inverse: {x_inv}")

# Method 2: Using solve (recommended)
x_solve = np.linalg.solve(A_system, b)
print(f"Solution using solve: {x_solve}")

# Verify solution
print(f"Verification A @ x = b: {np.allclose(A_system @ x_solve, b)}")

# 6. Matrix Visualization
print("\n=== Matrix Visualization ===")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Visualize matrices as heatmaps
matrices = [A, B, A @ B, S, Q, A_full_rank]
titles = ['Matrix A', 'Matrix B', 'A @ B', 'Symmetric S', 'Orthogonal Q', 'Full Rank A']

for ax, matrix, title in zip(axes.flat, matrices, titles):
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(title)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

---

## 2. Eigenvalues and Eigenvectors

### 📖 Core Concepts

#### Definition
For a square matrix A, if there exists a non-zero vector v and scalar λ such that:
```
Av = λv
```
Then:
- λ is an eigenvalue
- v is the corresponding eigenvector

#### Geometric Interpretation
Eigenvectors represent directions that are unchanged by the transformation A (only scaled by λ).

### 🔢 Mathematical Foundation

#### Characteristic Equation
To find eigenvalues:
```
det(A - λI) = 0
```

#### Properties
1. Sum of eigenvalues = trace(A)
2. Product of eigenvalues = det(A)
3. For symmetric matrices, eigenvalues are real
4. For positive definite matrices, eigenvalues > 0

### 💻 Eigenvalues and Eigenvectors Code

```python
# 7. Eigenvalues and Eigenvectors
print("\n=== Eigenvalues and Eigenvectors ===")

# Example matrix
A_eigen = np.array([[4, -2],
                    [1, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A_eigen)
print(f"Matrix A:\n{A_eigen}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")

# Verify Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    λ = eigenvalues[i]
    Av = A_eigen @ v
    λv = λ * v
    print(f"\nEigenvalue {i+1}: {λ}")
    print(f"Av = {Av}")
    print(f"λv = {λv}")
    print(f"Equal? {np.allclose(Av, λv)}")

# Visualization of eigenvector transformation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original vectors
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Plot original
ax = axes[0]
ax.plot(circle[0], circle[1], 'b-', alpha=0.3, label='Unit circle')
ax.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], 
          angles='xy', scale_units='xy', scale=1, color='red', width=0.01,
          label=f'v1 (λ={eigenvalues[0]:.2f})')
ax.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], 
          angles='xy', scale_units='xy', scale=1, color='green', width=0.01,
          label=f'v2 (λ={eigenvalues[1]:.2f})')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Original Eigenvectors')

# Transform and plot
transformed = A_eigen @ circle
ax = axes[1]
ax.plot(transformed[0], transformed[1], 'b-', alpha=0.3, label='Transformed')

# Show transformed eigenvectors
v1_transformed = A_eigen @ eigenvectors[:, 0]
v2_transformed = A_eigen @ eigenvectors[:, 1]
ax.quiver(0, 0, v1_transformed[0], v1_transformed[1], 
          angles='xy', scale_units='xy', scale=1, color='red', width=0.01,
          label=f'Av1 = {eigenvalues[0]:.2f}v1')
ax.quiver(0, 0, v2_transformed[0], v2_transformed[1], 
          angles='xy', scale_units='xy', scale=1, color='green', width=0.01,
          label=f'Av2 = {eigenvalues[1]:.2f}v2')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Transformed by A')

plt.tight_layout()
plt.show()

# 8. Eigenvalues for Special Matrices
print("\n=== Eigenvalues of Special Matrices ===")

# Symmetric matrix - real eigenvalues
S_eigen = np.array([[3, 1, 1],
                    [1, 3, 1],
                    [1, 1, 3]])
eigenvalues_S, eigenvectors_S = eig(S_eigen)
print(f"Symmetric matrix eigenvalues: {eigenvalues_S}")
print(f"All real? {np.all(np.isreal(eigenvalues_S))}")

# Positive definite matrix - positive eigenvalues
P = S_eigen @ S_eigen.T  # Guaranteed positive definite
eigenvalues_P, _ = eig(P)
print(f"\nPositive definite eigenvalues: {eigenvalues_P}")
print(f"All positive? {np.all(eigenvalues_P > 0)}")

# Orthogonal matrix - eigenvalues on unit circle
Q_eigen, _ = qr(np.random.randn(3, 3))
eigenvalues_Q, _ = eig(Q_eigen)
print(f"\nOrthogonal matrix eigenvalues: {eigenvalues_Q}")
print(f"Magnitudes: {np.abs(eigenvalues_Q)}")
```

---

## 3. Matrix Decompositions

### 📖 Core Concepts

Matrix decompositions factor a matrix into product of simpler matrices, useful for:
- Solving linear systems
- Computing matrix properties
- Dimensionality reduction
- Numerical stability

### 🔢 Types of Decompositions

#### 1. **LU Decomposition**
```
A = LU
```
- L: Lower triangular
- U: Upper triangular

#### 2. **QR Decomposition**
```
A = QR
```
- Q: Orthogonal matrix
- R: Upper triangular

#### 3. **Eigendecomposition**
```
A = VΛV^(-1)
```
- V: Eigenvectors
- Λ: Diagonal matrix of eigenvalues

#### 4. **Singular Value Decomposition (SVD)**
```
A = UΣV^T
```
- U, V: Orthogonal matrices
- Σ: Diagonal matrix of singular values

### 💻 Matrix Decompositions Code

```python
# 9. LU Decomposition
print("\n=== LU Decomposition ===")

A_lu = np.array([[2, 1, 1],
                 [4, -6, 0],
                 [-2, 7, 2]])

# LU decomposition
P, L, U = lu(A_lu)
print(f"Original matrix A:\n{A_lu}")
print(f"\nPermutation matrix P:\n{P}")
print(f"\nLower triangular L:\n{L}")
print(f"\nUpper triangular U:\n{U}")
print(f"\nVerification P @ A = L @ U:\n{P @ A_lu}\n{L @ U}")

# 10. QR Decomposition
print("\n=== QR Decomposition ===")

A_qr = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 10]])

Q, R = qr(A_qr)
print(f"Original matrix A:\n{A_qr}")
print(f"\nOrthogonal matrix Q:\n{Q}")
print(f"\nUpper triangular R:\n{R}")
print(f"\nVerification A = Q @ R:\n{Q @ R}")
print(f"Q orthogonal? (Q @ Q.T = I): {np.allclose(Q @ Q.T, np.eye(3))}")

# 11. Eigendecomposition
print("\n=== Eigendecomposition ===")

# For symmetric matrix (guaranteed real eigenvalues)
A_sym = np.array([[4, 1, 1],
                  [1, 3, 2],
                  [1, 2, 5]])

eigenvalues, V = eig(A_sym)
Λ = np.diag(eigenvalues)

print(f"Original symmetric matrix A:\n{A_sym}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvector matrix V:\n{V}")
print(f"\nDiagonal Λ:\n{Λ}")
print(f"\nReconstruction A = V @ Λ @ V^(-1):\n{V @ Λ @ inv(V)}")

# 12. Singular Value Decomposition (SVD)
print("\n=== Singular Value Decomposition (SVD) ===")

# Can work with non-square matrices
A_svd = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

U, s, Vt = svd(A_svd)
Σ = np.zeros_like(A_svd, dtype=float)
Σ[:min(A_svd.shape), :min(A_svd.shape)] = np.diag(s)

print(f"Original matrix A (3×4):\n{A_svd}")
print(f"\nLeft singular vectors U (3×3):\n{U}")
print(f"\nSingular values: {s}")
print(f"\nRight singular vectors V^T (4×4):\n{Vt}")
print(f"\nSigma matrix Σ:\n{Σ}")
print(f"\nReconstruction A = U @ Σ @ V^T:\n{U @ Σ @ Vt}")

# Low-rank approximation
k = 2  # Keep top k singular values
U_k = U[:, :k]
Σ_k = Σ[:k, :k]
Vt_k = Vt[:k, :]
A_approx = U_k @ Σ_k @ Vt_k

print(f"\nRank-{k} approximation:\n{A_approx}")
print(f"Approximation error (Frobenius norm): {norm(A_svd - A_approx, 'fro'):.6f}")

# 13. Cholesky Decomposition
print("\n=== Cholesky Decomposition ===")

# For positive definite matrix
A_pd = np.array([[4, 2, 1],
                 [2, 5, 2],
                 [1, 2, 3]])

L_chol = cholesky(A_pd, lower=True)
print(f"Positive definite matrix A:\n{A_pd}")
print(f"\nCholesky factor L:\n{L_chol}")
print(f"\nVerification A = L @ L^T:\n{L_chol @ L_chol.T}")

# 14. Applications in ML
print("\n=== ML Applications ===")

# PCA using eigendecomposition
print("1. PCA using Eigendecomposition:")
# Generate correlated data
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)

# Center the data
data_centered = data - np.mean(data, axis=0)

# Compute covariance matrix
C = np.cov(data_centered.T)

# Eigendecomposition
eigenvalues, eigenvectors = eig(C)

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Covariance matrix:\n{C}")
print(f"Eigenvalues (variance explained): {eigenvalues}")
print(f"Principal components:\n{eigenvectors}")
print(f"Variance explained ratio: {eigenvalues / np.sum(eigenvalues)}")

# Visualize PCA
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Original Data')
ax.set_aspect('equal')

# Add principal components
origin = np.mean(data, axis=0)
for i in range(2):
    ax.arrow(origin[0], origin[1], 
             eigenvectors[0, i] * np.sqrt(eigenvalues[i]) * 2,
             eigenvectors[1, i] * np.sqrt(eigenvalues[i]) * 2,
             head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}',
             label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')
ax.legend()

# Transformed data
ax = axes[1]
data_transformed = data_centered @ eigenvectors
ax.scatter(data_transformed[:, 0], data_transformed[:, 1], alpha=0.5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Transformed Data')
ax.set_aspect('equal')

plt.tight_layout()
plt.show()

# SVD for collaborative filtering
print("\n2. SVD for Collaborative Filtering:")
# User-item matrix (ratings)
ratings = np.array([[5, 3, 0, 1],
                    [4, 0, 0, 1],
                    [1, 1, 0, 5],
                    [1, 0, 0, 4],
                    [0, 1, 5, 4]])

print(f"User-Item Rating Matrix:\n{ratings}")

# SVD
U, s, Vt = svd(ratings, full_matrices=False)

# Low-rank approximation (k=2)
k = 2
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Reconstruct ratings
ratings_pred = U_k @ np.diag(s_k) @ Vt_k
print(f"\nPredicted ratings (rank-{k} approximation):\n{ratings_pred}")

# Find recommendations for user 0
user_0_ratings = ratings_pred[0]
unrated_items = np.where(ratings[0] == 0)[0]
recommendations = [(item, user_0_ratings[item]) for item in unrated_items]
recommendations.sort(key=lambda x: x[1], reverse=True)
print(f"\nRecommendations for User 0: {recommendations}")
```

## 🎯 Interview Questions

1. **Q: What's the computational complexity of matrix multiplication?**
   - A: O(n³) for square matrices, O(mnp) for (m×n) × (n×p).

2. **Q: When is a matrix invertible?**
   - A: When det(A) ≠ 0, full rank, linearly independent columns/rows.

3. **Q: What's the difference between eigenvalues and singular values?**
   - A: Eigenvalues require square matrices, can be complex. Singular values work for any matrix, always non-negative real.

4. **Q: How is SVD used in dimensionality reduction?**
   - A: Keep top k singular values/vectors for rank-k approximation, minimizes reconstruction error.

5. **Q: What's the relationship between PCA and eigendecomposition?**
   - A: PCA finds eigenvectors of covariance matrix; principal components are eigenvectors, variance explained by eigenvalues.

## 📝 Practice Exercises

1. Implement power iteration method to find dominant eigenvalue
2. Use SVD to compress an image
3. Solve a linear system using LU decomposition
4. Implement PCA from scratch using eigendecomposition

## 🔗 Key Takeaways
- Matrix operations form the foundation of linear algebra in ML
- Eigenvalues/eigenvectors reveal matrix properties and transformations
- Decompositions provide numerical stability and efficiency
- SVD is particularly powerful for dimensionality reduction
- Understanding these concepts is crucial for advanced ML algorithms