# Machine Learning Interview Preparation - Week 1: Foundations

## Day 1: NumPy and Pandas

### NumPy - Numerical Python

NumPy is the foundation of scientific computing in Python, providing support for large multi-dimensional arrays and matrices.

#### Core Concepts

**1. Array Creation and Operations**

```python
import numpy as np

# Array creation methods
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 4))  # 3x4 matrix of zeros
arr3 = np.ones((2, 3))   # 2x3 matrix of ones
arr4 = np.eye(4)         # 4x4 identity matrix
arr5 = np.random.randn(3, 3)  # Random normal distribution
arr6 = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
arr7 = np.linspace(0, 1, 5)   # 5 evenly spaced points between 0 and 1

# Array attributes
print(f"Shape: {arr5.shape}")
print(f"Dimensions: {arr5.ndim}")
print(f"Data type: {arr5.dtype}")
print(f"Size: {arr5.size}")
```

**2. Array Indexing and Slicing**

```python
# 2D array
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# Basic indexing
element = matrix[1, 2]  # 6
row = matrix[1, :]      # [4, 5, 6]
col = matrix[:, 1]      # [2, 5, 8]

# Advanced indexing
bool_idx = matrix > 5   # Boolean mask
filtered = matrix[bool_idx]  # [6, 7, 8, 9]

# Fancy indexing
rows = np.array([0, 2])
cols = np.array([1, 2])
selected = matrix[rows, cols]  # [2, 9]
```

**3. Broadcasting**

Broadcasting allows operations between arrays of different shapes:

```python
# Broadcasting rules
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

# b is broadcast to match a's shape
result = a + b  # [[11, 22, 33], [14, 25, 36]]

# Scalar broadcasting
result2 = a * 2  # Element-wise multiplication
```

**4. Advanced Array Operations**

```python
# Reshaping
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vstacked = np.vstack([a, b])  # Vertical stack
hstacked = np.hstack([a, b])  # Horizontal stack

# Mathematical operations
matrix = np.array([[1, 2], [3, 4]])
transpose = matrix.T
inverse = np.linalg.inv(matrix)
determinant = np.linalg.det(matrix)
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Statistical operations
mean = np.mean(matrix, axis=0)  # Column means
std = np.std(matrix, axis=1)    # Row standard deviations
```

### Pandas - Data Analysis Library

Pandas provides high-performance data structures and data analysis tools.

#### Core Concepts

**1. Series - 1D labeled array**

```python
import pandas as pd

# Creating Series
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})

# Series operations
print(s2['b'])  # Accessing by label
print(s2[s2 > 1])  # Boolean indexing
print(s2.mean(), s2.std())  # Statistical operations
```

**2. DataFrame - 2D labeled data structure**

```python
# Creating DataFrames
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000],
    'Department': ['IT', 'HR', 'IT', 'Finance']
}
df = pd.DataFrame(data)

# From NumPy array
np_data = np.random.randn(4, 3)
df2 = pd.DataFrame(np_data, columns=['A', 'B', 'C'])

# Reading from files
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')
```

**3. DataFrame Operations**

```python
# Viewing data
print(df.head())      # First 5 rows
print(df.tail(3))     # Last 3 rows
print(df.info())      # DataFrame info
print(df.describe())  # Statistical summary

# Selecting data
ages = df['Age']                    # Select column
subset = df[['Name', 'Salary']]     # Select multiple columns
row = df.loc[0]                     # Select by index label
rows = df.iloc[1:3]                 # Select by position

# Conditional selection
it_employees = df[df['Department'] == 'IT']
high_earners = df[df['Salary'] > 60000]
complex_filter = df[(df['Age'] > 30) & (df['Salary'] < 75000)]
```

**4. Data Manipulation**

```python
# Adding/modifying columns
df['Bonus'] = df['Salary'] * 0.1
df['Years'] = df['Age'] - 22

# Applying functions
df['Tax'] = df['Salary'].apply(lambda x: x * 0.3)
df['Category'] = df['Age'].apply(lambda x: 'Senior' if x > 35 else 'Junior')

# Grouping and aggregation
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'max', 'min'],
    'Age': 'mean'
})

# Sorting
df_sorted = df.sort_values('Salary', ascending=False)
df_multi_sort = df.sort_values(['Department', 'Salary'], ascending=[True, False])
```

**5. Advanced Pandas Operations**

```python
# Handling missing data
df_with_nan = df.copy()
df_with_nan.loc[1, 'Salary'] = np.nan
df_filled = df_with_nan.fillna(df_with_nan['Salary'].mean())
df_dropped = df_with_nan.dropna()

# Merging DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [90, 85, 95]})

# Different types of joins
inner_join = pd.merge(df1, df2, on='ID', how='inner')
left_join = pd.merge(df1, df2, on='ID', how='left')
outer_join = pd.merge(df1, df2, on='ID', how='outer')

# Pivot tables
pivot = df.pivot_table(values='Salary', 
                       index='Department', 
                       columns='Category', 
                       aggfunc='mean')

# Time series operations
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)
monthly_mean = ts.resample('M').mean()
```

## Day 2: Matplotlib and Seaborn

### Matplotlib - Fundamental Plotting Library

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Line plots
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linestyle='--')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plot
axes[0, 0].scatter(np.random.randn(100), np.random.randn(100), alpha=0.5)
axes[0, 0].set_title('Scatter Plot')

# Histogram
axes[0, 1].hist(np.random.normal(0, 1, 1000), bins=30, edgecolor='black')
axes[0, 1].set_title('Histogram')

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 0].bar(categories, values, color=['red', 'green', 'blue', 'yellow'])
axes[1, 0].set_title('Bar Plot')

# Box plot
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
axes[1, 1].boxplot(data, labels=['S1', 'S2', 'S3'])
axes[1, 1].set_title('Box Plot')

plt.tight_layout()
plt.show()
```

### Seaborn - Statistical Data Visualization

```python
import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load sample data
tips = sns.load_dataset("tips")

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram with KDE
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Total Bill')

# Box plot with categories
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0, 1])
axes[0, 1].set_title('Total Bill by Day')

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True, ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot: Bill by Day and Sex')

# Scatter plot with regression
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', style='sex', ax=axes[1, 1])
axes[1, 1].set_title('Bill vs Tip')

plt.tight_layout()
plt.show()

# Correlation heatmap
numeric_cols = tips.select_dtypes(include=[np.number]).columns
correlation_matrix = tips[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pair plot for multivariate analysis
sns.pairplot(tips, hue='time', diag_kind='kde')
plt.suptitle('Pair Plot of Tips Dataset', y=1.02)
plt.show()
```

## Day 3: Linear Algebra - Matrix Operations

### Mathematical Foundations

**1. Matrix Basics**

A matrix is a rectangular array of numbers arranged in rows and columns.

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [...  ...  ...  ...]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

**2. Matrix Operations**

```python
import numpy as np

# Matrix creation
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Addition and Subtraction
C = A + B  # Element-wise addition
D = A - B  # Element-wise subtraction

# Scalar multiplication
E = 2 * A

# Matrix multiplication
F = A @ B  # or np.dot(A, B)
# Note: (m×n) @ (n×p) = (m×p)

# Element-wise multiplication (Hadamard product)
G = A * B

# Transpose
A_T = A.T
# Property: (AB)ᵀ = BᵀAᵀ

# Trace (sum of diagonal elements)
trace_A = np.trace(A)  # 1 + 5 + 9 = 15
```

**3. Special Matrices**

```python
# Identity matrix
I = np.eye(3)
# Property: AI = IA = A

# Diagonal matrix
diag_vals = [1, 2, 3]
D = np.diag(diag_vals)

# Symmetric matrix (A = Aᵀ)
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])

# Orthogonal matrix (QᵀQ = QQᵀ = I)
theta = np.pi/4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
```

### Eigenvalues and Eigenvectors

**Mathematical Definition:**
For a square matrix A, if there exists a non-zero vector v and scalar λ such that:
```
Av = λv
```
Then λ is an eigenvalue and v is the corresponding eigenvector.

**Characteristic Equation:**
```
det(A - λI) = 0
```

```python
# Computing eigenvalues and eigenvectors
A = np.array([[4, -2],
              [1, 1]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verification
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    
    # Check Av = λv
    Av = A @ v_i
    lambda_v = lambda_i * v_i
    
    print(f"\nEigenvalue {lambda_i}:")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")
    print(f"Close? {np.allclose(Av, lambda_v)}")
```

**Applications of Eigendecomposition:**

```python
# 1. Matrix Diagonalization
# If A has n linearly independent eigenvectors, then A = PDP⁻¹
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ D @ P_inv
print(f"Original A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")

# 2. Matrix Powers
# A^n = PD^nP⁻¹
n = 5
A_power_5 = P @ np.diag(eigenvalues**n) @ P_inv
A_power_5_direct = np.linalg.matrix_power(A, n)
print(f"A^5 using eigendecomposition:\n{A_power_5}")
print(f"A^5 direct computation:\n{A_power_5_direct}")

# 3. Principal Component Analysis (PCA) preview
# Generate correlated data
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)

# Compute covariance matrix
cov_matrix = np.cov(data.T)

# Eigendecomposition of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Principal components (eigenvectors):\n{eigenvectors}")
print(f"Explained variance (eigenvalues): {eigenvalues}")
print(f"Variance ratio: {eigenvalues / eigenvalues.sum()}")
```

### Advanced Matrix Concepts

```python
# Matrix Decompositions

# 1. LU Decomposition
from scipy.linalg import lu
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]])

P, L, U = lu(A)
print(f"P (Permutation):\n{P}")
print(f"L (Lower triangular):\n{L}")
print(f"U (Upper triangular):\n{U}")
print(f"Reconstruction PLU:\n{P @ L @ U}")

# 2. QR Decomposition
Q, R = np.linalg.qr(A)
print(f"Q (Orthogonal):\n{Q}")
print(f"R (Upper triangular):\n{R}")
print(f"QᵀQ = I? {np.allclose(Q.T @ Q, np.eye(3))}")

# 3. Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(A)
S = np.diag(s)
print(f"U:\n{U}")
print(f"Singular values: {s}")
print(f"Vᵀ:\n{Vt}")

# Applications in ML
# Rank of matrix
rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}")

# Condition number (numerical stability indicator)
cond = np.linalg.cond(A)
print(f"Condition number: {cond}")

# Determinant
det = np.linalg.det(A)
print(f"Determinant: {det}")

# Inverse (if exists)
if det != 0:
    A_inv = np.linalg.inv(A)
    print(f"A⁻¹:\n{A_inv}")
    print(f"AA⁻¹ = I? {np.allclose(A @ A_inv, np.eye(3))}")
```

## Day 4: Probability and KL-Divergence

### Probability Fundamentals

**1. Basic Probability Concepts**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample space and events
# Example: Rolling two dice
sample_space = [(i, j) for i in range(1, 7) for j in range(1, 7)]
print(f"Sample space size: {len(sample_space)}")

# Event: Sum equals 7
event_sum_7 = [(i, j) for i, j in sample_space if i + j == 7]
prob_sum_7 = len(event_sum_7) / len(sample_space)
print(f"P(sum = 7) = {prob_sum_7:.3f}")

# Conditional probability: P(A|B) = P(A ∩ B) / P(B)
# Event A: First die shows 6
# Event B: Sum is greater than 8
event_A = [(i, j) for i, j in sample_space if i == 6]
event_B = [(i, j) for i, j in sample_space if i + j > 8]
event_A_and_B = [(i, j) for i, j in sample_space if i == 6 and i + j > 8]

P_A = len(event_A) / len(sample_space)
P_B = len(event_B) / len(sample_space)
P_A_and_B = len(event_A_and_B) / len(sample_space)
P_A_given_B = P_A_and_B / P_B

print(f"P(A) = {P_A:.3f}")
print(f"P(B) = {P_B:.3f}")
print(f"P(A|B) = {P_A_given_B:.3f}")
```

**2. Probability Distributions**

```python
# Discrete Distributions

# Binomial Distribution
n, p = 10, 0.3
binomial_dist = stats.binom(n, p)

x = np.arange(0, n+1)
pmf = binomial_dist.pmf(x)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.bar(x, pmf)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('k')
plt.ylabel('P(X=k)')

# Poisson Distribution
lambda_param = 3
poisson_dist = stats.poisson(lambda_param)

x = np.arange(0, 15)
pmf = poisson_dist.pmf(x)

plt.subplot(2, 2, 2)
plt.bar(x, pmf)
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('k')
plt.ylabel('P(X=k)')

# Continuous Distributions

# Normal Distribution
mu, sigma = 0, 1
normal_dist = stats.norm(mu, sigma)

x = np.linspace(-4, 4, 1000)
pdf = normal_dist.pdf(x)

plt.subplot(2, 2, 3)
plt.plot(x, pdf)
plt.fill_between(x, pdf, alpha=0.3)
plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
plt.xlabel('x')
plt.ylabel('f(x)')

# Exponential Distribution
lambda_exp = 2
exp_dist = stats.expon(scale=1/lambda_exp)

x = np.linspace(0, 3, 1000)
pdf = exp_dist.pdf(x)

plt.subplot(2, 2, 4)
plt.plot(x, pdf)
plt.fill_between(x, pdf, alpha=0.3)
plt.title(f'Exponential Distribution (λ={lambda_exp})')
plt.xlabel('x')
plt.ylabel('f(x)')

plt.tight_layout()
plt.show()
```

**3. Bayes' Theorem**

```
P(A|B) = P(B|A) × P(A) / P(B)
```

```python
# Medical diagnosis example
# Disease prevalence
P_disease = 0.01  # 1% of population has the disease

# Test accuracy
P_positive_given_disease = 0.99  # Sensitivity
P_negative_given_healthy = 0.95  # Specificity

# Derived probabilities
P_healthy = 1 - P_disease
P_positive_given_healthy = 1 - P_negative_given_healthy

# Total probability of positive test
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_healthy * P_healthy)

# Bayes' theorem: P(disease|positive test)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"Prior probability of disease: {P_disease:.3f}")
print(f"Probability of positive test: {P_positive:.3f}")
print(f"Posterior probability of disease given positive test: {P_disease_given_positive:.3f}")
```

### Kullback-Leibler (KL) Divergence

**Mathematical Definition:**
For discrete probability distributions P and Q:
```
D_KL(P||Q) = Σᵢ P(i) log(P(i)/Q(i))
```

For continuous distributions:
```
D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
```

**Properties:**
- D_KL(P||Q) ≥ 0 (non-negative)
- D_KL(P||Q) = 0 if and only if P = Q
- D_KL(P||Q) ≠ D_KL(Q||P) (not symmetric)

```python
def kl_divergence_discrete(p, q):
    """Calculate KL divergence for discrete distributions"""
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))

# Example 1: Comparing two dice
# Fair die
p_fair = np.ones(6) / 6

# Biased die (favors 6)
p_biased = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.4])

kl_fair_to_biased = kl_divergence_discrete(p_fair, p_biased)
kl_biased_to_fair = kl_divergence_discrete(p_biased, p_fair)

print(f"KL(fair||biased) = {kl_fair_to_biased:.4f}")
print(f"KL(biased||fair) = {kl_biased_to_fair:.4f}")
print(f"Symmetric? {kl_fair_to_biased == kl_biased_to_fair}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(1, 7)
width = 0.35

axes[0].bar(x - width/2, p_fair, width, label='Fair die', alpha=0.8)
axes[0].bar(x + width/2, p_biased, width, label='Biased die', alpha=0.8)
axes[0].set_xlabel('Outcome')
axes[0].set_ylabel('Probability')
axes[0].set_title('Probability Distributions')
axes[0].legend()
axes[0].set_xticks(x)

# KL divergence for continuous distributions
mu1, sigma1 = 0, 1
mu2, sigma2 = 2, 1.5

x = np.linspace(-5, 7, 1000)
p = stats.norm.pdf(x, mu1, sigma1)
q = stats.norm.pdf(x, mu2, sigma2)

# Analytical KL divergence for Gaussians
kl_analytical = (np.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5)

# Numerical approximation
dx = x[1] - x[0]
kl_numerical = np.sum(p * np.log(p / (q + 1e-10)) * dx)

axes[1].plot(x, p, label=f'P: N({mu1}, {sigma1}²)', linewidth=2)
axes[1].plot(x, q, label=f'Q: N({mu2}, {sigma2}²)', linewidth=2)
axes[1].fill_between(x, p, alpha=0.3)
axes[1].fill_between(x, q, alpha=0.3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Probability density')
axes[1].set_title(f'KL(P||Q) = {kl_analytical:.3f}')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Applications in Machine Learning:**

```python
# 1. Loss function in Variational Autoencoders (VAE)
def vae_kl_loss(mu, log_var):
    """KL divergence between N(mu, sigma) and N(0, 1)"""
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

# 2. Model comparison
# Comparing predicted distribution to true distribution
true_probs = np.array([0.2, 0.3, 0.3, 0.2])
model1_probs = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform
model2_probs = np.array([0.15, 0.35, 0.35, 0.15])  # Closer to true

kl_model1 = kl_divergence_discrete(true_probs, model1_probs)
kl_model2 = kl_divergence_discrete(true_probs, model2_probs)

print(f"KL(true||model1) = {kl_model1:.4f}")
print(f"KL(true||model2) = {kl_model2:.4f}")
print(f"Better model: Model {'2' if kl_model2 < kl_model1 else '1'}")

# 3. Information gain in decision trees
def information_gain(parent, children, weights):
    """Calculate information gain using KL divergence concept"""
    weighted_entropy = sum(w * stats.entropy(child) for child, w in zip(children, weights))
    return stats.entropy(parent) - weighted_entropy

# Example: Binary classification
parent_dist = np.array([0.5, 0.5])  # 50-50 split
left_child = np.array([0.8, 0.2])   # 80-20 split
right_child = np.array([0.2, 0.8])  # 20-80 split
weights = [0.6, 0.4]  # 60% go left, 40% go right

ig = information_gain(parent_dist, [left_child, right_child], weights)
print(f"Information gain: {ig:.4f}")
```

## Day 5: Statistics - Central Tendency and Spread

### Measures of Central Tendency

**1. Mean (Arithmetic Average)**
```
μ = (1/n) Σᵢ xᵢ
```

**2. Median**
- Middle value when data is sorted
- Robust to outliers

**3. Mode**
- Most frequent value
- Can have multiple modes (bimodal, multimodal)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
normal_data = np.random.normal(100, 15, 1000)
skewed_data = np.random.exponential(50, 1000)
bimodal_data = np.concatenate([np.random.normal(50, 10, 500), 
                               np.random.normal(100, 10, 500)])

def analyze_central_tendency(data, name):
    """Comprehensive analysis of central tendency"""
    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data, keepdims=True)
    mode = mode_result.mode[0]
    
    print(f"\n{name}:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")
    print(f"Skewness: {stats.skew(data):.2f}")
    
    return mean, median, mode

# Analyze different distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Normal distribution
mean1, median1, mode1 = analyze_central_tendency(normal_data, "Normal Distribution")
axes[0, 0].hist(normal_data, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(mean1, color='red', linestyle='--', label=f'Mean: {mean1:.1f}')
axes[0, 0].axvline(median1, color='green', linestyle='--', label=f'Median: {median1:.1f}')
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].legend()

# Skewed distribution
mean2, median2, mode2 = analyze_central_tendency(skewed_data, "Skewed Distribution")
axes[0, 1].hist(skewed_data, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(mean2, color='red', linestyle='--', label=f'Mean: {mean2:.1f}')
axes[0, 1].axvline(median2, color='green', linestyle='--', label=f'Median: {median2:.1f}')
axes[0, 1].set_title('Right-Skewed Distribution')
axes[0, 1].legend()

# Bimodal distribution
mean3, median3, mode3 = analyze_central_tendency(bimodal_data, "Bimodal Distribution")
axes[0, 2].hist(bimodal_data, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0, 2].axvline(mean3, color='red', linestyle='--', label=f'Mean: {mean3:.1f}')
axes[0, 2].axvline(median3, color='green', linestyle='--', label=f'Median: {median3:.1f}')
axes[0, 2].set_title('Bimodal Distribution')
axes[0, 2].legend()
```

### Measures of Spread

**1. Variance**
```
σ² = (1/n) Σᵢ (xᵢ - μ)²  (Population)
s² = (1/(n-1)) Σᵢ (xᵢ - x̄)²  (Sample)
```

**2. Standard Deviation**
```
σ = √(σ²)
```

**3. Covariance**
```
Cov(X,Y) = (1/n) Σᵢ (xᵢ - μₓ)(yᵢ - μᵧ)
```

```python
# Variance and Standard Deviation
def analyze_spread(data, name):
    """Analyze measures of spread"""
    variance = np.var(data, ddof=0)  # Population variance
    sample_variance = np.var(data, ddof=1)  # Sample variance
    std_dev = np.std(data, ddof=0)
    sample_std = np.std(data, ddof=1)
    
    # Percentiles and IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Coefficient of variation
    cv = (std_dev / np.mean(data)) * 100
    
    print(f"\n{name} - Spread Measures:")
    print(f"Population Variance: {variance:.2f}")
    print(f"Sample Variance: {sample_variance:.2f}")
    print(f"Population Std Dev: {std_dev:.2f}")
    print(f"Sample Std Dev: {sample_std:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Coefficient of Variation: {cv:.2f}%")
    
    return variance, std_dev, iqr

# Demonstrating effect of outliers
clean_data = np.random.normal(100, 10, 100)
outlier_data = np.append(clean_data, [300, 350, 400])  # Add outliers

axes[1, 0].boxplot([clean_data, outlier_data], labels=['Clean', 'With Outliers'])
axes[1, 0].set_title('Effect of Outliers on Spread')
axes[1, 0].set_ylabel('Value')

analyze_spread(clean_data, "Clean Data")
analyze_spread(outlier_data, "Data with Outliers")

# Covariance and its interpretation
# Generate correlated data
mean = [0, 0]
cov_matrix = [[1, 0.8], [0.8, 1]]  # High positive correlation
data_corr = np.random.multivariate_normal(mean, cov_matrix, 500)

x_corr = data_corr[:, 0]
y_corr = data_corr[:, 1]

# Calculate covariance
cov_xy = np.cov(x_corr, y_corr)[0, 1]
print(f"\nCovariance between X and Y: {cov_xy:.3f}")

# Visualize
axes[1, 1].scatter(x_corr, y_corr, alpha=0.5)
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].set_title(f'Positive Covariance: {cov_xy:.3f}')

# Multiple variables covariance matrix
# Generate multivariate data
mean_multi = [100, 50, 75]
cov_matrix_multi = [[225, 100, 50],    # Var(X1)=225, Cov(X1,X2)=100, Cov(X1,X3)=50
                    [100, 400, -100],   # Var(X2)=400, Cov(X2,X3)=-100
                    [50, -100, 625]]    # Var(X3)=625

data_multi = np.random.multivariate_normal(mean_multi, cov_matrix_multi, 1000)

# Calculate covariance matrix
cov_calculated = np.cov(data_multi.T)
print("\nCovariance Matrix:")
print(pd.DataFrame(cov_calculated, 
                  columns=['X1', 'X2', 'X3'],
                  index=['X1', 'X2', 'X3']).round(2))

# Visualize covariance matrix
im = axes[1, 2].imshow(cov_calculated, cmap='coolwarm', aspect='auto')
axes[1, 2].set_xticks([0, 1, 2])
axes[1, 2].set_yticks([0, 1, 2])
axes[1, 2].set_xticklabels(['X1', 'X2', 'X3'])
axes[1, 2].set_yticklabels(['X1', 'X2', 'X3'])
axes[1, 2].set_title('Covariance Matrix Heatmap')

# Add values to heatmap
for i in range(3):
    for j in range(3):
        text = axes[1, 2].text(j, i, f'{cov_calculated[i, j]:.0f}',
                              ha="center", va="center", color="black")

plt.colorbar(im, ax=axes[1, 2])
plt.tight_layout()
plt.show()
```

### Advanced Statistical Concepts

```python
# Robust measures of spread
def mad(data):
    """Median Absolute Deviation"""
    median = np.median(data)
    return np.median(np.abs(data - median))

def trimmed_mean(data, trim_percent=0.1):
    """Trimmed mean (removing extreme values)"""
    data_sorted = np.sort(data)
    trim_count = int(len(data) * trim_percent)
    return np.mean(data_sorted[trim_count:-trim_count])

# Example with outliers
data_with_outliers = np.concatenate([
    np.random.normal(100, 10, 100),
    np.array([500, 600, 700])  # Extreme outliers
])

print("Comparison of Robust vs Non-robust Measures:")
print(f"Mean: {np.mean(data_with_outliers):.2f}")
print(f"Trimmed Mean (10%): {trimmed_mean(data_with_outliers):.2f}")
print(f"Median: {np.median(data_with_outliers):.2f}")
print(f"Standard Deviation: {np.std(data_with_outliers):.2f}")
print(f"MAD: {mad(data_with_outliers):.2f}")

# Moments of distribution
def calculate_moments(data):
    """Calculate first four moments"""
    mean = np.mean(data)
    
    # Raw moments
    first_moment = mean
    second_moment = np.mean(data**2)
    third_moment = np.mean(data**3)
    fourth_moment = np.mean(data**4)
    
    # Central moments
    variance = np.var(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    return {
        'mean': first_moment,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

# Compare different distributions
distributions = {
    'Normal': np.random.normal(0, 1, 10000),
    'Uniform': np.random.uniform(-2, 2, 10000),
    'Exponential': np.random.exponential(1, 10000),
    'Student-t (df=3)': np.random.standard_t(3, 10000)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, data) in enumerate(distributions.items()):
    moments = calculate_moments(data)
    
    axes[idx].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{name}\nSkew: {moments["skewness"]:.2f}, Kurt: {moments["kurtosis"]:.2f}')
    axes[idx].set_xlim(-5, 5)
    
    # Add normal curve for reference
    x = np.linspace(-5, 5, 100)
    axes[idx].plot(x, stats.norm.pdf(x), 'r--', label='Standard Normal')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

## Day 6: Correlation and Multicollinearity

### Correlation

**Pearson Correlation Coefficient:**
```
r = Σᵢ(xᵢ - x̄)(yᵢ - ȳ) / √[Σᵢ(xᵢ - x̄)² × Σᵢ(yᵢ - ȳ)²]
```

Properties:
- Range: [-1, 1]
- r = 1: Perfect positive linear correlation
- r = -1: Perfect negative linear correlation
- r = 0: No linear correlation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate different types of correlations
np.random.seed(42)
n_points = 200

# 1. Strong positive correlation
x1 = np.random.normal(0, 1, n_points)
y1 = 2 * x1 + np.random.normal(0, 0.5, n_points)

# 2. Strong negative correlation
x2 = np.random.normal(0, 1, n_points)
y2 = -2 * x2 + np.random.normal(0, 0.5, n_points)

# 3. No correlation
x3 = np.random.normal(0, 1, n_points)
y3 = np.random.normal(0, 1, n_points)

# 4. Non-linear relationship (zero correlation but related)
x4 = np.random.uniform(-3, 3, n_points)
y4 = x4**2 + np.random.normal(0, 0.5, n_points)

# Calculate correlations
correlations = [
    (x1, y1, "Strong Positive"),
    (x2, y2, "Strong Negative"),
    (x3, y3, "No Correlation"),
    (x4, y4, "Non-linear (Quadratic)")
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (x, y, title) in enumerate(correlations):
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Spearman correlation (rank-based)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Kendall's tau (another rank-based)
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    # Plot
    axes[idx].scatter(x, y, alpha=0.6)
    axes[idx].set_title(f'{title}\nPearson: {pearson_r:.3f}, Spearman: {spearman_r:.3f}')
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('Y')
    
    # Add regression line for linear relationships
    if idx < 3:  # Only for linear relationships
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[idx].plot(x, p(x), "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.show()

# Correlation matrix example
# Generate multivariate data with known correlations
mean = [0, 0, 0, 0]
correlation_matrix = np.array([
    [1.0, 0.8, -0.5, 0.2],
    [0.8, 1.0, -0.3, 0.1],
    [-0.5, -0.3, 1.0, 0.7],
    [0.2, 0.1, 0.7, 1.0]
])

# Convert correlation to covariance (assuming unit variance)
std_devs = [1, 2, 1.5, 3]
D = np.diag(std_devs)
covariance_matrix = D @ correlation_matrix @ D

# Generate data
data = np.random.multivariate_normal(mean, covariance_matrix, 500)
df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4'])

# Calculate and visualize correlation matrix
plt.figure(figsize=(10, 8))
correlation_calculated = df.corr()

# Heatmap with annotations
mask = np.triu(np.ones_like(correlation_calculated), k=1)
sns.heatmap(correlation_calculated, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap')
plt.show()

# Partial correlation (correlation between X and Y controlling for Z)
def partial_correlation(data, x_col, y_col, control_cols):
    """Calculate partial correlation between x and y controlling for other variables"""
    from sklearn.linear_model import LinearRegression
    
    # Residualize x
    lr_x = LinearRegression()
    lr_x.fit(data[control_cols], data[x_col])
    x_residual = data[x_col] - lr_x.predict(data[control_cols])
    
    # Residualize y
    lr_y = LinearRegression()
    lr_y.fit(data[control_cols], data[y_col])
    y_residual = data[y_col] - lr_y.predict(data[control_cols])
    
    # Correlation of residuals
    return np.corrcoef(x_residual, y_residual)[0, 1]

# Example: Partial correlation between X1 and X2 controlling for X3
partial_corr = partial_correlation(df, 'X1', 'X2', ['X3'])
simple_corr = df['X1'].corr(df['X2'])

print(f"\nSimple correlation between X1 and X2: {simple_corr:.3f}")
print(f"Partial correlation between X1 and X2 (controlling for X3): {partial_corr:.3f}")
```

### Multicollinearity

Multicollinearity occurs when predictor variables are highly correlated with each other.

**Detection Methods:**

1. **Correlation Matrix**: Check for high correlations (|r| > 0.8)
2. **Variance Inflation Factor (VIF)**:
   ```
   VIF_j = 1 / (1 - R²_j)
   ```
   where R²_j is the R-squared from regressing X_j on all other predictors

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Create dataset with multicollinearity
np.random.seed(42)
n_samples = 1000

# Independent variables
X1 = np.random.normal(0, 1, n_samples)
X2 = 2 * X1 + np.random.normal(0, 0.1, n_samples)  # Highly correlated with X1
X3 = np.random.normal(0, 1, n_samples)  # Independent
X4 = 0.5 * X1 + 0.5 * X3 + np.random.normal(0, 0.2, n_samples)  # Moderately correlated

# Combine into dataframe
X = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4
})

# Target variable
y = 3 * X1 + 2 * X3 + np.random.normal(0, 0.5, n_samples)

# Calculate VIF
def calculate_vif(df):
    """Calculate VIF for each feature"""
    vif = pd.DataFrame()
    vif["Variable"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

vif_results = calculate_vif(X)
print("Variance Inflation Factors:")
print(vif_results)
print("\nInterpretation:")
print("VIF > 10: High multicollinearity")
print("VIF 5-10: Moderate multicollinearity")
print("VIF < 5: Low multicollinearity")

# Visualize multicollinearity
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Correlation heatmap
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, ax=axes[0, 0], vmin=-1, vmax=1)
axes[0, 0].set_title('Correlation Heatmap')

# 2. Scatter plot matrix
pd.plotting.scatter_matrix(X, ax=axes.ravel()[1:], diagonal='hist', alpha=0.5)
plt.suptitle('Scatter Plot Matrix', y=0.5)

plt.tight_layout()
plt.show()

# Effects of multicollinearity on regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fit model with all features
model_full = LinearRegression()
model_full.fit(X, y)

# Fit model without X2 (highly correlated with X1)
model_reduced = LinearRegression()
model_reduced.fit(X[['X1', 'X3', 'X4']], y)

print("\nRegression Coefficients Comparison:")
print("Full Model (with multicollinearity):")
for i, coef in enumerate(model_full.coef_):
    print(f"  {X.columns[i]}: {coef:.3f}")

print("\nReduced Model (without X2):")
for i, col in enumerate(['X1', 'X3', 'X4']):
    print(f"  {col}: {model_reduced.coef_[i]:.3f}")

# Compare R-squared
r2_full = r2_score(y, model_full.predict(X))
r2_reduced = r2_score(y, model_reduced.predict(X[['X1', 'X3', 'X4']]))

print(f"\nR² Full Model: {r2_full:.4f}")
print(f"R² Reduced Model: {r2_reduced:.4f}")
```

### Handling Multicollinearity

```python
# Methods to handle multicollinearity

# 1. Remove highly correlated features
def remove_correlated_features(df, threshold=0.9):
    """Remove features with correlation above threshold"""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold)]
    
    return df.drop(columns=to_drop), to_drop

X_reduced, dropped_cols = remove_correlated_features(X.copy())
print(f"Dropped columns: {dropped_cols}")

# 2. Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. Ridge Regression (L2 regularization)
from sklearn.linear_model import Ridge, RidgeCV

# Find optimal alpha using cross-validation
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X, y)

print(f"\nOptimal Ridge alpha: {ridge_cv.alpha_:.4f}")

# Compare coefficients
ridge_model = Ridge(alpha=ridge_cv.alpha_)
ridge_model.fit(X, y)

# Visualize coefficient shrinkage
fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients for different alpha values
coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

for i in range(X.shape[1]):
    ax.plot(alphas, coefs[:, i], label=X.columns[i])

ax.set_xscale('log')
ax.set_xlabel('Alpha (Regularization strength)')
ax.set_ylabel('Coefficient value')
ax.set_title('Ridge Regression Coefficient Paths')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# 4. Condition Number (numerical stability indicator)
from numpy.linalg import cond

condition_number = cond(X.corr())
print(f"\nCondition Number: {condition_number:.2f}")
print("Interpretation:")
print("< 30: No multicollinearity")
print("30-100: Moderate multicollinearity")
print("> 100: Severe multicollinearity")
```

## Day 7: Statistical Tests

### T-test

Used to compare means between two groups.

**Types:**
1. One-sample t-test: Compare sample mean to population mean
2. Independent samples t-test: Compare means of two independent groups
3. Paired samples t-test: Compare means of paired observations

**Formula:**
```
t = (x̄₁ - x̄₂) / √(s²/n₁ + s²/n₂)
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 1. One-sample t-test
# H0: μ = μ0
# H1: μ ≠ μ0

# Example: Testing if average height is 170 cm
np.random.seed(42)
heights = np.random.normal(172, 8, 100)  # True mean = 172
population_mean = 170

t_stat, p_value = stats.ttest_1samp(heights, population_mean)
print("One-sample t-test:")
print(f"Sample mean: {np.mean(heights):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H0: {p_value < 0.05}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution and test
axes[0, 0].hist(heights, bins=20, density=True, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(population_mean, color='red', linestyle='--', 
                   label=f'H0: μ = {population_mean}')
axes[0, 0].axvline(np.mean(heights), color='green', linestyle='--', 
                   label=f'Sample mean = {np.mean(heights):.1f}')
axes[0, 0].set_title(f'One-sample t-test (p = {p_value:.4f})')
axes[0, 0].legend()

# 2. Independent samples t-test
# H0: μ1 = μ2
# H1: μ1 ≠ μ2

# Example: Comparing test scores between two teaching methods
method_A = np.random.normal(75, 10, 50)
method_B = np.random.normal(80, 12, 60)

# Check assumptions
# a) Normality
_, p_norm_A = stats.normaltest(method_A)
_, p_norm_B = stats.normaltest(method_B)
print(f"\nNormality test p-values: A={p_norm_A:.4f}, B={p_norm_B:.4f}")

# b) Equal variances (Levene's test)
_, p_levene = stats.levene(method_A, method_B)
print(f"Levene's test p-value: {p_levene:.4f}")

# Perform t-test
t_stat, p_value = stats.ttest_ind(method_A, method_B, equal_var=True)
print(f"\nIndependent samples t-test:")
print(f"Method A mean: {np.mean(method_A):.2f}")
print(f"Method B mean: {np.mean(method_B):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Effect size (Cohen's d): {(np.mean(method_B) - np.mean(method_A)) / np.sqrt((np.var(method_A) + np.var(method_B)) / 2):.4f}")

# Visualize
axes[0, 1].boxplot([method_A, method_B], labels=['Method A', 'Method B'])
axes[0, 1].set_title(f'Independent t-test (p = {p_value:.4f})')
axes[0, 1].set_ylabel('Test Score')

# 3. Paired samples t-test
# Example: Before and after treatment
before = np.random.normal(120, 15, 30)  # Blood pressure before
after = before - np.random.normal(10, 5, 30)  # Reduction after treatment

t_stat, p_value = stats.ttest_rel(before, after)
differences = before - after

print(f"\nPaired samples t-test:")
print(f"Mean difference: {np.mean(differences):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualize
axes[1, 0].scatter(before, after, alpha=0.6)
axes[1, 0].plot([80, 160], [80, 160], 'r--', label='No change line')
axes[1, 0].set_xlabel('Before Treatment')
axes[1, 0].set_ylabel('After Treatment')
axes[1, 0].set_title(f'Paired t-test (p = {p_value:.4f})')
axes[1, 0].legend()

# Power analysis
from statsmodels.stats.power import ttest_power

effect_size = 0.5  # Medium effect
alpha = 0.05
power = 0.8
sample_size = 30

# Calculate power for given sample size
calculated_power = ttest_power(effect_size, sample_size, alpha)

# Calculate required sample size for desired power
from statsmodels.stats.power import tt_ind_solve_power
required_n = tt_ind_solve_power(effect_size=effect_size, alpha=alpha, 
                                power=power, ratio=1)

axes[1, 1].text(0.1, 0.8, f"Power Analysis:", fontsize=14, fontweight='bold')
axes[1, 1].text(0.1, 0.6, f"Effect size (d): {effect_size}")
axes[1, 1].text(0.1, 0.5, f"Sample size: {sample_size}")
axes[1, 1].text(0.1, 0.4, f"Alpha: {alpha}")
axes[1, 1].text(0.1, 0.3, f"Calculated power: {calculated_power:.3f}")
axes[1, 1].text(0.1, 0.1, f"Required n for power={power}: {required_n:.0f}")
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
```

### Z-test

Used when population standard deviation is known or sample size is large (n > 30).

```python
# Z-test implementation
def z_test(sample_mean, population_mean, population_std, n):
    """Perform one-sample z-test"""
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return z_score, p_value

# Example: Quality control in manufacturing
# Population parameters (known from historical data)
pop_mean = 100  # Target weight in grams
pop_std = 5     # Known population standard deviation

# Sample data
sample_weights = np.random.normal(101.5, 5, 100)
sample_mean = np.mean(sample_weights)
n = len(sample_weights)

z_score, p_value = z_test(sample_mean, pop_mean, pop_std, n)

print("Z-test Results:")
print(f"Population mean (μ0): {pop_mean}")
print(f"Sample mean (x̄): {sample_mean:.2f}")
print(f"Z-score: {z_score:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Sample distribution
ax1.hist(sample_weights, bins=20, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(85, 115, 100)
ax1.plot(x, stats.norm.pdf(x, pop_mean, pop_std), 'r-', linewidth=2, 
         label=f'Population N({pop_mean}, {pop_std}²)')
ax1.axvline(sample_mean, color='green', linestyle='--', linewidth=2,
            label=f'Sample mean = {sample_mean:.2f}')
ax1.set_xlabel('Weight (g)')
ax1.set_ylabel('Density')
ax1.set_title('Z-test: Sample vs Population')
ax1.legend()

# Z-distribution
z_range = np.linspace(-4, 4, 100)
ax2.plot(z_range, stats.norm.pdf(z_range), 'b-', linewidth=2)
ax2.fill_between(z_range, stats.norm.pdf(z_range), 
                 where=(np.abs(z_range) > abs(z_score)), alpha=0.3, color='red',
                 label=f'p-value = {p_value:.4f}')
ax2.axvline(z_score, color='green', linestyle='--', linewidth=2,
            label=f'z = {z_score:.3f}')
ax2.axvline(-z_score, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Z-score')
ax2.set_ylabel('Density')
ax2.set_title('Standard Normal Distribution')
ax2.legend()

plt.tight_layout()
plt.show()
```

### Chi-square Test

Used for testing relationships between categorical variables.

```python
# Chi-square test of independence
# Example: Is there a relationship between gender and product preference?

# Create contingency table
data = {
    'Product A': [45, 35],  # [Male, Female]
    'Product B': [30, 40],
    'Product C': [25, 25]
}

contingency_table = pd.DataFrame(data, index=['Male', 'Female'])
print("Contingency Table:")
print(contingency_table)

# Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square test results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Significant at α=0.05: {p_value < 0.05}")

# Expected frequencies
expected_df = pd.DataFrame(expected, 
                          index=contingency_table.index,
                          columns=contingency_table.columns)
print("\nExpected frequencies:")
print(expected_df.round(2))

# Calculate effect size (Cramér's V)
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"\nCramér's V: {cramers_v:.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Observed frequencies
contingency_table.plot(kind='bar', ax=axes[0])
axes[0].set_title('Observed Frequencies')
axes[0].set_ylabel('Count')
axes[0].legend(title='Product')

# Expected frequencies
expected_df.plot(kind='bar', ax=axes[1])
axes[1].set_title('Expected Frequencies (under H0)')
axes[1].set_ylabel('Count')
axes[1].legend(title='Product')

# Residuals
residuals = (contingency_table - expected_df) / np.sqrt(expected_df)
sns.heatmap(residuals, annot=True, cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title('Standardized Residuals')

plt.tight_layout()
plt.show()

# Chi-square goodness of fit test
# Example: Testing if a die is fair
observed_freq = np.array([8, 12, 10, 15, 13, 12])  # Frequencies for faces 1-6
expected_freq = np.array([10, 10, 10, 10, 10, 10])  # Expected for fair die
# Note: Total rolls = 70

chi2, p_value = stats.chisquare(observed_freq, expected_freq)

print(f"\nChi-square goodness of fit test:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Fair die? {p_value > 0.05}")
```

### ANOVA (Analysis of Variance)

Used to compare means across multiple groups.

```python
# One-way ANOVA
# Example: Comparing effectiveness of 4 different fertilizers

# Generate data
np.random.seed(42)
fertilizer_A = np.random.normal(25, 3, 30)  # Plant height in cm
fertilizer_B = np.random.normal(28, 3, 30)
fertilizer_C = np.random.normal(27, 3, 30)
fertilizer_D = np.random.normal(30, 3, 30)

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(fertilizer_A, fertilizer_B, 
                                 fertilizer_C, fertilizer_D)

print("One-way ANOVA:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant difference: {p_value < 0.05}")

# Create dataframe for visualization
data_list = []
for i, (data, label) in enumerate([(fertilizer_A, 'A'), 
                                   (fertilizer_B, 'B'),
                                   (fertilizer_C, 'C'), 
                                   (fertilizer_D, 'D')]):
    for value in data:
        data_list.append({'Fertilizer': label, 'Height': value})

df_anova = pd.DataFrame(data_list)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot
df_anova.boxplot(column='Height', by='Fertilizer', ax=axes[0, 0])
axes[0, 0].set_title('Plant Height by Fertilizer Type')
axes[0, 0].set_xlabel('Fertilizer')
axes[0, 0].set_ylabel('Height (cm)')

# Violin plot
sns.violinplot(data=df_anova, x='Fertilizer', y='Height', ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Plant Heights')

# Post-hoc analysis (Tukey's HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_results = pairwise_tukeyhsd(df_anova['Height'], 
                                  df_anova['Fertilizer'], 
                                  alpha=0.05)

# Convert to dataframe for easier viewing
tukey_df = pd.DataFrame(data=tukey_results.summary().data[1:], 
                       columns=tukey_results.summary().data[0])

# Plot Tukey results
axes[1, 0].text(0.1, 0.9, "Tukey's HSD Post-hoc Test:", 
                fontsize=12, fontweight='bold', transform=axes[1, 0].transAxes)

y_pos = 0.7
for _, row in tukey_df.iterrows():
    text = f"{row['group1']} vs {row['group2']}: "
    text += f"diff={float(row['meandiff']):.2f}, "
    text += f"p={float(row['p-adj']):.4f}"
    if float(row['p-adj']) < 0.05:
        text += " *"
    axes[1, 0].text(0.1, y_pos, text, fontsize=10, 
                    transform=axes[1, 0].transAxes)
    y_pos -= 0.1

axes[1, 0].axis('off')

# Residual analysis
# Fit ANOVA model
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('Height ~ C(Fertilizer)', data=df_anova).fit()
residuals = model.resid

# Q-Q plot for normality check
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

# Two-way ANOVA
# Example: Effect of fertilizer and watering frequency on plant growth

# Generate data with two factors
data_2way = []
fertilizers = ['A', 'B', 'C']
watering = ['Daily', 'Weekly']

# Interaction effect included
np.random.seed(42)
for fert in fertilizers:
    for water in watering:
        if fert == 'A' and water == 'Daily':
            heights = np.random.normal(30, 3, 20)  # Best combination
        elif fert == 'C' and water == 'Weekly':
            heights = np.random.normal(20, 3, 20)  # Worst combination
        else:
            heights = np.random.normal(25, 3, 20)  # Medium
        
        for h in heights:
            data_2way.append({
                'Fertilizer': fert,
                'Watering': water,
                'Height': h
            })

df_2way = pd.DataFrame(data_2way)

# Perform two-way ANOVA
model_2way = ols('Height ~ C(Fertilizer) + C(Watering) + C(Fertilizer):C(Watering)', 
                 data=df_2way).fit()
anova_table = anova_lm(model_2way, typ=2)

print("\nTwo-way ANOVA results:")
print(anova_table)

# Interaction plot
fig, ax = plt.subplots(figsize=(8, 6))

for water in watering:
    means = []
    for fert in fertilizers:
        mean_height = df_2way[(df_2way['Fertilizer'] == fert) & 
                              (df_2way['Watering'] == water)]['Height'].mean()
        means.append(mean_height)
    ax.plot(fertilizers, means, marker='o', label=water, linewidth=2, markersize=8)

ax.set_xlabel('Fertilizer Type')
ax.set_ylabel('Mean Height (cm)')
ax.set_title('Interaction Plot: Fertilizer × Watering Frequency')
ax.legend(title='Watering')
ax.grid(True, alpha=0.3)
plt.show()
```

### Summary of Statistical Tests

```python
# Create a summary reference table
test_summary = pd.DataFrame({
    'Test': ['One-sample t-test', 'Independent t-test', 'Paired t-test', 
             'Z-test', 'Chi-square independence', 'Chi-square goodness of fit',
             'One-way ANOVA', 'Two-way ANOVA'],
    'Use Case': ['Compare sample mean to population mean',
                 'Compare means of two independent groups',
                 'Compare paired observations',
                 'Compare sample to population (known σ)',
                 'Test association between categorical variables',
                 'Test if sample follows expected distribution',
                 'Compare means across 3+ groups',
                 'Test effects of two factors'],
    'Assumptions': ['Normal distribution, Random sampling',
                    'Normal distributions, Equal variances, Independence',
                    'Normal distribution of differences',
                    'Known population σ, Large sample or normal distribution',
                    'Expected frequency ≥ 5 in each cell',
                    'Expected frequency ≥ 5 in each category',
                    'Normal distributions, Equal variances, Independence',
                    'Same as one-way ANOVA + no interaction (if testing)'],
    'H0': ['μ = μ₀',
           'μ₁ = μ₂',
           'μd = 0',
           'μ = μ₀',
           'Variables are independent',
           'Data follows expected distribution',
           'μ₁ = μ₂ = ... = μₖ',
           'No main effects or interactions']
})

print("Statistical Tests Summary:")
print(test_summary.to_string(index=False))

# Effect size interpretation
effect_sizes = pd.DataFrame({
    'Measure': ["Cohen's d", "Pearson's r", "Cramér's V", "Eta squared (η²)"],
    'Small': [0.2, 0.1, 0.1, 0.01],
    'Medium': [0.5, 0.3, 0.3, 0.06],
    'Large': [0.8, 0.5, 0.5, 0.14]
})

print("\n\nEffect Size Guidelines:")
print(effect_sizes.to_string(index=False))
```

---

This completes Week 1 of your ML interview preparation. The content covers:

1. **Day 1**: NumPy (arrays, operations, broadcasting) and Pandas (Series, DataFrames, data manipulation)
2. **Day 2**: Matplotlib and Seaborn for data visualization
3. **Day 3**: Linear algebra fundamentals including matrix operations, eigenvalues, and eigenvectors
4. **Day 4**: Probability theory and KL-Divergence with practical applications
5. **Day 5**: Statistical measures (mean, median, mode, variance, standard deviation, covariance)
6. **Day 6**: Correlation analysis and multicollinearity detection/handling
7. **Day 7**: Statistical hypothesis testing (t-test, z-test, chi-square, ANOVA)

Each topic includes mathematical formulations, detailed code implementations, visualizations, and practical examples relevant to machine learning applications.