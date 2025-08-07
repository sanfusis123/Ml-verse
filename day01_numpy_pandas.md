# Day 1: NumPy and Pandas Fundamentals

## 📚 Topics
- NumPy arrays and operations
- Pandas DataFrames and indexing
- Data manipulation techniques

---

## 1. NumPy - Numerical Python

### 📖 Core Concepts

#### What is NumPy?
NumPy is the fundamental package for scientific computing in Python. It provides:
- Powerful N-dimensional array object
- Broadcasting functions
- Tools for integrating C/C++ and Fortran code
- Linear algebra, Fourier transform, and random number capabilities

### 🔢 Mathematical Foundation

#### Arrays vs Lists
- **Python List**: `O(n)` for element-wise operations
- **NumPy Array**: `O(1)` for vectorized operations (due to contiguous memory)

#### Memory Layout
```
List: [ptr] -> [obj1] [ptr] -> [obj2] [ptr] -> [obj3]
Array: [1][2][3][4][5] (contiguous block)
```

### 💻 Advanced NumPy Code

```python
import numpy as np

# 1. Array Creation Methods
print("=== Array Creation ===")
# From lists
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Using built-in functions
zeros = np.zeros((3, 4))  # 3x4 matrix of zeros
ones = np.ones((2, 3, 4))  # 2x3x4 tensor of ones
identity = np.eye(4)  # 4x4 identity matrix
random_uniform = np.random.rand(3, 3)  # Uniform [0,1)
random_normal = np.random.randn(3, 3)  # Standard normal

# Sequences
arange = np.arange(0, 10, 0.5)  # Start, stop, step
linspace = np.linspace(0, 1, 11)  # 11 points from 0 to 1

print(f"Identity Matrix:\n{identity}")

# 2. Array Properties
print("\n=== Array Properties ===")
arr = np.random.randn(3, 4, 5)
print(f"Shape: {arr.shape}")
print(f"Dimensions: {arr.ndim}")
print(f"Size: {arr.size}")
print(f"Data type: {arr.dtype}")
print(f"Memory usage: {arr.nbytes} bytes")

# 3. Indexing and Slicing
print("\n=== Advanced Indexing ===")
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Basic slicing
print(f"First row: {arr_2d[0, :]}")
print(f"Last column: {arr_2d[:, -1]}")
print(f"Sub-matrix: \n{arr_2d[1:, 2:]}")

# Boolean indexing
mask = arr_2d > 6
print(f"Elements > 6: {arr_2d[mask]}")

# Fancy indexing
rows = np.array([0, 2])
cols = np.array([1, 3])
print(f"Fancy indexing result: {arr_2d[rows, cols]}")

# 4. Broadcasting
print("\n=== Broadcasting ===")
# Broadcasting allows operations on arrays of different shapes
a = np.array([[1, 2, 3]])  # Shape: (1, 3)
b = np.array([[1], [2], [3]])  # Shape: (3, 1)
c = a + b  # Result shape: (3, 3)
print(f"Broadcasting result:\n{c}")

# 5. Vectorized Operations
print("\n=== Vectorized Operations ===")
arr = np.array([1, 2, 3, 4, 5])

# Element-wise operations
squared = arr ** 2
sqrt = np.sqrt(arr)
exp = np.exp(arr)
log = np.log(arr)

print(f"Original: {arr}")
print(f"Squared: {squared}")
print(f"Square root: {sqrt}")

# 6. Linear Algebra Operations
print("\n=== Linear Algebra ===")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
dot_product = np.dot(A, B)  # or A @ B
print(f"Matrix multiplication:\n{dot_product}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Matrix operations
det = np.linalg.det(A)
inv = np.linalg.inv(A)
trace = np.trace(A)
print(f"Determinant: {det}")
print(f"Trace: {trace}")

# 7. Statistical Operations
print("\n=== Statistical Operations ===")
data = np.random.randn(1000)
print(f"Mean: {np.mean(data):.4f}")
print(f"Std: {np.std(data):.4f}")
print(f"Variance: {np.var(data):.4f}")
print(f"Min/Max: {np.min(data):.4f} / {np.max(data):.4f}")
print(f"25th, 50th, 75th percentiles: {np.percentile(data, [25, 50, 75])}")

# 8. Array Manipulation
print("\n=== Array Manipulation ===")
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
transposed = reshaped.T
flattened = reshaped.flatten()
raveled = reshaped.ravel()  # Returns view if possible

print(f"Original shape: {arr.shape}")
print(f"Reshaped to 3x4:\n{reshaped}")
print(f"Transposed:\n{transposed}")

# Concatenation and splitting
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
v_stack = np.vstack([a, b])
h_stack = np.hstack([a, b])
print(f"Vertical stack:\n{v_stack}")
print(f"Horizontal stack:\n{h_stack}")
```

---

## 2. Pandas - Data Analysis Library

### 📖 Core Concepts

#### What is Pandas?
Pandas provides:
- DataFrame: 2D labeled data structure
- Series: 1D labeled array
- Tools for reading/writing data
- Data alignment and integrated handling of missing data
- Reshaping and pivoting of datasets
- Label-based slicing, fancy indexing

### 🔢 Mathematical Foundation

#### Index Alignment
Pandas automatically aligns data based on index labels:
```
Series A: [1, 2, 3] with index [0, 1, 2]
Series B: [4, 5, 6] with index [1, 2, 3]
A + B: [NaN, 6, 8, NaN] with index [0, 1, 2, 3]
```

### 💻 Advanced Pandas Code

```python
import pandas as pd
import numpy as np

# 1. Series Creation and Operations
print("=== Pandas Series ===")
# Creating Series
s1 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series({'a': 10, 'b': 20, 'c': 30, 'd': 40})

print(f"Series 1:\n{s1}")
print(f"\nSeries 2:\n{s2}")

# Series operations with alignment
print(f"\nAligned addition:\n{s1 + s2}")

# 2. DataFrame Creation
print("\n=== DataFrame Creation ===")
# From dictionary
df_dict = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}, index=['row1', 'row2', 'row3', 'row4'])

# From numpy array
df_array = pd.DataFrame(
    np.random.randn(5, 3),
    columns=['col1', 'col2', 'col3'],
    index=pd.date_range('2024-01-01', periods=5)
)

print(f"DataFrame from dict:\n{df_dict}")
print(f"\nDataFrame from array:\n{df_array}")

# 3. Data Selection and Indexing
print("\n=== Advanced Indexing ===")
df = pd.DataFrame(np.random.randn(6, 4),
                  index=pd.date_range('2024-01-01', periods=6),
                  columns=['A', 'B', 'C', 'D'])

# Selection by label
print(f"Select column 'A':\n{df['A']}")
print(f"\nSelect multiple columns:\n{df[['A', 'C']]}")

# .loc - label based
print(f"\n.loc['2024-01-02']:\n{df.loc['2024-01-02']}")
print(f"\n.loc with slicing:\n{df.loc['2024-01-02':'2024-01-04', ['A', 'B']]}")

# .iloc - integer based
print(f"\n.iloc[1:3, 0:2]:\n{df.iloc[1:3, 0:2]}")

# Boolean indexing
print(f"\nRows where A > 0:\n{df[df['A'] > 0]}")

# 4. Data Manipulation
print("\n=== Data Manipulation ===")
# Adding new columns
df['E'] = df['A'] + df['B']
df['F'] = df['A'].apply(lambda x: x**2)

# Conditional column creation
df['Category'] = np.where(df['A'] > 0, 'Positive', 'Negative')

print(f"DataFrame with new columns:\n{df}")

# 5. Handling Missing Data
print("\n=== Missing Data ===")
df_missing = df.copy()
df_missing.loc['2024-01-02':'2024-01-03', 'B'] = np.nan
df_missing.loc['2024-01-04', 'C'] = np.nan

print(f"DataFrame with NaN:\n{df_missing}")
print(f"\nDropping NaN (rows):\n{df_missing.dropna()}")
print(f"\nFilling NaN with 0:\n{df_missing.fillna(0)}")
print(f"\nForward fill:\n{df_missing.fillna(method='ffill')}")

# 6. GroupBy Operations
print("\n=== GroupBy Operations ===")
df_group = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Values': [10, 20, 30, 40, 50, 60],
    'Count': [1, 2, 3, 4, 5, 6]
})

grouped = df_group.groupby('Category')
print(f"Group means:\n{grouped.mean()}")
print(f"\nGroup sums:\n{grouped.sum()}")
print(f"\nMultiple aggregations:\n{grouped.agg(['mean', 'std', 'count'])}")

# 7. Merge, Join, and Concatenate
print("\n=== Merge and Join ===")
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# Different merge types
inner_merge = pd.merge(df1, df2, on='key', how='inner')
outer_merge = pd.merge(df1, df2, on='key', how='outer')
left_merge = pd.merge(df1, df2, on='key', how='left')

print(f"Inner merge:\n{inner_merge}")
print(f"\nOuter merge:\n{outer_merge}")
print(f"\nLeft merge:\n{left_merge}")

# 8. Pivot Tables
print("\n=== Pivot Tables ===")
df_pivot = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=9),
    'Category': ['A', 'B', 'C'] * 3,
    'Values': np.random.randint(10, 100, 9)
})

pivot_table = df_pivot.pivot_table(
    values='Values',
    index='Date',
    columns='Category',
    aggfunc='sum'
)
print(f"Pivot table:\n{pivot_table}")

# 9. Time Series Operations
print("\n=== Time Series ===")
ts = pd.Series(np.random.randn(365),
               index=pd.date_range('2024-01-01', periods=365))

# Resampling
monthly_mean = ts.resample('M').mean()
weekly_sum = ts.resample('W').sum()

print(f"Monthly means:\n{monthly_mean.head()}")
print(f"\nRolling window (7-day mean):\n{ts.rolling(window=7).mean().head(10)}")

# 10. Advanced String Operations
print("\n=== String Operations ===")
df_str = pd.DataFrame({
    'Names': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'Emails': ['john@email.com', 'jane@company.com', 'bob@org.net']
})

df_str['First_Name'] = df_str['Names'].str.split().str[0]
df_str['Last_Name'] = df_str['Names'].str.split().str[1]
df_str['Domain'] = df_str['Emails'].str.split('@').str[1]

print(f"String operations result:\n{df_str}")
```

## 🎯 Interview Questions

### NumPy Questions
1. **Q: What's the difference between `np.array([1,2,3])` and `list([1,2,3])`?**
   - A: NumPy arrays have fixed type, contiguous memory, support vectorized operations, and are more memory efficient.

2. **Q: Explain broadcasting with an example.**
   - A: Broadcasting allows operations between arrays of different shapes by extending smaller dimensions.
   ```python
   a = np.array([1, 2, 3])  # Shape: (3,)
   b = np.array([[1], [2]])  # Shape: (2, 1)
   # a + b broadcasts to (2, 3)
   ```

3. **Q: What's the difference between `copy()` and `view()` in NumPy?**
   - A: `view()` creates a shallow copy sharing the same data, `copy()` creates a deep copy with independent data.

### Pandas Questions
1. **Q: Difference between `loc` and `iloc`?**
   - A: `loc` is label-based indexing, `iloc` is integer position-based indexing.

2. **Q: How do you handle missing data in Pandas?**
   - A: Use `dropna()`, `fillna()`, `interpolate()`, or `replace()` methods.

3. **Q: Explain the difference between `merge`, `join`, and `concat`.**
   - A: `merge` is database-style join, `join` is index-based merge, `concat` stacks DataFrames along an axis.

## 📝 Practice Exercises

1. Create a NumPy array of shape (100, 100) with random values and find:
   - Elements greater than 0.5
   - Row and column means
   - Correlation matrix

2. Load a CSV file into Pandas and:
   - Handle missing values
   - Create pivot tables
   - Perform group-by analysis

3. Implement a function that uses NumPy broadcasting to normalize a matrix by subtracting mean and dividing by std along specified axis.

## 🔗 Key Takeaways
- NumPy provides efficient array operations through vectorization
- Pandas excels at structured data manipulation
- Understanding indexing and broadcasting is crucial
- Both libraries form the foundation of Python data science stack