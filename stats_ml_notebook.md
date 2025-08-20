# Statistics for Machine Learning - Complete Learning Notebook

## Table of Contents
1. [Introduction to Statistics](#1-introduction-to-statistics)
2. [Data Collection and Sampling](#2-data-collection-and-sampling)
3. [Variables and Data Visualization](#3-variables-and-data-visualization)
4. [Descriptive Statistics](#4-descriptive-statistics)
5. [Probability Fundamentals](#5-probability-fundamentals)
6. [Probability Distributions](#6-probability-distributions)
7. [Sampling Distributions](#7-sampling-distributions)
8. [Statistical Inference](#8-statistical-inference)
9. [Multivariate Statistics](#9-multivariate-statistics)
10. [Interview Questions](#10-interview-questions)

---

## 1. Introduction to Statistics

### What is Statistics?
Statistics is the science of collecting, analyzing, interpreting, and presenting data. In machine learning, statistics provides the mathematical foundation for understanding data patterns and making predictions.

### Types of Statistics

#### Descriptive Statistics
- **Purpose**: Summarize and describe data
- **Examples**: Mean, median, standard deviation, histograms
- **Use in ML**: Data exploration, feature understanding

#### Inferential Statistics
- **Purpose**: Make predictions and draw conclusions about populations
- **Examples**: Hypothesis testing, confidence intervals, regression
- **Use in ML**: Model validation, significance testing, uncertainty quantification

### Example
```python
import numpy as np
import matplotlib.pyplot as plt

# Descriptive: Summarizing student scores
scores = [85, 92, 78, 96, 88, 85, 90, 94, 87, 91]
print(f"Mean: {np.mean(scores)}")
print(f"Median: {np.median(scores)}")
print(f"Std Dev: {np.std(scores)}")

# Inferential: Can we conclude the class average > 80?
```

---

## 2. Data Collection and Sampling

### Population vs Sample

#### Population
- **Definition**: Complete set of all items of interest
- **Example**: All customers of a company
- **Challenge**: Often impossible/expensive to study entirely

#### Sample
- **Definition**: Subset of the population
- **Example**: 1000 randomly selected customers
- **Purpose**: Make inferences about the population

### Sampling Techniques

#### 1. Simple Random Sampling
- Every member has equal chance of selection
- **Example**: Using random number generator to select customers

#### 2. Stratified Sampling
- Population divided into strata, random sampling within each
- **Example**: Sampling equal numbers from each age group

#### 3. Systematic Sampling
- Select every kth item
- **Example**: Every 10th customer from a list

#### 4. Cluster Sampling
- Population divided into clusters, randomly select clusters
- **Example**: Randomly select cities, survey all residents

#### 5. Convenience Sampling
- Non-random, based on availability
- **Example**: Surveying people at a mall

### Example
```python
import random
import pandas as pd

# Population of 10,000 customers
population = list(range(1, 10001))

# Simple Random Sampling
sample_size = 100
simple_random_sample = random.sample(population, sample_size)

# Systematic Sampling
k = len(population) // sample_size  # Every kth element
systematic_sample = population[::k]
```

---

## 3. Variables and Data Visualization

### Types of Variables

#### Quantitative (Numerical)
1. **Continuous**: Can take any value within a range
   - Examples: Height, weight, temperature
   - Graphs: Histogram, box plot, density plot

2. **Discrete**: Countable distinct values
   - Examples: Number of children, goals scored
   - Graphs: Bar chart, histogram

#### Qualitative (Categorical)
1. **Nominal**: No natural order
   - Examples: Color, gender, city
   - Graphs: Bar chart, pie chart

2. **Ordinal**: Natural order exists
   - Examples: Education level, satisfaction rating
   - Graphs: Bar chart, stacked bar chart

### Choosing Graphs Based on Variable Types

#### For Single Variables

| Variable Type | Graph Type | Purpose & Information Gained | When to Use |
|---------------|------------|------------------------------|-------------|
| **Continuous** | Histogram | Shows distribution shape, frequency of ranges, identifies skewness/outliers | Understanding data distribution, checking normality |
| | Box Plot | Shows quartiles, median, outliers, spread | Comparing groups, outlier detection, understanding spread |
| | Density Plot | Smooth distribution curve, probability density | When you want smooth representation of distribution |
| | Violin Plot | Combines box plot with density, shows distribution shape | Comparing distributions across groups |
| **Discrete** | Bar Chart | Frequency of each value, most/least common values | Categorical-like discrete data (small number of values) |
| | Histogram | Distribution pattern for large range of discrete values | Many possible discrete values |
| | Dot Plot | Exact frequency of each value, good for small datasets | Small datasets, exact counts important |
| **Nominal** | Bar Chart | Frequency/proportion of each category | Comparing category sizes, finding most common |
| | Pie Chart | Proportion of whole for each category | When showing parts of a whole (limited categories) |
| | Stacked Bar Chart | Comparing categories across different groups | Multiple grouping variables |
| **Ordinal** | Bar Chart | Frequency while maintaining order | Shows ranking and frequency together |
| | Stacked Bar Chart | Ordered categories across different groups | Comparing ordered categories by groups |

#### For Multiple Variables

| Variable Combination | Graph Type | Purpose & Information Gained | When to Use |
|---------------------|------------|------------------------------|-------------|
| **Continuous vs Continuous** | Scatter Plot | Linear/non-linear relationships, correlation strength, outliers | Exploring relationships, correlation analysis |
| | Line Plot | Trend over time/ordered variable | Time series, sequential data |
| | Heatmap (correlation) | Correlation matrix visualization | Multiple variable relationships |
| **Categorical vs Continuous** | Box Plot | Distribution differences across categories | Comparing groups, identifying differences |
| | Violin Plot | Distribution shape differences across groups | When distribution shape matters |
| | Bar Plot (with error bars) | Mean/median differences with uncertainty | Comparing averages across groups |
| **Categorical vs Categorical** | Stacked Bar Chart | Relationship between two categorical variables | Cross-tabulation visualization |
| | Mosaic Plot | Proportional relationships, independence testing | Complex categorical relationships |
| | Heatmap (contingency table) | Frequency patterns across categories | Large contingency tables |

#### Detailed Graph Purposes

**Histogram**
- **Information**: Frequency distribution, central tendency, spread, skewness
- **Identifies**: Outliers, multimodal distributions, normality
- **Best for**: Understanding single continuous variable distribution

**Box Plot**
- **Information**: Median, quartiles (Q1, Q3), IQR, outliers, symmetry
- **Identifies**: Extreme values, data spread, group comparisons
- **Best for**: Comparing distributions across groups, outlier detection

**Scatter Plot**
- **Information**: Relationship strength, direction, linearity, outliers
- **Identifies**: Correlation patterns, clusters, influential points
- **Best for**: Exploring relationships between two continuous variables

**Bar Chart**
- **Information**: Category frequencies, relative sizes, rankings
- **Identifies**: Most/least common categories, patterns
- **Best for**: Comparing categorical data, showing counts/proportions

**Density Plot**
- **Information**: Probability density, distribution shape, modality
- **Identifies**: Multiple peaks, distribution smoothness
- **Best for**: Smooth representation of continuous distributions

**Violin Plot**
- **Information**: Distribution shape + quartile information
- **Identifies**: Distribution differences, skewness across groups
- **Best for**: Comparing distribution shapes across categories

### Example with Practical Interpretations

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data
np.random.seed(42)
ages = np.random.normal(35, 10, 1000)
departments = np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 1000, p=[0.15, 0.4, 0.3, 0.15])
salaries = 30000 + ages * 1000 + np.random.normal(0, 5000, 1000)
satisfaction = np.random.choice(['Low', 'Medium', 'High'], 1000, p=[0.2, 0.5, 0.3])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Histogram - Continuous variable (Ages)
axes[0,0].hist(ages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Age Distribution (Histogram)')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Frequency')
# Information gained: Distribution is roughly normal, centered around 35, 
# range from ~5 to 65, no obvious outliers

# 2. Box Plot - Comparing continuous across categories
salary_by_dept = [salaries[departments == dept] for dept in ['HR', 'IT', 'Sales', 'Marketing']]
axes[0,1].boxplot(salary_by_dept, labels=['HR', 'IT', 'Sales', 'Marketing'])
axes[0,1].set_title('Salary by Department (Box Plot)')
axes[0,1].set_ylabel('Salary')
# Information gained: IT has highest median salary, HR has most outliers,
# Sales has largest spread, Marketing has lowest median

# 3. Scatter Plot - Two continuous variables
axes[0,2].scatter(ages, salaries, alpha=0.6, color='green')
axes[0,2].set_title('Age vs Salary (Scatter Plot)')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Salary')
# Information gained: Positive correlation between age and salary,
# linear relationship, some scatter around trend

# 4. Bar Chart - Categorical variable
dept_counts = [sum(departments == dept) for dept in ['HR', 'IT', 'Sales', 'Marketing']]
axes[1,0].bar(['HR', 'IT', 'Sales', 'Marketing'], dept_counts, color=['red', 'blue', 'orange', 'purple'])
axes[1,0].set_title('Department Distribution (Bar Chart)')
axes[1,0].set_ylabel('Count')
# Information gained: IT is largest department, HR and Marketing are smallest,
# relatively balanced distribution

# 5. Violin Plot - Distribution shape across groups
salary_data = [salaries[departments == dept] for dept in ['HR', 'IT', 'Sales', 'Marketing']]
axes[1,1].violinplot(salary_data, positions=[1,2,3,4], showmedians=True)
axes[1,1].set_xticks([1,2,3,4])
axes[1,1].set_xticklabels(['HR', 'IT', 'Sales', 'Marketing'])
axes[1,1].set_title('Salary Distribution by Department (Violin Plot)')
axes[1,1].set_ylabel('Salary')
# Information gained: IT has bimodal salary distribution, HR is right-skewed,
# Sales has normal distribution, Marketing is left-skewed

# 6. Stacked Bar Chart - Two categorical variables
satisfaction_by_dept = {}
for dept in ['HR', 'IT', 'Sales', 'Marketing']:
    dept_mask = departments == dept
    satisfaction_by_dept[dept] = [
        sum((departments == dept) & (satisfaction == 'Low')),
        sum((departments == dept) & (satisfaction == 'Medium')),
        sum((departments == dept) & (satisfaction == 'High'))
    ]

x = np.arange(4)
width = 0.8
axes[1,2].bar(x, [satisfaction_by_dept[dept][0] for dept in ['HR', 'IT', 'Sales', 'Marketing']], 
              width, label='Low', color='red', alpha=0.7)
axes[1,2].bar(x, [satisfaction_by_dept[dept][1] for dept in ['HR', 'IT', 'Sales', 'Marketing']], 
              width, bottom=[satisfaction_by_dept[dept][0] for dept in ['HR', 'IT', 'Sales', 'Marketing']], 
              label='Medium', color='yellow', alpha=0.7)
axes[1,2].bar(x, [satisfaction_by_dept[dept][2] for dept in ['HR', 'IT', 'Sales', 'Marketing']], 
              width, bottom=[satisfaction_by_dept[dept][0] + satisfaction_by_dept[dept][1] 
                            for dept in ['HR', 'IT', 'Sales', 'Marketing']], 
              label='High', color='green', alpha=0.7)

axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(['HR', 'IT', 'Sales', 'Marketing'])
axes[1,2].set_title('Satisfaction by Department (Stacked Bar)')
axes[1,2].legend()
# Information gained: IT has highest proportion of high satisfaction,
# HR has most low satisfaction, proportional differences across departments

plt.tight_layout()
plt.show()

# Summary of insights from each graph type:
print("INSIGHTS FROM DIFFERENT GRAPH TYPES:")
print("="*50)
print("1. Histogram (Age): Normal distribution, mean ≈ 35, range 5-65")
print("2. Box Plot (Salary by Dept): IT highest median, HR most variable")
print("3. Scatter Plot (Age vs Salary): Strong positive correlation")
print("4. Bar Chart (Department): IT largest (40%), HR/Marketing smallest (15% each)")
print("5. Violin Plot (Salary by Dept): Shows distribution shapes, IT bimodal")
print("6. Stacked Bar (Satisfaction by Dept): IT most satisfied, HR least satisfied")
```

---

## 4. Descriptive Statistics

### Measures of Central Tendency

#### Mean (Arithmetic Average)
- **Formula**: μ = (Σx) / n
- **Use**: When data is normally distributed
- **Sensitive to**: Outliers

#### Median
- **Definition**: Middle value when data is ordered
- **Use**: When data is skewed or has outliers
- **Robust to**: Outliers

#### Mode
- **Definition**: Most frequently occurring value
- **Use**: Categorical data or discrete data
- **Can have**: Multiple modes or no mode

### Measures of Dispersion

#### Range
- **Formula**: Max - Min
- **Limitation**: Affected by outliers

#### Variance
- **Population Variance**: σ² = Σ(x - μ)² / N
- **Sample Variance**: s² = Σ(x - x̄)² / (n-1)

#### Standard Deviation
- **Formula**: σ = √variance
- **Interpretation**: Average distance from mean

#### Coefficient of Variation
- **Formula**: CV = (σ/μ) × 100%
- **Use**: Compare variability between different datasets

### Percentiles and Quartiles

#### Percentiles
- **Definition**: Value below which a percentage of data falls
- **Example**: 75th percentile means 75% of data is below this value

#### Quartiles
- **Q1 (25th percentile)**: First quartile
- **Q2 (50th percentile)**: Median
- **Q3 (75th percentile)**: Third quartile
- **IQR**: Q3 - Q1 (Interquartile Range)

#### Five Number Summary
1. Minimum
2. Q1
3. Median (Q2)
4. Q3
5. Maximum

### Outliers Detection
- **IQR Method**: Outliers are values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- **Z-score Method**: |z| > 2 or 3 (depending on threshold)

### Example
```python
data = [12, 15, 18, 20, 22, 25, 28, 30, 35, 100]  # 100 is an outlier

# Central Tendency
mean = np.mean(data)
median = np.median(data)
print(f"Mean: {mean}, Median: {median}")

# Dispersion
std_dev = np.std(data, ddof=1)
variance = np.var(data, ddof=1)
print(f"Std Dev: {std_dev}, Variance: {variance}")

# Quartiles
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")

# Outlier detection
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = [x for x in data if x < lower_bound or x > upper_bound]
print(f"Outliers: {outliers}")
```

### Scaling/Normalization Techniques (For Already Normal Data)

#### Z-score Normalization (Standardization)
- **Formula**: z = (x - μ) / σ
- **Result**: Mean = 0, Std Dev = 1
- **Use**: When data is normally distributed

#### Min-Max Normalization
- **Formula**: x_norm = (x - min) / (max - min)
- **Result**: Values between 0 and 1
- **Use**: When you need bounded values

---

## 4.1. Data Transformations for Non-Normal Data

### Why Transform Data to Normal Distribution?

Many statistical methods and machine learning algorithms assume data follows a normal distribution:
- **Linear Regression**: Residuals should be normally distributed
- **ANOVA**: Groups should be normally distributed
- **t-tests**: Data should be approximately normal
- **Some ML algorithms**: Naive Bayes, Linear Discriminant Analysis

### When Data is NOT Normal:
- **Skewed data**: Long tail on one side
- **Heavy-tailed data**: More extreme values than normal
- **Multi-modal data**: Multiple peaks
- **Bounded data**: Values constrained to certain ranges

---

### Types of Non-Normal Data and Their Transformations

## 1. Right-Skewed Data (Positive Skew) 📈

### Characteristics:
- **Long tail to the right**
- **Mean > Median**
- **Common in**: Income, house prices, website traffic, response times

### Transformations to Use:

#### A) Log Transformation
- **Best for**: Highly right-skewed data
- **Formula**: y = log(x) or y = ln(x)
- **When to use**: When data spans several orders of magnitude
- **Examples**: Income data, population sizes, stock prices

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Generate right-skewed data (exponential-like)
np.random.seed(42)
right_skewed_data = np.random.exponential(2, 1000)

# Apply log transformation
log_transformed = np.log(right_skewed_data)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original data
axes[0,0].hist(right_skewed_data, bins=30, alpha=0.7, color='red')
axes[0,0].set_title('Original Right-Skewed Data')
axes[0,0].axvline(np.mean(right_skewed_data), color='blue', linestyle='--', label=f'Mean: {np.mean(right_skewed_data):.2f}')
axes[0,0].axvline(np.median(right_skewed_data), color='green', linestyle='--', label=f'Median: {np.median(right_skewed_data):.2f}')
axes[0,0].legend()

# Log transformed
axes[0,1].hist(log_transformed, bins=30, alpha=0.7, color='green')
axes[0,1].set_title('After Log Transformation')
axes[0,1].axvline(np.mean(log_transformed), color='blue', linestyle='--', label=f'Mean: {np.mean(log_transformed):.2f}')
axes[0,1].axvline(np.median(log_transformed), color='green', linestyle='--', label=f'Median: {np.median(log_transformed):.2f}')
axes[0,1].legend()

# Q-Q plots to check normality
stats.probplot(right_skewed_data, dist="norm", plot=axes[0,2])
axes[0,2].set_title('Q-Q Plot: Original Data')

print("RIGHT-SKEWED DATA TRANSFORMATION:")
print(f"Original data - Skewness: {stats.skew(right_skewed_data):.3f}")
print(f"Log transformed - Skewness: {stats.skew(log_transformed):.3f}")
print(f"Shapiro-Wilk test (original): p-value = {stats.shapiro(right_skewed_data[:100])[1]:.6f}")
print(f"Shapiro-Wilk test (log): p-value = {stats.shapiro(log_transformed[:100])[1]:.6f}")
```

#### B) Square Root Transformation
- **Best for**: Moderately right-skewed data
- **Formula**: y = √x
- **When to use**: Count data, less extreme skewness
- **Examples**: Number of website visits, count of events

```python
# Generate moderately right-skewed data (Poisson-like)
moderately_skewed = np.random.poisson(3, 1000)
sqrt_transformed = np.sqrt(moderately_skewed)

axes[1,0].hist(moderately_skewed, bins=15, alpha=0.7, color='orange')
axes[1,0].set_title('Moderately Right-Skewed (Poisson)')

axes[1,1].hist(sqrt_transformed, bins=15, alpha=0.7, color='purple')
axes[1,1].set_title('After Square Root Transformation')

print(f"\nMODERATE SKEW TRANSFORMATION:")
print(f"Original Poisson data - Skewness: {stats.skew(moderately_skewed):.3f}")
print(f"Square root transformed - Skewness: {stats.skew(sqrt_transformed):.3f}")
```

#### C) Box-Cox Transformation
- **Best for**: Finding optimal transformation automatically
- **Formula**: y = (x^λ - 1) / λ when λ ≠ 0, y = ln(x) when λ = 0
- **When to use**: When unsure which transformation to use
- **Advantage**: Automatically finds best λ parameter

```python
# Box-Cox transformation
from scipy.stats import boxcox

# Box-Cox requires positive values
positive_data = right_skewed_data + 1  # Add 1 to ensure all positive
boxcox_transformed, lambda_param = boxcox(positive_data)

axes[1,2].hist(boxcox_transformed, bins=30, alpha=0.7, color='brown')
axes[1,2].set_title(f'Box-Cox Transformation (λ={lambda_param:.3f})')

plt.tight_layout()
plt.show()

print(f"Box-Cox optimal λ: {lambda_param:.3f}")
print(f"Box-Cox transformed - Skewness: {stats.skew(boxcox_transformed):.3f}")
```

---

## 2. Left-Skewed Data (Negative Skew) 📉

### Characteristics:
- **Long tail to the left**
- **Mean < Median**
- **Common in**: Test scores (ceiling effect), age at death, product ratings

### Transformations to Use:

#### A) Reflect and Transform Method
- **Step 1**: Reflect data: x_reflected = max(x) + 1 - x
- **Step 2**: Apply right-skew transformation (log, sqrt)
- **Step 3**: Reflect back if needed

```python
# Generate left-skewed data (beta distribution)
left_skewed_data = np.random.beta(5, 2, 1000) * 100  # Scale to 0-100

# Method 1: Reflect and Log Transform
max_val = np.max(left_skewed_data)
reflected = max_val + 1 - left_skewed_data
log_reflected = np.log(reflected)
final_transformed = -(log_reflected - np.max(log_reflected))  # Reflect back

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(left_skewed_data, bins=30, alpha=0.7, color='red')
axes[0].set_title('Original Left-Skewed Data')

axes[1].hist(reflected, bins=30, alpha=0.7, color='blue')
axes[1].set_title('Step 1: Reflected Data')

axes[2].hist(final_transformed, bins=30, alpha=0.7, color='green')
axes[2].set_title('Step 2: Transformed & Reflected Back')

plt.tight_layout()
plt.show()

print("LEFT-SKEWED DATA TRANSFORMATION:")
print(f"Original - Skewness: {stats.skew(left_skewed_data):.3f}")
print(f"Final transformed - Skewness: {stats.skew(final_transformed):.3f}")
```

#### B) Power Transformation (x^n where n > 1)
- **Formula**: y = x^2 or y = x^3
- **When to use**: Mild left skewness

```python
# For mild left skewness
power_transformed = left_skewed_data ** 2

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(left_skewed_data, bins=30, alpha=0.7, color='red')
axes[0].set_title('Original Left-Skewed')

axes[1].hist(power_transformed, bins=30, alpha=0.7, color='blue')
axes[1].set_title('Power Transformation (x²)')

plt.tight_layout()
plt.show()

print(f"Power transformation - Skewness: {stats.skew(power_transformed):.3f}")
```

---

## 3. Heavy-Tailed Data (High Kurtosis) 🎯

### Characteristics:
- **More extreme values than normal distribution**
- **Thick tails, sharp peak**
- **Common in**: Financial returns, error measurements

### Transformations:

#### Winsorization
- **Method**: Cap extreme values at percentiles
- **Formula**: Replace values below 5th percentile and above 95th percentile

```python
# Generate heavy-tailed data
heavy_tailed = np.random.standard_t(3, 1000)  # t-distribution with 3 degrees of freedom

# Winsorization
from scipy.stats.mstats import winsorize
winsorized_data = winsorize(heavy_tailed, limits=[0.05, 0.05])  # 5% from each tail

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(heavy_tailed, bins=50, alpha=0.7, color='red')
axes[0].set_title('Original Heavy-Tailed Data')
axes[0].set_ylim(0, 100)

axes[1].hist(winsorized_data, bins=50, alpha=0.7, color='green')
axes[1].set_title('After Winsorization')
axes[1].set_ylim(0, 100)

# Comparison with normal
normal_data = np.random.normal(0, 1, 1000)
axes[2].hist(normal_data, bins=50, alpha=0.5, color='blue', label='Normal')
axes[2].hist(winsorized_data, bins=50, alpha=0.5, color='green', label='Winsorized')
axes[2].set_title('Comparison with Normal')
axes[2].legend()

plt.tight_layout()
plt.show()

print("HEAVY-TAILED DATA TREATMENT:")
print(f"Original kurtosis: {stats.kurtosis(heavy_tailed):.3f}")
print(f"Winsorized kurtosis: {stats.kurtosis(winsorized_data):.3f}")
print(f"Normal kurtosis: {stats.kurtosis(normal_data):.3f}")
```

---

## 4. Proportion/Percentage Data (Bounded 0-1) 📊

### Characteristics:
- **Values between 0 and 1 (or 0% and 100%)**
- **Often U-shaped or skewed**
- **Common in**: Success rates, percentages, probabilities

### Transformations:

#### Logit Transformation
- **Formula**: y = ln(p/(1-p)) where p is proportion
- **Use**: When data is proportion between 0 and 1

```python
# Generate proportion data (beta distribution)
proportion_data = np.random.beta(2, 5, 1000)  # Skewed towards 0

# Logit transformation
# Add small epsilon to avoid log(0) issues
epsilon = 1e-6
proportion_clean = np.clip(proportion_data, epsilon, 1-epsilon)
logit_transformed = np.log(proportion_clean / (1 - proportion_clean))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(proportion_data, bins=30, alpha=0.7, color='purple')
axes[0].set_title('Original Proportion Data')
axes[0].set_xlabel('Proportion (0-1)')

axes[1].hist(logit_transformed, bins=30, alpha=0.7, color='orange')
axes[1].set_title('After Logit Transformation')
axes[1].set_xlabel('Logit(p)')

plt.tight_layout()
plt.show()

print("PROPORTION DATA TRANSFORMATION:")
print(f"Original - Skewness: {stats.skew(proportion_data):.3f}")
print(f"Logit transformed - Skewness: {stats.skew(logit_transformed):.3f}")
```

#### Arcsine Transformation
- **Formula**: y = arcsin(√p)
- **Use**: Alternative for proportion data

```python
# Arcsine transformation
arcsine_transformed = np.arcsin(np.sqrt(proportion_data))

plt.figure(figsize=(6, 4))
plt.hist(arcsine_transformed, bins=30, alpha=0.7, color='cyan')
plt.title('Arcsine Transformation')
plt.show()

print(f"Arcsine transformed - Skewness: {stats.skew(arcsine_transformed):.3f}")
```

---

## 5. Count Data (Non-negative Integers) 🔢

### Characteristics:
- **Only non-negative integers: 0, 1, 2, 3, ...**
- **Often right-skewed**
- **Common in**: Number of events, counts, frequencies

### Transformations:

#### Square Root Transformation
- **Formula**: y = √(x + 0.5)
- **Use**: Poisson-like count data

```python
# Generate count data
count_data = np.random.poisson(5, 1000)

# Square root transformation (add 0.5 for zero values)
sqrt_count = np.sqrt(count_data + 0.5)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(count_data, bins=20, alpha=0.7, color='red')
axes[0].set_title('Original Count Data (Poisson)')

axes[1].hist(sqrt_count, bins=20, alpha=0.7, color='blue')
axes[1].set_title('Square Root Transformed')

plt.tight_layout()
plt.show()

print("COUNT DATA TRANSFORMATION:")
print(f"Original - Skewness: {stats.skew(count_data):.3f}")
print(f"Square root transformed - Skewness: {stats.skew(sqrt_count):.3f}")
```

---

### How to Choose the Right Transformation

#### Step-by-Step Guide:

```python
def choose_transformation(data):
    """
    Function to suggest appropriate transformation based on data characteristics
    """
    data = np.array(data)
    
    # Calculate statistics
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    print("DATA ANALYSIS:")
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    print(f"Range: [{min_val:.3f}, {max_val:.3f}]")
    print(f"Data type: {type(data[0])}")
    
    print("\nRECOMMENDED TRANSFORMATIONS:")
    
    # Check data range
    if min_val >= 0 and max_val <= 1:
        print("• Data appears to be proportions (0-1)")
        print("  → Try: Logit or Arcsine transformation")
    
    elif min_val >= 0 and all(isinstance(x, (int, np.integer)) for x in data):
        print("• Data appears to be counts (non-negative integers)")
        print("  → Try: Square root transformation")
    
    elif min_val > 0:  # Positive values
        if skewness > 1:
            print("• Highly right-skewed positive data")
            print("  → Try: Log transformation")
        elif 0.5 < skewness <= 1:
            print("• Moderately right-skewed positive data") 
            print("  → Try: Square root transformation")
        elif skewness < -0.5:
            print("• Left-skewed data")
            print("  → Try: Power transformation (x²) or reflect-and-transform")
    
    else:  # Contains negative values
        if abs(kurtosis) > 2:
            print("• Heavy-tailed data with negative values")
            print("  → Try: Winsorization")
        else:
            print("• Data contains negative values")
            print("  → Consider: Standardization or robust scaling")
    
    if abs(kurtosis) > 3:
        print("• High kurtosis detected")
        print("  → Also consider: Winsorization to handle outliers")

# Test the function with different data types
datasets = {
    "Income (right-skewed)": np.random.exponential(50000, 100),
    "Test scores (left-skewed)": 100 - np.random.exponential(10, 100),
    "Proportions": np.random.beta(2, 5, 100),
    "Counts": np.random.poisson(3, 100),
    "Heavy-tailed": np.random.standard_t(3, 100)
}

for name, data in datasets.items():
    print(f"\n{'='*50}")
    print(f"ANALYZING: {name}")
    print('='*50)
    choose_transformation(data)
```

---

### Transformation Decision Tree

```python
print("\nTRANSFORMATION DECISION TREE:")
print("="*40)
print("""
1. CHECK DATA TYPE AND RANGE:
   ├── Proportions (0-1) → Logit or Arcsine
   ├── Counts (0,1,2,3...) → Square root
   ├── Positive only → Continue to step 2
   └── Contains negative → Standardization/Winsorization

2. CHECK SKEWNESS (for positive data):
   ├── Highly right-skewed (>1.0) → Log transformation
   ├── Moderately right-skewed (0.5-1.0) → Square root
   ├── Approximately symmetric (-0.5 to 0.5) → No transformation needed
   └── Left-skewed (<-0.5) → Power transformation or reflect-and-transform

3. CHECK KURTOSIS:
   ├── High kurtosis (>3) → Consider Winsorization
   └── Normal kurtosis → Use skewness-based transformation

4. VERIFY TRANSFORMATION:
   ├── Calculate new skewness and kurtosis
   ├── Visual inspection (histogram, Q-Q plot)
   ├── Statistical tests (Shapiro-Wilk, Anderson-Darling)
   └── If not satisfied, try Box-Cox for automatic optimization
""")
```

### Summary Table

| Data Characteristics | Transformation | Formula | Use Cases |
|---------------------|----------------|---------|-----------|
| **Highly right-skewed, positive** | Log | y = log(x) | Income, prices, population |
| **Moderately right-skewed** | Square root | y = √x | Counts, moderate skew |
| **Left-skewed** | Reflect + Log | y = log(max-x) | Test scores, ratings |
| **Proportions (0-1)** | Logit | y = log(p/(1-p)) | Success rates, percentages |
| **Count data** | Square root | y = √(x+0.5) | Event counts, frequencies |
| **Heavy-tailed** | Winsorization | Cap at percentiles | Financial data, outliers |
| **Unknown optimal** | Box-Cox | y = (x^λ-1)/λ | Automatic optimization |

This comprehensive transformation guide helps you convert non-normal data into approximately normal distributions for better statistical analysis and machine learning performance!

### Skewness
- **Definition**: Measure of asymmetry in distribution
- **Right Skewed (Positive)**: Tail extends to right, mean > median
- **Left Skewed (Negative)**: Tail extends to left, mean < median
- **Symmetric**: Skewness ≈ 0, mean ≈ median

---

## 5. Probability Fundamentals

### Basic Probability Concepts

#### Probability
- **Definition**: Likelihood of an event occurring
- **Range**: 0 ≤ P(A) ≤ 1
- **Formula**: P(A) = Number of favorable outcomes / Total outcomes

#### Sample Space
- **Definition**: Set of all possible outcomes
- **Example**: Rolling a die: {1, 2, 3, 4, 5, 6}

#### Event
- **Definition**: Subset of sample space
- **Example**: Rolling even number: {2, 4, 6}

### Probability Rules

#### Addition Rule
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- For mutually exclusive events: P(A ∪ B) = P(A) + P(B)

#### Multiplication Rule
- P(A ∩ B) = P(A) × P(B|A) = P(B) × P(A|B)
- For independent events: P(A ∩ B) = P(A) × P(B)

#### Conditional Probability
- P(A|B) = P(A ∩ B) / P(B)

### Example
```python
# Probability of drawing cards
total_cards = 52
red_cards = 26
face_cards = 12
red_face_cards = 6

P_red = red_cards / total_cards
P_face = face_cards / total_cards
P_red_and_face = red_face_cards / total_cards

# P(Red OR Face)
P_red_or_face = P_red + P_face - P_red_and_face
print(f"P(Red or Face) = {P_red_or_face}")
```

---

## 6. Probability Distributions

### What is a Random Variable? (Simple Explanation)

Think of a **random variable** as a way to assign numbers to the outcomes of a random event.

**Real-life Example**: Rolling a dice
- **Event**: Rolling a dice
- **Outcomes**: {⚀, ⚁, ⚂, ⚃, ⚄, ⚅}
- **Random Variable X**: Assigns numbers {1, 2, 3, 4, 5, 6} to each face

### Types of Random Variables

#### Discrete Random Variables
- **Simple Definition**: You can count the possible values (like counting on your fingers)
- **Examples**: 
  - Number of students in a class: 0, 1, 2, 3, ... (countable)
  - Number of cars in parking lot: 0, 1, 2, 3, ... (countable)
  - Result of dice roll: 1, 2, 3, 4, 5, 6 (exactly 6 possibilities)

#### Continuous Random Variables
- **Simple Definition**: Values can be any number within a range (like measuring with a ruler)
- **Examples**:
  - Height of students: 150.5 cm, 150.51 cm, 150.512 cm... (infinite possibilities)
  - Time to complete a task: 2.5 minutes, 2.51 minutes... (infinite possibilities)
  - Temperature: 25.3°C, 25.31°C, 25.312°C... (infinite possibilities)

---

### Discrete Probability Distributions (Step by Step)

## 1. Bernoulli Distribution 🎯

### What is it?
**The simplest distribution**: Only TWO possible outcomes (like flipping a coin)

### Real-Life Examples:
- **Coin flip**: Heads (success) or Tails (failure)
- **Exam**: Pass (success) or Fail (failure)
- **Product quality**: Good (success) or Defective (failure)
- **Customer**: Buys (success) or Doesn't buy (failure)

### Parameters:
- **p**: Probability of success (between 0 and 1)

### Mathematical Expressions:

#### Probability Mass Function (PMF):
```
P(X = 1) = p         (probability of success)
P(X = 0) = 1 - p     (probability of failure)
```

#### Mean (Expected Value):
```
E[X] = μ = p
```

#### Variance:
```
Var(X) = σ² = p(1-p)
```

#### Standard Deviation:
```
σ = √[p(1-p)]
```

### How it works:
- If success happens: X = 1 (with probability p)
- If failure happens: X = 0 (with probability 1-p)

### Example: Job Interview
```python
# Job Interview Example
# p = 0.7 means 70% chance of getting the job

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

p = 0.7  # 70% chance of success
outcomes = [0, 1]  # 0 = rejection, 1 = job offer
probabilities = [1-p, p]  # [30%, 70%]

plt.bar(['Rejection', 'Job Offer'], probabilities, color=['red', 'green'], alpha=0.7)
plt.title('Bernoulli Distribution - Job Interview')
plt.ylabel('Probability')
plt.show()

print(f"Probability of getting job: {p}")
print(f"Probability of rejection: {1-p}")
print(f"Expected value (average outcome): {p}")
```

---

## 2. Binomial Distribution 🎲

### What is it?
**Multiple coin flips**: Repeating a Bernoulli trial multiple times and counting successes

### Think of it as:
"If I repeat the same yes/no experiment N times, how many times will I get YES?"

### Real-Life Examples:
- **10 coin flips**: How many heads will you get?
- **20 customers**: How many will make a purchase?
- **50 products**: How many will be defective?
- **100 emails**: How many will be spam?

### Parameters:
- **n**: Number of trials (how many times you repeat)
- **p**: Probability of success in each trial

### Mathematical Expressions:

#### Probability Mass Function (PMF):
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

Where:
C(n,k) = n! / (k!(n-k)!)  (combination formula)
k = number of successes (0, 1, 2, ..., n)
```

#### Mean (Expected Value):
```
E[X] = μ = np
```

#### Variance:
```
Var(X) = σ² = np(1-p)
```

#### Standard Deviation:
```
σ = √[np(1-p)]
```

#### Conditions for Binomial:
1. Fixed number of trials (n)
2. Each trial has only two outcomes (success/failure)
3. Probability of success (p) is constant for each trial
4. Trials are independent

### Example: Marketing Campaign
```python
# Marketing Campaign Example
# Send emails to 20 customers, each has 30% chance of buying

from scipy.stats import binom

n = 20  # 20 customers
p = 0.3  # 30% chance each customer buys

# Calculate probabilities for 0, 1, 2, ... 20 purchases
x_values = np.arange(0, n+1)
probabilities = binom.pmf(x_values, n, p)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(x_values, probabilities, alpha=0.7, color='skyblue')
plt.title(f'Binomial Distribution\n({n} customers, {p} success rate)')
plt.xlabel('Number of Purchases')
plt.ylabel('Probability')

# Most likely outcomes
most_likely = x_values[np.argmax(probabilities)]
expected_purchases = n * p

plt.axvline(expected_purchases, color='red', linestyle='--', 
           label=f'Expected: {expected_purchases}')
plt.axvline(most_likely, color='green', linestyle='--', 
           label=f'Most likely: {most_likely}')
plt.legend()

# Practical interpretation
print("MARKETING CAMPAIGN INSIGHTS:")
print(f"Expected number of purchases: {expected_purchases}")
print(f"Most likely number of purchases: {most_likely}")

# Mathematical probability calculations
print(f"\nMATHEMATICAL CALCULATIONS:")
print(f"P(X = 6) = C(20,6) × (0.3)^6 × (0.7)^14")
print(f"P(X = 6) = {binom.pmf(6, n, p):.4f}")
print(f"P(X ≤ 5) = Σ P(X = k) for k = 0 to 5 = {binom.cdf(5, n, p):.4f}")
print(f"P(X > 8) = 1 - P(X ≤ 8) = 1 - {binom.cdf(8, n, p):.4f} = {1 - binom.cdf(8, n, p):.4f}")

print(f"\nBUSINESS INTERPRETATIONS:")
print(f"• Expected purchases: μ = np = {n} × {p} = {expected_purchases}")
print(f"• Standard deviation: σ = √[np(1-p)] = √[{n}×{p}×{1-p}] = {np.sqrt(n*p*(1-p)):.2f}")
print(f"• Most results will be between {expected_purchases - np.sqrt(n*p*(1-p)):.1f} and {expected_purchases + np.sqrt(n*p*(1-p)):.1f} purchases")
```

### Understanding Binomial with Different Scenarios:
```python
# Compare different scenarios
scenarios = [
    (10, 0.1, "10 emails, 10% response rate"),
    (10, 0.5, "10 emails, 50% response rate"), 
    (10, 0.9, "10 emails, 90% response rate"),
]

plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green']

for i, (n, p, label) in enumerate(scenarios):
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    plt.plot(x, pmf, 'o-', color=colors[i], label=label, alpha=0.7)

plt.title('Binomial Distribution - Different Success Rates')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 3. Poisson Distribution 📞

### What is it?
**Counting rare events**: How many times something happens in a fixed period when events are rare and random

### Key Idea:
"Events happen randomly over time, and we want to count how many occur"

### Real-Life Examples:
- **Phone calls**: How many calls does a call center receive per hour?
- **Website visits**: How many people visit a website per minute?
- **Defects**: How many defective products in a batch?
- **Accidents**: How many car accidents per day in a city?
- **Goals**: How many goals scored in a football match?

### When to Use Poisson:
1. **Events are rare** (don't happen very often)
2. **Events are independent** (one doesn't affect another)
3. **Rate is constant** (average doesn't change over time)
4. **Fixed time/space interval**

### Parameter:
- **λ (lambda)**: Average number of events in the time period

### Mathematical Expressions:

#### Probability Mass Function (PMF):
```
P(X = k) = (λ^k × e^(-λ)) / k!

Where:
λ = average rate (lambda)
k = number of events (0, 1, 2, 3, ...)
e ≈ 2.71828 (Euler's number)
k! = k factorial = k × (k-1) × (k-2) × ... × 1
```

#### Mean (Expected Value):
```
E[X] = μ = λ
```

#### Variance:
```
Var(X) = σ² = λ
```

#### Standard Deviation:
```
σ = √λ
```

#### Key Property:
For Poisson distribution, **Mean = Variance = λ**

#### Conditions for Poisson:
1. Events occur independently
2. Average rate (λ) is constant over time
3. Events are rare (small probability)
4. No two events occur simultaneously

### Example: Customer Service Calls
```python
# Customer Service Example
# Call center receives average of 4 calls per hour

from scipy.stats import poisson

lambda_rate = 4  # Average 4 calls per hour

# Possible number of calls: 0, 1, 2, 3, ... 15
x_values = np.arange(0, 16)
probabilities = poisson.pmf(x_values, lambda_rate)

plt.figure(figsize=(15, 10))

# Plot 1: Basic Poisson Distribution
plt.subplot(2, 2, 1)
plt.bar(x_values, probabilities, alpha=0.7, color='orange')
plt.title(f'Poisson Distribution - Call Center\n(Average {lambda_rate} calls/hour)')
plt.xlabel('Number of Calls in One Hour')
plt.ylabel('Probability')
plt.axvline(lambda_rate, color='red', linestyle='--', label=f'Average: {lambda_rate}')
plt.legend()

# Practical questions and answers
print("CALL CENTER INSIGHTS:")
print(f"Average calls per hour: λ = {lambda_rate}")
print(f"Standard deviation: σ = √λ = √{lambda_rate} = {np.sqrt(lambda_rate):.2f}")

# Mathematical probability calculations
print(f"\nMATHEMATICAL CALCULATIONS:")
print(f"P(X = 4) = (λ^4 × e^(-λ)) / 4!")
print(f"P(X = 4) = ({lambda_rate}^4 × e^(-{lambda_rate})) / 24 = {poisson.pmf(4, lambda_rate):.4f}")

print(f"P(X = 0) = (λ^0 × e^(-λ)) / 0!")
print(f"P(X = 0) = (1 × e^(-{lambda_rate})) / 1 = {poisson.pmf(0, lambda_rate):.4f}")

print(f"P(X > 6) = 1 - P(X ≤ 6) = 1 - {poisson.cdf(6, lambda_rate):.4f} = {1 - poisson.cdf(6, lambda_rate):.4f}")

print(f"\nBUSINESS INTERPRETATIONS:")
print(f"• Most likely number of calls: {lambda_rate} calls/hour")
print(f"• 68% of hours will have between {lambda_rate - np.sqrt(lambda_rate):.1f} and {lambda_rate + np.sqrt(lambda_rate):.1f} calls")
print(f"• Very quiet hour (0 calls): {poisson.pmf(0, lambda_rate)*100:.1f}% chance")
print(f"• Very busy hour (>6 calls): {(1 - poisson.cdf(6, lambda_rate))*100:.1f}% chance")
```

### Comparing Different Rates:
```python
# Compare different call volumes
rates = [1, 3, 5, 10]
colors = ['red', 'blue', 'green', 'purple']

plt.subplot(2, 2, 2)
for i, rate in enumerate(rates):
    x = np.arange(0, 20)
    pmf = poisson.pmf(x, rate)
    plt.plot(x, pmf, 'o-', color=colors[i], label=f'λ = {rate}', alpha=0.7)

plt.title('Poisson Distribution - Different Rates')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()

# Real-world interpretation
interpretations = {
    1: "Quiet business (1 call/hour avg)",
    3: "Moderate business (3 calls/hour avg)", 
    5: "Busy business (5 calls/hour avg)",
    10: "Very busy business (10 calls/hour avg)"
}

for rate, meaning in interpretations.items():
    print(f"λ = {rate}: {meaning}")
```

### More Poisson Examples:
```python
# Different real-world Poisson scenarios
scenarios = [
    (2, "Website crashes per month"),
    (0.5, "Major earthquakes per year in a region"),
    (8, "Customers entering store per hour"),
    (12, "Emails received per day")
]

plt.subplot(2, 2, 3)
for i, (rate, description) in enumerate(scenarios):
    x = np.arange(0, int(rate * 3) + 5)
    pmf = poisson.pmf(x, rate)
    plt.plot(x, pmf, 'o-', label=f'{description}\n(λ={rate})', alpha=0.7)

plt.title('Real-World Poisson Examples')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()
```

---

### Poisson Process 🔄

### Simple Explanation:
A **Poisson Process** is like having a Poisson distribution happening continuously over time.

### Real Example - Hospital Emergency Room:
```python
# Emergency Room - patients arrive following Poisson process
# Average 3 patients per hour

# Simulate one day (24 hours)
hours = 24
avg_per_hour = 3
total_patients = np.random.poisson(avg_per_hour * hours)

print(f"HOSPITAL EMERGENCY ROOM - One Day Simulation")
print(f"Average rate: {avg_per_hour} patients/hour")
print(f"Expected patients in 24 hours: {avg_per_hour * hours}")
print(f"Actual patients today: {total_patients}")

# Hour by hour breakdown
hourly_arrivals = np.random.poisson(avg_per_hour, hours)

plt.subplot(2, 2, 4)
plt.plot(range(24), hourly_arrivals, 'o-', color='red', alpha=0.7)
plt.axhline(avg_per_hour, color='blue', linestyle='--', label=f'Average: {avg_per_hour}')
plt.title('Poisson Process - Hospital Arrivals')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Patients')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Busiest hour: Hour {np.argmax(hourly_arrivals)} with {max(hourly_arrivals)} patients")
print(f"Quietest hour: Hour {np.argmin(hourly_arrivals)} with {min(hourly_arrivals)} patients")
```

---

### Expected Value (Simple Explanation) 🎯

### What is Expected Value?
**Expected Value** = Average outcome if you repeat the experiment many, many times

### Mathematical Expressions:

#### For Discrete Random Variables:
```
E[X] = μ = Σ [x × P(X = x)]
     = x₁ × P(X = x₁) + x₂ × P(X = x₂) + ... + xₙ × P(X = xₙ)
```

#### For Continuous Random Variables:
```
E[X] = μ = ∫ x × f(x) dx
```

#### Properties of Expected Value:
```
1. E[a] = a                    (constant)
2. E[aX] = a × E[X]           (linearity)
3. E[X + Y] = E[X] + E[Y]     (additivity)
4. E[aX + b] = a × E[X] + b   (linear transformation)
```

#### For Common Distributions:
```
Bernoulli:  E[X] = p
Binomial:   E[X] = np  
Poisson:    E[X] = λ
```

### Think of it as:
"If I do this 1000 times, what would be the average result?"

### Real Examples:

```python
# Example 1: Casino Dice Game - Mathematical Calculation
# Pay $1 to play, win $4 if you roll a 6, win nothing otherwise

dice_outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6  # Each outcome has 1/6 probability

# Winnings for each outcome
winnings = [-1, -1, -1, -1, -1, 3]  # Lose $1 for 1-5, Win $3 for 6 (net: $4-$1=$3)

# Mathematical calculation of expected value
expected_winnings = sum(w * p for w, p in zip(winnings, probabilities))
# E[X] = (-1)×(1/6) + (-1)×(1/6) + (-1)×(1/6) + (-1)×(1/6) + (-1)×(1/6) + (3)×(1/6)
# E[X] = (-5/6) + (3/6) = -2/6 = -1/3 ≈ -0.33

print("CASINO DICE GAME - MATHEMATICAL ANALYSIS:")
print(f"Cost to play: $1")
print(f"Win $4 if you roll 6, nothing otherwise")
print("Mathematical calculation:")
print("E[X] = (-1)×(1/6) + (-1)×(1/6) + (-1)×(1/6) + (-1)×(1/6) + (-1)×(1/6) + (3)×(1/6)")
print(f"E[X] = (-5/6) + (3/6) = -2/6 = {expected_winnings:.3f}")
print(f"Expected value per game: ${expected_winnings:.2f}")
if expected_winnings < 0:
    print("This is a losing game in the long run!")
else:
    print("This is a winning game!")

# Example 2: Insurance Company - Mathematical Analysis
# Sell $100,000 life insurance for $500 premium
# Probability of death in one year = 0.001 (0.1%)

premium = 500
payout = 100000
prob_death = 0.001
prob_survive = 1 - prob_death

# Company's expected profit per policy
# E[Profit] = (Profit if survive) × P(survive) + (Profit if death) × P(death)
# E[Profit] = ($500) × (0.999) + ($500 - $100,000) × (0.001)
expected_profit = premium * prob_survive + (premium - payout) * prob_death

print(f"\nINSURANCE COMPANY - MATHEMATICAL ANALYSIS:")
print(f"Premium collected: ${premium}")
print(f"Payout if death: ${payout}")
print(f"Probability of death: {prob_death} = {prob_death * 100}%")
print(f"Probability of survival: {prob_survive} = {prob_survive * 100}%")
print("Mathematical calculation:")
print(f"E[Profit] = ${premium} × {prob_survive} + (${premium} - ${payout}) × {prob_death}")
print(f"E[Profit] = ${premium * prob_survive:.2f} + ${(premium - payout) * prob_death:.2f}")
print(f"E[Profit] = ${expected_profit:.2f}")

# Example 3: Stock Investment - Mathematical Analysis
stock_scenarios = [
    (0.3, -20, "Market crash"),      # 30% chance, lose 20%
    (0.5, 8, "Normal growth"),       # 50% chance, gain 8%
    (0.2, 25, "Bull market")         # 20% chance, gain 25%
]

# E[Return] = Σ [Return × Probability]
expected_return = sum(prob * return_pct for prob, return_pct, _ in stock_scenarios)

print(f"\nSTOCK INVESTMENT - MATHEMATICAL ANALYSIS:")
print("Scenarios:")
for prob, return_pct, scenario in stock_scenarios:
    print(f"  {scenario}: P = {prob}, Return = {return_pct:+}%")

print("Mathematical calculation:")
print("E[Return] = (-20%) × (0.3) + (8%) × (0.5) + (25%) × (0.2)")
calculation_parts = [f"({return_pct:+}%) × ({prob})" for prob, return_pct, _ in stock_scenarios]
print(f"E[Return] = {' + '.join(calculation_parts)}")
individual_contributions = [prob * return_pct for prob, return_pct, _ in stock_scenarios]
print(f"E[Return] = {individual_contributions[0]:+.1f}% + {individual_contributions[1]:+.1f}% + {individual_contributions[2]:+.1f}%")
print(f"E[Return] = {expected_return:.1f}%")
```

### Summary with Simple Memory Tricks 🧠

```python
print("\n" + "="*60)
print("PROBABILITY DISTRIBUTIONS - MEMORY TRICKS")
print("="*60)

print("\n🎯 BERNOULLI - 'Single Shot'")
print("   Think: One coin flip, one exam, one job interview")
print("   Question: 'Will this ONE thing succeed or fail?'")
print("   Example: Will this customer buy? (Yes/No)")

print("\n🎲 BINOMIAL - 'Multiple Shots'") 
print("   Think: Multiple coin flips, multiple customers")
print("   Question: 'Out of N tries, how many will succeed?'")
print("   Example: Out of 20 customers, how many will buy?")

print("\n📞 POISSON - 'Counting Rare Events'")
print("   Think: Phone calls, accidents, defects")
print("   Question: 'How many times will this rare thing happen?'")
print("   Example: How many calls in the next hour?")

print("\n🎯 EXPECTED VALUE - 'Long-term Average'")
print("   Think: Casino winnings, investment returns")
print("   Question: 'If I repeat this 1000 times, what's the average?'")
print("   Example: Average winnings from playing this game")

plt.show()
```

---

## 7. Sampling Distributions

### Sampling Distribution of Sample Mean
- **Definition**: Distribution of sample means from all possible samples
- **Key Properties**:
  - Mean of sampling distribution = Population mean (μ)
  - Standard deviation = σ/√n (Standard Error)

### Standard Error of Mean
- **Formula**: SE = σ/√n
- **Interpretation**: Standard deviation of sample means
- **Decreases**: As sample size increases

### Central Limit Theorem (CLT)
- **Statement**: For large sample sizes (n ≥ 30), sampling distribution of mean approaches normal distribution
- **Applies**: Regardless of population distribution shape
- **Parameters**: Mean = μ, Standard deviation = σ/√n
- **Importance**: Foundation for confidence intervals and hypothesis testing

### Example
```python
# Demonstrate CLT with non-normal population
# Population: Exponential distribution
population = np.random.exponential(2, 10000)

# Draw many samples and calculate means
sample_means = []
sample_size = 30
num_samples = 1000

for _ in range(num_samples):
    sample = np.random.choice(population, sample_size)
    sample_means.append(np.mean(sample))

# Plot sampling distribution
plt.hist(sample_means, bins=50, alpha=0.7, density=True)
plt.title('Sampling Distribution of Mean (CLT)')
plt.axvline(np.mean(population), color='red', linestyle='--', label='Population Mean')
plt.legend()

# Verify CLT properties
print(f"Population mean: {np.mean(population):.3f}")
print(f"Sample means average: {np.mean(sample_means):.3f}")
print(f"Theoretical SE: {np.std(population)/np.sqrt(sample_size):.3f}")
print(f"Actual SE: {np.std(sample_means):.3f}")
```

---

## 8. Statistical Inference

### Significance Level (α)
- **Definition**: Probability of Type I error (rejecting true null hypothesis)
- **Common values**: 0.05, 0.01, 0.10
- **Interpretation**: If α = 0.05, we're willing to be wrong 5% of the time

### Hypothesis Testing Process
1. State null (H₀) and alternative (H₁) hypotheses
2. Choose significance level (α)
3. Calculate test statistic
4. Find p-value or critical value
5. Make decision and interpret

### Types of Errors
- **Type I Error**: Reject true H₀ (False Positive)
- **Type II Error**: Fail to reject false H₀ (False Negative)
- **Power**: 1 - P(Type II Error) = Probability of correctly rejecting false H₀

### Example
```python
from scipy import stats

# Example: Testing if mean height > 170 cm
# H₀: μ ≤ 170, H₁: μ > 170
sample_heights = [172, 168, 175, 171, 169, 174, 173, 170, 176, 172]
hypothesized_mean = 170
alpha = 0.05

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(sample_heights, hypothesized_mean)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value/2:.3f}")  # One-tailed test

if p_value/2 < alpha:
    print(f"Reject H₀ at α = {alpha}")
else:
    print(f"Fail to reject H₀ at α = {alpha}")
```

---

## 9. Multivariate Statistics

### Covariance
- **Definition**: Measure of how two variables change together
- **Formula**: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
- **Interpretation**: 
  - Positive: Variables increase together
  - Negative: One increases as other decreases
  - Zero: No linear relationship

### Correlation
- **Definition**: Standardized measure of linear relationship
- **Pearson Correlation**: r = Cov(X,Y) / (σₓ × σᵧ)
- **Range**: -1 ≤ r ≤ 1
- **Interpretation**:
  - r = 1: Perfect positive correlation
  - r = -1: Perfect negative correlation
  - r = 0: No linear correlation

### Properties of Correlation
- Unitless measure
- Unaffected by linear transformations
- Only measures linear relationships
- Correlation ≠ Causation

### Example
```python
# Generate correlated data
np.random.seed(42)
x = np.random.normal(50, 10, 100)
y = 2 * x + np.random.normal(0, 5, 100)  # y is related to x with noise

# Calculate covariance and correlation
covariance = np.cov(x, y)[0, 1]
correlation = np.corrcoef(x, y)[0, 1]

print(f"Covariance: {covariance:.2f}")
print(f"Correlation: {correlation:.2f}")

# Visualize relationship
plt.scatter(x, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Scatter Plot (r = {correlation:.2f})')
```

---

## 10. Interview Questions

### Fundamental Concepts

1. **What's the difference between population and sample?**
   - Population: Complete set of all observations
   - Sample: Subset used to make inferences about population

2. **Explain different types of sampling methods and when to use each.**
   - Simple random: Equal probability, unbiased
   - Stratified: When population has distinct groups
   - Systematic: When list is available, easier implementation
   - Cluster: When population is geographically distributed

3. **What are the different types of variables? Give examples.**
   - Quantitative: Continuous (height), Discrete (count)
   - Qualitative: Nominal (color), Ordinal (rating)

### Descriptive Statistics

4. **When would you use mean vs median vs mode?**
   - Mean: Normal distribution, no outliers
   - Median: Skewed data, outliers present
   - Mode: Categorical data, most common value needed

5. **How do you detect and handle outliers?**
   - Detection: IQR method, Z-score method, box plots
   - Handling: Remove, transform, or use robust statistics

6. **What is the difference between variance and standard deviation?**
   - Variance: Average squared deviation (units²)
   - Standard deviation: Square root of variance (original units)

7. **Explain skewness and its implications.**
   - Right skewed: Mean > Median, tail to right
   - Left skewed: Mean < Median, tail to left
   - Affects choice of central tendency measure

### Probability and Distributions

8. **What's the difference between Bernoulli and Binomial distributions?**
   - Bernoulli: Single trial, two outcomes
   - Binomial: n independent Bernoulli trials

9. **When would you use Poisson distribution?**
   - Counting events in fixed interval
   - Events are independent and rare
   - Examples: Defects per product, calls per hour

10. **Explain the concept of expected value.**
    - Average value over many trials
    - Weighted average of all possible outcomes
    - E[X] = Σ x × P(X = x) for discrete variables

### Central Limit Theorem

11. **State and explain the Central Limit Theorem.**
    - Sample means approach normal distribution for large n
    - True regardless of population distribution
    - Mean = μ, Standard deviation = σ/√n

12. **What's the minimum sample size for CLT to apply?**
    - Generally n ≥ 30
    - Can be smaller for normal populations
    - May need larger for highly skewed populations

13. **What is standard error and how does it relate to sample size?**
    - Standard deviation of sampling distribution
    - SE = σ/√n
    - Decreases as sample size increases

### Statistical Inference

14. **Explain Type I and Type II errors.**
    - Type I: Reject true null hypothesis (false positive)
    - Type II: Fail to reject false null hypothesis (false negative)
    - Trade-off: Reducing one increases the other

15. **What does statistical significance mean?**
    - Result is unlikely due to chance alone
    - p-value < significance level (α)
    - Doesn't necessarily mean practical significance

16. **How do you choose significance level?**
    - Consider consequences of errors
    - Industry standards (0.05 common)
    - More stringent (0.01) for critical decisions

### Correlation and Relationships

17. **What's the difference between covariance and correlation?**
    - Covariance: Units of X × Y, unbounded
    - Correlation: Unitless, bounded [-1, 1]
    - Correlation is standardized covariance

18. **Can correlation imply causation? Why not?**
    - No, correlation ≠ causation
    - Third variables, reverse causation possible
    - Need controlled experiments for causation

19. **What are the assumptions of Pearson correlation?**
    - Linear relationship
    - Continuous variables
    - Normal distribution (for significance testing)
    - No outliers

### Applied Questions

20. **How would you determine if a dataset is normally distributed?**
    - Visual: Histogram, Q-Q plot, box plot
    - Statistical: Shapiro-Wilk test, Kolmogorov-Smirnov test
    - Check skewness and kurtosis

21. **A dataset has mean=50 and std=10. What can you say about the distribution?**
    - If normal: ~68% within [40, 60], ~95% within [30, 70]
    - Need additional info about shape
    - CV = 20% (moderate variability)

22. **How do you standardize a dataset and why?**
    - Z-score: (x - μ) / σ
    - Why: Compare different scales, ML algorithms, normality

23. **In A/B testing, how would you determine sample size?**
    - Power analysis
    - Specify effect size, power (0.8), significance (0.05)
    - Use statistical software or formulas

24. **How does sample size affect confidence intervals?**
    - Larger sample → Smaller standard error
    - Smaller standard error → Narrower confidence intervals
    - More precise estimates

25. **What's the difference between p-value and confidence interval?**
    - p-value: Probability of observing result given H₀ is true
    - CI: Range of plausible parameter values
    - CI provides more information about effect size

### Machine Learning Context

26. **How do these statistical concepts apply to machine learning?**
    - Data exploration and preprocessing
    - Feature selection and engineering
    - Model validation and evaluation
    - Uncertainty quantification

27. **Why is normality important in machine learning?**
    - Some algorithms assume normality
    - Affects convergence and performance
    - Important for linear regression, naive Bayes

28. **How do you handle imbalanced datasets statistically?**
    - Understand through descriptive statistics
    - Use stratified sampling
    - Adjust performance metrics
    - Consider resampling techniques

## Summary

This notebook covers the essential statistical concepts needed for machine learning. The key takeaways are:

1. **Foundation**: Understand data types and collection methods
2. **Description**: Summarize data effectively with appropriate measures
3. **Probability**: Grasp uncertainty and randomness
4. **Inference**: Make conclusions about populations from samples
5. **Relationships**: Understand how variables relate to each other
6. **Application**: Apply concepts to real-world ML problems

Practice with real datasets and continue exploring advanced topics like Bayesian statistics, experimental design, and multivariate analysis to deepen your understanding.