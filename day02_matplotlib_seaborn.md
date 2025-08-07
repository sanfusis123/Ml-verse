# Day 2: Data Visualization with Matplotlib and Seaborn

## 📚 Topics
- Matplotlib fundamentals and advanced plotting
- Seaborn statistical visualizations
- Creating publication-quality figures
- Interactive and 3D visualizations

---

## 1. Matplotlib - The Foundation of Python Visualization

### 📖 Core Concepts

#### Architecture
Matplotlib has three layers:
1. **Backend Layer**: Handles rendering (Agg, TkAgg, Qt5Agg)
2. **Artist Layer**: All visual elements (Figure, Axes, Line2D, Text)
3. **Scripting Layer**: pyplot interface for ease of use

#### Figure and Axes
- **Figure**: The entire window/page
- **Axes**: The plotting area (can have multiple per figure)
- **Axis**: The x or y axis

### 🔢 Mathematical Foundation

#### Coordinate Systems
1. **Data coordinates**: Your actual data values
2. **Axes coordinates**: (0,0) to (1,1) within axes
3. **Figure coordinates**: (0,0) to (1,1) for entire figure
4. **Display coordinates**: Pixel coordinates

#### Transformations
```
Data -> Axes Transform -> Figure Transform -> Display Transform
(x, y) -> normalized -> figure fraction -> pixels
```

### 💻 Advanced Matplotlib Code

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Figure Architecture and Subplots
print("=== Advanced Subplot Layouts ===")
# Using GridSpec for complex layouts
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Create subplots with different sizes
ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
ax2 = fig.add_subplot(gs[1, :-1])  # Middle row, first 2 columns
ax3 = fig.add_subplot(gs[1:, -1])  # Last 2 rows, last column
ax4 = fig.add_subplot(gs[-1, 0])  # Bottom left
ax5 = fig.add_subplot(gs[-1, -2])  # Bottom middle

# Plot different data in each
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), 'b-', label='sin(x)')
ax1.plot(x, np.cos(x), 'r--', label='cos(x)')
ax1.legend()
ax1.set_title('Trigonometric Functions')

ax2.scatter(np.random.randn(50), np.random.randn(50), alpha=0.6)
ax2.set_title('Random Scatter')

ax3.hist(np.random.randn(1000), bins=30, orientation='horizontal')
ax3.set_title('Distribution')

ax4.bar(['A', 'B', 'C'], [3, 7, 5])
ax4.set_title('Bar Chart')

ax5.pie([30, 25, 20, 25], labels=['Q1', 'Q2', 'Q3', 'Q4'])
ax5.set_title('Pie Chart')

plt.tight_layout()
plt.savefig('complex_layout.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Customizing Plots - Publication Quality
print("\n=== Publication Quality Plots ===")
fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
x = np.linspace(0, 4*np.pi, 1000)
y1 = np.sin(x) * np.exp(-x/10)
y2 = np.cos(x) * np.exp(-x/10)

# Plot with custom styling
line1 = ax.plot(x, y1, 'b-', linewidth=2.5, label='Damped Sine')
line2 = ax.plot(x, y2, 'r--', linewidth=2.5, label='Damped Cosine')

# Customize axes
ax.set_xlim(0, 4*np.pi)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')
ax.set_title('Damped Oscillations', fontsize=16, fontweight='bold', pad=20)

# Add grid with custom style
ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
ax.set_xticklabels(['0', 'π', '2π', '3π', '4π'])

# Add annotation
ax.annotate('Decay Point', xy=(2*np.pi, 0.1), xytext=(2.5*np.pi, 0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
            fontsize=12, ha='center')

# Legend with custom location and styling
ax.legend(loc='upper right', frameon=True, shadow=True, 
          fancybox=True, framealpha=0.9, fontsize=12)

# Add mathematical expression
ax.text(0.5, 0.85, r'$y = e^{-x/10} \cdot \sin(x)$', 
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.show()

# 3. 3D Plotting
print("\n=== 3D Visualization ===")
fig = plt.figure(figsize=(12, 5))

# 3D Surface Plot
ax1 = fig.add_subplot(121, projection='3d')
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.set_title('3D Surface Plot')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 3D Scatter Plot
ax2 = fig.add_subplot(122, projection='3d')
n = 100
xs = np.random.randn(n)
ys = np.random.randn(n)
zs = np.random.randn(n)
colors = np.random.rand(n)

scatter = ax2.scatter(xs, ys, zs, c=colors, s=50, alpha=0.6)
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.set_title('3D Scatter Plot')

plt.tight_layout()
plt.show()

# 4. Advanced Plot Types
print("\n=== Advanced Plot Types ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Contour Plot
ax = axes[0, 0]
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-(X**2 + Y**2)) * np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)

contour = ax.contour(X, Y, Z, levels=15, colors='black', linewidths=0.5)
contourf = ax.contourf(X, Y, Z, levels=15, cmap='RdBu')
ax.clabel(contour, inline=True, fontsize=8)
fig.colorbar(contourf, ax=ax)
ax.set_title('Contour Plot')

# Heatmap
ax = axes[0, 1]
data = np.random.randn(10, 10)
im = ax.imshow(data, cmap='hot', interpolation='nearest')
ax.set_title('Heatmap')
fig.colorbar(im, ax=ax)

# Violin Plot
ax = axes[1, 0]
data = [np.random.normal(0, std, 100) for std in range(1, 5)]
violin_parts = ax.violinplot(data, showmeans=True, showmedians=True)
ax.set_title('Violin Plot')
ax.set_xlabel('Distribution')
ax.set_ylabel('Value')

# Stream Plot (Vector Field)
ax = axes[1, 1]
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
U = -Y
V = X
ax.streamplot(X, Y, U, V, density=1.5, color='blue', linewidth=1)
ax.set_title('Stream Plot (Vector Field)')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.tight_layout()
plt.show()

# 5. Animation Example
print("\n=== Animation ===")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Animated Sine Wave')

def init():
    line.set_data([], [])
    return line,

def animate(frame):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x + frame/10)
    line.set_data(x, y)
    return line,

# Note: This would create an animation if saved or shown in appropriate environment
# anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=True)
# anim.save('sine_animation.gif', writer='pillow')

# 6. Custom Artists and Patches
print("\n=== Custom Artists ===")
fig, ax = plt.subplots(figsize=(8, 8))

# Add various shapes
rect = Rectangle((0.2, 0.2), 0.3, 0.3, facecolor='blue', alpha=0.5)
circle = Circle((0.7, 0.7), 0.2, facecolor='red', alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circle)

# Add custom polygon
polygon = plt.Polygon([[0.1, 0.5], [0.3, 0.9], [0.5, 0.5]], 
                      facecolor='green', alpha=0.5)
ax.add_patch(polygon)

# Add arrow
ax.arrow(0.1, 0.1, 0.5, 0.3, head_width=0.05, head_length=0.05, 
         fc='black', ec='black')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_title('Custom Artists and Patches')
plt.show()
```

---

## 2. Seaborn - Statistical Data Visualization

### 📖 Core Concepts

#### What is Seaborn?
- Built on top of Matplotlib
- Provides high-level interface for statistical graphics
- Integrates closely with Pandas DataFrames
- Better default styles and color palettes

### 🔢 Statistical Visualizations

#### Key Plot Types
1. **Relational**: scatterplot, lineplot
2. **Distributional**: histplot, kdeplot, ecdfplot
3. **Categorical**: boxplot, violinplot, swarmplot
4. **Matrix**: heatmap, clustermap
5. **Regression**: regplot, lmplot

### 💻 Advanced Seaborn Code

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set Seaborn style
sns.set_theme(style="whitegrid", palette="deep")

# Generate sample dataset
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'z': np.random.randn(n) * 0.5,
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'subcategory': np.random.choice(['X', 'Y', 'Z'], n),
    'value': np.random.exponential(2, n),
    'score': np.random.randint(0, 100, n)
})

# Add correlations
data['y'] = data['x'] * 0.5 + np.random.randn(n) * 0.5
data['z'] = data['x'] * 0.3 + data['y'] * 0.5 + np.random.randn(n) * 0.3

# 1. Advanced Relational Plots
print("=== Relational Plots ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Scatter plot with multiple variables
sns.scatterplot(data=data, x='x', y='y', hue='category', 
                size='value', style='subcategory', alpha=0.7, ax=axes[0,0])
axes[0,0].set_title('Multi-dimensional Scatter Plot')

# Line plot with confidence intervals
time_data = pd.DataFrame({
    'time': np.tile(np.arange(50), 10),
    'value': np.concatenate([np.cumsum(np.random.randn(50)) + i*5 
                             for i in range(10)]),
    'group': np.repeat(['A', 'B'], 250)
})
sns.lineplot(data=time_data, x='time', y='value', hue='group', 
             errorbar='sd', ax=axes[0,1])
axes[0,1].set_title('Time Series with Confidence Bands')

# Regression plot with residuals
sns.regplot(data=data, x='x', y='y', scatter_kws={'alpha':0.5}, 
            ax=axes[1,0])
axes[1,0].set_title('Linear Regression with Confidence Interval')

# Joint plot (requires separate figure)
plt.figure(figsize=(8, 8))
g = sns.jointplot(data=data, x='x', y='y', kind='hex', 
                  marginal_kws=dict(bins=30))
g.fig.suptitle('Hexbin Joint Plot with Marginal Distributions', y=1.02)
plt.show()

# 2. Advanced Distribution Plots
print("\n=== Distribution Plots ===")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# KDE plot with multiple distributions
for cat in data['category'].unique():
    subset = data[data['category'] == cat]
    sns.kdeplot(data=subset, x='x', label=cat, ax=axes[0,0])
axes[0,0].set_title('Kernel Density Estimation by Category')
axes[0,0].legend()

# Bivariate KDE
sns.kdeplot(data=data, x='x', y='y', cmap='Blues', shade=True, 
            ax=axes[0,1])
axes[0,1].set_title('2D Kernel Density Estimation')

# ECDF plot
sns.ecdfplot(data=data, x='value', hue='category', ax=axes[0,2])
axes[0,2].set_title('Empirical Cumulative Distribution Function')

# Histogram with KDE overlay
sns.histplot(data=data, x='x', kde=True, bins=30, ax=axes[1,0])
axes[1,0].set_title('Histogram with KDE Overlay')

# Rug plot with KDE
sns.kdeplot(data=data, x='x', ax=axes[1,1])
sns.rugplot(data=data, x='x', ax=axes[1,1])
axes[1,1].set_title('KDE with Rug Plot')

# QQ plot (using scipy)
from scipy import stats
stats.probplot(data['x'], dist="norm", plot=axes[1,2])
axes[1,2].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# 3. Advanced Categorical Plots
print("\n=== Categorical Plots ===")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Enhanced box plot
sns.boxplot(data=data, x='category', y='value', hue='subcategory', 
            ax=axes[0,0])
axes[0,0].set_title('Grouped Box Plot')

# Violin plot with inner details
sns.violinplot(data=data, x='category', y='value', inner='box', 
               ax=axes[0,1])
axes[0,1].set_title('Violin Plot with Inner Box')

# Swarm plot with hue
subset_data = data.sample(300)  # Reduce points for clarity
sns.swarmplot(data=subset_data, x='category', y='score', 
              hue='subcategory', ax=axes[0,2])
axes[0,2].set_title('Swarm Plot')

# Strip plot with jitter
sns.stripplot(data=data, x='category', y='value', 
              jitter=True, alpha=0.5, ax=axes[1,0])
axes[1,0].set_title('Strip Plot with Jitter')

# Point plot with confidence intervals
sns.pointplot(data=data, x='category', y='value', hue='subcategory',
              capsize=0.1, ax=axes[1,1])
axes[1,1].set_title('Point Plot with Error Bars')

# Count plot
sns.countplot(data=data, x='category', hue='subcategory', ax=axes[1,2])
axes[1,2].set_title('Count Plot')

plt.tight_layout()
plt.show()

# 4. Matrix Plots
print("\n=== Matrix Plots ===")
# Correlation matrix
corr_data = data[['x', 'y', 'z', 'value', 'score']].corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Basic heatmap
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
            ax=axes[0])
axes[0].set_title('Correlation Heatmap')

# Clustermap
plt.figure(figsize=(10, 8))
sns.clustermap(corr_data, annot=True, cmap='coolwarm', center=0,
               method='average', metric='euclidean')
plt.title('Hierarchical Clustering Heatmap')
plt.show()

# 5. Regression Plots
print("\n=== Regression Analysis ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linear regression with different confidence levels
sns.regplot(data=data, x='x', y='y', ci=95, ax=axes[0,0])
axes[0,0].set_title('Linear Regression (95% CI)')

# Polynomial regression
sns.regplot(data=data, x='x', y='z', order=2, ax=axes[0,1])
axes[0,1].set_title('Polynomial Regression (Order 2)')

# Logistic regression
binary_data = data.copy()
binary_data['binary_outcome'] = (binary_data['y'] > 0).astype(int)
sns.regplot(data=binary_data, x='x', y='binary_outcome', 
            logistic=True, ax=axes[1,0])
axes[1,0].set_title('Logistic Regression')

# Residual plot
sns.residplot(data=data, x='x', y='y', ax=axes[1,1])
axes[1,1].set_title('Residual Plot')
axes[1,1].axhline(y=0, color='red', linestyle='--')

plt.tight_layout()
plt.show()

# 6. FacetGrid for Multi-panel Plots
print("\n=== FacetGrid ===")
g = sns.FacetGrid(data, col='category', row='subcategory', 
                  height=3, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.7)
g.add_legend()
g.fig.suptitle('FacetGrid: Scatter Plots by Category and Subcategory', 
               y=1.02)
plt.show()

# 7. PairGrid for Pairwise Relationships
print("\n=== PairGrid ===")
subset_cols = ['x', 'y', 'z', 'value']
g = sns.PairGrid(data[subset_cols])
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot, cmap='Blues')
g.map_diag(sns.histplot, kde=True)
plt.suptitle('PairGrid: Multiple Visualizations', y=1.02)
plt.show()

# 8. Advanced Styling
print("\n=== Advanced Styling ===")
# Create custom palette
custom_palette = sns.color_palette("husl", n_colors=4)

# Style context
with sns.plotting_context("notebook", font_scale=1.2):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=data, x='category', y='value', 
                   palette=custom_palette, ax=ax)
    ax.set_title('Custom Styled Violin Plot')
    
    # Add statistical annotations
    from scipy import stats
    categories = data['category'].unique()
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories[i+1:], i+1):
            data1 = data[data['category'] == cat1]['value']
            data2 = data[data['category'] == cat2]['value']
            stat, p = stats.mannwhitneyu(data1, data2)
            if p < 0.05:
                ax.plot([i, j], [data['value'].max() * 1.1] * 2, 'k-')
                ax.text((i + j) / 2, data['value'].max() * 1.15, 
                       f'p={p:.3f}', ha='center')
    
plt.show()
```

## 🎯 Interview Questions

### Matplotlib Questions
1. **Q: What's the difference between `plt.plot()` and `ax.plot()`?**
   - A: `plt.plot()` uses the current axes (state-based), `ax.plot()` uses specific axes object (object-oriented).

2. **Q: How do you create a figure with shared axes?**
   - A: Use `fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)`.

3. **Q: Explain the transform system in Matplotlib.**
   - A: Transforms convert between coordinate systems: data, axes, figure, and display coordinates.

### Seaborn Questions
1. **Q: When would you use Seaborn over Matplotlib?**
   - A: For statistical visualizations, working with DataFrames, and when you need better default aesthetics.

2. **Q: What's the difference between `regplot` and `lmplot`?**
   - A: `regplot` plots on existing axes, `lmplot` creates a FacetGrid for conditional relationships.

3. **Q: How do you handle large datasets in Seaborn?**
   - A: Use sampling, alpha transparency, hexbin plots, or aggregate data before plotting.

## 📝 Practice Exercises

1. Create a dashboard with 6 different plot types showing relationships in a dataset
2. Build an animated visualization showing data changing over time
3. Create a publication-ready figure with proper labels, annotations, and formatting
4. Implement a custom visualization combining Matplotlib and Seaborn

## 🔗 Key Takeaways
- Matplotlib provides fine-grained control over every plot element
- Seaborn excels at statistical visualizations with minimal code
- Understanding both libraries allows flexible, powerful visualizations
- Always consider your audience when choosing visualization types and styles