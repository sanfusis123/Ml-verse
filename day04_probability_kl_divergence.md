# Day 4: Probability Basics and KL-Divergence

## 📚 Topics
- Probability fundamentals
- Probability distributions
- Bayes' theorem
- Information theory basics
- Kullback-Leibler (KL) divergence

---

## 1. Probability Fundamentals

### 📖 Core Concepts

#### Basic Definitions
- **Sample Space (Ω)**: Set of all possible outcomes
- **Event**: Subset of sample space
- **Probability**: Measure of likelihood, P(A) ∈ [0, 1]

#### Probability Axioms (Kolmogorov)
1. Non-negativity: P(A) ≥ 0
2. Normalization: P(Ω) = 1
3. Additivity: P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅

### 🔢 Mathematical Foundation

#### Joint, Marginal, and Conditional Probability
- **Joint**: P(A, B) = P(A ∩ B)
- **Marginal**: P(A) = Σ_B P(A, B)
- **Conditional**: P(A|B) = P(A, B) / P(B)

#### Independence
- A and B are independent if: P(A, B) = P(A) × P(B)
- Conditional independence: P(A, B|C) = P(A|C) × P(B|C)

#### Bayes' Theorem
```
P(A|B) = P(B|A) × P(A) / P(B)
```

### 💻 Probability Implementation Code

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import rel_entr, kl_div
import pandas as pd

# Set style and random seed
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# 1. Basic Probability Concepts
print("=== Basic Probability ===")

# Simulate coin flips
n_flips = 10000
coin_flips = np.random.choice(['H', 'T'], size=n_flips, p=[0.5, 0.5])
p_heads = np.sum(coin_flips == 'H') / n_flips
print(f"Probability of heads (empirical): {p_heads:.4f}")
print(f"Theoretical probability: 0.5000")

# Law of Large Numbers visualization
cumulative_heads = np.cumsum(coin_flips == 'H')
cumulative_prob = cumulative_heads / np.arange(1, n_flips + 1)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_prob, alpha=0.7, linewidth=1)
plt.axhline(y=0.5, color='red', linestyle='--', label='True probability')
plt.xlabel('Number of flips')
plt.ylabel('Empirical probability of heads')
plt.title('Law of Large Numbers: Coin Flips')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2. Joint and Conditional Probability
print("\n=== Joint and Conditional Probability ===")

# Create a contingency table
# Weather (Sunny/Rainy) vs Activity (Indoor/Outdoor)
data = {
    'Weather': ['Sunny'] * 60 + ['Rainy'] * 40,
    'Activity': ['Outdoor'] * 45 + ['Indoor'] * 15 + ['Outdoor'] * 10 + ['Indoor'] * 30
}
df = pd.DataFrame(data)

# Create contingency table
contingency = pd.crosstab(df['Weather'], df['Activity'])
print("Contingency Table:")
print(contingency)

# Calculate probabilities
total = contingency.sum().sum()
joint_prob = contingency / total
print("\nJoint Probabilities:")
print(joint_prob)

# Marginal probabilities
marginal_weather = joint_prob.sum(axis=1)
marginal_activity = joint_prob.sum(axis=0)
print("\nMarginal Probabilities:")
print(f"P(Weather): {marginal_weather.to_dict()}")
print(f"P(Activity): {marginal_activity.to_dict()}")

# Conditional probabilities
# P(Activity|Weather)
conditional_activity_given_weather = contingency.div(contingency.sum(axis=1), axis=0)
print("\nP(Activity|Weather):")
print(conditional_activity_given_weather)

# Verify Bayes' theorem
# P(Sunny|Outdoor) = P(Outdoor|Sunny) * P(Sunny) / P(Outdoor)
p_outdoor_given_sunny = conditional_activity_given_weather.loc['Sunny', 'Outdoor']
p_sunny = marginal_weather['Sunny']
p_outdoor = marginal_activity['Outdoor']
p_sunny_given_outdoor_bayes = (p_outdoor_given_sunny * p_sunny) / p_outdoor

# Direct calculation
p_sunny_given_outdoor_direct = joint_prob.loc['Sunny', 'Outdoor'] / p_outdoor

print(f"\nBayes' Theorem Verification:")
print(f"P(Sunny|Outdoor) using Bayes: {p_sunny_given_outdoor_bayes:.4f}")
print(f"P(Sunny|Outdoor) direct calc: {p_sunny_given_outdoor_direct:.4f}")

# 3. Probability Distributions
print("\n=== Probability Distributions ===")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Discrete distributions
# Binomial
n, p = 20, 0.3
x = np.arange(0, n+1)
pmf_binom = stats.binom.pmf(x, n, p)
axes[0].bar(x, pmf_binom, alpha=0.7)
axes[0].set_title(f'Binomial Distribution (n={n}, p={p})')
axes[0].set_xlabel('k')
axes[0].set_ylabel('P(X=k)')

# Poisson
λ = 5
x = np.arange(0, 15)
pmf_poisson = stats.poisson.pmf(x, λ)
axes[1].bar(x, pmf_poisson, alpha=0.7, color='green')
axes[1].set_title(f'Poisson Distribution (λ={λ})')
axes[1].set_xlabel('k')
axes[1].set_ylabel('P(X=k)')

# Geometric
p = 0.3
x = np.arange(1, 20)
pmf_geom = stats.geom.pmf(x, p)
axes[2].bar(x, pmf_geom, alpha=0.7, color='orange')
axes[2].set_title(f'Geometric Distribution (p={p})')
axes[2].set_xlabel('k')
axes[2].set_ylabel('P(X=k)')

# Continuous distributions
x = np.linspace(-4, 4, 1000)

# Normal
μ, σ = 0, 1
pdf_normal = stats.norm.pdf(x, μ, σ)
axes[3].plot(x, pdf_normal, 'b-', lw=2)
axes[3].fill_between(x, pdf_normal, alpha=0.3)
axes[3].set_title(f'Normal Distribution (μ={μ}, σ={σ})')
axes[3].set_xlabel('x')
axes[3].set_ylabel('f(x)')

# Exponential
λ = 1.5
x_exp = np.linspace(0, 5, 1000)
pdf_exp = stats.expon.pdf(x_exp, scale=1/λ)
axes[4].plot(x_exp, pdf_exp, 'g-', lw=2)
axes[4].fill_between(x_exp, pdf_exp, alpha=0.3, color='green')
axes[4].set_title(f'Exponential Distribution (λ={λ})')
axes[4].set_xlabel('x')
axes[4].set_ylabel('f(x)')

# Beta
a, b = 2, 5
x_beta = np.linspace(0, 1, 1000)
pdf_beta = stats.beta.pdf(x_beta, a, b)
axes[5].plot(x_beta, pdf_beta, 'r-', lw=2)
axes[5].fill_between(x_beta, pdf_beta, alpha=0.3, color='red')
axes[5].set_title(f'Beta Distribution (α={a}, β={b})')
axes[5].set_xlabel('x')
axes[5].set_ylabel('f(x)')

plt.tight_layout()
plt.show()

# 4. Bayes' Theorem Application - Medical Testing
print("\n=== Bayes' Theorem Application ===")

# Disease testing scenario
prevalence = 0.01  # 1% of population has disease
sensitivity = 0.95  # True positive rate
specificity = 0.98  # True negative rate

# P(Disease|Positive Test) using Bayes
p_disease = prevalence
p_no_disease = 1 - prevalence
p_positive_given_disease = sensitivity
p_positive_given_no_disease = 1 - specificity

# Total probability of positive test
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)

# Bayes' theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"Disease Testing Scenario:")
print(f"Prevalence: {prevalence:.1%}")
print(f"Test Sensitivity: {sensitivity:.1%}")
print(f"Test Specificity: {specificity:.1%}")
print(f"\nP(Disease|Positive Test) = {p_disease_given_positive:.1%}")
print(f"P(No Disease|Positive Test) = {1-p_disease_given_positive:.1%}")

# Visualization of Bayes update
prior_range = np.linspace(0.001, 0.1, 100)
posterior_positive = (sensitivity * prior_range) / (
    sensitivity * prior_range + (1 - specificity) * (1 - prior_range))

plt.figure(figsize=(10, 6))
plt.plot(prior_range * 100, posterior_positive * 100, 'b-', lw=2)
plt.axvline(x=prevalence * 100, color='red', linestyle='--', 
            label=f'Current prevalence: {prevalence:.1%}')
plt.axhline(y=p_disease_given_positive * 100, color='green', linestyle='--',
            label=f'Posterior: {p_disease_given_positive:.1%}')
plt.xlabel('Prior Probability of Disease (%)')
plt.ylabel('Posterior Probability Given Positive Test (%)')
plt.title('Bayes Update: Disease Probability After Positive Test')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 2. Information Theory and KL-Divergence

### 📖 Core Concepts

#### Entropy
Measure of uncertainty in a probability distribution:
```
H(P) = -Σ p(x) log p(x)
```

#### Cross-Entropy
Average number of bits needed to encode data from P using code optimized for Q:
```
H(P, Q) = -Σ p(x) log q(x)
```

#### KL-Divergence (Relative Entropy)
Measure of how one probability distribution differs from another:
```
D_KL(P || Q) = Σ p(x) log(p(x) / q(x)) = H(P, Q) - H(P)
```

### 🔢 Properties of KL-Divergence

1. **Non-negativity**: D_KL(P || Q) ≥ 0
2. **Not symmetric**: D_KL(P || Q) ≠ D_KL(Q || P)
3. **Zero iff identical**: D_KL(P || Q) = 0 ⟺ P = Q
4. **Not a true metric**: Doesn't satisfy triangle inequality

### 💻 Information Theory and KL-Divergence Code

```python
# 5. Information Theory Basics
print("\n=== Information Theory ===")

def entropy(probs):
    """Calculate entropy of a probability distribution"""
    # Handle zero probabilities
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def cross_entropy(p, q):
    """Calculate cross-entropy H(p, q)"""
    p = np.array(p)
    q = np.array(q)
    # Avoid log(0)
    q = np.clip(q, 1e-10, 1)
    return -np.sum(p * np.log2(q))

def kl_divergence(p, q):
    """Calculate KL divergence D_KL(p || q)"""
    p = np.array(p)
    q = np.array(q)
    # Avoid division by zero and log(0)
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log2(p / q))

# Example: Fair vs Biased Coin
fair_coin = [0.5, 0.5]
biased_coin = [0.7, 0.3]
very_biased_coin = [0.9, 0.1]

print(f"Entropy of fair coin: {entropy(fair_coin):.4f} bits")
print(f"Entropy of biased coin: {entropy(biased_coin):.4f} bits")
print(f"Entropy of very biased coin: {entropy(very_biased_coin):.4f} bits")

print(f"\nKL(fair || biased): {kl_divergence(fair_coin, biased_coin):.4f}")
print(f"KL(biased || fair): {kl_divergence(biased_coin, fair_coin):.4f}")
print(f"KL(fair || very_biased): {kl_divergence(fair_coin, very_biased_coin):.4f}")

# Visualize entropy as function of probability
p_values = np.linspace(0.001, 0.999, 100)
entropies = [entropy([p, 1-p]) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, entropies, 'b-', lw=2)
plt.axvline(x=0.5, color='red', linestyle='--', label='Maximum entropy')
plt.xlabel('Probability of Heads')
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Binary Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. KL-Divergence for Continuous Distributions
print("\n=== KL-Divergence for Continuous Distributions ===")

# Normal distributions
x = np.linspace(-5, 5, 1000)
μ1, σ1 = 0, 1
μ2, σ2 = 1, 1.5

p = stats.norm.pdf(x, μ1, σ1)
q = stats.norm.pdf(x, μ2, σ2)

# Normalize to ensure they sum to 1 (discrete approximation)
dx = x[1] - x[0]
p = p * dx
q = q * dx

# Calculate KL divergence (discrete approximation)
kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

# Analytical KL divergence for normal distributions
kl_analytical = np.log(σ2/σ1) + (σ1**2 + (μ1-μ2)**2)/(2*σ2**2) - 0.5

print(f"KL(N({μ1},{σ1}) || N({μ2},{σ2})) numerical: {kl_pq:.4f}")
print(f"KL(N({μ1},{σ1}) || N({μ2},{σ2})) analytical: {kl_analytical:.4f}")
print(f"KL(N({μ2},{σ2}) || N({μ1},{σ1})): {kl_qp:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, p/dx, 'b-', lw=2, label=f'P: N({μ1}, {σ1})')
plt.plot(x, q/dx, 'r-', lw=2, label=f'Q: N({μ2}, {σ2})')
plt.fill_between(x, p/dx, alpha=0.3, color='blue')
plt.fill_between(x, q/dx, alpha=0.3, color='red')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title(f'KL Divergence Between Normal Distributions\n' + 
          f'D_KL(P||Q) = {kl_pq:.4f}, D_KL(Q||P) = {kl_qp:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. KL-Divergence in Model Selection
print("\n=== KL-Divergence in Model Selection ===")

# True distribution (mixture of Gaussians)
def true_distribution(x):
    return 0.3 * stats.norm.pdf(x, -2, 0.5) + 0.7 * stats.norm.pdf(x, 1, 1)

# Candidate models
def model1(x):  # Single Gaussian
    return stats.norm.pdf(x, 0, 1.5)

def model2(x):  # Different single Gaussian
    return stats.norm.pdf(x, 0.5, 1.2)

def model3(x):  # Mixture of two Gaussians
    return 0.4 * stats.norm.pdf(x, -1.5, 0.7) + 0.6 * stats.norm.pdf(x, 1.2, 0.9)

x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]

# Calculate distributions
p_true = true_distribution(x) * dx
p_model1 = model1(x) * dx
p_model2 = model2(x) * dx
p_model3 = model3(x) * dx

# Calculate KL divergences
kl_1 = kl_divergence(p_true, p_model1)
kl_2 = kl_divergence(p_true, p_model2)
kl_3 = kl_divergence(p_true, p_model3)

print(f"KL(True || Model1): {kl_1:.4f}")
print(f"KL(True || Model2): {kl_2:.4f}")
print(f"KL(True || Model3): {kl_3:.4f}")
print(f"Best model: Model{np.argmin([kl_1, kl_2, kl_3]) + 1}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

models = [(p_model1, 'Model 1', kl_1), 
          (p_model2, 'Model 2', kl_2), 
          (p_model3, 'Model 3', kl_3)]

for i, (model, name, kl) in enumerate(models):
    ax = axes[i]
    ax.plot(x, p_true/dx, 'k-', lw=2, label='True Distribution')
    ax.plot(x, model/dx, 'r--', lw=2, label=name)
    ax.fill_between(x, p_true/dx, alpha=0.3, color='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title(f'{name}: KL = {kl:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Combined plot
ax = axes[3]
ax.plot(x, p_true/dx, 'k-', lw=3, label='True')
ax.plot(x, p_model1/dx, 'r--', lw=2, alpha=0.7, label=f'Model 1 (KL={kl_1:.3f})')
ax.plot(x, p_model2/dx, 'g--', lw=2, alpha=0.7, label=f'Model 2 (KL={kl_2:.3f})')
ax.plot(x, p_model3/dx, 'b--', lw=2, alpha=0.7, label=f'Model 3 (KL={kl_3:.3f})')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('All Models Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. Jensen-Shannon Divergence (Symmetric version)
print("\n=== Jensen-Shannon Divergence ===")

def js_divergence(p, q):
    """Calculate Jensen-Shannon divergence"""
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Compare distributions
dist1 = [0.1, 0.2, 0.3, 0.4]
dist2 = [0.25, 0.25, 0.25, 0.25]
dist3 = [0.4, 0.3, 0.2, 0.1]

print("Distribution comparisons:")
print(f"JS(dist1, dist2): {js_divergence(dist1, dist2):.4f}")
print(f"JS(dist2, dist3): {js_divergence(dist2, dist3):.4f}")
print(f"JS(dist1, dist3): {js_divergence(dist1, dist3):.4f}")
print(f"\nNote: JS divergence is symmetric")
print(f"JS(dist1, dist2) = JS(dist2, dist1): {js_divergence(dist2, dist1):.4f}")

# 9. KL-Divergence in Machine Learning
print("\n=== KL-Divergence in ML Applications ===")

# Variational Autoencoder (VAE) KL loss component
# KL divergence between N(μ, σ²) and N(0, 1)
def kl_gaussian(mu, log_var):
    """KL divergence between N(mu, exp(log_var)) and N(0, 1)"""
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

# Example latent variables
mu = np.array([0.5, -0.3, 0.1])
log_var = np.array([-0.2, 0.1, -0.5])

kl_loss = kl_gaussian(mu, log_var)
print(f"VAE KL Loss: {kl_loss:.4f}")

# Visualization of KL regularization effect
mu_range = np.linspace(-3, 3, 50)
log_var_range = np.linspace(-2, 1, 50)
MU, LOG_VAR = np.meshgrid(mu_range, log_var_range)

# Calculate KL for each (mu, log_var) pair
KL = -0.5 * (1 + LOG_VAR - MU**2 - np.exp(LOG_VAR))

plt.figure(figsize=(10, 8))
contour = plt.contourf(MU, LOG_VAR, KL, levels=20, cmap='viridis')
plt.colorbar(contour, label='KL Divergence')
plt.contour(MU, LOG_VAR, KL, levels=10, colors='white', alpha=0.5, linewidths=0.5)
plt.xlabel('μ')
plt.ylabel('log(σ²)')
plt.title('KL Divergence: N(μ, σ²) || N(0, 1)')
plt.scatter([0], [0], color='red', s=100, marker='*', 
           label='Target: N(0, 1)', zorder=5)
plt.legend()
plt.show()
```

## 🎯 Interview Questions

1. **Q: What's the difference between likelihood and probability?**
   - A: Probability is P(data|parameters) for fixed parameters. Likelihood is L(parameters|data) for fixed data.

2. **Q: Why is KL-divergence not a true distance metric?**
   - A: It's not symmetric (D_KL(P||Q) ≠ D_KL(Q||P)) and doesn't satisfy triangle inequality.

3. **Q: How is KL-divergence used in variational inference?**
   - A: Minimizes KL between approximate posterior q(z) and true posterior p(z|x) to find best approximation.

4. **Q: What's the relationship between cross-entropy and KL-divergence?**
   - A: D_KL(P||Q) = H(P,Q) - H(P), where H(P,Q) is cross-entropy and H(P) is entropy.

5. **Q: When would you use Jensen-Shannon divergence over KL-divergence?**
   - A: When you need a symmetric measure or bounded metric (JS ∈ [0, 1] when using log base 2).

## 📝 Practice Exercises

1. Implement Bayes' theorem for spam classification
2. Calculate KL-divergence between categorical distributions
3. Derive the KL-divergence formula for multivariate Gaussians
4. Implement importance sampling using KL-divergence

## 🔗 Key Takeaways
- Probability theory forms the foundation of machine learning
- Bayes' theorem enables updating beliefs with new evidence
- KL-divergence measures difference between distributions
- Information theory provides tools for model comparison
- These concepts are essential for understanding ML algorithms