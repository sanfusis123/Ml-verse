# Day 4: Probability Basics & KL-Divergence

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Python Implementations](#python-implementations)
4. [ML Applications](#ml-applications)
5. [Interview Questions](#interview-questions)
6. [Practice Exercises](#practice-exercises)

## Core Concepts

### Probability Basics

**Probability** is the mathematical framework for quantifying uncertainty. In ML, it's fundamental for:
- Modeling uncertainty in predictions
- Building probabilistic models
- Understanding loss functions
- Bayesian inference

### Key Probability Concepts

1. **Sample Space (Ω)**: Set of all possible outcomes
2. **Event (E)**: Subset of the sample space
3. **Probability Measure P**: Function that assigns probabilities to events

### Probability Axioms (Kolmogorov)
1. Non-negativity: P(E) ≥ 0 for any event E
2. Normalization: P(Ω) = 1
3. Additivity: For disjoint events E₁, E₂, ..., P(⋃ᵢEᵢ) = ΣᵢP(Eᵢ)

### Types of Probability

1. **Joint Probability**: P(A, B) - probability of A and B occurring together
2. **Marginal Probability**: P(A) = Σ_B P(A, B)
3. **Conditional Probability**: P(A|B) = P(A, B) / P(B)

### Bayes' Theorem
```
P(A|B) = P(B|A) × P(A) / P(B)
```

### KL-Divergence (Kullback-Leibler Divergence)

**Definition**: Measures how one probability distribution differs from another reference distribution.

For discrete distributions P and Q:
```
D_KL(P||Q) = Σᵢ P(i) × log(P(i)/Q(i))
```

For continuous distributions:
```
D_KL(P||Q) = ∫ p(x) × log(p(x)/q(x)) dx
```

**Properties**:
- D_KL(P||Q) ≥ 0 (non-negative)
- D_KL(P||Q) = 0 if and only if P = Q
- Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
- Not a metric (doesn't satisfy triangle inequality)

## Mathematical Foundations

### Probability Distributions

#### Discrete Distributions

1. **Bernoulli Distribution**
   - PMF: P(X = k) = p^k × (1-p)^(1-k), k ∈ {0, 1}
   - Mean: μ = p
   - Variance: σ² = p(1-p)

2. **Binomial Distribution**
   - PMF: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
   - Mean: μ = np
   - Variance: σ² = np(1-p)

3. **Poisson Distribution**
   - PMF: P(X = k) = (λ^k × e^(-λ)) / k!
   - Mean: μ = λ
   - Variance: σ² = λ

#### Continuous Distributions

1. **Normal (Gaussian) Distribution**
   - PDF: f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
   - Mean: μ
   - Variance: σ²

2. **Exponential Distribution**
   - PDF: f(x) = λe^(-λx), x ≥ 0
   - Mean: μ = 1/λ
   - Variance: σ² = 1/λ²

### Information Theory Connection

**Entropy**: H(P) = -Σᵢ P(i) × log(P(i))

**Cross-Entropy**: H(P, Q) = -Σᵢ P(i) × log(Q(i))

**Relationship**: D_KL(P||Q) = H(P, Q) - H(P)

## Python Implementations

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import rel_entr
import seaborn as sns
from typing import List, Tuple, Union, Callable

class ProbabilityBasics:
    """Advanced probability computations for ML"""
    
    @staticmethod
    def joint_probability(data: np.ndarray) -> np.ndarray:
        """
        Calculate joint probability distribution from data
        
        Args:
            data: 2D array where each row is an observation
        
        Returns:
            Joint probability matrix
        """
        # Count occurrences
        unique_vals = [np.unique(data[:, i]) for i in range(data.shape[1])]
        joint_counts = np.zeros([len(uv) for uv in unique_vals])
        
        for row in data:
            indices = tuple(np.where(uv == row[i])[0][0] 
                          for i, uv in enumerate(unique_vals))
            joint_counts[indices] += 1
        
        # Normalize to get probabilities
        joint_prob = joint_counts / data.shape[0]
        return joint_prob
    
    @staticmethod
    def marginal_probability(joint_prob: np.ndarray, axis: int) -> np.ndarray:
        """
        Calculate marginal probability from joint distribution
        
        Args:
            joint_prob: Joint probability distribution
            axis: Axis to marginalize over
        
        Returns:
            Marginal probability distribution
        """
        return np.sum(joint_prob, axis=axis)
    
    @staticmethod
    def conditional_probability(joint_prob: np.ndarray, 
                              given_axis: int, 
                              given_value: int) -> np.ndarray:
        """
        Calculate conditional probability P(X|Y=y)
        
        Args:
            joint_prob: Joint probability distribution
            given_axis: Axis representing the given variable
            given_value: Value of the given variable
        
        Returns:
            Conditional probability distribution
        """
        # Get the slice for the given value
        slices = [slice(None)] * joint_prob.ndim
        slices[given_axis] = given_value
        
        conditional_unnorm = joint_prob[tuple(slices)]
        marginal = np.sum(conditional_unnorm)
        
        if marginal == 0:
            return np.zeros_like(conditional_unnorm)
        
        return conditional_unnorm / marginal
    
    @staticmethod
    def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
        """
        Apply Bayes' theorem
        
        Args:
            prior: P(H)
            likelihood: P(E|H)
            evidence: P(E)
        
        Returns:
            Posterior probability P(H|E)
        """
        if evidence == 0:
            raise ValueError("Evidence probability cannot be zero")
        return (likelihood * prior) / evidence

class KLDivergence:
    """KL-Divergence computations and visualizations"""
    
    @staticmethod
    def kl_divergence_discrete(p: np.ndarray, q: np.ndarray, 
                              epsilon: float = 1e-10) -> float:
        """
        Calculate KL divergence for discrete distributions
        
        Args:
            p: True distribution
            q: Approximating distribution
            epsilon: Small value to avoid log(0)
        
        Returns:
            KL divergence D_KL(P||Q)
        """
        # Ensure distributions are normalized
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Add epsilon to avoid log(0)
        q = q + epsilon
        q = q / np.sum(q)
        
        # Calculate KL divergence
        return np.sum(p * np.log(p / q))
    
    @staticmethod
    def kl_divergence_continuous(p_func: Callable, q_func: Callable, 
                               x_range: Tuple[float, float], 
                               num_points: int = 1000) -> float:
        """
        Approximate KL divergence for continuous distributions
        
        Args:
            p_func: PDF of true distribution
            q_func: PDF of approximating distribution
            x_range: Range for integration
            num_points: Number of points for approximation
        
        Returns:
            Approximated KL divergence
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        dx = (x_range[1] - x_range[0]) / num_points
        
        p_vals = p_func(x)
        q_vals = q_func(x)
        
        # Avoid numerical issues
        mask = (p_vals > 1e-10) & (q_vals > 1e-10)
        
        kl = np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask])) * dx
        return kl
    
    @staticmethod
    def kl_divergence_gaussians(mu1: float, sigma1: float, 
                               mu2: float, sigma2: float) -> float:
        """
        Analytical KL divergence between two Gaussians
        
        Args:
            mu1, sigma1: Parameters of first Gaussian
            mu2, sigma2: Parameters of second Gaussian
        
        Returns:
            KL divergence D_KL(N(μ₁,σ₁²)||N(μ₂,σ₂²))
        """
        return (np.log(sigma2/sigma1) + 
                (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)
    
    @staticmethod
    def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence (symmetric version of KL)
        
        Args:
            p, q: Probability distributions
        
        Returns:
            JS divergence
        """
        m = 0.5 * (p + q)
        kl_pm = KLDivergence.kl_divergence_discrete(p, m)
        kl_qm = KLDivergence.kl_divergence_discrete(q, m)
        return 0.5 * (kl_pm + kl_qm)

class ProbabilityVisualizer:
    """Visualization tools for probability concepts"""
    
    @staticmethod
    def plot_distributions_comparison(distributions: List[Tuple[str, np.ndarray, dict]],
                                    x_range: Tuple[float, float] = (-5, 5)):
        """
        Plot multiple probability distributions for comparison
        
        Args:
            distributions: List of (name, pdf_func, params)
            x_range: Range for x-axis
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        for name, dist_class, params in distributions:
            if dist_class == 'normal':
                y = stats.norm.pdf(x, **params)
            elif dist_class == 'exponential':
                y = stats.expon.pdf(x, **params)
            elif dist_class == 'uniform':
                y = stats.uniform.pdf(x, **params)
            
            ax1.plot(x, y, label=f"{name}: {params}", linewidth=2)
            ax2.plot(x, np.cumsum(y) * (x[1] - x[0]), label=f"{name} CDF", linewidth=2)
        
        ax1.set_title('Probability Density Functions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Cumulative Distribution Functions')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_kl_divergence(p: np.ndarray, q: np.ndarray, labels: List[str]):
        """
        Visualize KL divergence between two discrete distributions
        
        Args:
            p, q: Probability distributions
            labels: Labels for x-axis
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot distributions
        x = np.arange(len(p))
        width = 0.35
        
        ax1.bar(x - width/2, p, width, label='P (True)', alpha=0.8)
        ax1.bar(x + width/2, q, width, label='Q (Approx)', alpha=0.8)
        ax1.set_xlabel('Outcome')
        ax1.set_ylabel('Probability')
        ax1.set_title('Probability Distributions')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot KL divergence components
        kl_components = p * np.log(p / (q + 1e-10))
        ax2.bar(x, kl_components, alpha=0.8, color='red')
        ax2.set_xlabel('Outcome')
        ax2.set_ylabel('P(x) * log(P(x)/Q(x))')
        ax2.set_title(f'KL Divergence Components (Total: {np.sum(kl_components):.4f})')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class BayesianInference:
    """Bayesian inference implementations"""
    
    def __init__(self):
        self.prior = None
        self.posterior = None
        self.evidence = []
    
    def set_prior(self, prior: Union[np.ndarray, Callable]):
        """Set prior distribution"""
        self.prior = prior
        self.posterior = prior
    
    def update(self, likelihood: Union[np.ndarray, Callable], 
               observation: Union[int, float]):
        """
        Update posterior using Bayes' theorem
        
        Args:
            likelihood: Likelihood function P(evidence|hypothesis)
            observation: Observed evidence
        """
        if self.posterior is None:
            raise ValueError("Prior must be set before updating")
        
        # Calculate posterior
        if isinstance(self.posterior, np.ndarray):
            # Discrete case
            posterior_unnorm = self.posterior * likelihood[:, observation]
            self.posterior = posterior_unnorm / np.sum(posterior_unnorm)
        else:
            # Continuous case - would need numerical integration
            pass
        
        self.evidence.append(observation)
    
    def get_map_estimate(self) -> Union[int, float]:
        """Get Maximum A Posteriori estimate"""
        if isinstance(self.posterior, np.ndarray):
            return np.argmax(self.posterior)
        else:
            # For continuous, would need optimization
            pass

# Example usage and demonstrations
if __name__ == "__main__":
    # 1. Basic Probability Operations
    print("=== Basic Probability Operations ===")
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randint(0, 3, size=(1000, 2))
    
    prob_calc = ProbabilityBasics()
    joint_prob = prob_calc.joint_probability(data)
    
    print("Joint Probability Distribution:")
    print(joint_prob)
    print(f"\nSum of all probabilities: {np.sum(joint_prob):.4f}")
    
    # Marginal probabilities
    marginal_0 = prob_calc.marginal_probability(joint_prob, axis=1)
    marginal_1 = prob_calc.marginal_probability(joint_prob, axis=0)
    
    print(f"\nMarginal P(X): {marginal_0}")
    print(f"Marginal P(Y): {marginal_1}")
    
    # Conditional probability
    conditional = prob_calc.conditional_probability(joint_prob, given_axis=1, given_value=1)
    print(f"\nConditional P(X|Y=1): {conditional}")
    
    # 2. KL Divergence Examples
    print("\n=== KL Divergence Examples ===")
    
    kl_calc = KLDivergence()
    
    # Discrete distributions
    p_discrete = np.array([0.4, 0.3, 0.2, 0.1])
    q_discrete = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform
    
    kl_pq = kl_calc.kl_divergence_discrete(p_discrete, q_discrete)
    kl_qp = kl_calc.kl_divergence_discrete(q_discrete, p_discrete)
    js_div = kl_calc.jensen_shannon_divergence(p_discrete, q_discrete)
    
    print(f"D_KL(P||Q) = {kl_pq:.4f}")
    print(f"D_KL(Q||P) = {kl_qp:.4f}")
    print(f"JS(P,Q) = {js_div:.4f}")
    
    # Gaussian KL divergence
    kl_gauss = kl_calc.kl_divergence_gaussians(0, 1, 2, 1.5)
    print(f"\nKL divergence between N(0,1) and N(2,1.5²): {kl_gauss:.4f}")
    
    # 3. Visualizations
    print("\n=== Generating Visualizations ===")
    
    viz = ProbabilityVisualizer()
    
    # Compare distributions
    distributions = [
        ("Standard Normal", 'normal', {'loc': 0, 'scale': 1}),
        ("Shifted Normal", 'normal', {'loc': 2, 'scale': 1}),
        ("Wide Normal", 'normal', {'loc': 0, 'scale': 2})
    ]
    viz.plot_distributions_comparison(distributions)
    
    # Visualize KL divergence
    viz.visualize_kl_divergence(p_discrete, q_discrete, ['A', 'B', 'C', 'D'])
    
    # 4. Bayesian Inference Example
    print("\n=== Bayesian Inference Example ===")
    
    # Coin bias estimation
    bayes = BayesianInference()
    
    # Prior: uniform over possible biases
    prior = np.ones(11) / 11  # 11 possible biases: 0.0, 0.1, ..., 1.0
    bayes.set_prior(prior)
    
    # Likelihood for heads/tails
    biases = np.linspace(0, 1, 11)
    likelihood_heads = np.column_stack([1 - biases, biases])  # [P(T|bias), P(H|bias)]
    
    # Observe some coin flips (1 = heads, 0 = tails)
    observations = [1, 1, 0, 1, 1, 1, 0, 1]
    
    print("Prior:", bayes.posterior)
    
    for i, obs in enumerate(observations):
        bayes.update(likelihood_heads, obs)
        print(f"After flip {i+1} ({'H' if obs else 'T'}): "
              f"MAP estimate = {biases[bayes.get_map_estimate()]:.1f}")
    
    print("\nFinal posterior:", bayes.posterior)
    
    # Plot posterior evolution
    plt.figure(figsize=(10, 6))
    plt.bar(biases, bayes.posterior, alpha=0.7, width=0.08)
    plt.xlabel('Coin Bias (P(Heads))')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Distribution after 8 Coin Flips')
    plt.grid(True, alpha=0.3)
    plt.show()
```

## ML Applications

### 1. Loss Functions and KL Divergence

KL divergence is fundamental in many ML loss functions:

```python
class MLApplications:
    """ML applications of probability and KL divergence"""
    
    @staticmethod
    def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                          epsilon: float = 1e-7) -> float:
        """
        Cross-entropy loss for classification
        
        Note: Cross-entropy = H(p,q) = H(p) + D_KL(p||q)
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)
    
    @staticmethod
    def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray,
                 mu: np.ndarray, log_var: np.ndarray) -> float:
        """
        Variational Autoencoder loss = Reconstruction + KL divergence
        
        KL term ensures latent distribution is close to N(0,1)
        """
        # Reconstruction loss (e.g., MSE or BCE)
        reconstruction_loss = np.mean((x - x_reconstructed) ** 2)
        
        # KL divergence between N(μ,σ²) and N(0,1)
        kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        
        return reconstruction_loss + kl_loss
    
    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate mutual information I(X;Y) = D_KL(P(X,Y)||P(X)P(Y))
        
        Used in feature selection and information bottleneck methods
        """
        # Calculate joint and marginal distributions
        joint_prob = ProbabilityBasics.joint_probability(
            np.column_stack([x, y])
        )
        
        marginal_x = np.sum(joint_prob, axis=1)
        marginal_y = np.sum(joint_prob, axis=0)
        
        # Product of marginals
        product_marginals = np.outer(marginal_x, marginal_y)
        
        # MI as KL divergence
        mi = KLDivergence.kl_divergence_discrete(
            joint_prob.flatten(), 
            product_marginals.flatten()
        )
        
        return mi
```

### 2. Probabilistic Models

```python
class ProbabilisticModels:
    """Probabilistic model implementations"""
    
    @staticmethod
    def naive_bayes_classifier(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray) -> np.ndarray:
        """
        Simple Naive Bayes classifier implementation
        """
        classes = np.unique(y_train)
        n_features = X_train.shape[1]
        
        # Calculate class priors
        priors = {}
        for c in classes:
            priors[c] = np.sum(y_train == c) / len(y_train)
        
        # Calculate likelihoods (assuming Gaussian)
        likelihoods = {}
        for c in classes:
            X_c = X_train[y_train == c]
            likelihoods[c] = {
                'mean': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0) + 1e-6
            }
        
        # Predict
        predictions = []
        for x in X_test:
            posteriors = {}
            for c in classes:
                # Calculate log posterior (to avoid underflow)
                log_prior = np.log(priors[c])
                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * likelihoods[c]['std']**2) +
                    ((x - likelihoods[c]['mean'])**2) / (likelihoods[c]['std']**2)
                )
                posteriors[c] = log_prior + log_likelihood
            
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)
    
    @staticmethod
    def gmm_em(X: np.ndarray, n_components: int, max_iters: int = 100,
               tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gaussian Mixture Model using EM algorithm
        
        Returns:
            weights, means, covariances
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        weights = np.ones(n_components) / n_components
        means = X[np.random.choice(n_samples, n_components, replace=False)]
        covs = [np.eye(n_features) for _ in range(n_components)]
        
        log_likelihood_old = -np.inf
        
        for iteration in range(max_iters):
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((n_samples, n_components))
            
            for k in range(n_components):
                rv = stats.multivariate_normal(means[k], covs[k])
                responsibilities[:, k] = weights[k] * rv.pdf(X)
            
            # Normalize responsibilities
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            
            # M-step: Update parameters
            Nk = responsibilities.sum(axis=0)
            
            for k in range(n_components):
                weights[k] = Nk[k] / n_samples
                means[k] = (responsibilities[:, k] @ X) / Nk[k]
                
                # Update covariance
                diff = X - means[k]
                covs[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
                covs[k] += np.eye(n_features) * 1e-6  # Regularization
            
            # Check convergence
            log_likelihood = 0
            for k in range(n_components):
                rv = stats.multivariate_normal(means[k], covs[k])
                log_likelihood += np.sum(np.log(weights[k] * rv.pdf(X) + 1e-10))
            
            if log_likelihood - log_likelihood_old < tol:
                break
                
            log_likelihood_old = log_likelihood
        
        return weights, means, covs
```

### 3. Information-Theoretic Feature Selection

```python
def information_gain(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
    """
    Calculate information gain for a feature
    
    IG(Y|X) = H(Y) - H(Y|X) = I(X;Y)
    """
    # Discretize continuous feature if needed
    if len(np.unique(X[:, feature_idx])) > 10:
        X[:, feature_idx] = np.digitize(X[:, feature_idx], 
                                       bins=np.percentile(X[:, feature_idx], 
                                                         [25, 50, 75]))
    
    # Calculate entropy of y
    _, counts = np.unique(y, return_counts=True)
    p_y = counts / len(y)
    H_y = -np.sum(p_y * np.log2(p_y + 1e-10))
    
    # Calculate conditional entropy H(Y|X)
    feature_values = np.unique(X[:, feature_idx])
    H_y_given_x = 0
    
    for val in feature_values:
        mask = X[:, feature_idx] == val
        p_x_val = np.sum(mask) / len(y)
        
        if p_x_val > 0:
            y_given_x = y[mask]
            _, counts = np.unique(y_given_x, return_counts=True)
            p_y_given_x = counts / len(y_given_x)
            H_y_given_x_val = -np.sum(p_y_given_x * np.log2(p_y_given_x + 1e-10))
            H_y_given_x += p_x_val * H_y_given_x_val
    
    return H_y - H_y_given_x
```

## Interview Questions

### Basic Level

1. **What is the difference between joint, marginal, and conditional probability?**
   - Joint: P(A,B) - probability of both A and B
   - Marginal: P(A) = Σ_B P(A,B) - probability of A regardless of B
   - Conditional: P(A|B) = P(A,B)/P(B) - probability of A given B

2. **Explain Bayes' theorem and its significance in ML.**
   - Formula: P(A|B) = P(B|A)P(A)/P(B)
   - Allows updating beliefs with new evidence
   - Foundation of Bayesian ML methods

3. **What is KL divergence and why is it not a distance metric?**
   - Measures difference between distributions
   - Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
   - Doesn't satisfy triangle inequality

### Intermediate Level

4. **How is KL divergence used in VAEs?**
   - Regularization term in loss function
   - Forces latent distribution to match prior (usually N(0,1))
   - Balances reconstruction quality with latent space structure

5. **Explain the relationship between cross-entropy and KL divergence.**
   - H(P,Q) = H(P) + D_KL(P||Q)
   - Cross-entropy = entropy + KL divergence
   - Minimizing cross-entropy ≈ minimizing KL divergence

6. **What is mutual information and how is it related to KL divergence?**
   - I(X;Y) = D_KL(P(X,Y)||P(X)P(Y))
   - Measures shared information between variables
   - Used in feature selection and information bottleneck

### Advanced Level

7. **Derive the KL divergence between two Gaussian distributions.**
   ```
   For N(μ₁,σ₁²) and N(μ₂,σ₂²):
   D_KL = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
   ```

8. **Explain the connection between maximum likelihood estimation and KL divergence.**
   - MLE minimizes KL divergence between empirical and model distributions
   - argmax_θ L(θ) = argmin_θ D_KL(P_data||P_θ)

9. **How would you use KL divergence for distribution matching in GANs?**
   - Original GAN: min-max game approximates JS divergence
   - f-GAN generalizes using f-divergences including KL
   - Mode collapse related to asymmetry of KL

## Practice Exercises

### Exercise 1: Implement Entropy and Mutual Information
```python
def calculate_entropy(data: np.ndarray) -> float:
    """Calculate entropy H(X) of discrete data"""
    # TODO: Implement entropy calculation
    pass

def calculate_mutual_information(X: np.ndarray, Y: np.ndarray) -> float:
    """Calculate mutual information I(X;Y)"""
    # TODO: Implement using joint and marginal entropies
    pass
```

### Exercise 2: Bayesian A/B Testing
```python
class BayesianABTest:
    """Implement Bayesian A/B testing"""
    
    def __init__(self, prior_alpha: float = 1, prior_beta: float = 1):
        # TODO: Initialize Beta priors for both variants
        pass
    
    def update(self, variant: str, success: bool):
        # TODO: Update posterior with new observation
        pass
    
    def probability_A_better(self, n_samples: int = 10000) -> float:
        # TODO: Calculate P(θ_A > θ_B) using sampling
        pass
```

### Exercise 3: KL Divergence for Model Selection
```python
def model_selection_kl(data: np.ndarray, models: List[Callable], 
                      n_folds: int = 5) -> int:
    """
    Select best model using KL divergence on held-out data
    
    Returns:
        Index of best model
    """
    # TODO: Implement cross-validation with KL divergence scoring
    pass
```

### Exercise 4: Information Bottleneck
```python
class InformationBottleneck:
    """Implement information bottleneck method"""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def fit(self, X: np.ndarray, Y: np.ndarray, n_clusters: int):
        """
        Find clustering T that minimizes:
        L = I(X;T) - β*I(T;Y)
        """
        # TODO: Implement IB algorithm
        pass
```

### Exercise 5: Probabilistic PCA
```python
def probabilistic_pca(X: np.ndarray, n_components: int, 
                     max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement Probabilistic PCA using EM algorithm
    
    Returns:
        W: Loading matrix
        sigma2: Noise variance
    """
    # TODO: Implement PPCA
    pass
```

## Advanced Topics

### 1. f-Divergences
Generalization of KL divergence:
```
D_f(P||Q) = ∫ q(x) f(p(x)/q(x)) dx
```
- KL: f(t) = t log(t)
- JS: f(t) = -(t+1)log((t+1)/2) + t log(t)
- Total Variation: f(t) = |t-1|/2

### 2. Rényi Divergence
```
D_α(P||Q) = 1/(α-1) log(∫ p(x)^α q(x)^(1-α) dx)
```
- α→1: KL divergence
- α=0.5: Hellinger distance
- α=∞: Max divergence

### 3. Wasserstein Distance
- Considers geometry of probability space
- Used in Wasserstein GANs
- Better for distributions with disjoint support

## Summary

Key takeaways:
1. Probability provides the mathematical foundation for uncertainty in ML
2. KL divergence quantifies distribution differences
3. These concepts are fundamental to:
   - Loss functions (cross-entropy)
   - Generative models (VAEs, GANs)
   - Bayesian inference
   - Information theory in ML
4. Understanding the mathematical properties helps in:
   - Choosing appropriate loss functions
   - Debugging probabilistic models
   - Designing new algorithms