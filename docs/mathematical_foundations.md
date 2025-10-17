# Mathematical Foundations of ML Algorithms

This document provides mathematical formulations and theoretical foundations for all algorithms implemented in this library.

## Table of Contents

- [Regression](#regression)
- [Classification](#classification)
- [Clustering](#clustering)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Ensemble Methods](#ensemble-methods)
- [Neural Networks](#neural-networks)
- [Optimization](#optimization)

---

## Regression

### Linear Regression (Ordinary Least Squares)

**Problem Formulation:**
- Predict continuous target y given features X
- Model: `ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

**Cost Function (MSE):**
```
J(β) = (1/2m) * Σᵢ₌₁ᵐ (hβ(xⁱ) - yⁱ)²
```

**Solution Methods:**

1. **Normal Equation (Closed-form):**
   ```
   β = (XᵀX)⁻¹Xᵀy
   ```
   - Time complexity: O(n³)
   - Works well for small datasets (n < 10,000)

2. **Gradient Descent:**
   ```
   β := β - η * ∇J(β)
   ∇J(β) = (1/m) * Xᵀ(Xβ - y)
   ```
   - Time complexity: O(n²) per iteration
   - Better for large datasets

### Ridge Regression (L2 Regularization)

**Cost Function:**
```
J(β) = (1/2m) * Σ(hβ(x) - y)² + (λ/2m) * Σβⱼ²
```

**Normal Equation with Regularization:**
```
β = (XᵀX + λI)⁻¹Xᵀy
```

**Benefits:**
- Handles multicollinearity
- Prevents overfitting
- λ > 0: regularization strength

### Lasso Regression (L1 Regularization)

**Cost Function:**
```
J(β) = (1/2m) * Σ(hβ(x) - y)² + (λ/m) * Σ|βⱼ|
```

**Key Advantages:**
- Feature selection (shrinks some coefficients to zero)
- Sparse solutions
- No closed-form solution (uses coordinate descent)

**Coordinate Descent Update:**
```
βⱼ := S(ρⱼ, λ) where S(ρ, λ) = sign(ρ)max(|ρ| - λ, 0)
```

### Polynomial Regression

**Feature Expansion:**
```
Original: [x₁, x₂]
Degree 2: [1, x₁, x₂, x₁², x₁x₂, x₂²]
```

**Model:**
```
ŷ = β₀ + Σ βⱼ * (xⁱ)^(dⱼ)
```

---

## Classification

### Logistic Regression

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Hypothesis:**
```
P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))
```

**Cost Function (Binary Cross-Entropy):**
```
J(w) = -(1/m) * Σ[yⁱlog(hw(xⁱ)) + (1-yⁱ)log(1-hw(xⁱ))]
```

**Gradient:**
```
∇J(w) = (1/m) * Xᵀ(σ(Xw) - y)
```

### K-Nearest Neighbors

**Distance Metrics:**
```
Euclidean: d(x,y) = √(Σ(xᵢ - yᵢ)²)
Manhattan: d(x,y) = Σ|xᵢ - yᵢ|
Minkowski: d(x,y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

**Prediction:**
- Classification: Mode of k nearest neighbors
- Regression: Mean of k nearest neighbors

### Naive Bayes

**Bayes Theorem:**
```
P(y|X) = P(X|y) * P(y) / P(X)
```

**Naive Assumption:**
```
P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
```

**Gaussian Distribution:**
```
P(xⱼ|y) = (1/√(2πσⱼ²)) * exp(-(xⱼ - μⱼ)²/(2σⱼ²))
```

### Decision Trees (CART)

**Split Criterion - Gini Impurity:**
```
Gini(D) = 1 - Σₖ pₖ²
```

**Information Gain:**
```
IG(D,A) = Entropy(D) - Σ(|Dᵥ|/|D|) * Entropy(Dᵥ)
```

**Entropy:**
```
H(D) = -Σ pₖ * log₂(pₖ)
```

### Support Vector Machines

**Linear SVM Objective:**
```
minimize: (1/2)||w||² + C * Σ max(0, 1 - yᵢ(wᵀxᵢ + b))
```

**Hinge Loss:**
```
L(y, ŷ) = max(0, 1 - y*ŷ)
```

**Kernel Functions:**
```
Linear: K(x,y) = xᵀy
RBF: K(x,y) = exp(-γ||x-y||²)
Polynomial: K(x,y) = (γ*xᵀy + 1)^d
```

---

## Clustering

### K-Means

**Objective Function:**
```
J = Σᵢ₌₁ᴷ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Algorithm:**
1. Initialize K centroids randomly
2. Assign each point to nearest centroid: `Cᵢ = {x : ||x - μᵢ|| ≤ ||x - μⱼ||}`
3. Update centroids: `μᵢ = (1/|Cᵢ|) * Σₓ∈Cᵢ x`
4. Repeat until convergence

### DBSCAN

**Core Point Definition:**
```
Point p is core if |Nε(p)| ≥ MinPts
where Nε(p) = {q : d(p,q) ≤ ε}
```

**Density-Reachable:**
```
Point q is density-reachable from p if there exists
a chain p₁, p₂, ..., pₙ where p₁=p, pₙ=q, and each pᵢ₊₁
is directly density-reachable from pᵢ
```

### Gaussian Mixture Model (EM Algorithm)

**Model:**
```
P(x) = Σₖ πₖ * N(x | μₖ, Σₖ)
```

**E-step (Expectation):**
```
γₖ(xⁱ) = πₖ * N(xⁱ | μₖ, Σₖ) / Σⱼ πⱼ * N(xⁱ | μⱼ, Σⱼ)
```

**M-step (Maximization):**
```
πₖ = (1/m) * Σᵢ γₖ(xⁱ)
μₖ = Σᵢ γₖ(xⁱ) * xⁱ / Σᵢ γₖ(xⁱ)
Σₖ = Σᵢ γₖ(xⁱ) * (xⁱ - μₖ)(xⁱ - μₖ)ᵀ / Σᵢ γₖ(xⁱ)
```

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

**Goal:** Find directions of maximum variance

**Algorithm:**
1. Center data: `X_centered = X - μ`
2. Compute covariance: `Σ = (1/(m-1)) * X_centeredᵀ * X_centered`
3. Eigendecomposition: `Σ = U * Λ * Uᵀ`
4. Project: `X_transformed = X_centered * U[:, :k]`

**Explained Variance Ratio:**
```
EVR_i = λᵢ / Σⱼ λⱼ
```

### Linear Discriminant Analysis (LDA)

**Objective:** Maximize ratio of between-class to within-class variance

**Between-class Scatter:**
```
Sᵦ = Σᵢ nᵢ(μᵢ - μ)(μᵢ - μ)ᵀ
```

**Within-class Scatter:**
```
Sᵥ = Σᵢ Σₓ∈Cᵢ (x - μᵢ)(x - μᵢ)ᵀ
```

**Solution:**
```
Find w maximizing: w = argmax(wᵀSᵦw / wᵀSᵥw)
Solve: Sᵥ⁻¹Sᵦw = λw
```

### t-SNE

**High-dimensional Similarity (Gaussian):**
```
pⱼ|ᵢ = exp(-||xᵢ - xⱼ||²/2σᵢ²) / Σₖ exp(-||xᵢ - xₖ||²/2σᵢ²)
pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ) / 2n
```

**Low-dimensional Similarity (Student's t):**
```
qᵢⱼ = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖₗ (1 + ||yₖ - yₗ||²)⁻¹
```

**Cost Function (KL Divergence):**
```
C = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)
```

---

## Ensemble Methods

### Bagging (Bootstrap Aggregating)

**Algorithm:**
```
For b = 1 to B:
    Sample with replacement from training data
    Train model fᵦ on bootstrap sample
    
Prediction:
    Classification: mode{f₁(x), f₂(x), ..., fᵦ(x)}
    Regression: mean{f₁(x), f₂(x), ..., fᵦ(x)}
```

### Random Forest

**Key Difference from Bagging:**
- At each split, consider random subset of features
- Reduces correlation between trees

**Prediction:**
```
ŷ = mode{tree₁(x), ..., treeₙ(x)} (classification)
ŷ = (1/n) * Σ treeᵢ(x) (regression)
```

### AdaBoost

**Update Sample Weights:**
```
εₘ = Σᵢ wᵢ * I(hₘ(xⁱ) ≠ yⁱ) / Σᵢ wᵢ
αₘ = (1/2) * ln((1 - εₘ) / εₘ)
wᵢ := wᵢ * exp(-αₘ * yⁱ * hₘ(xⁱ))
```

**Final Prediction:**
```
H(x) = sign(Σₘ αₘ * hₘ(x))
```

### Gradient Boosting

**Sequential Error Correction:**
```
F₀(x) = argmin_γ Σ L(yⁱ, γ)

For m = 1 to M:
    rᵢₘ = -[∂L(yⁱ, F(xⁱ))/∂F(xⁱ)]_{F=F_{m-1}}
    Fit hₘ to residuals
    γₘ = argmin_γ Σ L(yⁱ, F_{m-1}(xⁱ) + γhₘ(xⁱ))
    Fₘ = F_{m-1} + ν*γₘ*hₘ
```

### Stacking

**Two-Level Model:**
```
Level 0: Train base learners L₁, L₂, ..., Lₖ
Level 1: Generate meta-features from base learner predictions
         Train meta-learner M on meta-features
         
Prediction: ŷ = M([L₁(x), L₂(x), ..., Lₖ(x)])
```

---

## Neural Networks

### Perceptron

**Update Rule:**
```
w := w + η * (y - ŷ) * x
b := b + η * (y - ŷ)
```

where `ŷ = sign(wᵀx + b)`

### Multi-Layer Perceptron

**Forward Pass:**
```
zˡ = Wˡ * aˡ⁻¹ + bˡ
aˡ = σ(zˡ)
```

**Backpropagation:**
```
δˡ = (Wˡ⁺¹)ᵀ * δˡ⁺¹ ⊙ σ'(zˡ)
∂L/∂Wˡ = δˡ * (aˡ⁻¹)ᵀ
∂L/∂bˡ = δˡ
```

### Activation Functions

**ReLU:**
```
f(z) = max(0, z)
f'(z) = 1 if z > 0, else 0
```

**Sigmoid:**
```
f(z) = 1 / (1 + e⁻ᶻ)
f'(z) = f(z) * (1 - f(z))
```

**Tanh:**
```
f(z) = tanh(z)
f'(z) = 1 - tanh²(z)
```

---

## Optimization

### Gradient Descent Variants

**Standard SGD:**
```
θ := θ - η * ∇J(θ)
```

**Momentum:**
```
v := β * v + η * ∇J(θ)
θ := θ - v
```

**Adam (Adaptive Moment Estimation):**
```
m := β₁ * m + (1 - β₁) * ∇J(θ)
v := β₂ * v + (1 - β₂) * (∇J(θ))²
m̂ := m / (1 - β₁ᵗ)
v̂ := v / (1 - β₂ᵗ)
θ := θ - η * m̂ / (√v̂ + ε)
```

### Learning Rate Schedules

**Step Decay:**
```
lr = initial_lr * drop_rate^floor(epoch / epochs_drop)
```

**Exponential Decay:**
```
lr = initial_lr * exp(-decay_rate * epoch)
```

**Cosine Annealing:**
```
lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * epoch / max_epochs))
```

---

## Key Concepts

### Bias-Variance Tradeoff

**Total Error = Bias² + Variance + Irreducible Error**

- **High Bias:** Model too simple, underfitting
- **High Variance:** Model too complex, overfitting
- **Goal:** Balance between bias and variance

### Regularization

**L1 (Lasso):** Σ|βⱼ| → Promotes sparsity
**L2 (Ridge):** Σβⱼ² → Shrinks coefficients smoothly

### Cross-Validation

**K-Fold:** Divide data into K folds, train on K-1, test on 1

### Evaluation Metrics

**Classification:**
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * (Precision * Recall) / (Precision + Recall)

**Regression:**
- MSE: (1/n) * Σ(y - ŷ)²
- RMSE: √MSE
- MAE: (1/n) * Σ|y - ŷ|
- R²: 1 - (SS_res / SS_tot)

---

## References

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
4. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
5. Ng, A. (2021). Machine Learning Yearning