# Algorithm Comparison Guide

Comprehensive comparison of ML algorithms implemented in this library to help you choose the right algorithm for your problem.

## Table of Contents

- [Regression Algorithms](#regression-algorithms)
- [Classification Algorithms](#classification-algorithms)
- [Clustering Algorithms](#clustering-algorithms)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Ensemble Methods](#ensemble-methods)
- [Neural Networks](#neural-networks)
- [Selection Guide](#selection-guide)

---

## Regression Algorithms

| Algorithm | Time Complexity | Space | Best For | Pros | Cons |
|-----------|-----------------|-------|----------|------|------|
| **Linear** | O(n³) or O(n²) | O(n²) | Simple linear relationships | Fast, interpretable, well-understood | Assumes linearity, sensitive to outliers |
| **Ridge** | O(n³) or O(n²) | O(n²) | High multicollinearity | Handles multicollinearity, stable | Requires λ tuning, less interpretable |
| **Lasso** | O(n²) × iterations | O(n) | Feature selection needed | Built-in feature selection, sparse | No closed-form, slower convergence |
| **Polynomial** | O(n³) or O(n²) | O(n^d) | Non-linear relationships | Flexible, captures curves | Overfitting risk, computational cost |

### When to Use:
- **Linear:** Baseline, simple relationships, interpretability needed
- **Ridge:** Correlated features, regularization needed
- **Lasso:** Feature selection, sparse solutions wanted
- **Polynomial:** Non-linear patterns, careful cross-validation needed

---

## Classification Algorithms

### Binary Classification Comparison

| Algorithm | Time | Space | Best For | Pros | Cons |
|-----------|------|-------|----------|------|------|
| **Logistic** | O(n²) | O(n) | Linear boundaries | Fast, interpretable, probabilistic | Linear only, needs feature scaling |
| **KNN** | O(nm) | O(nm) | Local patterns | Simple, no training, adapts | Slow prediction, curse of dimensionality |
| **Decision Tree** | O(n log n) | O(n) | Any pattern | Interpretable, handles non-linearity | Overfitting, unstable |
| **SVM** | O(n²) to O(n³) | O(n) | Complex patterns | Powerful, handles high-dim | Slow, hyperparameter tuning |
| **Naive Bayes** | O(n) | O(n) | Text, fast predictions | Very fast, works with small data | Strong independence assumption |
| **Neural Network** | Variable | Variable | Complex patterns | Highly flexible, non-linear | Needs lots of data, black-box |

### Multi-class Performance

| Algorithm | Support | Notes |
|-----------|---------|-------|
| Logistic | ✓ | One-vs-Rest or Softmax |
| KNN | ✓ | Natural extension |
| Decision Tree | ✓ | Handles naturally |
| SVM | ✓ | One-vs-One or One-vs-Rest |
| Naive Bayes | ✓ | Extends easily |
| Neural Network | ✓ | Multiple output neurons |

### When to Use:

**Linear Boundaries:**
- Logistic Regression (fast, interpretable)
- SVM Linear (if high-dimensional)

**Local Patterns:**
- KNN (small datasets)
- Decision Trees (interpretability)

**Complex Non-linear:**
- Neural Networks (large data)
- SVM RBF (medium data)
- Random Forest (ensemble needed)

**Fast Predictions:**
- Naive Bayes (text data)
- Decision Trees (real-time)
- Logistic (baseline)

---

## Clustering Algorithms

| Algorithm | Time | Space | Best For | Pros | Cons |
|-----------|------|-------|----------|------|------|
| **K-Means** | O(nkd × iter) | O(nk) | Spherical clusters | Fast, simple, scalable | Needs k, sensitive to init, spherical |
| **DBSCAN** | O(n²) | O(n) | Arbitrary shapes | No k needed, finds noise, any shape | Slow, parameter tuning |
| **Hierarchical** | O(n²) | O(n²) | Dendrograms needed | Dendrogram useful, flexible | Slow, merging irreversible |
| **GMM** | O(nkd × iter) | O(nkd) | Probabilistic labels | Soft clustering, probabilistic | Assumes Gaussian, EM convergence |

### Cluster Shape Handling

| Algorithm | Spherical | Elongated | Arbitrary |
|-----------|-----------|-----------|-----------|
| K-Means | ★★★★★ | ★★ | ★ |
| DBSCAN | ★★★★ | ★★★★★ | ★★★★★ |
| Hierarchical | ★★★ | ★★★ | ★★★★ |
| GMM | ★★★★ | ★★★★ | ★★★ |

### When to Use:

- **K-Means:** Fast clustering, known k, spherical clusters
- **DBSCAN:** Unknown number of clusters, arbitrary shapes, noise handling
- **Hierarchical:** Need dendrogram, exploratory analysis
- **GMM:** Probabilistic membership, soft assignments

---

## Dimensionality Reduction

| Algorithm | Supervised | Time | Best For | Pros | Cons |
|-----------|-----------|------|----------|------|------|
| **PCA** | No | O(n²d) | Linear patterns, visualization | Fast, unsupervised, interpretable | Linear only, variance != relevance |
| **LDA** | Yes | O(nd²) | Classification-aware reduction | Uses class info, supervised | Fewer components (k < c-1) |
| **t-SNE** | No | O(n²) | 2D/3D visualization | Preserves structure, beautiful plots | Slow, non-deterministic, non-linear |

### Comparison Table

| Aspect | PCA | LDA | t-SNE |
|--------|-----|-----|-------|
| Speed | Fast | Fast | Slow |
| Supervised | No | Yes | No |
| Non-linear | No | No | Yes |
| Visualization | Yes | Yes | Yes |
| Scalability | Good | Good | Poor |
| Interpretability | High | High | Low |

### When to Use:

- **PCA:** Quick preprocessing, visualization, feature reduction
- **LDA:** Classification task, need supervised reduction
- **t-SNE:** Explore 2D structure, understand clusters visually

---

## Ensemble Methods

| Algorithm | Type | Base Learner | Best For | Training Time | Memory |
|-----------|------|-------------|----------|---------------|--------|
| **Bagging** | Parallel | Unstable | Reduce variance | Moderate | Moderate |
| **Random Forest** | Parallel | Decision Trees | General purpose | Fast | Moderate |
| **AdaBoost** | Sequential | Weak learners | Reduce bias | Moderate | Low |
| **Gradient Boosting** | Sequential | Weak learners | Accuracy | Slow | Low |
| **Stacking** | Parallel | Heterogeneous | Complex patterns | Slow | High |

### Comparison Matrix

| Aspect | Bagging | RF | AdaBoost | GBM | Stacking |
|--------|---------|----|-----------|----|----------|
| Reduces Variance | ★★★★★ | ★★★★★ | ★★ | ★★★ | ★★★★ |
| Reduces Bias | ★★ | ★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| Speed | ★★★ | ★★★ | ★★ | ★ | ★ |
| Interpretability | ★★ | ★★★ | ★★ | ★ | ★ |
| Parallelizable | ★★★★★ | ★★★★★ | ★ | ★ | ★★★★★ |

### When to Use:

- **Bagging:** When base learner has high variance
- **Random Forest:** Default ensemble, feature importance needed
- **AdaBoost:** Hard examples need focus, reduce bias
- **Gradient Boosting:** Want best accuracy, have time/compute
- **Stacking:** Multiple diverse models available, maximum accuracy

---

## Neural Networks

| Type | Layers | Best For | Pros | Cons |
|------|--------|----------|------|------|
| **Perceptron** | 1 | Linear separation | Simple, interpretable | Very limited, linear only |
| **MLP** | 2+ | Non-linear patterns | Flexible, universal approx | Slow, needs lots of data, tuning |

### Architecture Selection

```
Simple patterns → Logistic/SVM
Non-linear patterns → MLP (1-2 hidden layers)
Complex high-dim → Deep networks
Image/Sequential → CNN/RNN (not in this library)
```

---

## Selection Guide

### By Problem Type

#### Regression
1. Start with: **Linear Regression**
2. If non-linear: **Polynomial Regression**
3. If multicollinearity: **Ridge Regression**
4. If feature selection: **Lasso Regression**
5. If complex: **Ensemble Methods** (Random Forest, Gradient Boosting)

#### Binary Classification
1. Start with: **Logistic Regression**
2. If non-linear: **Decision Tree** or **SVM**
3. If accuracy critical: **Random Forest** or **Gradient Boosting**
4. If very fast needed: **Naive Bayes**
5. If local patterns: **KNN**

#### Multi-class Classification
1. Start with: **Logistic Regression** (One-vs-Rest)
2. If interpretability: **Decision Tree**
3. If accuracy: **Random Forest**
4. If highly non-linear: **Neural Network**

#### Clustering
1. If know k: **K-Means**
2. If unknown k: **DBSCAN**
3. If need structure: **Hierarchical**
4. If probabilistic: **GMM**

#### Dimensionality Reduction
1. For preprocessing: **PCA**
2. For classification: **LDA**
3. For visualization: **t-SNE**

### By Dataset Size

| Size | Recommendation |
|------|-----------------|
| < 1,000 | KNN, Naive Bayes, Decision Trees |
| 1K - 100K | Logistic, SVM, Neural Networks |
| > 100K | Linear methods, Random Forest, Gradient Boosting |

### By Features

| Type | Recommendation |
|------|-----------------|
| Few (< 10) | Linear, KNN, Decision Trees |
| Medium (10-100) | SVM, Logistic, Ensemble |
| High (> 100) | Linear, SVM RBF, Dimensionality Reduction first |

### By Interpretability Needs

| Need | Best Choice |
|------|-------------|
| Maximum | Linear Regression, Decision Trees, Logistic |
| Moderate | SVM Linear, Random Forest (with feature importance) |
| Minimum OK | Neural Networks, Gradient Boosting |

### By Speed Requirements

| Requirement | Choice |
|-------------|--------|
| Real-time | Logistic, KNN, Decision Trees |
| Minutes | Naive Bayes, Linear SVM |
| Hours | Neural Networks, Gradient Boosting |

---

## Performance Expectations

### Typical Accuracy Ranges (on benchmark datasets)

| Task | Simple Baseline | Logistic/SVM | Ensemble | Deep Learning |
|------|-----------------|--------------|----------|---------------|
| Binary Classification | 50-60% | 75-85% | 85-92% | 90-98% |
| Multi-class (10 classes) | 10% | 60-75% | 75-88% | 85-95% |
| Regression (R²) | 0.3-0.5 | 0.6-0.8 | 0.8-0.92 | 0.85-0.98 |

### Computation Time (Relative)

```
Logistic Regression:        1x
Decision Tree:              1-2x
Random Forest (100 trees):  20x
Gradient Boosting:          30x
Neural Network:             50-200x
Stacking:                   100x
```

---

## Common Mistakes

### Regression
- ❌ Not checking linearity assumption
- ❌ Not scaling features for Ridge/Lasso
- ✅ Always visualize relationship first
- ✅ Try multiple regularization values

### Classification
- ❌ Using accuracy on imbalanced data
- ❌ Not tuning hyperparameters
- ✅ Use F1, precision-recall for imbalanced
- ✅ Cross-validate properly

### Clustering
- ❌ Choosing k arbitrarily
- ❌ Not scaling features
- ✅ Use elbow method or silhouette score
- ✅ Try multiple algorithms

### Ensemble
- ❌ Too many weak learners
- ❌ Not diversifying base learners
- ✅ Diversity is key for ensemble
- ✅ Monitor training curves

### Neural Networks
- ❌ Using too deep networks initially
- ❌ Not normalizing inputs
- ✅ Start simple, add complexity
- ✅ Monitor train/val loss

---

## Quick Reference Card

### Problem → Algorithm Decision Tree

```
Prediction Task?
├─ Continuous (Regression)
│  ├─ Linear relationship? → Linear Regression
│  ├─ Need feature selection? → Lasso
│  ├─ Multicollinearity? → Ridge
│  ├─ Non-linear? → Polynomial / RF / GBM
│  └─ Very complex? → Neural Network
│
├─ Categorical (Classification)
│  ├─ Need interpretability? → Decision Tree / Logistic
│  ├─ Linear separable? → Logistic / SVM Linear
│  ├─ Complex patterns? → RF / SVM RBF / GBM
│  ├─ Very fast needed? → Naive Bayes
│  └─ Accuracy critical? → Gradient Boosting / Stacking
│
└─ Unlabeled Data (Clustering)
   ├─ Know number of clusters? → K-Means
   ├─ Unknown clusters? → DBSCAN
   ├─ Need dendrogram? → Hierarchical
   └─ Want soft clusters? → GMM

Visualization Needed?
├─ High-dimensional to 2D? → PCA / t-SNE
├─ Classification-aware? → LDA
└─ Want structure preservation? → t-SNE
```

---

## Summary Recommendations

### Best Overall Performers
1. **Accuracy:** Gradient Boosting, Stacking, Neural Networks
2. **Speed:** Logistic, Naive Bayes, Decision Trees
3. **Simplicity:** Linear Regression, Logistic, KNN
4. **Interpretability:** Linear models, Decision Trees
5. **Robustness:** Random Forest, Ensemble methods

### Recommended Learning Path
1. Master: Linear Regression, Logistic, Decision Trees
2. Advance: SVM, Ensemble methods, Neural Networks
3. Specialize: Clustering, Dimensionality reduction, Advanced ensembles

### For Production Systems
- Start with ensemble methods (RF/GBM)
- Monitor with multiple metrics
- A/B test against baselines
- Retrain regularly
- Consider ensemble of multiple models