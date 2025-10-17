# API Reference

Complete API documentation for all algorithms and utilities in this library.

## Table of Contents

- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Ensemble Methods](#ensemble-methods)
- [Metrics](#metrics)
- [Preprocessing](#preprocessing)
- [Visualization](#visualization)
- [Optimizers](#optimizers)
- [Common Usage Patterns](#common-usage-patterns)
- [Type Hints](#type-hints)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)

---

## Supervised Learning

### Regression

#### LinearRegression
```python
from algorithms.supervised.regression import LinearRegression

model = LinearRegression(learning_rate=0.01, n_iterations=1000, 
                         fit_intercept=True, method='gradient_descent')
model.fit(X, y)
predictions = model.predict(X_test)
r2_score = model.score(X_test, y_test)
params = model.get_params()
```

**Methods:**
- `fit(X, y)` - Fit model to training data
- `predict(X)` - Make predictions
- `score(X, y)` - Calculate R² score
- `get_params()` - Return model parameters

#### RidgeRegression
```python
from algorithms.supervised.regression import RidgeRegression

model = RidgeRegression(alpha=1.0, learning_rate=0.01, 
                        n_iterations=1000, method='normal_equation')
model.fit(X, y)
predictions = model.predict(X_test)
```

**Parameters:**
- `alpha` - Regularization strength
- `learning_rate` - Learning rate for gradient descent
- `method` - 'normal_equation' or 'gradient_descent'

#### LassoRegression
```python
from algorithms.supervised.regression import LassoRegression

model = LassoRegression(alpha=0.1, learning_rate=0.01, 
                        n_iterations=1000, method='coordinate_descent')
model.fit(X, y)
```

**Parameters:**
- `alpha` - L1 regularization strength
- `method` - 'coordinate_descent' or 'subgradient'

#### PolynomialRegression
```python
from algorithms.supervised.regression import PolynomialRegression

model = PolynomialRegression(degree=2, learning_rate=0.01)
model.fit(X, y)
predictions = model.predict(X_test)
```

**Parameters:**
- `degree` - Polynomial degree
- `regularization` - None, 'l1', or 'l2'

#### LogisticRegression
```python
from algorithms.supervised.regression import LogisticRegression

model = LogisticRegression(learning_rate=0.01, n_iterations=1000,
                           regularization='l2', lambda_reg=0.01)
model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Methods:**
- `predict_proba(X)` - Predict probabilities for positive class

---

### Classification

#### KNearestNeighbors
```python
from algorithms.supervised.classification import KNearestNeighbors

model = KNearestNeighbors(n_neighbors=5, metric='euclidean',
                          weights='uniform', task='classification')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```

**Parameters:**
- `n_neighbors` - Number of neighbors
- `metric` - 'euclidean', 'manhattan', or 'minkowski'
- `weights` - 'uniform' or 'distance'
- `task` - 'classification' or 'regression'

#### GaussianNaiveBayes
```python
from algorithms.supervised.classification import GaussianNaiveBayes

model = GaussianNaiveBayes(var_smoothing=1e-9)
model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Methods:**
- `predict_proba(X)` - Predict class probabilities

#### DecisionTree
```python
from algorithms.supervised.classification import DecisionTree

model = DecisionTree(max_depth=10, min_samples_split=2,
                     criterion='gini', task='classification')
model.fit(X, y)
predictions = model.predict(X_test)
depth = model.get_depth()
```

**Parameters:**
- `criterion` - 'gini', 'entropy', or 'mse'
- `task` - 'classification' or 'regression'
- `max_features` - Number of features to consider

#### SVM
```python
from algorithms.supervised.classification import SVM

model = SVM(learning_rate=0.001, lambda_param=0.01,
            n_iterations=1000, kernel='linear', gamma=0.1)
model.fit(X, y)
predictions = model.predict(X_test)
scores = model.decision_function(X_test)
```

**Parameters:**
- `kernel` - 'linear', 'rbf', or 'poly'
- `gamma` - Kernel coefficient for RBF/poly

#### Perceptron
```python
from algorithms.supervised.neural_networks import Perceptron

model = Perceptron(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X_test)
```

#### MultilayerPerceptron
```python
from algorithms.supervised.neural_networks import MLP

model = MLP(hidden_layers=[64, 32], learning_rate=0.01,
            n_iterations=1000, activation='relu', task='classification')
model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Parameters:**
- `hidden_layers` - List of hidden layer sizes
- `activation` - 'relu', 'sigmoid', or 'tanh'
- `batch_size` - Mini-batch size

#### BackpropagationNN
```python
from algorithms.supervised.neural_networks import BackpropagationNN

model = BackpropagationNN(layer_sizes=[20, 16, 8, 1],
                          activation='relu', learning_rate=0.01)
model.fit(X, y)
predictions = model.predict(X_test)
```

**Parameters:**
- `layer_sizes` - List including input and output layer sizes
- `output_activation` - 'sigmoid', 'softmax', or 'linear'
- `loss` - 'mse', 'binary_cross_entropy', or 'categorical_cross_entropy'

---

## Unsupervised Learning

### Clustering

#### KMeans
```python
from algorithms.unsupervised.clustering import KMeans

model = KMeans(n_clusters=3, max_iterations=300, init='kmeans++')
model.fit(X)
labels = model.predict(X_new)
centroids = model.centroids
inertia = model.inertia
```

**Parameters:**
- `init` - 'random' or 'kmeans++'
- `max_iterations` - Maximum iterations
- `tol` - Convergence tolerance

#### DBSCAN
```python
from algorithms.unsupervised.clustering import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
labels = model.fit_predict(X)
n_clusters = model.n_clusters
```

**Returns:**
- Labels: -1 for noise points, cluster ID otherwise

#### HierarchicalClustering
```python
from algorithms.unsupervised.clustering import HierarchicalClustering

model = HierarchicalClustering(n_clusters=3, linkage='average')
labels = model.fit_predict(X)
```

**Parameters:**
- `linkage` - 'single', 'complete', 'average', or 'ward'

#### GaussianMixtureModel
```python
from algorithms.unsupervised.clustering import GaussianMixtureModel

model = GaussianMixtureModel(n_components=3, covariance_type='full')
model.fit(X)
labels = model.predict(X)
probabilities = model.predict_proba(X)
samples = model.sample(n_samples=10)
```

**Parameters:**
- `covariance_type` - 'full', 'diag', or 'spherical'

---

### Dimensionality Reduction

#### PCA
```python
from algorithms.unsupervised.dimensionality_reduction import PCA

model = PCA(n_components=2, whiten=False)
X_transformed = model.fit_transform(X)
X_reconstructed = model.inverse_transform(X_transformed)
```

**Attributes:**
- `explained_variance_` - Variance explained by each component
- `explained_variance_ratio_` - Ratio of variance explained
- `components_` - Principal axes

#### LDA
```python
from algorithms.unsupervised.dimensionality_reduction import LDA

model = LDA(n_components=2)
X_transformed = model.fit_transform(X, y)
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```

#### tSNE
```python
from algorithms.unsupervised.dimensionality_reduction import TSNE

model = TSNE(n_components=2, perplexity=30.0, n_iterations=1000)
X_embedded = model.fit_transform(X)
```

**Parameters:**
- `perplexity` - Related to number of nearest neighbors (5-50)
- `early_exaggeration` - Factor for early iterations
- `learning_rate` - Optimization learning rate

---

## Ensemble Methods

#### RandomForest
```python
from algorithms.ensemble import RandomForest

model = RandomForest(n_estimators=100, max_depth=10,
                     max_features='sqrt', task='classification', oob_score=True)
model.fit(X, y)
predictions = model.predict(X_test)
oob_score = model.oob_score_
```

**Parameters:**
- `max_features` - 'sqrt', 'log2', int, or None
- `bootstrap` - Use bootstrap samples
- `oob_score` - Calculate out-of-bag score

#### AdaBoost
```python
from algorithms.ensemble import AdaBoost

model = AdaBoost(n_estimators=50, learning_rate=1.0)
model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
scores = model.staged_score(X_test, y_test)
```

**Methods:**
- `staged_score(X, y)` - Score at each boosting iteration

#### GradientBoosting
```python
from algorithms.ensemble import GradientBoosting

model = GradientBoosting(n_estimators=100, learning_rate=0.1,
                         max_depth=3, task='regression', loss='mse')
model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # For classification
```

**Parameters:**
- `loss` - 'mse' (regression) or 'log_loss' (classification)
- `subsample` - Fraction of samples for each tree

#### Bagging
```python
from algorithms.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                          n_estimators=10, max_samples=1.0, oob_score=True)
model.fit(X, y)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Parameters:**
- `base_estimator` - Any sklearn-compatible estimator
- `max_features` - Fraction/number of features

#### Stacking
```python
from algorithms.ensemble import StackingClassifier

base_learners = [
    ('tree', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression())
]
meta_learner = LogisticRegression()

model = StackingClassifier(base_learners=base_learners,
                           meta_learner=meta_learner, use_proba=True)
model.fit(X, y)
predictions = model.predict(X_test)
```

---

## Metrics

### Classification Metrics

```python
from algorithms.utils.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)

# Binary classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Multi-class
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Probabilities
auc = roc_auc_score(y_true, y_proba)
loss = log_loss(y_true, y_proba)

# Matrix and Report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)
```

### Regression Metrics

```python
from algorithms.utils.metrics import (
    mean_squared_error, root_mean_squared_error,
    mean_absolute_error, r2_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Clustering Metrics

```python
from algorithms.utils.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

silhouette = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_score = davies_bouldin_score(X, labels)
```

---

## Preprocessing

### Scaling

```python
from algorithms.utils.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
X_inverse = scaler.inverse_transform(X_scaled)

# MinMax scaling (scale to range)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_inverse = scaler.inverse_transform(X_scaled)

# Robust scaling (robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Encoding

```python
from algorithms.utils.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding (encode target labels to integers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_original = le.inverse_transform(y_encoded)
classes = le.classes_

# One-hot encoding (categorical features)
ohe = OneHotEncoder()
X_onehot = ohe.fit_transform(X_categorical)
X_original = ohe.inverse_transform(X_onehot)
```

### Data Splitting

```python
from algorithms.utils.preprocessing import train_test_split, k_fold_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# K-fold cross-validation
for train_idx, test_idx in k_fold_split(X, y, n_splits=5, shuffle=True):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

### Feature Engineering

```python
from algorithms.utils.preprocessing import PolynomialFeatures, SimpleImputer

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
n_output_features = poly.n_output_features_

# Impute missing values (NaN)
imputer = SimpleImputer(strategy='mean')  # 'mean', 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)
```

---

## Visualization

### Classification Plots

```python
from algorithms.utils.visualization import (
    plot_decision_boundary, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve
)

# Plot decision boundary for 2D data
fig = plot_decision_boundary(model, X_2d, y, title="Decision Boundary",
                             figsize=(10, 6), resolution=100)

# Plot confusion matrix heatmap
fig = plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'],
                            normalize=False, figsize=(8, 6))

# Plot ROC curve with AUC
fig = plot_roc_curve(y_true, y_scores, figsize=(8, 6))

# Plot Precision-Recall curve
fig = plot_precision_recall_curve(y_true, y_scores, figsize=(8, 6))
```

### Regression Plots

```python
from algorithms.utils.visualization import plot_regression_results, plot_learning_curve

# Plot predicted vs actual + residuals
fig = plot_regression_results(y_true, y_pred, figsize=(12, 5))

# Plot learning curves
fig = plot_learning_curve(train_sizes, train_scores, val_scores,
                         title="Learning Curve", figsize=(8, 6))
```

### Clustering Plots

```python
from algorithms.utils.visualization import plot_clusters

# Plot 2D clusters with centers
fig = plot_clusters(X_2d, labels, centers=centroids,
                   title="Clusters", figsize=(10, 6))
```

### Dimensionality Reduction Plots

```python
from algorithms.utils.visualization import plot_pca_variance, plot_2d_embedding

# Plot explained variance per PCA component
fig = plot_pca_variance(explained_variance_ratio, figsize=(10, 6))

# Plot 2D embedding (PCA, t-SNE, LDA)
fig = plot_2d_embedding(X_embedded, y=labels, title="t-SNE Embedding",
                       figsize=(10, 8))
```

### General Plots

```python
from algorithms.utils.visualization import (
    plot_feature_importance, plot_training_history,
    plot_data_distribution, plot_correlation_matrix,
    plot_model_comparison
)

# Plot feature importance
fig = plot_feature_importance(feature_names, importances,
                             top_n=10, figsize=(10, 6))

# Plot training history (loss/metrics over epochs)
fig = plot_training_history({'loss': loss_history, 'val_loss': val_loss_history},
                           figsize=(12, 5))

# Plot data distribution
fig = plot_data_distribution(X, feature_names, figsize=(12, 8))

# Plot correlation matrix heatmap
fig = plot_correlation_matrix(X, feature_names, figsize=(10, 8))

# Compare multiple models
fig = plot_model_comparison(model_names, scores, metric_name='Accuracy',
                           figsize=(10, 6))
```

---

## Optimizers

### Gradient-Based Optimizers

```python
from algorithms.utils.optimizers import (
    GradientDescent, MomentumOptimizer, NesterovMomentum,
    AdaGrad, RMSprop, Adam, AdaMax, Nadam
)

# Standard SGD
optimizer = GradientDescent(learning_rate=0.01)
updated_params = optimizer.update(params, gradients)

# Momentum SGD
optimizer = MomentumOptimizer(learning_rate=0.01, momentum=0.9)
updated_params = optimizer.update(params, gradients)

# Nesterov momentum
optimizer = NesterovMomentum(learning_rate=0.01, momentum=0.9)
updated_params = optimizer.update(params, gradients)

# AdaGrad (per-parameter learning rate)
optimizer = AdaGrad(learning_rate=0.01, epsilon=1e-8)
updated_params = optimizer.update(params, gradients)

# RMSprop
optimizer = RMSprop(learning_rate=0.001, decay_rate=0.9, epsilon=1e-8)
updated_params = optimizer.update(params, gradients)

# Adam (most popular - recommended!)
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
updated_params = optimizer.update(params, gradients)

# AdaMax (Adam with infinity norm)
optimizer = AdaMax(learning_rate=0.002, beta1=0.9, beta2=0.999)
updated_params = optimizer.update(params, gradients)

# Nadam (Nesterov + Adam)
optimizer = Nadam(learning_rate=0.001, beta1=0.9, beta2=0.999)
updated_params = optimizer.update(params, gradients)

# Reset optimizer state between epochs
optimizer.reset()
```

**Optimizer Comparison:**

| Optimizer | Best For | Speed | Memory |
|-----------|----------|-------|--------|
| SGD | Simple problems | Fast | Low |
| Momentum | Accelerating convergence | Medium | Low |
| AdaGrad | Sparse gradients | Medium | Medium |
| RMSprop | Non-stationary problems | Fast | Medium |
| Adam | General use (recommended) | Fast | Medium |
| Nadam | Fine-tuning | Medium | Medium |

### Learning Rate Schedules

```python
from algorithms.utils.optimizers import StepDecay, ExponentialDecay, CosineAnnealing

# Step decay: drop learning rate at intervals
schedule = StepDecay(initial_lr=0.1, drop_rate=0.5, epochs_drop=10)
lr_epoch_5 = schedule(5)    # 0.1
lr_epoch_10 = schedule(10)  # 0.05
lr_epoch_20 = schedule(20)  # 0.025

# Exponential decay: smooth exponential decrease
schedule = ExponentialDecay(initial_lr=0.1, decay_rate=0.1)
lr_epoch_0 = schedule(0)    # 0.1
lr_epoch_10 = schedule(10)  # ~0.0366

# Cosine annealing: cosine curve schedule
schedule = CosineAnnealing(max_lr=0.1, min_lr=0.001, max_epochs=100)
lr_epoch_0 = schedule(0)    # 0.1
lr_epoch_50 = schedule(50)  # 0.0505
lr_epoch_100 = schedule(100) # 0.001
```

---

## Common Usage Patterns

### Standard ML Workflow

```python
from algorithms.utils.preprocessing import train_test_split, StandardScaler
from algorithms.supervised.classification import LogisticRegression
from algorithms.utils.metrics import accuracy_score, classification_report

# 1. Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Preprocess (scale features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# 4. Evaluate
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, predictions))

# 5. Tune hyperparameters and repeat
```

### Cross-Validation Workflow

```python
from algorithms.utils.preprocessing import k_fold_split
from algorithms.utils.metrics import accuracy_score
import numpy as np

scores = []
for train_idx, test_idx in k_fold_split(X, y, n_splits=5, shuffle=True):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    scores.append(score)

mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"Cross-validation: {mean_score:.4f} (+/- {std_score:.4f})")
```

### Ensemble Workflow

```python
from algorithms.ensemble import RandomForest, GradientBoosting, AdaBoost
from algorithms.utils.metrics import accuracy_score

models = [
    ('Random Forest', RandomForest(n_estimators=100)),
    ('Gradient Boosting', GradientBoosting(n_estimators=100)),
    ('AdaBoost', AdaBoost(n_estimators=50)),
]

for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    print(f"{name}: {score:.4f}")
```

### Hyperparameter Tuning

```python
from algorithms.supervised.classification import SVM

best_score = 0
best_params = {}

# Grid search
for lambda_param in [0.1, 1, 10, 100]:
    for kernel in ['linear', 'rbf']:
        for gamma in [0.001, 0.1, 1]:
            model = SVM(lambda_param=lambda_param, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            if score > best_score:
                best_score = score
                best_params = {'lambda': lambda_param, 'kernel': kernel, 'gamma': gamma}

print(f"Best params: {best_params}")
print(f"Best score: {best_score:.4f}")
```

---

## Type Hints

All functions use type hints for clarity:

```python
from typing import Optional, List, Tuple, Union
import numpy as np

def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassName':
    """Fit model to data."""
    return self

def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions."""
    return predictions

def score(self, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate performance metric."""
    return score
```

**Common Type Annotations:**
- `np.ndarray` - NumPy array
- `float` - Floating point
- `int` - Integer
- `bool` - Boolean
- `Optional[type]` - Optional type
- `List[type]` - List of type
- `Tuple[type, ...]` - Tuple
- `Union[type1, type2]` - Either type

---

## Error Handling

All algorithms validate inputs and raise appropriate errors:

```python
# Input validation
model.fit(X, y)  # Raises ValueError if y has wrong length
model.predict(X)  # Raises ValueError if X has wrong features

# Common errors
try:
    model.fit(X, y)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Training failed: {e}")
```

**Error Types:**
- `ValueError` - Invalid input shapes or values
- `TypeError` - Wrong data type passed
- `RuntimeError` - Execution error during training

---

## Performance Tips

1. **Scale features before training:**
   ```python
   scaler = StandardScaler()
   X = scaler.fit_transform(X)
   ```

2. **Use random_state for reproducibility:**
   ```python
   model = Algorithm(random_state=42)
   ```

3. **Monitor training with loss history:**
   ```python
   plt.plot(model.cost_history)
   plt.show()
   ```

4. **Choose right algorithm by data size:**
   - Small (< 1,000): KNN, Naive Bayes
   - Medium (1K-100K): Logistic, SVM, Trees
   - Large (> 100K): Linear, Ensemble

5. **Always cross-validate for robust estimates:**
   ```python
   for train_idx, test_idx in k_fold_split(X, y, n_splits=5):
       # Evaluate on each fold
   ```

---

## Quick Reference Card

| Task | Algorithm | Import |
|------|-----------|--------|
| Linear Regression | LinearRegression | `algorithms.supervised.regression` |
| Logistic/Binary | LogisticRegression | `algorithms.supervised.regression` |
| Multi-class | DecisionTree, RF | `algorithms.supervised.classification` |
| Clustering | KMeans, DBSCAN | `algorithms.unsupervised.clustering` |
| Reduce Dimensions | PCA, t-SNE | `algorithms.unsupervised.dimensionality_reduction` |
| Ensemble | RandomForest, GBM | `algorithms.ensemble` |
| Metrics | accuracy_score, f1_score | `algorithms.utils.metrics` |
| Scaling | StandardScaler | `algorithms.utils.preprocessing` |
| Visualization | plot_confusion_matrix | `algorithms.utils.visualization` |
| Optimization | Adam | `algorithms.utils.optimizers` |

---

## Additional Resources

- **Mathematical Foundations:** See `docs/mathematical_foundations.md`
- **Algorithm Comparison:** See `docs/algorithm_comparison.md`
- **Contributing Guide:** See `docs/contributing.md`
- **Working Examples:** Check `if __name__ == "__main__"` in each algorithm file
- **Unit Tests:** See `tests/` directory for validation examples#