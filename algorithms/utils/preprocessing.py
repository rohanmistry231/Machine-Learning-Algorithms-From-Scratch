"""
Data Preprocessing Utilities Implementation from Scratch

This module provides various data preprocessing techniques including
scaling, encoding, and data splitting.
"""

import numpy as np
from typing import Optional, Tuple, List


# ==================== SCALING ====================

class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    z = (x - μ) / σ
    
    Attributes
    ----------
    mean_ : np.ndarray
        Mean of each feature
    scale_ : np.ndarray
        Standard deviation of each feature
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute mean and std for later scaling."""
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_ = X.shape[1]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize features."""
        X = np.array(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization."""
        X = np.array(X)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1]).
    
    X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Attributes
    ----------
    data_min_ : np.ndarray
        Minimum value of each feature
    data_max_ : np.ndarray
        Maximum value of each feature
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max for scaling."""
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        # Avoid division by zero
        self.data_range_[self.data_range_ == 0] = 1.0
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to range."""
        X = np.array(X)
        feature_min, feature_max = self.feature_range
        X_std = (X - self.data_min_) / self.data_range_
        return X_std * (feature_max - feature_min) + feature_min
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling."""
        X = np.array(X)
        feature_min, feature_max = self.feature_range
        X_std = (X - feature_min) / (feature_max - feature_min)
        return X_std * self.data_range_ + self.data_min_


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers.
    
    Uses median and interquartile range (IQR).
    
    X_scaled = (X - median) / IQR
    """
    
    def __init__(self):
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """Compute median and IQR."""
        X = np.array(X)
        self.center_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.scale_ = q75 - q25
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features."""
        X = np.array(X)
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


class Normalizer:
    """
    Normalize samples individually to unit norm.
    
    Each sample (row) is normalized independently.
    
    Parameters
    ----------
    norm : str, default='l2'
        Norm to use: 'l1', 'l2', or 'max'
    """
    
    def __init__(self, norm: str = 'l2'):
        self.norm = norm
    
    def fit(self, X: np.ndarray) -> 'Normalizer':
        """Normalizer doesn't need fitting."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize samples."""
        X = np.array(X, dtype=float)
        
        if self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        return X / norms
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.transform(X)


# ==================== ENCODING ====================

class LabelEncoder:
    """
    Encode target labels with values between 0 and n_classes-1.
    
    Attributes
    ----------
    classes_ : np.ndarray
        Unique classes found during fit
    """
    
    def __init__(self):
        self.classes_ = None
    
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """Fit label encoder."""
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to normalized encoding."""
        y = np.array(y)
        encoded = np.zeros(len(y), dtype=int)
        for i, cls in enumerate(self.classes_):
            encoded[y == cls] = i
        return encoded
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels back to original encoding."""
        y = np.array(y)
        return self.classes_[y]


class OneHotEncoder:
    """
    Encode categorical features as one-hot numeric array.
    
    Attributes
    ----------
    categories_ : List[np.ndarray]
        Categories for each feature
    """
    
    def __init__(self, sparse: bool = False):
        self.sparse = sparse
        self.categories_ = None
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Fit encoder to X."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.categories_ = []
        for col in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, col]))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to one-hot encoding."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        encoded_cols = []
        
        for col_idx, categories in enumerate(self.categories_):
            n_categories = len(categories)
            encoded = np.zeros((n_samples, n_categories))
            
            for i, category in enumerate(categories):
                mask = X[:, col_idx] == category
                encoded[mask, i] = 1
            
            encoded_cols.append(encoded)
        
        return np.hstack(encoded_cols)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform one-hot back to original."""
        X = np.array(X)
        n_samples = X.shape[0]
        n_features = len(self.categories_)
        
        result = np.zeros((n_samples, n_features), dtype=object)
        
        col_idx = 0
        for feat_idx, categories in enumerate(self.categories_):
            n_categories = len(categories)
            one_hot_cols = X[:, col_idx:col_idx + n_categories]
            
            # Find which category is 1
            category_indices = np.argmax(one_hot_cols, axis=1)
            result[:, feat_idx] = categories[category_indices]
            
            col_idx += n_categories
        
        return result


# ==================== DATA SPLITTING ====================

def train_test_split(
    *arrays,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Parameters
    ----------
    *arrays : sequence of arrays
        Arrays to split
    test_size : float, default=0.25
        Proportion of dataset to include in test split
    random_state : int or None
        Random seed
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    stratify : array-like or None
        If not None, split in a stratified fashion using this as class labels
    
    Returns
    -------
    splitting : List[np.ndarray]
        List of arrays: train-test split of inputs
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(arrays[0])
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if stratify is not None:
        # Stratified split
        stratify = np.array(stratify)
        unique_classes = np.unique(stratify)
        
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            n_cls_samples = len(cls_indices)
            n_cls_test = int(n_cls_samples * test_size)
            
            if shuffle:
                np.random.shuffle(cls_indices)
            
            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
    
    else:
        # Regular split
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        train_indices = indices[n_test:]
        test_indices = indices[:n_test]
    
    # Split all arrays
    result = []
    for array in arrays:
        array = np.array(array)
        result.append(array[train_indices])
        result.append(array[test_indices])
    
    return result


def k_fold_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
):
    """
    Generate K-Fold cross-validation indices.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle before splitting
    random_state : int or None
        Random seed
    
    Yields
    ------
    train_indices : np.ndarray
        Training set indices
    test_indices : np.ndarray
        Test set indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        
        yield train_indices, test_indices
        
        current = stop


# ==================== FEATURE ENGINEERING ====================

class PolynomialFeatures:
    """
    Generate polynomial and interaction features.
    
    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features
    include_bias : bool, default=True
        Include bias column (ones)
    interaction_only : bool, default=False
        Only interaction features (no powers)
    """
    
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_input_features_ = None
        self.n_output_features_ = None
    
    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """Compute number of output features."""
        X = np.array(X)
        self.n_input_features_ = X.shape[1]
        
        # Calculate number of output features
        # This is a simplified calculation
        if not self.interaction_only:
            # All combinations with repetition
            from math import comb
            self.n_output_features_ = sum(
                comb(self.n_input_features_ + d - 1, d)
                for d in range(1 if not self.include_bias else 0, self.degree + 1)
            )
        else:
            # Only interactions (no repeated features)
            from math import comb
            self.n_output_features_ = sum(
                comb(self.n_input_features_, d)
                for d in range(1 if not self.include_bias else 0, min(self.degree + 1, self.n_input_features_ + 1))
            )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to polynomial features."""
        X = np.array(X)
        n_samples = X.shape[0]
        
        features = []
        
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        # Original features
        features.append(X)
        
        # Higher degree features
        if self.degree >= 2:
            for d in range(2, self.degree + 1):
                if self.interaction_only:
                    # Only interaction terms
                    from itertools import combinations
                    for combo in combinations(range(self.n_input_features_), d):
                        feature = np.prod(X[:, combo], axis=1).reshape(-1, 1)
                        features.append(feature)
                else:
                    # All polynomial terms
                    from itertools import combinations_with_replacement
                    for combo in combinations_with_replacement(range(self.n_input_features_), d):
                        feature = np.prod(X[:, combo], axis=1).reshape(-1, 1)
                        features.append(feature)
        
        return np.hstack(features)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


# ==================== MISSING VALUE HANDLING ====================

class SimpleImputer:
    """
    Impute missing values using mean, median, or mode.
    
    Parameters
    ----------
    strategy : str, default='mean'
        Imputation strategy: 'mean', 'median', 'most_frequent', or 'constant'
    fill_value : float or None
        Value to use when strategy='constant'
    """
    
    def __init__(self, strategy: str = 'mean', fill_value: Optional[float] = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
    
    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        """Compute statistics for imputation."""
        X = np.array(X, dtype=float)
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            # Mode for each column
            self.statistics_ = []
            for col in range(X.shape[1]):
                values = X[:, col]
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    unique, counts = np.unique(values, return_counts=True)
                    self.statistics_.append(unique[np.argmax(counts)])
                else:
                    self.statistics_.append(0)
            self.statistics_ = np.array(self.statistics_)
        elif self.strategy == 'constant':
            if self.fill_value is None:
                self.fill_value = 0
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values."""
        X = np.array(X, dtype=float).copy()
        
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            X[mask, col] = self.statistics_[col]
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    print("=== StandardScaler Example ===")
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Original:\n{X}")
    print(f"Scaled:\n{X_scaled}")
    print(f"Mean: {scaler.mean_}")
    print(f"Std: {scaler.scale_}")
    
    print("\n=== MinMaxScaler Example ===")
    minmax = MinMaxScaler(feature_range=(0, 1))
    X_minmax = minmax.fit_transform(X)
    print(f"MinMax Scaled:\n{X_minmax}")
    
    print("\n=== LabelEncoder Example ===")
    y = np.array(['cat', 'dog', 'cat', 'bird', 'dog', 'cat'])
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Original: {y}")
    print(f"Encoded: {y_encoded}")
    print(f"Classes: {le.classes_}")
    print(f"Decoded: {le.inverse_transform(y_encoded)}")
    
    print("\n=== OneHotEncoder Example ===")
    X_cat = np.array(['red', 'green', 'blue', 'red', 'green'])
    ohe = OneHotEncoder()
    X_onehot = ohe.fit_transform(X_cat)
    print(f"Original: {X_cat}")
    print(f"One-Hot:\n{X_onehot}")
    print(f"Categories: {ohe.categories_}")
    
    print("\n=== Train-Test Split Example ===")
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")
    
    print("\n=== SimpleImputer Example ===")
    X_missing = np.array([[1, 2], [np.nan, 4], [5, np.nan], [7, 8]])
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_missing)
    print(f"With missing:\n{X_missing}")
    print(f"Imputed:\n{X_imputed}")