"""
Decision Tree Implementation from Scratch (CART Algorithm)

Decision Trees use a tree structure to make decisions by recursively
splitting the data based on feature values.

Mathematical Foundation:
    Information Gain (Classification):
    IG(D,A) = Entropy(D) - Σ(|Dᵥ|/|D|) * Entropy(Dᵥ)
    
    Entropy: H(D) = -Σ p(c) * log₂(p(c))
    
    Gini Impurity (Classification):
    Gini(D) = 1 - Σ p(c)²
    
    MSE (Regression):
    MSE(D) = (1/n) * Σ(yᵢ - ȳ)²
"""

import numpy as np
from collections import Counter
from typing import Optional, Union


class Node:
    """
    Node class for decision tree.
    
    Attributes
    ----------
    feature_index : int or None
        Index of feature to split on
    threshold : float or None
        Threshold value for splitting
    left : Node or None
        Left child node
    right : Node or None
        Right child node
    value : float or int or None
        Predicted value for leaf nodes
    """
    
    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[Union[float, int]] = None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree for classification and regression.
    
    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node
    criterion : str, default='gini'
        Function to measure split quality:
        - 'gini': Gini impurity (classification)
        - 'entropy': Information gain (classification)
        - 'mse': Mean squared error (regression)
    task : str, default='classification'
        Task type: 'classification' or 'regression'
    max_features : int or None, default=None
        Number of features to consider for best split
        
    Attributes
    ----------
    root : Node
        Root node of the decision tree
    n_features : int
        Number of features in training data
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        task: str = 'classification',
        max_features: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.max_features = max_features
        self.root = None
        self.n_features = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Build decision tree.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : DecisionTree
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features = X.shape[1]
        
        if self.max_features is None:
            self.max_features = self.n_features
        
        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0)
        
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf(y)
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        # Check minimum samples per leaf
        if np.sum(left_indices) < self.min_samples_leaf or \
           np.sum(right_indices) < self.min_samples_leaf:
            return self._create_leaf(y)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _create_leaf(self, y: np.ndarray) -> Node:
        """Create a leaf node."""
        if self.task == 'classification':
            # Most common class
            value = Counter(y).most_common(1)[0][0]
        else:  # regression
            # Mean value
            value = np.mean(y)
        
        return Node(value=value)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best feature and threshold to split on.
        
        Returns
        -------
        best_feature : int or None
            Index of best feature
        best_threshold : float or None
            Best threshold value
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        # Randomly select features to consider
        feature_indices = np.random.choice(
            self.n_features,
            size=min(self.max_features, self.n_features),
            replace=False
        )
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate information gain
                gain = self._calculate_information_gain(
                    y, y[left_mask], y[right_mask]
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_information_gain(
        self,
        parent: np.ndarray,
        left: np.ndarray,
        right: np.ndarray
    ) -> float:
        """Calculate information gain for a split."""
        n_parent = len(parent)
        n_left = len(left)
        n_right = len(right)
        
        if self.criterion == 'gini':
            parent_impurity = self._gini_impurity(parent)
            left_impurity = self._gini_impurity(left)
            right_impurity = self._gini_impurity(right)
        elif self.criterion == 'entropy':
            parent_impurity = self._entropy(parent)
            left_impurity = self._entropy(left)
            right_impurity = self._entropy(right)
        elif self.criterion == 'mse':
            parent_impurity = self._mse(parent)
            left_impurity = self._mse(left)
            right_impurity = self._mse(right)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        # Calculate weighted impurity of children
        child_impurity = (n_left / n_parent) * left_impurity + \
                        (n_right / n_parent) * right_impurity
        
        # Information gain
        return parent_impurity - child_impurity
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity.
        
        Gini = 1 - Σ p²
        """
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy.
        
        H = -Σ p * log₂(p)
        """
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Add small epsilon to avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _mse(self, y: np.ndarray) -> float:
        """
        Calculate mean squared error.
        
        MSE = (1/n) * Σ(y - ȳ)²
        """
        if len(y) == 0:
            return 0
        
        return np.mean((y - np.mean(y)) ** 2)
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> Union[int, float]:
        """Predict single sample by traversing the tree."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate score (accuracy for classification, R² for regression).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True labels/values
            
        Returns
        -------
        score : float
            Accuracy (classification) or R² (regression)
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(y_pred == y)
        else:  # regression
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def get_depth(self, node: Optional[Node] = None) -> int:
        """Get the depth of the tree."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 0
        
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("=== Classification Example ===")
    X_clf, y_clf = make_classification(n_samples=300, n_features=10, 
                                       n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Train classifier
    clf = DecisionTree(max_depth=10, min_samples_split=2, 
                      criterion='gini', task='classification')
    clf.fit(X_train, y_train)
    
    print(f"Training Accuracy: {clf.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {clf.score(X_test, y_test):.4f}")
    print(f"Tree Depth: {clf.get_depth()}")
    
    print("\n=== Regression Example ===")
    X_reg, y_reg = make_regression(n_samples=300, n_features=10, noise=10, 
                                   random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Train regressor
    reg = DecisionTree(max_depth=10, min_samples_split=5, 
                      criterion='mse', task='regression')
    reg.fit(X_train, y_train)
    
    print(f"Training R²: {reg.score(X_train, y_train):.4f}")
    print(f"Test R²: {reg.score(X_test, y_test):.4f}")
    print(f"Tree Depth: {reg.get_depth()}")