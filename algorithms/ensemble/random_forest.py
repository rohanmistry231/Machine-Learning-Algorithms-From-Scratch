"""
Random Forest Implementation from Scratch

Random Forest is an ensemble learning method that combines multiple
decision trees trained on random subsets of data and features.

Mathematical Foundation:
    Bagging (Bootstrap Aggregating):
    - Create N bootstrap samples from training data
    - Train a decision tree on each sample
    - For each tree, only consider random subset of features at each split
    
    Prediction:
    - Classification: Majority vote across all trees
    - Regression: Average prediction across all trees
    
    Out-of-Bag (OOB) Error:
    - Use samples not in bootstrap for validation
"""

import numpy as np
from typing import List, Optional
from collections import Counter


class DecisionTreeForest:
    """
    Decision tree specifically designed for Random Forest.
    Simplified version with random feature selection.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        task: str = 'classification'
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.tree = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate MSE for regression."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> tuple:
        """Find best split among random features."""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                if self.task == 'classification':
                    parent_impurity = self._gini_impurity(y)
                    left_impurity = self._gini_impurity(y[left_mask])
                    right_impurity = self._gini_impurity(y[right_mask])
                else:
                    parent_impurity = self._mse(y)
                    left_impurity = self._mse(y[left_mask])
                    right_impurity = self._mse(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                gain = parent_impurity - (n_left/n_total * left_impurity + 
                                         n_right/n_total * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """Build decision tree recursively."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or n_samples < self.min_samples_split:
            if self.task == 'classification':
                value = Counter(y).most_common(1)[0][0]
            else:
                value = np.mean(y)
            return {'is_leaf': True, 'value': value}
        
        # Random feature selection
        max_features = self.max_features or n_features
        feature_indices = np.random.choice(n_features, max_features, replace=False)
        
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)
        
        if best_feature is None:
            if self.task == 'classification':
                value = Counter(y).most_common(1)[0][0]
            else:
                value = np.mean(y)
            return {'is_leaf': True, 'value': value}
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            if self.task == 'classification':
                value = Counter(y).most_common(1)[0][0]
            else:
                value = np.mean(y)
            return {'is_leaf': True, 'value': value}
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeForest':
        """Fit decision tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: dict):
        """Predict single sample."""
        if node['is_leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for all samples."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForest:
    """
    Random Forest classifier and regressor.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int or None, default=None
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples required to split
    min_samples_leaf : int, default=1
        Minimum samples required at leaf node
    max_features : str or int or None, default='sqrt'
        Number of features to consider for best split:
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - int: specific number
        - None: all features
    bootstrap : bool, default=True
        Whether to use bootstrap samples
    task : str, default='classification'
        Task type: 'classification' or 'regression'
    random_state : int or None, default=None
        Random seed
    oob_score : bool, default=False
        Whether to calculate out-of-bag score
        
    Attributes
    ----------
    trees : List[DecisionTreeForest]
        List of decision trees in the forest
    oob_score_ : float or None
        Out-of-bag score
    feature_importances_ : np.ndarray
        Feature importance scores
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        task: str = 'classification',
        random_state: Optional[int] = None,
        oob_score: bool = False
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.task = task
        self.random_state = random_state
        self.oob_score = oob_score
        
        self.trees = []
        self.oob_score_ = None
        self.feature_importances_ = None
    
    def _get_max_features(self, n_features: int) -> int:
        """Calculate number of features to use."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Create bootstrap sample."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Fit random forest.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : RandomForest
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        oob_predictions = [[] for _ in range(n_samples)]
        
        # Train each tree
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample, indices = self._bootstrap_sample(X, y)
                oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            else:
                X_sample, y_sample = X, y
                oob_indices = []
            
            # Create and train tree
            tree = DecisionTreeForest(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                task=self.task
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, oob_pred):
                    oob_predictions[idx].append(pred)
        
        # Calculate OOB score
        if self.oob_score:
            valid_samples = [i for i, preds in enumerate(oob_predictions) if len(preds) > 0]
            if len(valid_samples) > 0:
                if self.task == 'classification':
                    oob_y_pred = np.array([
                        Counter(oob_predictions[i]).most_common(1)[0][0]
                        for i in valid_samples
                    ])
                else:
                    oob_y_pred = np.array([
                        np.mean(oob_predictions[i])
                        for i in valid_samples
                    ])
                
                if self.task == 'classification':
                    self.oob_score_ = np.mean(oob_y_pred == y[valid_samples])
                else:
                    ss_tot = np.sum((y[valid_samples] - np.mean(y[valid_samples])) ** 2)
                    ss_res = np.sum((y[valid_samples] - oob_y_pred) ** 2)
                    self.oob_score_ = 1 - (ss_res / ss_tot)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
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
        
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        if self.task == 'classification':
            # Majority vote
            predictions = []
            for i in range(X.shape[0]):
                predictions.append(Counter(tree_predictions[:, i]).most_common(1)[0][0])
            return np.array(predictions)
        else:
            # Average
            return np.mean(tree_predictions, axis=0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate score (accuracy or R²).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            Accuracy (classification) or R² (regression)
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(y_pred == y)
        else:
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'n_trees_trained': len(self.trees),
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'task': self.task,
            'oob_score': self.oob_score_
        }


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("=== Random Forest Classification ===")
    X_clf, y_clf = make_classification(n_samples=500, n_features=10,
                                       n_informative=7, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    rf_clf = RandomForest(n_estimators=50, max_depth=10, max_features='sqrt',
                          task='classification', random_state=42, oob_score=True)
    rf_clf.fit(X_train, y_train)
    
    print(f"Training Accuracy: {rf_clf.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {rf_clf.score(X_test, y_test):.4f}")
    print(f"OOB Score: {rf_clf.oob_score_:.4f}")
    print(f"Number of trees: {rf_clf.get_params()['n_trees_trained']}")
    
    print("\n=== Random Forest Regression ===")
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    rf_reg = RandomForest(n_estimators=50, max_depth=10, max_features='sqrt',
                         task='regression', random_state=42, oob_score=True)
    rf_reg.fit(X_train, y_train)
    
    print(f"Training R²: {rf_reg.score(X_train, y_train):.4f}")
    print(f"Test R²: {rf_reg.score(X_test, y_test):.4f}")
    print(f"OOB Score: {rf_reg.oob_score_:.4f}")