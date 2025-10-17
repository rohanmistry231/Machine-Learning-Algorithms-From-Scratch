"""
Gradient Boosting Implementation from Scratch

Gradient Boosting builds an ensemble of weak learners (typically decision trees)
sequentially, where each new tree corrects errors made by previous trees.

Mathematical Foundation:
    F₀(x) = argmin_γ Σ L(yᵢ, γ)
    
    For m = 1 to M:
        1. Compute pseudo-residuals:
           rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=F_{m-1}}
        
        2. Fit weak learner hₘ(x) to pseudo-residuals
        
        3. Compute multiplier γₘ:
           γₘ = argmin_γ Σ L(yᵢ, F_{m-1}(xᵢ) + γhₘ(xᵢ))
        
        4. Update model:
           Fₘ(x) = F_{m-1}(x) + ν·γₘ·hₘ(x)
    
    where ν is the learning rate
"""

import numpy as np
from typing import Optional, List


class GradientBoostingTree:
    """
    Simple decision tree for gradient boosting (regression tree).
    """
    
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate MSE."""
        if len(y) == 0:
            return 0
        return np.var(y)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find best split."""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                parent_var = self._mse(y)
                left_var = self._mse(y[left_mask])
                right_var = self._mse(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                gain = parent_var - (n_left/n_total * left_var + n_right/n_total * right_var)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """Build regression tree."""
        n_samples = len(y)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'is_leaf': True, 'value': np.mean(y)}
        
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return {'is_leaf': True, 'value': np.mean(y)}
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {'is_leaf': True, 'value': np.mean(y)}
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingTree':
        """Fit regression tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: dict) -> float:
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


class GradientBoosting:
    """
    Gradient Boosting for regression and binary classification.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Learning rate (shrinkage parameter)
    max_depth : int, default=3
        Maximum depth of individual trees
    min_samples_split : int, default=2
        Minimum samples required to split
    subsample : float, default=1.0
        Fraction of samples to use for fitting each tree
    task : str, default='regression'
        Task type: 'regression' or 'classification'
    loss : str, default='mse'
        Loss function: 'mse' (regression), 'log_loss' (classification)
    random_state : int or None, default=None
        Random seed
        
    Attributes
    ----------
    trees : List[GradientBoostingTree]
        List of fitted trees
    init_prediction : float
        Initial prediction value
    train_loss_history : List[float]
        Training loss at each iteration
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        task: str = 'regression',
        loss: str = 'mse',
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.task = task
        self.loss = loss
        self.random_state = random_state
        
        self.trees = []
        self.init_prediction = None
        self.train_loss_history = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss."""
        if self.loss == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss == 'log_loss':
            # Binary cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
    
    def _compute_gradients(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute negative gradients (pseudo-residuals).
        
        For MSE: gradient = -(y_true - y_pred) = y_pred - y_true
        Negative gradient = y_true - y_pred (residuals)
        
        For log loss with sigmoid:
        gradient = y_pred - y_true (in probability space)
        """
        if self.loss == 'mse':
            return y_true - y_pred
        elif self.loss == 'log_loss':
            # For binary classification with log loss
            return y_true - y_pred
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoosting':
        """
        Fit gradient boosting model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : GradientBoosting
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        
        # Initialize predictions
        if self.task == 'regression':
            self.init_prediction = np.mean(y)
            F = np.full(n_samples, self.init_prediction)
        else:  # classification
            # Initialize with log odds
            pos_ratio = np.mean(y)
            self.init_prediction = np.log(pos_ratio / (1 - pos_ratio + 1e-15))
            F = np.full(n_samples, self.init_prediction)
        
        # Boosting iterations
        for i in range(self.n_estimators):
            # Convert to probabilities for classification
            if self.task == 'classification':
                y_pred_proba = self._sigmoid(F)
                gradients = self._compute_gradients(y, y_pred_proba)
            else:
                gradients = self._compute_gradients(y, F)
            
            # Subsample data
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                gradients_sample = gradients[indices]
            else:
                X_sample = X
                gradients_sample = gradients
            
            # Fit tree to negative gradients
            tree = GradientBoostingTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, gradients_sample)
            
            # Update predictions
            update = tree.predict(X)
            F += self.learning_rate * update
            
            self.trees.append(tree)
            
            # Record loss
            if self.task == 'classification':
                y_pred_proba = self._sigmoid(F)
                loss = self._compute_loss(y, y_pred_proba)
            else:
                loss = self._compute_loss(y, F)
            
            self.train_loss_history.append(loss)
        
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
            Predicted values or class labels
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        F = np.full(n_samples, self.init_prediction)
        
        # Add predictions from all trees
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        if self.task == 'classification':
            # Convert to probabilities and return class labels
            probabilities = self._sigmoid(F)
            return (probabilities >= 0.5).astype(int)
        else:
            return F
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification only).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples,)
            Predicted probabilities for positive class
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        F = np.full(n_samples, self.init_prediction)
        
        # Add predictions from all trees
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        return self._sigmoid(F)
    
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
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'task': self.task,
            'final_train_loss': self.train_loss_history[-1] if self.train_loss_history else None
        }


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    print("=== Gradient Boosting Classification ===")
    X_clf, y_clf = make_classification(n_samples=500, n_features=10,
                                       n_informative=7, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    gb_clf = GradientBoosting(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        task='classification',
        loss='log_loss',
        random_state=42
    )
    gb_clf.fit(X_train, y_train)
    
    print(f"Training Accuracy: {gb_clf.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {gb_clf.score(X_test, y_test):.4f}")
    print(f"Number of trees: {gb_clf.get_params()['n_trees_trained']}")
    print(f"Final training loss: {gb_clf.get_params()['final_train_loss']:.6f}")
    
    print("\n=== Gradient Boosting Regression ===")
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    gb_reg = GradientBoosting(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        task='regression',
        loss='mse',
        random_state=42
    )
    gb_reg.fit(X_train, y_train)
    
    print(f"Training R²: {gb_reg.score(X_train, y_train):.4f}")
    print(f"Test R²: {gb_reg.score(X_test, y_test):.4f}")
    print(f"Number of trees: {gb_reg.get_params()['n_trees_trained']}")
    print(f"Final training loss: {gb_reg.get_params()['final_train_loss']:.6f}")
    
    # Plot training loss history
    print("\n=== Loss History (first 20 iterations) ===")
    for i in range(min(20, len(gb_reg.train_loss_history))):
        if i % 5 == 0:
            print(f"Iteration {i}: Loss = {gb_reg.train_loss_history[i]:.6f}")