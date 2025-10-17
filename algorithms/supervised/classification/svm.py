"""
Support Vector Machine (SVM) Implementation from Scratch

SVM finds the optimal hyperplane that maximally separates classes
by maximizing the margin between support vectors.

Mathematical Foundation:
    Linear SVM objective:
    min (1/2)||w||² + C * Σ max(0, 1 - yᵢ(wᵀxᵢ + b))
    
    where:
    - w is the weight vector
    - b is the bias
    - C is the regularization parameter
    - The second term is the hinge loss
    
    Decision function: f(x) = sign(wᵀx + b)
"""

import numpy as np
from typing import Optional


class SVM:
    """
    Support Vector Machine classifier.
    
    Uses gradient descent with hinge loss for optimization.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    lambda_param : float, default=0.01
        Regularization parameter (1/C)
    n_iterations : int, default=1000
        Number of training iterations
    kernel : str, default='linear'
        Kernel type: 'linear', 'rbf', 'poly'
    gamma : float, default=0.1
        Kernel coefficient for 'rbf' and 'poly'
    degree : int, default=3
        Degree for 'poly' kernel
        
    Attributes
    ----------
    weights : np.ndarray
        Weight vector
    bias : float
        Bias term
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_iterations: int = 1000,
        kernel: str = 'linear',
        gamma: float = 0.1,
        degree: int = 3
    ):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.weights = None
        self.bias = None
        self.X_train = None
        self.y_train = None
    
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel function."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Fit SVM classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels (must be -1 or 1)
            
        Returns
        -------
        self : SVM
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([-1, 1])):
            y = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        
        if self.kernel == 'linear':
            # Linear SVM using gradient descent
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for _ in range(self.n_iterations):
                for idx, x_i in enumerate(X):
                    # Check if sample satisfies margin condition
                    condition = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                    
                    if condition:
                        # Update weights (no hinge loss)
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                    else:
                        # Update weights (hinge loss active)
                        self.weights -= self.learning_rate * (
                            2 * self.lambda_param * self.weights - np.dot(x_i, y[idx])
                        )
                        self.bias -= self.learning_rate * y[idx]
        else:
            # For non-linear kernels, store training data
            self.X_train = X
            self.y_train = y
            # Initialize dual coefficients (simplified approach)
            self.alpha = np.zeros(n_samples)
            
            # Simplified kernel SVM training
            for _ in range(self.n_iterations):
                for idx in range(n_samples):
                    # Compute decision value
                    decision = self._compute_kernel_decision(X[idx])
                    
                    # Update alpha based on margin violation
                    margin = y[idx] * decision
                    if margin < 1:
                        self.alpha[idx] += self.learning_rate * (1 - margin)
        
        return self
    
    def _compute_kernel_decision(self, x: np.ndarray) -> float:
        """Compute decision value for kernel SVM."""
        decision = 0
        for i in range(len(self.X_train)):
            decision += self.alpha[i] * self.y_train[i] * \
                       self._kernel_function(self.X_train[i], x)
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels (-1 or 1)
        """
        X = np.array(X)
        
        if self.kernel == 'linear':
            linear_output = np.dot(X, self.weights) + self.bias
            return np.sign(linear_output)
        else:
            predictions = []
            for x in X:
                decision = self._compute_kernel_decision(x)
                predictions.append(np.sign(decision))
            return np.array(predictions)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        decision : np.ndarray of shape (n_samples,)
            Decision function values
        """
        X = np.array(X)
        
        if self.kernel == 'linear':
            return np.dot(X, self.weights) + self.bias
        else:
            return np.array([self._compute_kernel_decision(x) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True labels
            
        Returns
        -------
        accuracy : float
            Accuracy score
        """
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([-1, 1])):
            y = np.where(y == unique_labels[0], -1, 1)
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = {
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'n_iterations': self.n_iterations,
            'kernel': self.kernel
        }
        
        if self.kernel == 'linear':
            params['weights'] = self.weights
            params['bias'] = self.bias
        else:
            params['n_support_vectors'] = np.sum(self.alpha > 1e-5) if hasattr(self, 'alpha') else 0
        
        return params


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate linearly separable data
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=2.0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=== Linear SVM ===")
    # Train linear SVM
    svm_linear = SVM(learning_rate=0.001, lambda_param=0.01, 
                     n_iterations=1000, kernel='linear')
    svm_linear.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {svm_linear.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {svm_linear.score(X_test_scaled, y_test):.4f}")
    
    print("\n=== RBF Kernel SVM ===")
    # Train RBF kernel SVM
    svm_rbf = SVM(learning_rate=0.001, lambda_param=0.01,
                  n_iterations=500, kernel='rbf', gamma=0.1)
    svm_rbf.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {svm_rbf.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {svm_rbf.score(X_test_scaled, y_test):.4f}")