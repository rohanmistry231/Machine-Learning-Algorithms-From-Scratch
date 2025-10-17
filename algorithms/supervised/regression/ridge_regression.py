"""
Ridge Regression (L2 Regularization) Implementation from Scratch

Ridge Regression adds L2 regularization penalty to linear regression
to prevent overfitting by penalizing large coefficients.

Mathematical Foundation:
    Cost Function: J(β) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ/2m) * Σβⱼ²
    
    where λ is the regularization parameter
    
    Normal Equation: β = (XᵀX + λI)⁻¹Xᵀy
"""

import numpy as np
from typing import Optional


class RidgeRegression:
    """
    Ridge Regression with L2 regularization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (λ). Must be positive.
        Larger values specify stronger regularization.
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    method : str, default='normal_equation'
        Method to use: 'gradient_descent' or 'normal_equation'
        
    Attributes
    ----------
    weights : np.ndarray
        Coefficients of the linear model
    bias : float
        Intercept term
    cost_history : list
        History of cost function values during training
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        method: str = 'normal_equation'
    ):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit the ridge regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : RidgeRegression
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y, n_samples, n_features)
        elif self.method == 'normal_equation':
            self._fit_normal_equation(X, y, n_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def _fit_gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        n_features: int
    ) -> None:
        """Fit using gradient descent with L2 regularization."""
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute cost with L2 regularization
            mse_cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            l2_cost = (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
            cost = mse_cost + l2_cost
            self.cost_history.append(cost)
            
            # Compute gradients with L2 regularization
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray, n_features: int) -> None:
        """
        Fit using the closed-form solution with L2 regularization.
        
        Ridge Normal Equation: β = (XᵀX + λI)⁻¹Xᵀy
        """
        # Add bias column
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Create regularization matrix (don't regularize bias term)
        regularization_matrix = self.alpha * np.eye(X_with_bias.shape[1])
        regularization_matrix[0, 0] = 0  # Don't regularize bias
        
        # Compute parameters using ridge normal equation
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias + regularization_matrix) @ X_with_bias.T @ y
        
        self.bias = theta[0, 0]
        self.weights = theta[1:].reshape(-1, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ridge regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples, 1)
            Predicted values
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            R² score
        """
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'alpha': self.alpha,
            'cost_history': self.cost_history
        }


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample data with multiple features
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RidgeRegression(alpha=1.0, method='normal_equation')
    model.fit(X_train, y_train)
    
    print("Ridge Regression Results:")
    print(f"R² Score: {model.score(X_test, y_test):.4f}")
    print(f"Weights shape: {model.weights.shape}")
    print(f"Bias: {model.bias:.4f}")
    print(f"Alpha (regularization): {model.alpha}")